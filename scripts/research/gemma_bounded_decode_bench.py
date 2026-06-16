"""gemma-4 bounded-decode concurrency ceiling (native hybrid cache, small window).

ADR 0015 item #2 on gemma-4. KEY FINDING (probed separately): gemma-4 keeps
recall 1.0 with the sliding window shrunk to ~68 *natively* — no Kakeya
restoration needed — because its 5 full-attention layers (of 30) carry recall.
So on gemma-4 "bounded decode" reduces to `sliding_window=W` on the native
HybridCache; this bench measures the resulting concurrency ceiling at long
context to compare against vLLM.

(Honest caveat: this is native-window tuning that vLLM can also apply, so it is
NOT a Kakeya algorithmic advantage on gemma-4 — that requires a full-attention
model where shrinking the window kills recall and only restoration recovers it.)

Sweeps N (concurrent sessions) at a fixed context, greedy, reporting per-N peak
GPU memory, per-session recall, and aggregate decode tok/s; the ceiling is the
largest N that fits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--sliding-window", type=int, default=68)
    ap.add_argument("--haystack-lines", type=int, default=3100)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,24,32")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "."); sys.path.insert(0, "sdks/python")
    from inference_engine.v04.niah_eval import make_niah_dataset

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    print(f"[gb] loading {args.verifier_id} sdpa bf16", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    tc = model.config.get_text_config()
    native_sw = tc.sliding_window
    tc.sliding_window = args.sliding_window
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = args.sliding_window
    print(f"[gb] sliding_window {native_sw} -> {args.sliding_window}", file=sys.stderr, flush=True)

    def enc(t):
        ids = tok.apply_chat_template([{"role": "user", "content": t}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        return (ids["input_ids"] if hasattr(ids, "keys") else ids)[0].tolist()

    pool = make_niah_dataset(n_samples=args.pool, haystack_min_lines=args.haystack_lines,
                             haystack_max_lines=args.haystack_lines, seed=args.seed)
    encs = [(enc(s.prompt_text), s.answer_text) for s in pool]
    modal = Counter(len(e[0]) for e in encs).most_common(1)[0][0]
    bucket = [(i, a) for i, a in encs if len(i) == modal]
    print(f"[gb] modal prompt len={modal}, {len(bucket)} equal-length", file=sys.stderr, flush=True)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    need = max(batch_sizes)
    while len(bucket) < need:
        bucket += bucket[: need - len(bucket)]

    def recall(ids_out, ans):
        return ans in tok.decode(ids_out, skip_special_tokens=True)

    @torch.no_grad()
    def run(N):
        sel = bucket[:N]
        ids = torch.tensor([s[0] for s in sel], device=device)
        ans = [s[1] for s in sel]
        torch.cuda.reset_peak_memory_stats(device)
        # prefill
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        gen = [[int(nxt[i])] for i in range(N)]
        T = ids.size(1)
        torch.cuda.synchronize(device); t0 = time.perf_counter()
        for step in range(args.gen_tokens - 1):
            cur = nxt.view(N, 1)
            cpos = torch.tensor([T + step], device=device)
            out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                        cache_position=cpos, logits_to_keep=1)
            cache = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1)
            for i in range(N):
                gen[i].append(int(nxt[i]))
        torch.cuda.synchronize(device); dt = time.perf_counter() - t0
        tps = (N * args.gen_tokens) / dt
        rec = sum(recall(gen[i], ans[i]) for i in range(N)) / N
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return tps, rec, peak

    # warmup
    try:
        run(1)
    except Exception as e:  # noqa: BLE001
        print(f"[gb] warmup note: {e}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    single_tps = None
    for N in batch_sizes:
        try:
            tps, rec, peak = run(N)
        except torch.OutOfMemoryError as e:  # noqa: BLE001
            print(f"[gb] N={N}: OOM ({e})", file=sys.stderr, flush=True)
            rows.append({"agents": N, "oom": True})
            break
        if single_tps is None:
            single_tps = tps
        row = {"agents": N, "decode_aggregate_tps": round(tps, 2),
               "recall": round(rec, 3),
               "parallel_speedup_vs_n1": round(tps / single_tps, 2),
               "peak_gpu_gb": round(peak, 2)}
        rows.append(row)
        print(f"[gb] N={N:3d} | {row['decode_aggregate_tps']} tok/s "
              f"(x{row['parallel_speedup_vs_n1']}) | recall {row['recall']} | "
              f"peak {row['peak_gpu_gb']} GB", file=sys.stderr, flush=True)

    report = {
        "kind": "gemma_bounded_decode",
        "config": {"verifier_id": args.verifier_id, "sliding_window": args.sliding_window,
                   "native_sliding_window": native_sw, "modal_prompt_len": modal,
                   "gen_tokens": args.gen_tokens, "batch_sizes": batch_sizes,
                   "note": ("native gemma-4 hybrid cache with shrunk sliding window; "
                            "no Kakeya restoration (recall comes from the 5 full-attn "
                            "layers). vLLM can apply the same window — not a Kakeya moat.")},
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "results": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[gb] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
