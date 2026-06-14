"""PR-A3c throughput: batched scheduler vs serialized, on the served per-session
adapters (recall-preserving S5).

§3.6 made the served path correct multi-tenant but RPC-serialized. This bench
takes N per-session restored adapters (from PerSessionVerifierRegistry), prefills
each, and decodes the cohort two ways on CUDA:
  * serialized  — each session's decode forward run alone, summed (the §3.6 path)
  * batched     — BatchedDecodeScheduler fuses all N into one forward per step

and reports aggregate decode tok/s for each, the speedup, and per-session recall
(must be 1.0 — recall is the bottom line).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch


@torch.no_grad()
def _serial_decode(model, adapter, max_tokens, eos_ids, device):
    """Single-session greedy decode against the adapter's own _past."""
    cache = adapter._past
    T = int(adapter._past_len)
    logits = adapter.next_token_logits
    gen: List[int] = []
    for step in range(max_tokens):
        tok = int(logits.argmax(-1).item())
        gen.append(tok)
        if tok in eos_ids:
            break
        cur = torch.tensor([[tok]], device=device)
        pos = torch.tensor([[T + step]], device=device)
        out = model(input_ids=cur, position_ids=pos,
                    cache_position=torch.tensor([T + step], device=device),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1, :]
    return gen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--f-theta-dir", default="results/research/f_theta_v5_s5_sliding")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--haystack-lines", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[sched] CUDA required.", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    from transformers import AutoTokenizer
    from inference_engine.v04 import make_niah_dataset
    from inference_engine.v04.build_restored import load_restored_verifier
    from inference_engine.session.verifier_registry import PerSessionVerifierRegistry
    from inference_engine.session.batch_scheduler import BatchedDecodeScheduler

    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    eos_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()

    print("[sched] loading restored verifier ...", file=sys.stderr, flush=True)
    base = load_restored_verifier(
        verifier_id=args.verifier_id, drafter_id=args.drafter_id,
        f_theta_dir=args.f_theta_dir, sink_size=args.sink, window_size=args.window,
        s5_exact_full_attn=True, device="cuda", incremental=True)
    model = base.model
    registry = PerSessionVerifierRegistry(factory=base.spawn)
    scheduler = BatchedDecodeScheduler(model, device)

    N = args.sessions
    pool = make_niah_dataset(n_samples=N * 3, haystack_min_lines=args.haystack_lines,
                             haystack_max_lines=args.haystack_lines, seed=0)

    def encode(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids[0].tolist()

    enc = [(encode(s.prompt_text), s.answer_text) for s in pool]
    modal = Counter(len(e[0]) for e in enc).most_common(1)[0][0]
    bucket = [(i, a) for i, a in enc if len(i) == modal][:N]
    while len(bucket) < N:
        bucket += bucket[: N - len(bucket)]
    prompts = [b[0] for b in bucket]
    answers = [b[1] for b in bucket]
    print(f"[sched] {N} sessions, modal prompt len={modal}", file=sys.stderr, flush=True)

    def recall(toks, ans):
        return ans in tok.decode(toks, skip_special_tokens=True)

    def fresh_adapters():
        ads = []
        for i in range(N):
            registry.remove(f"s{i}")
            a = registry.get(f"s{i}")
            a.prefill(prompts[i])
            ads.append(a)
        return ads

    # warmup
    try:
        wa = fresh_adapters()[:2]
        scheduler.run_cohort(wa, max_tokens=4, eos_ids=eos_ids)
    except Exception as e:  # noqa: BLE001
        print(f"[sched] warmup note: {e}", file=sys.stderr)

    # --- batched ---
    ads = fresh_adapters()
    bres = scheduler.run_cohort(ads, max_tokens=args.max_new_tokens, eos_ids=eos_ids)
    batched_tps = bres["decode_tokens_per_s"]
    batched_recall = sum(recall(bres["tokens"][i], answers[i]) for i in range(N)) / N

    # --- serialized (§3.6 path) ---
    ads = fresh_adapters()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    ser_tokens = []
    for i in range(N):
        ser_tokens.append(_serial_decode(model, ads[i], args.max_new_tokens, eos_ids, device))
    torch.cuda.synchronize(device)
    ser_dt = time.perf_counter() - t0
    ser_total = sum(len(t) for t in ser_tokens)
    serial_tps = round(ser_total / ser_dt, 3)
    serial_recall = sum(recall(ser_tokens[i], answers[i]) for i in range(N)) / N

    speedup = round(batched_tps / serial_tps, 2) if serial_tps else None
    report = {
        "kind": "k3_served_batched_scheduler",
        "config": {"sessions": N, "modal_prompt_len": modal,
                   "max_new_tokens": args.max_new_tokens,
                   "sink": args.sink, "window": args.window},
        "env": {"gpu": torch.cuda.get_device_name(0)},
        "serialized": {"aggregate_tps": serial_tps, "recall": round(serial_recall, 3)},
        "batched_scheduler": {"aggregate_tps": batched_tps,
                              "recall": round(batched_recall, 3)},
        "batched_speedup_vs_serialized": speedup,
    }
    print(f"[sched] N={N}: serialized {serial_tps} tok/s (recall {serial_recall}) | "
          f"batched {batched_tps} tok/s (recall {batched_recall}) | "
          f"speedup {speedup}x", file=sys.stderr, flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[sched] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
