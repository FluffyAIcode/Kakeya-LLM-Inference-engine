"""Kakeya Inference Engine — long-context concurrency / throughput evaluation.

Drives the product engine (`inference_engine.engine.KakeyaEngine`,
NativeHybridBounded policy) on a long-context NIAH workload: sweeps the
concurrent-session count N at a fixed context, reporting the peak-window
admission ceiling (the engine's predicted max concurrency), per-session recall,
aggregate throughput, and measured peak GPU memory. Intended for the
side-by-side with vLLM (`vllm_multitenant_parallel_bench.py`).
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
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--haystack-lines", type=int, default=3100)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,24,32")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-budget-gb", type=float, default=139.8)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "."); sys.path.insert(0, "sdks/python")
    from inference_engine.engine import KakeyaEngine
    from inference_engine.v04.niah_eval import make_niah_dataset

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    print(f"[kge] loading {args.verifier_id}", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    engine = KakeyaEngine(model, tok, sink=args.sink, window=args.window,
                          chunk_size=args.chunk_size)

    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    kvm = engine.bounded_kv_model()

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
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    while len(bucket) < max(batch_sizes):
        bucket += bucket[: max(batch_sizes) - len(bucket)]

    budget = int(args.mem_budget_gb * 1e9)
    admit = engine.max_concurrent(memory_budget_bytes=budget,
                                  model_weight_bytes=model_bytes,
                                  context_len=modal + args.gen_tokens)
    per_sess = kvm.resident_bytes(modal + args.gen_tokens)
    print(f"[kge] modal={modal} exact_layers={engine.exact_layer_indices} "
          f"resident_kv/session={per_sess/1e9:.2f}GB model={model_bytes/1e9:.1f}GB "
          f"admission_ceiling={admit}", file=sys.stderr, flush=True)

    def recall(out_ids, ans):
        return ans in tok.decode(out_ids, skip_special_tokens=True)

    try:
        engine.generate_cohort([bucket[0][0]], max_new_tokens=4)  # warmup
    except Exception as e:  # noqa: BLE001
        print(f"[kge] warmup note: {e}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for N in batch_sizes:
        sel = bucket[:N]
        torch.cuda.reset_peak_memory_stats(device)
        try:
            t0 = time.perf_counter()
            gens = engine.generate_cohort([s[0] for s in sel],
                                          max_new_tokens=args.gen_tokens)
            dt = time.perf_counter() - t0
        except torch.OutOfMemoryError:
            print(f"[kge] N={N}: OOM", file=sys.stderr, flush=True)
            rows.append({"agents": N, "oom": True})
            break
        rec = sum(recall(gens[i], sel[i][1]) for i in range(N)) / N
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        tps = (N * args.gen_tokens) / dt
        row = {"agents": N, "aggregate_tps_e2e": round(tps, 2),
               "recall": round(rec, 3), "peak_gpu_gb": round(peak, 2)}
        rows.append(row)
        print(f"[kge] N={N:3d} | {row['aggregate_tps_e2e']} tok/s (e2e) | "
              f"recall {row['recall']} | peak {row['peak_gpu_gb']} GB",
              file=sys.stderr, flush=True)

    report = {
        "kind": "kakeya_engine_throughput",
        "config": {"verifier_id": args.verifier_id, "policy": engine.policy,
                   "sink": args.sink, "window": args.window,
                   "chunk_size": args.chunk_size, "modal_prompt_len": modal,
                   "gen_tokens": args.gen_tokens, "exact_layers": engine.exact_layer_indices,
                   "resident_kv_bytes_per_session": per_sess,
                   "model_weight_bytes": model_bytes,
                   "admission_ceiling": admit, "mem_budget_gb": args.mem_budget_gb},
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "results": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[kge] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
