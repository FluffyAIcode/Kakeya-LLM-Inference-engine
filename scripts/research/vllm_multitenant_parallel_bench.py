"""vLLM apples-to-apples baseline for the PR-A3c restored-S5 multitenant bench.

Mirrors ``k3_cuda_multitenant_parallel_bench.py`` exactly so the two can be put
side by side on the SAME GPU, model, precision, prompts, gen length, and recall
predicate:

  * model      : google/gemma-4-26B-A4B-it (bf16 by default — matches the
                 Kakeya restored bench, which loads bf16)
  * prompts    : ``make_niah_dataset`` (same seed / haystack lines), the modal
                 token-length bucket tiled to N, fed to vLLM as token-ids so the
                 inputs are bit-identical to the Kakeya run
  * decode     : greedy (temperature 0), ``gen_tokens`` new tokens
  * for each N : submit the N prompts together; vLLM batches them (continuous
                 batching = its parallel path)

Reports, per N (= concurrent sessions):
  * aggregate **decode** tok/s — total decoded tokens / decode-phase wall-time
    (from vLLM per-request metrics: last finish − first first-token), the direct
    analog of the Kakeya bench's decode-loop timing (prefill excluded)
  * aggregate end-to-end tok/s (prefill+decode) — secondary
  * parallel speedup vs N=1
  * per-session recall (NIAH needle substring)
  * KV/memory notes (vLLM pre-reserves a KV pool via gpu_memory_utilization;
    contrast with Kakeya's bounded per-session S5 KV)

vLLM is full-KV PagedAttention; Kakeya restored-S5 is bounded-KV with recall
restoration — the comparison is throughput-at-equal-recall and the memory model.
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
    ap.add_argument("--haystack-lines", type=int, default=160)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None,
                    help="e.g. bitsandbytes for 4-bit; default None = bf16")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--sliding-window", type=int, default=0,
                    help="KIE-v2: override the model sliding_window (e.g. 68 for "
                         "Kakeya S5 bounded attention on the vLLM runtime; 0 = "
                         "model default).")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sys.path.insert(0, ".")
    sys.path.insert(0, "sdks/python")
    # Import the submodule directly (not the v04 package __init__, which pulls
    # the full restored-verifier stack) so this runs cleanly in the vLLM venv.
    from inference_engine.v04.niah_eval import make_niah_dataset

    print(f"[vllm-mt] tokenizer {args.verifier_id}", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)

    def encode_chat(text: str) -> List[int]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids[0].tolist()

    pool = make_niah_dataset(n_samples=args.pool,
                             haystack_min_lines=args.haystack_lines,
                             haystack_max_lines=args.haystack_lines, seed=args.seed)
    enc = [(encode_chat(s.prompt_text), s.answer_text) for s in pool]
    lengths = Counter(len(e[0]) for e in enc)
    modal_len, _ = lengths.most_common(1)[0]
    bucket = [(ids, ans) for ids, ans in enc if len(ids) == modal_len]
    print(f"[vllm-mt] modal prompt len={modal_len}, {len(bucket)} equal-length",
          file=sys.stderr, flush=True)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    need = max(batch_sizes)
    while len(bucket) < need:
        bucket += bucket[: need - len(bucket)]

    def recall(token_ids, ans):
        return ans in tok.decode(token_ids, skip_special_tokens=True)

    print(f"[vllm-mt] loading vLLM {args.verifier_id} dtype={args.dtype} "
          f"quant={args.quantization} sliding_window={args.sliding_window or 'default'}",
          file=sys.stderr, flush=True)
    llm_kwargs = dict(model=args.verifier_id, dtype=args.dtype,
                      quantization=args.quantization,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len, enforce_eager=False,
                      disable_log_stats=True)
    if args.sliding_window and args.sliding_window > 0:
        # KIE-v2: Kakeya S5 bounded window on vLLM. gemma-4 nests sliding_window
        # under text_config; override both so the hybrid KV bounds tighter.
        sw = int(args.sliding_window)
        llm_kwargs["hf_overrides"] = {
            "sliding_window": sw, "text_config": {"sliding_window": sw},
        }
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen_tokens,
                        ignore_eos=True)  # match Kakeya: always gen_tokens steps

    # warmup
    llm.generate([TokensPrompt(prompt_token_ids=bucket[0][0])],
                 SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))

    rows: List[Dict[str, Any]] = []
    single: Dict[str, float] = {}
    for N in batch_sizes:
        sel = bucket[:N]
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in sel]
        ans = [a for _, a in sel]
        torch.cuda.reset_peak_memory_stats()
        t_wall0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t_wall0

        # decode-phase window from per-request metrics
        first_tok_times, finish_times, decode_toks, e2e_toks = [], [], 0, 0
        recalls = 0
        for o, a in zip(outs, ans):
            comp = o.outputs[0]
            n_out = len(comp.token_ids)
            e2e_toks += n_out
            decode_toks += max(0, n_out - 1)
            m = o.metrics
            if m is not None and m.first_token_time and m.finished_time:
                first_tok_times.append(m.first_token_time)
                finish_times.append(m.finished_time)
            recalls += 1 if recall(list(comp.token_ids), a) else 0
        if first_tok_times and finish_times:
            decode_window = max(finish_times) - min(first_tok_times)
        else:
            decode_window = wall
        decode_tps = decode_toks / decode_window if decode_window > 0 else 0.0
        e2e_tps = e2e_toks / wall if wall > 0 else 0.0
        rec = recalls / N
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if N == 1:
            single = {"decode": decode_tps, "e2e": e2e_tps}
        row = {
            "agents": N,
            "decode_aggregate_tps": round(decode_tps, 2),
            "e2e_aggregate_tps": round(e2e_tps, 2),
            "recall": round(rec, 3),
            "decode_parallel_speedup_vs_n1":
                round(decode_tps / single["decode"], 2) if single.get("decode") else None,
            "e2e_parallel_speedup_vs_n1":
                round(e2e_tps / single["e2e"], 2) if single.get("e2e") else None,
            "torch_peak_alloc_gb": peak,
        }
        rows.append(row)
        print(f"[vllm-mt] N={N:3d} | decode {row['decode_aggregate_tps']} tok/s "
              f"(x{row['decode_parallel_speedup_vs_n1']}) | e2e "
              f"{row['e2e_aggregate_tps']} tok/s | recall {row['recall']} | "
              f"torch_peak {peak}GB", file=sys.stderr, flush=True)

    # vLLM KV pool facts (architectural memory story)
    kv_note = None
    try:
        cc = llm.llm_engine.cache_config
        kv_note = {"num_gpu_blocks": getattr(cc, "num_gpu_blocks", None),
                   "block_size": getattr(cc, "block_size", None),
                   "gpu_memory_utilization": args.gpu_mem_util}
    except Exception:  # noqa: BLE001
        pass

    report = {
        "kind": "vllm_multitenant_parallel",
        "schema_version": 1,
        "config": {
            "verifier_id": args.verifier_id, "dtype": args.dtype,
            "quantization": args.quantization,
            "sliding_window": args.sliding_window or "default",
            "haystack_lines": args.haystack_lines, "modal_prompt_len": modal_len,
            "gen_tokens": args.gen_tokens, "batch_sizes": batch_sizes,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_model_len": args.max_model_len,
            "note": ("vLLM PagedAttention (full KV) baseline; continuous "
                     "batching is the parallel path. decode tok/s excludes "
                     "prefill via per-request metrics, matching the Kakeya "
                     "decode-loop timing. vLLM pre-reserves a KV pool "
                     "(gpu_memory_utilization), so torch_peak reflects the "
                     "reserved pool, not per-request growth — see kv_pool."),
        },
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "kv_pool": kv_note,
        "results": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[vllm-mt] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
