"""K1.E NIAH validation runner — Gemma 3-1B-it on Mac M4 (or CUDA).

ADR 0008 §11.8 gate (a): mid-context recall ≥ 95 % at 100k-token
context. This script provides the runnable harness; it loads
google/gemma-3-1b-it, generates a NIAH dataset at the requested
context length, and evaluates three configurations:

  (a) full-attention oracle (model.forward) — upper bound, target ≈ 1.0
  (b) v0.3 sink+window (sink=4, window=64) — confirms the regression,
      target ≈ 0.17 per the 2026-06-06 A/B benchmark
  (c) v0.4 DLMRestoredVerifier (sink=4, window=64 + dLM K/V
      restoration) — the hypothesis under test, gate target ≥ 0.95

Outputs a structured JSON report under results/research/ along with
a stderr-tee'd log. The Mac M4 reviewer
(scripts/review_pr_k1e_on_mac.sh) wraps this script with the
boilerplate.

Defaults are conservative for a Mac M4 24 GB box. Larger contexts
(64k, 100k) require either explicit --haystack-* flags + sufficient
memory or a beefier device (vast.ai A100/H100). A 100k-token oracle
forward on Gemma 3-1B-it bf16 needs ~10 GB just for the KV cache;
v0.4 DLMRestoredVerifier sustained memory is constant in context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default="google/gemma-3-1b-it",
        help="HF model id (gated; needs HF_TOKEN). v0.4 K1 uses the same "
             "checkpoint for proposer and verifier (identity projection f_θ).",
    )
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
    )
    ap.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of NIAH samples to evaluate per configuration.",
    )
    ap.add_argument(
        "--haystack-min-lines", type=int, default=60,
        help="Minimum padding-line count per haystack. ~12-15 tokens / line.",
    )
    ap.add_argument(
        "--haystack-max-lines", type=int, default=80,
    )
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-v03", action="store_true",
        help="Skip the v0.3 sink+window baseline (it's slow and the "
             "result is well-established at ~0.17 per the A/B benchmark).",
    )
    ap.add_argument(
        "--skip-v04", action="store_true",
        help="Skip v0.4. Useful for quick smoke testing of the harness.",
    )
    ap.add_argument(
        "--skip-oracle", action="store_true",
        help="Skip the full-attention oracle. Not recommended; the "
             "oracle is the upper-bound reference.",
    )
    ap.add_argument(
        "--attn-impl",
        choices=["eager", "sdpa"],
        default="eager",
        help="HF transformers attention implementation for the wrapped model. "
             "'eager' (default) materialises the full [B, H, T, T] attention "
             "matrix per layer — fits comfortably at <= 16k context but OOMs "
             "long-context oracle/v0.3/v0.4 forwards on a single H200 at >= 88k "
             "tokens (62 GB just for one layer's attention matrix at 88k bf16). "
             "'sdpa' uses HF's memory-efficient scaled-dot-product-attention path; "
             "the K1.D patched forward already dispatches through ALL_ATTENTION_"
             "FUNCTIONS[impl] when impl != 'eager', so v0.4 K/V Restoration also "
             "works under SDPA. Use 'sdpa' for the 64k+ context rungs that "
             "validate ADR 0008 §11.8 gate (a) at canonical scale.",
    )
    ap.add_argument(
        "--output", default=None,
        help="JSON report path. Default: results/research/k1e_niah_<stamp>.json",
    )
    return ap.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()

    device = pick_device(args.device)
    print(f"[k1e] device={device}", file=sys.stderr)

    print(f"[k1e] loading {args.model}", file=sys.stderr, flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma3.modeling_gemma3 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    print(f"[k1e] attn_implementation={args.attn_impl}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn_impl,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    from inference_engine.v04 import (
        DLMRestoredVerifier,
        NIAHEvalResult,
        evaluate,
        format_memory_summary,
        greedy_decode_oracle,
        greedy_decode_sink_window,
        greedy_decode_v04,
        make_niah_dataset,
        record_memory,
        reset_memory_peak,
    )

    samples = make_niah_dataset(
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )

    # Encode prompts via chat template (ADR 0008 §2.4 / R1b Bug C: the
    # runtime is template-free; the harness applies the template).
    def encode_chat(prompt_text: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt_text}]
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(ids, list):
            ids = torch.tensor([ids])
        return ids.to(device)

    # Print summary of the evaluation set
    sample_ids = [encode_chat(s.prompt_text) for s in samples]
    seq_lens = [int(t.size(1)) for t in sample_ids]
    print(
        f"[k1e] dataset: {len(samples)} samples, prompt token len "
        f"min={min(seq_lens)} max={max(seq_lens)} mean={sum(seq_lens)//len(seq_lens)}",
        file=sys.stderr,
    )

    # K1.G: baseline memory snapshot. Captured BEFORE any config
    # runs, after model + tokenizer + dataset are loaded — represents
    # the minimum sustained working set for this run. Per-config
    # peak is reported relative to this baseline so the
    # constant-memory claim of ADR 0008 §11.5 §"Five properties"
    # item 1 is empirically verifiable from the JSON evidence.
    reset_memory_peak(device)
    baseline_memory = record_memory(device)
    print(
        f"[k1e] baseline memory after model+dataset load: "
        f"{format_memory_summary(baseline_memory)}",
        file=sys.stderr,
    )

    results = {}
    memory_per_config = {}

    # ----------------------------------------------------------------
    # (a) full-attention oracle
    # ----------------------------------------------------------------
    if not args.skip_oracle:
        print("[k1e] (a) full-attention oracle", file=sys.stderr, flush=True)

        def oracle_decode(sample) -> Tuple[str, float]:
            idx = samples.index(sample)
            prompt_ids = sample_ids[idx]
            t0 = time.perf_counter()
            text = greedy_decode_oracle(
                model=model, prompt_ids=prompt_ids, tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            return text, time.perf_counter() - t0

        reset_memory_peak(device)
        oracle = evaluate("oracle_full_attention", samples, oracle_decode)
        oracle_memory = record_memory(device)
        results["oracle_full_attention"] = _result_to_dict(oracle)
        memory_per_config["oracle_full_attention"] = oracle_memory
        print(
            f"[k1e]    oracle recall={oracle.recall:.3f} "
            f"({oracle.samples_correct}/{oracle.samples_total})  "
            f"mean_latency={oracle.mean_latency_s:.2f}s",
            file=sys.stderr,
        )
        print(
            f"[k1e]    oracle memory:  {format_memory_summary(oracle_memory)}",
            file=sys.stderr,
        )

    # ----------------------------------------------------------------
    # (b) v0.3 sink+window baseline
    # ----------------------------------------------------------------
    if not args.skip_v03:
        print(
            f"[k1e] (b) v0.3 sink+window (sink={args.sink_size}, "
            f"window={args.window_size})", file=sys.stderr, flush=True,
        )

        def v03_decode(sample) -> Tuple[str, float]:
            idx = samples.index(sample)
            prompt_ids = sample_ids[idx]
            t0 = time.perf_counter()
            text = greedy_decode_sink_window(
                model=model, prompt_ids=prompt_ids, tokenizer=tokenizer,
                sink_size=args.sink_size, window_size=args.window_size,
                is_gemma3=True,
                max_new_tokens=args.max_new_tokens,
            )
            return text, time.perf_counter() - t0

        reset_memory_peak(device)
        v03 = evaluate("v03_sink_window", samples, v03_decode)
        v03_memory = record_memory(device)
        results["v03_sink_window"] = _result_to_dict(v03)
        memory_per_config["v03_sink_window"] = v03_memory
        print(
            f"[k1e]    v0.3 recall={v03.recall:.3f} "
            f"({v03.samples_correct}/{v03.samples_total})  "
            f"mean_latency={v03.mean_latency_s:.2f}s",
            file=sys.stderr,
        )
        print(
            f"[k1e]    v0.3 memory:   {format_memory_summary(v03_memory)}",
            file=sys.stderr,
        )

    # ----------------------------------------------------------------
    # (c) v0.4 DLMRestoredVerifier
    # ----------------------------------------------------------------
    if not args.skip_v04:
        print(
            f"[k1e] (c) v0.4 DLMRestoredVerifier (sink={args.sink_size}, "
            f"window={args.window_size})", file=sys.stderr, flush=True,
        )
        verifier = DLMRestoredVerifier(
            model, sink_size=args.sink_size, window_size=args.window_size,
        )

        def v04_decode(sample) -> Tuple[str, float]:
            idx = samples.index(sample)
            prompt_ids = sample_ids[idx]
            t0 = time.perf_counter()
            text = greedy_decode_v04(
                verifier=verifier, prompt_ids=prompt_ids, tokenizer=tokenizer,
                apply_rotary_pos_emb=apply_rotary_pos_emb,
                eager_attention_forward=eager_attention_forward,
                all_attention_functions=ALL_ATTENTION_FUNCTIONS,
                max_new_tokens=args.max_new_tokens,
            )
            return text, time.perf_counter() - t0

        reset_memory_peak(device)
        v04 = evaluate("v04_dlm_restored", samples, v04_decode)
        v04_memory = record_memory(device)
        results["v04_dlm_restored"] = _result_to_dict(v04)
        memory_per_config["v04_dlm_restored"] = v04_memory
        print(
            f"[k1e]    v0.4 recall={v04.recall:.3f} "
            f"({v04.samples_correct}/{v04.samples_total})  "
            f"mean_latency={v04.mean_latency_s:.2f}s",
            file=sys.stderr,
        )
        print(
            f"[k1e]    v0.4 memory:   {format_memory_summary(v04_memory)}",
            file=sys.stderr,
        )

    # ----------------------------------------------------------------
    # Gate evaluation (only meaningful if both oracle and v04 ran)
    # ----------------------------------------------------------------
    gate = {}
    if "oracle_full_attention" in results and "v04_dlm_restored" in results:
        oracle_recall = results["oracle_full_attention"]["recall"]
        v04_recall = results["v04_dlm_restored"]["recall"]
        v04_vs_oracle = v04_recall - oracle_recall
        # ADR 0008 §11.8 gate (a): >= 95% at 100k. We don't always run
        # at 100k, so we report at the run's actual context length.
        gate["v04_vs_oracle_delta"] = v04_vs_oracle
        gate["v04_recall_ge_0_95"] = v04_recall >= 0.95
        gate["v04_within_5pct_of_oracle"] = (oracle_recall - v04_recall) <= 0.05
        if "v03_sink_window" in results:
            v03_recall = results["v03_sink_window"]["recall"]
            gate["v04_vs_v03_improvement"] = v04_recall - v03_recall
            gate["v04_dominates_v03"] = v04_recall > v03_recall

    report = {
        # schema v2: K1.G adds 'baseline_memory' and 'memory_per_config'.
        # v1 consumers must default the memory blocks to {} on read.
        "schema_version": 2,
        "kind": "k1e_niah_validation",
        "config": {
            "model": args.model,
            "device": str(device),
            "dtype": str(dtype),
            "attn_impl": args.attn_impl,
            "n_samples": args.n_samples,
            "haystack_min_lines": args.haystack_min_lines,
            "haystack_max_lines": args.haystack_max_lines,
            "sink_size": args.sink_size,
            "window_size": args.window_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "prompt_token_len_min": min(seq_lens),
            "prompt_token_len_max": max(seq_lens),
            "prompt_token_len_mean": sum(seq_lens) // len(seq_lens),
        },
        "results": results,
        "memory": {
            "baseline": baseline_memory,
            "per_config": memory_per_config,
        },
        "gate": gate,
    }

    output_path = (
        Path(args.output) if args.output is not None
        else Path(f"results/research/k1e_niah_{int(time.time())}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[k1e] report -> {output_path}", file=sys.stderr)

    # Top-line summary
    print("[k1e] ─── SUMMARY ──────────────────────────────────────", file=sys.stderr)
    for name, r in results.items():
        mem = memory_per_config.get(name, {})
        mem_str = ""
        if mem.get("device_kind") == "cuda" and mem.get("peak_allocated_bytes") is not None:
            mem_str = f"  peak_mem={mem['peak_allocated_bytes'] / 1e9:.2f}GB"
        elif mem.get("device_kind") == "mps" and mem.get("current_allocated_bytes") is not None:
            mem_str = f"  current_mem={mem['current_allocated_bytes'] / 1e9:.2f}GB"
        print(
            f"[k1e]   {name:<24s}  recall={r['recall']:.3f}  "
            f"mean_latency={r['mean_latency_s']:.2f}s{mem_str}",
            file=sys.stderr,
        )
    if gate:
        print("[k1e] Gate predicates:", file=sys.stderr)
        for k, v in gate.items():
            print(f"[k1e]   {k}: {v}", file=sys.stderr)
    return 0


def _result_to_dict(r) -> dict:
    return {
        "name": r.name,
        "recall": r.recall,
        "samples_correct": r.samples_correct,
        "samples_total": r.samples_total,
        "mean_latency_s": r.mean_latency_s,
        "median_latency_s": r.median_latency_s,
        "per_sample_decoded": r.per_sample_decoded,
        "per_sample_correct": r.per_sample_correct,
    }


if __name__ == "__main__":
    sys.exit(main())
