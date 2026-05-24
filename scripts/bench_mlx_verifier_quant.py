"""Mac-only bench: bf16 vs 4-bit MLX verifier, same proposer, same prompt.

Loads two ``MLXSinkWindowVerifier`` instances back-to-back —
``Qwen/Qwen3-1.7B`` (bf16) and ``mlx-community/Qwen3-1.7B-4bit``
by default — drives the same prompt through both with the same MLX
sparse proposer, and reports:

    * weight bytes (verifier-only) and quantization metadata
    * wall time, generated tokens, acceptance rate
    * peak KV bytes, peak activation bytes
    * per-config token sequence (so output equivalence can be inspected
      by eye even though bf16 vs 4-bit is not bit-equivalent)

This is the empirical evidence that supports ADR 0002 §2.2's "60 %
memory rule" (bf16 path on a 24 GB Mac risks swap; 4-bit comfortably
fits with ~14 GB headroom). Run before / after every change to the
verifier or proposer that might affect memory or wall time.

Output goes to JSON at ``results/platform-tests/bench_mlx_verifier_quant_<ts>.json``
when ``--report`` is set; otherwise prints to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from inference_engine.backends.mlx.env import probe_environment
from kv_cache_proposer.proposer import ProposerConfig
from kv_cache_proposer.speculative import SpeculativeDecoder
from kv_cache_proposer.verifier import VerifierConfig


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _run_one(label, verifier, proposer, prompt_ids, max_new, block_size,
             num_diffusion_steps, eos_set):
    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    t0 = time.perf_counter()
    result = decoder.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new,
        eos_token_ids=eos_set,
    )
    elapsed = time.perf_counter() - t0
    quant = verifier.quantization
    return {
        "label": label,
        "verifier_id": verifier.config.model_id,
        "wall_time_s": elapsed,
        "n_tokens": len(result.output_token_ids),
        "tok_per_s": len(result.output_token_ids) / max(elapsed, 1e-9),
        "acceptance_rate": result.acceptance_rate,
        "proposer_forward_calls": result.proposer_forward_calls,
        "verifier_forward_calls": result.verifier_forward_calls,
        "verifier_peak_kv_bytes": result.verifier_peak_kv_bytes,
        "verifier_peak_activation_bytes": result.verifier_peak_activation_bytes,
        "proposer_peak_activation_bytes": result.proposer_peak_activation_bytes,
        "verifier_weight_bytes": verifier.stats.weight_bytes,
        "quantization": {
            "is_quantized": quant.is_quantized,
            "bits": quant.bits,
            "group_size": quant.group_size,
            "quantized_weight_bytes": quant.quantized_weight_bytes,
            "full_precision_weight_bytes": quant.full_precision_weight_bytes,
            "total_weight_bytes": quant.total_weight_bytes,
            "full_precision_param_count": quant.full_precision_param_count,
            "quantized_param_count": quant.quantized_param_count,
            "effective_bits_per_param": quant.effective_bits_per_param,
        },
        "output_token_ids": result.output_token_ids,
    }


def _format_summary(rows: list) -> str:
    """Two-line-per-row tabular summary printed to stdout."""
    out = []
    out.append("")
    out.append(f"{'config':<12} {'verifier':<40} {'GB':>5} {'bits':>5} {'wall_s':>7} {'tok':>5} {'acc':>5} {'tok/s':>7}")
    out.append("-" * 96)
    for r in rows:
        gb = r["verifier_weight_bytes"] / 1e9
        if r["quantization"]["is_quantized"]:
            bits = f"{r['quantization']['effective_bits_per_param']:.2f}"
        else:
            bits = "16.0"
        out.append(
            f"{r['label']:<12} {r['verifier_id']:<40} "
            f"{gb:5.2f} {bits:>5} {r['wall_time_s']:7.2f} "
            f"{r['n_tokens']:5d} {r['acceptance_rate']:5.3f} {r['tok_per_s']:7.2f}"
        )
    out.append("")
    if len(rows) >= 2:
        bf16, q = rows[0], rows[1]
        weight_ratio = bf16["verifier_weight_bytes"] / max(q["verifier_weight_bytes"], 1)
        time_ratio = q["wall_time_s"] / max(bf16["wall_time_s"], 1e-9)
        out.append(
            f"weight memory: {weight_ratio:.2f}x reduction (bf16 / 4-bit)"
        )
        out.append(
            f"wall time   : {time_ratio:.2f}x slowdown (4-bit / bf16) — expected "
            f"~1.4-1.7x for Qwen3-1.7B per ADR 0002 §2"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bf16-verifier-id", default="Qwen/Qwen3-1.7B",
        help="HuggingFace repo id for the bf16 verifier baseline.",
    )
    ap.add_argument(
        "--quant-verifier-id", default="mlx-community/Qwen3-1.7B-4bit",
        help="HuggingFace repo id for the quantized verifier (4-bit by "
             "default; ADR 0002 §2.2 mandates 4-bit for verifiers >= 4 B).",
    )
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Hard cap on generated tokens.",
    )
    ap.add_argument(
        "--block-size", type=int, default=16,
        help="L; per the param sweep on M4, L=16 K=2 is fastest for the "
             "current proposer.",
    )
    ap.add_argument(
        "--num-diffusion-steps", type=int, default=2,
        help="K=2 is the param-sweep optimum.",
    )
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    env = probe_environment()
    if not env.is_available:
        print(f"[bench] MLX unavailable: {env.failure_reason}")
        return 2
    print(f"[bench] env: {env.render()}", flush=True)
    print(f"[bench] prompt: {args.prompt!r}", flush=True)
    print(f"[bench] bf16 verifier:  {args.bf16_verifier_id}", flush=True)
    print(f"[bench] quant verifier: {args.quant_verifier_id}", flush=True)

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    bf16_cfg = VerifierConfig(
        model_id=args.bf16_verifier_id,
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink_size, window_size=args.window_size,
    )
    quant_cfg = VerifierConfig(
        model_id=args.quant_verifier_id,
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink_size, window_size=args.window_size,
    )

    # Late import — Mac-only modules.
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
    from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier

    print("[bench] loading MLX sparse proposer ...", flush=True)
    proposer = MLXSparseLogitsProposer(proposer_cfg)

    rows = []

    print("\n[bench] === [bf16] verifier ===", flush=True)
    bf16_v = MLXSinkWindowVerifier(bf16_cfg)
    print(f"[bench] bf16 verifier quant: {bf16_v.quantization.render_short()}", flush=True)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = bf16_v.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=False, enable_thinking=False,
    )
    eos_set = _eos_ids(bf16_v.tokenizer)
    print(f"[bench] prompt tokens: {len(prompt_ids)}  eos: {eos_set}", flush=True)

    common = dict(
        prompt_ids=prompt_ids,
        max_new=args.max_new_tokens,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
        eos_set=eos_set,
    )
    rows.append(_run_one("bf16", bf16_v, proposer, **common))
    print(f"[bench] [bf16] wall={rows[-1]['wall_time_s']:6.2f}s  tokens={rows[-1]['n_tokens']}  "
          f"acc={rows[-1]['acceptance_rate']:.3f}", flush=True)

    # Free bf16 verifier before loading quantized one. Each is ~3.5 GB
    # for Qwen3-1.7B bf16 / ~1 GB for 4-bit; keeping both resident is
    # fine on 24 GB but wasteful — the bench compares wall-time, not
    # peak-resident, so we drop bf16 explicitly before measuring 4-bit.
    del bf16_v

    print("\n[bench] === [quant] verifier ===", flush=True)
    quant_v = MLXSinkWindowVerifier(quant_cfg)
    print(f"[bench] quant verifier quant: {quant_v.quantization.render_short()}", flush=True)
    rows.append(_run_one("quant", quant_v, proposer, **common))
    print(f"[bench] [quant] wall={rows[-1]['wall_time_s']:6.2f}s  tokens={rows[-1]['n_tokens']}  "
          f"acc={rows[-1]['acceptance_rate']:.3f}", flush=True)

    summary_text = _format_summary(rows)
    print(summary_text, flush=True)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = int(time.time())
        report_path = Path("results/platform-tests") / f"bench_mlx_verifier_quant_{ts}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": vars(args),
        "env": {
            "mlx_version": env.mlx_version,
            "mlx_lm_version": env.mlx_lm_version,
            "platform": env.platform_str,
            "machine": env.machine,
            "python": env.python_version,
        },
        "prompt_token_count": len(prompt_ids),
        "results": rows,
        "summary_text": summary_text,
    }
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[bench] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
