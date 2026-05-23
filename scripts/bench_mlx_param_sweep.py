"""Mac-only parameter sweep: find the best (block_size, diffusion_steps) for
the current MLX proposer + verifier on a given prompt.

The current default ``L=16, K=10`` is what the original Phase B bench
used. With acceptance ~0.07 (this proposer + Qwen3-1.7B verifier
without Repr-Align alignment), large L wastes compute: the expected
accepted-tokens-per-block ``(1-α^L)/(1-α)`` saturates near 1/(1-α)
for any L >> 1. Smaller L + smaller K hits the same per-block yield
with less waste.

This script sweeps L ∈ {2, 4, 8, 16} × K ∈ {2, 4, 8} and reports
wall time, tokens/sec, acceptance, and proposer-forward count for
each combination. Writes JSON to
``results/platform-tests/bench_param_sweep_<ts>.json`` for review.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from inference_engine.backends.mlx.env import probe_environment


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="是解释如何节省kv cache，而不是解释kv cache的作用")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--block-sizes", type=str, default="2,4,8,16")
    ap.add_argument("--num-diffusion-steps", type=str, default="2,4,8")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    env = probe_environment()
    if not env.is_available:
        print(f"[sweep] MLX unavailable: {env.failure_reason}")
        return 2
    print(f"[sweep] env: {env.render()}", flush=True)

    from kv_cache_proposer.proposer import ProposerConfig
    from kv_cache_proposer.speculative import SpeculativeDecoder
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
    from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink_size, window_size=args.window_size,
    )

    print("[sweep] loading proposer + verifier (one-time) ...", flush=True)
    # Use the CPU verifier just for tokenizer access (doesn't run inference).
    cpu_v_tok = SinkWindowVerifier(verifier_cfg)
    mlx_proposer = MLXSparseLogitsProposer(proposer_cfg)
    print("[sweep] models loaded.\n", flush=True)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = cpu_v_tok.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    eos_set = _eos_ids(cpu_v_tok.tokenizer)
    print(f"[sweep] prompt tokens: {len(prompt_ids)}", flush=True)

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    num_steps_list = [int(x) for x in args.num_diffusion_steps.split(",")]

    results = []
    print(
        f"\n  {'L':>3} {'K':>3} {'wall_s':>8} {'tok/s':>7} "
        f"{'acc':>6} {'p_fwd':>6} {'v_fwd':>6} {'tokens':>7}",
        flush=True,
    )
    print("  " + "-" * 50, flush=True)

    for L in block_sizes:
        for K in num_steps_list:
            if K > L:
                continue  # K is clamped to <= L by propose_block; skip dups
            # Fresh verifier per run for clean stats.
            verifier = MLXSinkWindowVerifier(verifier_cfg)
            decoder = SpeculativeDecoder(
                proposer=mlx_proposer,
                verifier=verifier,
                block_size=L,
                num_diffusion_steps=K,
            )
            t0 = time.perf_counter()
            r = decoder.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_set,
            )
            wall = time.perf_counter() - t0
            n = len(r.output_token_ids)
            tps = n / max(wall, 1e-9)
            row = {
                "block_size": L,
                "num_diffusion_steps": K,
                "wall_time_s": wall,
                "tokens_per_sec": tps,
                "acceptance_rate": r.acceptance_rate,
                "proposer_forward_calls": r.proposer_forward_calls,
                "verifier_forward_calls": r.verifier_forward_calls,
                "n_tokens": n,
                "output_token_ids": r.output_token_ids,
            }
            results.append(row)
            print(
                f"  {L:>3} {K:>3} {wall:>8.2f} {tps:>7.2f} "
                f"{r.acceptance_rate:>6.3f} {r.proposer_forward_calls:>6d} "
                f"{r.verifier_forward_calls:>6d} {n:>7d}",
                flush=True,
            )

    # Pick the best by tok/s
    best = max(results, key=lambda r: r["tokens_per_sec"])
    print(
        f"\n[sweep] best: L={best['block_size']} K={best['num_diffusion_steps']}  "
        f"wall={best['wall_time_s']:.2f}s  tok/s={best['tokens_per_sec']:.2f}",
        flush=True,
    )

    payload = {
        "config": vars(args),
        "env": {
            "mlx_version": env.mlx_version,
            "mlx_lm_version": env.mlx_lm_version,
        },
        "prompt_token_count": len(prompt_ids),
        "results": results,
        "best": {
            "block_size": best["block_size"],
            "num_diffusion_steps": best["num_diffusion_steps"],
            "wall_time_s": best["wall_time_s"],
            "tokens_per_sec": best["tokens_per_sec"],
            "speedup_vs_L16K10": (
                next(
                    (r["wall_time_s"] for r in results
                     if r["block_size"] == 16 and r["num_diffusion_steps"] == 10),
                    None,
                ) or 0.0
            ) / max(best["wall_time_s"], 1e-9),
        },
    }
    if args.report is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "results" / "platform-tests"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"bench_param_sweep_{int(time.time())}.json"
    else:
        report_path = Path(args.report)
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[sweep] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
