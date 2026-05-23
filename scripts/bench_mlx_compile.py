"""Mac-only benchmark: mx.compile speedup on the MLX proposer.

Runs the speculative decoder twice on the same prompt — once with the
proposer's bidirectional backbone wrapped in ``mx.compile``, once
without — and reports per-config wall time, output token sequences,
and the speedup ratio.

Caveats expected from the math at our current operating point:

  * The compiled path eats a one-time JIT cost on the first forward
    of any new (T_prompt + L_block) shape. For a chat session the
    second turn onwards reuses the cached graph; for a single bench
    we see the JIT cost mixed in with steady-state throughput.
  * mx.compile mostly saves Python kernel-launch overhead, not pure
    matmul FLOPs. M4 base GPU is memory-bandwidth bound on the
    dominant matmuls, so the achievable compile speedup is in the
    1.2–1.8x range, NOT the 2x+ that would be plausible on a
    compute-bound (e.g. H100) backend.

Use this on the Mac mini to quantify the real win:

    source .venv-mac/bin/activate
    PYTHONPATH=. python3 scripts/bench_mlx_compile.py \\
        --prompt 'Why is the sky blue?' \\
        --max-new-tokens 256 \\
        --block-size 16 --num-diffusion-steps 2

Writes JSON to ``results/platform-tests/bench_mlx_compile_<ts>.json``;
returns non-zero if compiled and uncompiled paths emit different token
sequences (a real bug we want surfaced).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _run(proposer, verifier_factory, prompt_ids, max_new, block_size,
         num_diffusion_steps, eos_set):
    from kv_cache_proposer.speculative import SpeculativeDecoder
    verifier = verifier_factory()
    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    # Warm-up: one forward to pay the first-shape JIT cost so the
    # measured wall time reflects steady-state throughput.
    decoder.generate(
        prompt_ids=prompt_ids, max_new_tokens=4, eos_token_ids=eos_set,
    )

    verifier = verifier_factory()  # fresh state for the timed run
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
    return {
        "wall_time_s": elapsed,
        "n_tokens": len(result.output_token_ids),
        "tok_per_s": len(result.output_token_ids) / max(elapsed, 1e-9),
        "acceptance_rate": result.acceptance_rate,
        "proposer_forward_calls": result.proposer_forward_calls,
        "verifier_forward_calls": result.verifier_forward_calls,
        "output_token_ids": result.output_token_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-diffusion-steps", type=int, default=2)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    from inference_engine.backends.mlx.env import probe_environment
    env = probe_environment()
    if not env.is_available:
        print(f"[bench] MLX unavailable: {env.failure_reason}")
        return 2

    print(f"[bench] env: {env.render()}", flush=True)
    print(f"[bench] prompt: {args.prompt!r}", flush=True)
    print(f"[bench] config: L={args.block_size}, K={args.num_diffusion_steps}, "
          f"max_new={args.max_new_tokens}", flush=True)

    from kv_cache_proposer.proposer import ProposerConfig
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
    from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink_size, window_size=args.window_size,
    )

    print("[bench] loading two MLX proposers (compile=True, compile=False) ...",
          flush=True)
    proposer_compiled = MLXSparseLogitsProposer(
        proposer_cfg, compile_backbone=True
    )
    proposer_uncompiled = MLXSparseLogitsProposer(
        proposer_cfg, compile_backbone=False
    )

    # Load the verifier tokenizer for prompt encoding.
    cpu_v_for_tok = SinkWindowVerifier(verifier_cfg)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = cpu_v_for_tok.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    eos_set = _eos_ids(cpu_v_for_tok.tokenizer)
    print(f"[bench] prompt tokens: {len(prompt_ids)}", flush=True)

    mlx_v_factory = lambda: MLXSinkWindowVerifier(verifier_cfg)

    common = dict(
        prompt_ids=prompt_ids,
        max_new=args.max_new_tokens,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
        eos_set=eos_set,
    )

    print("\n[bench] === [A] uncompiled backbone (mx.compile = False) ===",
          flush=True)
    a = _run(proposer_uncompiled, mlx_v_factory, **common)
    print(f"[bench] [A] wall={a['wall_time_s']:6.2f}s  tok/s={a['tok_per_s']:5.2f}  "
          f"tokens={a['n_tokens']}  acc={a['acceptance_rate']:.3f}", flush=True)

    print("\n[bench] === [B] compiled backbone (mx.compile = True) ===", flush=True)
    b = _run(proposer_compiled, mlx_v_factory, **common)
    print(f"[bench] [B] wall={b['wall_time_s']:6.2f}s  tok/s={b['tok_per_s']:5.2f}  "
          f"tokens={b['n_tokens']}  acc={b['acceptance_rate']:.3f}", flush=True)

    print("\n[bench] === comparison ===", flush=True)
    eq = a["output_token_ids"] == b["output_token_ids"]
    speedup = a["wall_time_s"] / max(b["wall_time_s"], 1e-9)
    print(f"[bench] outputs identical: {eq}", flush=True)
    print(f"[bench] wall-time speedup B/A (compile vs none): {speedup:.2f}x",
          flush=True)
    if not eq:
        print(
            f"[bench] FAIL: compiled and uncompiled paths produced "
            f"DIFFERENT token sequences. This is a real correctness bug.",
            flush=True,
        )

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
        "uncompiled": a,
        "compiled": b,
        "comparison": {
            "outputs_identical": eq,
            "wall_time_speedup_compile": speedup,
        },
    }
    if args.report is None:
        repo_root = Path(__file__).resolve().parents[1]
        out = repo_root / "results" / "platform-tests"
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / f"bench_mlx_compile_{int(time.time())}.json"
    else:
        report_path = Path(args.report)
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {report_path}", flush=True)
    return 0 if eq else 3


if __name__ == "__main__":
    sys.exit(main())
