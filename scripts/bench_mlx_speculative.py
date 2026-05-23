"""Mac-only end-to-end speculative bench: MLX vs PyTorch CPU.

Drives the same prompt through three configurations:

  1. PyTorch CPU verifier + PyTorch CPU sparse proposer (the Phase B
     baseline; this is what produced the 127 s / 48 token result on
     the M4 mini)
  2. MLX verifier + PyTorch CPU sparse proposer (cross-backend; what
     MLX-1b shipped)
  3. MLX verifier + MLX sparse proposer (full MLX path; what MLX-1c
     adds)

Reports per-config wall time, acceptance rate, output token sequences,
and a tabular summary so the speedup of each step is visible. Writes
JSON to ``results/platform-tests/bench_mlx_speculative_<ts>.json``.

Returns non-zero if any path produces an output that disagrees with
the PyTorch CPU baseline at the FIRST generated token (a real bug
indicator — bf16 noise alone shouldn't flip the first pick).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from inference_engine.backends.mlx.env import probe_environment
from inference_engine.proposer import SparseLogitsProposer
from kv_cache_proposer.proposer import DLMProposer, ProposerConfig
from kv_cache_proposer.speculative import SpeculativeDecoder
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig


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
    verifier = verifier_factory()
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
        "output_token_ids": result.output_token_ids,
        "acceptance_rate": result.acceptance_rate,
        "proposer_forward_calls": result.proposer_forward_calls,
        "verifier_forward_calls": result.verifier_forward_calls,
        "verifier_peak_kv_bytes": result.verifier_peak_kv_bytes,
        "verifier_peak_activation_bytes": result.verifier_peak_activation_bytes,
        "proposer_peak_activation_bytes": result.proposer_peak_activation_bytes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument(
        "--max-new-tokens", type=int, default=256,
        help=(
            "Hard cap on generated tokens. Default raised from 32 to 256 "
            "so EOS naturally terminates short-to-medium answers; bumped to "
            "512+ for long-form outputs. Mid-sentence truncation at low "
            "values is the dominant 'output looks cut off' UX issue."
        ),
    )
    ap.add_argument(
        "--block-size", type=int, default=16,
        help="Empirical sweet spot per the param sweep on M4 (L=16, K=2).",
    )
    ap.add_argument(
        "--num-diffusion-steps", type=int, default=2,
        help="K=2 is the param-sweep optimum: acceptance is ~0.07 "
             "regardless of K (proposer-quality limit) so doubling K "
             "doubles wall time without buying anything.",
    )
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--report", default=None)
    ap.add_argument(
        "--skip-cpu-cpu", action="store_true",
        help="Skip the PyTorch CPU verifier + CPU proposer baseline (slow on Mac)",
    )
    args = ap.parse_args()

    env = probe_environment()
    if not env.is_available:
        print(f"[bench] MLX unavailable: {env.failure_reason}")
        return 2

    print(f"[bench] env: {env.render()}", flush=True)
    print(f"[bench] prompt: {args.prompt!r}", flush=True)

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink_size, window_size=args.window_size,
    )

    # Common: tokenize prompt with the CPU verifier's tokenizer.
    print(f"[bench] loading PyTorch CPU verifier (for tokenizer + baseline) ...", flush=True)
    cpu_v = SinkWindowVerifier(verifier_cfg)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = cpu_v.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    eos_set = _eos_ids(cpu_v.tokenizer)
    print(f"[bench] prompt tokens: {len(prompt_ids)}  eos: {eos_set}", flush=True)

    print(f"[bench] loading PyTorch CPU sparse proposer ...", flush=True)
    cpu_proposer = SparseLogitsProposer(proposer_cfg)

    print(f"[bench] loading MLX verifier ...", flush=True)
    from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier

    print(f"[bench] loading MLX sparse proposer (downloading dllm-hub safetensors) ...", flush=True)
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
    mlx_proposer = MLXSparseLogitsProposer(proposer_cfg)

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
        "results": {},
    }

    common = dict(
        prompt_ids=prompt_ids,
        max_new=args.max_new_tokens,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
        eos_set=eos_set,
    )

    cpu_v_factory = lambda: SinkWindowVerifier(verifier_cfg)
    mlx_v_factory = lambda: MLXSinkWindowVerifier(verifier_cfg)

    if not args.skip_cpu_cpu:
        print("\n[bench] === [A] PyTorch CPU verifier + PyTorch CPU sparse proposer (Phase B baseline) ===", flush=True)
        a = _run(cpu_proposer, cpu_v_factory, **common)
        payload["results"]["cpu_v__cpu_p"] = a
        print(f"[bench] [A] wall={a['wall_time_s']:6.2f}s  tokens={a['n_tokens']}  acc={a['acceptance_rate']:.3f}", flush=True)
    else:
        a = None
        payload["results"]["cpu_v__cpu_p"] = None

    print("\n[bench] === [B] MLX verifier + PyTorch CPU sparse proposer (MLX-1b cross-backend) ===", flush=True)
    b = _run(cpu_proposer, mlx_v_factory, **common)
    payload["results"]["mlx_v__cpu_p"] = b
    print(f"[bench] [B] wall={b['wall_time_s']:6.2f}s  tokens={b['n_tokens']}  acc={b['acceptance_rate']:.3f}", flush=True)

    print("\n[bench] === [C] MLX verifier + MLX sparse proposer (MLX-1c full path) ===", flush=True)
    c = _run(mlx_proposer, mlx_v_factory, **common)
    payload["results"]["mlx_v__mlx_p"] = c
    print(f"[bench] [C] wall={c['wall_time_s']:6.2f}s  tokens={c['n_tokens']}  acc={c['acceptance_rate']:.3f}", flush=True)

    print("\n[bench] === Summary ===", flush=True)
    if a is not None:
        print(f"  [A] CPU/CPU  : wall={a['wall_time_s']:7.2f}s  acc={a['acceptance_rate']:.3f}", flush=True)
    print(f"  [B] MLX/CPU  : wall={b['wall_time_s']:7.2f}s  acc={b['acceptance_rate']:.3f}", flush=True)
    print(f"  [C] MLX/MLX  : wall={c['wall_time_s']:7.2f}s  acc={c['acceptance_rate']:.3f}", flush=True)
    if a is not None:
        speedup_b = a["wall_time_s"] / max(b["wall_time_s"], 1e-9)
        speedup_c = a["wall_time_s"] / max(c["wall_time_s"], 1e-9)
        print(f"  speedup B / A = {speedup_b:5.2f}x", flush=True)
        print(f"  speedup C / A = {speedup_c:5.2f}x", flush=True)
        payload["summary"] = {
            "speedup_B_over_A": speedup_b,
            "speedup_C_over_A": speedup_c,
        }
    speedup_c_over_b = b["wall_time_s"] / max(c["wall_time_s"], 1e-9)
    print(f"  speedup C / B = {speedup_c_over_b:5.2f}x  (the MLX-1c proposer-port win)", flush=True)
    payload.setdefault("summary", {})["speedup_C_over_B"] = speedup_c_over_b

    # First-token correctness gate
    if a is not None and a["output_token_ids"] and c["output_token_ids"]:
        if a["output_token_ids"][0] != c["output_token_ids"][0]:
            print(f"[bench] FAIL: first-token differs A vs C: "
                  f"a={a['output_token_ids'][0]} c={c['output_token_ids'][0]}", flush=True)
            return 3

    if args.report is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "results" / "platform-tests"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"bench_mlx_speculative_{int(time.time())}.json"
    else:
        report_path = Path(args.report)
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
