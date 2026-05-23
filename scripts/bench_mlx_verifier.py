"""Mac-only benchmark: MLX verifier vs PyTorch CPU verifier.

Drives both verifiers on the same prompt with the same parameters, then
reports per-call wall time, output equivalence (argmax agreement on the
first generated token), peak resident memory, and KV size after a
typical generation pattern.

Usage:
    source .venv-mac/bin/activate
    PYTHONPATH=. python3 scripts/bench_mlx_verifier.py \
        --prompt "Why is the sky blue?" \
        --max-new-tokens 32 \
        --sink-size 4 --window-size 64

Writes a structured JSON report to
`results/platform-tests/bench_mlx_verifier_<ts>.json` so the result
can be committed back for cross-host comparison. Exits non-zero if the
two verifiers disagree on the first-token argmax (which would indicate
a real backend bug, not bf16 noise — Qwen3-1.7B's first-token margin
is much larger than bf16 reduction error).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig
from inference_engine.backends.mlx.env import probe_environment
from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _greedy_generate(verifier, max_new_tokens: int, eos_set):
    """Pure greedy autoregressive generation (no proposer).

    Calls `forward_block([t])` for each emitted token. Returns the
    token list and total wall time."""
    out = []
    t0 = time.perf_counter()
    while len(out) < max_new_tokens:
        tok = int(torch.argmax(verifier.next_token_logits).item())
        out.append(tok)
        if tok in eos_set:
            break
        verifier.append_token(tok)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def _bench_one(verifier, prompt_ids, max_new_tokens, eos_set):
    t0 = time.perf_counter()
    verifier.prefill(prompt_ids)
    prefill_time = time.perf_counter() - t0
    out, gen_time = _greedy_generate(verifier, max_new_tokens, eos_set)
    return {
        "prefill_time_s": prefill_time,
        "generation_time_s": gen_time,
        "wall_time_s": prefill_time + gen_time,
        "output_token_ids": out,
        "n_tokens": len(out),
        "peak_kv_bytes": verifier.stats.peak_kv_bytes,
        "peak_activation_bytes": verifier.stats.peak_activation_bytes,
        "weight_bytes": verifier.stats.weight_bytes,
        "forward_calls": verifier.stats.forward_calls,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    env = probe_environment()
    if not env.is_available:
        print(f"[bench] MLX unavailable: {env.failure_reason}")
        return 2

    cpu_cfg = VerifierConfig(
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink_size, window_size=args.window_size,
    )

    print(f"[bench] env: {env.render()}", flush=True)
    print(f"[bench] loading PyTorch CPU verifier ...", flush=True)
    cpu_v = SinkWindowVerifier(cpu_cfg)
    print(f"[bench] loading MLX verifier ...", flush=True)
    mlx_v = MLXSinkWindowVerifier(cpu_cfg)
    print(f"[bench] both loaded.", flush=True)

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
    print(f"[bench] prompt tokens: {len(prompt_ids)}  eos_ids: {eos_set}", flush=True)

    print("\n[bench] === PyTorch CPU ===", flush=True)
    cpu_result = _bench_one(cpu_v, prompt_ids, args.max_new_tokens, eos_set)
    print(f"[bench] cpu  prefill={cpu_result['prefill_time_s']:6.2f}s  "
          f"gen={cpu_result['generation_time_s']:6.2f}s  "
          f"wall={cpu_result['wall_time_s']:6.2f}s  "
          f"tokens={cpu_result['n_tokens']:3d}", flush=True)

    print("\n[bench] === MLX (Apple Silicon) ===", flush=True)
    mlx_result = _bench_one(mlx_v, prompt_ids, args.max_new_tokens, eos_set)
    print(f"[bench] mlx  prefill={mlx_result['prefill_time_s']:6.2f}s  "
          f"gen={mlx_result['generation_time_s']:6.2f}s  "
          f"wall={mlx_result['wall_time_s']:6.2f}s  "
          f"tokens={mlx_result['n_tokens']:3d}", flush=True)

    print("\n[bench] === comparison ===", flush=True)
    eq = cpu_result["output_token_ids"] == mlx_result["output_token_ids"]
    common_prefix = 0
    for a, b in zip(cpu_result["output_token_ids"], mlx_result["output_token_ids"]):
        if a == b:
            common_prefix += 1
        else:
            break
    speedup_wall = cpu_result["wall_time_s"] / max(mlx_result["wall_time_s"], 1e-9)
    speedup_gen = cpu_result["generation_time_s"] / max(mlx_result["generation_time_s"], 1e-9)
    print(f"[bench] outputs identical: {eq}  (common prefix length: {common_prefix})", flush=True)
    print(f"[bench] wall-time speedup:        {speedup_wall:5.2f}x", flush=True)
    print(f"[bench] generation-only speedup:  {speedup_gen:5.2f}x", flush=True)

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
        "cpu": cpu_result,
        "mlx": mlx_result,
        "comparison": {
            "outputs_identical": eq,
            "common_prefix_length": common_prefix,
            "wall_time_speedup": speedup_wall,
            "generation_speedup": speedup_gen,
        },
    }

    if args.report is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "results" / "platform-tests"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"bench_mlx_verifier_{int(time.time())}.json"
    else:
        report_path = Path(args.report)
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {report_path}", flush=True)

    # Exit non-zero on argmax-disagreement (real bug indicator). The
    # first emitted token is what we trust most; it's allowed to diverge
    # later via accumulated bf16 noise, but if even the first picks
    # disagree there's a backend issue.
    if cpu_result["output_token_ids"] and mlx_result["output_token_ids"]:
        if cpu_result["output_token_ids"][0] != mlx_result["output_token_ids"][0]:
            print(
                f"[bench] FAIL: first-token argmax differs: "
                f"cpu={cpu_result['output_token_ids'][0]} "
                f"mlx={mlx_result['output_token_ids'][0]}",
                flush=True,
            )
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
