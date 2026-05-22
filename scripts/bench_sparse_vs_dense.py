"""Wall-time benchmark: dense `DLMProposer` vs `SparseLogitsProposer`.

Real Qwen3 weights, no mocks. The benchmark drives the speculative
decoder twice on the same prompt — once with the dense proposer, once
with the sparse proposer — and reports per-call wall time, output
equivalence, peak proposer activation, and acceptance rate.

Use this on the Mac mini to quantify Phase B's actual win on M-series
hardware:

    source .venv-mac/bin/activate
    PYTHONPATH=. python3 scripts/bench_sparse_vs_dense.py \
        --prompt "Why is sky blue?" \
        --max-new-tokens 32 \
        --block-size 8 \
        --num-diffusion-steps 4

The script prints a summary table and writes a structured JSON to
`results/platform-tests/bench_sparse_vs_dense-<ts>.json` so it can be
committed back for cross-host comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from inference_engine.proposer import SparseLogitsProposer
from kv_cache_proposer.proposer import DLMProposer, ProposerConfig
from kv_cache_proposer.speculative import SpeculativeDecoder, SpeculativeRunResult
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _run_one(
    proposer,
    verifier_cfg: VerifierConfig,
    prompt_ids,
    max_new_tokens: int,
    block_size: int,
    num_diffusion_steps: int,
    eos_set,
):
    verifier = SinkWindowVerifier(verifier_cfg)
    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    t0 = time.perf_counter()
    result = decoder.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_set,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Reply with exactly: OK.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--num-diffusion-steps", type=int, default=4)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional path to write JSON results. Defaults to results/platform-tests/bench_sparse_vs_dense-<ts>.json",
    )
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    cfg = ProposerConfig(dtype=dtype, device=args.device)
    vcfg = VerifierConfig(
        dtype=dtype, device=args.device,
        sink_size=args.sink_size, window_size=args.window_size,
    )

    print(f"[bench] device={args.device} dtype={args.dtype}", flush=True)
    print(f"[bench] L={args.block_size}  K={args.num_diffusion_steps}  "
          f"sink={args.sink_size} window={args.window_size}", flush=True)
    print(f"[bench] prompt: {args.prompt!r}", flush=True)
    print(f"[bench] loading dense proposer (DLMProposer) ...", flush=True)
    dense = DLMProposer(cfg)
    print(f"[bench] loading sparse proposer (SparseLogitsProposer) ...", flush=True)
    sparse = SparseLogitsProposer(cfg)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = dense.encode_chat(messages)
    print(f"[bench] prompt token length: {len(prompt_ids)}", flush=True)

    # Use the verifier's tokenizer (loaded transiently via the SinkWindowVerifier
    # construction inside _run_one) for EOS — we just need the chat template
    # terminator. Re-use dense.tokenizer since it's the same Qwen3 family.
    eos_set = _eos_ids(dense.tokenizer)

    # ---- dense run ----
    print("\n[bench] === dense (DLMProposer, full [1,T,V] logits) ===", flush=True)
    dense_result, dense_time = _run_one(
        dense, vcfg, prompt_ids, args.max_new_tokens,
        args.block_size, args.num_diffusion_steps, eos_set,
    )
    print(f"[bench] dense  wall={dense_time:7.2f}s  tokens={len(dense_result.output_token_ids):3d}  "
          f"acc={dense_result.acceptance_rate:.3f}  "
          f"prop_fwd={dense_result.proposer_forward_calls}  "
          f"prop_act_peak={dense_result.proposer_peak_activation_bytes/1024/1024:6.2f} MB", flush=True)

    # ---- sparse run ----
    print("\n[bench] === sparse (SparseLogitsProposer, [1,n_masked,V] logits) ===", flush=True)
    sparse_result, sparse_time = _run_one(
        sparse, vcfg, prompt_ids, args.max_new_tokens,
        args.block_size, args.num_diffusion_steps, eos_set,
    )
    print(f"[bench] sparse wall={sparse_time:7.2f}s  tokens={len(sparse_result.output_token_ids):3d}  "
          f"acc={sparse_result.acceptance_rate:.3f}  "
          f"prop_fwd={sparse_result.proposer_forward_calls}  "
          f"prop_act_peak={sparse_result.proposer_peak_activation_bytes/1024/1024:6.2f} MB", flush=True)

    # ---- comparison ----
    print("\n[bench] === comparison ===", flush=True)
    eq = dense_result.output_token_ids == sparse_result.output_token_ids
    print(f"[bench] output token sequences identical: {eq}", flush=True)
    if not eq:
        # bf16 numerical drift could in principle flip an argmax — log and exit non-zero.
        print(f"[bench]   dense:  {dense_result.output_token_ids[:16]}", flush=True)
        print(f"[bench]   sparse: {sparse_result.output_token_ids[:16]}", flush=True)
    speedup = dense_time / max(sparse_time, 1e-9)
    print(f"[bench] wall-time speedup: {speedup:.2f}x  ({dense_time:.2f}s -> {sparse_time:.2f}s)", flush=True)
    act_ratio = (
        sparse_result.proposer_peak_activation_bytes
        / max(dense_result.proposer_peak_activation_bytes, 1)
    )
    print(f"[bench] activation peak ratio: {act_ratio:.3f}  "
          f"({dense_result.proposer_peak_activation_bytes/1024/1024:.2f} MB -> "
          f"{sparse_result.proposer_peak_activation_bytes/1024/1024:.2f} MB)", flush=True)

    if args.report is None:
        repo_root = Path(__file__).resolve().parents[1]
        rep_dir = repo_root / "results" / "platform-tests"
        rep_dir.mkdir(parents=True, exist_ok=True)
        report = rep_dir / f"bench_sparse_vs_dense-{int(time.time())}.json"
    else:
        report = Path(args.report)

    payload = {
        "config": vars(args),
        "prompt_token_count": len(prompt_ids),
        "dense": {
            "wall_time_s": dense_time,
            "n_tokens": len(dense_result.output_token_ids),
            "acceptance_rate": dense_result.acceptance_rate,
            "proposer_forward_calls": dense_result.proposer_forward_calls,
            "proposer_peak_activation_bytes": dense_result.proposer_peak_activation_bytes,
            "verifier_forward_calls": dense_result.verifier_forward_calls,
            "verifier_peak_kv_bytes": dense_result.verifier_peak_kv_bytes,
            "output_token_ids": dense_result.output_token_ids,
        },
        "sparse": {
            "wall_time_s": sparse_time,
            "n_tokens": len(sparse_result.output_token_ids),
            "acceptance_rate": sparse_result.acceptance_rate,
            "proposer_forward_calls": sparse_result.proposer_forward_calls,
            "proposer_peak_activation_bytes": sparse_result.proposer_peak_activation_bytes,
            "verifier_forward_calls": sparse_result.verifier_forward_calls,
            "verifier_peak_kv_bytes": sparse_result.verifier_peak_kv_bytes,
            "output_token_ids": sparse_result.output_token_ids,
        },
        "comparison": {
            "outputs_identical": eq,
            "wall_time_speedup": speedup,
            "activation_peak_ratio_sparse_over_dense": act_ratio,
        },
    }
    report.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {report}", flush=True)

    return 0 if eq else 2


if __name__ == "__main__":
    sys.exit(main())
