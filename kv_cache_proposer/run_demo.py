"""End-to-end demo: DLM Proposer + AR Verifier with sink+window KV.

Runs the same prompt through:

    (1) the verifier alone with full DynamicCache (baseline)
    (2) the proposer + verifier speculative decoder with sink+window KV

Then prints:

  * the generated text from each path,
  * a token-level equivalence check (under greedy decoding the two outputs
    must share their prefix; with a large enough window they are identical),
  * the NBT report (KV bytes/token, with proposer overhead accounted for).

There is no mock and no fallback: every forward pass runs the real model
weights, and any inconsistency raises immediately rather than silently
degrading to plain AR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import List

import torch

from kv_cache_proposer.proposer import DLMProposer, ProposerConfig
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig
from kv_cache_proposer.baseline import BaselineDecoder, BaselineConfig
from kv_cache_proposer.speculative import SpeculativeDecoder
from kv_cache_proposer.metrics import NBTReport


PROMPTS = [
    "Explain in two short sentences what an attention sink is in transformer models.",
    "List three properties of prime numbers, one per line, no extra commentary.",
    "Write a short Python function add(a, b) that returns a + b. No tests.",
]


def _eos_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    # Qwen3 chat template terminator
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposer-id",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help="HF id for the DLM proposer.",
    )
    parser.add_argument(
        "--verifier-id",
        default="Qwen/Qwen3-1.7B",
        help="HF id for the AR verifier.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-diffusion-steps", type=int, default=16)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument(
        "--batch-size-for-amortization",
        type=int,
        default=1,
        help=(
            "Operating-point batch size used solely to amortize proposer "
            "weight/activation bytes in the NBT formula."
        ),
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        choices=list(range(len(PROMPTS))),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional override; if set, this single prompt is used.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="If set, write results dict to this JSON path.",
    )
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[demo] proposer={args.proposer_id}", flush=True)
    print(f"[demo] verifier={args.verifier_id}", flush=True)
    print(f"[demo] device={args.device}  dtype={args.dtype}", flush=True)
    print(f"[demo] block_size={args.block_size}  diffusion_steps={args.num_diffusion_steps}", flush=True)
    print(f"[demo] sink={args.sink_size}  window={args.window_size}", flush=True)
    print(f"[demo] max_new_tokens={args.max_new_tokens}", flush=True)

    # ---------------- proposer ---------------- #
    print("[demo] loading proposer ...", flush=True)
    proposer = DLMProposer(
        ProposerConfig(model_id=args.proposer_id, dtype=dtype, device=args.device)
    )
    print(
        f"[demo] proposer params: {proposer.stats.weight_bytes/1e6:.1f} MB",
        flush=True,
    )

    # ---------------- verifier ---------------- #
    print("[demo] loading verifier ...", flush=True)
    verifier = SinkWindowVerifier(
        VerifierConfig(
            model_id=args.verifier_id,
            dtype=dtype,
            device=args.device,
            sink_size=args.sink_size,
            window_size=args.window_size,
        )
    )
    print(
        f"[demo] verifier params: {verifier.stats.weight_bytes/1e6:.1f} MB",
        flush=True,
    )

    # ---------------- baseline (re-uses verifier.tokenizer) ---------------- #
    print("[demo] loading baseline (full-KV verifier) ...", flush=True)
    baseline = BaselineDecoder(
        BaselineConfig(model_id=args.verifier_id, dtype=dtype, device=args.device)
    )

    # We build the prompt with the *verifier* tokenizer, then encode the same
    # prompt with the *proposer* tokenizer for its independent input. Both
    # tokenizers come from the Qwen3 family and share BPE merges.
    if args.prompt is not None:
        user_text = args.prompt
    else:
        user_text = PROMPTS[args.prompt_index]

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_text},
    ]
    prompt_ids_verifier = verifier.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
    )
    prompt_ids_proposer = proposer.encode_chat(messages)
    if prompt_ids_verifier != prompt_ids_proposer:
        # Both come from the Qwen3-0.6B family tokenizer (same merges) but the
        # MDLM checkpoint adds a <|mask|> id. They must agree on prompt tokens.
        raise RuntimeError(
            "Tokenizer mismatch between proposer and verifier on the prompt: "
            f"proposer-len={len(prompt_ids_proposer)}, verifier-len={len(prompt_ids_verifier)}"
        )
    eos_ids = _eos_ids(verifier.tokenizer)

    print(f"[demo] prompt: {user_text!r}", flush=True)
    print(f"[demo] prompt token length: {len(prompt_ids_verifier)}", flush=True)
    print(f"[demo] eos ids: {eos_ids}", flush=True)

    # ---------------- baseline run ---------------- #
    print("\n[demo] === Baseline (full-KV greedy AR) ===", flush=True)
    baseline_result = baseline.generate(
        prompt_ids=prompt_ids_verifier,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    baseline_text = verifier.tokenizer.decode(
        baseline_result.output_token_ids, skip_special_tokens=False
    )
    print(f"[baseline] tokens: {len(baseline_result.output_token_ids)}", flush=True)
    print(f"[baseline] forward calls: {baseline_result.forward_calls}", flush=True)
    print(
        f"[baseline] peak KV: {baseline_result.peak_kv_bytes/1024:.1f} KB "
        f"({baseline_result.peak_kv_bytes/baseline_result.final_kv_token_count:.1f} B/token)",
        flush=True,
    )
    print(f"[baseline] text: {baseline_text!r}", flush=True)

    # ---------------- speculative run ---------------- #
    print("\n[demo] === Speculative (DLM proposer + sink+window verifier) ===", flush=True)
    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    spec_result = decoder.generate(
        prompt_ids=prompt_ids_verifier,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    spec_text = verifier.tokenizer.decode(
        spec_result.output_token_ids, skip_special_tokens=False
    )
    print(f"[spec] tokens: {len(spec_result.output_token_ids)}", flush=True)
    print(f"[spec] proposer forwards: {spec_result.proposer_forward_calls}", flush=True)
    print(f"[spec] verifier forwards: {spec_result.verifier_forward_calls}", flush=True)
    print(f"[spec] verifier peak KV: {spec_result.verifier_peak_kv_bytes/1024:.1f} KB", flush=True)
    print(f"[spec] verifier final KV slots: {spec_result.verifier_final_kv_token_count}", flush=True)
    print(f"[spec] acceptance rate: {spec_result.acceptance_rate:.3f}", flush=True)
    print(
        f"[spec] per-block accepted/proposed: "
        f"{list(zip(spec_result.accepted_per_block, spec_result.proposed_per_block))}",
        flush=True,
    )
    print(f"[spec] text: {spec_text!r}", flush=True)

    # ---------------- NBT report ---------------- #
    report = NBTReport.compute(
        speculative=spec_result,
        baseline=baseline_result,
        sink_size=args.sink_size,
        window_size=args.window_size,
        block_size=args.block_size,
        batch_size=args.batch_size_for_amortization,
    )
    print()
    print(report.render())

    if args.results_json:
        out = {
            "config": vars(args),
            "prompt": user_text,
            "prompt_tokens": len(prompt_ids_verifier),
            "eos_ids": eos_ids,
            "baseline": {
                "text": baseline_text,
                "n_tokens": len(baseline_result.output_token_ids),
                "peak_kv_bytes": baseline_result.peak_kv_bytes,
                "final_kv_token_count": baseline_result.final_kv_token_count,
                "forward_calls": baseline_result.forward_calls,
            },
            "speculative": {
                "text": spec_text,
                "n_tokens": len(spec_result.output_token_ids),
                "verifier_peak_kv_bytes": spec_result.verifier_peak_kv_bytes,
                "verifier_final_kv_token_count": spec_result.verifier_final_kv_token_count,
                "verifier_forward_calls": spec_result.verifier_forward_calls,
                "proposer_forward_calls": spec_result.proposer_forward_calls,
                "acceptance_rate": spec_result.acceptance_rate,
                "accepted_per_block": spec_result.accepted_per_block,
                "proposed_per_block": spec_result.proposed_per_block,
                "wall_time_seconds": spec_result.wall_time_seconds,
            },
            "nbt": asdict(report),
        }
        with open(args.results_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[demo] wrote {args.results_json}", flush=True)

    # Exit code: 0 if greedy outputs match (the "no intelligence loss" guarantee
    # in the regime where sink+window covers the full sequence), 2 otherwise.
    full_seq_len = len(prompt_ids_verifier) + len(baseline_result.output_token_ids)
    cache_budget = args.sink_size + args.window_size
    if cache_budget >= full_seq_len:
        if not report.output_exact_match:
            print(
                f"\n[demo] FAIL: sink+window={cache_budget} >= full_seq_len={full_seq_len} "
                f"yet outputs differ; this violates the equivalence theorem.",
                flush=True,
            )
            return 2
        else:
            print(
                "\n[demo] PASS: equivalence-regime test (sink+window covers full sequence) "
                "produced bit-identical greedy output.",
                flush=True,
            )
    else:
        print(
            f"\n[demo] info: sink+window={cache_budget} < full_seq_len={full_seq_len}; "
            f"some context is evicted by design — output may differ from baseline.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
