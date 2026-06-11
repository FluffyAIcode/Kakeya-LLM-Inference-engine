"""K3 Step 3b — native DFlash speculative-decoding acceptance eval on Mac M4.

Mac variant of ``scripts/research/k3_dflash_specdecode_eval.py`` (PR #93,
CUDA). Drives the engine's native PyTorch DFlash drafter
(``inference_engine.v04.dflash_drafter.DFlashDrafter``) against an MLX
4-bit Gemma-4 verifier and measures the same speculative-decoding
acceptance length / acceptance rate as the CUDA path — the metric that
determines DFlash speedup on production hardware.

Runtime architecture (cross-runtime spec decoding, MLX ↔ PyTorch):

    MLX 4-bit verifier (mlx_lm) on Apple Silicon
              ↕  scripts/research/k3_dflash_mlx_bridge.py (numpy bridge)
    PyTorch DFlash drafter on MPS / CPU (PR #93's native impl)

Self-speculative loop (lossless vs greedy AR; matches CUDA path):

  1. MLX verifier forward over `committed` → aux hiddens at
     ``aux_layer_ids = target_layer_ids + 1`` (+1 shift per vLLM
     PR #41703) + bonus token.
  2. Bridge MLX → torch (numpy intermediate).
  3. PyTorch ``DFlashProposer.propose_block(committed, L, num_steps=1)``
     → BlockProposal with L draft tokens.
  4. MLX verifier forward over ``committed + drafts`` → greedy verify,
     accept longest matching prefix.
  5. Commit accepted tokens (+1 always-correct bonus), repeat.

Reports per-prompt and aggregate ``acceptance_rate`` /
``acceptance_length``, plus ``lossless_vs_AR`` check on the same
prompts (run greedy AR over the verifier alone, compare token-for-token
with the spec-decoded output).

Comparable evidence to PR #93's CUDA evidence at
``results/research/k3_dflash_specdecode_corpus_heldout.json``:

    acceptance_rate / acceptance_length / lossless_vs_AR

Mac-specific environment notes:

  * Verifier: MLX 4-bit at ~13 GB weights. Combined with the drafter
    bf16 (~0.9 GB) + activations during forward, peak memory measured
    2026-06-09 was 15.49 GB on Mac M4 24 GB during verifier-only
    smoke. Spec decoding adds drafter activations during draft_block
    (small; drafter is 0.43B vs verifier's 26B). Joint peak expected
    in the 18-22 GB range — fits Mac M4 24 GB tight.

  * Tokenizer config patch: required at first run via
    ``scripts/research/k3_patch_gemma4_tokenizer_config.py`` (PR #101)
    to convert ``extra_special_tokens`` from list to dict. Idempotent.

  * Drafter device: MPS bf16 (or CPU fp32 via ``--drafter-device cpu``
    if MPS is misbehaving).

  * Default ``--drafter-state`` is the alignment-trained baseline
    shipped on main (``models/dflash-kakeya-baseline/``). Override
    with the upstream HF id ``z-lab/gemma-4-26B-A4B-it-DFlash``
    only for research-baseline comparison (not alignment-trained).

Usage (after PR #101's tokenizer patch ran once):

    HF_TOKEN=hf_xxx PYTHONPATH=.:sdks/python python3 \\
        scripts/research/k3_dflash_specdecode_eval_mac.py \\
        --max-new-tokens 48 --block-size 16 --num-steps 1 --n-prompts 4 \\
        --output results/research/k3_dflash_specdecode_mac_<stamp>.json

Validation gate (Mac M4 user run):

    end-to-end on the user's hardware — produces JSON with the same
    schema as PR #93's CUDA evidence. Acceptance rate is expected to
    be in the same neighbourhood as the CUDA held-out (~10.7%, with
    the same training-data-corpus-size limitation), since the
    verifier and drafter weights are identical and only the runtime
    differs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import torch

from inference_engine.v04.dflash_drafter import DFlashDrafter, DFlashProposer

# Local research-stage bridge (not engine API surface yet).
sys.path.insert(0, str(Path(__file__).parent))
from k3_dflash_mlx_bridge import (  # type: ignore  # noqa: E402
    MLXVerifierAuxProvider,
    build_mlx_verifier_callbacks,
    mlx_verify_block,
)


# Same prompt corpora as the CUDA path — direct comparability of evidence.
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a haiku about speculative decoding.",
]

HELD_OUT_PROMPTS = [
    "Write a Python function that counts vowels in a string.",
    "Name two differences between a list and a tuple in Python.",
    "What is the boiling point of water at sea level in Celsius?",
    "Write a short rhyming couplet about a robot learning to paint.",
    "Explain what a database index is and when to use one.",
    "Who painted the Mona Lisa?",
    "Write a function that returns whether a year is a leap year.",
    "Give two reasons code review is valuable.",
]


def mlx_greedy_ar(
    mlx_model, prompt_ids: List[int], max_new_tokens: int, eos_ids,
) -> tuple:
    """Greedy AR baseline on the MLX verifier alone — used for the
    ``lossless_vs_AR`` check matching PR #93's CUDA path.
    """
    import mlx.core as mx  # type: ignore
    cur = list(prompt_ids)
    forwards = 0
    for _ in range(max_new_tokens):
        inp = mx.array([cur])
        logits = mlx_model(inp)
        forwards += 1
        nxt = int(mx.argmax(logits[0, -1]).item())
        cur.append(nxt)
        if nxt in eos_ids:
            break
    return cur[len(prompt_ids):], forwards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verifier-path",
        default="models/gemma-4-26B-A4B-it-mlx-4bit",
        help="Local MLX 4-bit verifier directory (default: standard Mac path).",
    )
    ap.add_argument(
        "--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash",
        help="DFlash drafter source — local path or HF id. Default: the "
             "alignment-trained baseline on main (post PR #93 + #99 merge).",
    )
    ap.add_argument(
        "--drafter-device", default="mps", choices=["mps", "cpu"],
        help="Device for the PyTorch drafter. Default: MPS (Apple Silicon).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=1)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument(
        "--held-out", action="store_true",
        help="Use HELD_OUT_PROMPTS (disjoint from PR #93 alignment trainer's "
             "prompt corpus) for honest generalisation evidence.",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    prompts = HELD_OUT_PROMPTS if args.held_out else PROMPTS

    print(f"[k3-sd-mac] verifier_path: {args.verifier_path}", file=sys.stderr)
    print(f"[k3-sd-mac] drafter_id:    {args.drafter_id}", file=sys.stderr)
    print(f"[k3-sd-mac] drafter_device: {args.drafter_device}", file=sys.stderr)

    # ---------- Verifier (MLX 4-bit) ----------
    try:
        import mlx_lm  # type: ignore
    except ImportError:
        print("ERROR: mlx_lm not installed. On Mac: pip install --upgrade mlx-lm",
              file=sys.stderr)
        return 12

    print(f"[k3-sd-mac] loading MLX verifier ...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    mlx_model, mlx_tokenizer = mlx_lm.load(args.verifier_path)
    print(
        f"[k3-sd-mac]   loaded in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    # ---------- Drafter (PyTorch) ----------
    drafter_dtype = torch.bfloat16 if args.drafter_device == "mps" else torch.float32
    print(f"[k3-sd-mac] loading DFlash drafter ({drafter_dtype}) ...",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    drafter = DFlashDrafter.from_pretrained(
        args.drafter_id, dtype=drafter_dtype,
    ).to(args.drafter_device).eval()
    print(
        f"[k3-sd-mac]   loaded in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    cfg = drafter.cfg
    hidden = cfg.hidden_size
    softcap = cfg.final_logit_softcapping
    aux_layer_ids = list(cfg.aux_layer_ids)

    # Bridge: MLX verifier embed/lm_head → torch callbacks for drafter.
    embed_fn, lm_head_fn = build_mlx_verifier_callbacks(
        mlx_model, hidden_size=hidden, softcap=softcap,
        bridge_dtype=drafter_dtype, bridge_device=args.drafter_device,
    )

    aux_provider = MLXVerifierAuxProvider(
        mlx_model, aux_layer_ids,
        bridge_dtype=torch.float32, bridge_device=args.drafter_device,
    )

    proposer = DFlashProposer(drafter, aux_provider, embed_fn, lm_head_fn)

    eos_ids = set(
        x for x in [
            mlx_tokenizer.eos_token_id,
            getattr(mlx_tokenizer, "eot_token_id", None),
        ]
        if x is not None
    )

    # ---------- Spec decode loop ----------
    per_prompt = []
    tot_accepted = tot_drafted = tot_blocks = 0
    tot_spec_forwards = 0
    lossless = True

    for pi in range(min(args.n_prompts, len(prompts))):
        prompt = prompts[pi]
        msgs = [{"role": "user", "content": prompt}]
        enc = mlx_tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
        )
        # mlx_lm's apply_chat_template returns either a list[int] or a tensor
        if hasattr(enc, "tolist"):
            ids = enc.tolist() if not isinstance(enc, list) else enc
        else:
            ids = list(enc)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]

        committed = list(ids)
        generated: List[int] = []
        blk_accepts: List[int] = []
        aux_provider.forward_calls = 0
        t0 = time.perf_counter()

        while len(generated) < args.max_new_tokens:
            L = min(args.block_size, args.max_new_tokens - len(generated))
            # Aux-hidden + bonus from MLX verifier.
            aux_ctx, bonus_token_id = aux_provider.aux_hidden_context(committed)
            tot_spec_forwards += 1  # aux/prefill forward

            # DFlashProposer.propose_block uses these aux + bonus to draft L
            # tokens via a single non-causal forward through the PyTorch
            # drafter. num_steps is accepted for interface compatibility
            # (DFlash reference uses 1 pass).
            proposal = proposer.propose_block(
                committed_token_ids=committed,
                block_size=L,
                num_steps=args.num_steps,
            )
            drafts = proposal.tokens

            # Candidate = [bonus, drafts...]; verify greedily on MLX verifier.
            candidate = [bonus_token_id] + list(drafts)
            accepted, _correction = mlx_verify_block(mlx_model, committed, candidate)
            tot_spec_forwards += 1  # verify forward
            accepted = max(accepted, 1)  # bonus is guaranteed-correct
            committed += candidate[:accepted]
            generated += candidate[:accepted]
            draft_accepted = accepted - 1  # exclude the always-correct bonus
            blk_accepts.append(draft_accepted)
            tot_accepted += draft_accepted
            tot_drafted += L
            tot_blocks += 1
            if any(t in eos_ids for t in candidate[:accepted]):
                break

        spec_time = time.perf_counter() - t0
        spec_out = generated[: args.max_new_tokens]

        # ---------- AR baseline (lossless check) ----------
        ar_out, ar_forwards = mlx_greedy_ar(
            mlx_model, ids, len(spec_out), eos_ids,
        )
        match = spec_out[: len(ar_out)] == ar_out[: len(spec_out)]
        lossless = lossless and match
        mean_acc = sum(blk_accepts) / max(len(blk_accepts), 1)
        per_prompt.append({
            "prompt": prompt,
            "blocks": len(blk_accepts),
            "block_accepts": blk_accepts,
            "mean_accepted_per_block": mean_acc,
            "tokens_generated": len(spec_out),
            "verifier_aux_forwards": aux_provider.forward_calls,
            "lossless_vs_ar": match,
            "spec_seconds": spec_time,
            "decoded": mlx_tokenizer.decode(spec_out, skip_special_tokens=True)[:200] if hasattr(mlx_tokenizer, "decode") else "",
        })
        print(
            f"[k3-sd-mac] prompt {pi}: blocks={len(blk_accepts)} "
            f"mean_accept={mean_acc:.2f} accepts={blk_accepts} "
            f"lossless={match} spec_time={spec_time:.1f}s",
            file=sys.stderr,
        )

    acc_rate = tot_accepted / max(tot_drafted, 1)
    acc_length = (tot_accepted + tot_blocks) / max(tot_blocks, 1)

    report = {
        "schema_version": 1,
        "kind": "k3_dflash_specdecode_acceptance_mac",
        "config": {
            "verifier_path": args.verifier_path,
            "drafter_id": args.drafter_id,
            "drafter_device": args.drafter_device,
            "drafter_dtype": str(drafter_dtype),
            "block_size": args.block_size,
            "num_steps": args.num_steps,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": min(args.n_prompts, len(prompts)),
            "aux_layer_ids": aux_layer_ids,
            "held_out": bool(args.held_out),
        },
        "aggregate": {
            "acceptance_rate": acc_rate,
            "acceptance_length": acc_length,
            "total_accepted": tot_accepted,
            "total_drafted": tot_drafted,
            "total_blocks": tot_blocks,
            "lossless_vs_ar": lossless,
            "reference_humaneval": {"acceptance_length": 7.7, "acceptance_rate": 0.447},
            "reference_cuda_held_out": {
                "acceptance_length": 2.45, "acceptance_rate": 0.107,
                "source": "results/research/k3_dflash_specdecode_corpus_heldout.json",
                "note": "PR #93 CUDA held-out baseline; same drafter weights, "
                        "same verifier weights — Mac result should be in the "
                        "same neighbourhood (modulo MLX 4-bit vs CUDA bf16 "
                        "numerical differences).",
            },
        },
        "per_prompt": per_prompt,
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_dflash_specdecode_mac_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[k3-sd-mac] AGGREGATE acceptance_rate={acc_rate:.3f} "
        f"acceptance_length={acc_length:.2f} lossless={lossless} "
        f"(CUDA held-out ref: 0.107 / 2.45)  -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
