"""Parity gate: all-MLX DFlash drafter vs the torch reference (Apple Silicon).

Before the all-MLX drafter may carry any throughput claim, it must draft
the SAME tokens as the validated torch implementation on real inputs:
real verifier aux hidden (captured from the MLX Gemma-4 forward), real
shared embed/lm_head, several context lengths and blocks.

Procedure per sample:
  1. Build a NIAH prompt (same generator as the integrated eval; seed
     offset so this never reuses eval prompts).
  2. Capture aux hidden over the prompt from the MLX verifier (component
     A machinery, ``capture_aux_hidden``).
  3. Both drafters build their context K/V from the SAME aux (torch gets
     the bridged copy), then draft ``--n-blocks`` consecutive blocks with
     the same bonus token (verifier greedy next token).
  4. Compare drafted token ids position-by-position.

Gate: token agreement must be >= --min-agreement (default 1.0 — exact).
The report JSON records per-block tokens from both runtimes so any
mismatch is directly inspectable.

Run on the Mac via the bridge preset ``k3-drafter-parity`` or directly:

  PYTHONPATH=.:sdks/python python3 scripts/research/k3_mlx_drafter_parity.py \
      --verifier-path <mlx-4bit-dir> --drafter-id z-lab/gemma-4-26B-A4B-it-DFlash \
      --n-samples 3 --n-blocks 4 --block-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", default="models/gemma-4-26B-A4B-it-mlx-4bit")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--haystack-min-lines", type=int, default=20)
    ap.add_argument("--haystack-max-lines", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-agreement", type=float, default=1.0)
    ap.add_argument("--output",
                    default="results/research/k3_mlx_drafter_parity.json")
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    import torch

    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model,
    )
    from inference_engine.backends.mlx.dflash_drafter import (
        MLXDFlashDrafter, make_native_embed_lm_head,
    )
    from inference_engine.backends.mlx.fused_specdecode import (
        capture_aux_hidden, make_bridge_embed_lm_head,
    )
    from inference_engine.v04 import DFlashDrafter, make_niah_dataset
    from scripts.research.k3_dflash_mlx_bridge import mx_to_torch, torch_to_mx

    print(f"[parity] loading MLX verifier {args.verifier_path}", file=sys.stderr)
    mlx_model, tokenizer = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    embed_scale = float(getattr(text_model, "embed_scale", 1.0))
    softcap = None
    for obj in (getattr(mlx_model, "language_model", None), mlx_model):
        cap = getattr(obj, "final_logit_softcapping", None) if obj is not None else None
        if cap:
            softcap = float(cap); break

    print(f"[parity] loading torch drafter {args.drafter_id}", file=sys.stderr)
    t_drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=torch.float32)
    t_drafter = t_drafter.to("cpu").eval()
    print(f"[parity] loading MLX drafter {args.drafter_id}", file=sys.stderr)
    m_drafter = MLXDFlashDrafter.from_pretrained(args.drafter_id)
    aux_ids = tuple(m_drafter.cfg.aux_layer_ids)

    m_embed, m_head = make_native_embed_lm_head(text_model, softcap=softcap)
    t_embed, t_head = make_bridge_embed_lm_head(
        text_model, mx_to_torch=mx_to_torch, torch_to_mx=torch_to_mx,
        device=torch.device("cpu"), torch_dtype=torch.float32, softcap=softcap)

    samples = make_niah_dataset(
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )

    rows = []
    agree = total = 0
    for i, sample in enumerate(samples):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample.prompt_text}],
            add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        C = len(ids)
        aux_mx = capture_aux_hidden(mlx_model, ids, aux_ids, embed_scale=embed_scale)
        aux_t = [mx_to_torch(a, dtype=torch.float32, device="cpu") for a in aux_mx]

        # Bonus = verifier greedy next token over the prompt.
        out = mlx_model(mx.array([ids])); mx.eval(out)
        bonus = int(mx.argmax(out[0, -1]).item())

        m_ctx = m_drafter.make_context_kv(aux_mx, mx.arange(0, C))
        t_ctx = t_drafter.make_context_kv(aux_t, torch.arange(0, C))

        sample_row: dict = {"sample": i, "context_len": C, "blocks": []}
        ctx_len = C
        cur_bonus = bonus
        for b in range(args.n_blocks):
            t0 = time.perf_counter()
            m_tokens = m_drafter.draft_block_cached(
                m_ctx, cur_bonus, m_embed, m_head,
                block_size=args.block_size, context_len=ctx_len)
            m_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            t_tokens = t_drafter.draft_block_cached(
                t_ctx, cur_bonus, t_embed, t_head,
                block_size=args.block_size, context_len=ctx_len)
            t_s = time.perf_counter() - t0
            matches = sum(1 for a, c in zip(m_tokens, t_tokens) if a == c)
            agree += matches
            total += args.block_size
            sample_row["blocks"].append({
                "bonus": cur_bonus,
                "mlx_tokens": m_tokens,
                "torch_tokens": t_tokens,
                "matches": matches,
                "mlx_draft_s": round(m_s, 4),
                "torch_draft_s": round(t_s, 4),
            })
            # Next block conditions on a longer prefix: feed the torch
            # drafts (the reference) as "committed" by extending both
            # contexts with the same aux slice re-captured from the
            # verifier over prompt+drafts. Keep it simple and equal for
            # both: recompute aux over the extended ids.
            ids = ids + [cur_bonus] + t_tokens[: max(args.block_size - 1, 0)]
            ids = ids[: C + (b + 1) * args.block_size]
            aux_mx = capture_aux_hidden(
                mlx_model, ids, aux_ids, embed_scale=embed_scale)
            aux_t = [mx_to_torch(a, dtype=torch.float32, device="cpu")
                     for a in aux_mx]
            ctx_len = len(ids)
            m_ctx = m_drafter.make_context_kv(aux_mx, mx.arange(0, ctx_len))
            t_ctx = t_drafter.make_context_kv(aux_t, torch.arange(0, ctx_len))
            out = mlx_model(mx.array([ids])); mx.eval(out)
            cur_bonus = int(mx.argmax(out[0, -1]).item())
        rows.append(sample_row)
        print(f"[parity] sample {i}: agreement so far {agree}/{total}",
              file=sys.stderr)

    agreement = agree / max(total, 1)
    report = {
        "kind": "k3_mlx_drafter_parity",
        "schema_version": 1,
        "config": vars(args),
        "agreement": round(agreement, 4),
        "agreed_tokens": agree,
        "total_tokens": total,
        "samples": rows,
        "passed": agreement >= args.min_agreement,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[parity] agreement={agreement:.4f} "
          f"({agree}/{total}) min={args.min_agreement} -> {out_path}",
          file=sys.stderr)
    if agreement < args.min_agreement:
        print("[parity] FAIL: all-MLX drafter does not match the torch "
              "reference; throughput claims are blocked.", file=sys.stderr)
        return 1
    print("[parity] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
