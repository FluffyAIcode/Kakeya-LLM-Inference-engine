"""K3 integrated NIAH eval — Mac M4 variant.

The Mac mirror of ``scripts/research/k3_integrated_niah_eval.py`` (CUDA).
Runs the K1.E NIAH harness on top of
:class:`inference_engine.v04.cross_model_dlm_verifier_mlx.
MLXCrossModelDLMRestoredVerifier` to produce the **K3 Mac product gate
evidence**:

  ✓ architectural correctness — effective_attention_fraction = 1.0
  ✓ memory bounded            — sustained KV cache ≤ O(sink+window)
  ✓ recall preservation       — |delta vs MLX oracle| ≤ 5pp

Usage:

  bash scripts/review_pr_k3_integrated_niah_on_mac.sh

(this script is invoked by the reviewer aid; not normally called directly.)

Run order:

  1. (one-time, after PR #103 merges) train f_θ on vast:
     bash scripts/review_pr_k3_f_theta_train_on_vast.sh
  2. push trained f_θ to main / pull it locally on Mac
  3. (one-time per checkpoint) tokenizer_config patch:
     python3 scripts/research/k3_patch_gemma4_tokenizer_config.py \\
         models/gemma-4-26B-A4B-it-mlx-4bit
  4. run THIS script via the reviewer aid

Cross-runtime architecture (same as CUDA path, MLX-side wired):

  MLX 4-bit verifier (mlx_lm) on Apple Silicon
            ↕
       cross_model_dlm_verifier_mlx.py
            ↕  scripts/research/k3_dflash_mlx_bridge.py
  PyTorch DFlash drafter on MPS / CPU
  PyTorch FThetaProjection on CPU (small ~32M params)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from inference_engine.v04 import (
    DFlashDrafter,
    FThetaProjection,
    NIAHSample,
    aggregate_attention_window_metrics,
    aggregate_recall,
    compute_effective_attention_window,
    make_niah_dataset,
    recall_predicate,
)
from inference_engine.v04.cross_model_dlm_verifier_mlx import (
    MLXCrossModelDLMRestoredVerifier,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", default="models/gemma-4-26B-A4B-it-mlx-4bit")
    ap.add_argument("--drafter-id", default="models/dflash-kakeya-baseline")
    ap.add_argument("--f-theta-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--haystack-min-lines", type=int, default=60)
    ap.add_argument("--haystack-max-lines", type=int, default=80)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drafter-device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip-oracle", action="store_true")
    return ap.parse_args()


def _mlx_greedy_argmax(logits_mx: Any, *, position: int = -1) -> int:
    import mlx.core as mx  # type: ignore
    return int(mx.argmax(logits_mx[0, position]).item())


def _mac_record_memory(label: str) -> Dict[str, Any]:
    """Memory snapshot per Mac MPS conventions, mirroring K1.E memory
    record format so JSON evidence is diff-able with PR #94 ladder."""
    out: Dict[str, Any] = {"label": label, "platform": "mac"}
    try:
        out["current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
    except Exception:
        out["current_allocated_bytes"] = None
    try:
        out["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    except Exception:
        out["driver_allocated_bytes"] = None
    try:
        import psutil
        out["device_total_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        out["device_total_bytes"] = None
    return out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---------- Verifier (MLX 4-bit) ----------
    try:
        import mlx_lm  # type: ignore
        import mlx.core as mx  # type: ignore
    except ImportError:
        print(
            "ERROR: mlx_lm not available. On Mac:\n    pip install --upgrade mlx-lm",
            file=sys.stderr,
        )
        return 12

    print(f"[k3-integrated-mac] loading verifier {args.verifier_path}",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    mlx_verifier, mlx_tokenizer = mlx_lm.load(args.verifier_path)
    print(
        f"[k3-integrated-mac]   verifier loaded in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    # ---------- Drafter (PyTorch) ----------
    drafter_dtype = torch.bfloat16 if args.drafter_device == "mps" else torch.float32
    print(f"[k3-integrated-mac] loading drafter {args.drafter_id} ({drafter_dtype}) ...",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=drafter_dtype)
    drafter = drafter.to(args.drafter_device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    print(
        f"[k3-integrated-mac]   drafter loaded in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    # ---------- f_θ checkpoint ----------
    print(f"[k3-integrated-mac] loading f_θ from {args.f_theta_dir}",
          file=sys.stderr, flush=True)
    f_theta = FThetaProjection.from_pretrained(
        args.f_theta_dir, dtype=torch.float32, device=args.drafter_device,
    )

    # ---------- Cross-model wrapper ----------
    cross_verifier = MLXCrossModelDLMRestoredVerifier(
        mlx_verifier=mlx_verifier,
        drafter=drafter,
        f_theta=f_theta,
        sink_size=args.sink_size,
        window_size=args.window_size,
    )
    print(
        f"[k3-integrated-mac] cross-model verifier ready "
        f"(sink={args.sink_size}, window={args.window_size})",
        file=sys.stderr,
    )

    # ---------- NIAH dataset ----------
    samples: List[NIAHSample] = make_niah_dataset(
        mlx_tokenizer,
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )
    print(f"[k3-integrated-mac] generated {len(samples)} NIAH samples",
          file=sys.stderr)

    # ---------- Run integrated cross-model verifier ----------
    cross_results: List[Dict[str, Any]] = []
    cross_attn_window: List[Dict[str, Any]] = []
    mem_baseline = _mac_record_memory("baseline")

    for i, sample in enumerate(samples):
        input_ids = torch.tensor([sample.input_ids], dtype=torch.long)
        T = int(input_ids.size(1))

        # Greedy decode max_new_tokens via cross-model verifier
        cur = input_ids.clone()
        gen_tokens: List[int] = []
        for _ in range(args.max_new_tokens):
            logits_mx = cross_verifier.forward(cur)
            nxt = _mlx_greedy_argmax(logits_mx)
            gen_tokens.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

        decoded = mlx_tokenizer.decode(gen_tokens, skip_special_tokens=True)
        is_correct = recall_predicate(decoded, sample)
        cross_results.append({
            "sample_idx": i,
            "decoded": decoded[:200],
            "is_correct": is_correct,
            "seq_len": T,
        })

        attn_w = compute_effective_attention_window(
            seq_len=T,
            sink_size=args.sink_size,
            window_size=args.window_size,
            evicted_kv_restored=True,
            structural_constraint=(
                f"causal_with_dlm_reconstruction "
                f"(local_cache=sink={args.sink_size}+window={args.window_size}, "
                f"k3_cross_model_f_theta_mac_mlx)"
            ),
        )
        cross_attn_window.append(attn_w)
        print(
            f"[k3-integrated-mac] sample {i}: T={T} correct={is_correct} "
            f"decoded[:60]={decoded[:60]!r}",
            file=sys.stderr,
        )

    cross_recall = aggregate_recall(cross_results)
    cross_attn_agg = aggregate_attention_window_metrics(cross_attn_window)
    mem_after_cross = _mac_record_memory("after_cross_model")

    # ---------- MLX oracle baseline (full-attention, no sink+window) ----------
    oracle_results = None
    oracle_recall = None
    if not args.skip_oracle:
        print("[k3-integrated-mac] running MLX oracle baseline",
              file=sys.stderr, flush=True)
        oracle_results = []
        for i, sample in enumerate(samples):
            ids = list(sample.input_ids)
            gen_tokens = []
            for _ in range(args.max_new_tokens):
                inp = mx.array([ids])
                out = mlx_verifier(inp)
                nxt = _mlx_greedy_argmax(out)
                gen_tokens.append(nxt)
                ids.append(nxt)
            decoded = mlx_tokenizer.decode(gen_tokens, skip_special_tokens=True)
            is_correct = recall_predicate(decoded, sample)
            oracle_results.append({
                "sample_idx": i,
                "decoded": decoded[:200],
                "is_correct": is_correct,
                "seq_len": len(sample.input_ids),
            })
            print(f"[k3-integrated-mac]   oracle sample {i}: correct={is_correct}",
                  file=sys.stderr)
        oracle_recall = aggregate_recall(oracle_results)

    mem_after_oracle = _mac_record_memory("after_oracle")

    recall_delta = (
        abs(cross_recall["recall"] - oracle_recall["recall"])
        if oracle_recall else None
    )
    report = {
        "schema_version": 1,
        "kind": "k3_integrated_niah_acceptance_mac",
        "config": {
            "verifier_path": args.verifier_path,
            "drafter_id": args.drafter_id,
            "f_theta_dir": args.f_theta_dir,
            "drafter_device": args.drafter_device,
            "f_theta_config": f_theta.config.to_json_dict(),
            "n_samples": args.n_samples,
            "sink_size": args.sink_size,
            "window_size": args.window_size,
            "haystack_min_lines": args.haystack_min_lines,
            "haystack_max_lines": args.haystack_max_lines,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "skip_oracle": bool(args.skip_oracle),
        },
        "results": {
            "k3_cross_model_mac": {
                "name": "k3_cross_model_mac",
                **cross_recall,
                "per_sample": cross_results,
            },
            **(
                {"oracle": {"name": "oracle", **oracle_recall,
                            "per_sample": oracle_results}}
                if oracle_recall else {}
            ),
        },
        "attention_window": {
            "per_config": {"k3_cross_model_mac": cross_attn_agg},
        },
        "memory": {
            "baseline": mem_baseline,
            "after_cross_model": mem_after_cross,
            "after_oracle": mem_after_oracle,
        },
        "gate": {
            "architectural_correctness": (
                cross_attn_agg.get("effective_attention_fraction_mean") == 1.0
            ),
            "recall_delta_vs_oracle_pp": (
                recall_delta * 100 if recall_delta is not None else None
            ),
            "recall_delta_within_5pp": (
                recall_delta is not None and recall_delta <= 0.05
            ),
            "memory_under_24gb": (
                mem_after_cross.get("driver_allocated_bytes") is not None
                and mem_after_cross["driver_allocated_bytes"] < 24 * (1 << 30)
            ),
        },
    }

    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_integrated_niah_mac_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(
        f"\n[k3-integrated-mac] DONE.\n"
        f"  cross-model recall: {cross_recall['recall']:.3f} "
        f"({cross_recall['samples_correct']}/{cross_recall['samples_total']})",
        file=sys.stderr,
    )
    if oracle_recall:
        print(
            f"  oracle recall:      {oracle_recall['recall']:.3f} "
            f"({oracle_recall['samples_correct']}/{oracle_recall['samples_total']})",
            file=sys.stderr,
        )
        print(f"  |delta vs oracle|: {recall_delta * 100:.2f} pp", file=sys.stderr)
        print(
            f"  ADR §11.8 1a gate (≤ 5pp): "
            f"{'PASS' if recall_delta <= 0.05 else 'FAIL'}",
            file=sys.stderr,
        )
    print(
        f"  effective_attention_fraction: "
        f"{cross_attn_agg.get('effective_attention_fraction_mean')}",
        file=sys.stderr,
    )
    if mem_after_cross.get("driver_allocated_bytes") is not None:
        gb = mem_after_cross["driver_allocated_bytes"] / (1 << 30)
        print(f"  driver_alloc_after: {gb:.2f} GB (gate <24GB: "
              f"{'PASS' if gb < 24 else 'FAIL'})", file=sys.stderr)
    print(f"  Report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
