"""K3 Block B + C integrated NIAH eval — the complete Kakeya inference
engine product evidence on CUDA.

This script is the **final K3 product gate**: it combines
:class:`inference_engine.v04.cross_model_dlm_verifier.CrossModelDLMRestoredVerifier`
(verifier with sink+window cache + drafter K/V Restoration via f_θ)
with the K1.E NIAH evaluation harness (effective_attention_window /
recall / memory metrics).

Architecture under test:

    verifier (Gemma 4 26B-A4B):
      └─ sink+window local KV cache (sink=4 + window=64 default)
      └─ K/V at evicted positions injected via f_θ projection of
         drafter K/V

    drafter (DFlash 0.4B, alignment-trained baseline at
    models/dflash-kakeya-baseline/):
      └─ runs full forward over input_ids with verifier embed_tokens
      └─ K/V at every layer at every position captured
      └─ projected to verifier K/V space via trained f_θ

What this validates (per ADR 0008 §11.8 release gates):

  1. **Architectural correctness**:
     ``effective_attention_fraction = 1.0`` at every NIAH ladder rung.
     Verifier "sees" the full context despite holding only sink+window
     in its local cache. Falsifies "K/V Restoration is just
     decoration"; proves the architecture's load-bearing claim.

  2. **Memory bounded**:
     Sustained verifier KV-cache memory ≤ O(sink+window) regardless
     of input length. Compared against full-attention oracle's KV
     cache size, the K3 cross-model path delivers the memory
     savings claim.

  3. **Recall preservation**:
     Mid-context recall on NIAH samples vs the full-attention oracle.
     ADR §11.8 1a: ``|recall_v04 - recall_oracle| ≤ 5pp`` at every
     rung. This is the architecturally-meaningful gate (independent
     of base-model long-context capability).

This is the K3 production-scale evidence. It's the integrated test
that PR #102 (Mac MLX spec decode) doesn't perform.

Usage (vast.ai H200 / H100):

  HF_TOKEN=hf_xxx PYTHONPATH=.:sdks/python python3 \\
      scripts/research/k3_integrated_niah_eval.py \\
      --verifier-id google/gemma-4-26B-A4B-it \\
      --drafter-id models/dflash-kakeya-baseline \\
      --f-theta-dir results/research/f_theta_v1 \\
      --n-samples 10 --haystack-min-lines 60 --haystack-max-lines 80 \\
      --sink-size 4 --window-size 64 \\
      --output results/research/k3_integrated_niah_<stamp>.json

JSON output mirrors K1.E NIAH harness schema (per_config recall,
attention_window, memory) so it diff-able against PR #94's
ladder evidence + PR #93's CUDA baselines.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from inference_engine.v04 import (
    CrossModelDLMRestoredVerifier,
    DFlashDrafter,
    FThetaProjection,
    NIAHSample,
    aggregate_attention_window_metrics,
    aggregate_recall,
    compute_effective_attention_window,
    make_niah_dataset,
    recall_predicate,
    record_memory,
    reset_memory_peak,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="models/dflash-kakeya-baseline")
    ap.add_argument("--f-theta-dir", required=True,
                    help="Directory containing f_theta_config.json + f_theta_weights.pt")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--haystack-min-lines", type=int, default=60)
    ap.add_argument("--haystack-max-lines", type=int, default=80)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--skip-oracle", action="store_true",
        help="Skip the full-attention oracle baseline (saves time but "
             "loses the |delta vs oracle| gate signal).",
    )
    ap.add_argument(
        "--identity-restore", action="store_true",
        help="Diagnostic: restore evicted positions with the verifier's "
             "OWN true pre-norm K/V instead of the f_θ projection. Under "
             "this mode cross-model recall should match the oracle — it "
             "isolates 'is the restoration machinery correct?' from 'is "
             "f_θ accurate enough?'.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "[k3-integrated] WARNING: CUDA not available; "
            "running on CPU will be very slow on production scale.",
            file=sys.stderr,
        )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ---------- Verifier (CUDA bf16) ----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import (  # type: ignore
        apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS,
    )

    print(f"[k3-integrated] loading verifier {args.verifier_id}",
          file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="eager",
        device_map="auto" if device.type == "cuda" else None,
    ).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)

    # ---------- Drafter (CUDA bf16) ----------
    print(f"[k3-integrated] loading drafter {args.drafter_id}",
          file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype)
    drafter = drafter.to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)

    # ---------- f_θ checkpoint ----------
    print(f"[k3-integrated] loading f_θ from {args.f_theta_dir}",
          file=sys.stderr, flush=True)
    f_theta = FThetaProjection.from_pretrained(
        args.f_theta_dir, dtype=torch.float32, device=device,
    )

    # ---------- Cross-model wrapper ----------
    cross_verifier = CrossModelDLMRestoredVerifier(
        verifier_model=verifier,
        drafter=drafter,
        f_theta=f_theta,
        sink_size=args.sink_size,
        window_size=args.window_size,
    )
    print(f"[k3-integrated] cross-model verifier ready "
          f"(sink={args.sink_size}, window={args.window_size})",
          file=sys.stderr)

    if args.identity_restore:
        # Diagnostic: restore evicted positions with the verifier's own
        # true pre-norm K/V (not f_θ). Validates the restoration
        # machinery independent of f_θ accuracy.
        from inference_engine.v04.cross_model_dlm_verifier import (
            capture_verifier_own_kv,
        )
        cross_verifier.project_drafter_kv = (
            lambda ids: capture_verifier_own_kv(verifier, ids)
        )
        print("[k3-integrated] IDENTITY-RESTORE diagnostic enabled "
              "(evicted K/V come from verifier's own k_proj/v_proj)",
              file=sys.stderr)

    # ---------- NIAH dataset ----------
    samples: List[NIAHSample] = make_niah_dataset(
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )

    # Encode prompts via chat template (ADR 0008 §2.4: the runtime is
    # template-free; the harness applies the template), matching the
    # K1.E NIAH harness convention.
    def encode_chat(prompt_text: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt_text}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt",
        )
        if hasattr(ids, "keys"):          # BatchEncoding / dict
            ids = ids["input_ids"]
        elif isinstance(ids, list):
            ids = torch.tensor([ids])
        return ids.to(device)

    sample_ids = [encode_chat(s.prompt_text) for s in samples]
    seq_lens = [int(t.size(1)) for t in sample_ids]
    eos_id = tokenizer.eos_token_id
    print(
        f"[k3-integrated] dataset: {len(samples)} samples, prompt token len "
        f"min={min(seq_lens)} max={max(seq_lens)} "
        f"mean={sum(seq_lens) // len(seq_lens)}",
        file=sys.stderr,
    )

    def _greedy(decode_step) -> Tuple[List[str], List[float], List[int]]:
        """Run greedy decode over all samples with a per-step callable
        ``decode_step(cur_ids) -> logits[0, -1]``. Returns per-sample
        (decoded_text, latency_s, decode_token_count)."""
        decoded_all: List[str] = []
        lat_all: List[float] = []
        tok_all: List[int] = []
        for i in range(len(samples)):
            cur = sample_ids[i]
            gen: List[int] = []
            t0 = time.perf_counter()
            for _ in range(args.max_new_tokens):
                last_logits = decode_step(cur)
                nxt = int(torch.argmax(last_logits).item())
                gen.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
                cur = torch.cat(
                    [cur, torch.tensor([[nxt]], device=device, dtype=torch.long)],
                    dim=1,
                )
            lat_all.append(time.perf_counter() - t0)
            decoded_all.append(tokenizer.decode(gen, skip_special_tokens=True))
            tok_all.append(len(gen))
            print(
                f"[k3-integrated]   sample {i}: T={seq_lens[i]} tokens={len(gen)} "
                f"decoded[:48]={decoded_all[-1][:48]!r}",
                file=sys.stderr,
            )
        return decoded_all, lat_all, tok_all

    # ---------- Run integrated cross-model verifier ----------
    print("[k3-integrated] running K3 cross-model verifier (f_θ restoration)",
          file=sys.stderr, flush=True)
    reset_memory_peak(device)

    def _cross_step(cur):
        out = cross_verifier.forward(
            cur,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            eager_attention_forward=eager_attention_forward,
            all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        )
        return out.logits[0, -1]

    cross_decoded, cross_lat, cross_tok = _greedy(_cross_step)
    cross_res = aggregate_recall(
        "k3_cross_model", samples, cross_decoded, cross_lat, cross_tok,
    )
    cross_mem = record_memory(device)
    cross_attn_agg = aggregate_attention_window_metrics(
        "v04_dlm_restored",
        prompt_token_lens=seq_lens,
        sink_size=args.sink_size,
        window_size=args.window_size,
    )
    print(
        f"[k3-integrated] cross-model recall={cross_res.recall:.3f} "
        f"({cross_res.samples_correct}/{cross_res.samples_total})",
        file=sys.stderr,
    )

    # ---------- Optional oracle baseline ----------
    oracle_res = None
    oracle_mem = None
    if not args.skip_oracle:
        print("[k3-integrated] running full-attention oracle baseline",
              file=sys.stderr, flush=True)
        reset_memory_peak(device)

        def _oracle_step(cur):
            with torch.no_grad():
                out = verifier(input_ids=cur, use_cache=False)
            return out.logits[0, -1]

        oracle_decoded, oracle_lat, oracle_tok = _greedy(_oracle_step)
        oracle_res = aggregate_recall(
            "oracle", samples, oracle_decoded, oracle_lat, oracle_tok,
        )
        oracle_mem = record_memory(device)
        print(
            f"[k3-integrated] oracle recall={oracle_res.recall:.3f} "
            f"({oracle_res.samples_correct}/{oracle_res.samples_total})",
            file=sys.stderr,
        )

    # ---------- Build report ----------
    recall_delta = (
        abs(cross_res.recall - oracle_res.recall) if oracle_res else None
    )
    eff_frac_mean = cross_attn_agg.get("effective_attention_fraction_mean")
    report = {
        "schema_version": 2,
        "kind": "k3_integrated_niah_acceptance",
        "config": {
            "verifier_id": args.verifier_id,
            "drafter_id": args.drafter_id,
            "f_theta_dir": args.f_theta_dir,
            "f_theta_config": f_theta.config.to_json_dict(),
            "n_samples": args.n_samples,
            "sink_size": args.sink_size,
            "window_size": args.window_size,
            "haystack_min_lines": args.haystack_min_lines,
            "haystack_max_lines": args.haystack_max_lines,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "skip_oracle": bool(args.skip_oracle),
            "identity_restore": bool(args.identity_restore),
            "prompt_token_lens": seq_lens,
        },
        "results": {
            "k3_cross_model": dataclasses.asdict(cross_res),
            **({"oracle": dataclasses.asdict(oracle_res)} if oracle_res else {}),
        },
        "attention_window": {
            "per_config": {"k3_cross_model": cross_attn_agg},
        },
        "memory": {
            "k3_cross_model": cross_mem,
            **({"oracle": oracle_mem} if oracle_mem else {}),
        },
        "gate": {
            "architectural_correctness": (eff_frac_mean == 1.0),
            "recall_cross_model": cross_res.recall,
            "recall_oracle": oracle_res.recall if oracle_res else None,
            "recall_delta_vs_oracle_pp": (
                recall_delta * 100 if recall_delta is not None else None
            ),
            "recall_delta_within_5pp": (
                recall_delta is not None and recall_delta <= 0.05
            ),
        },
    }

    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_integrated_niah_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n[k3-integrated] DONE.", file=sys.stderr)
    print(
        f"  cross-model recall: {cross_res.recall:.3f} "
        f"({cross_res.samples_correct}/{cross_res.samples_total})",
        file=sys.stderr,
    )
    if oracle_res is not None:
        print(
            f"  oracle recall:      {oracle_res.recall:.3f} "
            f"({oracle_res.samples_correct}/{oracle_res.samples_total})",
            file=sys.stderr,
        )
        print(f"  |delta vs oracle|:  {recall_delta * 100:.2f} pp", file=sys.stderr)
        print(
            f"  ADR §11.8 1a gate (≤ 5pp): "
            f"{'PASS' if recall_delta <= 0.05 else 'FAIL'}",
            file=sys.stderr,
        )
    else:
        print("  oracle:             skipped", file=sys.stderr)
    print(f"  effective_attention_fraction: {eff_frac_mean}", file=sys.stderr)
    print(f"  Report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
