#!/usr/bin/env bash
# Mac M4 review aid for PR-K1.D — DLMRestoredVerifier integration smoke
# against real google/gemma-3-1b-it.
#
# This is a smoke test, not the empirical NIAH validation gate. The
# question this script answers is: "does the v0.4 K/V Restoration
# wrapper actually run end-to-end on real Gemma 3-1B-it without
# crashing or producing NaN/Inf logits?". The empirical "does it
# rescue mid-context recall?" question is K1.E (separate PR with a
# proper NIAH harness measuring v0.3 vs v0.4 vs full-attention
# oracle).
#
# What this script does
# ---------------------
#
# 1. Loads google/gemma-3-1b-it via HF transformers (downloads on
#    first run; cached afterwards).
# 2. Builds a synthetic 256-token input.
# 3. Runs three forwards on the same input:
#       (a) standard model.forward (full attention oracle)
#       (b) v0.4 DLMRestoredVerifier.forward with sink=4 window=64
#       (c) v0.4 DLMRestoredVerifier.forward with sink=10000 window=10000
#           (effectively no eviction — should match (a) bit-exactly
#           up to numerical noise)
# 4. Records summary statistics (last-token logit norm, argmax token,
#    KL divergence between (a) and (c)) into a JSON artifact.
# 5. Smoke gate: NaN/Inf-free in all three, and (c) ≈ (a) to numerical
#    precision (KL < 1e-3).
#
# This script does NOT make recall claims. Recall validation is K1.E.
#
# Time budget on Mac M4 24 GB with Gemma 3-1B-it: ~3-5 minutes.
#
# Usage:
#     export HF_TOKEN=hf_xxx   # only needed if model is gated for you
#     bash scripts/review_pr_k1d_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k1d_smoke_${stamp}.json"
log="${log_dir}/k1d_smoke_${stamp}.log"

echo "==> PR-K1.D smoke test (Mac M4)"
echo "    Model:        google/gemma-3-1b-it"
echo "    Device:       auto (mps)"
echo "    Input:        synthetic 256 tokens"
echo "    Configs:      (a) full attention | (b) sink=4 window=64 | (c) sink=10000 window=10000"
echo "    Report:       $report"
echo "    Log:          $log"
echo

PYTHONPATH=.:sdks/python python3 - "$report" 2>&1 | tee "$log" <<'PY'
"""K1.D Mac M4 smoke test — DLMRestoredVerifier on real Gemma 3-1B-it."""
import json
import math
import sys
import time
import traceback

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)

from inference_engine.v04 import DLMRestoredVerifier


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) on the last token's logits, in nats."""
    p_logp = torch.log_softmax(p_logits.float(), dim=-1)
    q_logp = torch.log_softmax(q_logits.float(), dim=-1)
    p = p_logp.exp()
    return float((p * (p_logp - q_logp)).sum().item())


def summarise_logits(name: str, logits: torch.Tensor) -> dict:
    """Summary statistics for a [1, T, vocab] logits tensor — focused on
    the last position (the next-token prediction)."""
    last = logits[0, -1, :].float()
    return {
        "name": name,
        "shape": list(logits.shape),
        "last_token_norm": float(last.norm().item()),
        "last_token_argmax": int(last.argmax().item()),
        "last_token_max": float(last.max().item()),
        "last_token_min": float(last.min().item()),
        "any_nan": bool(torch.isnan(logits).any().item()),
        "any_inf": bool(torch.isinf(logits).any().item()),
    }


def main():
    output_path = sys.argv[1]
    device = pick_device()
    print(f"[k1d] device={device}", file=sys.stderr)

    model_id = "google/gemma-3-1b-it"
    print(f"[k1d] loading {model_id}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Synthetic 256-token input. We don't need real semantic content
    # for the smoke gate; we just need the forward to run.
    seq_len = 256
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), device=device,
    )
    print(
        f"[k1d] input shape: {tuple(input_ids.shape)} dtype={input_ids.dtype}",
        file=sys.stderr,
    )

    results = {
        "schema_version": 1,
        "kind": "k1d_dlm_restored_verifier_smoke",
        "model": model_id,
        "device": str(device),
        "dtype": str(dtype),
        "seq_len": seq_len,
        "configs": [],
    }

    # ----------------------------------------------------------------
    # (a) Full attention oracle
    # ----------------------------------------------------------------
    print("[k1d] (a) running standard model.forward (oracle)", file=sys.stderr)
    t0 = time.perf_counter()
    with torch.no_grad():
        oracle_out = model(input_ids=input_ids, use_cache=False)
    oracle_logits = oracle_out.logits.detach().cpu()
    elapsed_a = time.perf_counter() - t0
    summary_a = summarise_logits("oracle_full_attention", oracle_logits)
    summary_a["elapsed_s"] = elapsed_a
    results["configs"].append(summary_a)
    print(f"[k1d]     elapsed {elapsed_a:.2f}s, last argmax={summary_a['last_token_argmax']}", file=sys.stderr)

    # ----------------------------------------------------------------
    # (b) v0.4 with aggressive sink+window (real eviction happens)
    # ----------------------------------------------------------------
    print("[k1d] (b) running DLMRestoredVerifier sink=4 window=64", file=sys.stderr)
    verifier_b = DLMRestoredVerifier(model, sink_size=4, window_size=64)
    t0 = time.perf_counter()
    logits_b = verifier_b.forward(
        input_ids,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    ).detach().cpu()
    elapsed_b = time.perf_counter() - t0
    summary_b = summarise_logits("v04_sink_4_window_64", logits_b)
    summary_b["elapsed_s"] = elapsed_b
    summary_b["kl_vs_oracle"] = kl_divergence(oracle_logits, logits_b)
    summary_b["argmax_matches_oracle"] = (
        summary_b["last_token_argmax"] == summary_a["last_token_argmax"]
    )
    results["configs"].append(summary_b)
    print(
        f"[k1d]     elapsed {elapsed_b:.2f}s, last argmax="
        f"{summary_b['last_token_argmax']} (oracle "
        f"argmax={summary_a['last_token_argmax']}), "
        f"KL vs oracle={summary_b['kl_vs_oracle']:.4f} nats",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # (c) v0.4 with no eviction (sink+window covers all positions)
    #
    # When sink+window >= seq_len, no evictions happen; the verifier
    # forward should produce logits bit-exactly matching the oracle
    # up to numerical noise from the patched-forward code path. This
    # is the K1.D smoke gate.
    # ----------------------------------------------------------------
    print("[k1d] (c) running DLMRestoredVerifier sink=10000 window=10000 (no eviction)", file=sys.stderr)
    verifier_c = DLMRestoredVerifier(model, sink_size=10000, window_size=10000)
    t0 = time.perf_counter()
    logits_c = verifier_c.forward(
        input_ids,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
    ).detach().cpu()
    elapsed_c = time.perf_counter() - t0
    summary_c = summarise_logits("v04_no_eviction", logits_c)
    summary_c["elapsed_s"] = elapsed_c
    summary_c["kl_vs_oracle"] = kl_divergence(oracle_logits, logits_c)
    summary_c["argmax_matches_oracle"] = (
        summary_c["last_token_argmax"] == summary_a["last_token_argmax"]
    )
    results["configs"].append(summary_c)
    print(
        f"[k1d]     elapsed {elapsed_c:.2f}s, last argmax="
        f"{summary_c['last_token_argmax']}, "
        f"KL vs oracle={summary_c['kl_vs_oracle']:.4e} nats "
        f"(should be ~0)",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Smoke gate
    # ----------------------------------------------------------------
    smoke_pass = True
    smoke_failures = []
    for cfg in results["configs"]:
        if cfg["any_nan"]:
            smoke_pass = False
            smoke_failures.append(f"{cfg['name']}: NaN logits")
        if cfg["any_inf"]:
            smoke_pass = False
            smoke_failures.append(f"{cfg['name']}: Inf logits")

    # (c) should ≈ (a) bit-exactly up to numerical noise. Acceptance:
    # KL < 1e-3 nats (extremely tight; the patched forward is
    # mathematically identical to the original when no merge runs).
    no_evict_kl = summary_c["kl_vs_oracle"]
    no_evict_kl_threshold = 1e-3
    if no_evict_kl > no_evict_kl_threshold:
        smoke_pass = False
        smoke_failures.append(
            f"no-eviction KL vs oracle = {no_evict_kl:.4e} > "
            f"{no_evict_kl_threshold:.4e}"
        )

    results["smoke_gate"] = {
        "pass": smoke_pass,
        "failures": smoke_failures,
        "no_eviction_kl_threshold": no_evict_kl_threshold,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[k1d] report -> {output_path}", file=sys.stderr)

    if smoke_pass:
        print("[k1d] SMOKE GATE: PASS", file=sys.stderr)
        return 0
    else:
        print("[k1d] SMOKE GATE: FAIL", file=sys.stderr)
        for f in smoke_failures:
            print(f"  - {f}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        rc = 2
    sys.exit(rc)
PY
exit_code=$?

echo
echo "==> Done. Report: $report"
echo "    exit code: $exit_code (0 = smoke gate PASS)"
echo
echo "Commit:"
echo "    git add $report $log"
echo "    git commit -m 'Mac M4 K1.D smoke evidence'"
echo "    git push"

exit $exit_code
