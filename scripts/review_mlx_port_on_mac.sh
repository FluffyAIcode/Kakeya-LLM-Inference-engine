#!/usr/bin/env bash
# Mac mini validation for the #107 MLX port (PR #109).
#
#   Step 1 (--incremental)      : restored decode via native cache + generate_step
#                                 → kills the O(T^2) re-forward collapse.
#   Step 2 (--fused-specdecode) : fused DFlash spec-decode (A+B+C).
#
# Each run also times the ORACLE = native mlx_lm AR (same model, no restoration),
# so the JSON carries `throughput.cross_model_speedup_vs_oracle_ar` and
# `gate.recall_delta_within_5pp` for a direct AR comparison. The speed gate is
# e2e over prefill+decode for both cross and oracle paths.
#
# Gates:
#   Step 1: speedup_vs_oracle ≈ 1.0 (no longer collapsed) AND recall == oracle.
#   Step 2: speedup_vs_oracle  > 1.0 (fused beats AR)     AND recall == oracle.
#
# Usage (from repo root, on the Mac mini):
#   bash scripts/review_mlx_port_on_mac.sh
# Override any knob via env, e.g.:
#   N_SAMPLES=8 MAX_NEW_TOKENS=64 BLOCK_SIZE=6 bash scripts/review_mlx_port_on_mac.sh
set -euo pipefail

VERIFIER_PATH="${VERIFIER_PATH:-models/gemma-4-26B-A4B-it-mlx-4bit}"
DRAFTER_ID="${DRAFTER_ID:-z-lab/gemma-4-26B-A4B-it-DFlash}"
F_THETA_DIR="${F_THETA_DIR:-results/research/f_theta_v5_s5_sliding}"
N_SAMPLES="${N_SAMPLES:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-512}"
DECODE_WARMUP_TOKENS="${DECODE_WARMUP_TOKENS:-1}"
SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
HAYSTACK_MIN="${HAYSTACK_MIN:-238}"
HAYSTACK_MAX="${HAYSTACK_MAX:-322}"
OUT_DIR="${OUT_DIR:-results/research}"
STAMP="$(date +%Y%m%d_%H%M%S)"

export PYTHONPATH="${PYTHONPATH:-.:sdks/python}"
# Let MLX use the unified-memory wired limit if the box is tight (optional).
export MLX_METAL_MEMORY_LIMIT_RATIO="${MLX_METAL_MEMORY_LIMIT_RATIO:-0.0}"

INCR_JSON="${OUT_DIR}/k3_mlx_incremental_${STAMP}.json"
FUSED_JSON="${OUT_DIR}/k3_mlx_fused_${STAMP}.json"

common_args=(
  --verifier-path "${VERIFIER_PATH}"
  --drafter-id "${DRAFTER_ID}"
  --f-theta-dir "${F_THETA_DIR}"
  --s5-exact-full-attn
  --n-samples "${N_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --sink-size "${SINK_SIZE}"
  --window-size "${WINDOW_SIZE}"
  --prefill-chunk-size "${PREFILL_CHUNK_SIZE}"
  --decode-warmup-tokens "${DECODE_WARMUP_TOKENS}"
  --haystack-min-lines "${HAYSTACK_MIN}"
  --haystack-max-lines "${HAYSTACK_MAX}"
)

echo "=========================================================="
echo "[mlx-port] Step 1: incremental restored decode (native cache + generate_step)"
echo "[mlx-port]   verifier=${VERIFIER_PATH}  drafter=${DRAFTER_ID}"
echo "[mlx-port]   f_theta=${F_THETA_DIR}  n=${N_SAMPLES}  gen=${MAX_NEW_TOKENS}"
echo "=========================================================="
python scripts/research/k3_integrated_niah_eval_mac.py \
  "${common_args[@]}" --incremental --output "${INCR_JSON}"

echo "=========================================================="
echo "[mlx-port] Step 2: fused DFlash spec-decode (A+B+C, block_size=${BLOCK_SIZE})"
echo "=========================================================="
python scripts/research/k3_integrated_niah_eval_mac.py \
  "${common_args[@]}" --fused-specdecode --block-size "${BLOCK_SIZE}" \
  --output "${FUSED_JSON}"

echo "=========================================================="
echo "[mlx-port] SUMMARY"
echo "=========================================================="
python - "${INCR_JSON}" "${FUSED_JSON}" <<'PY'
import json, sys
def show(tag, path, want):
    d = json.load(open(path))
    g, t = d["gate"], d["throughput"]
    cm = t["k3_cross_model"]; ar = t.get("oracle_native_ar") or {}
    spd = t.get("cross_model_speedup_vs_oracle_ar")
    rc, ro = g["recall_cross_model"], g.get("recall_oracle")
    mem = d["memory"]
    print(f"\n[{tag}]  ({d['config']['eval_mode']})")
    print(f"  recall: cross={rc}  oracle={ro}  within_5pp={g['recall_delta_within_5pp']}")
    print(f"  scope : cross={cm.get('timing_scope')}  oracle={ar.get('timing_scope')}")
    print(f"  tok/s : cross={cm.get('tokens_per_second')}  "
          f"oracle_AR={ar.get('tokens_per_second')}  speedup_vs_AR={spd}")
    print(f"  KV    : S5={mem['s5']['total_resident_mb']}MB  "
          f"naive={mem['naive_full_kv']['total_resident_mb']}MB  "
          f"savings={mem['savings_vs_naive_pct']}%")
    ok_recall = bool(g["recall_delta_within_5pp"])
    ok_speed = (spd is not None and spd >= want)
    print(f"  GATE  : recall {'PASS' if ok_recall else 'FAIL'}  | "
          f"speed {'PASS' if ok_speed else 'FAIL'} (need >= {want}x AR)")
    return ok_recall and ok_speed
s1 = show("Step 1 incremental", sys.argv[1], 0.85)   # ~= AR (collapse fixed)
s2 = show("Step 2 fused",       sys.argv[2], 1.00)   # > AR
print("\n[mlx-port] OVERALL:",
      "PASS" if (s1 and s2) else "see gates above")
print(f"[mlx-port] JSON: {sys.argv[1]}\n[mlx-port] JSON: {sys.argv[2]}")
PY
