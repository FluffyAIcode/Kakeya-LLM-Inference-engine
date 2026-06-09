#!/usr/bin/env bash
# vast.ai (CUDA) reviewer aid for PR-K1.E — GPU acceleration of the
# NIAH validation harness.
#
# Same K1.E harness as the Mac M4 reviewer
# (scripts/review_pr_k1e_on_mac.sh), but routed through the existing
# vast provisioning machinery (scripts/research/run_on_vast.sh) and
# tuned for CUDA-class hardware. Two modes:
#
#   * Single-context (default): evaluate one context length per run.
#     Useful for fast iteration during development.
#
#   * Multi-context scan (MULTI_CONTEXT=1): evaluate the same model
#     and configurations across several context lengths in one
#     invocation, producing a recall-vs-context-length curve. This
#     is the form that empirically validates the ADR 0008 §11.8
#     gate (a) target ("≥ 95 % at 100 k") AND demonstrates how v0.4
#     scales relative to v0.3 sink+window AND the full-attention
#     oracle.
#
# Time budget on a vast.ai NVIDIA H100 (80 GB):
#
#   * 2 k context, 30 samples, all 3 configs: ~5-8 min
#   * 4 k context, 30 samples, all 3 configs: ~10-15 min
#   * 16 k context, 30 samples, all 3 configs: ~30-45 min
#   * 64 k context, 20 samples: ~60-90 min
#   * 100 k context, 20 samples: ~90-150 min
#
# Multi-context scan default (1 k → 4 k → 16 k) runs in ~45-60 min.
# Default targets H100; on A100 80 GB add ~50-100 % for compute-bound
# v0.4 forwards. Smaller GPUs (A10G 24 GB) cap out around 16 k tokens
# for the oracle config but can still run v0.4 at any size (sustained
# memory is constant in context length by design).
#
# Acceptance signals — same as the Mac reviewer:
#
#   * v0.3 recall ≈ 0.17 at 1 k+ context (matches the
#     2026-06-06 A/B benchmark; sanity that the regression
#     reproduces)
#   * v0.4 recall close to oracle (within 5 pp; ADR 0008 §11.8
#     gate (a) at the run's context length)
#   * v0.4 ≫ v0.3 (target ≥ +50 pp; ADR 0008 §11.5 §"Five
#     properties" item 2 — intelligence approximates full attention)
#
# Usage:
#
#     # Setup: vast instance must be running, repo synced, HF_TOKEN exported
#     HF_TOKEN=hf_xxx bash scripts/review_pr_k1e_on_vast.sh
#
#     # Larger single-context run:
#     HAYSTACK_MIN=900 HAYSTACK_MAX=1100 N_SAMPLES=30 \
#         bash scripts/review_pr_k1e_on_vast.sh
#
#     # Multi-context scan with default ladder (~30, ~120, ~500 lines
#     #  ≈ 1-2k, 4k, 16k tokens):
#     MULTI_CONTEXT=1 bash scripts/review_pr_k1e_on_vast.sh
#
#     # Custom multi-context scan (lines per context — line ≈ 14 tokens):
#     MULTI_CONTEXT=1 \
#     CONTEXT_LADDER='80 320 1280 5000' \
#         bash scripts/review_pr_k1e_on_vast.sh
#
# Env knobs:
#
#   N_SAMPLES         (default 30)   samples per (config, context length)
#   HAYSTACK_MIN      (default 60)   single-context: min padding-line count
#   HAYSTACK_MAX      (default 80)   single-context: max padding-line count
#   SINK              (default 4)
#   WINDOW            (default 64)
#   MAX_NEW_TOKENS    (default 24)
#   SEED              (default 42)
#   SKIP_V03=1                       skip the v0.3 baseline
#   SKIP_V04=1                       skip v0.4 (oracle-only smoke)
#   SKIP_ORACLE=1                    skip the oracle (not recommended)
#   MULTI_CONTEXT=1                  enable multi-context scan
#   CONTEXT_LADDER='40 80 320 1280'  (only used when MULTI_CONTEXT=1)
#                                    space-separated padding-line counts;
#                                    each entry yields a haystack range of
#                                    [n × 0.85, n × 1.15] for variability.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N_SAMPLES="${N_SAMPLES:-30}"
HAYSTACK_MIN="${HAYSTACK_MIN:-60}"
HAYSTACK_MAX="${HAYSTACK_MAX:-80}"
SINK="${SINK:-4}"
WINDOW="${WINDOW:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-42}"
SKIP_V03="${SKIP_V03:-0}"
SKIP_V04="${SKIP_V04:-0}"
SKIP_ORACLE="${SKIP_ORACLE:-0}"
MULTI_CONTEXT="${MULTI_CONTEXT:-0}"
# Default ladder: ~1k, ~4k, ~16k tokens (line ≈ 14 tokens)
CONTEXT_LADDER="${CONTEXT_LADDER:-70 280 1100}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

flags_common=(
    --model google/gemma-3-1b-it
    --device cuda
    --n-samples "$N_SAMPLES"
    --sink-size "$SINK"
    --window-size "$WINDOW"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
)
[[ "$SKIP_V03"    == "1" ]] && flags_common+=(--skip-v03)
[[ "$SKIP_V04"    == "1" ]] && flags_common+=(--skip-v04)
[[ "$SKIP_ORACLE" == "1" ]] && flags_common+=(--skip-oracle)

# Tell the generic vast runner which Python script to invoke.
export KAKEYA_VAST_SCRIPT="scripts/research/k1e_niah_validation.py"

# Provision venv ONCE before any runs.
echo "==> provisioning venv (one-time)"
bash scripts/research/run_on_vast.sh --setup-only

run_one() {
    local label="$1"; local lo="$2"; local hi="$3"
    local report="${out_dir}/k1e_niah_vast_${label}_${stamp}.json"
    local log="${log_dir}/k1e_niah_vast_${label}_${stamp}.log"
    echo
    echo "==> Run $label: haystack lines [$lo, $hi]"
    echo "    Report: $report"
    echo "    Log:    $log"
    bash scripts/research/run_on_vast.sh \
        "${flags_common[@]}" \
        --haystack-min-lines "$lo" \
        --haystack-max-lines "$hi" \
        --output "$report" \
        2>&1 | tee "$log"
    echo "    -> finished $label"
}

if [[ "$MULTI_CONTEXT" == "1" ]]; then
    echo "==> PR-K1.E NIAH validation — vast.ai CUDA, multi-context scan"
    echo "    Model:        google/gemma-3-1b-it"
    echo "    Samples each: $N_SAMPLES"
    echo "    Sink x window: ${SINK} x ${WINDOW}"
    echo "    Context ladder (padding lines): $CONTEXT_LADDER"
    echo "    Configs:      oracle + v0.3 + v0.4 (modulo skip flags)"
    echo

    for n in $CONTEXT_LADDER; do
        # ±15 % range around target line count
        lo=$(( (n * 85 + 50) / 100 ))
        hi=$(( (n * 115 + 50) / 100 ))
        if [[ $lo -lt 10 ]]; then lo=10; fi
        if [[ $hi -lt $((lo + 1)) ]]; then hi=$((lo + 1)); fi
        run_one "ctx${n}" "$lo" "$hi"
    done

    echo
    echo "==> Multi-context scan complete. Reports under:"
    echo "    $out_dir/k1e_niah_vast_ctx*_${stamp}.json"
    echo "    $log_dir/k1e_niah_vast_ctx*_${stamp}.log"
else
    echo "==> PR-K1.E NIAH validation — vast.ai CUDA, single-context"
    echo "    Model:        google/gemma-3-1b-it"
    echo "    Samples:      $N_SAMPLES"
    echo "    Haystack:     [$HAYSTACK_MIN, $HAYSTACK_MAX] lines"
    echo "    Sink x window: ${SINK} x ${WINDOW}"
    echo "    Configs:      oracle + v0.3 + v0.4 (modulo skip flags)"
    echo

    run_one "single" "$HAYSTACK_MIN" "$HAYSTACK_MAX"
fi

echo
echo "Commit:"
echo "    git add $out_dir/k1e_niah_vast_*_${stamp}.json $log_dir/k1e_niah_vast_*_${stamp}.log"
echo "    git commit -m 'vast H100/A100 K1.E NIAH validation evidence'"
echo "    git push"
