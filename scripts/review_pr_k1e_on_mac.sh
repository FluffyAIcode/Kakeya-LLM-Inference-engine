#!/usr/bin/env bash
# Mac M4 review aid for PR-K1.E — empirical NIAH validation of v0.4
# K/V Restoration against the v0.3 sink+window baseline and the
# full-attention oracle, on real google/gemma-3-1b-it.
#
# This is the PR that makes ADR 0008 §11.8 gate (a) falsifiable on
# hardware. Three forwards per sample × 20 samples = 60 generation
# loops. Time budget on Mac M4 24 GB:
#
#   * Oracle           ~30-60 s per sample at ~1k tokens, ~5-10 min total
#   * v0.3 sink+window ~5-15 s per sample (cheap forward), ~2-5 min
#   * v0.4 DLM restore ~60-120 s per sample (proposer + verifier per step)
#                       ~20-40 min total
#
# Total: ~30-60 min for the default 20-sample 1-2k-token run. Larger
# context lengths or more samples scale linearly — for 100k context
# the oracle alone needs ~10 GB just for KV cache; budget 1-3 hours
# total at that scale.
#
# Acceptance signals:
#
#   * v0.4 recall >> v0.3 recall  (target ≥ +50pp; ADR 0008 §11.5
#     §"Five properties" item 2 — intelligence approximates full
#     attention)
#   * v0.4 recall close to oracle (target within 5pp; ADR 0008 §11.8
#     gate (a))
#   * v0.3 recall ≈ 0.17 (regression confirmed; matches the
#     2026-06-06 A/B benchmark in
#     results/platform-tests/sink_window_quality_ab_1780714635.json)
#
# Three orthogonal metrics are reported per config (in JSON +
# stderr summary):
#
#   * recall              — behavioural intelligence (does the
#                           model actually answer correctly)
#   * peak / current mem  — sustained memory cost (K1.G)
#   * attn_window         — structural attention coverage as a
#                           fraction of preceding context (K1.H);
#                           v0.3 caps at sink+window regardless of
#                           T, v0.4's dLM K/V Restoration restores
#                           the full causal range
#
# Time-saving knobs (env vars):
#
#   N_SAMPLES         (default 20)   how many NIAH samples per config
#   HAYSTACK_MIN      (default 60)   min padding-line count
#   HAYSTACK_MAX      (default 80)   max padding-line count
#   SINK              (default 4)    sink size
#   WINDOW            (default 64)   window size
#   ATTN_IMPL         (default eager) 'eager' (matches the 2026-06-08
#                                    Mac M4 baseline recorded in ADR
#                                    0008 §11.11) vs 'sdpa' (memory-
#                                    efficient; needed for >= 16k context
#                                    even on Mac, but introduces small
#                                    bf16 reduction-order numerical
#                                    differences vs eager — slightly
#                                    different recall numbers possible).
#   SKIP_V03=1                       skip the v0.3 baseline (saves ~5 min)
#   SKIP_V04=1                       skip v0.4 (smoke-only the oracle path)
#
# Usage:
#     export HF_TOKEN=hf_xxx        # if model is gated for you
#     bash scripts/review_pr_k1e_on_mac.sh
#
#     # Larger context (16k tokens):
#     HAYSTACK_MIN=900 HAYSTACK_MAX=1100 bash scripts/review_pr_k1e_on_mac.sh
#
#     # Quick smoke (5 samples):
#     N_SAMPLES=5 bash scripts/review_pr_k1e_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N_SAMPLES="${N_SAMPLES:-20}"
HAYSTACK_MIN="${HAYSTACK_MIN:-60}"
HAYSTACK_MAX="${HAYSTACK_MAX:-80}"
SINK="${SINK:-4}"
WINDOW="${WINDOW:-64}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
SKIP_V03="${SKIP_V03:-0}"
SKIP_V04="${SKIP_V04:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k1e_niah_${stamp}.json"
log="${log_dir}/k1e_niah_${stamp}.log"

echo "==> PR-K1.E NIAH validation (Mac M4 — google/gemma-3-1b-it)"
echo "    Samples:           $N_SAMPLES"
echo "    Haystack lines:    [$HAYSTACK_MIN, $HAYSTACK_MAX]"
echo "    Sink x window:     ${SINK}x${WINDOW}"
echo "    Configs:           oracle"
[[ "$SKIP_V03" == "1" ]] && echo "                       (v0.3 SKIPPED)" || echo "                       v0.3 sink+window"
[[ "$SKIP_V04" == "1" ]] && echo "                       (v0.4 SKIPPED)" || echo "                       v0.4 DLMRestoredVerifier"
echo "    Report:            $report"
echo "    Log:               $log"
echo

flags=(
  --model google/gemma-3-1b-it
  --device auto
  --attn-impl "$ATTN_IMPL"
  --n-samples "$N_SAMPLES"
  --haystack-min-lines "$HAYSTACK_MIN"
  --haystack-max-lines "$HAYSTACK_MAX"
  --sink-size "$SINK"
  --window-size "$WINDOW"
  --output "$report"
)
[[ "$SKIP_V03" == "1" ]] && flags+=(--skip-v03)
[[ "$SKIP_V04" == "1" ]] && flags+=(--skip-v04)

PYTHONPATH=.:sdks/python python3 scripts/research/k1e_niah_validation.py "${flags[@]}" 2>&1 | tee "$log"
exit_code=$?

echo
echo "==> Done. Report: $report"
echo "    exit code: $exit_code"
echo
echo "Commit:"
echo "    git add $report $log"
echo "    git commit -m 'Mac M4 K1.E NIAH validation evidence'"
echo "    git push"

exit $exit_code
