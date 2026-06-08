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
# Two run modes:
#
#   * Single-context (default): one (HAYSTACK_MIN, HAYSTACK_MAX) range,
#     producing one JSON report. Good for fast iteration; matches the
#     2026-06-08 Mac M4 baseline (cbdf13d) at 1.4k context.
#
#   * Multi-context scan (MULTI_CONTEXT=1): same harness across an
#     ordered ladder of context lengths in one invocation. Mirrors the
#     vast.ai reviewer (scripts/review_pr_k1e_on_vast.sh) so Mac M4
#     can fill the §11.12 evidence ladder rungs that vast.ai's CUDA
#     long-context run cannot reach in time, and so the K1.H
#     attention-window metric is exercised at multiple T values on
#     real hardware. Default Mac ladder is '70 280' (≈1.4k + 5.6k
#     tokens) — the 1.4k point reproduces the cbdf13d baseline under
#     the K1.H schema, and the 5.6k point fills the gap between Mac
#     M4 (cbdf13d) and vast.ai's pending 64k/100k results.
#
# Mac M4 24 GB context budget (with SDPA — required for >= 4k):
#
#   *  ~1.4k  (eager OK)        oracle ~30-60s/sample, full ladder ~30-50min
#   *  ~5.6k  (sdpa)            oracle ~3-5min/sample, full ladder ~3-5h
#   *  ~16k   (sdpa, tight)     oracle ~10-15min/sample, full ladder ~8-12h
#   *  >= 32k                   typically OOMs even under SDPA; use vast.ai
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
#   HAYSTACK_MIN      (default 60)   single-context: min padding-line count
#   HAYSTACK_MAX      (default 80)   single-context: max padding-line count
#   SINK              (default 4)    sink size
#   WINDOW            (default 64)   window size
#   ATTN_IMPL         (default auto) auto-selects 'eager' for single-
#                                    context (matches the 2026-06-08
#                                    Mac M4 baseline cbdf13d) and 'sdpa'
#                                    for MULTI_CONTEXT (required for
#                                    >= 4k context on 24 GB). Override
#                                    explicitly to force one or the
#                                    other; SDPA introduces small bf16
#                                    reduction-order numerical drift
#                                    vs eager — recall numbers may
#                                    differ by 1-2 samples per 20.
#   SKIP_V03=1                       skip the v0.3 baseline (saves ~5 min)
#   SKIP_V04=1                       skip v0.4 (smoke-only the oracle path)
#   SKIP_ORACLE=1                    skip the oracle (not recommended)
#   MULTI_CONTEXT=1                  enable the multi-context ladder
#   CONTEXT_LADDER='70 280'          (only used when MULTI_CONTEXT=1)
#                                    space-separated padding-line counts;
#                                    line ≈ 20 tokens with chat template;
#                                    each entry yields a haystack range
#                                    of [n × 0.85, n × 1.15] for sample
#                                    variability. Default '70 280'
#                                    targets ~1.4k + ~5.6k tokens — fits
#                                    Mac M4 24 GB comfortably with SDPA.
#                                    Add '800' (≈16k) only if 8-12h is
#                                    available.
#
# Usage:
#     export HF_TOKEN=hf_xxx        # if model is gated for you
#     bash scripts/review_pr_k1e_on_mac.sh
#
#     # Larger single-context (≈16k tokens):
#     HAYSTACK_MIN=720 HAYSTACK_MAX=880 ATTN_IMPL=sdpa \
#         bash scripts/review_pr_k1e_on_mac.sh
#
#     # Multi-context ladder (1.4k + 5.6k, ~5-6 h, fills §11.12 mid-rung):
#     MULTI_CONTEXT=1 bash scripts/review_pr_k1e_on_mac.sh
#
#     # Aggressive multi-context (1.4k + 5.6k + 16k, ~10-14 h):
#     MULTI_CONTEXT=1 CONTEXT_LADDER='70 280 800' \
#         bash scripts/review_pr_k1e_on_mac.sh
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
SKIP_V03="${SKIP_V03:-0}"
SKIP_V04="${SKIP_V04:-0}"
SKIP_ORACLE="${SKIP_ORACLE:-0}"
MULTI_CONTEXT="${MULTI_CONTEXT:-0}"
# Default Mac ladder: ~1.4k + ~5.6k tokens (line ≈ 20 tokens with chat
# template). The 1.4k point reproduces the 2026-06-08 Mac M4 baseline
# (cbdf13d) under the new K1.H attention-window schema. The 5.6k point
# fills the §11.12 evidence rung between Mac M4 (existing) and vast.ai
# (pending 64k/100k). Add '800' (≈16k) only if 8-12 h is available.
CONTEXT_LADDER="${CONTEXT_LADDER:-70 280}"
# ATTN_IMPL: 'auto' picks 'eager' for single-context (cbdf13d parity)
# and 'sdpa' for MULTI_CONTEXT (>= 4k context would OOM under eager
# even on Mac M4 24 GB — same pathology K1.F fixed for vast). Override
# to force a specific implementation.
ATTN_IMPL_RAW="${ATTN_IMPL:-auto}"
if [[ "$ATTN_IMPL_RAW" == "auto" ]]; then
    if [[ "$MULTI_CONTEXT" == "1" ]]; then
        ATTN_IMPL="sdpa"
    else
        ATTN_IMPL="eager"
    fi
else
    ATTN_IMPL="$ATTN_IMPL_RAW"
fi

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

flags_common=(
  --model google/gemma-3-1b-it
  --device auto
  --attn-impl "$ATTN_IMPL"
  --n-samples "$N_SAMPLES"
  --sink-size "$SINK"
  --window-size "$WINDOW"
)
[[ "$SKIP_V03"    == "1" ]] && flags_common+=(--skip-v03)
[[ "$SKIP_V04"    == "1" ]] && flags_common+=(--skip-v04)
[[ "$SKIP_ORACLE" == "1" ]] && flags_common+=(--skip-oracle)

run_one() {
    local label="$1"; local lo="$2"; local hi="$3"
    local report="${out_dir}/k1e_niah_mac_${label}_${stamp}.json"
    local log="${log_dir}/k1e_niah_mac_${label}_${stamp}.log"
    echo
    echo "==> Run $label: haystack lines [$lo, $hi]  attn=$ATTN_IMPL"
    echo "    Report: $report"
    echo "    Log:    $log"
    PYTHONPATH=.:sdks/python python3 scripts/research/k1e_niah_validation.py \
        "${flags_common[@]}" \
        --haystack-min-lines "$lo" \
        --haystack-max-lines "$hi" \
        --output "$report" 2>&1 | tee "$log"
    echo "    -> finished $label"
}

if [[ "$MULTI_CONTEXT" == "1" ]]; then
    echo "==> PR-K1.E NIAH validation — Mac M4 multi-context scan"
    echo "    Model:           google/gemma-3-1b-it"
    echo "    Samples each:    $N_SAMPLES"
    echo "    Sink x window:   ${SINK}x${WINDOW}"
    echo "    Attn impl:       $ATTN_IMPL  (auto: sdpa for multi-ctx)"
    echo "    Context ladder:  $CONTEXT_LADDER  (padding lines)"
    echo "    Configs:         oracle + v0.3 + v0.4 (modulo skip flags)"
    echo "    Time budget:     ~5-6 h for default '70 280' ladder"
    echo

    for n in $CONTEXT_LADDER; do
        # ±15 % range around target line count — mirrors vast variant
        lo=$(( (n * 85 + 50) / 100 ))
        hi=$(( (n * 115 + 50) / 100 ))
        if [[ $lo -lt 10 ]]; then lo=10; fi
        if [[ $hi -lt $((lo + 1)) ]]; then hi=$((lo + 1)); fi
        run_one "ctx${n}" "$lo" "$hi"
    done

    echo
    echo "==> Multi-context scan complete. Reports under:"
    echo "    $out_dir/k1e_niah_mac_ctx*_${stamp}.json"
    echo "    $log_dir/k1e_niah_mac_ctx*_${stamp}.log"
    echo
    echo "Commit:"
    echo "    git add $out_dir/k1e_niah_mac_ctx*_${stamp}.json $log_dir/k1e_niah_mac_ctx*_${stamp}.log"
    echo "    git commit -m 'Mac M4 K1.E multi-context NIAH evidence (recall + memory + attn_window)'"
    echo "    git push"
else
    echo "==> PR-K1.E NIAH validation (Mac M4 — google/gemma-3-1b-it)"
    echo "    Samples:           $N_SAMPLES"
    echo "    Haystack lines:    [$HAYSTACK_MIN, $HAYSTACK_MAX]"
    echo "    Sink x window:     ${SINK}x${WINDOW}"
    echo "    Attn impl:         $ATTN_IMPL  (auto: eager for single-ctx)"
    echo "    Configs:           oracle"
    [[ "$SKIP_V03"    == "1" ]] && echo "                       (v0.3 SKIPPED)"    || echo "                       v0.3 sink+window"
    [[ "$SKIP_V04"    == "1" ]] && echo "                       (v0.4 SKIPPED)"    || echo "                       v0.4 DLMRestoredVerifier"
    [[ "$SKIP_ORACLE" == "1" ]] && echo "                       (oracle SKIPPED)"
    echo

    run_one "single" "$HAYSTACK_MIN" "$HAYSTACK_MAX"
    echo
    echo "Commit:"
    echo "    git add $out_dir/k1e_niah_mac_single_${stamp}.json $log_dir/k1e_niah_mac_single_${stamp}.log"
    echo "    git commit -m 'Mac M4 K1.E NIAH validation evidence'"
    echo "    git push"
fi
