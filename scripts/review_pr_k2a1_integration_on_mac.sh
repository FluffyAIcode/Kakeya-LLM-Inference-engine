#!/usr/bin/env bash
# Mac M4 reviewer aid for PR-K2.A.1 — KakeyaLattice integration
# A/B at the §11.12 ladder rungs Mac M4 24 GB can handle.
#
# Mirror of scripts/review_pr_k2a1_integration_on_vast.sh, but
# scoped to Mac M4 budget: default ladder '70 280' (~1.4k +
# ~5.6k tokens). Adding 800 (~16k) to the ladder is feasible but
# pushes total time to ~10-14h.
#
# Runs the K1.E NIAH harness twice at every context rung — once
# with KL OFF (K1 baseline; reproduces 4fb947f evidence under the
# K2.A.1 schema v5) and once with KL ON (KakeyaLattice D4 Q=38
# on PyTorch MPS). Acceptance evidence as in the vast variant:
#
#   gate (b) recall delta <= 1pp — Mac binding gate for K2.A.
#                                  Should hold at every rung; if
#                                  Mac MPS fidelity (~7e-4 K rel
#                                  MSE per K2.A.0 smoke) is too
#                                  loose for retrieval at any rung,
#                                  tighten Q (e.g. Q=76 → +1
#                                  bit/coord, halves quantisation
#                                  error) per ADR §11.11.9 escape
#                                  hatch. Do NOT fail K2.A on this
#                                  alone.
#   gate (a) round-trip identity — sanity from K2.A.0 smoke;
#                                  Mac MPS calibrated bound is
#                                  1.5e-3 (per scripts/research/
#                                  k2a_kl_mac_smoke.py). The K1.E
#                                  harness exercises the codec on
#                                  real K/V values; if recall is
#                                  preserved, the codec is
#                                  functional.
#   gate (c) throughput >= 1.3x  — K2.A.1 stateless adds compress+
#                                  decompress overhead per step
#                                  WITHOUT cross-step caching
#                                  savings. Throughput on Mac is
#                                  expected to be SAME OR SLOWER
#                                  with KL on — gate (c) is closed
#                                  by K2.A.2 stateful caching, not
#                                  this PR.
#
# Time budget on Mac M4 24 GB (with SDPA — required for >= 4k
# context). Numbers reflect doubled work (KL on + KL off, each
# arm = a full §11.12 rung):
#
#   '70 280'        (1.4k + 5.6k):   ~7-9 h
#   '70 280 800'    (+ 16k):         ~16-22 h  (NOT recommended)
#
# Defaults: D4 Q=38 lattice, MPS device, attn=sdpa.
#
# Env knobs (defaults):
#
#   N_SAMPLES         (20)         samples per (config, context, kl-state)
#   SINK              (4)
#   WINDOW            (64)
#   MAX_NEW_TOKENS    (24)
#   SEED              (42)
#   KL_LATTICE        (D4)         D4 (v1.4) or E8 (v1.5)
#   KL_Q_RANGE        (38)         canonical D4 operating point
#   CONTEXT_LADDER    (70 280)     padding-line counts; line ≈ 20 tokens
#   ATTN_IMPL         (sdpa)       memory-efficient; required for >=4k Mac M4
#   SKIP_KL_OFF=1                  skip the KL OFF baseline arm
#   SKIP_KL_ON=1                   skip the KL ON arm
#   SKIP_V03=1                     skip v0.3 sink+window baseline (saves ~30%)
#
# Usage:
#
#   # Default 2-rung A/B at 1.4k + 5.6k (~7-9 h on Mac M4):
#   bash scripts/review_pr_k2a1_integration_on_mac.sh
#
#   # Quick 1.4k-only A/B (~2 h):
#   CONTEXT_LADDER='70' bash scripts/review_pr_k2a1_integration_on_mac.sh
#
# Pre-flight: kakeyalattice must be installed. The K1.E runner
# imports it lazily, so the failure mode if missing is a
# RuntimeError mid-run, not silent fallback. Confirm first:
#
#   PYTHONPATH=.:sdks/python python3 -c "import kakeyalattice; print(kakeyalattice.__name__)"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N_SAMPLES="${N_SAMPLES:-20}"
SINK="${SINK:-4}"
WINDOW="${WINDOW:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
KL_LATTICE="${KL_LATTICE:-D4}"
KL_Q_RANGE="${KL_Q_RANGE:-38}"
CONTEXT_LADDER="${CONTEXT_LADDER:-70 280}"
SKIP_KL_OFF="${SKIP_KL_OFF:-0}"
SKIP_KL_ON="${SKIP_KL_ON:-0}"
SKIP_V03="${SKIP_V03:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

# Pre-flight: kakeyalattice availability check.
echo "==> Pre-flight: kakeyalattice availability"
if PYTHONPATH=.:sdks/python python3 -c "import kakeyalattice; print(' kakeyalattice', kakeyalattice.__name__, 'OK')" 2>/dev/null; then
    echo "    kakeyalattice installed — KL ON arm will run."
else
    echo "    kakeyalattice NOT installed — KL ON arm will fail."
    echo "    Install: pip install kakeyalattice"
    if [[ "$SKIP_KL_ON" != "1" ]]; then
        echo "    Aborting (use SKIP_KL_ON=1 to run KL OFF only)."
        exit 1
    fi
fi
echo

flags_common=(
    --model google/gemma-3-1b-it
    --device auto
    --attn-impl "$ATTN_IMPL"
    --n-samples "$N_SAMPLES"
    --sink-size "$SINK"
    --window-size "$WINDOW"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
)
[[ "$SKIP_V03" == "1" ]] && flags_common+=(--skip-v03)

run_one_rung() {
    local n="$1"
    # ±15 % range around target line count — same formula as vast variant
    local lo=$(( (n * 85 + 50) / 100 ))
    local hi=$(( (n * 115 + 50) / 100 ))
    if [[ $lo -lt 10 ]]; then lo=10; fi
    if [[ $hi -lt $((lo + 1)) ]]; then hi=$((lo + 1)); fi

    if [[ "$SKIP_KL_OFF" != "1" ]]; then
        local kloff_report="${out_dir}/k2a1_niah_mac_ctx${n}_kloff_${stamp}.json"
        local kloff_log="${log_dir}/k2a1_niah_mac_ctx${n}_kloff_${stamp}.log"
        echo
        echo "==> ctx${n} KL OFF: lines [$lo, $hi]  attn=$ATTN_IMPL"
        echo "    Report: $kloff_report"
        PYTHONPATH=.:sdks/python python3 scripts/research/k1e_niah_validation.py \
            "${flags_common[@]}" \
            --haystack-min-lines "$lo" \
            --haystack-max-lines "$hi" \
            --output "$kloff_report" 2>&1 | tee "$kloff_log"
        echo "    -> finished ctx${n} KL OFF"
    fi

    if [[ "$SKIP_KL_ON" != "1" ]]; then
        local klon_report="${out_dir}/k2a1_niah_mac_ctx${n}_klon_${stamp}.json"
        local klon_log="${log_dir}/k2a1_niah_mac_ctx${n}_klon_${stamp}.log"
        echo
        echo "==> ctx${n} KL ON: lines [$lo, $hi]  lattice=$KL_LATTICE Q=$KL_Q_RANGE"
        echo "    Report: $klon_report"
        PYTHONPATH=.:sdks/python python3 scripts/research/k1e_niah_validation.py \
            "${flags_common[@]}" \
            --haystack-min-lines "$lo" \
            --haystack-max-lines "$hi" \
            --kl-on \
            --kl-lattice "$KL_LATTICE" \
            --kl-q-range "$KL_Q_RANGE" \
            --output "$klon_report" 2>&1 | tee "$klon_log"
        echo "    -> finished ctx${n} KL ON"
    fi
}

echo "==> PR-K2.A.1 KakeyaLattice integration A/B — Mac M4"
echo "    Model:           google/gemma-3-1b-it"
echo "    Samples / arm:   $N_SAMPLES"
echo "    Sink x window:   ${SINK} x ${WINDOW}"
echo "    Attn impl:       $ATTN_IMPL"
echo "    Lattice:         $KL_LATTICE  (Q=$KL_Q_RANGE)"
echo "    Context ladder:  $CONTEXT_LADDER  (padding lines)"
[[ "$SKIP_KL_OFF" == "1" ]] && echo "    SKIP_KL_OFF=1"
[[ "$SKIP_KL_ON"  == "1" ]] && echo "    SKIP_KL_ON=1"
[[ "$SKIP_V03"    == "1" ]] && echo "    SKIP_V03=1"
echo "    Time budget:     ~7-9 h for default '70 280' ladder"
echo

for n in $CONTEXT_LADDER; do
    run_one_rung "$n"
done

echo
echo "==> A/B scan complete. Reports under:"
echo "    $out_dir/k2a1_niah_mac_ctx*_kloff_${stamp}.json"
echo "    $out_dir/k2a1_niah_mac_ctx*_klon_${stamp}.json"
echo
echo "Acceptance gate (b): KL ON vs KL OFF v0.4 recall delta <= 1pp at every rung."
echo "If recall regresses > 1pp on Mac, tighten Q (KL_Q_RANGE=76) per §11.11.9 escape hatch."
echo
echo "Commit:"
echo "    git add $out_dir/k2a1_niah_mac_*_${stamp}.json"
echo "    git add $log_dir/k2a1_niah_mac_*_${stamp}.log"
echo "    git commit -m 'Mac M4 K2.A.1 KL on/off A/B evidence'"
echo "    git push"
