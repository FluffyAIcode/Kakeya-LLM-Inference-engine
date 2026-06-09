#!/usr/bin/env bash
# vast.ai (CUDA) reviewer aid for PR-K2.A.1 — KakeyaLattice
# integration A/B at the §11.12 ladder.
#
# Runs the K1.E NIAH harness twice at every context rung — once
# with KL OFF (K1 baseline, IdentityCompressor; reproduces
# aab8686 evidence) and once with KL ON (KakeyaLattice D4 Q=38).
# Schema v5 JSON evidence is committed back; the K2.A acceptance
# gates of ADR 0008 §11.11.5 are evaluated by reading both JSONs
# at each rung:
#
#   gate (a) round-trip identity  — sanity from KL on JSON;
#                                   KakeyaLattice's published
#                                   CUDA fidelity envelope (3e-5
#                                   K rel MSE on D4 Q=38) is the
#                                   reference.
#   gate (b) recall delta <= 1pp — compare results.v04_dlm_restored.recall
#                                   between the two JSONs at every
#                                   rung. THIS is the K2.A binding
#                                   gate.
#   gate (c) throughput >= 1.3x at 22k+ — compare
#                                   results.v04_dlm_restored.mean_throughput_tokens_per_sec
#                                   between KL on / KL off at the
#                                   16k, 64k, 100k rungs.
#                                   K2.A.1 (this PR, stateless KL
#                                   round-trip) is NOT expected to
#                                   pass gate (c) on its own — the
#                                   stateless round-trip adds compress+
#                                   decompress overhead per step
#                                   without caching savings. K2.A.2
#                                   (future PR, stateful caching)
#                                   closes gate (c) by eliminating
#                                   per-step verifier-forward
#                                   recomputation. Gate (c) data
#                                   collected here is the BASELINE
#                                   K2.A.2 will be measured against.
#
# Also closes the §11.11.9 sustained-memory empirical gap:
#   CUDA peak_allocated_bytes via K1.G (schema v2+) is precise on
#   CUDA — distinguishes oracle's full-prefix KV from v0.4's
#   sink+window-bounded resident KV. KL on / off comparison shows
#   the sustained-memory delta KL provides.
#
# Time budget on a vast.ai NVIDIA H100 (80 GB) — same as
# scripts/review_pr_k1e_on_vast.sh but doubled (KL on + KL off):
#
#   Default ladder (1k / 4k / 16k tokens), N=20:    ~2-3 h
#   Full ladder (1k / 4k / 16k / 64k / 100k):       ~6-8 h
#
# Default lattice = D4 Q=38 (canonical, lower per-block compute).
#
# Env knobs (defaults):
#
#   N_SAMPLES         (20)         samples per (config, context, kl-state)
#   SINK              (4)
#   WINDOW            (64)
#   MAX_NEW_TOKENS    (24)
#   SEED              (42)
#   ATTN_IMPL         (sdpa)       memory-efficient SDPA path; needed for >=88k
#   KL_LATTICE        (D4)         D4 (v1.4) or E8 (v1.5)
#   KL_Q_RANGE        (38)         canonical D4 operating point
#   CONTEXT_LADDER    (70 280 1100) padding-line counts; line ≈ 20 tokens
#                                  with chat template. To reach 64k+ rungs:
#                                  '70 280 1100 3200 5000'
#   SKIP_KL_OFF=1                  skip the KL OFF baseline rung (only run KL ON)
#   SKIP_KL_ON=1                   skip the KL ON arm
#   SKIP_V03=1                     skip the v0.3 sink+window baseline (saves ~30%)
#
# Usage:
#
#   # Default 3-rung A/B (~2-3 h on H100):
#   bash scripts/review_pr_k2a1_integration_on_vast.sh
#
#   # Full 5-rung A/B reaching 100k (~6-8 h):
#   CONTEXT_LADDER='70 280 1100 3200 5000' \
#       bash scripts/review_pr_k2a1_integration_on_vast.sh
#
# Acceptance signals:
#
#   * KL_ON / KL_OFF v0.4 recall delta <= 1pp at every rung
#     (gate (b) — binding for K2.A acceptance)
#   * KL_ON v0.4 peak_mem <= KL_OFF v0.4 peak_mem (KL is doing
#     useful work at the architectural level even if K2.A.1
#     stateless doesn't cash in on throughput)
#   * KL_ON v0.4 throughput >= K2.A.1 baseline (informational;
#     gate (c) requires K2.A.2 stateful caching)

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
CONTEXT_LADDER="${CONTEXT_LADDER:-70 280 1100}"
SKIP_KL_OFF="${SKIP_KL_OFF:-0}"
SKIP_KL_ON="${SKIP_KL_ON:-0}"
SKIP_V03="${SKIP_V03:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

flags_common=(
    --model google/gemma-3-1b-it
    --device cuda
    --attn-impl "$ATTN_IMPL"
    --n-samples "$N_SAMPLES"
    --sink-size "$SINK"
    --window-size "$WINDOW"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
)
[[ "$SKIP_V03" == "1" ]] && flags_common+=(--skip-v03)

export KAKEYA_VAST_SCRIPT="scripts/research/k1e_niah_validation.py"

echo "==> Provisioning venv (one-time)"
bash scripts/research/run_on_vast.sh --setup-only

run_one_rung() {
    local n="$1"
    # ±15 % range around target line count
    local lo=$(( (n * 85 + 50) / 100 ))
    local hi=$(( (n * 115 + 50) / 100 ))
    if [[ $lo -lt 10 ]]; then lo=10; fi
    if [[ $hi -lt $((lo + 1)) ]]; then hi=$((lo + 1)); fi

    if [[ "$SKIP_KL_OFF" != "1" ]]; then
        local kloff_report="${out_dir}/k2a1_niah_vast_ctx${n}_kloff_${stamp}.json"
        local kloff_log="${log_dir}/k2a1_niah_vast_ctx${n}_kloff_${stamp}.log"
        echo
        echo "==> ctx${n} KL OFF: lines [$lo, $hi]"
        echo "    Report: $kloff_report"
        bash scripts/research/run_on_vast.sh \
            "${flags_common[@]}" \
            --haystack-min-lines "$lo" \
            --haystack-max-lines "$hi" \
            --output "$kloff_report" \
            2>&1 | tee "$kloff_log"
        echo "    -> finished ctx${n} KL OFF"
    fi

    if [[ "$SKIP_KL_ON" != "1" ]]; then
        local klon_report="${out_dir}/k2a1_niah_vast_ctx${n}_klon_${stamp}.json"
        local klon_log="${log_dir}/k2a1_niah_vast_ctx${n}_klon_${stamp}.log"
        echo
        echo "==> ctx${n} KL ON: lines [$lo, $hi]  lattice=$KL_LATTICE Q=$KL_Q_RANGE"
        echo "    Report: $klon_report"
        bash scripts/research/run_on_vast.sh \
            "${flags_common[@]}" \
            --haystack-min-lines "$lo" \
            --haystack-max-lines "$hi" \
            --kl-on \
            --kl-lattice "$KL_LATTICE" \
            --kl-q-range "$KL_Q_RANGE" \
            --output "$klon_report" \
            2>&1 | tee "$klon_log"
        echo "    -> finished ctx${n} KL ON"
    fi
}

echo
echo "==> PR-K2.A.1 KakeyaLattice integration A/B — vast.ai CUDA"
echo "    Model:           google/gemma-3-1b-it"
echo "    Samples / arm:   $N_SAMPLES"
echo "    Sink x window:   ${SINK} x ${WINDOW}"
echo "    Attn impl:       $ATTN_IMPL"
echo "    Lattice:         $KL_LATTICE  (Q=$KL_Q_RANGE)"
echo "    Context ladder:  $CONTEXT_LADDER  (padding lines)"
[[ "$SKIP_KL_OFF" == "1" ]] && echo "    SKIP_KL_OFF=1"
[[ "$SKIP_KL_ON"  == "1" ]] && echo "    SKIP_KL_ON=1"
[[ "$SKIP_V03"    == "1" ]] && echo "    SKIP_V03=1"
echo

for n in $CONTEXT_LADDER; do
    run_one_rung "$n"
done

echo
echo "==> A/B scan complete. Reports under:"
echo "    $out_dir/k2a1_niah_vast_ctx*_kloff_${stamp}.json"
echo "    $out_dir/k2a1_niah_vast_ctx*_klon_${stamp}.json"
echo
echo "Compare KL OFF vs KL ON at each rung — gate (b) recall delta is the"
echo "K2.A binding signal. Gate (c) throughput delta at 22k+ is informational"
echo "(K2.A.1 stateless does not target throughput; K2.A.2 stateful does)."
echo
echo "Commit:"
echo "    git add $out_dir/k2a1_niah_vast_*_${stamp}.json"
echo "    git add $log_dir/k2a1_niah_vast_*_${stamp}.log"
echo "    git commit -m 'vast H100 K2.A.1 KL on/off A/B evidence'"
echo "    git push"
