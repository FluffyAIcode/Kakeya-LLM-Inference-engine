#!/usr/bin/env bash
# Mac M4 reviewer aid for PR-K2.A.0 — KakeyaLattice round-trip
# identity smoke on Apple Silicon (PyTorch MPS backend).
#
# This is the empirical-evidence path for ADR 0008 §11.11.9 (Mac
# M4 portability for K2.A): kakeyalattice is pure PyTorch, so
# in principle it runs on MPS unmodified — this script proves
# that claim end-to-end and produces the JSON evidence the K2.A
# integration PR will reference.
#
# What the smoke validates:
#
#   1. kakeyalattice imports cleanly on Mac M4
#   2. V14KakeyaZamirLatticeGPU (D4 lattice) constructs on
#      device='mps' without raising
#   3. codec.roundtrip(K) and codec.roundtrip(V) on synthetic
#      Gemma 3-1B-shape K/V (head_dim=256, num_kv_heads=1,
#      n_positions=256) produce reconstructions within the
#      relative-MSE bound (default 5e-4 — 10x looser than the
#      published CUDA envelope to absorb MPS bf16 reduction-
#      order numerics)
#   4. The inference_engine.v04.kv_compressor adapter layer
#      (IdentityCompressor + KakeyaLatticeCompressor) wraps the
#      codec correctly on MPS — compress / decompress / evict
#      all work, codec_name is self-describing, memory_bytes
#      reports a non-zero size after compress
#   5. make_default_compressor(device=mps, prefer_kakeya=True)
#      picks KakeyaLatticeCompressor (not the Identity fallback)
#
# Time budget on Mac M4 24 GB: ~30-90 s for 256 positions × 256
# head_dim. Codec construction is ~50-200 ms; round-trips are
# ~5-20 ms per K/V tensor.
#
# If kakeyalattice is not installed, the smoke exits 0 with a
# clear "skipped" status and an install hint — Mac M4 evidence
# for K2.A is then "kakeyalattice install required, then re-run".
# This is intentional: the smoke is a check, not a hard CI gate.
# The hard gate happens when K2.A integration ships with
# 'kakeyalattice' pinned in install_requires.
#
# Env knobs (defaults):
#
#   HEAD_DIM         (256)        Gemma 3-1B-it head_dim. Power of 2, divisible by 4.
#   N_POSITIONS      (256)        Synthetic batch size for round-trip
#   NUM_KV_HEADS     (1)          Gemma 3-1B has 1 kv head; production verifiers may differ
#   LATTICE          (D4)         D4 (v1.4) or E8 (v1.5)
#   Q_RANGE          (38)         Canonical D4 operating point per the KL README
#   RMSE_BOUND       (5e-4)       Pass threshold for relative MSE (10x CUDA envelope)
#   SEED             (42)
#
# Usage:
#
#   bash scripts/review_pr_k2a_kl_smoke_on_mac.sh
#
# Install kakeyalattice first if not already:
#
#   pip install kakeyalattice
#   # or, if you have a local clone of github.com/FluffyAIcode/LLM-KV--Cache-compress:
#   pip install -e <path-to-clone>/kakeyalattice/python
#
# Acceptance: exit 0 + report["summary"]["status"]=="pass" +
#   report["summary"]["mps_active"]==true (when on Mac).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HEAD_DIM="${HEAD_DIM:-256}"
N_POSITIONS="${N_POSITIONS:-256}"
NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
LATTICE="${LATTICE:-D4}"
Q_RANGE="${Q_RANGE:-38}"
RMSE_BOUND="${RMSE_BOUND:-5e-4}"
SEED="${SEED:-42}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k2a_kl_mac_smoke_${stamp}.json"
log="${log_dir}/k2a_kl_mac_smoke_${stamp}.log"

echo "==> PR-K2.A.0 KakeyaLattice Mac M4 smoke"
echo "    Lattice:           $LATTICE  (Q=$Q_RANGE)"
echo "    Head dim:          $HEAD_DIM"
echo "    Batch:             $N_POSITIONS positions × $NUM_KV_HEADS kv-heads"
echo "    RMSE bound:        $RMSE_BOUND  (pass threshold)"
echo "    Report:            $report"
echo "    Log:               $log"
echo

# Pre-flight: confirm kakeyalattice is reachable. The smoke
# itself handles the missing case gracefully (exits 0, status=skipped),
# but a quick stderr signal here helps the reviewer know what to expect.
echo "==> kakeyalattice availability check"
if PYTHONPATH=.:sdks/python python3 -c "import kakeyalattice; print(kakeyalattice.__name__, 'OK')" 2>/dev/null; then
    echo "    kakeyalattice installed; smoke will run all checks."
else
    echo "    kakeyalattice NOT installed; smoke will run in skipped mode."
    echo "    install with: pip install kakeyalattice"
fi
echo

PYTHONPATH=.:sdks/python python3 scripts/research/k2a_kl_mac_smoke.py \
    --device auto \
    --head-dim "$HEAD_DIM" \
    --n-positions "$N_POSITIONS" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --lattice "$LATTICE" \
    --q-range "$Q_RANGE" \
    --rmse-bound "$RMSE_BOUND" \
    --seed "$SEED" \
    --output "$report" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}

echo
echo "==> Done. Report: $report"
echo "    exit code: $exit_code"
echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "Commit:"
    echo "    git add $report $log"
    echo "    git commit -m 'Mac M4 K2.A.0 KakeyaLattice round-trip-identity smoke evidence'"
    echo "    git push"
fi

exit $exit_code
