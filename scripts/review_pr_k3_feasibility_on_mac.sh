#!/usr/bin/env bash
# Mac M4 reviewer aid for K3 hardware feasibility.
#
# K3 production scale per ADR 0008 §11.7 corrected:
#   verifier:  google/gemma-4-26B-A4B-it      (26B A4B MoE, 4B active)
#   drafter:   z-lab/gemma-4-26B-A4B-it-DFlash (0.4B block-diffusion)
#
# Mac M4 24 GB cannot fit the verifier at bf16 (~52 GB). Two-step
# Mac path:
#
#   Step 1 (one-time, ~5-15 min on broadband): download the published
#   PLE-safe community 4-bit MLX variant.
#
#     PYTHONPATH=.:sdks/python python3 \
#         scripts/research/k3_quantize_for_mac.py \
#         --output models/gemma-4-26B-A4B-it-mlx-4bit
#     Result: ~16.4 GB local directory.
#
#   NOTE 2026-06-09: Step 1 used to do mlx_lm.convert self-quantize.
#   That path is broken on mlx-lm 0.31.3 for Gemma 4 26B-A4B MoE due
#   to 5 upstream bugs (ml-explore/mlx-lm#1123). Default switched to
#   downloading FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit, a working
#   PLE-safe variant. To force self-quantize anyway (e.g. when a future
#   mlx-lm release fixes the upstream bugs), pass --mode self-quantize
#   to k3_quantize_for_mac.py.
#
#   Step 2 (each smoke run, ~5-15 min): load + smoke forward
#     bash scripts/review_pr_k3_feasibility_on_mac.sh
#     Result: results/research/k3_feasibility_smoke_<stamp>.json
#
# This script is Step 2. It detects whether Step 1 has already been
# done (looks for the quantized directory) and bails with an
# actionable error if not.
#
# Memory expectation at Mac M4 24 GB:
#   model weights:       ~13 GB (verifier 4-bit) + ~0.8 GB (drafter bf16)
#   KV cache (sink+window=4+64): negligible
#   activations:         ~1-2 GB transient
#   PyTorch MPS allocator overhead: 1.5-2x
#   ────────────────────────────────────
#   estimated peak:      ~18-22 GB; fits Mac M4 24 GB tight
#
# At longer prompt (4096+ tokens), peak grows due to prefill
# activations. Test with PROMPT_TOKENS=4096 only after the 512
# baseline passes.
#
# Env knobs:
#   VERIFIER_PATH    (models/gemma-4-26B-A4B-it-mlx-4bit)
#                     local 4-bit MLX model directory
#   PROMPT_TOKENS    (512)
#   GEN_TOKENS       (8)
#   SEED             (42)
#   SKIP_DRAFTER=1            verifier-only smoke
#   ALLOW_MISSING_QUANTIZE=1  proceed even if verifier path missing
#                             (will fail at load step; useful for
#                             scripted dry-run)
#
# Usage:
#   # First time (one-time, requires HF login + 30-90 min):
#   huggingface-cli login
#   PYTHONPATH=.:sdks/python python3 \
#       scripts/research/k3_quantize_for_mac.py \
#       --output models/gemma-4-26B-A4B-it-mlx-4bit
#
#   # Subsequent smokes:
#   bash scripts/review_pr_k3_feasibility_on_mac.sh
#
#   # Longer-context test:
#   PROMPT_TOKENS=4096 bash scripts/review_pr_k3_feasibility_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERIFIER_PATH="${VERIFIER_PATH:-models/gemma-4-26B-A4B-it-mlx-4bit}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-8}"
SEED="${SEED:-42}"
SKIP_DRAFTER="${SKIP_DRAFTER:-0}"
ALLOW_MISSING_QUANTIZE="${ALLOW_MISSING_QUANTIZE:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k3_feasibility_smoke_mac_${stamp}.json"
log="${log_dir}/k3_feasibility_smoke_mac_${stamp}.log"

echo "==> K3 hardware feasibility smoke (Mac M4)"
echo "    Verifier (4-bit MLX): $VERIFIER_PATH"
echo "    Drafter:              z-lab/gemma-4-26B-A4B-it-DFlash"
echo "    Prompt:               $PROMPT_TOKENS tokens"
echo "    Gen:                  $GEN_TOKENS tokens"
echo "    Skip drafter:         $SKIP_DRAFTER"
echo "    Report:               $report"
echo

# Pre-flight 1: quantized verifier exists?
if [[ ! -d "$VERIFIER_PATH" ]]; then
    if [[ "$ALLOW_MISSING_QUANTIZE" == "1" ]]; then
        echo "WARN: verifier path '$VERIFIER_PATH' missing; proceeding due to ALLOW_MISSING_QUANTIZE=1"
    else
        echo "ERROR: verifier path '$VERIFIER_PATH' does not exist."
        echo
        echo "On Mac, the verifier must be pre-quantized via:"
        echo "    huggingface-cli login   # Gemma 4 is gated"
        echo "    PYTHONPATH=.:sdks/python python3 \\"
        echo "        scripts/research/k3_quantize_for_mac.py \\"
        echo "        --output $VERIFIER_PATH"
        echo
        echo "This is a one-time step (~30-90 min) that produces a ~13 GB"
        echo "local 4-bit MLX directory."
        echo
        echo "Set ALLOW_MISSING_QUANTIZE=1 to bypass this check (smoke will"
        echo "fail at verifier load, but the JSON report still tells you why)."
        exit 1
    fi
fi

# Pre-flight 2: HF token (DFlash drafter on HF, may also be gated)
if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "WARN: no HF auth detected. Drafter download may fail."
    echo "      huggingface-cli login   # to authenticate"
fi

# Pre-flight 3: mlx_lm installed?
if ! python3 -c "import mlx_lm" > /dev/null 2>&1; then
    echo "ERROR: mlx-lm not installed. On Mac:"
    echo "    pip install --upgrade mlx-lm"
    exit 2
fi

flags=(
    --platform mac
    --verifier-path "$VERIFIER_PATH"
    --prompt-tokens "$PROMPT_TOKENS"
    --gen-tokens "$GEN_TOKENS"
    --seed "$SEED"
    --output "$report"
)
[[ "$SKIP_DRAFTER" == "1" ]] && flags+=(--skip-drafter)

echo "==> Running smoke"
PYTHONPATH=.:sdks/python python3 scripts/research/k3_feasibility_smoke.py \
    "${flags[@]}" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}

echo
echo "==> Done. exit=$exit_code"
echo "Report: $report"
echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "Commit:"
    echo "    git add $report $log"
    echo "    git commit -m 'Mac M4 K3 hardware feasibility evidence (4-bit verifier)'"
    echo "    git push"
fi

exit $exit_code
