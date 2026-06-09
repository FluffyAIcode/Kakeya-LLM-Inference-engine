#!/usr/bin/env bash
# vast.ai (CUDA) reviewer aid for K3 hardware feasibility.
#
# Loads the K3 production-scale models per ADR 0008 §11.7 corrected
# table (HF-verified §11.14.3):
#   verifier:  google/gemma-4-26B-A4B-it      (26B A4B MoE, 4B active)
#   drafter:   z-lab/gemma-4-26B-A4B-it-DFlash (0.4B block-diffusion)
#
# The smoke validates: load both models, run a smoke forward on each,
# report memory + latency. Does NOT exercise:
#   * cross-model DLMRestoredVerifier (see docs/design/k3-cross-model-
#     dlmrestored-verifier-contract.md for the contract; implementation
#     is the K2.B/K3 PR scope)
#   * trained f_θ projection (see docs/design/k3-f-theta-training-
#     pipeline.md)
#   * NIAH ladder evidence (requires the above two)
#
# Hardware target: vast.ai single H100 80 GB or H200 80 GB+.
# Total bf16 footprint at smoke prompt size: ~52 GB verifier + ~0.8 GB
# drafter + activations + KV cache. Fits 80 GB single GPU comfortably
# at 512-prompt smoke; 100k-context tests need separate budget analysis.
#
# Pre-flight: HF token must be set. Gemma 4 is gated. Either:
#   export HF_TOKEN=hf_xxx
# or
#   huggingface-cli login
#
# Env knobs:
#   PROMPT_TOKENS   (512)   smoke prompt length; raise (4096, 16384) for
#                           longer-context feasibility test
#   GEN_TOKENS      (8)     gen tokens to measure tok/s (greedy)
#   SEED            (42)
#   SKIP_DRAFTER=1          verifier-only smoke (isolates verifier memory)
#
# Usage:
#   bash scripts/review_pr_k3_feasibility_on_vast.sh
#
# Output:
#   results/research/k3_feasibility_smoke_<stamp>.json
#   results/research/logs/k3_feasibility_smoke_<stamp>.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-8}"
SEED="${SEED:-42}"
SKIP_DRAFTER="${SKIP_DRAFTER:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k3_feasibility_smoke_vast_${stamp}.json"
log="${log_dir}/k3_feasibility_smoke_vast_${stamp}.log"

echo "==> K3 hardware feasibility smoke (vast.ai CUDA)"
echo "    Verifier:        google/gemma-4-26B-A4B-it   (bf16)"
echo "    Drafter:         z-lab/gemma-4-26B-A4B-it-DFlash"
echo "    Prompt:          $PROMPT_TOKENS tokens (synthetic)"
echo "    Gen:             $GEN_TOKENS tokens (greedy)"
echo "    Skip drafter:    $SKIP_DRAFTER"
echo "    Report:          $report"
echo

# Pre-flight: HF token check
if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "ERROR: no HF auth detected. Run:"
    echo "    huggingface-cli login"
    echo "or:"
    echo "    export HF_TOKEN=hf_xxx"
    echo "Gemma 4 is a gated model; auth is required."
    exit 1
fi

# Build the runner via the existing run_on_vast.sh wrapper for consistency
# with K1.E vast runs. Set KAKEYA_VAST_SCRIPT to point at the smoke.
export KAKEYA_VAST_SCRIPT="scripts/research/k3_feasibility_smoke.py"

flags=(
    --platform cuda
    --prompt-tokens "$PROMPT_TOKENS"
    --gen-tokens "$GEN_TOKENS"
    --seed "$SEED"
    --output "$report"
)
[[ "$SKIP_DRAFTER" == "1" ]] && flags+=(--skip-drafter)

echo "==> Provisioning venv (one-time)"
bash scripts/research/run_on_vast.sh --setup-only

echo
echo "==> Running smoke"
bash scripts/research/run_on_vast.sh "${flags[@]}" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}

echo
echo "==> Done. exit=$exit_code"
echo "Report: $report"
echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "Commit:"
    echo "    git add $report $log"
    echo "    git commit -m 'vast K3 hardware feasibility evidence'"
    echo "    git push"
fi

exit $exit_code
