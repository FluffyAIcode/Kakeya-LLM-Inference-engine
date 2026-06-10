#!/usr/bin/env bash
# vast.ai (CUDA) reviewer aid for K3 Block C — f_θ K/V projection training.
#
# Pre-flight: Gemma 4 26B-A4B-it verifier (gated, needs HF_TOKEN) +
# DFlash drafter from models/dflash-kakeya-baseline/ (Git LFS, in main
# post-PR-#93). Joint memory budget: ~52 GB verifier bf16 + 0.9 GB
# drafter bf16 + ~8 GB K/V cache for 64 sequences × 128 tokens + 130 MB
# f_θ. Fits H200 80 GB single GPU comfortably; H100 80 GB also works.
#
# Output: trained f_θ checkpoint at $SAVE_DIR (default
# results/research/f_theta_v1/) containing f_theta_config.json +
# f_theta_weights.pt, plus a training report at $SAVE_DIR.json.
#
# Env knobs (defaults):
#
#   STEPS              4000        training steps; 4k is the K3 first-iteration target
#   LR                 1e-3        AdamW learning rate
#   RANK               256         f_θ low-rank bottleneck
#   N_PROMPTS          64          training corpus size (PR #93's PROMPTS)
#   GEN_LEN            128         tokens generated per prompt during data collection
#   SAMPLE_POSITIONS   256         random positions sampled per training step (memory)
#   SAVE_DIR           results/research/f_theta_v1
#   SEED               0
#
# Usage (from vast.ai host with repo synced):
#
#   HF_TOKEN=hf_xxx bash scripts/review_pr_k3_f_theta_train_on_vast.sh
#
#   # Quick sanity (10 prompts, 200 steps, ~5 min):
#   N_PROMPTS=10 STEPS=200 SAVE_DIR=results/research/f_theta_smoke \
#       HF_TOKEN=hf_xxx bash $0
#
# Expected timing on H200: data collection ~3-5 min for 64 prompts;
# training 4k steps × ~50ms/step ≈ 3-5 min. Total wall ~8-15 min.
#
# Validation gates (printed at end):
#   * loss_reduction_factor ≥ 2.0 (final loss ≤ initial / 2)
#   * f_theta_weights.pt non-empty (~130 MB at rank=256)
#
# These are sanity gates, not product gates. Product gate is the
# integrated NIAH ladder evidence (separate reviewer aid:
# scripts/review_pr_k3_integrated_niah_on_vast.sh, follow-up PR).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STEPS="${STEPS:-4000}"
LR="${LR:-1e-3}"
RANK="${RANK:-256}"
N_PROMPTS="${N_PROMPTS:-64}"
GEN_LEN="${GEN_LEN:-128}"
SAMPLE_POSITIONS="${SAMPLE_POSITIONS:-256}"
# v3: relative (magnitude-normalized) MSE + per-component diagnostics.
# Default output dir is f_theta_v3 so the v1 evidence is not overwritten.
SAVE_DIR="${SAVE_DIR:-results/research/f_theta_v3}"
LOSS_MODE="${LOSS_MODE:-relmse}"
SEED="${SEED:-0}"

stamp="$(date +%s)"
log_dir="results/research/logs"
mkdir -p "$log_dir"
log="${log_dir}/k3_f_theta_train_vast_${stamp}.log"

echo "==> K3 Block C — f_θ K/V projection training (vast.ai CUDA)"
echo "    Verifier:        google/gemma-4-26B-A4B-it (bf16, sdpa)"
echo "    Drafter:         models/dflash-kakeya-baseline (in main, Git LFS)"
echo "    Steps:           $STEPS"
echo "    LR:              $LR"
echo "    Rank:            $RANK"
echo "    N prompts:       $N_PROMPTS"
echo "    Gen len:         $GEN_LEN"
echo "    Sample positions: $SAMPLE_POSITIONS"
echo "    Loss mode:       $LOSS_MODE"
echo "    Save dir:        $SAVE_DIR"
echo "    Log:             $log"
echo

# Pre-flight 1: HF token
if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "ERROR: no HF auth detected. Run:"
    echo "    huggingface-cli login   # Gemma 4 is gated"
    echo "or:"
    echo "    export HF_TOKEN=hf_xxx"
    exit 1
fi

# Pre-flight 2: drafter checkpoint
if [[ ! -d "models/dflash-kakeya-baseline" ]]; then
    echo "ERROR: models/dflash-kakeya-baseline/ missing."
    echo "       This is Git LFS-tracked; pull via:"
    echo "           git lfs install"
    echo "           git lfs pull"
    exit 2
fi
if [[ ! -f "models/dflash-kakeya-baseline/model.safetensors" ]]; then
    echo "ERROR: models/dflash-kakeya-baseline/model.safetensors missing."
    exit 2
fi
size_bytes=$(stat -c%s "models/dflash-kakeya-baseline/model.safetensors" 2>/dev/null \
             || stat -f%z "models/dflash-kakeya-baseline/model.safetensors")
if [[ "$size_bytes" -lt 100000000 ]]; then
    echo "ERROR: model.safetensors is only $size_bytes bytes (likely LFS pointer)."
    echo "       Run 'git lfs pull' to fetch the real 859 MB file."
    exit 2
fi

# Pre-flight 3: torch + CUDA + transformers 5.x
if ! python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available', file=sys.stderr); sys.exit(2)
print(f'torch {torch.__version__} cuda={torch.version.cuda}', file=sys.stderr)
"; then
    exit 3
fi
if ! python3 -c "
import transformers, sys
v = transformers.__version__.split('.')
if int(v[0]) < 5:
    print(f'WARN: transformers {transformers.__version__} (need 5.x for Gemma 4)',
          file=sys.stderr)
print(f'transformers {transformers.__version__}', file=sys.stderr)
"; then
    exit 4
fi

# Run
echo "==> Running f_θ training"
PYTHONPATH=.:sdks/python python3 scripts/research/k3_f_theta_train.py \
    --steps "$STEPS" \
    --lr "$LR" \
    --rank "$RANK" \
    --n-prompts "$N_PROMPTS" \
    --gen-len "$GEN_LEN" \
    --sample-positions "$SAMPLE_POSITIONS" \
    --loss-mode "$LOSS_MODE" \
    --save "$SAVE_DIR" \
    --seed "$SEED" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}

echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "==> f_θ training PASS"
    echo "    Checkpoint: $SAVE_DIR/{f_theta_config.json, f_theta_weights.pt}"
    echo "    Report:     ${SAVE_DIR}.json"
    echo "    Log:        $log"
    echo
    echo "Inspect training report:"
    echo "    python3 -c 'import json; r = json.load(open(\"${SAVE_DIR}.json\"));"
    echo "        print(\"initial_loss:\", r[\"initial_loss\"]);"
    echo "        print(\"final_loss:\", r[\"final_loss\"]);"
    echo "        print(\"reduction_factor:\", r[\"loss_reduction_factor\"]);"
    echo "        print(\"train_seconds:\", r[\"train_seconds\"])'"
    echo
    echo "Commit checkpoint + report:"
    echo "    git add $SAVE_DIR/ ${SAVE_DIR}.json"
    echo "    git lfs track \"$SAVE_DIR/f_theta_weights.pt\""
    echo "    git add .gitattributes"
    echo "    git commit -m 'K3 f_θ trained checkpoint v1'"
    echo "    git push"
else
    echo "==> f_θ training FAILED (exit=$exit_code)"
    echo "    Log: $log"
fi

exit "$exit_code"
