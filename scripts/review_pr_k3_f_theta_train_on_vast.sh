#!/usr/bin/env bash
# vast.ai (CUDA) reviewer aid for K3 Block C — f_θ K/V projection training.
#
# v3 (2026-06-10) — ONE-SHOT principled trainer.
#   - --loss-type attn_distill   (attention-output distillation — the
#                                 mathematically right loss for K/V
#                                 projection; v1 was raw MSE on K/V,
#                                 v2 intermediate was cos+mag)
#   - --rank 768                 (3× v1's 256 capacity at f_θ bottleneck)
#   - --steps 20000              (5× v1; v1 was 4k → 59s, undertrained)
#   - --gen-len 512              (4× v1; v1 was 128)
#   - --lr-schedule cosine       (linear warmup → cosine decay to peak/100)
#   - +64 NIAH-style synthetic prompts (v1 had zero retrieval data)
# v1 reproduction: STEPS=4000 GEN_LEN=128 LR_SCHEDULE=const LOSS_TYPE=mse
#   N_NIAH_PROMPTS=0 RANK=256
#
# Pre-flight: Gemma 4 26B-A4B-it verifier (gated, needs HF_TOKEN) +
# DFlash drafter from models/dflash-kakeya-baseline/ (Git LFS, in main
# post-PR-#93). Joint memory budget: ~52 GB verifier bf16 + 0.9 GB
# drafter bf16 + ~30 GB K/V cache for 128 sequences × 512 tokens + 130 MB
# f_θ. Fits H200 80 GB single GPU; H100 80 GB also works.
#
# Output: trained f_θ checkpoint at $SAVE_DIR (default
# results/research/f_theta_v2/) containing f_theta_config.json +
# f_theta_weights.pt, plus a training report at $SAVE_DIR.json.
#
# Env knobs (v3 defaults):
#
#   STEPS              20000          training steps (v3 = 5× v1)
#   LR                 1e-3           peak AdamW learning rate
#   LR_SCHEDULE        cosine         const | cosine
#   WARMUP_STEPS       500
#   LOSS_TYPE          attn_distill   attn_distill | mse | cos_mag | combined
#   RANK               (auto)         empty = trainer auto-picks 768 for
#                                     attn_distill / 256 for legacy losses
#   N_PROMPTS          64
#   N_NIAH_PROMPTS     64
#   GEN_LEN            512
#   SAMPLE_POSITIONS   0              0 = full T (recommended for attn_distill)
#   SAVE_DIR           results/research/f_theta_v3
#   SEED               0
#
# Usage (from vast.ai host with repo synced):
#
#   HF_TOKEN=hf_xxx bash scripts/review_pr_k3_f_theta_train_on_vast.sh
#
#   # Quick sanity (10 prompts, 200 steps, NIAH off, ~5 min):
#   N_PROMPTS=10 N_NIAH_PROMPTS=0 STEPS=200 \
#       SAVE_DIR=results/research/f_theta_smoke \
#       HF_TOKEN=hf_xxx bash $0
#
#   # v1 reproduction (for direct comparability with PR #103 evidence):
#   STEPS=4000 GEN_LEN=128 LR_SCHEDULE=const LOSS_TYPE=mse \
#       RANK=256 N_NIAH_PROMPTS=0 \
#       SAVE_DIR=results/research/f_theta_v1_repro \
#       HF_TOKEN=hf_xxx bash $0
#
# Expected timing on H200:
#   - Data collection:  ~15-25 min (128 prompts × 512 gen_len each;
#     NIAH prompts longer due to haystack; eager-attn forward is
#     somewhat slower than sdpa)
#   - Training 20k steps × ~80 ms/step (attention forward through
#     all 30 layers per step) ≈ 25-30 min
#   - Total wall: ~40-60 min
#
# Validation gates (printed at end):
#   * loss_reduction_factor ≥ 5.0
#   * mseO/|O_tgt|^2 ratio < 0.05 → attention output preserved
#     (v3 attn_distill diagnostic)
#   * f_theta_weights.pt non-empty (~352 MB at rank=768)
#
# These are sanity gates, not product gates. Product gate is the
# integrated NIAH ladder evidence (separate reviewer aid:
# scripts/review_pr_k3_integrated_niah_on_vast.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Per docs/agent-workflow-rules.md R2 — print branch + HEAD + recipe
# at startup so the user (and reviewing agent) can verify what's
# about to spend GPU time IS what they think it is.
# shellcheck disable=SC1091
source "$ROOT/scripts/_lib/reviewer_aid_header.sh"

STEPS="${STEPS:-20000}"
LR="${LR:-1e-3}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
LOSS_TYPE="${LOSS_TYPE:-attn_distill}"
RANK="${RANK:-}"        # empty = trainer auto-picks (768 for attn_distill, else 256)
N_PROMPTS="${N_PROMPTS:-64}"
N_NIAH_PROMPTS="${N_NIAH_PROMPTS:-64}"
GEN_LEN="${GEN_LEN:-512}"
SAMPLE_POSITIONS="${SAMPLE_POSITIONS:-0}"   # 0 = full T (attn_distill default)
SAVE_DIR="${SAVE_DIR:-results/research/f_theta_v3}"
SEED="${SEED:-0}"

stamp="$(date +%s)"
log_dir="results/research/logs"
mkdir -p "$log_dir"
log="${log_dir}/k3_f_theta_train_vast_${stamp}.log"

attn_impl_msg="eager"
if [[ "$LOSS_TYPE" != "attn_distill" ]]; then attn_impl_msg="sdpa"; fi
rank_msg="$RANK"
if [[ -z "$RANK" ]]; then
    if [[ "$LOSS_TYPE" == "attn_distill" ]]; then rank_msg="auto (768)"; else rank_msg="auto (256)"; fi
fi

# Recipe summary for the R2 header — must include all knobs that
# affect "what code/config will run". Order: most-impactful first.
recipe="loss=$LOSS_TYPE rank=$rank_msg steps=$STEPS gen_len=$GEN_LEN"
recipe+=" lr_schedule=$LR_SCHEDULE warmup=$WARMUP_STEPS"
recipe+=" n_general=$N_PROMPTS n_niah=$N_NIAH_PROMPTS"
recipe+=" save=$SAVE_DIR"
print_aid_header "$0" "$recipe"
echo "==> K3 Block C — f_θ K/V projection training (vast.ai CUDA, v3)"
echo "    Verifier:          google/gemma-4-26B-A4B-it (bf16, $attn_impl_msg)"
echo "    Drafter:           models/dflash-kakeya-baseline (in main, Git LFS)"
echo "    Loss type:         $LOSS_TYPE"
echo "    Steps:             $STEPS"
echo "    Peak LR:           $LR (schedule: $LR_SCHEDULE, warmup: $WARMUP_STEPS)"
echo "    Rank:              $rank_msg"
echo "    N general prompts: $N_PROMPTS"
echo "    N NIAH prompts:    $N_NIAH_PROMPTS"
echo "    Gen len:           $GEN_LEN"
echo "    Sample positions:  $SAMPLE_POSITIONS  (0 = full T)"
echo "    Save dir:          $SAVE_DIR"
echo "    Log:               $log"
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
echo "==> Running f_θ training (v3)"
extra_flags=()
if [[ "$N_NIAH_PROMPTS" -eq 0 ]]; then
    extra_flags+=(--no-niah-prompts)
fi
if [[ -n "$RANK" ]]; then
    extra_flags+=(--rank "$RANK")
fi
PYTHONPATH=.:sdks/python python3 scripts/research/k3_f_theta_train.py \
    --steps "$STEPS" \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --warmup-steps "$WARMUP_STEPS" \
    --loss-type "$LOSS_TYPE" \
    --n-prompts "$N_PROMPTS" \
    --n-niah-prompts "$N_NIAH_PROMPTS" \
    --gen-len "$GEN_LEN" \
    --sample-positions "$SAMPLE_POSITIONS" \
    --save "$SAVE_DIR" \
    --seed "$SEED" "${extra_flags[@]}" 2>&1 | tee "$log"
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
