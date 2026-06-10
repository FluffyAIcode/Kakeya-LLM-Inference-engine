#!/usr/bin/env bash
# Mac M4 reviewer aid for K3 integrated NIAH eval —
# the complete Kakeya inference engine product evidence on Mac MLX.
#
# Combines MLXCrossModelDLMRestoredVerifier (verifier with sink+window
# cache + drafter K/V Restoration via f_θ on MLX) with the K1.E NIAH
# evaluation harness. This is the **K3 Mac product gate**.
#
# Pre-flight requires (each fails fast with actionable error):
#   1. mlx_lm importable
#   2. PyTorch with MPS (or DRAFTER_DEVICE=cpu fallback)
#   3. Verifier dir at $VERIFIER_PATH (default
#      models/gemma-4-26B-A4B-it-mlx-4bit) with config.json
#   4. tokenizer_config.json's extra_special_tokens IS a dict
#      (PR #101 patch state — points at the patch script if list)
#   5. Drafter at $DRAFTER_ID (default models/dflash-kakeya-baseline,
#      Git LFS in main post-PR-#93)
#   6. f_θ checkpoint at $F_THETA_DIR (produced by
#      review_pr_k3_f_theta_train_on_vast.sh on vast)
#
# Validates per ADR 0008 §11.8:
#   1. Architectural correctness:
#      effective_attention_fraction = 1.0 at every NIAH ladder rung
#   2. Memory bounded:
#      driver_alloc < 24 GB on Mac M4 24 GB
#   3. Recall preservation:
#      |recall_cross_model_mac - recall_mlx_oracle| ≤ 5 pp
#
# Env knobs (defaults):
#
#   VERIFIER_PATH      models/gemma-4-26B-A4B-it-mlx-4bit
#   DRAFTER_ID         models/dflash-kakeya-baseline
#   F_THETA_DIR        results/research/f_theta_v1
#   DRAFTER_DEVICE     mps
#   N_SAMPLES          4         per ladder rung
#   SINK_SIZE          4
#   WINDOW_SIZE        64
#   MAX_NEW_TOKENS     24
#   SEED               42
#   CONTEXT_LADDER     '70 280'  (≈1.4k + ≈5.6k tokens)
#   SKIP_ORACLE=1                skip MLX oracle baseline (saves time;
#                                loses recall_delta gate signal)
#
# Usage:
#
#   bash scripts/review_pr_k3_integrated_niah_on_mac.sh
#
# Quick sanity (1.4k context, 2 samples):
#
#   N_SAMPLES=2 CONTEXT_LADDER='70' bash $0
#
# Output JSONs at:
#   results/research/k3_integrated_niah_mac_ctx<N>_<stamp>.json (per rung)
#   results/research/logs/k3_integrated_niah_mac_<stamp>.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERIFIER_PATH="${VERIFIER_PATH:-models/gemma-4-26B-A4B-it-mlx-4bit}"
DRAFTER_ID="${DRAFTER_ID:-models/dflash-kakeya-baseline}"
F_THETA_DIR="${F_THETA_DIR:-results/research/f_theta_v1}"
DRAFTER_DEVICE="${DRAFTER_DEVICE:-mps}"
N_SAMPLES="${N_SAMPLES:-4}"
SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-42}"
CONTEXT_LADDER="${CONTEXT_LADDER:-70 280}"
SKIP_ORACLE="${SKIP_ORACLE:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
log="${log_dir}/k3_integrated_niah_mac_${stamp}.log"

echo "==> K3 integrated NIAH eval (Mac M4 MLX)"
echo "    Verifier (MLX 4-bit):  $VERIFIER_PATH"
echo "    Drafter:               $DRAFTER_ID"
echo "    Drafter device:        $DRAFTER_DEVICE"
echo "    f_θ checkpoint:        $F_THETA_DIR"
echo "    N samples / rung:      $N_SAMPLES"
echo "    Sink × window:         ${SINK_SIZE} × ${WINDOW_SIZE}"
echo "    Context ladder:        $CONTEXT_LADDER"
echo "    Skip oracle:           $SKIP_ORACLE"
echo "    Log:                   $log"
echo

# Pre-flight 1: mlx_lm
if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo "ERROR: mlx_lm not installed. On Mac:"
    echo "    pip install --upgrade mlx-lm"
    exit 1
fi

# Pre-flight 2: PyTorch + MPS
if [[ "$DRAFTER_DEVICE" == "mps" ]]; then
    if ! python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        echo "ERROR: torch.backends.mps not available; try DRAFTER_DEVICE=cpu"
        exit 2
    fi
fi

# Pre-flight 3: verifier dir
if [[ ! -d "$VERIFIER_PATH" ]] || [[ ! -f "$VERIFIER_PATH/config.json" ]]; then
    echo "ERROR: verifier '$VERIFIER_PATH' missing or no config.json"
    echo "  Pre-quantize via:"
    echo "    huggingface-cli login   # Gemma 4 gated"
    echo "    PYTHONPATH=.:sdks/python python3 scripts/research/k3_quantize_for_mac.py \\"
    echo "        --output $VERIFIER_PATH"
    exit 3
fi

# Pre-flight 4: tokenizer_config patch state
patch_state=$(python3 -c "
import json, sys
p = '$VERIFIER_PATH/tokenizer_config.json'
try:
    cfg = json.load(open(p))
    extra = cfg.get('extra_special_tokens')
    if extra is None:
        print('absent')
    elif isinstance(extra, dict):
        print('dict')
    elif isinstance(extra, list):
        print('list')
    else:
        print(f'unknown:{type(extra).__name__}')
except Exception as e:
    print(f'error:{e}')
" 2>&1)
if [[ "$patch_state" == "list" ]]; then
    echo "ERROR: tokenizer_config.json's extra_special_tokens is still a list."
    echo "       Run the PR #101 patch:"
    echo "    python3 scripts/research/k3_patch_gemma4_tokenizer_config.py $VERIFIER_PATH"
    exit 4
fi
echo "    tokenizer_config patch state: $patch_state ✓"

# Pre-flight 5: drafter
case "$DRAFTER_ID" in
    models/*|./*|../*|/*)
        if [[ ! -d "$DRAFTER_ID" ]] || [[ ! -f "$DRAFTER_ID/config.json" ]]; then
            echo "ERROR: drafter '$DRAFTER_ID' missing or incomplete"
            echo "       Run: git lfs install && git lfs pull"
            exit 5
        fi
        size=$(stat -f%z "$DRAFTER_ID/model.safetensors" 2>/dev/null \
               || stat -c%s "$DRAFTER_ID/model.safetensors" 2>/dev/null \
               || echo 0)
        if [[ "$size" -lt 100000000 ]]; then
            echo "ERROR: $DRAFTER_ID/model.safetensors is $size bytes (LFS pointer)."
            echo "       Run: git lfs pull"
            exit 5
        fi
        ;;
esac

# Pre-flight 6: f_θ checkpoint
if [[ ! -d "$F_THETA_DIR" ]]; then
    echo "ERROR: f_θ '$F_THETA_DIR' missing."
    echo "       Train it on vast first:"
    echo "    HF_TOKEN=hf_xxx bash scripts/review_pr_k3_f_theta_train_on_vast.sh"
    echo "       Then push the trained checkpoint to main and pull on Mac."
    exit 6
fi
if [[ ! -f "$F_THETA_DIR/f_theta_config.json" ]] || [[ ! -f "$F_THETA_DIR/f_theta_weights.pt" ]]; then
    echo "ERROR: '$F_THETA_DIR' missing f_theta_config.json or f_theta_weights.pt."
    ls -la "$F_THETA_DIR" 2>&1 | head -10
    exit 6
fi

# Pre-flight 7: PR #103 modules importable
if ! PYTHONPATH=.:sdks/python python3 -c "
from inference_engine.v04 import (
    DFlashDrafter, FThetaProjection,
)
from inference_engine.v04.cross_model_dlm_verifier_mlx import (
    MLXCrossModelDLMRestoredVerifier,
)
" 2>&1; then
    echo "ERROR: cannot import K3 modules. Verify PR #103 + this PR have merged + 'git pull' is current."
    exit 7
fi
echo "    K3 engine modules importable ✓"
echo

# Run per-rung
flags=(
    --verifier-path "$VERIFIER_PATH"
    --drafter-id "$DRAFTER_ID"
    --f-theta-dir "$F_THETA_DIR"
    --drafter-device "$DRAFTER_DEVICE"
    --n-samples "$N_SAMPLES"
    --sink-size "$SINK_SIZE"
    --window-size "$WINDOW_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
)
[[ "$SKIP_ORACLE" == "1" ]] && flags+=(--skip-oracle)

exit_code=0
for n_lines in $CONTEXT_LADDER; do
    lo=$(( (n_lines * 85 + 50) / 100 ))
    hi=$(( (n_lines * 115 + 50) / 100 ))
    [[ $lo -lt 10 ]] && lo=10
    [[ $hi -lt $((lo + 1)) ]] && hi=$((lo + 1))
    rung_report="${out_dir}/k3_integrated_niah_mac_ctx${n_lines}_${stamp}.json"

    echo "==> ctx${n_lines}: lines [$lo, $hi]  → $rung_report"
    PYTHONPATH=.:sdks/python python3 scripts/research/k3_integrated_niah_eval_mac.py \
        --haystack-min-lines "$lo" --haystack-max-lines "$hi" \
        --output "$rung_report" \
        "${flags[@]}" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "==> ctx${n_lines} FAILED (exit=$rc); continuing"
        exit_code="$rc"
    fi
done

echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "==> K3 Mac integrated NIAH PASS (all rungs)"
    echo "    Reports:"
    for n_lines in $CONTEXT_LADDER; do
        echo "        ${out_dir}/k3_integrated_niah_mac_ctx${n_lines}_${stamp}.json"
    done
    echo
    echo "Commit evidence:"
    echo "    git add ${out_dir}/k3_integrated_niah_mac_*_${stamp}.json $log"
    echo "    git commit -m 'Mac M4 K3 integrated NIAH evidence (cross-model + f_θ + sink+window)'"
    echo "    git push"
else
    echo "==> Some rungs FAILED (last exit=$exit_code). See $log."
fi

exit "$exit_code"
