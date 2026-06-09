#!/usr/bin/env bash
# Mac M4 reviewer aid for K3 cross-model speculative decoding (Step 3b).
#
# Background
# ----------
#
# Wraps scripts/research/k3_dflash_specdecode_eval_mac.py — the Mac
# variant of PR #93's CUDA spec decode eval. Drives:
#
#   verifier:  google/gemma-4-26B-A4B-it via mlx_lm 4-bit (Apple Silicon)
#   drafter:   models/dflash-kakeya-baseline/ (PR #93's alignment-trained
#              DFlash; PyTorch bf16 on MPS)
#
# Through the cross-runtime spec decode loop in
# scripts/research/k3_dflash_mlx_bridge.py.
#
# Pre-flight checks (each fails fast with actionable error if violated):
#
#   1. mlx_lm importable
#   2. PyTorch with MPS support importable
#   3. Verifier MLX dir present at $VERIFIER_PATH
#   4. Drafter at $DRAFTER_ID present (local path or HF id)
#   5. tokenizer_config.json's extra_special_tokens is a dict (per
#      PR #101 patch). If still a list, point at the patch script.
#   6. PR #93's DFlashDrafter importable from inference_engine.v04
#
# Pre-flight 5 is the "did the user run the patch script" check —
# without it, the eval script will fail at mlx_lm.load with a
# bug4 fingerprint. Pre-flight catches it 1 second in instead.
#
# Env knobs
# ---------
#
#   VERIFIER_PATH    (models/gemma-4-26B-A4B-it-mlx-4bit)
#   DRAFTER_ID       (models/dflash-kakeya-baseline)
#   DRAFTER_DEVICE   (mps)        cpu fallback if MPS misbehaves
#   N_PROMPTS        (4)
#   BLOCK_SIZE       (16)         DFlash standard
#   NUM_STEPS        (1)          DFlash uses 1 pass per block
#   MAX_NEW_TOKENS   (48)
#   HELD_OUT=1                    use HELD_OUT_PROMPTS (disjoint from
#                                 PR #93 alignment training corpus)
#
# Output
# ------
#
#   results/research/k3_dflash_specdecode_mac_<stamp>.json
#   results/research/logs/k3_dflash_specdecode_mac_<stamp>.log
#
# JSON schema mirrors PR #93's CUDA eval evidence at
#   results/research/k3_dflash_specdecode_corpus_heldout.json
# so the two are directly comparable. Aggregate.reference_cuda_held_out
# embeds the CUDA baseline (0.107 acceptance / 2.45 length) for direct
# comparison in the same JSON.
#
# Usage
# -----
#
# First run (after pulling main with PR #99 + #101 + #100 + Step 3b):
#
#     # one-time tokenizer patch (idempotent):
#     python3 scripts/research/k3_patch_gemma4_tokenizer_config.py \
#         models/gemma-4-26B-A4B-it-mlx-4bit
#
#     # run the spec decode eval:
#     bash scripts/review_pr_k3_dflash_specdecode_on_mac.sh
#
# Held-out evaluation (compares directly to PR #93's CUDA held-out):
#
#     HELD_OUT=1 bash scripts/review_pr_k3_dflash_specdecode_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERIFIER_PATH="${VERIFIER_PATH:-models/gemma-4-26B-A4B-it-mlx-4bit}"
DRAFTER_ID="${DRAFTER_ID:-models/dflash-kakeya-baseline}"
DRAFTER_DEVICE="${DRAFTER_DEVICE:-mps}"
N_PROMPTS="${N_PROMPTS:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_STEPS="${NUM_STEPS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
HELD_OUT="${HELD_OUT:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k3_dflash_specdecode_mac_${stamp}.json"
log="${log_dir}/k3_dflash_specdecode_mac_${stamp}.log"

echo "==> K3 cross-model speculative decoding (Mac M4)"
echo "    Verifier (MLX 4-bit): $VERIFIER_PATH"
echo "    Drafter:              $DRAFTER_ID"
echo "    Drafter device:       $DRAFTER_DEVICE"
echo "    Block size:           $BLOCK_SIZE"
echo "    Num steps:            $NUM_STEPS"
echo "    Max new tokens:       $MAX_NEW_TOKENS"
echo "    N prompts:            $N_PROMPTS"
echo "    Held out:             $HELD_OUT"
echo "    Report:               $report"
echo

# ---------------------------------------------------------------------------
# Pre-flight 1: mlx_lm importable
# ---------------------------------------------------------------------------
if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo "ERROR: mlx_lm not installed. On Mac:"
    echo "    pip install --upgrade mlx-lm"
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-flight 2: PyTorch + MPS check
# ---------------------------------------------------------------------------
if ! python3 -c "
import torch, sys
if not torch.backends.mps.is_available():
    print('NOTE: torch.backends.mps not available; will need DRAFTER_DEVICE=cpu', file=sys.stderr)
    sys.exit(2)
" 2>&1; then
    if [[ "$DRAFTER_DEVICE" == "mps" ]]; then
        echo "ERROR: torch.backends.mps unavailable but DRAFTER_DEVICE=mps."
        echo "       Try DRAFTER_DEVICE=cpu (slower but works without MPS)."
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Pre-flight 3: verifier dir exists
# ---------------------------------------------------------------------------
if [[ ! -d "$VERIFIER_PATH" ]]; then
    echo "ERROR: verifier path '$VERIFIER_PATH' missing."
    echo
    echo "Pre-quantize via:"
    echo "    huggingface-cli login"
    echo "    PYTHONPATH=.:sdks/python python3 \\"
    echo "        scripts/research/k3_quantize_for_mac.py \\"
    echo "        --output $VERIFIER_PATH"
    exit 3
fi
if [[ ! -f "$VERIFIER_PATH/config.json" ]]; then
    echo "ERROR: verifier dir '$VERIFIER_PATH' has no config.json."
    exit 3
fi

# ---------------------------------------------------------------------------
# Pre-flight 4: drafter source check (local path heuristic from PR #99's
# fail-fast logic — adapted for this script).
# ---------------------------------------------------------------------------
case "$DRAFTER_ID" in
    models/*|./*|../*|/*)
        if [[ ! -d "$DRAFTER_ID" ]]; then
            echo "ERROR: DRAFTER_ID='$DRAFTER_ID' looks like a local path but does not exist."
            echo
            echo "If you intended the alignment-trained baseline (the default),"
            echo "verify it's checked out via 'git lfs pull'. The baseline ships"
            echo "in-tree at models/dflash-kakeya-baseline/ via Git LFS as of"
            echo "PR #93 merge."
            echo
            echo "If you intended an HF repo id (research-baseline comparison),"
            echo "use the format 'org/repo' (e.g. 'z-lab/gemma-4-26B-A4B-it-DFlash')."
            exit 4
        fi
        if [[ ! -f "$DRAFTER_ID/config.json" ]]; then
            echo "ERROR: DRAFTER_ID='$DRAFTER_ID' is a directory but lacks config.json."
            ls -la "$DRAFTER_ID" 2>&1 | head -10
            exit 4
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# Pre-flight 5: tokenizer_config patch state
# ---------------------------------------------------------------------------
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
except FileNotFoundError:
    print('no_tokenizer_config')
except json.JSONDecodeError:
    print('invalid_json')
" 2>&1)
if [[ "$patch_state" == "list" ]]; then
    echo "ERROR: tokenizer_config.json's extra_special_tokens is still a list."
    echo "       This will trigger bug4 in mlx_lm.load. Run the PR #101 patch:"
    echo
    echo "    python3 scripts/research/k3_patch_gemma4_tokenizer_config.py \\"
    echo "        $VERIFIER_PATH"
    echo
    exit 5
fi
if [[ "$patch_state" == invalid_json* ]] || [[ "$patch_state" == no_tokenizer_config* ]]; then
    echo "ERROR: tokenizer_config.json missing or invalid: $patch_state"
    exit 5
fi
echo "    tokenizer_config patch state: $patch_state ✓"

# ---------------------------------------------------------------------------
# Pre-flight 6: PR #93 DFlashDrafter importable
# ---------------------------------------------------------------------------
if ! PYTHONPATH=.:sdks/python python3 -c "
from inference_engine.v04.dflash_drafter import DFlashDrafter, DFlashProposer
" 2>&1; then
    echo "ERROR: cannot import inference_engine.v04.dflash_drafter."
    echo "       Verify PR #93 has merged to main + 'git pull' is current."
    exit 6
fi
echo "    PR #93 DFlashDrafter importable ✓"

echo

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
flags=(
    --verifier-path "$VERIFIER_PATH"
    --drafter-id "$DRAFTER_ID"
    --drafter-device "$DRAFTER_DEVICE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --block-size "$BLOCK_SIZE"
    --num-steps "$NUM_STEPS"
    --n-prompts "$N_PROMPTS"
    --output "$report"
)
[[ "$HELD_OUT" == "1" ]] && flags+=(--held-out)

echo "==> Running cross-model spec decode eval"
PYTHONPATH=.:sdks/python python3 scripts/research/k3_dflash_specdecode_eval_mac.py \
    "${flags[@]}" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}

echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "==> PASS"
    echo "    Report:  $report"
    echo "    Log:     $log"
    echo
    echo "Inspect the four key product metrics:"
    echo "    python3 -c 'import json; r = json.load(open(\"$report\"));"
    echo "        a = r[\"aggregate\"];"
    echo "        print(f\"acceptance_rate:    {a[\\\"acceptance_rate\\\"]:.3f}\");"
    echo "        print(f\"acceptance_length:  {a[\\\"acceptance_length\\\"]:.2f}\");"
    echo "        print(f\"lossless_vs_ar:     {a[\\\"lossless_vs_ar\\\"]}\");"
    echo "        print(f\"total_blocks:       {a[\\\"total_blocks\\\"]}\");'"
    echo
    echo "Compare directly with PR #93's CUDA baseline at:"
    echo "    results/research/k3_dflash_specdecode_corpus_heldout.json"
    echo
    echo "Commit evidence:"
    echo "    git add $report $log"
    echo "    git commit -m 'Mac M4 K3 cross-model spec decode eval evidence'"
    echo "    git push"
else
    echo "==> FAILED (exit=$exit_code)"
    echo "    Log: $log"
    echo
    echo "If failure is at mlx_lm.load, push the smoke harness diagnostic"
    echo "(scripts/research/k3_feasibility_smoke.py) which captures structured"
    echo "JSON evidence with bug fingerprints (see PR #99 + PR #101)."
fi

exit "$exit_code"
