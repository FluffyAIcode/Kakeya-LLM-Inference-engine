#!/usr/bin/env bash
# Run the FULL Kakeya Inference Engine locally on a Mac (Apple Silicon).
#
# Launches an interactive chat on the complete engine:
#   gemma-4 verifier (MLX) + DFlash proposer (fused spec-decode)
#   + f_θ K/V restoration + S5 bounded KV.
# f_θ runs by DEFAULT (the full verifier/proposer/f_θ pipeline). Use --fast for
# the all-MLX proposer path (f_θ bypassed via S5 native prefill — much faster on
# Mac, but the f_θ projection does not execute).
#
# Model facts come from env vars (set on the kakeya-mac-m4 runner), with sane
# fallbacks; override on the CLI if needed:
#   KAKEYA_MAC_VERIFIER_PATH   local MLX gemma-4 dir
#   KAKEYA_MAC_DRAFTER_ID      DFlash drafter repo/dir
#   KAKEYA_MAC_FTHETA_DIR      trained f_θ projection dir
#
# Usage:
#   bash scripts/run_kakeya_mac.sh                 # full engine (f_θ on), interactive
#   bash scripts/run_kakeya_mac.sh --fast          # proposer-only (f_θ bypassed), faster
#   bash scripts/run_kakeya_mac.sh --max-new-tokens 2048 --window 128
#   bash scripts/run_kakeya_mac.sh --dry-run       # print the command, run nothing
#   echo 'Explain proof-of-work.' | bash scripts/run_kakeya_mac.sh   # one-shot via stdin
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

VERIFIER="${KAKEYA_MAC_VERIFIER_PATH:-$HOME/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit}"
DRAFTER="${KAKEYA_MAC_DRAFTER_ID:-z-lab/gemma-4-26B-A4B-it-DFlash}"
FTHETA="${KAKEYA_MAC_FTHETA_DIR:-results/research/f_theta_v5_s5_sliding}"
SINK="${KAKEYA_SINK:-4}"
WINDOW="${KAKEYA_WINDOW:-64}"
BLOCK="${KAKEYA_BLOCK_SIZE:-4}"
MAX_NEW="${KAKEYA_MAX_NEW_TOKENS:-1024}"

FAST=0
DRY_RUN=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast)            FAST=1 ;;
    --dry-run)         DRY_RUN=1 ;;
    --verifier-path)   shift; VERIFIER="${1:?}" ;;
    --drafter-id)      shift; DRAFTER="${1:?}" ;;
    --f-theta-dir)     shift; FTHETA="${1:?}" ;;
    --max-new-tokens)  shift; MAX_NEW="${1:?}" ;;
    --window)          shift; WINDOW="${1:?}" ;;
    --sink)            shift; SINK="${1:?}" ;;
    --block-size)      shift; BLOCK="${1:?}" ;;
    -h|--help)         sed -n '2,28p' "$0"; exit 0 ;;
    *)                 EXTRA+=("$1") ;;   # pass-through (e.g. --chat-scripted ...)
  esac
  shift
done

log() { echo "[run-kakeya-mac] $*" >&2; }

# ---- argv for the full-engine harness chat ----
args=(
  --verifier-path "$VERIFIER"
  --drafter-id "$DRAFTER"
  --f-theta-dir "$FTHETA"
  --s5-exact-full-attn --fused-specdecode
  --sink-size "$SINK" --window-size "$WINDOW" --block-size "$BLOCK"
  --max-new-tokens "$MAX_NEW" --chat
)
if [[ "$FAST" == "1" ]]; then
  # all-MLX proposer + bounded trim: faster, but f_θ is bypassed (S5 free lunch).
  args+=( --all-mlx-drafter --cuda-trim )
  MODE="FAST (verifier + proposer + S5 bounded KV; f_θ BYPASSED)"
else
  # torch drafter + f_θ: the harness auto-enables --force-f-theta in --chat, so
  # f_θ projection ACTUALLY RUNS each turn (the full pipeline).
  MODE="FULL (verifier + proposer + f_θ + S5 bounded KV; f_θ runs)"
fi

log "mode    : $MODE"
log "verifier: $VERIFIER"
log "drafter : $DRAFTER"
log "f_theta : $FTHETA"
log "params  : sink=$SINK window=$WINDOW block=$BLOCK max_new=$MAX_NEW"

cmd=( python3 scripts/research/k3_integrated_niah_eval_mac.py "${args[@]}" "${EXTRA[@]}" )

if [[ "$DRY_RUN" == "1" ]]; then
  echo "PYTHONPATH=.:sdks/python ${cmd[*]}"
  exit 0
fi

# ---- preflight (Apple Silicon + MLX + model) ----
command -v python3 >/dev/null || { log "python3 not found"; exit 1; }
python3 -c "import mlx.core" 2>/dev/null \
  || { log "MLX not importable — this needs Apple Silicon + 'pip install mlx mlx-lm'"; exit 2; }
[[ -d "$VERIFIER" ]] \
  || { log "verifier model dir not found: $VERIFIER (set KAKEYA_MAC_VERIFIER_PATH)"; exit 3; }
if [[ "$FAST" != "1" && ! -e "$FTHETA" ]]; then
  log "f_θ dir not found: $FTHETA — set KAKEYA_MAC_FTHETA_DIR, or use --fast (f_θ bypassed)"
  exit 4
fi

log "starting... (type a message, blank line / Ctrl-D to quit)"
PYTHONPATH=".:sdks/python" exec "${cmd[@]}"
