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
# LONG ANSWERS ARE SAFE (PR #146). The full path runs on gemma-4's native hybrid
# cache (sliding RotatingKVCache, max_size≈1024). Past that ring wrap the engine
# automatically commits single tokens (no speculative rollback to mis-trim on the
# wrapped ring), so generations stay coherent well beyond ~1024 tokens — they
# just lose the spec-decode speedup past the wrap. So the default budget below is
# generous; you no longer need to keep answers under the window.
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
#   bash scripts/run_kakeya_mac.sh --max-new-tokens 4096 --window 128
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
# Default budget reaches past the ~1024 native-cache wrap; coherent there since
# PR #146 (single-token commits past the wrap). Raise/lower freely.
MAX_NEW="${KAKEYA_MAX_NEW_TOKENS:-2048}"

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
    -h|--help)         sed -n '2,29p' "$0"; exit 0 ;;
    *)                 EXTRA+=("$1") ;;   # pass-through (e.g. --chat-scripted ...)
  esac
  shift
done

log() { echo "[run-kakeya-mac] $*" >&2; }

# Pinned interpreter (Layer B): AUTO-DISCOVER the venv python that actually has
# mlx_lm (a bare `python3` is often the system/pyenv one without it, especially
# after a reboot or in a fresh shell). Try KAKEYA_MAC_PYTHON first, then common
# venv locations, then PATH pythons; pick the first that can `import mlx_lm`.
# Set KAKEYA_MAC_PYTHON to override. See
# docs/skills/pin-selfhosted-runner-python-env-skill.md.
_can_import_mlx_lm() { [ -n "${1:-}" ] && "$1" -c 'import mlx_lm' >/dev/null 2>&1; }
_resolve_pybin() {
  local c
  for c in \
      "${KAKEYA_MAC_PYTHON:-}" \
      "$repo_root/.venv-mac/bin/python3.13" \
      "$repo_root/.venv-mac/bin/python" \
      "$repo_root/.venv/bin/python" \
      "$HOME/kakeya-venv/bin/python" \
      "$HOME/.venv/bin/python" \
      "$(command -v python3.13 2>/dev/null || true)" \
      "$(command -v python3 2>/dev/null || true)"; do
    if _can_import_mlx_lm "$c"; then printf '%s' "$c"; return 0; fi
  done
  # Nothing with mlx_lm — fall back to a sensible value so dry-run/preflight can
  # still report a clear error.
  printf '%s' "${KAKEYA_MAC_PYTHON:-python3}"
  return 1
}
PYBIN="$(_resolve_pybin)" || true
log "python  : $PYBIN$([ -n "${KAKEYA_MAC_PYTHON:-}" ] && echo ' (KAKEYA_MAC_PYTHON)')"

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
  # f_θ projection ACTUALLY RUNS each turn (the full pipeline). Coherent past the
  # ~1024 native-cache wrap (PR #146: single-token commits once the ring wraps).
  MODE="FULL (verifier + proposer + f_θ + S5 bounded KV; f_θ runs; long-answer safe)"
fi

log "mode    : $MODE"
log "verifier: $VERIFIER"
log "drafter : $DRAFTER"
log "f_theta : $FTHETA"
log "params  : sink=$SINK window=$WINDOW block=$BLOCK max_new=$MAX_NEW"

# NOTE: ``${EXTRA[@]+"${EXTRA[@]}"}`` (not a bare ``"${EXTRA[@]}"``) — under
# ``set -u`` macOS's default bash 3.2 treats expanding an EMPTY array as an
# "unbound variable" error; the ``+`` form expands to nothing when EXTRA is
# empty and to the quoted elements otherwise.
cmd=( "$PYBIN" scripts/research/k3_integrated_niah_eval_mac.py "${args[@]}" ${EXTRA[@]+"${EXTRA[@]}"} )

if [[ "$DRY_RUN" == "1" ]]; then
  echo "PYTHONPATH=.:sdks/python ${cmd[*]}"
  exit 0
fi

# ---- preflight (Apple Silicon + MLX + model) ----
"$PYBIN" -c "import mlx.core, mlx_lm" 2>/dev/null \
  || { log "no Python with mlx_lm found (auto-searched KAKEYA_MAC_PYTHON, "\
"$repo_root/.venv-mac, ~/kakeya-venv, ~/.venv, python3.13, python3; tried '$PYBIN'). "\
"Activate your MLX venv or set KAKEYA_MAC_PYTHON=/path/to/venv/bin/python "\
"(needs 'mlx mlx-lm torch transformers'). See docs/skills/pin-selfhosted-runner-python-env-skill.md"; exit 2; }
[[ -d "$VERIFIER" ]] \
  || { log "verifier model dir not found: $VERIFIER (set KAKEYA_MAC_VERIFIER_PATH)"; exit 3; }
if [[ "$FAST" != "1" && ! -e "$FTHETA" ]]; then
  log "f_θ dir not found: $FTHETA — set KAKEYA_MAC_FTHETA_DIR, or use --fast (f_θ bypassed)"
  exit 4
fi

log "starting... (type a message, blank line / Ctrl-D to quit)"
PYTHONPATH=".:sdks/python" exec "${cmd[@]}"
