#!/usr/bin/env bash
# Fetch the alignment-trained DFlash drafter baseline
# (models/dflash-kakeya-baseline) from PR #93's branch into a local
# cache, without polluting the current worktree's git state.
#
# Why this exists
# ---------------
#
# The alignment-trained baseline lives at:
#
#   repo:    FluffyAIcode/Kakeya-LLM-Inference-engine
#   branch:  AgentMemory/v04-pr-k3-dflash-native-integration-2815  (PR #93)
#   path:    models/dflash-kakeya-baseline/
#   commit:  19a2d5c
#   size:    859 MB bf16 (Git LFS)
#
# But ALL the K3 Mac smoke / DFlashDrafter product API work
# (PRs #95–#98) is on a SEPARATE stack rooted at PR #92, and PR #93
# is parallel — not in their lineage. So a user's K3 worktree
# (e.g. Kakeya-LLM-Inference-engine-k3-dlm-proposer/) does NOT have
# the LFS-tracked baseline checked out, and `DRAFTER_ID=models/
# dflash-kakeya-baseline` resolves to a missing path.
#
# Three options to handle this:
#
#   (A) Merge PR #93 into main first, then rebase the K3 stack.
#       Cleanest long-term; requires PR #93 review. Tracked separately
#       (user offered: "需要的话我可以把 PR #93 转成非 draft 或推进合并").
#
#   (B) git-checkout PR #93's path into the user's K3 worktree, then
#       git rm --cached. Works but pollutes the working tree with
#       untracked files and risks accidental commit if the user runs
#       `git add .`.
#
#   (C) [THIS SCRIPT] Sparse-checkout PR #93's branch into a separate
#       cache directory under $HOME/.cache/kakeya/. The user's K3
#       worktree git state stays clean; the smoke gets DRAFTER_ID
#       pointing at the cache path. Idempotent.
#
# Usage
# -----
#
# Standalone (prints the cache path on stdout, for piping):
#
#   CACHE_PATH="$(bash scripts/research/k3_fetch_alignment_baseline.sh)"
#   echo "$CACHE_PATH"  # e.g. /Users/me/.cache/kakeya/dflash-kakeya-baseline
#
# Auto-invoked by review_pr_k3_feasibility_on_mac.sh when
# DRAFTER_ID == 'models/dflash-kakeya-baseline' AND that path doesn't
# exist locally AND AUTO_FETCH_BASELINE=1 (default).
#
# Override the cache root:
#
#   KAKEYA_BASELINE_CACHE_ROOT=/path/to/scratch \
#       bash scripts/research/k3_fetch_alignment_baseline.sh
#
# Force re-fetch (ignore cache):
#
#   KAKEYA_BASELINE_FORCE_REFETCH=1 bash scripts/research/k3_fetch_alignment_baseline.sh
#
# Exit codes
# ----------
#   0   success; cache populated; cache path printed on stdout
#   1   git not installed
#   2   git-lfs not installed
#   3   network / fetch failure
#   4   LFS pull verification failed (model.safetensors < 100 MB)
#   5   sparse-checkout failed

set -euo pipefail

REPO_URL="https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine.git"
PR93_BRANCH="AgentMemory/v04-pr-k3-dflash-native-integration-2815"
SUBPATH="models/dflash-kakeya-baseline"

# All script output except the final cache path goes to stderr so the
# stdout is purely the cache path (for piping into command substitution).
log() { echo "[k3-fetch-baseline] $*" >&2; }

CACHE_ROOT="${KAKEYA_BASELINE_CACHE_ROOT:-$HOME/.cache/kakeya}"
SCRATCH_DIR="$CACHE_ROOT/dflash-baseline-fetch"
TARGET_DIR="$CACHE_ROOT/dflash-kakeya-baseline"

mkdir -p "$CACHE_ROOT"

# ---------------------------------------------------------------------------
# Idempotency check: cache already populated with a real (non-pointer)
# safetensors file?
# ---------------------------------------------------------------------------
if [[ "${KAKEYA_BASELINE_FORCE_REFETCH:-0}" != "1" ]] \
   && [[ -f "$TARGET_DIR/config.json" ]] \
   && [[ -f "$TARGET_DIR/model.safetensors" ]]; then
    # stat -c%s on Linux, -f%z on macOS; try both
    safetensors_size=$(stat -f%z "$TARGET_DIR/model.safetensors" 2>/dev/null \
                       || stat -c%s "$TARGET_DIR/model.safetensors" 2>/dev/null \
                       || echo 0)
    if [[ "$safetensors_size" -gt 100000000 ]]; then
        log "cache hit: $TARGET_DIR (model.safetensors = $((safetensors_size / 1024 / 1024)) MB); skipping fetch"
        echo "$TARGET_DIR"
        exit 0
    else
        log "cache present but model.safetensors is $safetensors_size bytes (likely an LFS pointer); re-fetching"
    fi
fi

# ---------------------------------------------------------------------------
# Pre-flight: git + git-lfs availability
# ---------------------------------------------------------------------------
if ! command -v git > /dev/null 2>&1; then
    log "ERROR: git not installed."
    exit 1
fi
if ! git lfs version > /dev/null 2>&1; then
    log "ERROR: git-lfs not installed."
    log "       On Mac:  brew install git-lfs && git lfs install"
    log "       On Linux: apt-get install -y git-lfs && git lfs install"
    exit 2
fi

# ---------------------------------------------------------------------------
# Sparse-checkout fetch into scratch dir
# ---------------------------------------------------------------------------
log "fetching $SUBPATH from PR #93 branch into scratch: $SCRATCH_DIR"

if [[ ! -d "$SCRATCH_DIR/.git" ]]; then
    log "scratch is empty; bare clone (no blobs, no checkout)..."
    rm -rf "$SCRATCH_DIR"
    if ! git clone --no-checkout --filter=blob:none \
            "$REPO_URL" "$SCRATCH_DIR" 2>&1 | sed 's/^/[k3-fetch-baseline]   /' >&2; then
        log "ERROR: git clone failed"
        exit 3
    fi
fi

cd "$SCRATCH_DIR"

# Fetch the specific branch (idempotent, safe to re-run)
log "fetching origin/$PR93_BRANCH..."
if ! git fetch origin "$PR93_BRANCH" 2>&1 | sed 's/^/[k3-fetch-baseline]   /' >&2; then
    log "ERROR: git fetch failed"
    exit 3
fi

# Sparse-checkout: only models/dflash-kakeya-baseline/
log "sparse-checkout: $SUBPATH"
if ! git sparse-checkout init --cone 2>&1 | sed 's/^/[k3-fetch-baseline]   /' >&2; then
    log "ERROR: git sparse-checkout init failed"
    exit 5
fi
git sparse-checkout set "$SUBPATH" 2>&1 | sed 's/^/[k3-fetch-baseline]   /' >&2 || true

log "checking out FETCH_HEAD into sparse cone..."
if ! git checkout FETCH_HEAD 2>&1 | sed 's/^/[k3-fetch-baseline]   /' >&2; then
    log "ERROR: git checkout FETCH_HEAD failed"
    exit 5
fi

# Now pull the actual LFS content
log "git lfs pull (859 MB; ~5-10 min on broadband)..."
git lfs install --local > /dev/null 2>&1 || true
if ! git lfs pull --include="$SUBPATH/*" 2>&1 | sed 's/^/[k3-fetch-baseline]   /' >&2; then
    log "ERROR: git lfs pull failed"
    exit 3
fi

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
src="$SCRATCH_DIR/$SUBPATH"
if [[ ! -f "$src/model.safetensors" ]]; then
    log "ERROR: $src/model.safetensors missing after LFS pull"
    exit 4
fi
safetensors_size=$(stat -f%z "$src/model.safetensors" 2>/dev/null \
                   || stat -c%s "$src/model.safetensors" 2>/dev/null \
                   || echo 0)
if [[ "$safetensors_size" -lt 100000000 ]]; then
    log "ERROR: $src/model.safetensors is only $safetensors_size bytes (expected ~859 MB)"
    log "       LFS pull seems to have failed; the file may still be a pointer."
    log "       Manually retry: cd $SCRATCH_DIR && git lfs pull --include=$SUBPATH/*"
    exit 4
fi
log "verified: model.safetensors = $((safetensors_size / 1024 / 1024)) MB"

# ---------------------------------------------------------------------------
# Materialise into final target dir
# ---------------------------------------------------------------------------
# Use a separate target dir (not the scratch sparse checkout) so:
#   - the user can delete the scratch dir to save space without losing the cache
#   - the target dir is a plain directory (no .git/), making it identical
#     to a manually-pulled local checkout from the user's perspective
log "copying scratch → final target: $TARGET_DIR"
mkdir -p "$TARGET_DIR"
cp -r "$src/." "$TARGET_DIR/"

log "OK: $TARGET_DIR ready."
log "    config.json:        $(stat -f%z "$TARGET_DIR/config.json" 2>/dev/null || stat -c%s "$TARGET_DIR/config.json" 2>/dev/null) bytes"
log "    model.safetensors:  $((safetensors_size / 1024 / 1024)) MB"
[[ -f "$TARGET_DIR/manifest.json" ]] \
    && log "    manifest.json:      $(stat -f%z "$TARGET_DIR/manifest.json" 2>/dev/null || stat -c%s "$TARGET_DIR/manifest.json" 2>/dev/null) bytes"
[[ -f "$TARGET_DIR/README.md" ]] \
    && log "    README.md:          $(stat -f%z "$TARGET_DIR/README.md" 2>/dev/null || stat -c%s "$TARGET_DIR/README.md" 2>/dev/null) bytes"
log ""
log "Use this path as DRAFTER_ID:"
log "    DRAFTER_ID=$TARGET_DIR"

# Final stdout: just the path, for command-substitution piping
echo "$TARGET_DIR"
