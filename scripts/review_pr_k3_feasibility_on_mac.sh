#!/usr/bin/env bash
# Mac M4 reviewer aid for K3 hardware feasibility.
#
# K3 production scale per ADR 0008 §11.7 corrected:
#   verifier:  google/gemma-4-26B-A4B-it      (26B A4B MoE, 4B active)
#   drafter:   models/dflash-kakeya-baseline  (alignment-trained DFlash
#                                              drafter; 859 MB bf16, Git
#                                              LFS, commit 19a2d5c — the
#                                              new authoritative baseline
#                                              for all Kakeya inference
#                                              tests/dev as of 2026-06-09).
#                                              Override with the upstream
#                                              HF id z-lab/gemma-4-26B-
#                                              A4B-it-DFlash via
#                                              DRAFTER_ID env var ONLY for
#                                              research-baseline comparison
#                                              (NOT alignment-trained).
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
#   DRAFTER_ID       (models/dflash-kakeya-baseline)
#                     Alignment-trained Kakeya inference baseline
#                     (default; LFS, 859 MB bf16, on PR #93 branch).
#                     If missing locally AND AUTO_FETCH_BASELINE=1
#                     (default), automatically fetched into local
#                     cache via scripts/research/k3_fetch_alignment_
#                     baseline.sh. Override with
#                     'z-lab/gemma-4-26B-A4B-it-DFlash' for
#                     research-baseline comparison only (NOT
#                     alignment-trained).
#   AUTO_FETCH_BASELINE  (1)  if DRAFTER_ID defaults to the
#                             alignment baseline AND it's missing
#                             locally, sparse-checkout PR #93's
#                             branch into $HOME/.cache/kakeya/ on
#                             demand (one-time, ~5-10 min for the
#                             859 MB LFS download). Set to 0 to
#                             disable and require a pre-existing
#                             local checkout.
#   PROMPT_TOKENS    (512)
#   GEN_TOKENS       (8)
#   SEED             (42)
#   SKIP_DRAFTER=1            verifier-only smoke
#   SKIP_VERIFIER=1           drafter-only smoke (useful while
#                             upstream mlx_lm Gemma 4 MoE compat
#                             bug blocks the verifier path)
#   PROPOSER_KV_CAPTURE=1     after drafter loads, exercise
#                             drafter.propose_kv() — the v0.4
#                             dLM K/V Restoration proposer-role
#                             primitive (per ADR §11.5)
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
DRAFTER_ID="${DRAFTER_ID:-models/dflash-kakeya-baseline}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-8}"
SEED="${SEED:-42}"
SKIP_DRAFTER="${SKIP_DRAFTER:-0}"
SKIP_VERIFIER="${SKIP_VERIFIER:-0}"
PROPOSER_KV_CAPTURE="${PROPOSER_KV_CAPTURE:-0}"
ALLOW_MISSING_QUANTIZE="${ALLOW_MISSING_QUANTIZE:-0}"
AUTO_FETCH_BASELINE="${AUTO_FETCH_BASELINE:-1}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"
report="${out_dir}/k3_feasibility_smoke_mac_${stamp}.json"
log="${log_dir}/k3_feasibility_smoke_mac_${stamp}.log"

echo "==> K3 hardware feasibility smoke (Mac M4)"
echo "    Verifier (4-bit MLX): $VERIFIER_PATH"
echo "    Drafter:              $DRAFTER_ID"
echo "    Prompt:               $PROMPT_TOKENS tokens"
echo "    Gen:                  $GEN_TOKENS tokens"
echo "    Skip drafter:         $SKIP_DRAFTER"
echo "    Skip verifier:        $SKIP_VERIFIER"
echo "    Proposer KV capture:  $PROPOSER_KV_CAPTURE"
echo "    Report:               $report"
echo

# Pre-flight 0a: auto-fetch the alignment-trained baseline from PR #93's
# branch into a local cache when the default DRAFTER_ID is missing locally.
#
# The alignment-trained baseline (models/dflash-kakeya-baseline, 859 MB
# bf16, Git LFS, commit 19a2d5c) lives on PR #93's branch
# AgentMemory/v04-pr-k3-dflash-native-integration-2815. The K3 Mac
# smoke / DFlashDrafter API (PRs #95-#98) is on a SEPARATE stack rooted
# at PR #92, and the baseline is NOT in that lineage. So a user on the
# K3 stack worktree doesn't have the LFS-pulled baseline available
# under 'models/dflash-kakeya-baseline'.
#
# This pre-flight detects the missing-default case and auto-invokes
# scripts/research/k3_fetch_alignment_baseline.sh which sparse-checkouts
# PR #93's branch into $HOME/.cache/kakeya/ — without polluting the
# user's git state. DRAFTER_ID is then re-pointed at the cache path.
#
# Disable: AUTO_FETCH_BASELINE=0 bash $0
if [[ "$SKIP_DRAFTER" != "1" ]] \
   && [[ "$DRAFTER_ID" == "models/dflash-kakeya-baseline" ]] \
   && [[ ! -d "$DRAFTER_ID" ]] \
   && [[ "$AUTO_FETCH_BASELINE" == "1" ]]; then
    echo "==> Pre-flight: alignment-trained baseline not found at '$DRAFTER_ID' locally."
    echo "    Auto-fetching from PR #93 branch into local cache..."
    echo "    (set AUTO_FETCH_BASELINE=0 to disable; one-time ~5-10 min for 859 MB LFS download.)"
    echo
    if cached_path="$(bash scripts/research/k3_fetch_alignment_baseline.sh)"; then
        DRAFTER_ID="$cached_path"
        echo "==> Auto-fetch complete. DRAFTER_ID re-pointed to cache: $DRAFTER_ID"
        echo
    else
        rc=$?
        echo "ERROR: auto-fetch of baseline failed (exit=$rc)."
        echo "       See above for the fetch script's diagnostic output."
        echo "       To bypass auto-fetch: AUTO_FETCH_BASELINE=0 bash $0"
        echo "       To manually fetch:   bash scripts/research/k3_fetch_alignment_baseline.sh"
        exit "$rc"
    fi
fi

# Pre-flight 0b: drafter source check (skipped when SKIP_DRAFTER=1).
#
# When DRAFTER_ID is a local path (starts with 'models/', './', '../', '/'),
# verify the directory exists with config.json + safetensors before
# invoking Python — a missing local path otherwise silently falls through
# to HF Hub fetch, which 404s with a misleading error message far from
# the actual root cause (commonly a missing 'git lfs pull' or wrong cwd).
if [[ "$SKIP_DRAFTER" != "1" ]]; then
    case "$DRAFTER_ID" in
        models/*|./*|../*|/*)
            if [[ ! -d "$DRAFTER_ID" ]]; then
                echo "ERROR: DRAFTER_ID='$DRAFTER_ID' looks like a local path but does not exist."
                echo
                echo "Common causes:"
                echo "  1. The Git LFS pointer for the model has not been pulled."
                echo "     Run from the repo root:"
                echo "         git lfs install"
                echo "         git lfs pull"
                echo
                echo "  2. The current working directory is not the repo root."
                echo "         pwd        # should be the repo root"
                echo "         ls models/ # should list dflash-kakeya-baseline"
                echo
                echo "  3. You are on a worktree without the model checkpoint."
                echo "         git status   # confirm worktree"
                echo
                echo "If you intended a HuggingFace repo id instead, override:"
                echo "    DRAFTER_ID=z-lab/gemma-4-26B-A4B-it-DFlash bash $0"
                echo "    (note: that variant is NOT alignment-trained — research only)"
                exit 1
            fi
            if [[ ! -f "$DRAFTER_ID/config.json" ]]; then
                echo "ERROR: DRAFTER_ID='$DRAFTER_ID' is a directory but lacks config.json."
                echo "       The model checkpoint at this path is incomplete or corrupted."
                ls -la "$DRAFTER_ID" 2>&1 | head -20
                exit 1
            fi
            ;;
    esac
fi

# Pre-flight 1: quantized verifier exists? (skipped when SKIP_VERIFIER=1)
if [[ "$SKIP_VERIFIER" == "1" ]]; then
    echo "[pre-flight] SKIP_VERIFIER=1 set; bypassing verifier-dir check."
elif [[ ! -d "$VERIFIER_PATH" ]]; then
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
    --drafter-id "$DRAFTER_ID"
    --prompt-tokens "$PROMPT_TOKENS"
    --gen-tokens "$GEN_TOKENS"
    --seed "$SEED"
    --output "$report"
)
[[ "$SKIP_DRAFTER" == "1" ]] && flags+=(--skip-drafter)
[[ "$SKIP_VERIFIER" == "1" ]] && flags+=(--skip-verifier)
[[ "$PROPOSER_KV_CAPTURE" == "1" ]] && flags+=(--proposer-kv-capture)

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
