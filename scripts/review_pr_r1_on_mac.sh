#!/usr/bin/env bash
# Mac M4 review aid for PR-R1 (ADR 0011 + cross-attention toy prototype).
#
# This is a research-track PR. Linux CI gate covers ADR markdown +
# script syntax. The empirical question — does cross-attention coupling
# actually let a bounded verifier recover long-context recall? — is
# what this Mac aid measures.
#
# Phase 1 (G-X1) acceptance per ADR 0011 §4:
#   bounded baseline recall ~ 20 %
#   cross-attn recall >= 80 %  ← target
#
# Time budget on Mac M4 24 GB with Gemma 3-1B-it: ~30-60 minutes.
# Larger models (Gemma 3-2B) take 1-3 hours.
#
# Usage:
#     bash scripts/review_pr_r1_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/research"
mkdir -p "$out_dir"
report="$out_dir/cross_attn_toy_${stamp}.json"

echo "==> ADR 0011 toy prototype (Gate G-X1 — Mac M4)"
echo "    Model: google/gemma-3-1b-it (smallest viable; substitute as needed)"
echo "    Device: mps (auto)"
echo

PYTHONPATH=.:sdks/python python3 scripts/research/cross_attn_toy_prototype.py \
    --model google/gemma-3-1b-it \
    --device auto \
    --cross-attn-depth 8 \
    --sink 4 --window 64 \
    --num-heads 8 --head-dim 64 \
    --train-steps 200 \
    --lr 3e-4 \
    --n-train 200 \
    --n-eval 50 \
    --eval-every 50 \
    --haystack-min-tokens 256 \
    --haystack-max-tokens 1024 \
    --output "$report"
exit_code=$?

echo
echo "==> Done. Report: $report"
echo "    exit code: $exit_code (0 = G-X1 PASS)"
echo
echo "Commit:"
echo "    git add $report"
echo "    git commit -m 'Mac M4 evidence for PR-R1 (ADR 0011 G-X1 toy prototype)'"
echo "    git push"

exit $exit_code
