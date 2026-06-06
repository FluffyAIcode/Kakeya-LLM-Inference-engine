#!/usr/bin/env bash
# Mac M4 review aid for PR-R1 / R1b (ADR 0011 + cross-attention toy
# prototype).
#
# This is a research-track script. Linux CI covers ADR markdown +
# the cross-attention forward / mask / hook unit tests. The
# empirical question — does cross-attention coupling actually let a
# bounded verifier recover long-context recall? — is what this Mac
# aid measures.
#
# Phase 1 (G-X1) acceptance per ADR 0011 §4 (R1b: 3 predicates):
#   oracle (full-attn, no bridge)            ≥ 0.80    ← sanity
#   bounded baseline (sink+window, no bridge) ≤ 0.30   ← bound is real
#   cross-attn (sink+window + bridge)         ≥ 0.80   ← hypothesis
#
# Time budget on Mac M4 24 GB with Gemma 3-1B-it: ~45-75 minutes
# (R1b: slightly longer than R1 because eval now has 3 baselines per
# sample × 24 max_new_tokens). Larger models (Gemma 3-2B) 1.5-3
# hours.
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

echo "==> ADR 0011 toy prototype (Gate G-X1 — Mac M4, R1b)"
echo "    Model:        google/gemma-3-1b-it (26 layers, 1152 hidden)"
echo "    Device:       mps (auto)"
echo "    Cross-attn:   forward-hook on layer 20 output (R1b architectural fix)"
echo "    Mask:         4D sink+window via attention_mask dict (R1b Bug B fix)"
echo "    Prompts:      tokenizer.apply_chat_template (R1b Bug C fix)"
echo "    Eval:         oracle + bounded + cross_attn (R1b Bug D fix)"
echo

PYTHONPATH=.:sdks/python python3 scripts/research/cross_attn_toy_prototype.py \
    --model google/gemma-3-1b-it \
    --device auto \
    --cross-attn-depth 20 \
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
echo "    exit code: $exit_code (0 = all 3 G-X1 predicates pass)"
echo
echo "Commit:"
echo "    git add $report"
echo "    git commit -m 'Mac M4 evidence for PR-R1b (ADR 0011 G-X1 toy prototype, attempt #2)'"
echo "    git push"

exit $exit_code
