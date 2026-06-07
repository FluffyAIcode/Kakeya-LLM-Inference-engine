#!/usr/bin/env bash
# vast.ai (CUDA) review aid for PR-R1c (ADR 0011 cross-attention toy,
# Gate G-X1, capacity-bumped + plateau-escape attempt #3).
#
# Background — why R1c:
#   R1b fixed the four toy bugs and the loss curve became interpretable,
#   but recall stayed at 0 % on both pathways. The loss was still
#   descending at the 200-step cap:
#       step  50 -> 2.975
#       step 100 -> 2.363
#       step 150 -> 2.131
#       step 200 -> 2.046   (slope ~ -0.001/step, decelerating)
#   Per-token answer probability was only ~13 %; cross_attn_recall only
#   turns non-zero below ~0.7 loss, which linear extrapolation puts at
#   ~1500-2500 steps. R1c therefore (a) trains 10x longer (2000 steps),
#   (b) bumps bridge capacity (16 heads x 128 dim), (c) seeds W_o with a
#   small non-zero std (0.01) so the bridge is shaped by the loss from
#   step 1, and (d) adds --needle-debug-mode to test the mechanism on an
#   easy low-entropy target before the full task.
#
# This script launches TWO runs IN PARALLEL on the GPU host:
#   A (full)  : the real task, --needle-debug-mode off, 2000 steps.
#   B (small) : easy probe,   --needle-debug-mode small, 2000 steps.
#
# Both write a schema_version=3 JSON report under results/research/.
# Run this ON the vast host (repo synced there) with HF_TOKEN exported.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/review_pr_r1c_on_vast.sh
#
# Env knobs:
#   TRAIN_STEPS  (default 2000)   training steps for both runs
#   WAIT         (default 1)      1 = block until both finish; 0 = launch
#                                 in background, print PIDs/logs, return.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_STEPS="${TRAIN_STEPS:-2000}"
WAIT="${WAIT:-1}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

report_a="${out_dir}/cross_attn_toy_vast_full_${stamp}.json"
report_b="${out_dir}/cross_attn_toy_vast_needle_small_${stamp}.json"
log_a="${log_dir}/r1c_full_${stamp}.log"
log_b="${log_dir}/r1c_needle_small_${stamp}.log"

echo "==> PR-R1c ADR 0011 toy (Gate G-X1 — vast CUDA, attempt #3)"
echo "    Model:        google/gemma-3-1b-it (gated; needs HF_TOKEN)"
echo "    Device:       auto (cuda)"
echo "    Capacity:     16 heads x 128 dim (R1c bump from 8 x 64)"
echo "    o_proj init:  std 0.01 (R1c; R1b used strict 0.0)"
echo "    Steps:        $TRAIN_STEPS"
echo "    Run A (full): --needle-debug-mode off   -> $report_a"
echo "    Run B (small):--needle-debug-mode small -> $report_b"
echo

# Provision the venv ONCE before launching parallel runs so the two
# pip installs don't race each other.
echo "==> provisioning venv (one-time)"
bash scripts/research/run_on_vast.sh --setup-only

common_args=(
    --model google/gemma-3-1b-it
    --cross-attn-depth 20
    --sink 4 --window 64
    --num-heads 16 --head-dim 128
    --train-steps "$TRAIN_STEPS"
    --o-proj-init-std 0.01
    --lr 3e-4
    --n-train 200 --n-eval 50
    --eval-every 100
    --haystack-min-tokens 256 --haystack-max-tokens 1024
)

echo "==> launching Run A (full) and Run B (needle small) in parallel"

bash scripts/research/run_on_vast.sh \
    "${common_args[@]}" \
    --needle-debug-mode off \
    --output "$report_a" \
    >"$log_a" 2>&1 &
pid_a=$!
echo "    Run A pid=$pid_a  log=$log_a"

bash scripts/research/run_on_vast.sh \
    "${common_args[@]}" \
    --needle-debug-mode small \
    --output "$report_b" \
    >"$log_b" 2>&1 &
pid_b=$!
echo "    Run B pid=$pid_b  log=$log_b"
echo

if [[ "$WAIT" != "1" ]]; then
    echo "==> WAIT=0: runs detached. Tail logs with:"
    echo "    tail -f $log_a"
    echo "    tail -f $log_b"
    echo "    PIDs: A=$pid_a B=$pid_b"
    exit 0
fi

echo "==> waiting for both runs (tail the logs above to watch progress)"
rc_a=0; rc_b=0
wait "$pid_a" || rc_a=$?
echo "    Run A finished rc=$rc_a"
wait "$pid_b" || rc_b=$?
echo "    Run B finished rc=$rc_b"
echo
echo "==> Reports:"
echo "    A (full):  $report_a  (exit $rc_a; 0 = all 3 G-X1 predicates pass)"
echo "    B (small): $report_b  (exit $rc_b)"
echo
echo "Commit:"
echo "    git add $report_a $report_b"
echo "    git commit -m 'vast H200 evidence for PR-R1c (ADR 0011 G-X1 attempt #3)'"
echo "    git push"

# Non-zero only if BOTH failed to even run (exit >= 2); a plain gate
# FAIL (exit 1) is an expected scientific outcome, not a script error.
if [[ "$rc_a" -ge 2 && "$rc_b" -ge 2 ]]; then
    exit 1
fi
exit 0
