#!/usr/bin/env bash
# vast.ai (CUDA) review aid for PR-R1d-β (ADR 0011 cross-attention toy,
# Gate G-X1, retrieval-auxiliary loss + attention-localization
# instrumentation, attempt #4).
#
# Background — why R1d-β:
#   R1c lifted the four hyperparameter caps R1b had flagged (2000 steps,
#   16x128 bridge, W_o init 0.01, --needle-debug-mode). Result on vast
#   H200:
#       Run A (full,  135k vocab): cross_attn=0.000  (FAIL)
#       Run B (small,  20 vocab):  cross_attn=0.16   (FAIL but signal:
#                                                     3.2x over random)
#   Conclusion: the cross-attention bridge mechanism partially works, but
#   the single-layer-at-depth-20 architecture caps far below G-X1's 80%.
#   R1d-β does NOT add capacity; it adds a SUPERVISED RETRIEVAL signal
#   (retrieval-aux loss pushes the bridge's attention probability mass
#   onto the known needle token range during training) plus a diagnostic
#   metric (attention_localization_rate: how often does the bridge's
#   argmax over the proposer bank actually land in the needle area?).
#
#   This single GPU run produces TWO numbers that disambiguate "the
#   mechanism is fundamentally limited" from "it just needs more
#   capacity": (recall, localization_rate) per the decision matrix in
#   PR-R1c's description (#65) and ADR 0011 §10.
#
# This script launches THREE runs IN PARALLEL on the GPU host:
#   A : full task,    aux=0.0   (control: reproduces PR-R1c Run A)
#   B : small task,   aux=0.0   (control: reproduces PR-R1c Run B)
#   C : small task,   aux=0.1   (THE EXPERIMENT)
#
# Three runs (not two) because: the aux-loss mechanism only works if the
# bridge can locate AND project; if Run C jumps recall on small-vocab
# but Run A still 0% on full, that's diagnostic. If Run C also fails,
# the mechanism is genuinely capped.
#
# Each writes a schema_version=4 JSON report under results/research/.
# Run this ON the vast host (repo synced there) with HF_TOKEN exported.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/review_pr_r1d_on_vast.sh
#
# Env knobs:
#   TRAIN_STEPS         (default 2000)  steps for all three runs
#   AUX_WEIGHT          (default 0.1)   retrieval-aux weight for Run C
#   WAIT                (default 1)     1 = block; 0 = detach + return.
#   SKIP_FULL_CONTROL   (default 0)     1 = skip Run A (the control on
#                                       full vocab) to save GPU minutes
#                                       when only the diagnostic matters.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_STEPS="${TRAIN_STEPS:-2000}"
AUX_WEIGHT="${AUX_WEIGHT:-0.1}"
WAIT="${WAIT:-1}"
SKIP_FULL_CONTROL="${SKIP_FULL_CONTROL:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

report_a="${out_dir}/cross_attn_toy_vast_r1d_full_aux0_${stamp}.json"
report_b="${out_dir}/cross_attn_toy_vast_r1d_small_aux0_${stamp}.json"
report_c="${out_dir}/cross_attn_toy_vast_r1d_small_aux${AUX_WEIGHT//./p}_${stamp}.json"
log_a="${log_dir}/r1d_full_aux0_${stamp}.log"
log_b="${log_dir}/r1d_small_aux0_${stamp}.log"
log_c="${log_dir}/r1d_small_aux${AUX_WEIGHT//./p}_${stamp}.log"

echo "==> PR-R1d-beta ADR 0011 toy (Gate G-X1 — vast CUDA, attempt #4)"
echo "    Model:        google/gemma-3-1b-it (gated; needs HF_TOKEN)"
echo "    Device:       auto (cuda)"
echo "    Capacity:     16 heads x 128 dim (preserved from R1c)"
echo "    o_proj init:  std 0.01 (preserved from R1c)"
echo "    Steps:        $TRAIN_STEPS"
echo "    Aux weight:   $AUX_WEIGHT (only Run C; A and B are aux=0 controls)"
if [[ "$SKIP_FULL_CONTROL" == "1" ]]; then
echo "    Run A:        SKIPPED (SKIP_FULL_CONTROL=1)"
else
echo "    Run A:        full   --aux 0.0 -> $report_a"
fi
echo "    Run B:        small  --aux 0.0 -> $report_b"
echo "    Run C:        small  --aux $AUX_WEIGHT -> $report_c"
echo

# Provision the venv ONCE before launching parallel runs so the pip
# installs don't race each other.
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

pids=()
labels=()

if [[ "$SKIP_FULL_CONTROL" != "1" ]]; then
    bash scripts/research/run_on_vast.sh \
        "${common_args[@]}" \
        --needle-debug-mode off \
        --retrieval-aux-weight 0.0 \
        --output "$report_a" \
        >"$log_a" 2>&1 &
    pids+=($!); labels+=("A(full,aux=0)")
    echo "    Run A pid=${pids[-1]}  log=$log_a"
fi

bash scripts/research/run_on_vast.sh \
    "${common_args[@]}" \
    --needle-debug-mode small \
    --retrieval-aux-weight 0.0 \
    --output "$report_b" \
    >"$log_b" 2>&1 &
pids+=($!); labels+=("B(small,aux=0)")
echo "    Run B pid=${pids[-1]}  log=$log_b"

bash scripts/research/run_on_vast.sh \
    "${common_args[@]}" \
    --needle-debug-mode small \
    --retrieval-aux-weight "$AUX_WEIGHT" \
    --output "$report_c" \
    >"$log_c" 2>&1 &
pids+=($!); labels+=("C(small,aux=$AUX_WEIGHT)")
echo "    Run C pid=${pids[-1]}  log=$log_c"
echo

if [[ "$WAIT" != "1" ]]; then
    echo "==> WAIT=0: runs detached. PIDs:"
    for i in "${!pids[@]}"; do
        echo "    ${labels[$i]} pid=${pids[$i]}"
    done
    exit 0
fi

echo "==> waiting for runs (tail the logs above to watch progress)"
rcs=()
for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    rcs+=("$rc")
    echo "    ${labels[$i]} finished rc=$rc"
done
echo
echo "==> Reports:"
if [[ "$SKIP_FULL_CONTROL" != "1" ]]; then
    echo "    A: $report_a"
fi
echo "    B: $report_b"
echo "    C: $report_c"
echo
echo "Commit:"
echo "    git add $out_dir/cross_attn_toy_vast_r1d_*${stamp}.json $log_dir/r1d_*${stamp}.log"
echo "    git commit -m 'vast H200 evidence for PR-R1d-beta (ADR 0011 G-X1 attempt #4)'"
echo "    git push"

# Non-zero only if every run failed to even start (exit >= 2). A plain
# gate FAIL (exit 1) is an expected scientific outcome.
all_hard_fail=1
for rc in "${rcs[@]}"; do
    if [[ "$rc" -lt 2 ]]; then all_hard_fail=0; fi
done
if [[ "$all_hard_fail" == "1" ]]; then exit 1; fi
exit 0
