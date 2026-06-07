#!/usr/bin/env bash
# vast.ai (CUDA) review aid for PR-R1e (ADR 0011 cross-attention toy,
# Gate G-X1, write-path expansion attempt #5).
#
# Background — why R1e:
#   R1d-β (post needle-matcher fix) gave the decisive write-path-bottleneck
#   diagnosis:
#     • Run B (small task, aux=0):   localization=0.25  recall=0.12
#     • Run C (small task, aux=0.1): localization=0.50  recall=0.10
#   Doubling localization (more attention mass on the needle) had ZERO
#   effect on recall — the bottleneck is the WRITE path, not the READ
#   path. Per ADR 0011 §10's decision matrix this is the (I-1) cell:
#   bridge can locate but cannot project. R1e tests three architecturally
#   distinct write-path expansions, all on the small-vocab task with
#   the supervised aux loss left on (we know it doesn't hurt and lifts
#   localization "for free"):
#
#     R1e-α (FFN write path)     : single bridge + SiLU 4× FFN after o_proj
#     R1e-β (multi-layer)        : 3 bridges at depths 8/14/20, no FFN
#     R1e-γ (full block)         : single bridge replaced by full pre-norm
#                                  cross-attn + FFN block (+ 2 LayerNorms)
#
# Each writes a schema_version=5 JSON report. Three runs, sequential by
# default (parallel runs of multi-bridge + capture_attention OOM'd a
# single H200 in R1d's first attempt).
#
# Run this ON the vast host (repo synced there) with HF_TOKEN exported.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/review_pr_r1e_on_vast.sh
#
# Env knobs:
#   TRAIN_STEPS    (default 2000)
#   AUX_WEIGHT     (default 0.1)   retrieval-aux weight, applied to ALL three
#                                  runs (we know it lifts localization "for
#                                  free" and was decisive in R1d).
#   PARALLEL       (default 0)     1 = launch all three in background
#                                  (DANGEROUS — see OOM history above);
#                                  0 = sequential.
#   SKIP_ALPHA     (default 0)
#   SKIP_BETA      (default 0)
#   SKIP_GAMMA     (default 0)     opt-out individual runs to save GPU time

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_STEPS="${TRAIN_STEPS:-2000}"
AUX_WEIGHT="${AUX_WEIGHT:-0.1}"
PARALLEL="${PARALLEL:-0}"
SKIP_ALPHA="${SKIP_ALPHA:-0}"
SKIP_BETA="${SKIP_BETA:-0}"
SKIP_GAMMA="${SKIP_GAMMA:-0}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

aux_tag="${AUX_WEIGHT//./p}"
report_a="${out_dir}/cross_attn_toy_vast_r1e_alpha_ffn_aux${aux_tag}_${stamp}.json"
report_b="${out_dir}/cross_attn_toy_vast_r1e_beta_multilayer_aux${aux_tag}_${stamp}.json"
report_c="${out_dir}/cross_attn_toy_vast_r1e_gamma_block_aux${aux_tag}_${stamp}.json"
log_a="${log_dir}/r1e_alpha_ffn_aux${aux_tag}_${stamp}.log"
log_b="${log_dir}/r1e_beta_multilayer_aux${aux_tag}_${stamp}.log"
log_c="${log_dir}/r1e_gamma_block_aux${aux_tag}_${stamp}.log"

echo "==> PR-R1e ADR 0011 toy (Gate G-X1 — vast CUDA, attempt #5)"
echo "    Model:       google/gemma-3-1b-it (gated; needs HF_TOKEN)"
echo "    Device:      auto (cuda)"
echo "    Steps:       $TRAIN_STEPS"
echo "    Aux weight:  $AUX_WEIGHT (R1d found 0.1 lifts localization 0.25 -> 0.50)"
echo "    Task:       --needle-debug-mode small (the R1c/R1d discriminator)"
echo "    Sequential by default (PARALLEL=$PARALLEL)"
[[ "$SKIP_ALPHA" == "1" ]] && echo "    Run alpha:   SKIPPED" || \
    echo "    Run alpha:   FFN write path (single bridge + 4x SiLU FFN)  -> $report_a"
[[ "$SKIP_BETA"  == "1" ]] && echo "    Run beta:    SKIPPED" || \
    echo "    Run beta:    multi-layer (3 bridges @ depths 8,14,20)      -> $report_b"
[[ "$SKIP_GAMMA" == "1" ]] && echo "    Run gamma:   SKIPPED" || \
    echo "    Run gamma:   full block (cross-attn + LN + FFN + LN)        -> $report_c"
echo

# Provision the venv ONCE before launching runs.
echo "==> provisioning venv (one-time)"
bash scripts/research/run_on_vast.sh --setup-only

base_args=(
    --model google/gemma-3-1b-it
    --sink 4 --window 64
    --num-heads 16 --head-dim 128
    --train-steps "$TRAIN_STEPS"
    --o-proj-init-std 0.01
    --retrieval-aux-weight "$AUX_WEIGHT"
    --needle-debug-mode small
    --lr 3e-4
    --n-train 200 --n-eval 50
    --eval-every 100
    --haystack-min-tokens 256 --haystack-max-tokens 1024
)

# R1e-α: FFN write path on a single bridge at depth 20.
alpha_args=(
    "${base_args[@]}"
    --cross-attn-depth 20
    --bridge-use-ffn-write-path
    --ffn-expansion 4
    --output "$report_a"
)
# R1e-β: 3 independent bridges at depths 8/14/20, no FFN.
beta_args=(
    "${base_args[@]}"
    --cross-attn-depths "8,14,20"
    --output "$report_b"
)
# R1e-γ: full pre-norm transformer block at depth 20.
gamma_args=(
    "${base_args[@]}"
    --cross-attn-depth 20
    --bridge-use-block-architecture
    --ffn-expansion 4
    --output "$report_c"
)

run_one() {
    # In each mode, ONE line goes to stdout (captured by $(...) by the
    # caller — the pid in PARALLEL=1 mode, the rc in sequential mode).
    # Everything else is human-readable info on stderr.
    local label="$1"; local logfile="$2"; shift 2
    local rc=0
    if [[ "$PARALLEL" == "1" ]]; then
        bash scripts/research/run_on_vast.sh "$@" >"$logfile" 2>&1 &
        local pid=$!
        echo "    $label pid=$pid  log=$logfile" >&2
        echo "$pid"
    else
        echo "==> launching $label (sequential, log=$logfile)" >&2
        bash scripts/research/run_on_vast.sh "$@" >"$logfile" 2>&1 || rc=$?
        echo "    $label finished rc=$rc" >&2
        echo "$rc"
    fi
}

if [[ "$PARALLEL" == "1" ]]; then
    pids=()
    labels=()
    [[ "$SKIP_ALPHA" != "1" ]] && { pid=$(run_one "alpha" "$log_a" "${alpha_args[@]}"); pids+=("$pid"); labels+=("alpha"); }
    [[ "$SKIP_BETA"  != "1" ]] && { pid=$(run_one "beta"  "$log_b" "${beta_args[@]}");  pids+=("$pid"); labels+=("beta");  }
    [[ "$SKIP_GAMMA" != "1" ]] && { pid=$(run_one "gamma" "$log_c" "${gamma_args[@]}"); pids+=("$pid"); labels+=("gamma"); }
    echo
    echo "==> all three running in parallel; tail logs to watch progress"
    rcs=()
    for i in "${!pids[@]}"; do
        rc=0; wait "${pids[$i]}" || rc=$?; rcs+=("$rc")
        echo "    ${labels[$i]} finished rc=$rc"
    done
else
    rcs=()
    [[ "$SKIP_ALPHA" != "1" ]] && rcs+=("$(run_one "alpha" "$log_a" "${alpha_args[@]}")")
    [[ "$SKIP_BETA"  != "1" ]] && rcs+=("$(run_one "beta"  "$log_b" "${beta_args[@]}")")
    [[ "$SKIP_GAMMA" != "1" ]] && rcs+=("$(run_one "gamma" "$log_c" "${gamma_args[@]}")")
fi

echo
echo "==> Reports:"
[[ "$SKIP_ALPHA" != "1" ]] && echo "    alpha (FFN write path):     $report_a"
[[ "$SKIP_BETA"  != "1" ]] && echo "    beta  (multi-layer):        $report_b"
[[ "$SKIP_GAMMA" != "1" ]] && echo "    gamma (full block):         $report_c"
echo
echo "Commit:"
echo "    git add $out_dir/cross_attn_toy_vast_r1e_*${stamp}.json $log_dir/r1e_*${stamp}.log"
echo "    git commit -m 'vast H200 evidence for PR-R1e (ADR 0011 G-X1 attempt #5)'"
echo "    git push"

# Non-zero exit only if EVERY run hard-failed (rc >= 2). A plain gate
# FAIL (rc 1) is an expected scientific outcome for this PR.
all_hard_fail=1
for rc in "${rcs[@]}"; do
    if [[ "$rc" -lt 2 ]]; then all_hard_fail=0; fi
done
if [[ "$all_hard_fail" == "1" ]]; then exit 1; fi
exit 0
