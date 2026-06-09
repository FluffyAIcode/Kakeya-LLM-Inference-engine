#!/usr/bin/env bash
# Mac M4 K2.A production-shape smoke (per user directive 2026-06-09).
#
# This script answers the **product-aligned** question: under a
# realistic single-user single-request shape, does the K2.A KL ON +
# K2.A.2 stateful path deliver acceptable behaviour on Mac M4?
#
# Specifically:
#
#   * Recall hits the needle once per request (not statistical
#     averaging over 20 samples).
#   * Per-step dtype crash (the bf16 / fp32 round-trip failure
#     fixed by PR #87) does NOT recur on MPS bf16.
#   * First-token latency is recorded so we can see what the user
#     actually waits for, not just a research mean throughput.
#
# What this script does NOT validate (deliberately):
#
#   * Statistical recall delta over a sample distribution. That's
#     a research A/B question, answered by
#     scripts/review_pr_k2a1_integration_on_mac.sh (~7-9h, BINDING
#     gate evidence for PR-K2.A.1 merge).
#   * Throughput parity vs full-attention oracle (no oracle arm
#     run here — gate (c) work that K2.A.2 + K3 close).
#   * MLX/Metal native path latency. v0.4 today is PyTorch MPS
#     research harness; the user-facing latency floor will only
#     drop when K3 ships the MLX path. This smoke nonetheless
#     records the PyTorch MPS number as a CONSERVATIVE upper bound
#     on what users experience: anything K3 ships will be faster.
#
# Usage
# -----
#
# Default — single 5.6k-context request, KL ON + stateful, ~3-5 min
# on Mac M4 24 GB:
#
#   bash scripts/review_pr_k2a_production_smoke_on_mac.sh
#
# Single 1.4k request (~30-60 s):
#
#   CTX_LINES=70 bash scripts/review_pr_k2a_production_smoke_on_mac.sh
#
# Wider context (slower, may OOM on 24 GB):
#
#   CTX_LINES=800 bash scripts/review_pr_k2a_production_smoke_on_mac.sh
#
# Env knobs (defaults):
#
#   CTX_LINES         (280)       padding-line count; 70=1.4k, 280=5.6k,
#                                 800=16k tokens
#   SINK              (4)
#   WINDOW            (64)
#   MAX_NEW_TOKENS    (24)
#   SEED              (42)
#   KL_LATTICE        (D4)        D4 (v1.4) or E8 (v1.5)
#   KL_Q_RANGE        (38)        canonical D4 operating point
#   ATTN_IMPL         (sdpa)      memory-efficient; required for >=4k Mac M4
#   STATEFUL          (1)         1 = K2.A.2 (default; production path);
#                                 0 = K2.A.1 stateless (each-step recompute,
#                                     research-only — DO NOT ship to users)
#
# Output
# ------
#
# Single JSON report at:
#
#   results/research/k2a_production_smoke_mac_ctx<CTX_LINES>_<stamp>.json
#
# JSON contains: recall (hit/miss for the single request), prefill
# latency, first-token latency, mean tok/s, peak resident memory,
# effective attention window, dtype/numerics warnings if any.
#
# Pre-flight
# ----------
#
# kakeyalattice must be installed (for KL ON arm). PyTorch with MPS
# support must be importable. HF_TOKEN must be exported (Gemma 3-1B
# is gated):
#
#   export HF_TOKEN=hf_xxx
#   pip install kakeyalattice
#
# Lineage (why this script exists alongside the A/B script)
# ---------------------------------------------------------
#
# The A/B reviewer scripts/review_pr_k2a1_integration_on_mac.sh is a
# **research evidence collector** for PR-K2.A.1 merging — it takes
# 7-9h, runs 20 samples × 2 context rungs × {KL OFF, KL ON} × {oracle,
# v0.3, v0.4} arms because the BINDING gate for that PR (recall delta
# ≤ 1pp at every rung) needs statistical signal.
#
# That A/B shape is wrong for the **product** question — users send
# one request, expect one answer, and care about the latency of that
# one request. Per the user's 2026-06-09 directive ("严重浪费时间..."):
# do not run oracle / v0.3 / A/B for product-experience verification;
# run a single production arm (KL ON + stateful), single request,
# report user-facing latency.
#
# Both scripts have a place — research signal vs product signal.
# This file is the product-signal entry point.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CTX_LINES="${CTX_LINES:-280}"
SINK="${SINK:-4}"
WINDOW="${WINDOW:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
KL_LATTICE="${KL_LATTICE:-D4}"
KL_Q_RANGE="${KL_Q_RANGE:-38}"
STATEFUL="${STATEFUL:-1}"

stamp="$(date +%s)"
out_dir="results/research"
log_dir="${out_dir}/logs"
mkdir -p "$out_dir" "$log_dir"

# ±15 % range around the target line count — same formula as A/B
# script so the haystack length distribution matches research evidence.
lo=$(( (CTX_LINES * 85 + 50) / 100 ))
hi=$(( (CTX_LINES * 115 + 50) / 100 ))
if [[ $lo -lt 10 ]]; then lo=10; fi
if [[ $hi -lt $((lo + 1)) ]]; then hi=$((lo + 1)); fi

report="${out_dir}/k2a_production_smoke_mac_ctx${CTX_LINES}_${stamp}.json"
log="${log_dir}/k2a_production_smoke_mac_ctx${CTX_LINES}_${stamp}.log"

echo "==> Mac M4 K2.A production-shape smoke (single request)"
echo "    Model:           google/gemma-3-1b-it"
echo "    Context:         ~$(( CTX_LINES * 20 )) tokens (lines [$lo, $hi])"
echo "    Sink x window:   ${SINK} x ${WINDOW}"
echo "    Attn impl:       $ATTN_IMPL"
echo "    Lattice:         $KL_LATTICE  (Q=$KL_Q_RANGE)"
echo "    Stateful (K2.A.2): $STATEFUL"
echo "    Samples:         1   (product shape, NOT 20)"
echo "    Arms:            v0.4 KL ON only (NO oracle, NO v0.3, NO KL OFF)"
echo "    Time budget:     ~30 s @ 1.4k, ~3-5 min @ 5.6k, ~12-18 min @ 16k"
echo "    Report:          $report"
echo

echo "==> Pre-flight checks"

# (1) kakeyalattice availability
if PYTHONPATH=.:sdks/python python3 -c "import kakeyalattice" 2>/dev/null; then
    echo "    [OK]  kakeyalattice importable"
else
    echo "    [FAIL] kakeyalattice NOT installed. Install:"
    echo "           pip install kakeyalattice"
    exit 1
fi

# (2) K1.E runner supports --stateful (added in K2.A.2 / PR #90).
#     This script's whole product premise is 'KL ON + K2.A.2 stateful',
#     so the runner MUST recognise --stateful. If not, this branch is
#     based on a pre-K2.A.2 main and the smoke would silently fall
#     back to stateless (or, worse, error halfway). We refuse to start.
runner="scripts/research/k1e_niah_validation.py"
if grep -q -- '"--stateful"' "$runner"; then
    echo "    [OK]  k1e_niah_validation.py exposes --stateful (K2.A.2 path available)"
else
    echo "    [FAIL] $runner does NOT expose --stateful."
    echo
    echo "    Root cause: this branch is based on a main that does NOT yet"
    echo "    include the K2.A.2 stateful caching commit (PR #90). The"
    echo "    production-shape smoke is meaningless without --stateful;"
    echo "    aborting rather than silently producing a stateless-only result."
    echo
    echo "    Fix: pull the latest Mac reviewer branch (it's stacked on the"
    echo "    K3 stack tip which includes K2.A.2):"
    echo
    echo "        git fetch origin AgentMemory/mac-k2a1-reviewer-production-shape-8e7f"
    echo "        git checkout AgentMemory/mac-k2a1-reviewer-production-shape-8e7f"
    echo "        git pull --ff-only   # or: git reset --hard origin/<branch>"
    echo
    echo "    Then re-run this script."
    exit 2
fi

# (3) Quick parser-level smoke: verify the runner accepts our flag set
#     without trying to load the model. Bail early if any flag is
#     unknown — that gives a clean error message instead of failing
#     mid-run after the model has loaded.
if PYTHONPATH=.:sdks/python python3 -c "
import sys, importlib.util
spec = importlib.util.spec_from_file_location('k1e', '$runner')
mod = importlib.util.module_from_spec(spec)
sys.argv = ['k1e_niah_validation.py',
            '--stateful', '--kl-on', '--skip-oracle', '--skip-v03',
            '--n-samples', '1', '--haystack-min-lines', '10',
            '--haystack-max-lines', '11', '--max-new-tokens', '1',
            '--output', '/tmp/_k1e_parser_check.json']
import argparse
spec.loader.exec_module(mod)
mod.parse_args()
print('parser-OK')
" 2>&1 | tail -1 | grep -q "parser-OK"; then
    echo "    [OK]  k1e_niah_validation.py parser accepts production-smoke flag set"
else
    echo "    [FAIL] $runner parser rejected one of:"
    echo "           --stateful --kl-on --skip-oracle --skip-v03 --n-samples 1"
    echo "    Re-run with the diagnostic command:"
    echo "        PYTHONPATH=.:sdks/python python3 $runner --help"
    exit 3
fi
echo

# Build the K1.E runner invocation. Production-shape constraints:
#
#   --n-samples 1        single request, not 20
#   --skip-oracle        no oracle arm (research-only baseline)
#   --skip-v03           no v0.3 arm (research-only baseline)
#   --kl-on              production arm uses KL compression
#   (no --kl-on toggle for KL OFF)
#   --stateful           K2.A.2 incremental decode path (production
#                        path; without this each step recompresses
#                        the whole resident window, which is what
#                        the user's directive specifically forbids)
#
flags=(
    --model google/gemma-3-1b-it
    --device auto
    --attn-impl "$ATTN_IMPL"
    --n-samples 1
    --haystack-min-lines "$lo"
    --haystack-max-lines "$hi"
    --sink-size "$SINK"
    --window-size "$WINDOW"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED"
    --skip-oracle
    --skip-v03
    --kl-on
    --kl-lattice "$KL_LATTICE"
    --kl-q-range "$KL_Q_RANGE"
    --output "$report"
)
[[ "$STATEFUL" == "1" ]] && flags+=(--stateful)

echo "==> Running v0.4 KL ON $([[ "$STATEFUL" == "1" ]] && echo "+ stateful (K2.A.2)") on a single request"
PYTHONPATH=.:sdks/python python3 scripts/research/k1e_niah_validation.py \
    "${flags[@]}" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}

echo
if [[ "$exit_code" -eq 0 ]]; then
    echo "==> Production smoke OK."
    echo "    Report:  $report"
    echo "    Log:     $log"
    echo
    echo "Inspect first-token latency / recall:"
    echo "    python3 -c 'import json,sys;r=json.load(open(\"$report\"));"
    echo "        v=r[\"per_config\"].get(\"v0.4\");"
    echo "        print(\"recall:\", v[\"recall\"]);"
    echo "        print(\"first_token_s:\", v.get(\"first_token_seconds\"));"
    echo "        print(\"mean_tok_s:\", v.get(\"mean_throughput_tokens_per_sec\"));"
    echo "        print(\"peak_mem_GB:\", v.get(\"peak_memory_bytes\",0)/1e9)'"
    echo
    echo "Commit (only if recall=hit AND no dtype crash):"
    echo "    git add $report $log"
    echo "    git commit -m 'Mac M4 K2.A production-shape smoke evidence'"
    echo "    git push"
else
    echo "==> Production smoke FAILED (exit=$exit_code)."
    echo "    Report (partial): $report"
    echo "    Log:              $log"
fi

exit "$exit_code"
