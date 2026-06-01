#!/usr/bin/env bash
# Mac M4 review aid for PR-E1b (ADR 0008 §6.5 gRPC long-session
# bench: bench_session_long_run.py).
#
# This script runs a 30-minute SMOKE invocation of the bench against
# a locally-started gRPC server, then prints instructions for the
# 4-hour run. The 30-min smoke is what reviewers commit; the 4h run
# is the GA evidence and is run separately when you have wall-clock
# budget for it.
#
# Both runs validate the two ADR 0008 §7 GA gates the deprecated
# HTTP shim's bench_long_session.py cannot answer:
#
#   * memory bounded:    agg.kv_bounded is True.
#   * prefill bounded:   agg.prefill_bounded is True.
#
# Usage (from repo root, on Mac M4 / arm64):
#
#     bash scripts/review_pr_e1b_on_mac.sh
#
# Then commit the smoke artifact:
#
#     git add results/platform-tests/pr-e1b-mac-bench-session-30min-*
#     git commit -m "Mac M4 review evidence for PR-E1b (30-min smoke)"
#     git push
#
# Then optionally launch the 4h:
#
#     # in one terminal:
#     bash -c "$(scripts/review_pr_e1b_on_mac.sh --print-server-cmd)"
#     # in another:
#     bash -c "$(scripts/review_pr_e1b_on_mac.sh --print-4h-cmd)"
#
# The 4h JSON gets committed under results/platform-tests/ to the
# PR branch as the binding evidence for v0.3 GA gate G2.
#
# Pre-requisites:
#   - Qwen3-0.6B in HF cache (smoke + 4h both load it).
#   - Free port 50051 (the script binds to 127.0.0.1:50051).
#   - PYTHONPATH unset / not pointing at a stale checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Branch B: bare command printers (used by the `bash -c "$( ... )"` idiom).
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "--print-server-cmd" ]]; then
    cat <<'CMD'
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --bind 127.0.0.1:50051 \
    --capacity 1 --sink 4 --window 64
CMD
    exit 0
fi

if [[ "${1:-}" == "--print-4h-cmd" ]]; then
    stamp="$(date +%s)"
    cat <<CMD
PYTHONPATH=.:sdks/python python3 \\
    scripts/bench_agentic/bench_session_long_run.py \\
    --grpc-address 127.0.0.1:50051 \\
    --tokenizer-id Qwen/Qwen3-0.6B \\
    --duration-s 14400 --turn-spacing-s 30 \\
    --max-tokens 64 \\
    --output results/platform-tests/bench_session_4h_${stamp}.json
CMD
    exit 0
fi

# ---------------------------------------------------------------------------
# Branch A: the 30-min smoke. Default behavior of the script.
# ---------------------------------------------------------------------------

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"
out_json="$out_dir/pr-e1b-mac-bench-session-30min-${stamp}.json"
server_log="$out_dir/pr-e1b-mac-bench-session-30min-${stamp}.server.log"
bench_log="$out_dir/pr-e1b-mac-bench-session-30min-${stamp}.bench.log"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        echo "==> stopping gRPC server (pid=$server_pid)"
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "==> starting gRPC server (logs: $server_log)"
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu \
    --verifier-id Qwen/Qwen3-0.6B \
    --bind 127.0.0.1:50051 \
    --capacity 1 \
    --sink 4 --window 64 \
    --log-level INFO \
    >"$server_log" 2>&1 &
server_pid=$!
echo "    pid=$server_pid"

echo "==> waiting up to 60s for gRPC server to become ready"
ready=0
for _ in $(seq 1 60); do
    if grep -q "kakeya gRPC RuntimeService listening on" "$server_log" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" != "1" ]]; then
    echo "!!! gRPC server did not become ready in 60s. Last 20 lines of log:"
    tail -20 "$server_log" || true
    exit 1
fi

echo "==> running bench_session_long_run.py (1800s = 30min smoke)"
PYTHONPATH=.:sdks/python python3 \
    scripts/bench_agentic/bench_session_long_run.py \
    --grpc-address 127.0.0.1:50051 \
    --tokenizer-id Qwen/Qwen3-0.6B \
    --duration-s 1800 --turn-spacing-s 30 \
    --max-tokens 64 \
    --output "$out_json" \
    2>&1 | tee "$bench_log"

echo
echo "==> Smoke complete. Headline KPIs from $out_json:"
PYTHONPATH=.:sdks/python python3 - "$out_json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)
agg = payload["agg"]
print(f"    n_turns         = {agg['n_turns']}")
print(f"    n_errors        = {agg['n_errors']}")
print(f"    p50_latency_s   = {agg['p50_latency_s']}")
print(f"    p95_latency_s   = {agg['p95_latency_s']}")
print(f"    kv min/mean/max = "
      f"{agg['min_kv_live_bytes']} / "
      f"{agg['mean_kv_live_bytes']} / "
      f"{agg['max_kv_live_bytes']}")
print(f"    kv_bounded      = {agg['kv_bounded']}")
print(f"    prefill_bounded = {agg['prefill_bounded']}")
print(f"    latency_drift_p50_s = {agg['latency_drift_p50_s']}")
PY

echo
echo "==> Done. Commit the artifact:"
echo "    git add $out_dir/pr-e1b-mac-bench-session-30min-${stamp}.*"
echo "    git commit -m 'Mac M4 review evidence for PR-E1b (30-min smoke)'"
echo "    git push"
echo
echo "==> When you're ready for the 4-hour GA evidence run:"
echo "    # in one terminal:"
echo '    bash -c "$(scripts/review_pr_e1b_on_mac.sh --print-server-cmd)"'
echo "    # in another:"
echo '    bash -c "$(scripts/review_pr_e1b_on_mac.sh --print-4h-cmd)"'
echo "    # then commit the resulting bench_session_4h_<stamp>.json"
