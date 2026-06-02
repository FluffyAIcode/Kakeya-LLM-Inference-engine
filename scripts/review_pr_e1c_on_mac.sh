#!/usr/bin/env bash
# Mac M4 review aid for PR-E1c (kv_live_bytes reporting fix).
#
# This PR closes the GetSessionInfo.kv_live_bytes=0 reporting bug
# PR-E1b's 4-hour bench surfaced. The Linux unit gate exercises the
# coordinator-level slab-write-through against a deterministic
# FakeVerifier. The Mac M4 review here adds two further checks:
#
#   1. The CPU verifier's kv_live_bytes accessor against real
#      Qwen3-0.6B numerics — non-zero, plateaus at sink+window
#      capacity, equals k_seq_length × per-token bytes.
#   2. A short (5-min) gRPC bench run that confirms
#      GetSessionInfo.kv_live_bytes is no longer 0 over the wire.
#
# Produces 2 artifacts:
#
#   results/platform-tests/pr-e1c-mac-verifier-tests-<unix>.json
#     pytest tests/core/test_verifier.py + tests/backends/mlx/test_verifier.py
#     (the kv_live_bytes-related tests + INV-1 baseline).
#
#   results/platform-tests/pr-e1c-mac-bench-session-5min-<unix>.json
#     bench_session_long_run.py @ 300s. Purpose: visually confirm
#     kv_live_bytes goes 0 -> capped multi-MB once cache hits
#     sink+window. Expected: kv_bounded=True, prefill_bounded=True,
#     min/mean/max kv_live_bytes all > 0.
#
# Usage (from repo root, on Mac M4):
#
#     bash scripts/review_pr_e1c_on_mac.sh
#
# Then commit:
#
#     git add results/platform-tests/pr-e1c-mac-*
#     git commit -m "Mac M4 review evidence for PR-E1c"
#     git push

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

# --- Part 1: verifier-level tests -----------------------------------------
verif_junit="$out_dir/pr-e1c-mac-verifier-tests-${stamp}.junit.xml"
verif_report="$out_dir/pr-e1c-mac-verifier-tests-${stamp}.json"

echo "==> CPU + MLX verifier tests covering kv_live_bytes (PR-E1c)"
PYTHONPATH=.:sdks/python python3 -m pytest \
    tests/core/test_verifier.py \
    tests/backends/mlx/test_verifier.py \
    -k "kv_live_bytes or k_seq_length or cache_inspector" \
    --junitxml="$verif_junit" \
    -v

PYTHONPATH=.:sdks/python python3 - "$verif_junit" "$verif_report" <<'PY'
import json
import platform
import sys
import xml.etree.ElementTree as ET
junit_path, out_path = sys.argv[1:3]
jr = ET.parse(junit_path).getroot()
testsuites = list(jr.iter("testsuite"))
total_tests = sum(int(ts.get("tests", "0")) for ts in testsuites)
total_failures = sum(int(ts.get("failures", "0")) for ts in testsuites)
total_errors = sum(int(ts.get("errors", "0")) for ts in testsuites)
total_skipped = sum(int(ts.get("skipped", "0")) for ts in testsuites)
report = {
    "schema_version": 1,
    "kind": "pr_e1c_mac_verifier_tests",
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "junit": {
        "tests": total_tests, "failures": total_failures,
        "errors": total_errors, "skipped": total_skipped,
    },
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"  -> {out_path}")
PY

# --- Part 2: 5-min gRPC bench ---------------------------------------------
# This part requires PR-E1b's scripts/start_grpc_runtime_server.py and
# scripts/bench_agentic/bench_session_long_run.py to be present on the
# checked-out tree. PR-E1c merges *after* PR-E1b in the recommended
# sequence; if PR-E1c is exercised against a tree where PR-E1b hasn't
# landed yet, skip the bench gracefully so Part 1 evidence still
# commits cleanly.
if [[ ! -f scripts/start_grpc_runtime_server.py \
   || ! -f scripts/bench_agentic/bench_session_long_run.py ]]; then
    echo
    echo "==> Part 2 skipped: PR-E1b artifacts not present on this tree."
    echo "    Re-run after PR-E1b lands to capture the bench evidence."
    echo
    echo "==> Done. Commit Part 1 evidence:"
    echo "    git add $out_dir/pr-e1c-mac-verifier-tests-${stamp}.*"
    echo "    git commit -m 'Mac M4 review evidence for PR-E1c (verifier tests)'"
    echo "    git push"
    exit 0
fi

bench_json="$out_dir/pr-e1c-mac-bench-session-5min-${stamp}.json"
server_log="$out_dir/pr-e1c-mac-bench-session-5min-${stamp}.server.log"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo
echo "==> starting gRPC server (logs: $server_log)"
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --bind 127.0.0.1:50051 --capacity 1 --sink 4 --window 64 \
    >"$server_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 60); do
    if grep -q "kakeya gRPC RuntimeService listening on" "$server_log" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" != "1" ]]; then
    echo "!!! gRPC server didn't become ready"
    tail -20 "$server_log" || true
    exit 1
fi

echo "==> running 5-min bench (validates kv_live_bytes is non-zero)"
PYTHONPATH=.:sdks/python python3 \
    scripts/bench_agentic/bench_session_long_run.py \
    --grpc-address 127.0.0.1:50051 \
    --tokenizer-id Qwen/Qwen3-0.6B \
    --duration-s 300 --turn-spacing-s 30 \
    --max-tokens 64 \
    --output "$bench_json"

echo
echo "==> Headline KPIs from $bench_json:"
PYTHONPATH=.:sdks/python python3 - "$bench_json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)
agg = payload["agg"]
print(f"    n_turns         = {agg['n_turns']}")
print(f"    n_errors        = {agg['n_errors']}")
print(f"    p50_latency_s   = {agg['p50_latency_s']}")
print(f"    kv min/mean/max = "
      f"{agg['min_kv_live_bytes']} / "
      f"{agg['mean_kv_live_bytes']} / "
      f"{agg['max_kv_live_bytes']}")
print(f"    kv_bounded      = {agg['kv_bounded']}")
print(f"    prefill_bounded = {agg['prefill_bounded']}")
m = agg["max_kv_live_bytes"]
if m and m > 0:
    print(f"    -> kv_live_bytes is non-zero; PR-E1c reporting fix VERIFIED.")
else:
    print(f"    -> kv_live_bytes is still 0; PR-E1c FAILED.")
    sys.exit(1)
PY

echo
echo "==> Done. Commit:"
echo "    git add $out_dir/pr-e1c-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-E1c'"
echo "    git push"
