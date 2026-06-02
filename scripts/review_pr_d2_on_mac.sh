#!/usr/bin/env bash
# Mac M4 review aid for PR-D2 (HTTP shim refactor onto SessionStore).
#
# PR-D2 retired the Scheduler + PooledVerifier + SpeculativeEngine
# machinery from the HTTP shim's request path. Each
# /v1/chat/completions request is now a single-shot session under
# SessionStore — same semantics as the gRPC RuntimeService. The
# integration suite's test_http_shim_real.py is the binding gate
# for this refactor: it drives the full FastAPI app (with real
# Qwen3-0.6B verifier) through OpenAI-compat surface tests
# including SSE streaming, auth, error envelopes, /metrics, and
# /v1/models.
#
# Produces 1 artifact:
#
#   results/platform-tests/pr-d2-mac-integration-tests-<unix>.json
#     pytest -m integration tests/integration/ — runs the full
#     accumulated integration suite (PR-E1 INV-3 + PR-N1 coordinator/
#     generator + PR-N2 scheduler + PR-N3 http_shim/engine/
#     tokenizer/streaming + PR-N4 SDK).
#
# Usage (from repo root, on Mac M4):
#
#     bash scripts/review_pr_d2_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

junit="$out_dir/pr-d2-mac-integration-tests-${stamp}.junit.xml"
report="$out_dir/pr-d2-mac-integration-tests-${stamp}.json"

echo "==> integration suite (HTTP shim onto SessionStore + full N1..N4 cumulative)"
PYTHONPATH=.:sdks/python python3 -m pytest \
    -m integration \
    tests/integration/ \
    --junitxml="$junit" \
    -v

PYTHONPATH=.:sdks/python python3 - "$junit" "$report" <<'PY'
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
    "kind": "pr_d2_mac_integration_tests",
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

echo
echo "==> Done. Commit:"
echo "    git add $out_dir/pr-d2-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-D2'"
echo "    git push"
