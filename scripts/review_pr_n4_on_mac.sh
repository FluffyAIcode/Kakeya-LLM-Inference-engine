#!/usr/bin/env bash
# Mac M4 review aid for PR-N4 (no-test-doubles cleanup, FINAL).
#
# PR-N4 retires the last verifier-protocol stand-in: the
# ``_MinimalVerifierStub`` (formerly ``FakeVerifier`` import) in
# ``tests/sdk/python/conftest.py``. The SDK transport tests
# (Client + Session) move to ``tests/integration/test_sdk_real.py``
# where they run against a real Qwen3-0.6B-backed gRPC runtime.
#
# After PR-N4: NO test doubles remain in the Linux test tree
# implementing the verifier / engine / tokenizer protocols. The
# Linux CI gate covers ONLY truly verifier-independent code; the
# integration suite is the binding gate for runtime correctness.
#
# Produces 1 artifact:
#
#   results/platform-tests/pr-n4-mac-integration-tests-<unix>.json
#     pytest -m integration tests/integration/ — runs the full
#     accumulated integration suite (PR-E1 INV-3 + PR-N1 coordinator/
#     generator + PR-N2 scheduler + PR-N3 http_shim/engine/tokenizer/
#     streaming + PR-N4 SDK).
#
# Usage (from repo root, on Mac M4):
#
#     bash scripts/review_pr_n4_on_mac.sh
#
# Then commit:
#
#     git add results/platform-tests/pr-n4-mac-*
#     git commit -m "Mac M4 review evidence for PR-N4"
#     git push

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

junit="$out_dir/pr-n4-mac-integration-tests-${stamp}.junit.xml"
report="$out_dir/pr-n4-mac-integration-tests-${stamp}.json"

echo "==> integration suite (full accumulated PR-N1..N4 + PR-E1 GA gate)"
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
    "kind": "pr_n4_mac_integration_tests",
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
echo "    git add $out_dir/pr-n4-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-N4'"
echo "    git push"
