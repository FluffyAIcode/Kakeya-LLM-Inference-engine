#!/usr/bin/env bash
# Mac M4 review aid for PR-N1 (no-test-doubles cleanup, scope =
# verifier-protocol mirror classes).
#
# PR-N1 retired FakeVerifier, _LyingVerifier, _RegressingVerifier,
# and _LyingFakeVerifier from the Linux test tree. Their dispatch /
# state-mirror tests moved to tests/integration/test_coordinator_real.py
# and tests/integration/test_generator_real.py, where they run
# against the real Qwen3-0.6B SinkWindowVerifier instead of a
# hand-coded mirror. This script runs that integration suite on
# Apple Silicon and produces the JSON evidence reviewers commit.
#
# Produces 1 artifact:
#
#   results/platform-tests/pr-n1-mac-integration-tests-<unix>.json
#     pytest -m integration tests/integration/ — coordinator and
#     generator integration tests against real Qwen3 + the existing
#     INV-3 byte-exact GA gate. Acceptance: all tests pass.
#
# Usage (from repo root, on Mac M4 / arm64):
#
#     bash scripts/review_pr_n1_on_mac.sh
#
# Then commit:
#
#     git add results/platform-tests/pr-n1-mac-*
#     git commit -m "Mac M4 review evidence for PR-N1"
#     git push

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

junit="$out_dir/pr-n1-mac-integration-tests-${stamp}.junit.xml"
report="$out_dir/pr-n1-mac-integration-tests-${stamp}.json"

echo "==> integration suite (PR-N1 migrated tests + INV-3 GA gate)"
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

cases = []
for tc in jr.iter("testcase"):
    cases.append({
        "classname": tc.get("classname"),
        "name": tc.get("name"),
        "time": float(tc.get("time", 0.0)),
        "outcome": (
            "failed" if tc.find("failure") is not None
            else "errored" if tc.find("error") is not None
            else "skipped" if tc.find("skipped") is not None
            else "passed"
        ),
    })

report = {
    "schema_version": 1,
    "kind": "pr_n1_mac_integration_tests",
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "junit": {
        "tests": total_tests,
        "failures": total_failures,
        "errors": total_errors,
        "skipped": total_skipped,
        "cases": cases,
    },
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"  -> {out_path}")
PY

echo
echo "==> Done. Commit:"
echo "    git add $out_dir/pr-n1-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-N1'"
echo "    git push"
