#!/usr/bin/env bash
# Mac M4 review aid for PR-E1 (ADR 0008 §6.5 integration suite +
# INV-3 GA gate).
#
# This is the first PR whose Mac M4 evidence is **load-bearing for
# v0.3 GA**: the INV-3 gate is GA gate G3 from ADR 0008 §7. Linux
# unit tests cover the dispatch logic with a deterministic
# FakeVerifier; the integration suite covers the same property
# against the real Qwen3-0.6B verifier on the actual sampler
# numerics, which only runs on Apple Silicon (or a CUDA host with
# the right HF cache).
#
# Produces 1 artifact under results/platform-tests/:
#
#   pr-e1-mac-integration-tests-<unix>.json
#     pytest -m integration tests/integration/ — 3 tests covering
#     the INV-3 byte-exact contract under three chunkings + a
#     repeated-run determinism check.
#
# Usage (from repo root, on Mac M4 / arm64):
#
#     bash scripts/review_pr_e1_on_mac.sh
#
# Then commit the artifact:
#
#     git add results/platform-tests/pr-e1-mac-*
#     git commit -m "Mac M4 review evidence for PR-E1"
#     git push
#
# Same `coverage run -m pytest` + `--include` filter pattern as
# review_pr_b3_on_mac.sh — no `--source` flag, no
# `COVERAGE_CORE=sysmon` env var, sidesteps the Python 3.13 /
# coverage / torch race.
#
# The integration suite has no module under coverage; we don't
# `--cov`-instrument it. The gate is functional (assert byte-equal
# token streams), not coverage-based.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

junit="$out_dir/pr-e1-mac-integration-tests-${stamp}.junit.xml"
report="$out_dir/pr-e1-mac-integration-tests-${stamp}.json"

echo "==> integration suite (INV-3 GA gate G3 against real Qwen3)"
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

# Same aggregate-from-inner-<testsuite> pattern as the other reviewer
# scripts (commit 9d1a250).
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
    "kind": "pr_e1_mac_integration_tests",
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
echo "==> Done."
echo "    Integration tests : $report"
echo "    JUnit             : $junit"
echo
echo "Next:"
echo "    git add $out_dir/pr-e1-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-E1'"
echo "    git push"
