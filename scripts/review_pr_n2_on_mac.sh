#!/usr/bin/env bash
# Mac M4 review aid for PR-N2 (no-test-doubles cleanup, scope =
# DeterministicEngine + DeterministicTokenizer in scheduler/conftest.py
# + the test_scheduler.py tests that depended on them).
#
# PR-N2 retired the scheduler/-side ``DeterministicEngine`` and
# ``DeterministicTokenizer`` test doubles. Their dispatch /
# admission-control / lifecycle tests moved to
# tests/integration/test_scheduler_real.py, where they run against
# the real ``SpeculativeEngine`` over Qwen3-0.6B.
#
# The HTTP shim's separate copy of these doubles (in
# ``tests/inference_engine/server/conftest.py``) and the engine-
# subtype doubles (``_RaisingEngine``, ``_ProxyEngine``, etc.) are
# PR-N3 scope and remain in place on this branch.
#
# Produces 1 artifact:
#
#   results/platform-tests/pr-n2-mac-integration-tests-<unix>.json
#     pytest -m integration tests/integration/test_scheduler_real.py
#     against real Qwen3-0.6B + SpeculativeEngine. Acceptance: all
#     pass; structural invariants hold (state transitions, slab
#     acquire/release, admission control, concurrency).
#
# Usage (from repo root, on Mac M4):
#
#     bash scripts/review_pr_n2_on_mac.sh
#
# Then commit:
#
#     git add results/platform-tests/pr-n2-mac-*
#     git commit -m "Mac M4 review evidence for PR-N2"
#     git push

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

junit="$out_dir/pr-n2-mac-integration-tests-${stamp}.junit.xml"
report="$out_dir/pr-n2-mac-integration-tests-${stamp}.json"

echo "==> integration suite (PR-N2 migrated scheduler tests + INV-3 GA gate)"
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
    "kind": "pr_n2_mac_integration_tests",
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
echo "    git add $out_dir/pr-n2-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-N2'"
echo "    git push"
