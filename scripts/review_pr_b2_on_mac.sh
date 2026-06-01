#!/usr/bin/env bash
# Mac M4 review aid for PR-B2 (ADR 0008 Phase B, AppendTokens RPC).
#
# Generates four artifacts under results/platform-tests/:
#
#   1. pr-b2-mac-coordinator-tests-<unix>.json
#        pytest run of tests/inference_engine/session/test_coordinator.py
#        (19 tests; 100% line coverage on inference_engine/session/coordinator.py).
#
#   2. pr-b2-mac-grpc-tests-<unix>.json
#        pytest run of tests/inference_engine/server/test_grpc_app.py
#        (28 tests after PR-B2's additions; 100% line coverage on
#        inference_engine/server/grpc_app.py).
#
#   3. pr-b2-mac-grpc-runtime-smoke-<unix>.json
#        Regression-smoke: scripts/smoke_grpc_runtime.py — confirms the
#        no-coordinator path (PR-B1's UNIMPLEMENTED contract) still
#        holds after PR-B2's wiring.
#
#   4. pr-b2-mac-grpc-appender-smoke-<unix>.json
#        New PR-B2 smoke: scripts/smoke_grpc_appender.py — 10 RPC
#        scenarios with AppendTokensCoordinator + FakeVerifier wired
#        in, including INV-1 violation -> FAILED_PRECONDITION mapping.
#
# Usage (from repo root, on Mac M4 / arm64):
#
#     bash scripts/review_pr_b2_on_mac.sh
#
# Then:
#
#     git add results/platform-tests/pr-b2-mac-*
#     git commit -m "Mac M4 review evidence for PR-B2"
#     git push
#
# Same pattern as scripts/review_pr_b1_on_mac.sh; uses `coverage run`
# instead of `pytest --cov` to sidestep the torch/coverage tracer race
# on Python 3.13 (see commit 79a82f2 on PR-B1).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

coord_junit="$out_dir/pr-b2-mac-coordinator-tests-${stamp}.junit.xml"
coord_cov="$out_dir/pr-b2-mac-coordinator-tests-${stamp}.coverage.xml"
coord_report="$out_dir/pr-b2-mac-coordinator-tests-${stamp}.json"

grpc_junit="$out_dir/pr-b2-mac-grpc-tests-${stamp}.junit.xml"
grpc_cov="$out_dir/pr-b2-mac-grpc-tests-${stamp}.coverage.xml"
grpc_report="$out_dir/pr-b2-mac-grpc-tests-${stamp}.json"

runtime_smoke="$out_dir/pr-b2-mac-grpc-runtime-smoke-${stamp}.json"
appender_smoke="$out_dir/pr-b2-mac-grpc-appender-smoke-${stamp}.json"


_summarize_pytest() {
    # $1 junit  $2 coverage  $3 out  $4 kind label
    PYTHONPATH=. python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, platform, sys, xml.etree.ElementTree as ET
junit_path, cov_path, out_path, kind = sys.argv[1:5]
jr = ET.parse(junit_path).getroot()

# pytest's --junitxml emits <testsuites> (root) > <testsuite> (inner);
# count attributes (tests / failures / errors / skipped) live on the
# *inner* element, not on the wrapper. Earlier versions of this
# helper read the wrapper and silently produced "tests": 0. Aggregate
# from every <testsuite> so we are correct whether the root is
# <testsuites> (pytest) or <testsuite> (other producers).
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
cov_root = ET.parse(cov_path).getroot()
report = {
    "schema_version": 1,
    "kind": kind,
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
    "coverage": {
        "line_rate": float(cov_root.get("line-rate", "0.0")),
        "branch_rate": float(cov_root.get("branch-rate", "0.0")),
        "lines_covered": int(cov_root.get("lines-covered", "0")),
        "lines_valid": int(cov_root.get("lines-valid", "0")),
    },
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"  -> {out_path}")
PY
}


echo "==> [1/4] coordinator unit tests"
# See review_pr_b1_on_mac.sh for the rationale on the no-`--source`
# / no-`COVERAGE_CORE=sysmon` choice; we mirror that here.
PYTHONPATH=. python3 -m coverage erase
PYTHONPATH=. python3 -m coverage run \
    -m pytest tests/inference_engine/session/test_coordinator.py \
        --junitxml="$coord_junit" -v
python3 -m coverage report \
    --include='inference_engine/session/coordinator.py' \
    --fail-under=100 -m
python3 -m coverage xml \
    --include='inference_engine/session/coordinator.py' \
    -o "$coord_cov"
_summarize_pytest "$coord_junit" "$coord_cov" "$coord_report" \
    "pr_b2_mac_coordinator_tests"

echo
echo "==> [2/4] gRPC tests (PR-B1 surface + PR-B2 AppendTokens)"
PYTHONPATH=. python3 -m coverage erase
PYTHONPATH=. python3 -m coverage run \
    -m pytest tests/inference_engine/server/test_grpc_app.py \
        --junitxml="$grpc_junit" -v
python3 -m coverage report \
    --include='inference_engine/server/grpc_app.py' \
    --fail-under=100 -m
python3 -m coverage xml \
    --include='inference_engine/server/grpc_app.py' \
    -o "$grpc_cov"
_summarize_pytest "$grpc_junit" "$grpc_cov" "$grpc_report" \
    "pr_b2_mac_grpc_tests"

echo
echo "==> [3/4] runtime smoke (PR-B1 contract regression: AppendTokens stays UNIMPLEMENTED without coordinator)"
PYTHONPATH=. python3 scripts/smoke_grpc_runtime.py --report "$runtime_smoke"

echo
echo "==> [4/4] appender smoke (PR-B2 AppendTokens with coordinator wired)"
PYTHONPATH=. python3 scripts/smoke_grpc_appender.py --report "$appender_smoke"

echo
echo "==> Done."
echo "    Coordinator tests : $coord_report"
echo "    gRPC tests        : $grpc_report"
echo "    Runtime smoke     : $runtime_smoke"
echo "    Appender smoke    : $appender_smoke"
echo
echo "Next:"
echo "    git add $out_dir/pr-b2-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-B2'"
echo "    git push"
