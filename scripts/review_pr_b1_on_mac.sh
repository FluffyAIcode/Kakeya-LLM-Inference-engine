#!/usr/bin/env bash
# Mac M4 review aid for PR-B1 (ADR 0008 Phase B, gRPC runtime stub).
#
# Generates two artifacts under results/platform-tests/ that you can
# commit back to the PR branch so the PR description has direct
# evidence of "I ran this on Mac and it behaves as advertised":
#
#   1. pr-b1-mac-grpc-tests-<unix>.json
#        pytest run of tests/inference_engine/server/test_grpc_app.py
#        (22 tests; 100% line coverage on inference_engine/server/grpc_app.py).
#
#   2. pr-b1-mac-grpc-smoke-<unix>.json
#        scripts/smoke_grpc_runtime.py — 10 RPC scenarios walked
#        through end-to-end on a real grpc.aio server bound to a
#        local free port. Visible per-step JSON-Lines output so the
#        wire-level behavior is auditable.
#
# Usage (from repo root, on Mac M4 / arm64):
#
#     bash scripts/review_pr_b1_on_mac.sh
#
# Then:
#
#     git add results/platform-tests/pr-b1-mac-grpc-*
#     git commit -m "Mac M4 review evidence for PR-B1"
#     git push
#
# This is NOT a CI-gating script. Linux CI on the PR is the binding
# gate (PR-B1 is Linux-only path per ADR 0008 §9 carve-out). This
# script gives you, the reviewer, the same evidence on your hardware
# so you can satisfy yourself the gRPC surface behaves correctly on
# Apple Silicon as well.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
tests_report="$out_dir/pr-b1-mac-grpc-tests-${stamp}.json"
tests_junit="$out_dir/pr-b1-mac-grpc-tests-${stamp}.junit.xml"
tests_cov="$out_dir/pr-b1-mac-grpc-tests-${stamp}.coverage.xml"
smoke_report="$out_dir/pr-b1-mac-grpc-smoke-${stamp}.json"

mkdir -p "$out_dir"

echo "==> [1/2] pytest tests/inference_engine/server/test_grpc_app.py"
# Run coverage via `coverage run -m pytest` rather than pytest-cov.
# Two design points:
#
#   1. No `--source=...` flag on `coverage run`. The user-reported
#      Mac M4 segfault (Python 3.13.12) reproduced reliably with
#      `--source=inference_engine.server.grpc_app`; the same coverage
#      version with no `--source` and `--include=...` applied at
#      report time runs cleanly. Tracing-then-filtering is slightly
#      slower than source-scoped tracing but it sidesteps a coverage
#      / torch-_C / sys.monitoring race that we don't control.
#
#   2. No `COVERAGE_CORE=sysmon` env var. .coveragerc already sets
#      `[run] core = sysmon` for Python 3.12+. Setting it via env
#      forces an earlier init path that, on Python 3.13, can
#      segfault inside torch's C extension. The config-file route
#      defers init until `coverage run` actually starts, which is
#      after torch's import has settled.
PYTHONPATH=. python3 -m coverage erase
PYTHONPATH=. python3 -m coverage run \
    -m pytest \
        tests/inference_engine/server/test_grpc_app.py \
        --junitxml="$tests_junit" \
        -v
python3 -m coverage report \
    --include='inference_engine/server/grpc_app.py' \
    --fail-under=100 -m
python3 -m coverage xml \
    --include='inference_engine/server/grpc_app.py' \
    -o "$tests_cov"

# Convert junit + summary into a JSON report for parity with the
# other artifacts under results/platform-tests/.
PYTHONPATH=. python3 - "$tests_junit" "$tests_cov" "$tests_report" <<'PY'
import json
import platform
import sys
import xml.etree.ElementTree as ET

junit_path, cov_path, out_path = sys.argv[1:4]

junit_root = ET.parse(junit_path).getroot()
cases = []
for tc in junit_root.iter("testcase"):
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
    "kind": "pr_b1_mac_grpc_tests",
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "junit": {
        "tests": int(junit_root.get("tests", "0")),
        "failures": int(junit_root.get("failures", "0")),
        "errors": int(junit_root.get("errors", "0")),
        "skipped": int(junit_root.get("skipped", "0")),
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
print(f"tests report -> {out_path}")
PY

echo
echo "==> [2/2] scripts/smoke_grpc_runtime.py"
PYTHONPATH=. python3 scripts/smoke_grpc_runtime.py --report "$smoke_report"

echo
echo "==> Done."
echo "    Tests   : $tests_report"
echo "    Smoke   : $smoke_report"
echo "    Junit   : $tests_junit"
echo "    Coverage: $tests_cov"
echo
echo "Next:"
echo "    git add $out_dir/pr-b1-mac-grpc-*"
echo "    git commit -m 'Mac M4 review evidence for PR-B1'"
echo "    git push"
