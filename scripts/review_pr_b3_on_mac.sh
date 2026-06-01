#!/usr/bin/env bash
# Mac M4 review aid for PR-B3 (ADR 0008 Phase B, Generate RPC).
#
# Per ADR 0008 §9, PR-B3 is the FIRST Phase-B PR with a mandatory
# Mac M4 integration test report — Linux CI is necessary but not
# sufficient because Generate exercises the verifier-sampler path
# that has MLX-specific behavior (argmax on bf16 tensors, post-trim
# K/V tensor shapes, etc.). This script produces 5 artifacts under
# results/platform-tests/:
#
#   1. pr-b3-mac-generator-tests-<unix>.json
#        pytest tests/inference_engine/session/test_generator.py
#        (31 tests; 100% line coverage on
#        inference_engine/session/generator.py).
#
#   2. pr-b3-mac-grpc-tests-<unix>.json
#        pytest tests/inference_engine/server/test_grpc_app.py
#        (39 tests after PR-B3 additions; 100% line coverage on
#        inference_engine/server/grpc_app.py).
#
#   3. pr-b3-mac-grpc-runtime-smoke-<unix>.json
#        Regression smoke: smoke_grpc_runtime.py (PR-B1 contract,
#        AppendTokens + Generate stay UNIMPLEMENTED with a bare
#        Servicer).
#
#   4. pr-b3-mac-grpc-appender-smoke-<unix>.json
#        Regression smoke: smoke_grpc_appender.py (PR-B2 contract,
#        AppendTokens reachable + Generate still UNIMPLEMENTED).
#
#   5. pr-b3-mac-grpc-generator-smoke-<unix>.json
#        New PR-B3 smoke: smoke_grpc_generator.py (10 RPC scenarios
#        with Generate fully wired, including HistoryTruncated and
#        STOP_REASON_EOS frames).
#
# Usage (from repo root, on Mac M4 / arm64):
#
#     bash scripts/review_pr_b3_on_mac.sh
#
# Then commit the artifacts:
#
#     git add results/platform-tests/pr-b3-mac-*
#     git commit -m "Mac M4 review evidence for PR-B3"
#     git push
#
# Same `coverage run -m pytest` + `--include` filter pattern as
# review_pr_b2_on_mac.sh — sidesteps the Python 3.13 / coverage /
# torch race documented in commit 9cb1c56 + 9d1a250 on PR #45.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

gen_junit="$out_dir/pr-b3-mac-generator-tests-${stamp}.junit.xml"
gen_cov="$out_dir/pr-b3-mac-generator-tests-${stamp}.coverage.xml"
gen_report="$out_dir/pr-b3-mac-generator-tests-${stamp}.json"

grpc_junit="$out_dir/pr-b3-mac-grpc-tests-${stamp}.junit.xml"
grpc_cov="$out_dir/pr-b3-mac-grpc-tests-${stamp}.coverage.xml"
grpc_report="$out_dir/pr-b3-mac-grpc-tests-${stamp}.json"

runtime_smoke="$out_dir/pr-b3-mac-grpc-runtime-smoke-${stamp}.json"
appender_smoke="$out_dir/pr-b3-mac-grpc-appender-smoke-${stamp}.json"
generator_smoke="$out_dir/pr-b3-mac-grpc-generator-smoke-${stamp}.json"


_summarize_pytest() {
    # $1 junit  $2 coverage  $3 out  $4 kind label
    PYTHONPATH=. python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, platform, sys, xml.etree.ElementTree as ET
junit_path, cov_path, out_path, kind = sys.argv[1:5]
jr = ET.parse(junit_path).getroot()

# Aggregate counts from <testsuite> elements (pytest's --junitxml
# emits the counts on the inner <testsuite>, not on the <testsuites>
# wrapper). Same fix as commit 9d1a250.
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


echo "==> [1/5] generator unit tests"
PYTHONPATH=. python3 -m coverage erase
PYTHONPATH=. python3 -m coverage run \
    -m pytest tests/inference_engine/session/test_generator.py \
        --junitxml="$gen_junit" -v
python3 -m coverage report \
    --include='inference_engine/session/generator.py' \
    --fail-under=100 -m
python3 -m coverage xml \
    --include='inference_engine/session/generator.py' \
    -o "$gen_cov"
_summarize_pytest "$gen_junit" "$gen_cov" "$gen_report" \
    "pr_b3_mac_generator_tests"

echo
echo "==> [2/5] gRPC tests (PR-B1 + PR-B2 + PR-B3 surface)"
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
    "pr_b3_mac_grpc_tests"

echo
echo "==> [3/5] runtime smoke (PR-B1 contract regression)"
PYTHONPATH=. python3 scripts/smoke_grpc_runtime.py --report "$runtime_smoke"

echo
echo "==> [4/5] appender smoke (PR-B2 contract regression)"
PYTHONPATH=. python3 scripts/smoke_grpc_appender.py --report "$appender_smoke"

echo
echo "==> [5/5] generator smoke (PR-B3 new)"
PYTHONPATH=. python3 scripts/smoke_grpc_generator.py --report "$generator_smoke"

echo
echo "==> Done."
echo "    Generator tests : $gen_report"
echo "    gRPC tests      : $grpc_report"
echo "    Runtime smoke   : $runtime_smoke"
echo "    Appender smoke  : $appender_smoke"
echo "    Generator smoke : $generator_smoke"
echo
echo "Next:"
echo "    git add $out_dir/pr-b3-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-B3'"
echo "    git push"
