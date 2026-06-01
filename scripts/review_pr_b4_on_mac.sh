#!/usr/bin/env bash
# Mac M4 review aid for PR-B4 (ADR 0008 Phase B, Python SDK).
#
# Per ADR 0008 §9, PR-B4 is a Linux-only path (the SDK is pure
# Python wrapping `grpc.insecure_channel` + the generated stubs;
# no MLX runtime code). The §9 carve-out applies, so this script
# is a review affordance — not a mandatory §9 report. Reviewers
# who want hardware-level evidence on Apple Silicon run it; the
# binding gate is Linux CI.
#
# Produces 4 artifacts under results/platform-tests/:
#
#   1. pr-b4-mac-sdk-tests-<unix>.json
#        pytest tests/sdk/python/ with 100% coverage on
#        sdks/python/kakeya/*. 66 tests total covering Client,
#        Session, errors, and end-to-end gRPC streaming through
#        the SDK.
#
#   2. pr-b4-mac-grpc-runtime-smoke-<unix>.json
#        Regression smoke: smoke_grpc_runtime.py (PR-B1 contract).
#
#   3. pr-b4-mac-grpc-appender-smoke-<unix>.json
#        Regression smoke: smoke_grpc_appender.py (PR-B2 contract).
#
#   4. pr-b4-mac-grpc-generator-smoke-<unix>.json
#        Regression smoke: smoke_grpc_generator.py (PR-B3 contract).
#
# Same `coverage run -m pytest` + `--include` filter pattern as
# review_pr_b3_on_mac.sh — sidesteps the Python 3.13 / coverage /
# torch race documented in commits 9cb1c56 + 9d1a250.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

sdk_junit="$out_dir/pr-b4-mac-sdk-tests-${stamp}.junit.xml"
sdk_cov="$out_dir/pr-b4-mac-sdk-tests-${stamp}.coverage.xml"
sdk_report="$out_dir/pr-b4-mac-sdk-tests-${stamp}.json"

runtime_smoke="$out_dir/pr-b4-mac-grpc-runtime-smoke-${stamp}.json"
appender_smoke="$out_dir/pr-b4-mac-grpc-appender-smoke-${stamp}.json"
generator_smoke="$out_dir/pr-b4-mac-grpc-generator-smoke-${stamp}.json"


_summarize_pytest() {
    # $1 junit  $2 coverage  $3 out  $4 kind label
    PYTHONPATH=.:sdks/python python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, platform, sys, xml.etree.ElementTree as ET
junit_path, cov_path, out_path, kind = sys.argv[1:5]
jr = ET.parse(junit_path).getroot()

# Aggregate counts from <testsuite> elements; pytest puts counts on
# the inner element, not on the <testsuites> wrapper. Same fix as
# commit 9d1a250.
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


echo "==> [1/4] SDK unit + integration tests"
PYTHONPATH=.:sdks/python python3 -m coverage erase
PYTHONPATH=.:sdks/python python3 -m coverage run \
    -m pytest tests/sdk/python/ \
        --junitxml="$sdk_junit" -v
python3 -m coverage report \
    --include='sdks/python/kakeya/*' \
    --fail-under=100 -m
python3 -m coverage xml \
    --include='sdks/python/kakeya/*' \
    -o "$sdk_cov"
_summarize_pytest "$sdk_junit" "$sdk_cov" "$sdk_report" \
    "pr_b4_mac_sdk_tests"

echo
echo "==> [2/4] runtime smoke (PR-B1 regression)"
PYTHONPATH=. python3 scripts/smoke_grpc_runtime.py --report "$runtime_smoke"

echo
echo "==> [3/4] appender smoke (PR-B2 regression)"
PYTHONPATH=. python3 scripts/smoke_grpc_appender.py --report "$appender_smoke"

echo
echo "==> [4/4] generator smoke (PR-B3 regression)"
PYTHONPATH=. python3 scripts/smoke_grpc_generator.py --report "$generator_smoke"

echo
echo "==> Done."
echo "    SDK tests       : $sdk_report"
echo "    Runtime smoke   : $runtime_smoke"
echo "    Appender smoke  : $appender_smoke"
echo "    Generator smoke : $generator_smoke"
echo
echo "Next:"
echo "    git add $out_dir/pr-b4-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-B4'"
echo "    git push"
