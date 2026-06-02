#!/usr/bin/env bash
# Mac M4 review aid for PR-N3 (no-test-doubles cleanup, scope =
# HTTP shim + engine + tokenizer + streaming doubles).
#
# PR-N3 retired the largest cluster of test doubles in the Linux
# tree:
#   - DeterministicEngine + DeterministicTokenizer in
#     tests/inference_engine/server/conftest.py
#   - Engine subtypes (_RaisingEngine, _ProxyEngine,
#     _AlwaysHoldingEngine, _KVAwareSlowEngine) in test_app_*.py
#   - Tokenizer subtypes (_BrokenTokenizer, _EmptyTemplateTokenizer,
#     _NoEosTokenizer) in test_app_*.py
#   - Verifier / decoder doubles (_VerifierDouble,
#     _LegacyVerifierDouble, _DecoderDouble, _DecoderResult) in
#     test_engine.py
#
# All HTTP-shim runtime tests, engine wrapper tests, tokenizer
# wrapper tests, and streaming-detokenizer tests moved to
# tests/integration/ where they run against the real
# ``SpeculativeEngine`` over Qwen3-0.6B.
#
# Produces 1 artifact:
#
#   results/platform-tests/pr-n3-mac-integration-tests-<unix>.json
#     pytest -m integration tests/integration/ — runs ALL integration
#     suites accumulated to date (PR-E1 INV-3 gate, PR-N1 coordinator
#     and generator, PR-N2 scheduler, PR-N3 http_shim + engine +
#     tokenizer + streaming).
#
# Usage (from repo root, on Mac M4):
#
#     bash scripts/review_pr_n3_on_mac.sh
#
# Then commit:
#
#     git add results/platform-tests/pr-n3-mac-*
#     git commit -m "Mac M4 review evidence for PR-N3"
#     git push

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

junit="$out_dir/pr-n3-mac-integration-tests-${stamp}.junit.xml"
report="$out_dir/pr-n3-mac-integration-tests-${stamp}.json"

echo "==> integration suite (all PR-N1/N2/N3 migrated tests + INV-3 GA gate)"
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
    "kind": "pr_n3_mac_integration_tests",
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
echo "    git add $out_dir/pr-n3-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-N3'"
echo "    git push"
