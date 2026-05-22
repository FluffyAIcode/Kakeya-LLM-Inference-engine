#!/usr/bin/env bash
# Unified platform-test runner.
#
# Usage:
#   ./scripts/run_platform_tests.sh --backend mlx   --report results/platform-tests/mac-$(date +%s).json
#   ./scripts/run_platform_tests.sh --backend cuda  --report results/platform-tests/cuda-$(date +%s).json
#   ./scripts/run_platform_tests.sh --backend cpu   --report results/platform-tests/cpu-$(date +%s).json
#
# This script:
#   1. Hard-checks the environment for the requested backend (no fallback).
#   2. Runs `pytest` over the platform-neutral suite + the backend-specific suite,
#      enforcing 100% line coverage on the targeted module(s).
#   3. Writes a structured JSON report to --report (caller commits it back so
#      the maintainer can review without needing the local hardware).

set -euo pipefail

backend=""
report=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --report)  report="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "[runner] unknown arg: $1" >&2
            exit 64
            ;;
    esac
done

if [[ -z "$backend" ]]; then
    echo "[runner] --backend is required (one of: mlx, cuda, cpu)" >&2
    exit 64
fi
if [[ -z "$report" ]]; then
    repo_root="$(cd "$(dirname "$0")/.." && pwd)"
    mkdir -p "${repo_root}/results/platform-tests"
    report="${repo_root}/results/platform-tests/${backend}-$(date +%s).json"
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# ---------- environment checks (hard, no fallback) ----------
case "$backend" in
    mlx)
        python3 -c "import mlx.core as mx; assert mx.metal.is_available(), 'Metal not available'" \
            || { echo "[runner] Metal/MLX unavailable on this host"; exit 2; }
        ;;
    cuda)
        python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
            || { echo "[runner] CUDA unavailable on this host"; exit 2; }
        ;;
    cpu)
        python3 -c "import torch; assert torch.__version__"
        ;;
    *)
        echo "[runner] unknown backend: $backend" >&2
        exit 64
        ;;
esac

# ---------- HF cache pre-flight (hard, no fallback) ----------
# The test suite loads real Qwen3 weights from the HuggingFace cache.
# If the cache is empty we error out IMMEDIATELY rather than let 78
# tests cascade-fail on network errors.
python3 - <<'PY' || exit 5
import sys
from huggingface_hub import try_to_load_from_cache
import os
required = [
    ("Qwen/Qwen3-1.7B", "config.json"),
    ("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "config.json"),
]
missing = []
for repo, fname in required:
    p = try_to_load_from_cache(repo_id=repo, filename=fname)
    if not p:
        missing.append(repo)
if missing:
    sys.stderr.write(
        "[runner] HF cache is missing the following repos:\n"
    )
    for r in missing:
        sys.stderr.write(f"  - {r}\n")
    sys.stderr.write(
        "[runner] re-run ./scripts/setup_mac.sh (or setup_cuda.sh) to download.\n"
        "[runner] if you're in mainland China, set HF_ENDPOINT=https://hf-mirror.com first.\n"
    )
    sys.exit(5)
print(f"[runner] HF cache contains both required model snapshots")
PY

# ---------- pytest ----------
# Target paths:
#   tests/core            — platform-neutral, runs everywhere
#   tests/backends/<bk>   — backend-specific
#
# Coverage is enforced against `inference_engine` (when present) and the
# legacy `kv_cache_proposer` package. We require 100% line coverage on the
# code targeted by the current test selection.

cov_targets=("--cov=kv_cache_proposer")
if [[ -d "$repo_root/inference_engine" ]]; then
    cov_targets+=("--cov=inference_engine")
fi
if [[ -d "$repo_root/inference_engine/backends/$backend" ]]; then
    cov_targets+=("--cov=inference_engine.backends.$backend")
fi

test_paths=("tests/core")
if [[ -d "$repo_root/tests/backends/$backend" ]]; then
    test_paths+=("tests/backends/$backend")
fi

mkdir -p "$(dirname "$report")"
junit="${report%.json}.junit.xml"
covxml="${report%.json}.coverage.xml"

set +e
pytest \
    "${test_paths[@]}" \
    "${cov_targets[@]}" \
    --cov-report="xml:${covxml}" \
    --cov-report=term-missing \
    --cov-fail-under=100 \
    --junitxml="${junit}" \
    -v
exit_code=$?
set -e

# ---------- structured JSON report ----------
python3 - "$report" "$junit" "$covxml" "$backend" "$exit_code" <<'PY'
import json, sys, os, platform
from xml.etree import ElementTree as ET

report_path, junit_path, cov_path, backend, exit_code = sys.argv[1:6]
exit_code = int(exit_code)

def parse_junit(p):
    if not os.path.exists(p):
        return None
    root = ET.parse(p).getroot()
    suites = root if root.tag == "testsuites" else [root]
    tests = failures = errors = skipped = 0
    cases = []
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
        for c in s.iter("testcase"):
            cases.append({
                "classname": c.get("classname"),
                "name": c.get("name"),
                "time": float(c.get("time", 0.0)),
                "status": (
                    "failed" if c.find("failure") is not None
                    else "errored" if c.find("error") is not None
                    else "skipped" if c.find("skipped") is not None
                    else "passed"
                ),
            })
    return {"tests": tests, "failures": failures, "errors": errors,
            "skipped": skipped, "cases": cases}

def parse_coverage(p):
    if not os.path.exists(p):
        return None
    root = ET.parse(p).getroot()
    return {
        "line_rate": float(root.get("line-rate", 0.0)),
        "branch_rate": float(root.get("branch-rate", 0.0)),
        "lines_covered": int(root.get("lines-covered", 0)),
        "lines_valid": int(root.get("lines-valid", 0)),
    }

out = {
    "backend": backend,
    "exit_code": exit_code,
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "junit": parse_junit(junit_path),
    "coverage": parse_coverage(cov_path),
}
with open(report_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"[runner] wrote {report_path}")
PY

echo "[runner] exit code: $exit_code"
exit "$exit_code"
