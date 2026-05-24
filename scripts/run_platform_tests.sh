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
#
# The required-models list is overridable via KAKEYA_REQUIRED_MODELS
# (CSV of HF repo ids). The default — proposer + bf16 verifier — is
# the v0.1.0 baseline that the platform-neutral tests assume. Users
# running 4-bit-only test slices can override; CI uses the default.
python3 - <<'PY' || exit 5
import os, sys
from huggingface_hub import try_to_load_from_cache

DEFAULT_REQUIRED = [
    ("Qwen/Qwen3-1.7B", "config.json"),
    ("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "config.json"),
]
override = os.environ.get("KAKEYA_REQUIRED_MODELS", "").strip()
if override:
    required = [(item.strip(), "config.json") for item in override.split(",") if item.strip()]
else:
    required = DEFAULT_REQUIRED

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
        "[runner] override the required list with KAKEYA_REQUIRED_MODELS='repo1,repo2,...'\n"
    )
    sys.exit(5)
print(f"[runner] HF cache contains all {len(required)} required model snapshots")
PY

# ---------- pytest ----------
# Target paths:
#   tests/core              — platform-neutral, runs everywhere
#   tests/inference_engine  — platform-neutral inference_engine.* subpackages
#   tests/backends/<bk>     — backend-specific (only when on that backend)
#
# Coverage scope is set DYNAMICALLY per backend so Linux VM with
# --backend=cpu doesn't fail because MLX/CUDA modules can't be
# imported. We require 100% line coverage on every module included
# below.

cov_targets=("--cov=kv_cache_proposer")

# Always-on platform-neutral subpackages of inference_engine. These run
# pure-CPU torch and have no backend-specific dependencies, so they are
# covered on every host. The `server` subpackage (E2 HTTP API) is in
# this list because FastAPI / httpx / pydantic are platform-neutral.
for sub in proposer memory scheduler server; do
    if [[ -d "$repo_root/inference_engine/$sub" ]]; then
        cov_targets+=("--cov=inference_engine.$sub")
    fi
done

# Always-on platform-neutral training subpackages. These run pure-CPU
# torch and have no backend-specific dependencies, so they are covered
# on every host.
for sub in repr_align; do
    if [[ -d "$repo_root/training/$sub" ]]; then
        cov_targets+=("--cov=training.$sub")
    fi
done

# Selected backend only.
if [[ -d "$repo_root/inference_engine/backends/$backend" ]]; then
    cov_targets+=("--cov=inference_engine.backends.$backend")
fi

test_paths=("tests/core")
if [[ -d "$repo_root/tests/inference_engine" ]]; then
    test_paths+=("tests/inference_engine")
fi
if [[ -d "$repo_root/tests/training" ]]; then
    test_paths+=("tests/training")
fi

# tests/backends/<bk>/test_env.py is platform-neutral (it monkeypatches
# the platform check, so its branches run on every host). The
# backend-specific test files (test_verifier.py etc.) are gated by
# importing the backend module — they error out cleanly on hosts that
# can't load it. We always include the directory; pytest's collection
# is filtered by import side-effects per file.
if [[ -d "$repo_root/tests/backends/mlx" ]]; then
    test_paths+=("tests/backends/mlx")
fi
if [[ -d "$repo_root/tests/backends/cuda" ]]; then
    test_paths+=("tests/backends/cuda")
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
