#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-.:sdks/python}"

# This script is the canonical local CI subset. GitHub Actions invokes this
# exact file, then adds Docker, package, proto, TypeScript, and Mac integration
# jobs around it.
lake build

"$PYTHON" -m coverage erase
"$PYTHON" -m coverage run -m pytest \
  tests/inference_engine/server/ \
  tests/inference_engine/memory/ \
  tests/inference_engine/scheduler/ \
  tests/inference_engine/pipeline/ \
  tests/inference_engine/session/ \
  tests/inference_engine/bench/ \
  tests/inference_engine/setup/ \
  tests/inference_engine/bridge/ \
  tests/inference_engine/distributed/ \
  tests/inference_engine/network/ \
  tests/sdk/python/ \
  tests/training/repr_align/ \
  tests/backends/mlx/test_env.py \
  --junitxml=junit.xml \
  -v

COVERAGE_INCLUDE='inference_engine/server/auth.py,inference_engine/server/config.py,inference_engine/server/errors.py,inference_engine/server/grpc_app.py,inference_engine/server/metrics.py,inference_engine/server/schemas.py,inference_engine/server/proto_gen/**/*.py,inference_engine/memory/*,inference_engine/bridge/*,inference_engine/distributed/*,inference_engine/network/*,inference_engine/scheduler/config.py,inference_engine/scheduler/session.py,inference_engine/pipeline/*,inference_engine/session/store.py,inference_engine/setup/*,sdks/python/kakeya/__init__.py,sdks/python/kakeya/errors.py,training/repr_align/*'

"$PYTHON" -m coverage report \
  --include="$COVERAGE_INCLUDE" \
  --fail-under=100
"$PYTHON" -m coverage xml \
  -o coverage.xml \
  --include="$COVERAGE_INCLUDE"
