#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${KAKEYA_BENCH_PYTHON:-$HOME/.venv-distwan/bin/python}"

exec env PYTHONPATH="$ROOT:$ROOT/sdks/python" \
  "$PYTHON" "$ROOT/autoresearch/prefill/supervisor.py" "$@"
