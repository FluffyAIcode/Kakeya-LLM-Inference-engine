#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${KAKEYA_BENCH_PYTHON:-$HOME/.venv-distwan/bin/python}"
MODEL="${KAKEYA_BENCH_MODEL:-$HOME/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit}"

exec env PYTHONPATH="$REPO_ROOT:$REPO_ROOT/sdks/python" \
  "$PYTHON" "$REPO_ROOT/scripts/agent_gan_inference_demo.py" \
  --tokenizer-id "$MODEL" \
  "$@"
