#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${KAKEYA_BENCH_PYTHON:-$HOME/.venv-distwan/bin/python}"
MODEL="${KAKEYA_BENCH_MODEL:-$HOME/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit}"

trap 'echo "[supervisor] external termination ignored; use /quit"' TERM HUP

while true; do
  set +e
  env PYTHONPATH="$REPO_ROOT:$REPO_ROOT/sdks/python" \
    "$PYTHON" "$REPO_ROOT/scripts/agent_gan_repl.py" \
    --tokenizer-id "$MODEL" \
    "$@"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    exit 0
  fi
  if [[ "$status" -eq 129 || "$status" -eq 137 || "$status" -eq 143 ]]; then
    echo "[supervisor] REPL exited from signal ($status); restarting in 2s..."
    sleep 2
    continue
  fi
  exit "$status"
done
