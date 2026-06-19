#!/usr/bin/env bash
# Run the ADR 0009 distributed speculative-decode demo as TWO local processes
# (proposer node + verifier node over real gRPC sockets) and assert the
# distributed output is byte-identical to local greedy. Used for on-device
# validation on the Mac bridge (and runnable anywhere with the deps).
#
# Picks an mlx_lm-free but torch+transformers+grpcio-capable Python: prefers
# KAKEYA_MAC_PYTHON, then the repo venv, then python3.
#
# Usage:
#   bash scripts/run_distributed_demo.sh [--verifier-id ID] [--max-new-tokens N]
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

VERIFIER_ID="Qwen/Qwen3-0.6B"
MAXNEW="48"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verifier-id) shift; VERIFIER_ID="${1:?}" ;;
    --max-new-tokens) shift; MAXNEW="${1:?}" ;;
    *) echo "[dist-demo] ignoring arg: $1" >&2 ;;
  esac
  shift
done

_can() { [ -n "${1:-}" ] && "$1" -c 'import grpc, torch, transformers' >/dev/null 2>&1; }
PYBIN=""
for c in "${KAKEYA_MAC_PYTHON:-}" "$repo_root/.venv-mac/bin/python3.13" \
         "$repo_root/.venv-mac/bin/python" "$HOME/kakeya-venv/bin/python" \
         "$(command -v python3 2>/dev/null || true)"; do
  if _can "$c"; then PYBIN="$c"; break; fi
done
if [[ -z "$PYBIN" ]]; then
  echo "[dist-demo] no Python with grpc+torch+transformers found; set KAKEYA_MAC_PYTHON" >&2
  exit 2
fi
echo "[dist-demo] python=$PYBIN verifier=$VERIFIER_ID max_new=$MAXNEW" >&2

export PYTHONPATH="$repo_root:$repo_root/sdks/python"
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Clean any stale demo procs, start the proposer node, run the verifier node.
for p in $(pgrep -f demo_distributed_spec_decode 2>/dev/null || true); do kill "$p" 2>/dev/null || true; done
sleep 1
"$PYBIN" scripts/demo_distributed_spec_decode.py \
    --role proposer-node --bind 127.0.0.1:50061 --node-id node-b \
    > /tmp/kakeya_dist_proposer.log 2>&1 &
PP=$!
trap 'kill "$PP" 2>/dev/null || true' EXIT
sleep 6
"$PYBIN" scripts/demo_distributed_spec_decode.py \
    --role verifier-node --bind 127.0.0.1:50060 --node-id node-a \
    --peer 127.0.0.1:50061 --verifier-id "$VERIFIER_ID" \
    --max-new-tokens "$MAXNEW"
RC=$?
echo "[dist-demo] VERIFIER_RC=$RC" >&2
exit $RC
