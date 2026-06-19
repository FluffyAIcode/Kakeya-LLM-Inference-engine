#!/usr/bin/env bash
# Start a local ProposerService and benchmark the distributed spec-decode path
# (token throughput, bounded-KV footprint, gRPC RTT) against it. On-device perf
# validation for the Mac bridge; runnable anywhere with the deps.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

VERIFIER_ID="Qwen/Qwen3-0.6B"
MAXNEW="48"; RTT="300"; LABEL="local"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verifier-id) shift; VERIFIER_ID="${1:?}" ;;
    --max-new-tokens) shift; MAXNEW="${1:?}" ;;
    --rtt-samples) shift; RTT="${1:?}" ;;
    --label) shift; LABEL="${1:?}" ;;
    *) echo "[dist-bench] ignoring arg: $1" >&2 ;;
  esac
  shift
done

_can() { [ -n "${1:-}" ] && "$1" -c 'import grpc, torch, transformers' >/dev/null 2>&1; }
PYBIN=""
for c in "${KAKEYA_MAC_PYTHON:-}" "$repo_root/.venv-mac/bin/python3.13" \
         "$repo_root/.venv-mac/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  if _can "$c"; then PYBIN="$c"; break; fi
done
[[ -z "$PYBIN" ]] && { echo "[dist-bench] no Python with grpc+torch+transformers" >&2; exit 2; }
echo "[dist-bench] python=$PYBIN label=$LABEL" >&2

export PYTHONPATH="$repo_root:$repo_root/sdks/python"
export HF_HUB_DISABLE_PROGRESS_BARS=1

for p in $(pgrep -f demo_distributed_spec_decode 2>/dev/null || true); do kill "$p" 2>/dev/null || true; done
sleep 1
"$PYBIN" scripts/demo_distributed_spec_decode.py \
    --role proposer-node --bind 127.0.0.1:50061 --node-id bench-proposer \
    > /tmp/kakeya_bench_proposer.log 2>&1 &
PP=$!
trap 'kill "$PP" 2>/dev/null || true' EXIT
sleep 6
"$PYBIN" scripts/bench_distributed_spec_decode.py \
    --peer 127.0.0.1:50061 --label "$LABEL" \
    --verifier-id "$VERIFIER_ID" --max-new-tokens "$MAXNEW" --rtt-samples "$RTT"
