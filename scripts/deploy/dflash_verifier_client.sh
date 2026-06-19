#!/usr/bin/env bash
# Host A (verifier) side of the distributed DFlash+f_θ engine: a gemma-4 MLX
# verifier driving the remote proposer (host B) over an SSH -L tunnel, asserting
# byte-identical-to-greedy and reporting throughput + cross-host RTT.
#
# Usage:
#   bash scripts/deploy/dflash_verifier_client.sh \
#       --verifier-path /path/to/gemma-4-26B-A4B-it-mlx-4bit \
#       --drafter-id z-lab/gemma-4-26B-A4B-it-DFlash \
#       [--port 50070] [--max-new 64] [--block 4] \
#       [--ssh "-p 43350 root@107.206.71.138" --ssh-key /path/key]   # auto-open tunnel
#
# If --ssh is omitted, assumes an SSH -L <port>:localhost:<port> tunnel to host B
# is ALREADY open (the vast/portal case: open it yourself with your own creds).
set -euo pipefail

PORT=50070
VERIFIER_PATH="${KAKEYA_MAC_VERIFIER_PATH:-}"
DRAFTER_ID="${KAKEYA_MAC_DRAFTER_ID:-z-lab/gemma-4-26B-A4B-it-DFlash}"
MAXNEW=64
BLOCK=4
SSH_TARGET=""
SSH_KEY=""
PYBIN="${KAKEYA_MAC_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) shift; PORT="${1:?}" ;;
    --verifier-path) shift; VERIFIER_PATH="${1:?}" ;;
    --drafter-id) shift; DRAFTER_ID="${1:?}" ;;
    --max-new) shift; MAXNEW="${1:?}" ;;
    --block) shift; BLOCK="${1:?}" ;;
    --ssh) shift; SSH_TARGET="${1:?}" ;;
    --ssh-key) shift; SSH_KEY="${1:?}" ;;
    --python) shift; PYBIN="${1:?}" ;;
    *) echo "[verifier-client] unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root:$repo_root/sdks/python"
log() { echo "[verifier-client] $*" >&2; }
[[ -n "$VERIFIER_PATH" ]] || { log "ERROR: --verifier-path (or KAKEYA_MAC_VERIFIER_PATH) required"; exit 1; }

tunnel_pid=""
cleanup() { [[ -n "$tunnel_pid" ]] && kill "$tunnel_pid" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -n "$SSH_TARGET" ]]; then
  key_opt=""; [[ -n "$SSH_KEY" ]] && key_opt="-i $SSH_KEY"
  log "opening SSH tunnel: localhost:$PORT -> host B :$PORT ($SSH_TARGET)"
  # shellcheck disable=SC2086
  ssh $key_opt -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
      -fN -L "$PORT:localhost:$PORT" $SSH_TARGET
  tunnel_pid=$(pgrep -f "$PORT:localhost:$PORT" | head -1 || true)
  sleep 3
fi

# Connectivity probe (helps distinguish "tunnel down" from "Caddy 401").
"$PYBIN" - "$PORT" <<'PY'
import socket, sys
p = int(sys.argv[1]); s = socket.socket(); s.settimeout(5)
try:
    s.connect(("127.0.0.1", p)); print(f"[verifier-client] tunnel OK -> localhost:{p}", file=sys.stderr)
except Exception as e:
    print(f"[verifier-client] NO tunnel on localhost:{p}: {e}\n"
          f"  open one: ssh -p <ssh_port> root@<gpu_host> -L {p}:localhost:{p}", file=sys.stderr)
    sys.exit(1)
finally:
    s.close()
PY

log "running cross-host E2E (verifier @here <-> proposer @localhost:$PORT)"
exec "$PYBIN" scripts/research/k3_distributed_dflash_e2e_mac.py \
    --verifier-path "$VERIFIER_PATH" --drafter-id "$DRAFTER_ID" \
    --remote-addr "localhost:$PORT" --max-new-tokens "$MAXNEW" --block-size "$BLOCK"
