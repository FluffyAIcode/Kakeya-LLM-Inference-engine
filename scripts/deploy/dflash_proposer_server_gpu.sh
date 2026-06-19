#!/usr/bin/env bash
# Deploy the remote DFlash+f_θ proposer (ADR 0009 §4 F3) on a CUDA host (host B).
#
# One command: ensure transformers 5.x, fetch the gemma-4 verifier (for its
# embedding) + DFlash drafter to a (RAM-disk) HF cache, and serve the
# DFlashProposerService. A gemma-4 MLX verifier on host A drives it via
# RemoteDFlashProposer (see scripts/deploy/dflash_verifier_client.sh).
#
# Usage:
#   bash scripts/deploy/dflash_proposer_server_gpu.sh \
#       [--port 50070] [--hf-cache /dev/shm/hf] \
#       [--verifier-id google/gemma-4-26B-A4B-it] \
#       [--drafter-id z-lab/gemma-4-26B-A4B-it-DFlash] \
#       [--f-theta-dir results/research/f_theta_v5_s5_sliding] \
#       [--python /path/to/venv/python] [--foreground]
#
# IMPORTANT — pick a port the vast/portal Caddy does NOT own. Portal ports
# (1111/8080/8384/6006 on vast) are Caddy-proxied (HTTP 401 to gRPC); use a
# plain high port like 50070 and reach it from host A over an SSH -L tunnel.
set -euo pipefail

PORT=50070
HF_CACHE="/dev/shm/hf"          # RAM-disk: the gemma-4 base is ~52GB, > many root disks
VERIFIER_ID="google/gemma-4-26B-A4B-it"
DRAFTER_ID="z-lab/gemma-4-26B-A4B-it-DFlash"
FTHETA_DIR="results/research/f_theta_v5_s5_sliding"
PYBIN="${KAKEYA_GPU_PYTHON:-python3}"
FOREGROUND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) shift; PORT="${1:?}" ;;
    --hf-cache) shift; HF_CACHE="${1:?}" ;;
    --verifier-id) shift; VERIFIER_ID="${1:?}" ;;
    --drafter-id) shift; DRAFTER_ID="${1:?}" ;;
    --f-theta-dir) shift; FTHETA_DIR="${1:?}" ;;
    --python) shift; PYBIN="${1:?}" ;;
    --foreground) FOREGROUND=1 ;;
    *) echo "[deploy-gpu] unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
export HF_HOME="$HF_CACHE"
export PYTHONPATH="$repo_root:$repo_root/sdks/python"

log() { echo "[deploy-gpu] $*" >&2; }

log "repo=$repo_root python=$PYBIN port=$PORT hf_cache=$HF_CACHE"
[[ -s "$FTHETA_DIR/f_theta_weights.pt" ]] || {
  log "ERROR: $FTHETA_DIR/f_theta_weights.pt missing (git lfs pull it, or scp from host A)"; exit 1; }

# gemma-4 needs transformers 5.x; the DFlash drafter + f_θ are framework-custom.
if ! "$PYBIN" -c 'import transformers,sys; sys.exit(0 if transformers.__version__>="5" else 1)' 2>/dev/null; then
  log "installing transformers>=5.0 (gemma-4 requires it)"
  "$PYBIN" -m pip install -q "transformers>=5.0,<6.0"
fi

log "fetching weights into $HF_CACHE (gemma-4 verifier embed + DFlash drafter)"
"$PYBIN" - "$VERIFIER_ID" "$DRAFTER_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download
v, d = sys.argv[1], sys.argv[2]
snapshot_download(v, allow_patterns=["*.json","*.model","tokenizer*","*.safetensors"])
snapshot_download(d)
print("[deploy-gpu] weights ready", file=sys.stderr)
PY

cmd=("$PYBIN" scripts/research/k3_dflash_proposer_server.py
     --verifier-id "$VERIFIER_ID" --drafter-id "$DRAFTER_ID"
     --f-theta-dir "$FTHETA_DIR" --bind "0.0.0.0:$PORT")

if [[ "$FOREGROUND" == "1" ]]; then
  log "serving in foreground on 0.0.0.0:$PORT"
  exec "${cmd[@]}"
fi
for p in $(pgrep -f k3_dflash_proposer_server 2>/dev/null || true); do kill "$p" 2>/dev/null || true; done
sleep 1
nohup "${cmd[@]}" > /tmp/dflash_proposer_server.log 2>&1 &
log "server pid $! -> /tmp/dflash_proposer_server.log (loading gemma-4 onto the GPU…)"
log "host A connects via: ssh -p <ssh_port> root@<gpu_host> -L $PORT:localhost:$PORT"
