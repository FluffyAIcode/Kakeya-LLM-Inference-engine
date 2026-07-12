#!/usr/bin/env bash
# Install the ADR 0017 prefill-compute worker as a per-user LaunchAgent.
set -euo pipefail

: "${KAKEYA_WORKER_REPO:?set KAKEYA_WORKER_REPO}"
: "${KAKEYA_WORKER_PYTHON:?set KAKEYA_WORKER_PYTHON}"
: "${KAKEYA_WORKER_MODEL:?set KAKEYA_WORKER_MODEL}"
: "${KAKEYA_WORKER_NODE_ID:?set KAKEYA_WORKER_NODE_ID}"
: "${KAKEYA_WORKER_ADVERTISE:?set KAKEYA_WORKER_ADVERTISE (host:port)}"
: "${KAKEYA_LAYER_GEOMETRY_HASH:?set KAKEYA_LAYER_GEOMETRY_HASH}"

BIND="${KAKEYA_WORKER_BIND:-0.0.0.0:53051}"
TENANT="${KAKEYA_TENANT_ID:-default}"
CACHE_GB="${KAKEYA_WORKER_CACHE_GB:-4}"
PSK_FILE="${KAKEYA_FLEET_PSK_FILE:-}"
CACHE_MODEL_ID="${KAKEYA_CACHE_MODEL_ID:-$KAKEYA_WORKER_MODEL}"
MODEL_REVISION="${KAKEYA_MODEL_REVISION:-}"
TOKENIZER_REVISION="${KAKEYA_TOKENIZER_REVISION:-}"
QUANTIZATION="${KAKEYA_CACHE_QUANTIZATION:-4bit-mlx}"
ROPE_HASH="${KAKEYA_ROPE_HASH:-}"
SINK="${KAKEYA_WORKER_SINK:-4}"
WINDOW="${KAKEYA_WORKER_WINDOW:-64}"
BLOCK_TOKENS="${KAKEYA_CACHE_BLOCK_TOKENS:-64}"
PREFILL_TPS="${KAKEYA_WORKER_PREFILL_TPS:-20}"
NETWORK="${KAKEYA_WORKER_NETWORK:-lan}"
PRIORITY="${KAKEYA_WORKER_PRIORITY:-50}"
RTT_MS="${KAKEYA_WORKER_RTT_MS:-1.0}"
PEER="${KAKEYA_WORKER_PEER:-}"
MAX_CONCURRENT_JOBS="${KAKEYA_WORKER_MAX_CONCURRENT_JOBS:-1}"
MAX_PROMPT_TOKENS="${KAKEYA_WORKER_MAX_PROMPT_TOKENS:-131072}"
LABEL="ai.kakeya.prefill-worker"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/.kakeya"
mkdir -p "$(dirname "$PLIST")" "$LOG_DIR"

psk_xml=""
if [[ -n "$PSK_FILE" ]]; then
  psk_xml="<string>--fleet-psk-file</string><string>$PSK_FILE</string>"
fi
peer_xml=""
if [[ -n "$PEER" ]]; then
  peer_xml="<string>--peer</string><string>$PEER</string>"
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$KAKEYA_WORKER_PYTHON</string>
    <string>$KAKEYA_WORKER_REPO/scripts/start_prefill_worker_node.py</string>
    <string>--node-id</string><string>$KAKEYA_WORKER_NODE_ID</string>
    <string>--bind</string><string>$BIND</string>
    <string>--advertise</string><string>$KAKEYA_WORKER_ADVERTISE</string>
    <string>--model-id</string><string>$KAKEYA_WORKER_MODEL</string>
    <string>--cache-model-id</string><string>$CACHE_MODEL_ID</string>
    <string>--model-revision</string><string>$MODEL_REVISION</string>
    <string>--tokenizer-revision</string><string>$TOKENIZER_REVISION</string>
    <string>--quantization</string><string>$QUANTIZATION</string>
    <string>--rope-hash</string><string>$ROPE_HASH</string>
    <string>--layer-geometry-hash</string><string>$KAKEYA_LAYER_GEOMETRY_HASH</string>
    <string>--tenant-id</string><string>$TENANT</string>
    <string>--cache-gb</string><string>$CACHE_GB</string>
    <string>--sink</string><string>$SINK</string>
    <string>--window</string><string>$WINDOW</string>
    <string>--block-size-tokens</string><string>$BLOCK_TOKENS</string>
    <string>--prefill-tps</string><string>$PREFILL_TPS</string>
    <string>--network</string><string>$NETWORK</string>
    <string>--priority</string><string>$PRIORITY</string>
    <string>--rtt-ms</string><string>$RTT_MS</string>
    <string>--max-concurrent-jobs</string><string>$MAX_CONCURRENT_JOBS</string>
    <string>--max-prompt-tokens</string><string>$MAX_PROMPT_TOKENS</string>
    $peer_xml
    $psk_xml
  </array>
  <key>WorkingDirectory</key><string>$KAKEYA_WORKER_REPO</string>
  <key>EnvironmentVariables</key><dict>
    <key>PYTHONPATH</key><string>$KAKEYA_WORKER_REPO:$KAKEYA_WORKER_REPO/sdks/python</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOG_DIR/prefill-worker.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/prefill-worker.log</string>
</dict></plist>
EOF

chmod 644 "$PLIST"
DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
for _ in {1..20}; do
  if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
for attempt in 1 2 3; do
  if launchctl bootstrap "$DOMAIN" "$PLIST"; then
    break
  fi
  if [[ "$attempt" -eq 3 ]]; then
    echo "failed to bootstrap $LABEL after $attempt attempts" >&2
    exit 1
  fi
  sleep 2
done
launchctl kickstart -k "$DOMAIN/$LABEL"
echo "installed $LABEL -> $PLIST"

