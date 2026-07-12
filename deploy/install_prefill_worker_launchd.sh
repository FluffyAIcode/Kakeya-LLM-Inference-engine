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
LABEL="ai.kakeya.prefill-worker"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/.kakeya"
mkdir -p "$(dirname "$PLIST")" "$LOG_DIR"

psk_xml=""
if [[ -n "$PSK_FILE" ]]; then
  psk_xml="<string>--fleet-psk-file</string><string>$PSK_FILE</string>"
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

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $PLIST"

