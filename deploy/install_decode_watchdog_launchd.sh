#!/usr/bin/env bash
# Install the external Primary decode watchdog as a per-user LaunchAgent.
set -euo pipefail

: "${KAKEYA_RUNTIME_REPO:?set KAKEYA_RUNTIME_REPO}"
: "${KAKEYA_RUNTIME_PYTHON:?set KAKEYA_RUNTIME_PYTHON}"

WATCHDOG_LABEL="${KAKEYA_WATCHDOG_LABEL:-ai.kakeya.decode-watchdog}"
RUNTIME_LABEL="${KAKEYA_RUNTIME_LABEL:-ai.kakeya.grpc-runtime-prefill}"
STALL_SECONDS="${KAKEYA_DECODE_STALL_SECONDS:-120}"
INTERVAL_SECONDS="${KAKEYA_WATCHDOG_INTERVAL_SECONDS:-30}"
LIVENESS_FILE="${KAKEYA_DECODE_LIVENESS_FILE:-$HOME/.kakeya/primary-decode-liveness.json}"
STATE_FILE="${KAKEYA_WATCHDOG_STATE_FILE:-$HOME/.kakeya/decode-watchdog-state.json}"
UNHEALTHY_FILE="${KAKEYA_RUNTIME_UNHEALTHY_FILE:-$HOME/.kakeya/primary-runtime-unhealthy.json}"
PLIST="$HOME/Library/LaunchAgents/$WATCHDOG_LABEL.plist"
LOG_FILE="$HOME/.kakeya/decode-watchdog.log"

mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG_FILE")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$WATCHDOG_LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$KAKEYA_RUNTIME_PYTHON</string>
    <string>$KAKEYA_RUNTIME_REPO/scripts/decode_watchdog.py</string>
    <string>--liveness-file</string><string>$LIVENESS_FILE</string>
    <string>--state-file</string><string>$STATE_FILE</string>
    <string>--unhealthy-file</string><string>$UNHEALTHY_FILE</string>
    <string>--runtime-label</string><string>$RUNTIME_LABEL</string>
    <string>--stall-seconds</string><string>$STALL_SECONDS</string>
  </array>
  <key>StartInterval</key><integer>$INTERVAL_SECONDS</integer>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$LOG_FILE</string>
  <key>StandardErrorPath</key><string>$LOG_FILE</string>
</dict></plist>
EOF

chmod 644 "$PLIST"
DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/$WATCHDOG_LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart -k "$DOMAIN/$WATCHDOG_LABEL"
echo "installed $WATCHDOG_LABEL -> $PLIST"
