#!/usr/bin/env bash
# Install a user LaunchAgent that re-checks the mac-bridge runner
# after reboot and periodically self-heals it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RECOVER_SCRIPT="${REPO_ROOT}/scripts/mac_bridge/recover_runner_after_reboot.sh"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/com.kakeya.mac-bridge-runner-autorecover.plist"
LABEL="com.kakeya.mac-bridge-runner-autorecover"
UID_NUM="$(id -u)"

mkdir -p "$PLIST_DIR"
[ -x "$RECOVER_SCRIPT" ] || chmod +x "$RECOVER_SCRIPT"

cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RECOVER_SCRIPT}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>

  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>60</integer>

  <key>StandardOutPath</key>
  <string>${HOME}/actions-runner/_diag/launchagent-autorecover.out.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/actions-runner/_diag/launchagent-autorecover.err.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID_NUM}" "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${UID_NUM}" "${PLIST_PATH}"
launchctl enable "gui/${UID_NUM}/${LABEL}" || true
launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" || true

echo "[mac-bridge-autorecover] installed: ${PLIST_PATH}"
echo "[mac-bridge-autorecover] label: ${LABEL}"
