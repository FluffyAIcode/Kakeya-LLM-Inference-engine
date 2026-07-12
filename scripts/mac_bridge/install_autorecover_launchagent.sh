#!/usr/bin/env bash
# Install a runner watchdog. `--system` installs a headless LaunchDaemon that
# runs before user login (recommended for remote Mac minis). Without it, a
# per-user LaunchAgent is installed as a fallback.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RECOVER_SCRIPT="${REPO_ROOT}/scripts/mac_bridge/recover_runner_after_reboot.sh"
SUPPORT_DIR="${HOME}/Library/Application Support/Kakeya"
INSTALLED_RECOVER_SCRIPT="${SUPPORT_DIR}/recover_runner_after_reboot.sh"
LABEL="com.kakeya.mac-bridge-runner-autorecover"
UID_NUM="$(id -u)"
USER_NAME="$(id -un)"
MODE="user"
if [ "${1:-}" = "--system" ]; then
  MODE="system"
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--system]" >&2
  exit 2
fi

[ -x "$RECOVER_SCRIPT" ] || chmod +x "$RECOVER_SCRIPT"
mkdir -p "${HOME}/actions-runner/_diag"
mkdir -p "$SUPPORT_DIR"
install -m 755 "$RECOVER_SCRIPT" "$INSTALLED_RECOVER_SCRIPT"
TMP_PLIST="$(mktemp)"
trap 'rm -f "$TMP_PLIST"' EXIT

cat >"$TMP_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${INSTALLED_RECOVER_SCRIPT}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${HOME}/actions-runner</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>${HOME}</string>
    <key>RUNNER_DIR</key><string>${HOME}/actions-runner</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>
  <key>AbandonProcessGroup</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key><false/>
  </dict>
  <key>StartInterval</key>
  <integer>60</integer>

  <key>StandardOutPath</key>
  <string>${HOME}/actions-runner/_diag/launchagent-autorecover.out.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/actions-runner/_diag/launchagent-autorecover.err.log</string>
</dict>
</plist>
EOF

if [ "$MODE" = "system" ]; then
  PLIST_PATH="/Library/LaunchDaemons/${LABEL}.plist"
  # UserName lets the runner retain access to its registration, models and
  # workspace while the daemon itself starts before GUI login.
  /usr/libexec/PlistBuddy -c "Add :UserName string ${USER_NAME}" "$TMP_PLIST"
  sudo install -o root -g wheel -m 644 "$TMP_PLIST" "$PLIST_PATH"
  sudo launchctl bootout system "$PLIST_PATH" >/dev/null 2>&1 || true
  sudo launchctl bootstrap system "$PLIST_PATH"
  sudo launchctl enable "system/${LABEL}" || true
  sudo launchctl kickstart -k "system/${LABEL}" || true
  DOMAIN="system"
else
  PLIST_DIR="${HOME}/Library/LaunchAgents"
  PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
  mkdir -p "$PLIST_DIR"
  install -m 644 "$TMP_PLIST" "$PLIST_PATH"
  launchctl bootout "gui/${UID_NUM}" "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/${UID_NUM}" "$PLIST_PATH"
  launchctl enable "gui/${UID_NUM}/${LABEL}" || true
  launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" || true
  DOMAIN="gui/${UID_NUM}"
fi

echo "[mac-bridge-autorecover] installed: ${PLIST_PATH}"
echo "[mac-bridge-autorecover] label: ${DOMAIN}/${LABEL}"
