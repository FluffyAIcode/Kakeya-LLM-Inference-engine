#!/usr/bin/env bash
# Self-heal the GitHub Actions runner after Mac reboot.
#
# Idempotent:
#  - if Runner.Listener is already alive: no-op
#  - else try svc.sh start (when service is installed)
#  - if service is not installed/usable, fall back to foreground run.sh
#    via nohup so bridge jobs can still execute.

set -euo pipefail

RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
LOG_DIR="${RUNNER_LOG_DIR:-$HOME/actions-runner/_diag}"
mkdir -p "$LOG_DIR"

log() { echo "[mac-bridge-recover] $*" >&2; }

if pgrep -f "Runner.Listener.*${RUNNER_DIR}" >/dev/null 2>&1; then
  log "Runner.Listener already running."
  exit 0
fi

if [ ! -x "${RUNNER_DIR}/run.sh" ]; then
  log "runner not found at ${RUNNER_DIR}; skipping."
  exit 0
fi

if [ -x "${RUNNER_DIR}/svc.sh" ] && "${RUNNER_DIR}/svc.sh" status 2>/dev/null | grep -q "installed"; then
  log "service installed; attempting svc.sh start"
  if "${RUNNER_DIR}/svc.sh" start >/dev/null 2>&1; then
    sleep 2
    if pgrep -f "Runner.Listener.*${RUNNER_DIR}" >/dev/null 2>&1; then
      log "runner started via service."
      exit 0
    fi
  fi
  log "service start did not bring up listener, falling back to run.sh"
fi

ts="$(date +%Y%m%d_%H%M%S)"
nohup "${RUNNER_DIR}/run.sh" >"${LOG_DIR}/runner-nohup-${ts}.log" 2>&1 &
sleep 2
if pgrep -f "Runner.Listener.*${RUNNER_DIR}" >/dev/null 2>&1; then
  log "runner started via nohup run.sh fallback."
  exit 0
fi

log "failed to start runner listener."
exit 1
