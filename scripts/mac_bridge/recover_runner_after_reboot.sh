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
LOCK_DIR="${TMPDIR:-/tmp}/kakeya-runner-recover.lock"

log() { echo "[mac-bridge-recover] $*" >&2; }

listener_running() {
  pgrep -f "${RUNNER_DIR}/bin/Runner.Listener" >/dev/null 2>&1 \
    || pgrep -f "Runner.Listener run" >/dev/null 2>&1
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
    log "another recovery attempt is active (pid=$lock_pid)."
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
echo "$$" >"$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT

if listener_running; then
  log "Runner.Listener already running."
  exit 0
fi

if [ ! -x "${RUNNER_DIR}/run.sh" ]; then
  log "runner not found at ${RUNNER_DIR}."
  exit 1
fi
if [ ! -s "${RUNNER_DIR}/.runner" ]; then
  log "runner exists but is not registered (.runner missing)."
  exit 1
fi

if ! curl -fsSI --max-time 10 https://github.com/ >/dev/null 2>&1; then
  log "github.com is unreachable; leaving runner stopped for the next retry."
  exit 1
fi

if [ -x "${RUNNER_DIR}/svc.sh" ]; then
  status="$("${RUNNER_DIR}/svc.sh" status 2>&1 || true)"
  # Do not match the phrase "not installed".
  if echo "$status" | grep -qi "installed" \
      && ! echo "$status" | grep -qi "not installed"; then
    log "official service is installed; attempting svc.sh start"
    if [ "$(id -u)" -eq 0 ]; then
      "${RUNNER_DIR}/svc.sh" start >/dev/null 2>&1 || true
    elif sudo -n true >/dev/null 2>&1; then
      sudo -n "${RUNNER_DIR}/svc.sh" start >/dev/null 2>&1 || true
    fi
    for _ in 1 2 3 4 5; do
      sleep 2
      if listener_running; then
        log "runner started via official service."
        exit 0
      fi
    done
    log "official service did not bring up listener; using direct fallback."
  fi
fi

ts="$(date +%Y%m%d_%H%M%S)"
(
  cd "$RUNNER_DIR"
  nohup ./run.sh >"${LOG_DIR}/runner-nohup-${ts}.log" 2>&1 &
)
for _ in 1 2 3 4 5 6 7 8; do
  sleep 2
  if listener_running; then
    log "runner started via direct run.sh fallback."
    exit 0
  fi
done

latest_log="$(ls -t "$LOG_DIR"/runner-nohup-*.log 2>/dev/null | head -1 || true)"
if [ -n "$latest_log" ]; then
  log "last runner output:"
  tail -20 "$latest_log" >&2 || true
fi

log "failed to start runner listener."
exit 1
