#!/usr/bin/env bash
# Diagnose why GitHub Actions jobs are queued for kakeya-mac-m4.
set -uo pipefail

RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
LABEL="com.kakeya.mac-bridge-runner-autorecover"
failed=0

check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf 'PASS  %s\n' "$name"
  else
    printf 'FAIL  %s\n' "$name"
    failed=1
  fi
}

echo "runner_dir=$RUNNER_DIR"
check "runner registration (.runner)" test -s "$RUNNER_DIR/.runner"
check "runner executable" test -x "$RUNNER_DIR/run.sh"
check "github.com connectivity" curl -fsSI --max-time 10 https://github.com/
check "Runner.Listener process" sh -c \
  "pgrep -f '$RUNNER_DIR/bin/Runner.Listener' >/dev/null || pgrep -f 'Runner.Listener run' >/dev/null"
check "disk has >=10 GiB free" sh -c \
  "[ \$(df -k '$RUNNER_DIR' | awk 'NR==2 {print \$4}') -ge 10485760 ]"

echo
echo "official service:"
"$RUNNER_DIR/svc.sh" status 2>&1 || true
echo
echo "user watchdog:"
launchctl print "gui/$(id -u)/$LABEL" 2>&1 | head -30 || true
echo
echo "system watchdog:"
sudo -n launchctl print "system/$LABEL" 2>&1 | head -30 || true
echo
echo "recent runner diagnostics:"
ls -t "$RUNNER_DIR"/_diag/Runner_*.log \
  "$RUNNER_DIR"/_diag/runner-nohup-*.log 2>/dev/null \
  | head -2 | while read -r log; do
      echo "=== $log ==="
      tail -20 "$log"
    done

exit "$failed"

