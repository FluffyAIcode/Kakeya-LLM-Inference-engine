#!/usr/bin/env bash
# Polling watcher for K3 v4 training/eval evidence.
#
# Triggers on three evidence types and prints unique markers when each
# arrives, so a parent process (the agent) can use Await pattern matching
# to react and run analysis.
#
# Markers (regex-friendly, fixed text):
#   "TRIGGER:V4A_NIAH"        — v4a NIAH eval JSON appeared
#   "TRIGGER:V4A_SWEEP"       — v4a alpha-sweep JSON appeared
#   "TRIGGER:V4B_TRAIN"       — v4b training report JSON appeared
#   "TRIGGER:V4B_NIAH"        — v4b NIAH eval JSON appeared
#   "TRIGGER:V4B_SWEEP"       — v4b alpha-sweep JSON appeared
#   "TRIGGER:WATCHER_TIMEOUT" — global wall-clock timeout
#
# Each trigger line includes the path of the file that triggered it.
#
# Usage (from any cwd, normally /workspace):
#   bash scripts/research/k3_v4_evidence_watcher.sh \
#       AgentMemory/v04-pr-k3-block-c-f-theta-v2-trainer-fix-recall-8e7f
#
# State (so the same trigger doesn't fire twice):
#   /tmp/k3_v4_watcher_state — set of already-fired triggers

set -uo pipefail

BRANCH="${1:-AgentMemory/v04-pr-k3-block-c-f-theta-v2-trainer-fix-recall-8e7f}"
INTERVAL="${WATCHER_INTERVAL_SEC:-300}"        # 5 min default
TIMEOUT="${WATCHER_TIMEOUT_SEC:-21600}"        # 6 hr default
STATE_FILE="${WATCHER_STATE_FILE:-/tmp/k3_v4_watcher_state}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
touch "$STATE_FILE"

start_ts=$(date +%s)
echo "[watcher] start $(date -u +%FT%TZ) branch=$BRANCH interval=${INTERVAL}s timeout=${TIMEOUT}s"

# Trigger patterns — file globs to scan + the marker text.
# Each line: "MARKER|GLOB"
TRIGGERS=(
    "TRIGGER:V4A_NIAH|results/research/k3_integrated_niah_*v4a*.json"
    "TRIGGER:V4A_SWEEP|results/research/k3_alpha_sweep_*v4a*.json"
    "TRIGGER:V4B_TRAIN|results/research/f_theta_v4b*.json"
    "TRIGGER:V4B_NIAH|results/research/k3_integrated_niah_*v4b*.json"
    "TRIGGER:V4B_SWEEP|results/research/k3_alpha_sweep_*v4b*.json"
)

fire_trigger() {
    local marker="$1" path="$2"
    local key="${marker}::${path}"
    if grep -Fxq -- "$key" "$STATE_FILE" 2>/dev/null; then
        return
    fi
    echo "$key" >> "$STATE_FILE"
    echo "[watcher] $(date -u +%FT%TZ) ${marker} path=${path}"
}

scan_once() {
    if ! git fetch origin "$BRANCH" >/dev/null 2>&1; then
        echo "[watcher] $(date -u +%FT%TZ) WARN git fetch failed"
        return
    fi

    local remote_head local_head
    remote_head=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "?")
    local_head=$(git rev-parse HEAD 2>/dev/null || echo "?")
    if [[ "$remote_head" != "$local_head" ]]; then
        echo "[watcher] $(date -u +%FT%TZ) new commit on remote: $remote_head"
        if git checkout "$BRANCH" >/dev/null 2>&1 && \
           git pull --ff-only origin "$BRANCH" >/dev/null 2>&1; then
            echo "[watcher] $(date -u +%FT%TZ) synced to $remote_head"
        else
            echo "[watcher] $(date -u +%FT%TZ) WARN pull failed"
        fi
    fi

    # Check trigger globs
    for entry in "${TRIGGERS[@]}"; do
        local marker="${entry%%|*}"
        local glob="${entry##*|}"
        # `compgen -G` returns matching files; `for ... in` with no match
        # would expand to literal pattern, so guard.
        local matched
        matched=$(compgen -G "$glob" 2>/dev/null || true)
        if [[ -z "$matched" ]]; then
            continue
        fi
        while IFS= read -r path; do
            fire_trigger "$marker" "$path"
        done <<< "$matched"
    done
}

# First scan happens immediately; subsequent every $INTERVAL.
while :; do
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
        echo "[watcher] $(date -u +%FT%TZ) TRIGGER:WATCHER_TIMEOUT elapsed=${elapsed}s"
        break
    fi

    scan_once

    # Idle log (every cycle, even if nothing matched) so the agent can
    # see the watcher is alive.
    echo "[watcher] $(date -u +%FT%TZ) idle elapsed=${elapsed}s next_check_in=${INTERVAL}s"
    sleep "$INTERVAL"
done
echo "[watcher] $(date -u +%FT%TZ) exit"
