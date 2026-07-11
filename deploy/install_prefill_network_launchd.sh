#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="${HOME}"
VENV="${VENV:-$HOME/.venv-distwan}"
KEY_FILE="${KEY_FILE:-$HOME/.kakeya/network_api_key}"
TEMPLATE="$ROOT/deploy/launchd/ai.kakeya.prefill-network-head.plist"
TARGET="$HOME/Library/LaunchAgents/ai.kakeya.prefill-network.plist"

[ -x "$VENV/bin/python" ] || { echo "missing $VENV/bin/python" >&2; exit 2; }
[ -s "$KEY_FILE" ] || { echo "missing network API key: $KEY_FILE" >&2; exit 2; }
mkdir -p "$HOME/.kakeya" "$HOME/Library/LaunchAgents"

API_KEY="$(tr -d '\n' < "$KEY_FILE")"
python3 - "$TEMPLATE" "$TARGET" "$ROOT" "$HOME_DIR" "$VENV" "$API_KEY" <<'PY'
from pathlib import Path
import sys
source, target, repo, home, venv, key = sys.argv[1:]
text = Path(source).read_text()
for old, new in {
    "__REPO__": repo,
    "__HOME__": home,
    "__VENV__": venv,
    "__PYTHON__": f"{venv}/bin/python",
    "__API_KEY__": key,
}.items():
    text = text.replace(old, new)
Path(target).write_text(text)
PY
chmod 600 "$TARGET"

launchctl bootout "gui/$(id -u)/ai.kakeya.prefill-network" 2>/dev/null || true
pkill -f start_prefill_cache_node.py 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
launchctl kickstart -k "gui/$(id -u)/ai.kakeya.prefill-network"
echo "installed ai.kakeya.prefill-network"
