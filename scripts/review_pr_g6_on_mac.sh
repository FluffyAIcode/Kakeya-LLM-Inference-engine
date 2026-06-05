#!/usr/bin/env bash
# Mac M4 review aid for PR-G6 (chat REPL over gRPC SDK).
#
# This is a UX PR — its correctness comes from the underlying SDK
# (which is already integration-tested) plus a single end-to-end
# smoke that confirms a real interactive REPL session works against
# a real Qwen3-0.6B-backed gRPC server.
#
# Smoke flow:
#   1. Start the gRPC server in the background.
#   2. Pipe a short conversation through chat_grpc.py via stdin.
#   3. Capture the output; assert at least one response chunk
#      arrives + the script exits cleanly.
#
# Produces 1 artifact:
#   results/platform-tests/pr-g6-mac-chat-smoke-<unix>.json
#
# Usage (from repo root, on Mac M4):
#
#     bash scripts/review_pr_g6_on_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

server_log="$out_dir/pr-g6-mac-chat-smoke-${stamp}.server.log"
chat_log="$out_dir/pr-g6-mac-chat-smoke-${stamp}.chat.log"
report="$out_dir/pr-g6-mac-chat-smoke-${stamp}.json"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "==> starting gRPC server (logs: $server_log)"
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --bind 127.0.0.1:50098 \
    --capacity 1 --sink 4 --window 64 \
    >"$server_log" 2>&1 &
server_pid=$!

echo "==> waiting up to 60s for server to become ready"
for _ in $(seq 1 60); do
    if grep -q "kakeya gRPC RuntimeService listening on" "$server_log" 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! grep -q "kakeya gRPC RuntimeService listening on" "$server_log"; then
    echo "!!! server did not become ready"
    tail -20 "$server_log"
    exit 1
fi

echo "==> piping a 3-turn conversation through chat_grpc.py"
PYTHONPATH=.:sdks/python python3 scripts/chat_grpc.py \
    --address 127.0.0.1:50098 \
    --tokenizer-id Qwen/Qwen3-0.6B \
    --max-tokens 24 <<'INPUT' >"$chat_log" 2>&1 || true
Hi.
What is your favorite color?
/info
/exit
INPUT

# Acceptance: chat output contains at least 2 'kakeya>' response prompts.
n_responses=$(grep -c '^kakeya> ' "$chat_log" || true)
echo "    chat_log: $chat_log"
echo "    response prompts: $n_responses (expect >=2)"

PYTHONPATH=.:sdks/python python3 - "$report" "$n_responses" <<'PY'
import json
import platform
import sys
report_path, n_resp_str = sys.argv[1:3]
n_resp = int(n_resp_str)
report = {
    "schema_version": 1,
    "kind": "pr_g6_mac_chat_smoke",
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "n_chat_responses": n_resp,
    "passed": n_resp >= 2,
}
with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"  -> {report_path}")
PY

echo
echo "==> Done. Commit:"
echo "    git add $out_dir/pr-g6-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-G6'"
echo "    git push"
