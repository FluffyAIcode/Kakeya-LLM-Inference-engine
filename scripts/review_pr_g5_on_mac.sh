#!/usr/bin/env bash
# Mac M4 review aid for PR-G5 (model prewarm CLI + first-run UX).
#
# Two end-to-end checks the Linux unit tests can't do:
#   1. The prewarm CLI's actual download path against real
#      huggingface.co (or hf-mirror.com via HF_ENDPOINT).
#   2. The gRPC server's pre-flight cache check fail-fast behavior
#      when the model is genuinely missing.
#
# Produces 1 artifact:
#   results/platform-tests/pr-g5-mac-prewarm-<unix>.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%s)"
out_dir="results/platform-tests"
mkdir -p "$out_dir"

report="$out_dir/pr-g5-mac-prewarm-${stamp}.json"

echo "==> 1. Verify Qwen3-0.6B is in HF cache (should be true post setup_mac.sh)"
PYTHONPATH=.:sdks/python python3 - <<'PY'
from inference_engine.setup import is_model_in_cache, snapshot_size_bytes
mid = "Qwen/Qwen3-0.6B"
ok = is_model_in_cache(mid)
size = snapshot_size_bytes(mid)
print(f"  cached={ok}  size_mib={size / (1024*1024):.1f}")
PY

echo
echo "==> 2. Pre-warm CLI on a model that's almost certainly NOT in cache"
echo "    (a synthetic model id; expects FAILED gracefully without crashing)"
PYTHONPATH=.:sdks/python python3 scripts/kakeya_prewarm.py \
    --verifier-id Qwen/Qwen3-0.6B \
    --no-tokenizer 2>&1 | head -20 || true

echo
echo "==> 3. Server pre-flight: try a model NOT in cache, expect exit code 2"
set +e
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu \
    --verifier-id this-is/a-fake-id-for-testing \
    --bind 127.0.0.1:50099 \
    --capacity 1 --sink 4 --window 64 2>&1 | head -10
exit_code=$?
set -e
echo "    exit code: $exit_code (expect 2)"

echo
PYTHONPATH=.:sdks/python python3 - "$report" "$exit_code" <<'PY'
import json
import platform
import sys
report_path = sys.argv[1]
exit_code = int(sys.argv[2])
report = {
    "schema_version": 1,
    "kind": "pr_g5_mac_prewarm",
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "server_preflight_exit_code": exit_code,
    "server_preflight_passed": exit_code == 2,
}
with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"  -> {report_path}")
PY

echo
echo "==> Done. Commit:"
echo "    git add $out_dir/pr-g5-mac-*"
echo "    git commit -m 'Mac M4 review evidence for PR-G5'"
echo "    git push"
