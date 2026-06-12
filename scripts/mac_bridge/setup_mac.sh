#!/usr/bin/env bash
# One-click Mac-side setup for the Kakeya Mac bridge.
#
# Run ON the Mac mini, from the repo root:
#
#   # Runner already installed (existing kakeya-mac-m4 host):
#   bash scripts/mac_bridge/setup_mac.sh
#
#   # Fresh Mac, install + register the Actions runner too
#   # (get the token from GitHub: Settings -> Actions -> Runners ->
#   #  New self-hosted runner -> copy the --token value):
#   bash scripts/mac_bridge/setup_mac.sh --runner-token <TOKEN> \
#       --repo-url https://github.com/<owner>/<repo>
#
#   # Also prepare M2 interactive access (Tailscale SSH):
#   bash scripts/mac_bridge/setup_mac.sh --with-tailscale
#
# Idempotent: every step checks before it changes anything. Ends with a
# bridge self-test (manifest validation + dry-run argv resolution) so a
# green exit means the next `mac-bridge/**` push will execute.
#
# See docs/design/mac-bridge-cloud-agent-access.md and
# docs/ops/mac-m4-runner-setup.md.

set -euo pipefail

RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_LABELS="self-hosted,macOS,ARM64,kakeya-mac-m4"
RUNNER_TOKEN=""
REPO_URL=""
WITH_TAILSCALE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --runner-token) RUNNER_TOKEN="$2"; shift 2 ;;
        --repo-url)     REPO_URL="$2"; shift 2 ;;
        --with-tailscale) WITH_TAILSCALE=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok()   { printf '   \033[32mOK\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33mWARN\033[0m %s\n' "$*"; }
die()  { printf '   \033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

step "1/6 Host shape"
[ "$(uname -s)" = "Darwin" ] || die "this script runs on macOS (got $(uname -s))"
[ "$(uname -m)" = "arm64" ] || die "Apple Silicon required (got $(uname -m))"
ok "macOS arm64"
PYVER="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" \
    || die "python3 not on PATH"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
    || die "Python >= 3.12 required (got ${PYVER}); brew install python@3.12"
ok "python3 ${PYVER}"
[ -f "scripts/mac_bridge/run_preset.py" ] || die "run from the repo root"
ok "repo root: $(pwd)"

step "2/6 Python dependencies (into the runner's python3)"
# The Actions runner executes jobs with the host's plain `python3`
# (see integration.yaml's install step) — NOT the .venv-mac that
# scripts/setup_mac.sh builds for interactive dev. Install into the
# same interpreter the bridge workflow will use.
if python3 -c 'import mlx.core, mlx_lm, torch, pytest' 2>/dev/null; then
    ok "mlx / mlx_lm / torch / pytest importable"
else
    warn "installing project deps into $(command -v python3) (first run takes a few minutes)"
    python3 -m pip install --upgrade pip --quiet
    python3 -m pip install --quiet -r requirements.txt
    python3 -m pip install --quiet 'mlx>=0.20' 'mlx-lm>=0.18' \
        pytest pytest-asyncio pytest-timeout
    python3 -c 'import mlx.core, mlx_lm, torch, pytest' \
        || die "deps still not importable after install"
    ok "deps installed"
fi
# Version sanity for the K3 path: transformers >= 5.0 is required by
# Gemma 4 / DFlash / current mlx-lm (requirements.txt dropped the <5
# pin; scripts/setup_mac.sh used to enforce it and broke setups with
# transformers 5.x — fixed alongside this script).
python3 - <<'PY'
import sys
from importlib.metadata import version
from packaging.version import Version
v = Version(version("transformers"))
if v < Version("4.45"):
    sys.exit(f"transformers {v} < 4.45 floor; pip install -U transformers")
print(f"   transformers {v} (K3 path wants >= 5.0: "
      f"{'OK' if v >= Version('5.0') else 'WARN — k3-* presets may fail'})")
PY
ok "dependency versions consistent with requirements.txt"

step "3/6 GitHub Actions runner (${RUNNER_DIR})"
if [ -f "${RUNNER_DIR}/.runner" ]; then
    ok "runner already configured: $(grep -o '"agentName": *"[^"]*"' "${RUNNER_DIR}/.runner" || true)"
    if "${RUNNER_DIR}/svc.sh" status 2>/dev/null | grep -q "Started"; then
        ok "runner service running"
    else
        warn "runner service not running; starting"
        (cd "${RUNNER_DIR}" && sudo ./svc.sh start)
    fi
else
    [ -n "${RUNNER_TOKEN}" ] || die "no runner at ${RUNNER_DIR}; rerun with --runner-token <TOKEN> --repo-url <URL> (GitHub: Settings->Actions->Runners->New self-hosted runner)"
    [ -n "${REPO_URL}" ] || die "--repo-url required with --runner-token"
    mkdir -p "${RUNNER_DIR}"
    cd "${RUNNER_DIR}"
    LATEST="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"
    echo "   downloading actions-runner v${LATEST} (osx-arm64)"
    curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${LATEST}/actions-runner-osx-arm64-${LATEST}.tar.gz"
    tar xzf runner.tar.gz && rm runner.tar.gz
    ./config.sh --unattended --url "${REPO_URL}" --token "${RUNNER_TOKEN}" \
        --name "kakeya-mac-m4" --labels "${RUNNER_LABELS}" --replace
    sudo ./svc.sh install && sudo ./svc.sh start
    cd - >/dev/null
    ok "runner installed + started with labels ${RUNNER_LABELS}"
fi

step "4/6 Bridge model locations (k3-* presets)"
VERIFIER="${KAKEYA_MAC_VERIFIER_PATH:-models/gemma-4-26B-A4B-it-mlx-4bit}"
FTHETA="${KAKEYA_MAC_FTHETA_DIR:-results/research/f_theta_v5_s5_sliding}"
if [ -d "${VERIFIER}" ]; then
    ok "verifier: ${VERIFIER}"
else
    warn "verifier not at '${VERIFIER}'. The k3-* presets need it."
    warn "Either place/link it there, or set the repo Actions variable"
    warn "KAKEYA_MAC_VERIFIER_PATH to its absolute path"
    warn "(GitHub: Settings -> Secrets and variables -> Actions -> Variables)."
fi
if [ -d "${FTHETA}" ]; then
    ok "f_theta: ${FTHETA}"
else
    warn "f_theta dir not at '${FTHETA}' (set KAKEYA_MAC_FTHETA_DIR var)."
fi
if [ -d "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B" ]; then
    ok "HF cache: Qwen3-0.6B pre-warmed (integration-tests preset ready)"
else
    warn "Qwen3-0.6B not in HF cache; integration-tests preset will fail."
    warn "Pre-warm with: PYTHONPATH=. python3 scripts/kakeya_prewarm.py"
fi

step "5/6 Bridge self-test (manifest validation + dry-run argv)"
TMP_MANIFEST="$(mktemp)"
python3 - "$TMP_MANIFEST" <<'PY'
import json, sys, time
json.dump({
    "schema_version": 1, "preset": "mlx-env-probe", "params": {},
    "ref": "HEAD", "requested_by": "setup-self-test",
    "nonce": f"{int(time.time())}-selftest",
}, open(sys.argv[1], "w"))
PY
PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/run_preset.py \
    --manifest "$TMP_MANIFEST" --dry-run >/dev/null \
    || die "bridge self-test failed"
rm -f "$TMP_MANIFEST"
ok "executor validates + resolves presets"
PYTHONPATH=.:sdks/python python3 -c \
    'from inference_engine.backends.mlx.env import probe_environment; print("   " + probe_environment().render())'

step "6/6 Optional: Tailscale (M2 interactive access)"
if [ "${WITH_TAILSCALE}" = "1" ]; then
    command -v brew >/dev/null || die "Homebrew required for --with-tailscale"
    command -v tailscale >/dev/null || brew install tailscale
    sudo brew services start tailscale 2>/dev/null || true
    warn "complete login + enable Tailscale SSH manually:"
    warn "  sudo tailscale up --ssh --advertise-tags=tag:kakeya-mac"
else
    ok "skipped (rerun with --with-tailscale to enable M2 interactive SSH)"
fi

printf '\n\033[1m\033[32mMac bridge ready.\033[0m Any push to mac-bridge/** (or AgentMemory/mac-bridge-*) now executes here.\n'
printf 'Smoke it from any clone with push rights:\n'
printf '  PYTHONPATH=.:sdks/python python3 scripts/mac_bridge/kakeya_mac.py run --preset mlx-env-probe --wait 600\n'
