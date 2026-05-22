#!/usr/bin/env bash
# Set up a clean venv on macOS / Apple Silicon for this project.
#
# Why a venv: the proposer checkpoint requires transformers 4.x, which
# conflicts with newer system installs (e.g. macOS 26 ships fine with
# transformers 5.x for other tools). We never touch system Python.
#
# Idempotent: re-running upgrades the venv if needed and verifies all
# required imports succeed. Any missing or wrong-version package raises a
# hard error — there is no silent fallback.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
venv_dir="${repo_root}/.venv-mac"

ensure_macos_arm64() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "[setup_mac] this script must run on macOS, got $(uname -s)" >&2
        exit 1
    fi
    if [[ "$(uname -m)" != "arm64" ]]; then
        echo "[setup_mac] this script targets Apple Silicon (arm64), got $(uname -m)" >&2
        echo "[setup_mac] you appear to be running under Rosetta; switch to a native arm64 shell" >&2
        exit 1
    fi
}

ensure_xcode_clt() {
    if ! xcode-select -p >/dev/null 2>&1; then
        echo "[setup_mac] Xcode Command Line Tools not installed; running 'xcode-select --install'" >&2
        xcode-select --install
        echo "[setup_mac] complete the GUI prompt then re-run this script" >&2
        exit 1
    fi
}

pick_python() {
    # Prefer a 3.12 install (most stable for our deps). Fall back to
    # 3.11 or 3.13. We refuse to use 3.14+ because the wheel ecosystem for
    # transformers 4.x and dllm-hub's custom code is not yet validated
    # there.
    for cmd in python3.12 python3.11 python3.13; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "$cmd"
            return 0
        fi
    done
    echo "[setup_mac] no compatible Python found." >&2
    echo "[setup_mac] install Python 3.12 with: brew install python@3.12" >&2
    exit 1
}

ensure_venv() {
    local py="$1"
    if [[ ! -d "$venv_dir" ]]; then
        echo "[setup_mac] creating venv at $venv_dir using $py"
        "$py" -m venv "$venv_dir"
    else
        echo "[setup_mac] reusing existing venv at $venv_dir"
    fi
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    python -m pip install --upgrade pip --quiet
}

install_deps() {
    echo "[setup_mac] installing project deps"
    pip install --quiet -r "${repo_root}/requirements.txt"
    echo "[setup_mac] installing MLX backend deps"
    pip install --quiet 'mlx>=0.20' 'mlx-lm>=0.18'
}

install_dllm_stub() {
    # The dllm-hub modeling file imports `dllm` only inside an
    # `if __name__ == "__main__":` block, but transformers' static
    # check_imports flags it. Install a no-op stub.
    python -c "import site, os; \
        p = os.path.join(site.getsitepackages()[0], 'dllm'); \
        os.makedirs(p, exist_ok=True); \
        open(os.path.join(p, '__init__.py'), 'a').close()"
}

verify_imports() {
    echo "[setup_mac] verifying imports"
    python - <<'PY'
import sys
required = {
    "torch":        ("2.4", None),
    "transformers": ("4.45", "5.0"),  # hard-pin: 4.x only
    "mlx":          ("0.20", None),
    "huggingface_hub": ("0.24", None),
    "safetensors":  ("0.4", None),
    "pytest":       ("8.0", None),
}
import importlib
from packaging.version import Version
problems = []
for name, (lo, hi) in required.items():
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        problems.append(f"{name}: import failed ({e})")
        continue
    v = Version(getattr(mod, "__version__", "0"))
    if Version(lo) > v:
        problems.append(f"{name}: version {v} < required {lo}")
    if hi is not None and v >= Version(hi):
        problems.append(f"{name}: version {v} >= forbidden upper {hi}")
    print(f"  {name:20s} {v}  OK")
import platform
print(f"  Python              {platform.python_version()}  ({platform.machine()})")
if problems:
    print("\n[setup_mac] verification FAILED:")
    for p in problems: print("  -", p)
    sys.exit(1)
PY
}

main() {
    echo "[setup_mac] repo: $repo_root"
    ensure_macos_arm64
    ensure_xcode_clt
    py=$(pick_python)
    ensure_venv "$py"
    install_deps
    install_dllm_stub
    verify_imports
    echo
    echo "[setup_mac] DONE. To use:"
    echo "  source $venv_dir/bin/activate"
    echo "  ./scripts/run_platform_tests.sh --backend mlx"
}

main "$@"
