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

probe_hf_connectivity() {
    # Determine endpoint: respect HF_ENDPOINT, else huggingface.co.
    local ep="${HF_ENDPOINT:-https://huggingface.co}"
    echo "[setup_mac] probing HF endpoint: $ep"
    # Probe a tiny existing-model API endpoint with a short timeout.
    if ! curl -fsSL --max-time 15 "$ep/api/models/Qwen/Qwen3-1.7B" \
            -o /dev/null 2>/dev/null; then
        cat >&2 <<EOF
[setup_mac] cannot reach $ep within 15 s.
[setup_mac] tests load real Qwen3 weights, so we must populate the
[setup_mac] HuggingFace cache before running them.
[setup_mac]
[setup_mac] remediation:
[setup_mac]   1. check VPN / corporate proxy
[setup_mac]   2. if you are in mainland China, set the official mirror:
[setup_mac]        export HF_ENDPOINT=https://hf-mirror.com
[setup_mac]      then re-run: ./scripts/setup_mac.sh
[setup_mac]   3. if you have the cache already on another machine, copy it to:
[setup_mac]        ~/.cache/huggingface/hub/
[setup_mac]      then re-run; the script will skip the download step.
EOF
        exit 4
    fi
}

download_models() {
    echo "[setup_mac] populating HF cache with Qwen3-1.7B (~3.5 GB) and"
    echo "[setup_mac] dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 (~1.5 GB)"
    python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
print(f"[download] endpoint: {endpoint}")

REQUIRED = [
    "Qwen/Qwen3-1.7B",
    "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
]
for repo in REQUIRED:
    print(f"[download] {repo} ...")
    try:
        snapshot_download(repo_id=repo, allow_patterns=None)
    except Exception as e:
        sys.stderr.write(f"\n[download] FAILED to fetch {repo}\n")
        sys.stderr.write(f"  {type(e).__name__}: {e}\n")
        sys.stderr.write(f"  endpoint was: {endpoint}\n")
        sys.exit(5)
print("[download] all required checkpoints are present")
PY
}

clear_offline_mode() {
    # If the calling shell had HF_HUB_OFFLINE=1 set (which causes
    # transformers' `local_files_only=True` to be the default), unset it
    # for this script's process so the download succeeds. Tests run by
    # `run_platform_tests.sh` will inherit the user's normal environment.
    if [[ -n "${HF_HUB_OFFLINE:-}" ]] || [[ -n "${TRANSFORMERS_OFFLINE:-}" ]]; then
        echo "[setup_mac] note: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE were set;"
        echo "[setup_mac]       unsetting for the duration of this script."
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    fi
}

verify_imports() {
    echo "[setup_mac] verifying imports"
    python - <<'PY'
import sys
import importlib
import importlib.metadata as md
import platform
from packaging.version import Version

# Each entry is (import_name, optional_distribution_name, lo, hi).
# distribution_name is the name pip uses; defaults to import_name when None.
required = [
    ("torch",           None,                  "2.4",   "3.0"),
    ("transformers",    None,                  "4.45",  "5.0"),  # hard-pin to 4.x
    ("mlx",             None,                  "0.20",  None),
    ("mlx_lm",          "mlx-lm",              "0.18",  None),
    ("huggingface_hub", None,                  "0.24",  None),
    ("safetensors",     None,                  "0.4",   None),
    ("pytest",          None,                  "8.0",   None),
]

def get_version(import_name: str, dist_name) -> str:
    """Robust version lookup.

    Order of attempts:
      1. importlib.metadata.version(dist_name or import_name) — canonical.
      2. importlib.metadata with `_`→`-` substitution (PEP 503 normalization).
      3. The imported module's __version__ attribute.

    `mlx` does NOT expose __version__ as a module attribute, so step 1 is
    required. We never silently fall back to '0' (which was the prior bug
    that misclassified mlx 0.31.1 as failing the >=0.20 floor).
    """
    candidates = [dist_name or import_name]
    # importlib.metadata accepts either '_' or '-' on most Python versions,
    # but be explicit so older 3.10 hosts work too.
    canon = (dist_name or import_name).replace("_", "-")
    if canon not in candidates:
        candidates.append(canon)
    for c in candidates:
        try:
            return md.version(c)
        except md.PackageNotFoundError:
            continue
    # Fall back to attribute (rare; required only for editable installs
    # where dist metadata is missing for some reason).
    try:
        mod = importlib.import_module(import_name)
    except Exception as e:
        raise RuntimeError(f"cannot import {import_name}: {e}")
    v = getattr(mod, "__version__", None)
    if v is None:
        raise RuntimeError(
            f"{import_name}: neither importlib.metadata nor module.__version__ "
            f"yields a version (tried distributions: {candidates})"
        )
    return v

problems = []
for import_name, dist_name, lo, hi in required:
    try:
        importlib.import_module(import_name)
    except Exception as e:
        problems.append(f"{import_name}: import failed ({e})")
        continue
    try:
        v_str = get_version(import_name, dist_name)
    except Exception as e:
        problems.append(f"{import_name}: version lookup failed ({e})")
        continue
    v = Version(v_str)
    if Version(lo) > v:
        problems.append(f"{import_name}: version {v} < required {lo}")
        continue
    if hi is not None and v >= Version(hi):
        problems.append(f"{import_name}: version {v} >= forbidden upper {hi}")
        continue
    print(f"  {import_name:20s} {v}  OK")
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
    clear_offline_mode
    probe_hf_connectivity
    download_models
    echo
    echo "[setup_mac] DONE. To use:"
    echo "  source $venv_dir/bin/activate"
    echo "  ./scripts/run_platform_tests.sh --backend mlx"
}

main "$@"
