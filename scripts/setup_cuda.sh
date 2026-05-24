#!/usr/bin/env bash
# Set up a clean venv on Linux / NVIDIA CUDA for this project.
#
# Status: this script is staged. Final wheel selection (Flash-Attention 3
# vs 2, Marlin variant, FP8 KV support) depends on the GPU compute
# capability and CUDA toolkit version on the target host. Hard-errors out
# until the maintainer fills in the GPU-specific block below.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
venv_dir="${repo_root}/.venv-cuda"

ensure_linux_x86_64() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "[setup_cuda] this script must run on Linux, got $(uname -s)" >&2
        exit 1
    fi
    if [[ "$(uname -m)" != "x86_64" ]] && [[ "$(uname -m)" != "aarch64" ]]; then
        echo "[setup_cuda] unexpected arch $(uname -m)" >&2
        exit 1
    fi
}

ensure_nvidia() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[setup_cuda] nvidia-smi not found; install NVIDIA driver first." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    echo "[setup_cuda] detected compute capability: ${cap}"
    if [[ -z "$cap" ]]; then
        echo "[setup_cuda] failed to read compute capability" >&2
        exit 1
    fi
    # Save for downstream use:
    export INFERENCE_ENGINE_GPU_COMPUTE_CAP="$cap"
}

ensure_cuda_toolkit() {
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[setup_cuda] nvcc not on PATH; install CUDA Toolkit (cuda-toolkit-12-* or 13-*)" >&2
        exit 1
    fi
    nvcc --version | tail -2
}

pick_python() {
    for cmd in python3.12 python3.11 python3.13 python3.10; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "$cmd"
            return 0
        fi
    done
    echo "[setup_cuda] no compatible Python (3.10–3.13) found" >&2
    exit 1
}

ensure_venv() {
    local py="$1"
    if [[ ! -d "$venv_dir" ]]; then
        echo "[setup_cuda] creating venv at $venv_dir using $py"
        "$py" -m venv "$venv_dir"
    else
        echo "[setup_cuda] reusing venv at $venv_dir"
    fi
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    python -m pip install --upgrade pip --quiet
}

install_torch() {
    # CUDA major version from nvcc:
    cuda_ver=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+)\.([0-9]+).*/\1\2/' | head -1)
    case "$cuda_ver" in
        118|119) idx="cu118" ;;
        120|121|122) idx="cu121" ;;
        123|124) idx="cu124" ;;
        125|126) idx="cu126" ;;
        130|131|132) idx="cu126" ;;  # cu13 wheels not yet GA on PyTorch index
        *)
            echo "[setup_cuda] unsupported CUDA version $cuda_ver; aborting" >&2
            exit 1
            ;;
    esac
    echo "[setup_cuda] installing torch with index $idx"
    pip install --quiet "torch>=2.4,<3.0" --index-url "https://download.pytorch.org/whl/${idx}"
}

install_deps() {
    echo "[setup_cuda] installing project deps"
    pip install --quiet -r "${repo_root}/requirements.txt"
}

install_attention_kernel() {
    cap="$INFERENCE_ENGINE_GPU_COMPUTE_CAP"
    # Strip dot if present, e.g. 8.0 -> 80, 9.0 -> 90, 12.0 -> 120
    cap_int=$(echo "$cap" | tr -d '.' | sed -E 's/^([0-9]+).*/\1/')
    if [[ "$cap_int" -ge 90 ]]; then
        echo "[setup_cuda] Hopper or newer (cap=$cap) — Flash-Attention 3 will be used"
        echo "[setup_cuda] FA3 install path is staged; aborting until verified on a real Hopper/Blackwell host." >&2
        exit 2
    elif [[ "$cap_int" -ge 80 ]]; then
        echo "[setup_cuda] Ampere/Ada (cap=$cap) — installing Flash-Attention 2"
        pip install --quiet 'flash-attn>=2.6,<3' --no-build-isolation
    else
        echo "[setup_cuda] compute capability $cap is below 8.0; Flash-Attention not available." >&2
        echo "[setup_cuda] Speculative decoding will run but without fused-attention acceleration." >&2
        exit 3
    fi
}

install_quant_kernels() {
    pip install --quiet 'autoawq>=0.2'
}

install_dllm_stub() {
    python -c "import site, os; \
        p = os.path.join(site.getsitepackages()[0], 'dllm'); \
        os.makedirs(p, exist_ok=True); \
        open(os.path.join(p, '__init__.py'), 'a').close()"
}

clear_offline_mode() {
    if [[ -n "${HF_HUB_OFFLINE:-}" ]] || [[ -n "${TRANSFORMERS_OFFLINE:-}" ]]; then
        echo "[setup_cuda] note: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE were set;"
        echo "[setup_cuda]       unsetting for the duration of this script."
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    fi
}

probe_hf_connectivity() {
    local ep="${HF_ENDPOINT:-https://huggingface.co}"
    echo "[setup_cuda] probing HF endpoint: $ep"
    if ! curl -fsSL --max-time 15 "$ep/api/models/Qwen/Qwen3-1.7B" \
            -o /dev/null 2>/dev/null; then
        cat >&2 <<EOF
[setup_cuda] cannot reach $ep within 15 s.
[setup_cuda] remediation:
[setup_cuda]   1. check network / corporate proxy
[setup_cuda]   2. if you are in mainland China, use the mirror:
[setup_cuda]        export HF_ENDPOINT=https://hf-mirror.com
[setup_cuda]      then re-run this script.
[setup_cuda]   3. if cache is pre-populated on disk, copy it to:
[setup_cuda]        ~/.cache/huggingface/hub/
EOF
        exit 4
    fi
}

download_models() {
    # Same KAKEYA_VERIFIER_IDS extension protocol as setup_mac.sh; see
    # that script's download_models() comment block for full details.
    # CUDA users typically opt into AWQ/GPTQ 4-bit verifiers (when those
    # checkpoints exist on HuggingFace) by setting:
    #
    #   export KAKEYA_VERIFIER_IDS="Qwen/Qwen3-1.7B-AWQ"
    local extra_ids="${KAKEYA_VERIFIER_IDS:-}"
    echo "[setup_cuda] populating HF cache (~5 GB total) for the base set:"
    echo "[setup_cuda]   - Qwen/Qwen3-1.7B"
    echo "[setup_cuda]   - dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
    if [[ -n "$extra_ids" ]]; then
        echo "[setup_cuda] plus KAKEYA_VERIFIER_IDS extras: $extra_ids"
    fi
    KAKEYA_VERIFIER_IDS="$extra_ids" python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
print(f"[download] endpoint: {endpoint}")

REQUIRED = [
    "Qwen/Qwen3-1.7B",
    "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
]
extra_csv = os.environ.get("KAKEYA_VERIFIER_IDS", "").strip()
if extra_csv:
    for item in extra_csv.split(","):
        item = item.strip()
        if item and item not in REQUIRED:
            REQUIRED.append(item)

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

verify_imports() {
    echo "[setup_cuda] verifying imports"
    python - <<'PY'
import sys, importlib, platform
import importlib.metadata as md
from packaging.version import Version

required = [
    # (import_name, dist_name_override, lo, hi)
    ("torch",           None,           "2.4",   "3.0"),
    ("transformers",    None,           "4.45",  "5.0"),
    ("huggingface_hub", None,           "0.24",  None),
    ("safetensors",     None,           "0.4",   None),
    ("pytest",          None,           "8.0",   None),
    ("flash_attn",      "flash-attn",   "2.6",   "3.0"),
    ("awq",             "autoawq",      "0.2",   None),
]

def get_version(import_name, dist_name):
    candidates = [dist_name or import_name]
    canon = (dist_name or import_name).replace("_", "-")
    if canon not in candidates:
        candidates.append(canon)
    for c in candidates:
        try:
            return md.version(c)
        except md.PackageNotFoundError:
            continue
    try:
        mod = importlib.import_module(import_name)
    except Exception as e:
        raise RuntimeError(f"cannot import {import_name}: {e}")
    v = getattr(mod, "__version__", None)
    if v is None:
        raise RuntimeError(
            f"{import_name}: neither importlib.metadata nor module.__version__ "
            f"yields a version (tried: {candidates})"
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
import torch
print(f"  cuda available     {torch.cuda.is_available()}")
print(f"  cuda devices       {torch.cuda.device_count()}")
print(f"  Python             {platform.python_version()}  ({platform.machine()})")
if problems:
    print("\n[setup_cuda] verification FAILED:")
    for p in problems: print("  -", p)
    sys.exit(1)
PY
}

main() {
    echo "[setup_cuda] repo: $repo_root"
    ensure_linux_x86_64
    ensure_nvidia
    ensure_cuda_toolkit
    py=$(pick_python)
    ensure_venv "$py"
    install_torch
    install_deps
    install_attention_kernel
    install_quant_kernels
    install_dllm_stub
    verify_imports
    clear_offline_mode
    probe_hf_connectivity
    download_models
    echo
    echo "[setup_cuda] DONE. To use:"
    echo "  source $venv_dir/bin/activate"
    echo "  ./scripts/run_platform_tests.sh --backend cuda"
}

main "$@"
