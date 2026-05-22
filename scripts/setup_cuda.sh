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

verify_imports() {
    echo "[setup_cuda] verifying imports"
    python - <<'PY'
import sys, importlib, platform
from packaging.version import Version
required = {
    "torch":           ("2.4", "3.0"),
    "transformers":    ("4.45", "5.0"),
    "huggingface_hub": ("0.24", None),
    "safetensors":     ("0.4", None),
    "pytest":          ("8.0", None),
    "flash_attn":      ("2.6", "3.0"),
    "awq":             ("0.2", None),
}
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
    echo
    echo "[setup_cuda] DONE. To use:"
    echo "  source $venv_dir/bin/activate"
    echo "  ./scripts/run_platform_tests.sh --backend cuda"
}

main "$@"
