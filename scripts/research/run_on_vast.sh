#!/usr/bin/env bash
# Linux / NVIDIA (vast.ai) runner — generic GPU provisioning + Python
# script invocation for project research scripts. Originally written
# for the ADR 0011 toy (PR-R1c, since closed) but kept on main as
# reusable infrastructure: PR-K1.E and beyond use this same runner
# for their vast.ai-side reviewer aids.
#
# Compatibility: developed/validated on an H200 (compute capability
# 9.0, CUDA 13.0); also works on H100 / A100 / L4 / A10G with the
# same wheel channel (cu128 forward-compatible). Run on the vast
# host with the repo synced there.
#
# It is intentionally self-contained and idempotent:
#
#   1. Creates / reuses a venv at .venv-vast.
#   2. Installs a CUDA-enabled torch + transformers stack (pinned to the
#      project's transformers 4.x line — see requirements.txt).
#   3. Verifies the GPU is visible to torch.
#   4. Runs scripts/research/cross_attn_toy_prototype.py once, forwarding
#      every argument after the script name straight through to the toy.
#
# The toy's default model (google/gemma-3-1b-it) is gated on HuggingFace.
# Export HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) before running; the script
# refuses to start without one rather than failing 401 mid-download
# (ADR 0008 §6.2: no silent fallback).
#
# Usage (run ON the vast host, repo synced there):
#
#   # one full run, defaults (2000 steps, capacity-bumped):
#   HF_TOKEN=hf_xxx bash scripts/research/run_on_vast.sh \
#       --output results/research/cross_attn_toy_vast_full.json
#
#   # just provision the venv (used by review_pr_r1c_on_vast.sh before
#   # it launches two runs in parallel):
#   HF_TOKEN=hf_xxx bash scripts/research/run_on_vast.sh --setup-only

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
venv_dir="${repo_root}/.venv-vast"

# Default torch CUDA wheel channel. cu128/cu126 wheels run fine against
# newer drivers (forward-compatible); override with KAKEYA_TORCH_INDEX
# if the host needs a different channel.
TORCH_INDEX="${KAKEYA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

log() { echo "[run_on_vast] $*" >&2; }

ensure_token() {
    if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        cat >&2 <<'EOF'
[run_on_vast] HF_TOKEN is not set, but the toy's default model
[run_on_vast] (google/gemma-3-1b-it) is GATED on HuggingFace. Export a
[run_on_vast] token that has accepted the Gemma license:
[run_on_vast]     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
[run_on_vast] then re-run. (ADR 0008 §6.2 forbids silent fallbacks.)
EOF
        exit 4
    fi
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
}

ensure_gpu_present() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi not found — this script targets a CUDA GPU host."
        exit 1
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
        --format=csv,noheader >&2
}

pick_python() {
    for cmd in python3.12 python3.11 python3.13 python3.10 python3; do
        if command -v "$cmd" >/dev/null 2>&1; then echo "$cmd"; return 0; fi
    done
    log "no compatible Python (3.10-3.13) found"; exit 1
}

ensure_venv() {
    local py="$1"
    if [[ ! -d "$venv_dir" ]]; then
        log "creating venv at $venv_dir using $py"
        "$py" -m venv "$venv_dir"
    else
        log "reusing venv at $venv_dir"
    fi
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    python -m pip install --upgrade pip --quiet
}

install_stack() {
    if python -c "import torch" 2>/dev/null && \
       python -c "import transformers" 2>/dev/null; then
        log "torch + transformers already importable; skipping install"
        return 0
    fi
    log "installing CUDA torch from $TORCH_INDEX"
    pip install --quiet "torch>=2.4,<3.0" --index-url "$TORCH_INDEX"
    log "installing transformers/accelerate stack (4.x pin)"
    pip install --quiet \
        "transformers>=4.45,<5.0" \
        "accelerate>=0.34" \
        "safetensors>=0.4" \
        "huggingface_hub>=0.24" \
        "numpy>=1.26"
}

verify_torch_cuda() {
    python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available()
print(f"[run_on_vast] torch={torch.__version__} cuda_available={ok} "
      f"cuda={torch.version.cuda}", file=sys.stderr)
if ok:
    print(f"[run_on_vast] device0={torch.cuda.get_device_name(0)}",
          file=sys.stderr)
else:
    print("[run_on_vast] WARNING: torch cannot see the GPU; the toy will "
          "fall back to CPU and be extremely slow.", file=sys.stderr)
    sys.exit(5)
import transformers
print(f"[run_on_vast] transformers={transformers.__version__}",
      file=sys.stderr)
PY
}

provision() {
    ensure_gpu_present
    local py; py="$(pick_python)"
    ensure_venv "$py"
    install_stack
    verify_torch_cuda
}

main() {
    ensure_token

    local setup_only=0
    local fwd=()
    for arg in "$@"; do
        if [[ "$arg" == "--setup-only" ]]; then
            setup_only=1
        else
            fwd+=("$arg")
        fi
    done

    provision

    if [[ "$setup_only" == "1" ]]; then
        log "setup-only complete; venv ready at $venv_dir"
        return 0
    fi

    # Pick the Python script to run. KAKEYA_VAST_SCRIPT env var is
    # the explicit override; reviewer aids set it to point at their
    # own runner (e.g., scripts/research/k1e_niah_validation.py).
    # The default keeps backward compatibility with the original
    # ADR 0011 toy reviewer scripts (PR-R1c, since closed) so
    # historical reproducibility is preserved.
    local script="${KAKEYA_VAST_SCRIPT:-scripts/research/cross_attn_toy_prototype.py}"
    if [[ ! -f "$script" ]]; then
        log "script $script not found in repo (cwd=$PWD); pass "
        log "KAKEYA_VAST_SCRIPT=path/to/your_script.py to override"
        exit 6
    fi

    log "launching $script: ${fwd[*]:-<defaults>}"
    PYTHONPATH=".:sdks/python" python "$script" \
        "${fwd[@]}"
}

main "$@"
