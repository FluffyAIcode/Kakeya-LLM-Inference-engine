"""K3 Mac M4 verifier setup — get a working 4-bit MLX-quantized
Gemma 4 26B-A4B-it locally.

**Default path (added 2026-06-09)**: download the published PLE-safe
community variant `FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit` rather
than self-quantize. Rationale below.

**Why we don't self-quantize anymore**

The original version of this script called `mlx_lm.convert` to
4-bit-quantize `google/gemma-4-26B-A4B-it` directly. That path
is broken on mlx-lm 0.31.3 (and the few releases around it) due
to FIVE interlocking bugs in mlx-lm / mlx-vlm's handling of
Gemma 4's PLE (Per-Layer Embedding) architecture + MoE
(SwitchLinear) expert layers (verified 2026-06-09 via GitHub
issue ml-explore/mlx-lm#1123 and the FakeRocket543/mlx-gemma4
write-up). Surface symptoms:

  * `AttributeError: 'list' object has no attribute 'keys'` —
    the MoE-not-included-in-quantization bug (#4)
  * Or quantize succeeds but model emits degenerate token-
    repetition garbage like `ionoxffionoxff...` — the PLE-layers-
    quantized bug (#2)

Both are upstream issues in mlx-lm/mlx-vlm, NOT in this repo or
the K3 design. The mlx-vlm 0.4.4 release reportedly fixes all
five bugs but the `mlx_lm.convert` Python API in mlx-lm 0.31.3
hasn't yet inherited the fix.

**The fix**: use the published community 4-bit variant.
`FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit` is a PLE-safe
quantization that:

  * Quantizes only the large `nn.Linear` and `SwitchLinear`
    (MoE expert) layers
  * Keeps `ScaledLinear` (PLE), `ScaledEmbedding`, vision
    encoder, all norms and scalars in bf16
  * 16.4 GB on disk (vs the ~13 GB an unsafe naive quant
    produces — MoE expert layers correctly quantized adds
    ~3.4 GB)
  * Apache 2 license (per Gemma 4 family upstream)

Mac M4 24 GB fit at 16.4 GB model:

  * model weights:   ~16.4 GB
  * KV cache:        sink+window=4+64 ≈ negligible at K1 setup
  * activations:     ~1-2 GB transient (depends on context)
  * MPS allocator:   1.3-1.5x overhead (PyTorch MPS quirks)
  * DFlash drafter:  ~0.8 GB if loaded
  * --------
  * estimated peak:  ~22-26 GB at 512-prompt smoke
  * Mac M4 24 GB:    feasible at 512-prompt baseline; 16k context
                     likely OK; 64k+ probably triggers macOS
                     unified-memory swap

**This script supports two modes**:

  * `--mode download` (DEFAULT): fetch
    `FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit` via
    `huggingface_hub.snapshot_download` to the local output
    directory. ~5-15 min depending on network. No quantize
    compute needed.
  * `--mode self-quantize`: falls back to the original
    `mlx_lm.convert` path, with `--upstream-fix` flag to switch
    on `trust_remote_code=True` and any other workaround we
    discover empirically. **Likely to fail** on mlx-lm 0.31.3
    per the bugs above; preserved for debugging / when a future
    mlx-lm release fixes the upstream bugs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["download", "self-quantize"],
        default="download",
        help="'download' (DEFAULT) fetches the published PLE-safe community "
             "MLX 4-bit variant. 'self-quantize' falls back to mlx_lm.convert "
             "(likely fails on mlx-lm 0.31.3 due to upstream bugs; preserved "
             "for diagnostic purposes and future-mlx-lm-release recovery).",
    )
    ap.add_argument(
        "--source-download",
        default="FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit",
        help="HF repo id of the PLE-safe community 4-bit variant. Verified "
             "2026-06-09 via the FakeRocket543/mlx-gemma4 write-up.",
    )
    ap.add_argument(
        "--source-quantize",
        default="google/gemma-4-26B-A4B-it",
        help="HF repo id for self-quantize (the bf16 source).",
    )
    ap.add_argument(
        "--output", default="models/gemma-4-26B-A4B-it-mlx-4bit",
        help="Local directory where the model will be saved.",
    )
    ap.add_argument(
        "--q-bits", type=int, default=4, choices=[2, 3, 4, 6, 8],
        help="Quantization bits (only used in --mode self-quantize). "
             "Default 4. Note: PLE-safe quantization skips PLE layers "
             "regardless of this setting.",
    )
    ap.add_argument(
        "--q-group-size", type=int, default=64,
        help="Quantization group size (only used in --mode self-quantize).",
    )
    ap.add_argument(
        "--manifest", default=None,
        help="JSON manifest output path. Default: <output>/k3_setup_manifest.json",
    )
    return ap.parse_args()


def _check_hf_login() -> None:
    """Confirm an HF token is available. Both source models (gated Gemma 4
    + the community variant which mirrors Gemma's gating) need auth."""
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    if token is None:
        try:
            r = subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                user = r.stdout.strip()
                print(
                    f"[k3-setup] HF login: {user}",
                    file=sys.stderr,
                )
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        print(
            "ERROR: no HuggingFace credentials detected. Either:\n"
            "    huggingface-cli login\n"
            "or set HF_TOKEN environment variable.\n"
            "Gemma 4 family is gated; auth is required even for the "
            "community PLE-safe variant.",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[k3-setup] HF token: present (env)", file=sys.stderr)


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _do_download(args: argparse.Namespace, output: Path) -> int:
    """Fetch the PLE-safe community variant via huggingface_hub."""
    print(
        f"[k3-setup] mode=download  source={args.source_download}",
        file=sys.stderr,
    )
    print(
        f"[k3-setup] destination: {output}",
        file=sys.stderr,
    )
    print(
        "[k3-setup] downloading PLE-safe 4-bit MLX variant; ~5-15 min "
        "depending on network. The variant is ~16.4 GB.",
        file=sys.stderr, flush=True,
    )
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub not installed. Install with:\n"
            "    pip install --upgrade huggingface_hub",
            file=sys.stderr,
        )
        return 4

    t_start = time.perf_counter()
    try:
        snapshot_download(
            repo_id=args.source_download,
            local_dir=str(output),
            # Keep file structure flat (don't create symlinks to
            # ~/.cache/huggingface; we want the model self-contained
            # in `output`).
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(
            f"[k3-setup] download FAILED: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 5
    elapsed = time.perf_counter() - t_start

    size_bytes = _directory_size_bytes(output)
    size_gb = size_bytes / 1e9
    print(
        f"[k3-setup] DONE. Size on disk: {size_gb:.2f} GB. Elapsed: {elapsed/60:.1f} min.",
        file=sys.stderr,
    )

    manifest_path = (
        Path(args.manifest) if args.manifest is not None
        else output / "k3_setup_manifest.json"
    )
    manifest = {
        "kind": "k3_setup_manifest",
        "schema_version": 2,  # v2: introduces 'mode' field
        "mode": "download",
        "source": args.source_download,
        "output": str(output),
        "size_bytes": size_bytes,
        "size_gb": size_gb,
        "elapsed_seconds": elapsed,
        "host": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
        },
        "ple_safe_quantization": True,
        "rationale": (
            "Self-quantize via mlx_lm.convert is broken on mlx-lm 0.31.3 "
            "for Gemma 4 26B-A4B MoE due to 5 interlocking upstream bugs "
            "(see ml-explore/mlx-lm#1123). The community variant "
            f"{args.source_download} is PLE-safe and produces correct output."
        ),
        "instructions_for_smoke": (
            f"Pass --verifier-path {output} to "
            "scripts/research/k3_feasibility_smoke.py"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[k3-setup] manifest -> {manifest_path}", file=sys.stderr)
    return 0


def _do_self_quantize(args: argparse.Namespace, output: Path) -> int:
    """Original self-quantize path. Likely fails on mlx-lm 0.31.3 due to
    upstream Gemma 4 MoE bugs; preserved for diagnostic purposes."""
    print(
        f"[k3-setup] mode=self-quantize  source={args.source_quantize}",
        file=sys.stderr,
    )
    print(
        "[k3-setup] WARNING: this path is known to fail on mlx-lm 0.31.3 for "
        "Gemma 4 26B-A4B MoE. See ml-explore/mlx-lm#1123. Use --mode download "
        "for a working PLE-safe variant. Continuing anyway for diagnostic "
        "purposes — full traceback will be captured to "
        f"{output}.quantize_traceback.txt on failure.",
        file=sys.stderr,
    )

    try:
        import mlx_lm  # type: ignore
    except ImportError as e:
        print(
            "ERROR: mlx-lm not installed. On Mac mini:\n"
            "    pip install --upgrade mlx-lm",
            file=sys.stderr,
        )
        return 1
    version = getattr(mlx_lm, "__version__", "0.0.0")
    print(f"[k3-setup] mlx-lm version: {version}", file=sys.stderr)

    print(
        f"[k3-setup] starting mlx_lm.convert (q_bits={args.q_bits}, "
        f"q_group_size={args.q_group_size})",
        file=sys.stderr,
    )
    t_start = time.perf_counter()
    from mlx_lm import convert  # type: ignore
    try:
        convert(
            hf_path=args.source_quantize,
            mlx_path=str(output),
            quantize=True,
            q_bits=args.q_bits,
            q_group_size=args.q_group_size,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        # Capture full traceback to a file for diagnosis.
        tb_path = Path(f"{str(output)}.quantize_traceback.txt")
        tb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(
                f"k3_quantize_for_mac.py self-quantize failure\n"
                f"=========================================\n\n"
                f"Source: {args.source_quantize}\n"
                f"Destination: {output}\n"
                f"q_bits: {args.q_bits}\n"
                f"q_group_size: {args.q_group_size}\n"
                f"mlx-lm version: {version}\n"
                f"Elapsed before failure: {elapsed/60:.1f} min\n\n"
                f"Exception type: {type(e).__name__}\n"
                f"Exception args: {e.args!r}\n\n"
            )
            traceback.print_exc(file=f)
        print(
            f"[k3-setup] FAIL: mlx_lm.convert raised {type(e).__name__}: {e}\n"
            f"[k3-setup] full traceback -> {tb_path}\n"
            f"[k3-setup] If this is the known Gemma 4 MoE bug "
            "(`AttributeError: 'list' object has no attribute 'keys'` or "
            "degenerate output after success), use --mode download instead.",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 4
    elapsed = time.perf_counter() - t_start

    if not output.exists():
        print(
            f"[k3-setup] FAIL: convert returned but output {output} does not exist",
            file=sys.stderr,
        )
        return 5
    size_bytes = _directory_size_bytes(output)
    size_gb = size_bytes / 1e9
    print(
        f"[k3-setup] DONE. Size on disk: {size_gb:.2f} GB. Elapsed: {elapsed/60:.1f} min.",
        file=sys.stderr,
    )
    if size_gb < 14:
        print(
            f"WARNING: quantized model is {size_gb:.1f} GB, smaller than "
            "the PLE-safe 16.4 GB reference. This MAY indicate the MoE "
            "expert layers (SwitchLinear) were silently SKIPPED from "
            "quantization (mlx-lm bug #4 — manifests as a smaller model "
            "with degenerate output rather than a crash). Validate output "
            "with a real prompt before trusting this checkpoint.",
            file=sys.stderr,
        )

    manifest_path = (
        Path(args.manifest) if args.manifest is not None
        else output / "k3_setup_manifest.json"
    )
    manifest = {
        "kind": "k3_setup_manifest",
        "schema_version": 2,
        "mode": "self-quantize",
        "source": args.source_quantize,
        "output": str(output),
        "q_bits": args.q_bits,
        "q_group_size": args.q_group_size,
        "size_bytes": size_bytes,
        "size_gb": size_gb,
        "elapsed_seconds": elapsed,
        "host": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
        },
        "ple_safe_quantization": False,  # mlx_lm.convert is NOT PLE-safe
        "warning": (
            "Self-quantized via mlx_lm.convert. May produce degenerate "
            "output due to Gemma 4 PLE / MoE bugs. Use --mode download "
            "for a working PLE-safe variant."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        print(
            f"ERROR: output directory {output} already exists and is non-empty. "
            "Either remove it first or pick a different --output path.",
            file=sys.stderr,
        )
        return 3
    output.parent.mkdir(parents=True, exist_ok=True)

    _check_hf_login()

    if args.mode == "download":
        return _do_download(args, output)
    return _do_self_quantize(args, output)


if __name__ == "__main__":
    sys.exit(main())
