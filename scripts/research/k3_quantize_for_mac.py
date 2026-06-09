"""K3 Mac M4 one-time setup: quantize Gemma 4 26B-A4B-it to 4-bit MLX.

ADR 0008 §11.7 corrected K3 model (HF-verified, §11.14.3) is
``google/gemma-4-26B-A4B-it`` (26B A4B MoE, 4B active / 26B total).
At bf16 the weights are ~52 GB — does not fit Mac M4 24 GB
unified memory. 4-bit quantization via MLX brings that to
~13 GB, leaving ~10 GB headroom for KV cache + activations +
the DFlash drafter (~0.8 GB). This script does the one-time
quantize-and-save step.

Why MLX over bitsandbytes / GPTQ / AWQ:
  * MLX is Apple's native framework for Apple Silicon — uses
    the unified memory architecture directly, no PyTorch MPS
    dispatch overhead.
  * mlx_lm.convert ships a 4-bit quantizer that's well-tested
    on Gemma family and produces a self-contained model
    directory we can load with mlx_lm.load.
  * The quantization is one-time; subsequent feasibility runs
    use the local quantized directory.

Output: a self-contained MLX model directory at the path
specified by --output. The directory contains:
  * config.json          (with quantization metadata)
  * model.safetensors    (4-bit quantized weights, group-quantized)
  * tokenizer files
Total size ~13 GB.

Usage (Mac mini):
  # First, log in to HuggingFace (Gemma 4 is gated):
  huggingface-cli login

  # Then quantize (one-time, ~30-90 min on Mac M4 24 GB):
  PYTHONPATH=.:sdks/python python3 scripts/research/k3_quantize_for_mac.py \\
      --output models/gemma-4-26B-A4B-it-mlx-4bit

The output directory is then the --verifier-path for
scripts/research/k3_feasibility_smoke.py and the K3 reviewer.

Memory note: quantization itself needs ~52 GB working memory
because mlx_lm.convert loads weights at fp16 before quantizing.
On a 24 GB Mac, this requires that mlx_lm uses lazy loading +
disk-backed temporaries — the upstream mlx_lm.convert handles
this automatically as of mlx-lm 0.21+. If your mlx-lm is older,
upgrade first: ``pip install --upgrade mlx-lm``.

This script is a thin wrapper that:
  1. Checks mlx-lm is installed and recent enough
  2. Calls mlx_lm.convert programmatically with sensible defaults
     for K3 (--quantize, --q-bits 4, --q-group-size 64)
  3. Prints a summary of the resulting model directory size
  4. Emits a JSON manifest with the conversion params + sizes
     for reproducibility / audit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", default="google/gemma-4-26B-A4B-it",
        help="HF model id of the source bf16 model. Default is the K3 "
             "production verifier per ADR 0008 §11.14.3.",
    )
    ap.add_argument(
        "--output", default="models/gemma-4-26B-A4B-it-mlx-4bit",
        help="Local directory where the quantized MLX model will be saved.",
    )
    ap.add_argument(
        "--q-bits", type=int, default=4, choices=[2, 3, 4, 6, 8],
        help="Quantization bits per weight. Default 4 = the K3 Mac fit target.",
    )
    ap.add_argument(
        "--q-group-size", type=int, default=64,
        help="Quantization group size (mlx-lm default 64). Smaller group "
             "= better quality, larger model. 64 is the well-tested default.",
    )
    ap.add_argument(
        "--manifest", default=None,
        help="JSON manifest output path. Default: <output>/k3_quantize_manifest.json",
    )
    return ap.parse_args()


def _check_mlx_lm() -> None:
    """Confirm mlx-lm is importable and recent enough."""
    try:
        import mlx_lm  # type: ignore
    except ImportError as e:
        print(
            "ERROR: mlx-lm is not installed. On Mac mini, install with:\n"
            "    pip install --upgrade mlx-lm\n"
            f"Original ImportError: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    # Check version >= 0.21 for lazy loading support on 24 GB Mac.
    version = getattr(mlx_lm, "__version__", "0.0.0")
    print(f"[k3-quantize] mlx-lm version: {version}", file=sys.stderr)
    try:
        major, minor, *_ = version.split(".")
        if int(major) == 0 and int(minor) < 21:
            print(
                f"WARNING: mlx-lm {version} may not support lazy loading "
                "for 26B-class models on 24 GB Mac. Upgrade with:\n"
                "    pip install --upgrade mlx-lm\n"
                "Continuing anyway, but quantization may OOM.",
                file=sys.stderr,
            )
    except Exception:
        pass


def _check_hf_login() -> None:
    """Confirm an HF token is available (Gemma 4 is gated)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    if token is None:
        # Fall back to checking if huggingface-cli is logged in.
        try:
            r = subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                user = r.stdout.strip()
                print(
                    f"[k3-quantize] HF login: {user}", file=sys.stderr,
                )
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        print(
            "ERROR: no HuggingFace credentials detected. Gemma 4 is a gated "
            "model — you must authenticate before downloading. Either:\n"
            "    huggingface-cli login\n"
            "or set HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[k3-quantize] HF token: present (env)", file=sys.stderr)


def _directory_size_bytes(path: Path) -> int:
    """Total size of all files in a directory (recursive)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    args = parse_args()

    _check_mlx_lm()
    _check_hf_login()

    source = args.source
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        print(
            f"ERROR: output directory {output} already exists and is non-empty. "
            "Either remove it first or pick a different --output path.",
            file=sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[k3-quantize] source:      {source}\n"
        f"[k3-quantize] destination: {output}\n"
        f"[k3-quantize] q-bits:      {args.q_bits}\n"
        f"[k3-quantize] q-group:     {args.q_group_size}\n"
        f"[k3-quantize] starting conversion (may take 30-90 min on Mac M4)...",
        file=sys.stderr,
    )

    t_start = time.perf_counter()
    # Call mlx_lm.convert programmatically. The function signature in
    # mlx-lm 0.21+ is:
    #   convert(hf_path, mlx_path, quantize=False, q_group_size=64,
    #           q_bits=4, dtype="float16", upload_repo=None,
    #           revision=None, dequantize=False, ...)
    from mlx_lm import convert  # type: ignore
    try:
        convert(
            hf_path=source,
            mlx_path=str(output),
            quantize=True,
            q_bits=args.q_bits,
            q_group_size=args.q_group_size,
        )
    except Exception as e:
        print(
            f"[k3-quantize] FAIL during mlx_lm.convert: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 4
    elapsed = time.perf_counter() - t_start

    # Post-conversion: report size + write manifest.
    if not output.exists():
        print(
            f"[k3-quantize] FAIL: convert returned but output {output} does not exist",
            file=sys.stderr,
        )
        return 5
    size_bytes = _directory_size_bytes(output)
    size_gb = size_bytes / 1e9
    print(
        f"[k3-quantize] DONE. Size on disk: {size_gb:.2f} GB. Elapsed: {elapsed/60:.1f} min.",
        file=sys.stderr,
    )

    manifest_path = (
        Path(args.manifest) if args.manifest is not None
        else output / "k3_quantize_manifest.json"
    )
    manifest = {
        "kind": "k3_quantize_manifest",
        "schema_version": 1,
        "source": source,
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
        "instructions_for_smoke": (
            "Pass --verifier-path "
            f"{output} to scripts/research/k3_feasibility_smoke.py "
            "(the smoke script auto-detects MLX format from "
            "config.json and uses mlx_lm.load instead of "
            "transformers.AutoModelForCausalLM)."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[k3-quantize] manifest -> {manifest_path}",
        file=sys.stderr,
    )

    # Mac M4 24 GB sanity: warn if size is unexpectedly large.
    if size_gb > 16:
        print(
            f"WARNING: quantized model is {size_gb:.1f} GB, larger than the "
            "estimated ~13 GB. Mac M4 24 GB will have less headroom for KV "
            "cache + activations than expected. Consider --q-bits 3 (smaller, "
            "lower quality) or accept tighter peak memory.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
