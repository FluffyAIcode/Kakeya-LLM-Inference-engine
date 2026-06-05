"""Pre-warm the HuggingFace model cache for v0.3 runtime use.

The v0.3 gRPC server fails fast at startup when the verifier model
isn't already in the HF cache (per
:func:`inference_engine.setup.prewarm.assert_cached_or_raise`). This
CLI is the canonical way to fill the cache: it surfaces the standard
huggingface_hub progress bar, resumes interrupted downloads, and
reports total disk footprint when done.

Usage::

    PYTHONPATH=. python3 scripts/kakeya_prewarm.py \\
        --verifier-id Qwen/Qwen3-0.6B

    # Multiple at once (e.g. for switching between Mac CPU + MLX-4bit)
    PYTHONPATH=. python3 scripts/kakeya_prewarm.py \\
        --verifier-id Qwen/Qwen3-0.6B \\
        --verifier-id mlx-community/Qwen3-1.7B-4bit

    # Custom cache root
    PYTHONPATH=. python3 scripts/kakeya_prewarm.py \\
        --verifier-id Qwen/Qwen3-0.6B \\
        --cache-root /data/hf-cache

Mainland-China users behind GFW: ``export HF_ENDPOINT=https://hf-mirror.com``
before invoking. huggingface_hub respects ``HF_ENDPOINT`` per its
documented contract.

Per the project's CLI-plumbing convention this script is exempt
from the unit-test coverage gate (the underlying helpers in
``inference_engine.setup.prewarm`` are 100% covered). End-to-end
correctness is exercised by the Mac M4 reviewer aid + the integration
suite, both of which depend on a populated HF cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inference_engine.setup import (
    HF_CACHE_DEFAULT,
    free_disk_bytes,
    prewarm_model_id,
)


def _format_bytes(n: int) -> str:
    """Human-readable byte count for log lines."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verifier-id",
        action="append",
        required=True,
        metavar="MODEL_ID",
        help=(
            "HuggingFace model id (owner/repo). Pass multiple times to "
            "pre-warm a set, e.g. one bf16 and one MLX-4bit variant."
        ),
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            f"HF cache root. Defaults to $HF_HUB_CACHE / $HF_HOME / "
            f"~/.cache/huggingface (resolved as: {HF_CACHE_DEFAULT})."
        ),
    )
    ap.add_argument(
        "--no-tokenizer",
        action="store_true",
        help=(
            "Skip tokenizer files (saves <50 MB; only useful for "
            "inference-only workflows that already have the tokenizer)."
        ),
    )
    args = ap.parse_args()

    free = free_disk_bytes(args.cache_root)
    print(
        f"[prewarm] cache root: {args.cache_root or HF_CACHE_DEFAULT}\n"
        f"[prewarm] free disk:  {_format_bytes(free)}",
        file=sys.stderr, flush=True,
    )
    if free < 5 * 1024 * 1024 * 1024:
        print(
            f"[prewarm] WARNING: less than 5 GiB free; large models may "
            f"not fit. Free up space or pass --cache-root to a different "
            f"filesystem.",
            file=sys.stderr,
        )

    overall_ok = True
    for model_id in args.verifier_id:
        print(
            f"\n[prewarm] {model_id} ...",
            file=sys.stderr, flush=True,
        )
        try:
            status = prewarm_model_id(
                model_id,
                cache_root=args.cache_root,
                include_tokenizer=not args.no_tokenizer,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(
                f"[prewarm] FAILED {model_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            overall_ok = False
            continue

        print(f"[prewarm] {status.human()}", file=sys.stderr)

    if not overall_ok:
        print(
            "\n[prewarm] one or more downloads failed; see error lines "
            "above. Re-run after fixing the cause; downloads resume from "
            "wherever the previous attempt left off.",
            file=sys.stderr,
        )
        return 1

    print(
        "\n[prewarm] done. Start the runtime now:\n"
        "  PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \\\n"
        "      --backend cpu --verifier-id "
        f"{args.verifier_id[0]}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
