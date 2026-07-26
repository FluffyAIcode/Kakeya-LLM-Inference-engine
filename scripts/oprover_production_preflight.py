#!/usr/bin/env python3
"""Read-only acceptance gate before any production residency swap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from autoresearch.prefill.oprover_advisor import OFFICIAL_REVISION
from autoresearch.prefill.cursor_strategy import cursor_keychain_configured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--quant", choices=("q4", "q5"), required=True)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--minimum-free-gib", type=float, default=12.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    free = shutil.disk_usage(model_dir.parent if model_dir.parent.exists()
                             else Path.home()).free
    cursor_env = bool(os.getenv("CURSOR_API_KEY", "").strip())
    cursor_keychain = cursor_keychain_configured()
    cursor_key = cursor_env or cursor_keychain
    cursor_model = bool(
        os.getenv("KAKEYA_CURSOR_STRATEGY_MODEL", "").strip()
    )
    source_manifest = model_dir / "source-manifest.json"
    try:
        source_manifest_payload = json.loads(
            source_manifest.read_text(encoding="utf-8")
        )
        source_manifest_hash = hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        source_manifest_payload = {}
        source_manifest_hash = ""
    try:
        model_config = json.loads(
            (model_dir / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        model_config = {}
    quantization = model_config.get("quantization", {})
    quant_bits = int(quantization.get("bits", 0) or 0)
    expected_bits = 5 if args.quant == "q5" else 4
    converted_model = (
        quant_bits == expected_bits
        and any(model_dir.glob("*.safetensors"))
    )
    revision_ok = args.revision == OFFICIAL_REVISION
    permissions_ok = (
        model_dir.is_dir()
        and not bool(model_dir.stat().st_mode & stat.S_IWOTH)
    )
    # Names only: never serialize command lines because they may contain keys.
    process_names = subprocess.run(
        ["ps", "-axo", "comm="], text=True, capture_output=True, check=True,
    ).stdout.lower()
    gemma_detected = "python" in process_names
    report = {
        "schema_version": 1,
        "destructive_actions_performed": False,
        "cursor_api_key": "configured" if cursor_key else "missing",
        "cursor_credential_reference": (
            "environment" if cursor_env
            else "macos-keychain" if cursor_keychain
            else "missing"
        ),
        "cursor_model_id": "configured" if cursor_model else "missing",
        "model_dir": str(model_dir),
        "model_dir_present": model_dir.is_dir(),
        "quantization": args.quant,
        "revision": args.revision,
        "revision_pinned": revision_ok,
        "source_manifest_present": source_manifest.is_file(),
        "source_manifest_sha256": source_manifest_hash,
        "source_manifest_revision_pinned": (
            source_manifest_payload.get("revision") == OFFICIAL_REVISION
        ),
        "mlx_quantized_model": converted_model,
        "quantization_bits": quant_bits,
        "permissions_ok": permissions_ok,
        "free_bytes": free,
        "free_space_ok": free >= args.minimum_free_gib * 1024 ** 3,
        "gemma_process_class_detected": gemma_detected,
        "ready": all((
            cursor_key, cursor_model, model_dir.is_dir(), revision_ok,
            source_manifest.is_file(),
            source_manifest_payload.get("revision") == OFFICIAL_REVISION,
            converted_model, permissions_ok,
            free >= args.minimum_free_gib * 1024 ** 3,
        )),
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
