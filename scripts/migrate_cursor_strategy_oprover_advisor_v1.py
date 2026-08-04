#!/usr/bin/env python3
"""Offline checkpoint cutover; production invocation requires prior snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    ARCHITECTURE_VERSION,
    SCHEMA_VERSION,
    STRATEGY_TOURNAMENT_MIGRATION_EVENT,
    current_capability_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    snapshot_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    snapshot = snapshot_dir / source.name
    shutil.copy2(source, snapshot)
    os.chmod(snapshot, 0o600)
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    artifacts = raw.setdefault("validated_artifacts", {})
    invalidated = raw.setdefault("invalidated_artifacts", {})
    for role in ("strategy", "generator", "critic"):
        reference = artifacts.pop(role, None)
        if reference:
            invalidated[reference["sha256"]] = {
                **reference,
                "audit_only": True,
                "reason_codes": ["LEGACY_PROVIDER_DIRECT_CUTOVER"],
                "migration_event": STRATEGY_TOURNAMENT_MIGRATION_EVENT,
            }
    raw.update(current_capability_manifest())
    raw.update({
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "state": "STRATEGY_TOURNAMENT",
        "current_role": "strategy_tournament",
        "migration_event": STRATEGY_TOURNAMENT_MIGRATION_EVENT,
        "migration_snapshot": f"sha256:{snapshot_hash}",
        "adapter_status": "INTEGRATION_BLOCKED",
        "blocked_reason": (
            "STRATEGY_PROVIDER_UNAVAILABLE:CONFIGURATION_REQUIRED"
        ),
        "strategy_provider": "cursor-sdk",
        "strategy_provider_configured": False,
        "strategy_run_status": "STRATEGY_PROVIDER_UNAVAILABLE",
        "residency_phase": "GEMMA_SERVING",
        "active_model": "gemma",
        "updated_at": time.time(),
    })
    encoded = json.dumps(raw, ensure_ascii=False, indent=2)
    temporary = source.with_name(f".{source.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, source)
    print(json.dumps({
        "migration_event": STRATEGY_TOURNAMENT_MIGRATION_EVENT,
        "snapshot_sha256": snapshot_hash,
        "checkpoint": str(source),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
