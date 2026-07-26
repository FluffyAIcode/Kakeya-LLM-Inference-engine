#!/usr/bin/env python3
"""Crash-safe activation migration for definition resolution v1."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.definition_resolution import (
    MIGRATION_EVENT,
    migrate_legacy_registry,
    store_hash,
    resolution_environment_hash,
)
from autoresearch.prefill.orchestration_state import (
    ProofState,
    current_capability_manifest,
    load_checkpoint,
    save_checkpoint,
)
from autoresearch.prefill.theorem_cards import pinned_environment_hash


def snapshot_state(home: Path, destination: Path, supervisor_pid: int) -> Path:
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    sources = (
        home / "autoresearch/proof_orchestration.json",
        home / "autoresearch/proof_orchestration.journal.jsonl",
        home / "autoresearch/proof_orchestration.artifacts",
        home / "autoresearch/proof_orchestration.definition_registry.json",
        home / "agent_gan_proof_ledger.json",
        home / "agent_gan_state.json",
        home / "proof_live_status.json",
        home / "autoresearch/results.tsv",
    )
    copied = []
    for source in sources:
        if not source.exists():
            continue
        target = destination / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
        copied.append({"source": str(source), "snapshot": str(target)})
    manifest = {
        "migration_event": MIGRATION_EVENT,
        "old_supervisor_pid": supervisor_pid,
        "created_at": time.time(),
        "files": copied,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8",
    )
    return destination


def migrate(*, home: Path, project_root: Path, supervisor_pid: int, snapshot: Path) -> dict:
    try:
        os.kill(supervisor_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("supervisor must be stopped before migration")
    checkpoint_path = home / "autoresearch/proof_orchestration.json"
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if raw.get("migration_event") == MIGRATION_EVENT:
        return {
            "event": MIGRATION_EVENT,
            "status": "IDEMPOTENT_REPLAY",
            "snapshot": raw.get("migration_snapshot", ""),
            "current_concept": raw.get("current_definition_gap_id", ""),
        }
    snapshot_state(home, snapshot, supervisor_pid)
    legacy_path = checkpoint_path.with_name(
        "proof_orchestration.definition_registry.json",
    )
    legacy = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else {}
    store = migrate_legacy_registry(
        legacy, base_environment_hash=pinned_environment_hash(project_root),
    )
    store_path = checkpoint_path.with_name(
        "proof_orchestration.definition_resolution.json",
    )
    store_path.write_text(
        json.dumps(store, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.chmod(store_path, 0o600)
    raw.update(current_capability_manifest())
    raw["schema_version"] = 10
    raw["architecture_version"] = 8
    raw["state"] = ProofState.DEFINITION_RESOLUTION.value
    raw["current_role"] = "definition_resolution"
    raw["migration_event"] = MIGRATION_EVENT
    raw["migration_snapshot"] = str(snapshot)
    raw["selected_move_id"] = "RESOLVE_ONE_DEFINITION_QUERY"
    raw["active_gate"] = "AUTONOMOUS_DEFINITION_RESOLUTION"
    raw["current_definition_gap_id"] = "DEF_SEQUENCE_DENSITY"
    raw.pop("definition_registry_hash", None)
    raw.pop("definition_registry_hash_delta", None)
    raw["definition_store_hash"] = store_hash(store)
    raw["definition_environment_hash"] = resolution_environment_hash(store)
    raw["definition_store_hash_delta"] = ""
    raw["definition_environment_hash_delta"] = ""
    raw["definition_query_hash"] = ""
    raw["definition_source_statuses"] = {}
    raw["definition_property_statuses"] = {}
    raw["definition_branch_hashes"] = []
    raw["definition_exhaustion_hash"] = ""
    raw["definition_interface_hash"] = ""
    raw["definition_backjump_target"] = ""
    raw["semantic_stagnation_count"] = 0
    raw["stagnation_reason"] = ""
    raw["progress_fingerprint"] = ""
    raw["forbidden_semantic_fingerprints"] = []
    raw["typed_ir_hash"] = ""
    raw["proposition_hash"] = ""
    raw["elaborated_theorem_id"] = ""
    raw["adapter_status"] = ""
    raw["blocked_reason"] = ""
    raw["last_transition_reason"] = "migration:autonomous-definition-resolution-v1"
    raw.setdefault("recovery_events", []).append({
        "event_type": "OPERATOR_MIGRATION",
        "event_id": MIGRATION_EVENT,
        "target_state": ProofState.DEFINITION_RESOLUTION.value,
        "current_concept": "DEF_SEQUENCE_DENSITY",
        "legacy_definitions": len(store["historical_audit"]),
        "legacy_policy": "AUDIT_ONLY_PENDING_SEMANTIC_VALIDATION",
        "created_at": time.time(),
    })
    checkpoint_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise RuntimeError("checkpoint disappeared during migration")
    save_checkpoint(checkpoint_path, checkpoint)
    return {
        "event": MIGRATION_EVENT,
        "status": "MIGRATED",
        "snapshot": str(snapshot),
        "current_concept": checkpoint.current_definition_gap_id,
        "legacy_audit_only": len(store["historical_audit"]),
        "store_hash": store["store_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path.home() / ".kakeya")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(
        home=args.home.expanduser(),
        project_root=args.project_root.expanduser(),
        supervisor_pid=args.supervisor_pid,
        snapshot=args.snapshot.expanduser(),
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
