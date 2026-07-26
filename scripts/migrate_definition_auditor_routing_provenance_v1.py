#!/usr/bin/env python3
"""Snapshot and migrate the v93 Definition Auditor routing defect."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    DefinitionAuditOutcomeType,
    load_checkpoint,
    persist_validated_artifact,
    route_definition_audit_outcome,
    save_checkpoint,
)


EVENT_ID = "definition_auditor_routing_provenance_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(
    *,
    checkpoint_path: Path,
    ledger_path: Path,
    report_path: Path,
    snapshot_root: Path,
) -> Path:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None or checkpoint.ledger_version != 93:
        raise RuntimeError("snapshot requires the live v93 checkpoint")
    snapshot_dir = snapshot_root / (
        "definition-auditor-routing-provenance-v1-"
        + time.strftime("%Y%m%dT%H%M%S%z")
    )
    snapshot_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    journal_path = checkpoint_path.with_name("proof_orchestration.journal.jsonl")
    sources = {
        "checkpoint": checkpoint_path,
        "ledger": ledger_path,
        "journal": journal_path,
        "report": report_path,
        "report_log": report_path.with_suffix(".log"),
    }
    for role, reference in checkpoint.validated_artifacts.items():
        sources[f"artifact_{role}"] = Path(reference.path)
    manifest = {
        "schema_version": 1,
        "event_id": EVENT_ID,
        "ledger_version": checkpoint.ledger_version,
        "checkpoint_state": checkpoint.state,
        "checkpoint_role": checkpoint.current_role,
        "created_at": time.time(),
        "files": {},
    }
    for name, source in sources.items():
        if not source.is_file():
            raise RuntimeError(f"required snapshot source is missing: {source}")
        destination = snapshot_dir / f"{name}{source.suffix}"
        shutil.copy2(source, destination)
        manifest["files"][name] = {
            "source": str(source),
            "snapshot": destination.name,
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snapshot_dir


def migrate(
    *,
    checkpoint_path: Path,
    ledger_path: Path,
    report_path: Path,
    snapshot_dir: Path,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise RuntimeError("checkpoint is missing")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("version") != 93 or checkpoint.ledger_version != 93:
        raise RuntimeError("migration requires matching ledger/checkpoint v93")
    if checkpoint.migration_event == EVENT_ID:
        return
    reference = checkpoint.validated_artifacts.get("definition_auditor")
    if reference is None:
        raise RuntimeError("validated Definition Auditor artifact is missing")
    artifact_path = Path(reference.path)
    if _sha256(artifact_path) != reference.sha256:
        raise RuntimeError("Definition Auditor artifact hash is stale")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_log = report_path.with_suffix(".log").read_text(encoding="utf-8")
    if (
        report.get("id") != "br_172ab0943c602e01"
        or "audit_outcome REFRAME_REQUIRED;" not in report_log
        or [stage.get("name") for stage in report.get("stages", [])]
        != ["agent_definition_auditor"]
    ):
        raise RuntimeError("production report does not prove exact REFRAME_REQUIRED")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["audit_outcome"] = DefinitionAuditOutcomeType.REFRAME_REQUIRED.value
    migrated_reference = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload=payload,
        dependencies=list(reference.dependencies),
        source_run_id=reference.source_run_id,
    )
    route_definition_audit_outcome(
        checkpoint,
        outcome=DefinitionAuditOutcomeType.REFRAME_REQUIRED,
        artifact_hash=migrated_reference.sha256,
        source_run_id=migrated_reference.source_run_id,
        missing_definition_ids=(
            str(item.get("definition_id", ""))
            for item in payload.get("missing_definitions", ())
        ),
    )
    checkpoint.migration_event = EVENT_ID
    checkpoint.migration_snapshot = str(snapshot_dir)
    checkpoint.recovery_events.append({
        "event_type": EVENT_ID,
        "event_id": EVENT_ID,
        "source_run_id": report["id"],
        "source_artifact_hash": reference.sha256,
        "migrated_artifact_hash": migrated_reference.sha256,
        "target_state": checkpoint.state,
        "snapshot": str(snapshot_dir),
        "created_at": time.time(),
    })
    save_checkpoint(checkpoint_path, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args()
    if args.snapshot_only == args.migrate:
        parser.error("choose exactly one of --snapshot-only or --migrate")
    if args.snapshot_only:
        created = snapshot(
            checkpoint_path=args.checkpoint,
            ledger_path=args.ledger,
            report_path=args.report,
            snapshot_root=args.snapshot_root,
        )
        print(created)
        return 0
    if args.snapshot_dir is None:
        parser.error("--migrate requires --snapshot-dir")
    migrate(
        checkpoint_path=args.checkpoint,
        ledger_path=args.ledger,
        report_path=args.report,
        snapshot_dir=args.snapshot_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
