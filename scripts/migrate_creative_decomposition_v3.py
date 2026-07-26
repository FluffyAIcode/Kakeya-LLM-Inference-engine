#!/usr/bin/env python3
"""Atomically snapshot and migrate a quiescent proof to evidence planning."""
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
    ProofState,
    current_capability_manifest,
    load_checkpoint,
    save_checkpoint,
)

MIGRATION_EVENT = "host_feasibility_evidence_planner_v1"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_snapshot(
    *,
    snapshot_root: Path,
    files: tuple[Path, ...],
    old_supervisor_pid: int,
    ledger_version: int = 87,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final = snapshot_root / f"evidence-planner-ledger-v{ledger_version}-{stamp}"
    temporary = snapshot_root / f".{final.name}.{os.getpid()}.tmp"
    snapshot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if final.exists() or temporary.exists():
        raise FileExistsError(final)
    temporary.mkdir(mode=0o700)
    manifest_files = []
    try:
        used_names: set[str] = set()
        for source in files:
            if not source.exists():
                continue
            snapshot_name = source.name
            if snapshot_name in used_names:
                snapshot_name = f"{_sha256(source)[:12]}-{snapshot_name}"
            used_names.add(snapshot_name)
            destination = temporary / snapshot_name
            shutil.copy2(source, destination)
            os.chmod(destination, 0o600)
            manifest_files.append({
                "source": str(source),
                "snapshot_name": destination.name,
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            })
        manifest = {
            "schema_version": 1,
            "migration_event": MIGRATION_EVENT,
            "ledger_version": ledger_version,
            "old_supervisor_pid": old_supervisor_pid,
            "created_at": time.time(),
            "files": manifest_files,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o600)
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, final)
        parent = os.open(snapshot_root, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final


def migrate_checkpoint(
    checkpoint_path: Path,
    snapshot: Path,
    *,
    ledger_version: int | None = None,
    failed_run_id: str = "",
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(checkpoint_path)
    ledger_version = checkpoint.ledger_version if ledger_version is None else ledger_version
    prior_checkpoint_ledger_version = checkpoint.ledger_version
    if (
        checkpoint.ledger_version != ledger_version
        and not (
            checkpoint.ledger_version == ledger_version + 1
            and not checkpoint.committed
        )
    ):
        raise ValueError(
            f"expected ledger v{ledger_version}, found v{checkpoint.ledger_version}"
        )
    checkpoint.ledger_version = ledger_version
    checkpoint.architecture_version = ARCHITECTURE_VERSION
    checkpoint.schema_version = SCHEMA_VERSION
    for name, value in current_capability_manifest().items():
        setattr(checkpoint, name, value)
    reusable = set(checkpoint.validated_artifacts)
    target = (
        ProofState.SYNTHESIS
        if "definition_auditor" in reusable
        else ProofState.DEFINITION_AUDITOR
    )
    checkpoint.state = target.value
    checkpoint.current_role = target.value.lower()
    checkpoint.resume_origin = "DECOMPOSER"
    checkpoint.strategy_reused = True
    checkpoint.adapter_status = ""
    checkpoint.blocked_reason = ""
    checkpoint.migration_event = MIGRATION_EVENT
    checkpoint.migration_snapshot = str(snapshot)
    checkpoint.stagnation_reason = "LEGACY_DECOMPOSER_OUTPUT_AUDIT_ONLY"
    checkpoint.candidate_set_hash = ""
    checkpoint.candidate_hashes = []
    checkpoint.candidate_count = 0
    checkpoint.ranking_hash = ""
    checkpoint.ranked_candidate_ids = []
    checkpoint.selected_move_id = ""
    checkpoint.evidence_gap_graph_hash = ""
    checkpoint.proof_plan_hash = ""
    checkpoint.proof_plan_id = ""
    checkpoint.executable_plan_node_id = ""
    checkpoint.plan_score_explanation = {}
    checkpoint.last_transition_reason = (
        f"{MIGRATION_EVENT}:resume-{target.value.lower()}"
    )
    checkpoint.recovery_events.append({
        "event_type": "OPERATOR_MIGRATION",
        "event_id": MIGRATION_EVENT,
        "from_ledger_version": ledger_version,
        "from_checkpoint_ledger_version": prior_checkpoint_ledger_version,
        "target_state": target.value,
        "snapshot": str(snapshot),
        "failed_run_id": failed_run_id,
        "failed_legacy_output": "AUDIT_ONLY_NOT_REUSABLE",
        "preserved_validated_artifacts": sorted(reusable),
        "created_at": time.time(),
    })
    save_checkpoint(checkpoint_path, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home", type=Path, default=Path.home() / ".kakeya",
    )
    parser.add_argument("--old-supervisor-pid", type=int, required=True)
    parser.add_argument(
        "--failed-run-id", default="br_48bdab7c1d75a51f",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path.cwd() / "autoresearch/prefill/candidate.py",
    )
    parser.add_argument(
        "--failed-run-artifact",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Reuse a quiescent pre-stop snapshot instead of creating one",
    )
    args = parser.parse_args()
    home = args.home.expanduser()
    orchestration = home / "autoresearch/proof_orchestration.json"
    ledger = home / "agent_gan_proof_ledger.json"
    ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
    ledger_version = int(ledger_payload.get("version", 0))
    if ledger_version <= 0:
        raise SystemExit("refusing migration: production ledger has no version")
    files = (
        orchestration,
        ledger,
        args.candidate.expanduser(),
        home / "autoresearch/proof_orchestration.journal.jsonl",
        home / "agent_gan_state.json",
        home / "proof_live_status.json",
        *(item.expanduser() for item in args.failed_run_artifact),
    )
    snapshot = (
        args.snapshot.expanduser()
        if args.snapshot is not None
        else atomic_snapshot(
            snapshot_root=home / "autoresearch/snapshots",
            files=files,
            old_supervisor_pid=args.old_supervisor_pid,
            ledger_version=ledger_version,
        )
    )
    if not snapshot.is_dir():
        raise SystemExit(f"refusing migration: snapshot does not exist: {snapshot}")
    migrate_checkpoint(
        orchestration,
        snapshot,
        ledger_version=ledger_version,
        failed_run_id=args.failed_run_id,
    )
    checkpoint = load_checkpoint(orchestration)
    assert checkpoint is not None
    journal = orchestration.with_name("proof_orchestration.journal.jsonl")
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "operator_event",
            "event_id": MIGRATION_EVENT,
            "ledger_version": ledger_version,
            "old_supervisor_pid": args.old_supervisor_pid,
            "snapshot": str(snapshot),
            "target_state": checkpoint.state,
            "created_at": time.time(),
        }, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "migration_event": MIGRATION_EVENT,
        "snapshot": str(snapshot),
        "ledger_version": ledger_version,
        "old_supervisor_pid": args.old_supervisor_pid,
        "target_state": checkpoint.state,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
