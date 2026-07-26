#!/usr/bin/env python3
"""Snapshot and migrate a pre-proof Contract deadlock to Decomposer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    ProofState,
    current_capability_manifest,
    load_checkpoint,
    save_checkpoint,
)


MIGRATION_EVENT = "research_contract_preproof_routing_v2"


def snapshot_runtime(
    snapshot_root: Path,
    sources: tuple[Path, ...],
    *,
    supervisor_pid: int,
    ledger_version: int,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final = snapshot_root / (
        f"research-contract-preproof-v{ledger_version}-{stamp}"
    )
    temporary = snapshot_root / f".{final.name}.{os.getpid()}.tmp"
    snapshot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary.mkdir(mode=0o700)
    records = []
    try:
        for index, source in enumerate(sources):
            if not source.exists():
                continue
            destination = temporary / f"{index:02d}-{source.name}"
            if source.is_dir():
                shutil.copytree(source, destination)
                digest = hashlib.sha256(json.dumps(sorted(
                    str(path.relative_to(destination))
                    + ":" + hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in destination.rglob("*") if path.is_file()
                ), separators=(",", ":")).encode()).hexdigest()
            else:
                shutil.copy2(source, destination)
                os.chmod(destination, 0o600)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            records.append({
                "source": str(source),
                "snapshot_name": destination.name,
                "sha256": digest,
            })
        manifest = {
            "schema_version": 1,
            "migration_event": MIGRATION_EVENT,
            "ledger_version": ledger_version,
            "old_supervisor_pid": supervisor_pid,
            "created_at": time.time(),
            "files": records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o600)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final


def migrate(
    checkpoint_path: Path,
    ledger_path: Path,
    snapshot: Path,
) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(checkpoint_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_version = int(ledger.get("version", 0))
    existing = [
        event for event in checkpoint.recovery_events
        if event.get("event_id") == MIGRATION_EVENT
    ]
    if (
        existing
        and checkpoint.proof_state == ProofState.DECOMPOSER
        and not checkpoint.research_contract_id
        and checkpoint.ledger_version == ledger_version
    ):
        return existing[-1]
    contract_ref = checkpoint.validated_artifacts.pop(
        "research_contract", None,
    )
    if contract_ref is not None:
        checkpoint.invalidated_artifacts[contract_ref.sha256] = {
            **asdict(contract_ref),
            "audit_only": True,
            "read_only": True,
            "reason_codes": ["PREPROOF_CONTRACT_INVALID"],
            "migration_event": MIGRATION_EVENT,
        }
    prior_state = checkpoint.state
    checkpoint.state = ProofState.DECOMPOSER.value
    checkpoint.current_role = "decomposer"
    checkpoint.resume_origin = prior_state
    checkpoint.strategy_reused = True
    checkpoint.ledger_version = ledger_version
    checkpoint.adapter_status = ""
    checkpoint.blocked_reason = ""
    checkpoint.research_contract_id = ""
    checkpoint.research_contract_hash = ""
    checkpoint.research_contract_rejection_codes = []
    checkpoint.active_gate = ""
    checkpoint.migration_event = MIGRATION_EVENT
    checkpoint.migration_snapshot = str(snapshot)
    checkpoint.last_transition_reason = (
        f"{MIGRATION_EVENT}:resume-at-earliest-semantic-owner"
    )
    for name, value in current_capability_manifest().items():
        setattr(checkpoint, name, value)
    event = {
        "event_type": "OPERATOR_MIGRATION",
        "event_id": MIGRATION_EVENT,
        "ledger_version": ledger_version,
        "from_state": prior_state,
        "target_state": ProofState.DECOMPOSER.value,
        "snapshot": str(snapshot),
        "strategy_tournament_hash": checkpoint.strategy_tournament_hash,
        "strategy_plan_ids": list(checkpoint.strategy_plan_ids),
        "selected_strategy_plan_id": checkpoint.selected_strategy_plan_id,
        "preserved_quarantine_evidence": bool(checkpoint.branch_history),
        "created_at": time.time(),
    }
    checkpoint.recovery_events.append(event)
    save_checkpoint(checkpoint_path, checkpoint)
    return event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path.home() / ".kakeya")
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    home = args.home.expanduser()
    checkpoint = home / "autoresearch/proof_orchestration.json"
    ledger = home / "agent_gan_proof_ledger.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    tournament_value = str(
        checkpoint_payload.get("validated_artifacts", {})
        .get("strategy_tournament", {})
        .get("path", "")
    )
    sources = [
        checkpoint,
        checkpoint.with_name("proof_orchestration.journal.jsonl"),
        checkpoint.with_suffix(".artifacts"),
        ledger,
        home / "proof_live_status.json",
    ]
    if tournament_value:
        sources.insert(2, Path(tournament_value))
    snapshot = (
        args.snapshot.expanduser() if args.snapshot else snapshot_runtime(
            home / "autoresearch/snapshots",
            tuple(sources),
            supervisor_pid=args.supervisor_pid,
            ledger_version=int(payload.get("version", 0)),
        )
    )
    print(json.dumps(migrate(checkpoint, ledger, snapshot), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
