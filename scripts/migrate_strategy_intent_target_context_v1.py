#!/usr/bin/env python3
"""Offline, idempotent Architecture-9 target-context cutover."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    ProofState,
    load_checkpoint,
    save_checkpoint,
)
from autoresearch.prefill.target_context import (
    MIGRATION_EVENT,
    activate_target_context,
)
from autoresearch.prefill.theorem_cards import pinned_environment_hash


def _copy_if_present(source: Path, destination: Path) -> str:
    if not source.exists():
        return ""
    target = destination / source.name
    shutil.copy2(source, target)
    os.chmod(target, 0o600)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    ledger_path = args.ledger.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    snapshot = args.snapshot_dir.expanduser().resolve()
    snapshot.mkdir(parents=True, exist_ok=False, mode=0o700)
    lock_path = checkpoint_path.with_name(".strategy-target-context-migration.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_hashes = {
            path.name: digest for path in (
                checkpoint_path,
                checkpoint_path.with_name("proof_orchestration.journal.jsonl"),
                checkpoint_path.with_name("proof_target_contexts.json"),
                ledger_path,
                state_path,
            ) if (digest := _copy_if_present(path, snapshot))
        }
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise ValueError("CHECKPOINT_REQUIRED")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        obligations = {
            str(item["obligation_id"]): item
            for item in ledger.get("obligations", ())
        }
        rh_c1 = obligations["RH-C1"]
        rh_c2_items = [
            item for key, item in obligations.items() if key.startswith("RH-C2")
        ]
        rh_c2 = max(
            rh_c2_items,
            key=lambda item: len(str(item["obligation_id"])),
        )
        environment_hash = pinned_environment_hash(args.project_root.resolve())

        # Materialize the contaminated branch as immutable audit history first.
        activate_target_context(
            checkpoint_path,
            checkpoint,
            target_obligation_id=str(rh_c2["obligation_id"]),
            statement=str(rh_c2["statement"]),
            environment_hash=environment_hash,
            strategy_plan_hash="LEGACY_AUDIT",
            evidence={
                "last_evidence": str(rh_c2.get("last_evidence", "")),
                "migration_source": "pre_target_context_active_pointers",
            },
        )
        # Then create the only active namespace from concise RH-C1 evidence.
        activate_target_context(
            checkpoint_path,
            checkpoint,
            target_obligation_id="RH-C1",
            statement=str(rh_c1["statement"]),
            environment_hash=environment_hash,
            strategy_plan_hash="STRATEGY_PENDING",
            evidence={
                "last_evidence": str(rh_c1.get("last_evidence", "")),
                "formal_status": str(rh_c1.get("formal_status", "")),
                "ledger_version": int(ledger.get("version", 0)),
            },
        )
        checkpoint.state = ProofState.STRATEGY_TOURNAMENT.value
        checkpoint.current_role = "strategy_tournament"
        checkpoint.adapter_status = ""
        checkpoint.blocked_reason = ""
        checkpoint.strategy_run_status = "MIGRATED_AWAITING_STRATEGY"
        checkpoint.migration_event = MIGRATION_EVENT
        checkpoint.migration_snapshot = "sha256:" + hashlib.sha256(
            json.dumps(snapshot_hashes, sort_keys=True).encode()
        ).hexdigest()
        checkpoint.last_transition_reason = MIGRATION_EVENT
        checkpoint.updated_at = time.time()
        save_checkpoint(checkpoint_path, checkpoint)
    print(json.dumps({
        "migration_event": MIGRATION_EVENT,
        "active_target": "RH-C1",
        "active_context_hash": checkpoint.target_context_hash,
        "snapshot_dir": str(snapshot),
        "snapshot_hashes": snapshot_hashes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
