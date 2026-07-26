#!/usr/bin/env python3
"""Atomically migrate production to architecture 7 and review its branch."""
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
    ARCHITECTURE_VERSION,
    SCHEMA_VERSION,
    STRATEGY_TOURNAMENT_MIGRATION_EVENT,
    ProofState,
    current_capability_manifest,
    load_checkpoint,
    save_checkpoint,
)
from autoresearch.prefill.strategy_tournament import (
    BranchEvidence,
    BranchHistory,
    branch_review_record,
    review_branch,
)


MIGRATION_EVENT = STRATEGY_TOURNAMENT_MIGRATION_EVENT


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def atomic_snapshot(
    snapshot_root: Path,
    sources: tuple[Path, ...],
    *,
    supervisor_pid: int,
    ledger_version: int,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final = snapshot_root / f"strategy-tournament-v{ledger_version}-{stamp}"
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
                digest = _digest(sorted(
                    str(path.relative_to(destination))
                    + ":" + hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in destination.rglob("*") if path.is_file()
                ))
            else:
                shutil.copy2(source, destination)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                os.chmod(destination, 0o600)
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


def recorded_density_singularity_review(ledger: dict) -> dict[str, object]:
    obligations = [
        item for item in ledger.get("obligations", ())
        if (
            "density" in str(item.get("statement", "")).lower()
            or "singular" in str(item.get("statement", "")).lower()
        )
    ]
    evidence = []
    for item in obligations:
        canonical = dict(item)
        evidence.append(BranchEvidence(
            evidence_id=str(item.get("obligation_id", "")),
            event_type="RECORDED_LEDGER_OBLIGATION",
            accepted_child_delta=int(
                bool(item.get("decomposition_certificate_hash"))
                and str(item.get("status", "")) != "REJECTED_DUPLICATE"
            ),
            lean_theorem_delta=int(
                str(item.get("formal_status", "")).upper() == "PROVED"
            ),
            verified_counterexample_delta=int(
                bool(item.get("quarantine_evidence_source"))
                and float(item.get("quarantine_confidence", 0.0)) >= 0.8
            ),
            semantic_failure_delta=int(
                str(item.get("status", "")).upper()
                in {"REJECTED", "REJECTED_DUPLICATE", "QUARANTINED"}
            ),
            reason_codes=tuple(filter(None, (
                str(item.get("status", "")),
                str(item.get("formal_status", "")),
                str(item.get("invalidation_kind", "")),
            ))),
            provenance_hash=_digest(canonical),
        ))
    history = BranchHistory(
        branch_id="density-singularity",
        plan_ids=[str(item.get("obligation_id", "")) for item in obligations],
        evidence=evidence,
    )
    review_branch(history, stagnation_threshold=4, failure_threshold=3)
    return branch_review_record(history)


def migrate(
    checkpoint_path: Path,
    ledger_path: Path,
    snapshot: Path,
) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(checkpoint_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    existing = [
        item for item in checkpoint.recovery_events
        if (
            item.get("event_id") == MIGRATION_EVENT
            and item.get("branch_review")
            and item.get("snapshot")
        )
    ]
    ledger_version = int(ledger.get("version", 0))
    if (
        existing
        and checkpoint.architecture_version == ARCHITECTURE_VERSION
        and checkpoint.ledger_version == ledger_version
        and set(checkpoint.validated_artifacts) <= {"definition_auditor"}
        and checkpoint.migration_snapshot
    ):
        return existing[-1]
    for role in tuple(checkpoint.validated_artifacts):
        if role == "definition_auditor":
            continue
        reference = checkpoint.validated_artifacts.pop(role, None)
        if reference is not None:
            checkpoint.invalidated_artifacts[reference.sha256] = {
                **asdict(reference),
                "audit_only": True,
                "read_only": True,
                "reason_codes": ["LEGACY_PRE_CONTRACT_EXECUTION"],
                "migration_event": MIGRATION_EVENT,
            }
    review = recorded_density_singularity_review(ledger)
    review_path = checkpoint_path.with_name(
        "density_singularity_branch_review.json",
    )
    review_path.write_text(
        json.dumps(review, sort_keys=True, indent=2), encoding="utf-8",
    )
    os.chmod(review_path, 0o600)
    checkpoint.architecture_version = ARCHITECTURE_VERSION
    checkpoint.schema_version = SCHEMA_VERSION
    for name, value in current_capability_manifest().items():
        setattr(checkpoint, name, value)
    prior_state = checkpoint.state
    checkpoint.state = ProofState.STRATEGY_TOURNAMENT.value
    checkpoint.current_role = "strategy_tournament"
    checkpoint.resume_origin = prior_state
    checkpoint.ledger_version = ledger_version
    checkpoint.strategy_reused = False
    checkpoint.adapter_status = ""
    checkpoint.blocked_reason = ""
    checkpoint.migration_event = MIGRATION_EVENT
    checkpoint.migration_snapshot = str(snapshot)
    checkpoint.strategy_event_id = ""
    checkpoint.strategy_event_type = ""
    checkpoint.strategy_plan_ids = []
    checkpoint.feasible_strategy_plan_ids = []
    checkpoint.selected_strategy_plan_id = ""
    checkpoint.research_contract_id = ""
    checkpoint.research_contract_hash = ""
    checkpoint.research_contract_rejection_codes = []
    checkpoint.typed_ir_hash = ""
    checkpoint.lean_declaration_hash = ""
    checkpoint.proposition_hash = ""
    checkpoint.elaborated_theorem_id = ""
    checkpoint.active_gate = ""
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
    checkpoint.branch_history["density-singularity"] = review
    checkpoint.branches_killed = int(review["status"] == "QUARANTINED")
    checkpoint.last_transition_reason = f"{MIGRATION_EVENT}:formal-branch-review"
    event = {
        "event_type": "OPERATOR_MIGRATION",
        "event_id": MIGRATION_EVENT,
        "ledger_version": ledger_version,
        "target_state": checkpoint.state,
        "snapshot": str(snapshot),
        "branch_review": str(review_path),
        "branch_review_hash": review["content_hash"],
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
    parser.add_argument(
        "--candidate", type=Path,
        default=Path.cwd() / "autoresearch/prefill/candidate.py",
    )
    args = parser.parse_args()
    home = args.home.expanduser()
    checkpoint = home / "autoresearch/proof_orchestration.json"
    ledger = home / "agent_gan_proof_ledger.json"
    ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
    sources = (
        checkpoint,
        checkpoint.with_name("proof_orchestration.journal.jsonl"),
        checkpoint.with_suffix(".artifacts"),
        ledger,
        home / "agent_gan_state.json",
        home / "proof_live_status.json",
        args.candidate.expanduser(),
        home / "autoresearch" / "runs",
    )
    snapshot = (
        args.snapshot.expanduser() if args.snapshot else atomic_snapshot(
            home / "autoresearch/snapshots",
            sources,
            supervisor_pid=args.supervisor_pid,
            ledger_version=int(ledger_payload.get("version", 0)),
        )
    )
    event = migrate(checkpoint, ledger, snapshot)
    print(json.dumps(event, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
