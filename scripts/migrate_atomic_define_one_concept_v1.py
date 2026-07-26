#!/usr/bin/env python3
"""Migrate the active Architecture-7 definition loop crash-safely."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.atomic_definition import (
    MIGRATION_EVENT,
    define_one_concept,
    dependency_graph_for_audit,
    first_dependency_closed_gap,
)
from autoresearch.prefill.orchestration_state import (
    ProofState,
    current_capability_manifest,
    load_checkpoint,
    save_checkpoint,
)
from autoresearch.prefill.theorem_cards import pinned_environment_hash


def _snapshot(home: Path, destination: Path, supervisor_pid: int) -> Path:
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    sources = (
        home / "autoresearch/proof_orchestration.json",
        home / "autoresearch/proof_orchestration.journal.jsonl",
        home / "autoresearch/proof_orchestration.artifacts",
        home / "agent_gan_proof_ledger.json",
        home / "agent_gan_state.json",
        home / "proof_live_status.json",
        home / "autoresearch/results.tsv",
    )
    records = []
    for source in sources:
        if not source.exists():
            continue
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        records.append({"source": str(source), "snapshot": str(target)})
    manifest = {
        "migration_event": MIGRATION_EVENT,
        "old_supervisor_pid": supervisor_pid,
        "created_at": time.time(),
        "files": records,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def migrate(
    *,
    home: Path,
    project_root: Path,
    supervisor_pid: int,
    snapshot: Path,
) -> dict:
    checkpoint_path = home / "autoresearch/proof_orchestration.json"
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if raw.get("migration_event") == MIGRATION_EVENT:
        raw.update(current_capability_manifest())
        raw["adapter_status"] = ""
        raw["blocked_reason"] = ""
        temporary = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp",
        )
        temporary.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, checkpoint_path)
        return {
            "event": MIGRATION_EVENT,
            "status": "IDEMPOTENT_REPLAY",
            "snapshot": raw.get("migration_snapshot", ""),
            "current_gap": raw.get("current_definition_gap_id", ""),
        }
    try:
        os.kill(supervisor_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("supervisor must be stopped before migration")
    _snapshot(home, snapshot, supervisor_pid)
    raw.update(current_capability_manifest())
    raw["capability_flags"] = current_capability_manifest()["capability_flags"]
    checkpoint_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise RuntimeError("checkpoint disappeared during migration")
    auditor = checkpoint.validated_artifacts.get("definition_auditor")
    if auditor is None:
        raise RuntimeError("migration requires a validated Definition Auditor")
    audit = json.loads(Path(auditor.path).read_text(encoding="utf-8"))
    missing = tuple(audit.get("missing_definitions", ()))
    graph = dependency_graph_for_audit(missing)
    environment_hash = pinned_environment_hash(project_root)
    registry_path = checkpoint_path.with_name(
        "proof_orchestration.definition_registry.json",
    )
    migrated_resolutions = []
    for restriction_id in ("DEF_EPSILON", "DEF_GENUS"):
        gap = next(
            (
                item for item in missing
                if item.get("definition_id") == restriction_id
            ),
            None,
        )
        if gap is None:
            continue
        result = define_one_concept(
            gap=gap,
            definition_auditor_hash=auditor.sha256,
            dependency_graph=graph,
            registry_path=registry_path,
            project_root=project_root,
            theorem_card_ids=(),
            current_environment_hash=environment_hash,
        )
        migrated_resolutions.append(asdict(result))
    resolved = {
        item["gap_id"] for item in migrated_resolutions
        if item["status"] in {"RESOLVED", "IDEMPOTENT_REPLAY"}
    }
    first_gap = first_dependency_closed_gap(
        missing,
        dependency_graph=graph,
        resolved_gap_ids=resolved,
    )
    for role in (
        "decomposer", "synthesis", "math_ir_translator",
        "host_typed_ir_gate", "strategy_tournament", "research_contract",
    ):
        reference = checkpoint.validated_artifacts.pop(role, None)
        if reference is not None:
            checkpoint.invalidated_artifacts[reference.sha256] = {
                **asdict(reference),
                "audit_only": True,
                "reason_codes": [
                    "LEGACY_DEFINITION_REGISTRATION_NOT_PROGRESS",
                    "PLACEHOLDER_TRUE_NOT_PROGRESS",
                ],
                "migration_event": MIGRATION_EVENT,
            }
    checkpoint.state = ProofState.STRATEGY_TOURNAMENT.value
    checkpoint.current_role = "strategy_tournament"
    checkpoint.adapter_status = ""
    checkpoint.blocked_reason = ""
    checkpoint.selected_move_id = "DEFINE_ONE_CONCEPT"
    checkpoint.active_gate = "HOST_DEFINITION_GATE"
    checkpoint.current_definition_gap_id = (
        str(first_gap.get("definition_id", "")) if first_gap else ""
    )
    checkpoint.candidate_count = 0
    checkpoint.definition_candidate_count = 0
    checkpoint.typed_ir_hash = ""
    checkpoint.lean_declaration_hash = ""
    checkpoint.proposition_hash = ""
    checkpoint.elaborated_theorem_id = ""
    checkpoint.new_elaborated_definitions = 0
    checkpoint.new_elaborated_lemmas = 0
    checkpoint.definitions_added = 0
    checkpoint.lemmas_proved = 0
    checkpoint.progress_vector = {
        "definitions_added": 0,
        "existing_definitions_resolved": 0,
        "lemmas_proved": 0,
        "accepted_children": 0,
        "subgoals_closed": 0,
        "verified_counterexamples": 0,
    }
    checkpoint.progress_fingerprint = ""
    checkpoint.semantic_stagnation_count = 0
    checkpoint.stagnation_reason = ""
    checkpoint.forbidden_semantic_fingerprints = []
    checkpoint.strategy_event_id = ""
    checkpoint.strategy_event_type = ""
    checkpoint.research_contract_id = ""
    checkpoint.research_contract_hash = ""
    checkpoint.research_contract_rejection_codes = []
    checkpoint.migration_event = MIGRATION_EVENT
    checkpoint.migration_snapshot = str(snapshot)
    checkpoint.last_transition_reason = (
        "migration:atomic-define-one-concept:first-dependency-closed-gap"
    )
    checkpoint.recovery_events.append({
        "event_type": "OPERATOR_MIGRATION",
        "event_id": MIGRATION_EVENT,
        "from_move": "REGISTER_DEFINITION_OBLIGATION",
        "target_state": checkpoint.state,
        "current_gap": checkpoint.current_definition_gap_id,
        "typed_restrictions_resolved": sorted(resolved),
        "created_at": time.time(),
    })
    save_checkpoint(checkpoint_path, checkpoint)
    return {
        "event": MIGRATION_EVENT,
        "status": "MIGRATED",
        "snapshot": str(snapshot),
        "current_gap": checkpoint.current_definition_gap_id,
        "typed_restrictions_resolved": sorted(resolved),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path.home() / ".kakeya")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    result = migrate(
        home=args.home.expanduser(),
        project_root=args.project_root.expanduser(),
        supervisor_pid=args.supervisor_pid,
        snapshot=args.snapshot.expanduser(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
