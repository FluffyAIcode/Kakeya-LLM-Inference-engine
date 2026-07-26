"""Single architecture-7 entry point for tournament and contract routing."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from autoresearch.prefill.definition_resolution import (
    build_definition_query,
    load_resolution_store,
    resolve_one_concept,
)
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    persist_validated_artifact,
    save_checkpoint,
)
from autoresearch.prefill.research_contract import gate_research_contract
from autoresearch.prefill.strategy_tournament import (
    CriticReason,
    StrategyEvent,
    build_host_plans,
    evaluate_feasibility,
    run_tournament,
)
from autoresearch.prefill.theorem_cards import (
    build_theorem_card_index,
    pinned_environment_hash,
)


def _definition_audit(
    checkpoint: OrchestrationCheckpoint,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    reference = checkpoint.validated_artifacts.get("definition_auditor")
    if reference is None:
        return (), (), (), ""
    try:
        payload = json.loads(Path(reference.path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return (), (), (), reference.sha256
    registered = tuple(sorted({
        str(item.get("definition_id", ""))
        for item in payload.get("definitions", ())
        if isinstance(item, Mapping) and item.get("definition_id")
    }))
    unresolved = tuple(sorted({
        str(item.get("definition_id", ""))
        for item in payload.get("missing_definitions", ())
        if isinstance(item, Mapping) and item.get("definition_id")
    }))
    gaps = tuple(f"gap:definition:{item}" for item in unresolved)
    return registered, unresolved, gaps, reference.sha256


def run_host_definition_gate(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    project_root: Path,
) -> tuple[OrchestrationCheckpoint, str]:
    """Execute exactly one Autonomous Definition Resolution transaction."""
    if checkpoint.proof_state not in {
        ProofState.DEFINITION_RESOLUTION,
        ProofState.DECOMPOSER,
        ProofState.STRATEGY_TOURNAMENT,
        ProofState.MATHEMATICAL_STAGNATION,
    }:
        return checkpoint, ""
    reference = checkpoint.validated_artifacts.get("definition_auditor")
    if reference is None:
        return checkpoint, ""
    try:
        audit = json.loads(Path(reference.path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return checkpoint, ""
    missing = tuple(
        item for item in audit.get("missing_definitions", ())
        if isinstance(item, Mapping) and item.get("definition_id")
    )
    if not missing:
        return checkpoint, ""
    base_environment_hash = pinned_environment_hash(project_root)
    store_path = checkpoint_path.with_name(
        "proof_orchestration.definition_resolution.json",
    )
    store = load_resolution_store(store_path, base_environment_hash)
    requested = checkpoint.current_definition_gap_id
    gap = next(
        (
            item for item in missing
            if str(item.get("definition_id", "")) == requested
        ),
        None,
    ) if requested else None
    if gap is None:
        committed = set(store.get("commits", {}))
        gap = next(
            (
                item for item in missing
                if str(item.get("definition_id", "")) not in committed
            ),
            None,
        )
    if gap is None:
        checkpoint.current_definition_gap_id = ""
        checkpoint.lean_definition_status = "NO_OPEN_DEFINITION_QUERY"
        checkpoint.stagnation_reason = "NO_OPEN_DEFINITION_QUERY"
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint, "NO_OPEN_DEFINITION_QUERY"
    gap_id = str(gap["definition_id"])
    if checkpoint.proof_state != ProofState.DEFINITION_RESOLUTION:
        checkpoint.transition(
            ProofState.DEFINITION_RESOLUTION,
            "missing-concept:autonomous-definition-resolution",
            strategy_reused=True,
        )
    evidence_refs = {
        role: item.sha256
        for role, item in checkpoint.validated_artifacts.items()
    }
    query = build_definition_query(
        gap,
        parent_hash=(
            checkpoint.parent_statement_sha256
            or checkpoint.proposition_hash
            or checkpoint.target_obligation_id
        ),
        typed_ir_hash=checkpoint.typed_ir_hash,
        auditor_hash=reference.sha256,
        critic_evidence_hashes=(
            evidence_refs.get("critic", ""),
            evidence_refs.get("adversarial_proponent", ""),
        ),
        counterexample_evidence_hashes=(
            evidence_refs.get("counterexample_worker", ""),
        ),
        theorem_dependencies=checkpoint.theorem_card_ids,
        environment_hash=base_environment_hash,
        prior_failure_hashes=checkpoint.forbidden_semantic_fingerprints,
    )
    history = [
        value for value in store.get("historical_audit", {}).values()
        if value.get("semantic_validation") == "VERIFIED"
    ]
    result = resolve_one_concept(
        query=query,
        store_path=store_path,
        project_root=project_root,
        source_context={"validated_history": history},
    )
    checkpoint.current_definition_gap_id = gap_id
    checkpoint.definition_query_hash = result.query_hash
    checkpoint.definition_candidate_count = len(result.candidate_hashes)
    checkpoint.candidate_count = len(result.candidate_hashes)
    checkpoint.selected_move_id = "RESOLVE_ONE_DEFINITION_QUERY"
    checkpoint.active_gate = "AUTONOMOUS_DEFINITION_RESOLUTION"
    checkpoint.lean_definition_status = result.status
    checkpoint.definition_store_hash = result.store_hash_after
    checkpoint.definition_environment_hash = result.environment_hash_after
    checkpoint.definition_store_hash_delta = (
        f"{result.store_hash_before}->{result.store_hash_after}"
    )
    checkpoint.definition_environment_hash_delta = (
        f"{result.environment_hash_before}->{result.environment_hash_after}"
    )
    checkpoint.definition_source_statuses = {
        item.source_id: item.status for item in result.source_statuses
    }
    checkpoint.definition_property_statuses = {
        key: dict(value) for key, value in result.property_statuses.items()
    }
    checkpoint.definition_branch_hashes = list(result.branch_hashes)
    checkpoint.definition_exhaustion_hash = result.exhaustion_hash
    checkpoint.definition_interface_hash = result.interface_hash
    committed = int(result.status == "COMMITTED")
    checkpoint.progress_vector = {
        "definitions_added": committed,
        "existing_definitions_resolved": 0,
        "lemmas_proved": 0,
        "accepted_children": 0,
        "subgoals_closed": 0,
        "verified_counterexamples": 0,
    }
    checkpoint.definitions_added += committed
    checkpoint.new_elaborated_definitions = checkpoint.definitions_added
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_resolution",
        payload={"schema_version": 1, **asdict(result)},
        dependencies=[reference.sha256],
        source_run_id=f"host:definition-resolution:{gap_id}",
    )
    checkpoint.semantic_stagnation_count = 0
    checkpoint.stagnation_reason = result.status + ":" + result.reason
    if result.query_hash not in checkpoint.forbidden_semantic_fingerprints:
        checkpoint.forbidden_semantic_fingerprints.append(result.query_hash)
    if result.status == "COMMITTED":
        checkpoint.stagnation_reason = ""
        checkpoint.transition(
            ProofState.STRATEGY_TOURNAMENT,
            "definition-resolution-commit:environment-changed",
            strategy_reused=False,
        )
    elif result.status == "PARENT_STATEMENT_UNDERSPECIFIED":
        checkpoint.transition(
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
            "definition-interpretations-change-parent-truth",
            strategy_reused=True,
        )
        checkpoint.definition_backjump_target = (
            checkpoint.parent_statement_sha256 or checkpoint.root_goal_sha256
        )
        checkpoint.transition(
            ProofState.PREMISE_AUDIT,
            "parent-statement-underspecified:premise-audit-and-backjump",
            strategy_reused=True,
        )
    elif result.status == "INTERFACE_REQUIRED":
        checkpoint.definition_backjump_target = (
            checkpoint.parent_statement_sha256 or checkpoint.root_goal_sha256
        )
        checkpoint.transition(
            ProofState.PREMISE_AUDIT,
            "definition-exhaustion:conditional-interface-axioms-required",
            strategy_reused=True,
        )
    else:
        checkpoint.definition_backjump_target = (
            checkpoint.parent_statement_sha256 or checkpoint.root_goal_sha256
        )
        checkpoint.transition(
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
            "definition-exhaustion:no-viable-interface:typed-backjump",
            strategy_reused=True,
        )
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint, result.status


def _target_is_quarantined(
    checkpoint: OrchestrationCheckpoint,
    target_ref: str,
) -> bool:
    for review in checkpoint.branch_history.values():
        if str(review.get("status", "")).upper() != "QUARANTINED":
            continue
        recorded_targets = {
            str(item)
            for key in ("plan_ids", "evidence_ids")
            for item in review.get(key, ())
        }
        if target_ref in recorded_targets:
            return True
    return False


def run_architecture_v7_entry(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    project_root: Path,
    target_ref: str,
    parent_obligation_ref: str,
    parent_complexity: int,
    event_type: StrategyEvent,
    event_id: str,
    elaborated_theorem_id: str = "",
    proposition_hash: str = "",
) -> OrchestrationCheckpoint:
    """Run exactly once per strategy event; never once per outer iteration."""
    if checkpoint.proof_state != ProofState.STRATEGY_TOURNAMENT:
        return checkpoint
    cards = build_theorem_card_index(project_root)
    card_ids = tuple(card.card_id for card in cards)
    environment_hash = pinned_environment_hash(project_root)
    (
        definitions,
        unresolved_definitions,
        definition_gap_ids,
        definition_auditor_hash,
    ) = _definition_audit(checkpoint)
    dependencies = tuple(
        reference.sha256
        for role, reference in sorted(checkpoint.validated_artifacts.items())
        if role not in {"strategy", "generator", "critic"}
    )
    evidence_refs = (*dependencies, *checkpoint.advisory_artifacts)
    plans = build_host_plans(
        target_ref=target_ref,
        parent_obligation_ref=parent_obligation_ref,
        parent_complexity=max(5, int(parent_complexity)),
        environment_hash=environment_hash,
        registered_definition_ids=definitions,
        theorem_card_ids=card_ids,
        dependency_ids=dependencies,
        evidence_refs=evidence_refs,
        unresolved_definition_ids=unresolved_definitions,
        definition_gap_ids=definition_gap_ids,
        definition_auditor_hash=definition_auditor_hash,
        elaborated_theorem_id=elaborated_theorem_id,
        proposition_hash=proposition_hash,
    )
    decisions = evaluate_feasibility(
        plans,
        registered_definition_ids=definitions,
        resolved_dependency_ids=dependencies,
        verified_theorem_card_ids=card_ids,
        allowed_assumption_ids=(),
        no_go_hashes=tuple(checkpoint.invalidated_artifacts),
    )
    tournament = run_tournament(
        event_id=event_id,
        event_type=event_type,
        plans=plans,
        decisions=decisions,
        critic_ranked_plan_ids=tuple(plan.plan_id for plan in reversed(plans)),
        critic_reason_codes=(CriticReason.MAXIMIZES_INFORMATION_GAIN,),
    )
    checkpoint.strategy_event_id = event_id
    checkpoint.strategy_event_type = event_type.value
    checkpoint.strategy_plan_ids = [plan.plan_id for plan in plans]
    checkpoint.feasible_strategy_plan_ids = [
        item.plan_id for item in decisions if item.feasible
    ]
    checkpoint.pareto_plan_ids = list(tournament.pareto_plan_ids)
    checkpoint.selected_strategy_plan_id = tournament.selected_plan_id
    checkpoint.strategy_tournament_hash = tournament.content_hash
    checkpoint.theorem_card_ids = list(card_ids)
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="strategy_tournament",
        payload={
            "schema_version": 1,
            "event_id": event_id,
            "event_type": event_type.value,
            "plan_ids": checkpoint.strategy_plan_ids,
            "plan_classes": [plan.plan_class for plan in plans],
            "plan_hashes": [plan.content_hash for plan in plans],
            "plans": [
                {
                    "plan_id": plan.plan_id,
                    "required_definition_ids": list(
                        plan.required_definition_ids
                    ),
                    "unresolved_definition_ids": list(
                        plan.unresolved_definition_ids
                    ),
                    "definition_gap_ids": list(plan.definition_gap_ids),
                    "definition_auditor_hash": plan.definition_auditor_hash,
                    "proposition_transformation_ref": (
                        plan.proposition_transformation_ref
                    ),
                    "case_partition_ids": list(plan.case_partition_ids),
                    "execution_status": plan.execution_status,
                }
                for plan in plans
            ],
            "feasibility": [
                {
                    "plan_id": item.plan_id,
                    "feasible": item.feasible,
                    "reason_codes": list(item.reason_codes),
                    "score_explanation": dict(item.score_explanation),
                }
                for item in decisions
            ],
            "pareto_plan_ids": list(tournament.pareto_plan_ids),
            "selected_plan_id": tournament.selected_plan_id,
            "content_hash": tournament.content_hash,
        },
        dependencies=list(dependencies),
        source_run_id=f"host:{event_id}",
    )
    precontract_reasons = []
    if unresolved_definitions:
        precontract_reasons.append("MISSING_DEFINITION")
    if not elaborated_theorem_id or not proposition_hash:
        precontract_reasons.append("UNELABORATED_TARGET")
    if _target_is_quarantined(checkpoint, target_ref):
        precontract_reasons.append("QUARANTINED_PARENT_REQUIRES_TYPED_REFRAME")
    if precontract_reasons:
        checkpoint.research_contract_id = ""
        checkpoint.research_contract_hash = ""
        checkpoint.research_contract_rejection_codes = []
        checkpoint.transition(
            ProofState.DECOMPOSER,
            "precontract-semantic-routing:" + ",".join(precontract_reasons),
            strategy_reused=True,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
    selected = next(
        (plan for plan in plans if plan.plan_id == tournament.selected_plan_id),
        None,
    )
    if selected is None:
        checkpoint.research_contract_rejection_codes = []
        checkpoint.transition(
            ProofState.DECOMPOSER,
            "precontract-semantic-routing:NO_EXECUTABLE_PLAN",
            strategy_reused=True,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
    checkpoint.transition(
        ProofState.RESEARCH_CONTRACT_GATE,
        "strategy-tournament-complete:elaborated-target",
        strategy_reused=False,
    )
    contract_decision = gate_research_contract(
        selected,
        elaborated_theorem_id=elaborated_theorem_id,
        elaborated_proposition_hash=proposition_hash,
        proof_obligation_id=target_ref,
        registered_definition_ids=definitions,
        resolved_dependency_ids=dependencies,
        verified_theorem_card_ids=card_ids,
        allowed_assumption_ids=(),
        environment_hash=environment_hash,
        expected_plan_hash=selected.content_hash,
    )
    checkpoint.research_contract_rejection_codes = list(
        contract_decision.reason_codes if not contract_decision.accepted else (),
    )
    if contract_decision.contract is not None:
        contract = contract_decision.contract
        checkpoint.research_contract_id = contract.contract_id
        checkpoint.research_contract_hash = contract.content_hash
        persist_validated_artifact(
            checkpoint_path,
            checkpoint,
            role="research_contract",
            payload={
                **contract.__dict__,
                "schema_version": 1,
            },
            dependencies=[tournament.content_hash],
            source_run_id=f"host:{event_id}:contract",
        )
    checkpoint.transition(
        ProofState(contract_decision.route_state),
        (
            "research-contract-accepted"
            if contract_decision.accepted
            else "research-contract-rejected:"
            + ",".join(contract_decision.reason_codes)
        ),
    )
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint
