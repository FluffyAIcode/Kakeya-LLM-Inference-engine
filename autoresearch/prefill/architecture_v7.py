"""Single architecture-7 entry point for tournament and contract routing."""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from autoresearch.prefill.cursor_strategy import (
    STRATEGY_INTENT_UNMAPPABLE,
    STRATEGY_PROVIDER_UNAVAILABLE,
    CursorStrategyAdapter,
    StrategyProviderError,
    compile_memo_to_plan_id,
)
from autoresearch.prefill.definition_resolution import (
    build_definition_query,
    load_resolution_store,
    resolve_one_concept,
)
from autoresearch.prefill.decomposition_exploration import (
    gate_decomposition_exploration,
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
    PlanClass,
    StrategyEvent,
    build_definition_resolution_plan,
    compile_strategy_intent,
    evaluate_feasibility,
    run_tournament,
)
from autoresearch.prefill.target_context import (
    activate_target_context,
    update_active_context,
)
from autoresearch.prefill.theorem_cards import (
    build_theorem_card_index,
    pinned_environment_hash,
)
from autoresearch.prefill.typed_interface_resolution import (
    build_target_interface_registry,
    resolve_target_interface,
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
    interface_strategy_adapter: CursorStrategyAdapter | None = None,
) -> tuple[OrchestrationCheckpoint, str]:
    """Execute exactly one Autonomous Definition Resolution transaction."""
    if checkpoint.proof_state not in {
        ProofState.DEFINITION_RESOLUTION,
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
        if (
            checkpoint.current_definition_gap_id
            == "GAP_ELABORATED_TARGET_REQUIRED"
        ):
            target_statement = str(
                checkpoint.target_evidence.get(
                    "EVIDENCE_TARGET_STATEMENT", "",
                ),
            )
            candidates, source_statuses, theorem_card_ids = (
                build_target_interface_registry(
                    target_ref=checkpoint.target_obligation_id,
                    target_statement=target_statement,
                    target_evidence=checkpoint.target_evidence,
                    auditor_hash=reference.sha256,
                    project_root=project_root,
                )
            )
            ranked_ids: tuple[str, ...] = ()
            provider_provenance: dict[str, object] = {
                "status": "NOT_REQUESTED",
            }
            if interface_strategy_adapter is not None:
                memo, telemetry = interface_strategy_adapter.advise(
                    evidence={
                        "target_ref": checkpoint.target_obligation_id,
                        "target_statement": target_statement,
                        "definition_auditor_hash": reference.sha256,
                        "target_context_hash": checkpoint.target_context_hash,
                        "candidate_schemas": [
                            {
                                "short_id": item.short_id,
                                "candidate_kind": item.candidate_kind,
                                "proposition_schema": item.proposition_schema,
                                "host_feasible": item.feasible,
                                "host_rejection_codes": item.rejection_codes,
                            }
                            for item in candidates
                        ],
                        "source_statuses": [
                            asdict(item) for item in source_statuses
                        ],
                        "theorem_card_ids": theorem_card_ids,
                    },
                    registered_plan_ids=tuple(
                        item.short_id for item in candidates
                    ),
                )
                ranked_ids = (
                    compile_memo_to_plan_id(
                        memo,
                        tuple(item.short_id for item in candidates),
                    ),
                )
                provider_provenance = {
                    **asdict(telemetry),
                    "selected_candidate_id": ranked_ids[0],
                }
            result = resolve_target_interface(
                target_ref=checkpoint.target_obligation_id,
                target_statement=target_statement,
                target_evidence=checkpoint.target_evidence,
                auditor_hash=reference.sha256,
                environment_hash=checkpoint.target_environment_hash,
                project_root=project_root,
                ranked_candidate_ids=ranked_ids,
                provider_provenance=provider_provenance,
            )
            source_run_id = str(
                provider_provenance.get("run_id")
                or "host:target-interface-exhaustion"
            )
            persist_validated_artifact(
                checkpoint_path,
                checkpoint,
                role="typed_interface_resolution",
                payload=asdict(result),
                dependencies=[reference.sha256],
                source_run_id=source_run_id,
            )
            checkpoint.selected_move_id = "EXHAUST_TARGET_INTERFACE_REGISTRY"
            checkpoint.active_gate = "TARGET_TYPED_INTERFACE_GATE"
            checkpoint.lean_definition_status = result.status
            checkpoint.definition_exhaustion_hash = result.exhaustion_hash
            checkpoint.stagnation_reason = (
                result.status + ":" + result.terminal_reason
            )
            checkpoint.progress_vector = {
                "definitions_added": 0,
                "existing_definitions_resolved": 0,
                "lemmas_proved": 0,
                "accepted_children": 0,
                "subgoals_closed": 0,
                "verified_counterexamples": 0,
            }
            persist_validated_artifact(
                checkpoint_path,
                checkpoint,
                role="interface_exhaustion_certificate",
                payload={
                    **asdict(result),
                    "certificate_kind": (
                        "target_interface_exhaustion_certificate"
                    ),
                    "typed_backjump_target": "ROOT_UNAVAILABLE",
                    "quarantine_target": checkpoint.target_obligation_id,
                    "proof_search_allowed": False,
                    "oprover_allowed": False,
                },
                dependencies=[
                    reference.sha256,
                    checkpoint.validated_artifacts[
                        "typed_interface_resolution"
                    ].sha256,
                ],
                source_run_id=source_run_id,
            )
            checkpoint.premise_outcome_type = (
                ProofState.PARENT_STATEMENT_UNDERSPECIFIED.value
            )
            checkpoint.premise_outcome_owner = "target_typed_interface_gate"
            checkpoint.premise_decision = result.status
            checkpoint.premise_confidence = 1.0
            checkpoint.premise_evidence = {
                "exhaustion_hash": result.exhaustion_hash,
                "target_statement_hash": result.target_statement_hash,
                "auditor_hash": result.auditor_hash,
                "candidate_hashes": [
                    item.content_hash for item in result.candidates
                ],
                "source_status_hash": hashlib.sha256(json.dumps(
                    [asdict(item) for item in result.source_statuses],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()).hexdigest(),
            }
            checkpoint.premise_backjump_target = "ROOT_UNAVAILABLE"
            checkpoint.branch_history[
                "interface-exhaustion:" + result.exhaustion_hash
            ] = {
                "status": "QUARANTINED",
                "reason": result.terminal_reason,
                "plan_ids": [checkpoint.target_obligation_id],
                "evidence_ids": [result.exhaustion_hash],
                "source_run_id": source_run_id,
                "typed_backjump_target": "ROOT_UNAVAILABLE",
                "created_at": checkpoint.updated_at,
            }
            checkpoint.transition(
                ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
                "target-interface-exhausted:typed-backjump",
                source_run_id=source_run_id,
                strategy_reused=False,
            )
            save_checkpoint(checkpoint_path, checkpoint)
            checkpoint.blocked_reason = (
                "MATHEMATICAL_TERMINAL_BLOCKER:"
                + result.exhaustion_hash
            )
            checkpoint.transition(
                ProofState.BLOCKED,
                checkpoint.blocked_reason,
                source_run_id=source_run_id,
                strategy_reused=False,
            )
            save_checkpoint(checkpoint_path, checkpoint)
            return checkpoint, result.status
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
    elif result.status in {
        "PARENT_STATEMENT_UNDERSPECIFIED",
        "INTERFACE_REQUIRED",
        "IDENTICAL_QUERY_EXHAUSTED",
    }:
        checkpoint.definition_backjump_target = (
            checkpoint.parent_statement_sha256 or checkpoint.root_goal_sha256
        )
        checkpoint.premise_outcome_type = (
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED.value
        )
        checkpoint.premise_outcome_owner = "definition_resolution"
        checkpoint.premise_decision = result.status
        checkpoint.premise_confidence = 1.0
        checkpoint.premise_evidence = {
            "query_hash": result.query_hash,
            "exhaustion_hash": result.exhaustion_hash,
            "interface_hash": result.interface_hash,
            "reason": result.reason,
        }
        checkpoint.premise_backjump_target = (
            checkpoint.definition_backjump_target
        )
        checkpoint.premise_outcome_fingerprint = result.query_hash
        if result.query_hash not in checkpoint.consumed_premise_fingerprints:
            checkpoint.consumed_premise_fingerprints.append(result.query_hash)
        checkpoint.transition(
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
            "definition-resolution-exhausted:parent-underspecified",
            strategy_reused=False,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        checkpoint.transition(
            ProofState.STRATEGY_TOURNAMENT,
            "parent-underspecified:typed-backjump-new-target",
            strategy_reused=False,
        )
    else:
        checkpoint.definition_backjump_target = (
            checkpoint.parent_statement_sha256 or checkpoint.root_goal_sha256
        )
        checkpoint.premise_outcome_type = (
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED.value
        )
        checkpoint.premise_outcome_owner = "definition_resolution"
        checkpoint.premise_decision = result.status
        checkpoint.premise_confidence = 1.0
        checkpoint.premise_evidence = {
            "query_hash": result.query_hash,
            "exhaustion_hash": result.exhaustion_hash,
            "reason": result.reason,
        }
        checkpoint.premise_backjump_target = (
            checkpoint.definition_backjump_target
        )
        checkpoint.premise_outcome_fingerprint = result.query_hash
        if result.query_hash not in checkpoint.consumed_premise_fingerprints:
            checkpoint.consumed_premise_fingerprints.append(result.query_hash)
        checkpoint.transition(
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
            "definition-exhaustion:no-viable-interface:typed-backjump",
            strategy_reused=False,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        checkpoint.transition(
            ProofState.STRATEGY_TOURNAMENT,
            "parent-underspecified:typed-backjump-new-target",
            strategy_reused=False,
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
    strategy_adapter: CursorStrategyAdapter | None = None,
    target_statement: str = "",
    target_evidence: Mapping[str, object] | None = None,
) -> OrchestrationCheckpoint:
    """Run exactly once per strategy event; never once per outer iteration."""
    if checkpoint.proof_state != ProofState.STRATEGY_TOURNAMENT:
        return checkpoint
    target_statement = str(target_statement or checkpoint.target_statement).strip()
    if not target_statement:
        target_statement = f"Unelaborated proof obligation {target_ref}"
    environment_hash = pinned_environment_hash(project_root)
    current_input_context = bool(
        checkpoint.target_context_hash
        and checkpoint.target_obligation_id == target_ref
        and checkpoint.parent_statement_sha256
        == hashlib.sha256(target_statement.encode()).hexdigest()
        and checkpoint.target_environment_hash == environment_hash
    )
    if not current_input_context:
        _context, context_changed = activate_target_context(
            checkpoint_path,
            checkpoint,
            target_obligation_id=target_ref,
            statement=target_statement,
            environment_hash=environment_hash,
            strategy_plan_hash="STRATEGY_PENDING",
            evidence=dict(target_evidence or {}),
        )
        if context_changed:
            save_checkpoint(checkpoint_path, checkpoint)
    all_cards = build_theorem_card_index(project_root)
    statement_words = {
        word.lower().strip(".,'\"()[]{}")
        for word in target_statement.split()
        if len(word) >= 5
    }
    cards = tuple(card for card in all_cards if statement_words.intersection({
        *card.applicability_tags,
        *card.required_hypotheses,
        *card.description.lower().split(),
    }))
    card_ids = tuple(card.card_id for card in cards)
    (
        definitions,
        unresolved_definitions,
        definition_gap_ids,
        definition_auditor_hash,
    ) = _definition_audit(checkpoint)
    dependencies = tuple(
        reference.sha256
        for role, reference in sorted(checkpoint.validated_artifacts.items())
        if role not in {
            "strategy", "strategy_tournament", "generator", "critic",
        }
    )
    evidence_refs = (*dependencies, *checkpoint.advisory_artifacts)
    if strategy_adapter is None:
        checkpoint.strategy_run_status = "STRATEGY_PROVIDER_UNAVAILABLE"
        checkpoint.adapter_blocked(
            "STRATEGY_PROVIDER_UNAVAILABLE:CURSOR_REQUIRED",
            status="INTEGRATION_BLOCKED",
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
    checkpoint.strategy_provider = "cursor-sdk"
    checkpoint.strategy_provider_configured = strategy_adapter.configured()
    checkpoint.strategy_model_id = strategy_adapter.model_id
    if not checkpoint.strategy_provider_configured:
        checkpoint.strategy_run_status = STRATEGY_PROVIDER_UNAVAILABLE
        checkpoint.adapter_blocked(
            f"{STRATEGY_PROVIDER_UNAVAILABLE}:CONFIGURATION_REQUIRED",
            status="INTEGRATION_BLOCKED",
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
    evidence_ids = tuple(sorted({
        *map(str, evidence_refs),
        "EVIDENCE_TARGET_STATEMENT",
    }))
    theorem_tags = tuple(sorted({
        tag for card in cards for tag in card.applicability_tags
    }))
    proof_plan_classes = tuple(
        item for item in PlanClass
        if item is not PlanClass.DEFINITION_RESOLUTION_PLAN
    )
    criteria = {
        "falsification_criterion_id": tuple(
            f"FALSIFY_{item.value}" for item in proof_plan_classes
        ),
        "success_criterion_id": tuple(
            f"SUCCESS_{item.value}" for item in proof_plan_classes
        ),
        "abandonment_criterion_id": tuple(
            f"ABANDON_{item.value}" for item in proof_plan_classes
        ),
    }
    try:
        remediation_only = not elaborated_theorem_id or not proposition_hash
        if remediation_only:
            remediation = build_definition_resolution_plan(
                target_ref=target_ref,
                parent_obligation_ref=parent_obligation_ref,
                parent_complexity=max(5, int(parent_complexity)),
                environment_hash=environment_hash,
                registered_definition_ids=definitions,
                unresolved_definition_ids=unresolved_definitions,
                definition_gap_ids=definition_gap_ids,
                definition_auditor_hash=definition_auditor_hash,
                dependency_ids=dependencies,
                evidence_refs=evidence_ids,
            )
        memo, telemetry = strategy_adapter.advise(
            evidence={
                "event_id": event_id,
                "event_type": event_type.value,
                "target_ref": target_ref,
                "target_statement": target_statement,
                "parent_obligation_ref": parent_obligation_ref,
                "elaborated_theorem_id": elaborated_theorem_id,
                "proposition_hash": proposition_hash,
                "definition_ids": definitions,
                "unresolved_definition_ids": unresolved_definitions,
                "gap_refs": definition_gap_ids,
                "theorem_card_ids": card_ids,
                "evidence_refs": evidence_ids,
            },
            registered_plan_ids=(
                (remediation.plan_id,)
                if remediation_only
                else tuple(item.value for item in proof_plan_classes)
            ),
        )
        if remediation_only:
            selected_id = compile_memo_to_plan_id(memo, (remediation.plan_id,))
            if selected_id != remediation.plan_id:
                raise ValueError("REMEDIATION_PLAN_SELECTION_MISMATCH")
            plans = (remediation,)
            intent_hash = hashlib.sha256(json.dumps({
                "selected_plan_id": selected_id,
                "selected_plan_hash": remediation.content_hash,
                "target_ref": target_ref,
                "definition_auditor_hash": definition_auditor_hash,
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            intent_run_id = telemetry.run_id
        else:
            intent = strategy_adapter.extract_intent(
                memo,
                registered_fields={
                "plan_class": tuple(item.value for item in proof_plan_classes),
                "target_ref": (target_ref,),
                "gap_ref": tuple(definition_gap_ids),
                "move_family": (
                    "MOVE_DIRECT", "MOVE_FALSIFY", "MOVE_REDUCE",
                    "MOVE_REFRAME", "MOVE_REGISTRY_EXPANSION",
                    "MOVE_EXPLORE_SUBPROBLEMS",
                ),
                "theorem_tag": theorem_tags,
                "evidence_ref": evidence_ids,
                    **criteria,
                },
            )
            cards_by_tag = {
                tag: tuple(
                    card.card_id for card in cards
                    if tag in card.applicability_tags
                )
                for tag in theorem_tags
            }
            plans = compile_strategy_intent(
                intent,
                target_ref=target_ref,
                parent_obligation_ref=parent_obligation_ref,
                parent_complexity=max(5, int(parent_complexity)),
                environment_hash=environment_hash,
                registered_definition_ids=definitions,
                unresolved_definition_ids=unresolved_definitions,
                registered_gap_ids=definition_gap_ids,
                theorem_cards_by_tag=cards_by_tag,
                registered_evidence_refs=evidence_ids,
                dependency_ids=dependencies,
                definition_auditor_hash=definition_auditor_hash,
                elaborated_theorem_id=elaborated_theorem_id,
                proposition_hash=proposition_hash,
            )
            intent_hash = intent.intent_hash
            intent_run_id = intent.provider_run_id
        checkpoint.strategy_run_status = telemetry.status
        checkpoint.strategy_agent_id = telemetry.agent_id
        checkpoint.strategy_run_id = telemetry.run_id
        checkpoint.strategy_prompt_hash = telemetry.prompt_hash
        checkpoint.strategy_evidence_hash = telemetry.evidence_hash
        checkpoint.strategy_memo_hash = telemetry.memo_hash
        checkpoint.strategy_intent_hash = intent_hash
        checkpoint.strategy_intent_run_id = intent_run_id
        checkpoint.strategy_intent_status = "MAPPED"
        checkpoint.strategy_latency_ms = telemetry.latency_ms
        checkpoint.strategy_provider_configured = telemetry.configured
        if checkpoint.adapter_status:
            checkpoint.clear_adapter_blocked("cursor-strategy-intent-mapped")
    except StrategyProviderError as exc:
        checkpoint.strategy_run_status = exc.code
        checkpoint.adapter_blocked(
            f"{exc.code}:{exc.classification}",
            status="INTEGRATION_BLOCKED",
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
    except ValueError as exc:
        checkpoint.strategy_intent_status = STRATEGY_INTENT_UNMAPPABLE
        checkpoint.strategy_run_status = STRATEGY_INTENT_UNMAPPABLE
        checkpoint.current_definition_gap_id = "REGISTRY_EVIDENCE_EXPANSION"
        checkpoint.stagnation_reason = str(exc)
        checkpoint.transition(
            ProofState.DEFINITION_RESOLUTION,
            f"{STRATEGY_INTENT_UNMAPPABLE}:{exc}",
            strategy_reused=False,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
    decisions = evaluate_feasibility(
        plans,
        registered_definition_ids=definitions,
        resolved_dependency_ids=dependencies,
        verified_theorem_card_ids=card_ids,
        allowed_assumption_ids=(),
        no_go_hashes=tuple(checkpoint.invalidated_artifacts),
    )
    critic_ranked_ids = tuple(plan.plan_id for plan in plans)
    tournament = run_tournament(
        event_id=event_id,
        event_type=event_type,
        plans=plans,
        decisions=decisions,
        critic_ranked_plan_ids=critic_ranked_ids,
        critic_reason_codes=(CriticReason.MAXIMIZES_INFORMATION_GAIN,),
    )
    selected_plan = next(
        (plan for plan in plans if plan.plan_id == tournament.selected_plan_id),
        None,
    )
    selected_hash = selected_plan.content_hash if selected_plan else ""
    # The active TargetContext is immutable input evidence. Selection is a
    # downstream pointer and must never rewrite/rebind its upstream auditor.
    checkpoint.target_strategy_plan_hash = (
        selected_hash or "NO_FEASIBLE_PLAN"
    )
    checkpoint.target_evidence = {
        **dict(target_evidence or checkpoint.target_evidence),
        "EVIDENCE_TARGET_STATEMENT": target_statement,
    }
    checkpoint.strategy_event_id = event_id
    checkpoint.strategy_event_type = event_type.value
    checkpoint.strategy_plan_ids = [plan.plan_id for plan in plans]
    checkpoint.strategy_plan_hashes = [plan.content_hash for plan in plans]
    checkpoint.feasible_strategy_plan_ids = [
        item.plan_id for item in decisions if item.feasible
    ]
    checkpoint.pareto_plan_ids = list(tournament.pareto_plan_ids)
    checkpoint.selected_strategy_plan_id = tournament.selected_plan_id
    checkpoint.selected_strategy_plan_hash = selected_hash
    checkpoint.strategy_tournament_hash = tournament.content_hash
    checkpoint.theorem_card_ids = list(card_ids)
    checkpoint.target_gap_ids = list(definition_gap_ids)
    checkpoint.strategy_selection_provenance = {
        "provider": checkpoint.strategy_provider,
        "provider_run_id": checkpoint.strategy_run_id,
        "intent_run_id": checkpoint.strategy_intent_run_id,
        "memo_hash": checkpoint.strategy_memo_hash,
        "intent_hash": checkpoint.strategy_intent_hash,
        "selected_plan_id": tournament.selected_plan_id,
        "selected_plan_hash": selected_hash,
        "target_context_hash": checkpoint.target_context_hash,
    }
    tournament_ref = persist_validated_artifact(
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
                    "plan_kind": plan.plan_class,
                    "target_ref": plan.target_ref,
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
                    "success_criterion_id": plan.success_criterion_id,
                    "abandonment_criterion_id": (
                        plan.abandonment_criterion_id
                    ),
                    "next_gate": (
                        "RESEARCH_CONTRACT_GATE"
                        if plan.plan_class
                        == PlanClass.DEFINITION_RESOLUTION_PLAN.value
                        else "DECOMPOSITION_EXPLORATION"
                        if plan.plan_class
                        == PlanClass.DECOMPOSE_TO_SUBPROBLEMS.value
                        else "PROOF_SEARCH"
                    ),
                    "proof_search_allowed": (
                        plan.plan_class not in {
                            PlanClass.DEFINITION_RESOLUTION_PLAN.value,
                            PlanClass.DECOMPOSE_TO_SUBPROBLEMS.value,
                        }
                    ),
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
        save=False,
    )
    update_active_context(
        checkpoint_path,
        checkpoint,
        gap_ids=definition_gap_ids,
        definition_ids=definitions,
        theorem_card_ids=card_ids,
        artifact_hashes=(
            checkpoint.validated_artifacts["strategy_tournament"].sha256,
        ),
    )
    # Memo metadata, constrained intent, compiled plans, feasibility, selection,
    # and the reachable artifact pointer become visible in one checkpoint swap.
    save_checkpoint(checkpoint_path, checkpoint)
    selected = next(
        (plan for plan in plans if plan.plan_id == tournament.selected_plan_id),
        None,
    )
    if (
        selected is not None
        and selected.plan_class == PlanClass.DECOMPOSE_TO_SUBPROBLEMS.value
    ):
        exploration = gate_decomposition_exploration(
            selected,
            target_obligation_id=target_ref,
            target_context_hash=checkpoint.target_context_hash,
            proposition_hash=proposition_hash,
            evidence_refs=selected.evidence_refs,
            no_go_refs=selected.known_no_go_refs,
            theorem_card_ids=selected.theorem_card_ids,
            candidate_budget=selected.candidate_budget,
        )
        checkpoint.exploration_contract_id = exploration.contract_id
        checkpoint.exploration_contract_hash = exploration.content_hash
        persist_validated_artifact(
            checkpoint_path,
            checkpoint,
            role="decomposition_exploration_contract",
            payload=asdict(exploration),
            dependencies=[tournament_ref.sha256],
            source_run_id=f"host:{event_id}:exploration-contract",
        )
        checkpoint.transition(
            ProofState.DECOMPOSITION_EXPLORATION,
            "decomposition-exploration-contract-accepted",
            strategy_reused=False,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
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
        if selected_plan and (
            selected_plan.plan_class
            == PlanClass.DEFINITION_RESOLUTION_PLAN.value
        ):
            checkpoint.current_definition_gap_id = (
                selected_plan.definition_gap_ids[0]
            )
            checkpoint.transition(
                ProofState.DEFINITION_RESOLUTION,
                "typed-definition-resolution-plan:"
                + ",".join(precontract_reasons),
                strategy_reused=True,
            )
        else:
            checkpoint.transition(
                ProofState.DECOMPOSER,
                "precontract-semantic-routing:" + ",".join(precontract_reasons),
                strategy_reused=True,
            )
        save_checkpoint(checkpoint_path, checkpoint)
        return checkpoint
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
            dependencies=[tournament_ref.sha256],
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
