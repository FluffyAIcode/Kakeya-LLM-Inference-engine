"""Evidence-driven, host-owned strategy tournament.

Models may rank the short IDs produced here, but cannot author plans, syntax,
assumptions, or feasibility decisions.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping


TOURNAMENT_VERSION = 2
MIGRATION_EVENT = "strategy_tournament_stepwise_generator_v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


class StrategyEvent(str, Enum):
    INITIAL_BRANCH = "INITIAL_BRANCH"
    CONFIRMED_BRANCH_FAILURE = "CONFIRMED_BRANCH_FAILURE"
    PREMISE_INVALIDATION = "PREMISE_INVALIDATION"
    TARGET_CHANGE = "TARGET_CHANGE"
    MATHEMATICAL_STAGNATION = "MATHEMATICAL_STAGNATION"


class PlanClass(str, Enum):
    DIRECT_PROOF = "DIRECT_PROOF"
    DISPROOF_OR_COUNTEREXAMPLE = "DISPROOF_OR_COUNTEREXAMPLE"
    REDUCTION_TO_KNOWN_RESULT = "REDUCTION_TO_KNOWN_RESULT"
    REFRAME_DEFINITIONS_OR_REPRESENTATION = (
        "REFRAME_DEFINITIONS_OR_REPRESENTATION"
    )


class FeasibilityReason(str, Enum):
    FEASIBLE = "FEASIBLE"
    UNMET_DEFINITION = "UNMET_DEFINITION"
    UNRESOLVED_DEPENDENCY = "UNRESOLVED_DEPENDENCY"
    HIDDEN_ASSUMPTION = "HIDDEN_ASSUMPTION"
    CIRCULAR_REDUCTION = "CIRCULAR_REDUCTION"
    MISSING_FALSIFIER = "MISSING_FALSIFIER"
    MISSING_THEOREM_SUPPORT = "MISSING_THEOREM_SUPPORT"
    DUPLICATE_ROUTE = "DUPLICATE_ROUTE"
    NO_GO_ROUTE = "NO_GO_ROUTE"
    NON_REDUCING_TARGET = "NON_REDUCING_TARGET"
    PLANNING_ONLY = "PLANNING_ONLY"


class CriticReason(str, Enum):
    MAXIMIZES_INFORMATION_GAIN = "MAXIMIZES_INFORMATION_GAIN"
    STRONGEST_THEOREM_SUPPORT = "STRONGEST_THEOREM_SUPPORT"
    LOWEST_COMPLEXITY = "LOWEST_COMPLEXITY"
    LOWEST_RISK = "LOWEST_RISK"
    STRICTEST_REDUCTION = "STRICTEST_REDUCTION"


class PlanExecutionStatus(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    PLANNING_ONLY = "PLANNING_ONLY"


@dataclass(frozen=True)
class LemmaNode:
    lemma_id: str
    dependency_ids: tuple[str, ...]
    target_ref: str
    complexity: int


@dataclass(frozen=True)
class StrategyPlan:
    plan_id: str
    plan_class: str
    target_ref: str
    required_definition_ids: tuple[str, ...]
    theorem_card_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    lemma_graph: tuple[LemmaNode, ...]
    falsification_test_id: str
    success_criterion_id: str
    abandonment_criterion_id: str
    expected_information_gain: int
    assumption_ids: tuple[str, ...]
    restriction_ids: tuple[str, ...]
    parent_obligation_ref: str
    parent_complexity: int
    target_complexity: int
    risk: int
    source_move_id: str
    environment_hash: str
    evidence_refs: tuple[str, ...]
    unresolved_definition_ids: tuple[str, ...]
    definition_gap_ids: tuple[str, ...]
    definition_auditor_hash: str
    proposition_transformation_ref: str
    case_partition_ids: tuple[str, ...]
    execution_status: str
    content_hash: str


@dataclass(frozen=True)
class FeasibilityDecision:
    plan_id: str
    feasible: bool
    reason_codes: tuple[str, ...]
    deterministic_score: tuple[int, int, int, int, int]
    score_explanation: Mapping[str, int]


@dataclass(frozen=True)
class TournamentResult:
    event_id: str
    event_type: str
    plans: tuple[StrategyPlan, ...]
    decisions: tuple[FeasibilityDecision, ...]
    pareto_plan_ids: tuple[str, ...]
    critic_ranked_plan_ids: tuple[str, ...]
    critic_reason_codes: tuple[str, ...]
    selected_plan_id: str
    content_hash: str


@dataclass(frozen=True)
class BranchEvidence:
    evidence_id: str
    event_type: str
    accepted_child_delta: int = 0
    lean_theorem_delta: int = 0
    verified_counterexample_delta: int = 0
    semantic_failure_delta: int = 0
    reason_codes: tuple[str, ...] = ()
    provenance_hash: str = ""


@dataclass
class BranchHistory:
    branch_id: str
    plan_ids: list[str] = field(default_factory=list)
    evidence: list[BranchEvidence] = field(default_factory=list)
    score: int = 0
    status: str = "ACTIVE"
    disposition_reason_codes: list[str] = field(default_factory=list)


def strategy_event_due(
    event_type: StrategyEvent | None,
    *,
    rounds_without_verified_progress: int,
    stagnation_threshold: int,
) -> bool:
    """Events are explicit; outer-loop iteration alone is never a trigger."""
    if event_type is None:
        return rounds_without_verified_progress >= stagnation_threshold
    return event_type in set(StrategyEvent)


def build_host_plans(
    *,
    target_ref: str,
    parent_obligation_ref: str,
    parent_complexity: int,
    environment_hash: str,
    registered_definition_ids: Iterable[str],
    theorem_card_ids: Iterable[str],
    dependency_ids: Iterable[str],
    evidence_refs: Iterable[str],
    unresolved_definition_ids: Iterable[str] = (),
    definition_gap_ids: Iterable[str] = (),
    definition_auditor_hash: str = "",
    elaborated_theorem_id: str = "",
    proposition_hash: str = "",
) -> tuple[StrategyPlan, ...]:
    """Construct one independent host plan per mandatory plan class."""
    registered = tuple(sorted(set(registered_definition_ids)))
    unresolved = tuple(sorted(set(unresolved_definition_ids)))
    definitions = tuple(sorted(set((*registered, *unresolved))))
    gaps = tuple(sorted(set(definition_gap_ids)))
    cards = tuple(sorted(set(theorem_card_ids)))
    dependencies = tuple(sorted(set(dependency_ids)))
    evidence = tuple(sorted(set(evidence_refs)))
    specs = (
        (PlanClass.DIRECT_PROOF, "MOVE_DIRECT", cards[:1], 3, 2),
        (PlanClass.DISPROOF_OR_COUNTEREXAMPLE, "MOVE_FALSIFY", (), 5, 2),
        (PlanClass.REDUCTION_TO_KNOWN_RESULT, "MOVE_REDUCE", cards[:2], 4, 3),
        (
            PlanClass.REFRAME_DEFINITIONS_OR_REPRESENTATION,
            "MOVE_REFRAME", (), 5, 1,
        ),
    )
    plans = []
    for index, (kind, move_id, support, gain, risk) in enumerate(specs, 1):
        target_complexity = max(0, parent_complexity - index)
        lemma_id = f"L{index}"
        transformation_ref = (
            "typed-reframe:" + proposition_hash
            if (
                kind == PlanClass.REFRAME_DEFINITIONS_OR_REPRESENTATION
                and elaborated_theorem_id
                and proposition_hash
            )
            else ""
        )
        case_partitions = (
            (f"case:{proposition_hash[:20]}",)
            if transformation_ref else ()
        )
        execution_status = (
            PlanExecutionStatus.EXECUTABLE.value
            if elaborated_theorem_id and proposition_hash
            else PlanExecutionStatus.PLANNING_ONLY.value
        )
        canonical = {
            "version": TOURNAMENT_VERSION,
            "plan_class": kind.value,
            "target_ref": target_ref,
            "definitions": definitions,
            "unresolved_definitions": unresolved,
            "definition_gap_ids": gaps,
            "definition_auditor_hash": definition_auditor_hash,
            "theorem_cards": support,
            "dependencies": dependencies,
            "lemma_graph": [{
                "lemma_id": lemma_id,
                "dependency_ids": dependencies,
                "target_ref": target_ref,
                "complexity": target_complexity,
            }],
            "falsification_test_id": f"FALSIFY_{kind.value}",
            "success_criterion_id": f"SUCCESS_{kind.value}",
            "abandonment_criterion_id": f"ABANDON_{kind.value}",
            "information_gain": gain,
            "assumptions": (),
            "restrictions": (),
            "parent": parent_obligation_ref,
            "parent_complexity": parent_complexity,
            "target_complexity": target_complexity,
            "risk": risk,
            "source_move_id": move_id,
            "environment_hash": environment_hash,
            "evidence_refs": evidence,
            "proposition_transformation_ref": transformation_ref,
            "case_partition_ids": case_partitions,
            "execution_status": execution_status,
        }
        content_hash = _digest(canonical)
        plans.append(StrategyPlan(
            plan_id=f"P{index}",
            plan_class=kind.value,
            target_ref=target_ref,
            required_definition_ids=definitions,
            theorem_card_ids=tuple(support),
            dependency_ids=dependencies,
            lemma_graph=(LemmaNode(
                lemma_id, dependencies, target_ref, target_complexity,
            ),),
            falsification_test_id=canonical["falsification_test_id"],
            success_criterion_id=canonical["success_criterion_id"],
            abandonment_criterion_id=canonical["abandonment_criterion_id"],
            expected_information_gain=gain,
            assumption_ids=(),
            restriction_ids=(),
            parent_obligation_ref=parent_obligation_ref,
            parent_complexity=parent_complexity,
            target_complexity=target_complexity,
            risk=risk,
            source_move_id=move_id,
            environment_hash=environment_hash,
            evidence_refs=evidence,
            unresolved_definition_ids=unresolved,
            definition_gap_ids=gaps,
            definition_auditor_hash=definition_auditor_hash,
            proposition_transformation_ref=transformation_ref,
            case_partition_ids=case_partitions,
            execution_status=execution_status,
            content_hash=content_hash,
        ))
    return tuple(plans)


def evaluate_feasibility(
    plans: Iterable[StrategyPlan],
    *,
    registered_definition_ids: Iterable[str],
    resolved_dependency_ids: Iterable[str],
    verified_theorem_card_ids: Iterable[str],
    allowed_assumption_ids: Iterable[str],
    no_go_hashes: Iterable[str] = (),
) -> tuple[FeasibilityDecision, ...]:
    definitions = set(registered_definition_ids)
    dependencies = set(resolved_dependency_ids)
    cards = set(verified_theorem_card_ids)
    assumptions = set(allowed_assumption_ids)
    no_go = set(no_go_hashes)
    seen_routes: set[tuple[str, str, str]] = set()
    decisions = []
    for plan in plans:
        reasons: list[str] = []
        if not set(plan.required_definition_ids) <= definitions:
            reasons.append(FeasibilityReason.UNMET_DEFINITION.value)
        if plan.execution_status != PlanExecutionStatus.EXECUTABLE.value:
            reasons.append(FeasibilityReason.PLANNING_ONLY.value)
        if (
            plan.plan_class
            == PlanClass.REFRAME_DEFINITIONS_OR_REPRESENTATION.value
            and plan.execution_status == PlanExecutionStatus.EXECUTABLE.value
            and not (
                plan.proposition_transformation_ref
                or plan.case_partition_ids
            )
        ):
            reasons.append(FeasibilityReason.PLANNING_ONLY.value)
        if not set(plan.dependency_ids) <= dependencies:
            reasons.append(FeasibilityReason.UNRESOLVED_DEPENDENCY.value)
        if not set(plan.assumption_ids) <= assumptions:
            reasons.append(FeasibilityReason.HIDDEN_ASSUMPTION.value)
        if not plan.falsification_test_id:
            reasons.append(FeasibilityReason.MISSING_FALSIFIER.value)
        if (
            plan.plan_class == PlanClass.REDUCTION_TO_KNOWN_RESULT.value
            and not plan.theorem_card_ids
        ):
            reasons.append(FeasibilityReason.MISSING_THEOREM_SUPPORT.value)
        if not set(plan.theorem_card_ids) <= cards:
            reasons.append(FeasibilityReason.MISSING_THEOREM_SUPPORT.value)
        graph_ids = {node.lemma_id for node in plan.lemma_graph}
        if plan.parent_obligation_ref in graph_ids:
            reasons.append(FeasibilityReason.CIRCULAR_REDUCTION.value)
        if plan.target_complexity >= plan.parent_complexity:
            reasons.append(FeasibilityReason.NON_REDUCING_TARGET.value)
        route = (plan.plan_class, plan.target_ref, plan.source_move_id)
        if route in seen_routes:
            reasons.append(FeasibilityReason.DUPLICATE_ROUTE.value)
        seen_routes.add(route)
        if plan.content_hash in no_go:
            reasons.append(FeasibilityReason.NO_GO_ROUTE.value)
        feasible = not reasons
        explanation = {
            "information_gain": plan.expected_information_gain,
            "theorem_support": len(plan.theorem_card_ids),
            "strict_reduction": plan.parent_complexity - plan.target_complexity,
            "complexity": plan.target_complexity,
            "risk": plan.risk,
        }
        score = (
            -explanation["information_gain"],
            -explanation["theorem_support"],
            -explanation["strict_reduction"],
            explanation["complexity"],
            explanation["risk"],
        )
        decisions.append(FeasibilityDecision(
            plan.plan_id, feasible,
            tuple(reasons or (FeasibilityReason.FEASIBLE.value,)),
            score, explanation,
        ))
    return tuple(decisions)


def _dominates(left: FeasibilityDecision, right: FeasibilityDecision) -> bool:
    a, b = left.score_explanation, right.score_explanation
    maximize = ("information_gain", "theorem_support", "strict_reduction")
    minimize = ("complexity", "risk")
    no_worse = (
        all(a[key] >= b[key] for key in maximize)
        and all(a[key] <= b[key] for key in minimize)
    )
    better = (
        any(a[key] > b[key] for key in maximize)
        or any(a[key] < b[key] for key in minimize)
    )
    return no_worse and better


def run_tournament(
    *,
    event_id: str,
    event_type: StrategyEvent,
    plans: Iterable[StrategyPlan],
    decisions: Iterable[FeasibilityDecision],
    critic_ranked_plan_ids: Iterable[str] = (),
    critic_reason_codes: Iterable[CriticReason] = (),
) -> TournamentResult:
    plans = tuple(plans)
    decisions = tuple(decisions)
    if {plan.plan_class for plan in plans} != {item.value for item in PlanClass}:
        raise ValueError("TOURNAMENT_REQUIRES_FOUR_INDEPENDENT_PLAN_CLASSES")
    feasible = tuple(item for item in decisions if item.feasible)
    frontier = tuple(sorted(
        item.plan_id for item in feasible
        if not any(_dominates(other, item) for other in feasible if other != item)
    ))
    requested = tuple(critic_ranked_plan_ids)
    feasible_ids = {item.plan_id for item in feasible}
    # Critic ordering is advisory and can never re-admit an infeasible plan.
    ranked = tuple(item for item in requested if item in feasible_ids)
    ranked += tuple(
        item.plan_id for item in sorted(
            feasible, key=lambda decision: (
                decision.deterministic_score, decision.plan_id,
            ),
        ) if item.plan_id not in ranked
    )
    selected = next((item for item in ranked if item in frontier), "")
    canonical = {
        "version": TOURNAMENT_VERSION,
        "event_id": event_id,
        "event_type": event_type.value,
        "plans": [asdict(item) for item in plans],
        "decisions": [asdict(item) for item in decisions],
        "pareto": frontier,
        "critic_ranking": ranked,
        "critic_reasons": [item.value for item in critic_reason_codes],
        "selected": selected,
    }
    return TournamentResult(
        event_id, event_type.value, plans, decisions, frontier, ranked,
        tuple(item.value for item in critic_reason_codes), selected,
        _digest(canonical),
    )


def review_branch(
    history: BranchHistory,
    *,
    stagnation_threshold: int,
    failure_threshold: int,
) -> BranchHistory:
    """Review only persisted evidence; callers cannot inject a verdict."""
    progress = sum(
        item.accepted_child_delta + item.lean_theorem_delta
        + item.verified_counterexample_delta
        for item in history.evidence
        if item.provenance_hash
    )
    failures = sum(
        item.semantic_failure_delta for item in history.evidence
        if item.provenance_hash
    )
    history.score = progress * 10 - failures
    if failures >= failure_threshold:
        history.status = "QUARANTINED"
        history.disposition_reason_codes = ["CONFIRMED_FAILURE_THRESHOLD"]
    elif len(history.evidence) >= stagnation_threshold and progress == 0:
        history.status = "REFRAME_REQUIRED"
        history.disposition_reason_codes = ["OBJECTIVE_MATHEMATICAL_STAGNATION"]
    else:
        history.status = "ACTIVE"
        history.disposition_reason_codes = []
    return history


def branch_review_record(history: BranchHistory) -> dict[str, object]:
    return {
        "schema_version": 1,
        "branch_id": history.branch_id,
        "reviewed_at": time.time(),
        "evidence_ids": [item.evidence_id for item in history.evidence],
        "evidence_provenance_hashes": [
            item.provenance_hash for item in history.evidence
        ],
        "score": history.score,
        "status": history.status,
        "disposition_reason_codes": list(history.disposition_reason_codes),
        "assistant_verdict": None,
        "content_hash": _digest({
            "branch_id": history.branch_id,
            "evidence": [asdict(item) for item in history.evidence],
            "score": history.score,
            "status": history.status,
            "reasons": history.disposition_reason_codes,
        }),
    }
