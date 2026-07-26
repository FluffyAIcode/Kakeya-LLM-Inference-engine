"""Host-owned gate between a strategy tournament and proof search."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping

from autoresearch.prefill.strategy_tournament import (
    PlanExecutionStatus,
    StrategyPlan,
)


CONTRACT_VERSION = 2


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


class ContractRejection(str, Enum):
    MISSING_DEFINITION = "MISSING_DEFINITION"
    UNELABORATED_TARGET = "UNELABORATED_TARGET"
    PROPOSITION_HASH_MISMATCH = "PROPOSITION_HASH_MISMATCH"
    UNRESOLVED_DEPENDENCY = "UNRESOLVED_DEPENDENCY"
    UNRESOLVED_THEOREM_CARD = "UNRESOLVED_THEOREM_CARD"
    MISSING_SUCCESS_CRITERION = "MISSING_SUCCESS_CRITERION"
    MISSING_FAILURE_CRITERION = "MISSING_FAILURE_CRITERION"
    MISSING_PROOF_OBLIGATION = "MISSING_PROOF_OBLIGATION"
    HIDDEN_ASSUMPTION = "HIDDEN_ASSUMPTION"
    NON_REDUCING_TARGET = "NON_REDUCING_TARGET"
    ENVIRONMENT_HASH_MISMATCH = "ENVIRONMENT_HASH_MISMATCH"
    PLAN_HASH_MISMATCH = "PLAN_HASH_MISMATCH"
    PLANNING_ONLY = "PLANNING_ONLY"


@dataclass(frozen=True)
class ResearchContract:
    contract_version: int
    contract_id: str
    plan_id: str
    plan_hash: str
    target_ref: str
    theorem_id: str
    proposition_hash: str
    proof_obligation_id: str
    definition_ids: tuple[str, ...]
    definition_gap_ids: tuple[str, ...]
    definition_auditor_hash: str
    dependency_ids: tuple[str, ...]
    theorem_card_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    success_criterion_id: str
    failure_criterion_id: str
    parent_obligation_ref: str
    parent_complexity: int
    target_complexity: int
    environment_hash: str
    content_hash: str


@dataclass(frozen=True)
class ContractDecision:
    accepted: bool
    reason_codes: tuple[str, ...]
    route_state: str
    contract: ResearchContract | None


def gate_research_contract(
    plan: StrategyPlan,
    *,
    elaborated_theorem_id: str,
    elaborated_proposition_hash: str,
    proof_obligation_id: str,
    registered_definition_ids: Iterable[str],
    resolved_dependency_ids: Iterable[str],
    verified_theorem_card_ids: Iterable[str],
    allowed_assumption_ids: Iterable[str],
    environment_hash: str,
    expected_plan_hash: str,
) -> ContractDecision:
    reasons: list[str] = []
    definitions = set(registered_definition_ids)
    dependencies = set(resolved_dependency_ids)
    cards = set(verified_theorem_card_ids)
    assumptions = set(allowed_assumption_ids)
    if not set(plan.required_definition_ids) <= definitions:
        reasons.append(ContractRejection.MISSING_DEFINITION.value)
    if plan.execution_status != PlanExecutionStatus.EXECUTABLE.value:
        reasons.append(ContractRejection.PLANNING_ONLY.value)
    if (
        plan.plan_class == "REFRAME_DEFINITIONS_OR_REPRESENTATION"
        and not (
            plan.proposition_transformation_ref
            or plan.case_partition_ids
        )
    ):
        reasons.append(ContractRejection.PLANNING_ONLY.value)
    if not elaborated_theorem_id or not elaborated_proposition_hash:
        reasons.append(ContractRejection.UNELABORATED_TARGET.value)
    if not set(plan.dependency_ids) <= dependencies:
        reasons.append(ContractRejection.UNRESOLVED_DEPENDENCY.value)
    if not set(plan.theorem_card_ids) <= cards:
        reasons.append(ContractRejection.UNRESOLVED_THEOREM_CARD.value)
    if not plan.success_criterion_id:
        reasons.append(ContractRejection.MISSING_SUCCESS_CRITERION.value)
    if not plan.abandonment_criterion_id:
        reasons.append(ContractRejection.MISSING_FAILURE_CRITERION.value)
    if not proof_obligation_id:
        reasons.append(ContractRejection.MISSING_PROOF_OBLIGATION.value)
    if not set(plan.assumption_ids) <= assumptions:
        reasons.append(ContractRejection.HIDDEN_ASSUMPTION.value)
    if plan.target_complexity >= plan.parent_complexity:
        reasons.append(ContractRejection.NON_REDUCING_TARGET.value)
    if plan.environment_hash != environment_hash:
        reasons.append(ContractRejection.ENVIRONMENT_HASH_MISMATCH.value)
    if plan.content_hash != expected_plan_hash:
        reasons.append(ContractRejection.PLAN_HASH_MISMATCH.value)
    if reasons:
        reason_set = set(reasons)
        if reason_set & {
            ContractRejection.MISSING_DEFINITION.value,
            ContractRejection.UNELABORATED_TARGET.value,
            ContractRejection.MISSING_PROOF_OBLIGATION.value,
            ContractRejection.PLANNING_ONLY.value,
        }:
            route = "DECOMPOSER"
        elif reason_set & {
            ContractRejection.PROPOSITION_HASH_MISMATCH.value,
            ContractRejection.UNRESOLVED_DEPENDENCY.value,
        }:
            route = "MATH_IR_TRANSLATION"
        elif reason_set & {
            ContractRejection.ENVIRONMENT_HASH_MISMATCH.value,
            ContractRejection.PLAN_HASH_MISMATCH.value,
        }:
            route = "HOST_TYPED_IR_GATE"
        else:
            route = "STRATEGY_TOURNAMENT"
        return ContractDecision(False, tuple(dict.fromkeys(reasons)), route, None)
    body: Mapping[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "plan_id": plan.plan_id,
        "plan_hash": plan.content_hash,
        "target_ref": plan.target_ref,
        "theorem_id": elaborated_theorem_id,
        "proposition_hash": elaborated_proposition_hash,
        "proof_obligation_id": proof_obligation_id,
        "definition_ids": plan.required_definition_ids,
        "definition_gap_ids": plan.definition_gap_ids,
        "definition_auditor_hash": plan.definition_auditor_hash,
        "dependency_ids": plan.dependency_ids,
        "theorem_card_ids": plan.theorem_card_ids,
        "assumption_ids": plan.assumption_ids,
        "success_criterion_id": plan.success_criterion_id,
        "failure_criterion_id": plan.abandonment_criterion_id,
        "parent_obligation_ref": plan.parent_obligation_ref,
        "parent_complexity": plan.parent_complexity,
        "target_complexity": plan.target_complexity,
        "environment_hash": environment_hash,
    }
    content_hash = _digest(body)
    contract = ResearchContract(
        contract_id="RC-" + content_hash[:20],
        content_hash=content_hash,
        **body,  # type: ignore[arg-type]
    )
    return ContractDecision(True, ("ACCEPTED",), "PROOF_SEARCH", contract)


def verify_contract_binding(
    contract: ResearchContract,
    *,
    proposition_hash: str,
    environment_hash: str,
) -> None:
    if contract.proposition_hash != proposition_hash:
        raise ValueError(ContractRejection.PROPOSITION_HASH_MISMATCH.value)
    if contract.environment_hash != environment_hash:
        raise ValueError(ContractRejection.ENVIRONMENT_HASH_MISMATCH.value)
    body = asdict(contract)
    content_hash = body.pop("content_hash")
    body.pop("contract_id")
    if _digest(body) != content_hash:
        raise ValueError("RESEARCH_CONTRACT_CONTENT_HASH_MISMATCH")
