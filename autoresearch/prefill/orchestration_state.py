"""Crash-consistent, event-driven proof orchestration checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 13
ARCHITECTURE_VERSION = 9
TYPED_ARCHITECTURE_MIN_VERSION = 7
TYPED_IR_MIGRATION_EVENT = "typed_ir_host_constrained_selector_v2"
CREATIVE_DECOMPOSITION_MIGRATION_EVENT = (
    "creative_decomposition_synthesis_moves_v3"
)
STRATEGY_TOURNAMENT_MIGRATION_EVENT = (
    "cursor_strategy_oprover_advisor_v1"
)
STRATEGY_INTENT_TARGET_CONTEXT_MIGRATION_EVENT = (
    "strategy_intent_target_context_v1"
)
CANDIDATE_REPRESENTATION_MIGRATION_EVENT = (
    "candidate_representation_analysis_v1"
)
REQUIRED_TYPED_CAPABILITIES = {
    "typed_role_transport": True,
    "host_owned_public_assumptions": True,
    "explicit_restriction_moves": True,
    "legacy_model_artifact_execution": False,
    "journaled_checkpoint_changes": True,
    "strategy_tournament": True,
    "research_contract_gate": True,
    "stepwise_lean_feedback": True,
    "legacy_strategy_generator_execution": False,
    "atomic_define_one_concept": True,
    "mathematical_progress_invariant": True,
    "autonomous_definition_resolution": True,
    "legacy_definition_registry_execution": False,
    "generic_definition_reframe_execution": False,
    "cursor_strategy_adapter_only": True,
    "oprover_proof_advisor": True,
    "exclusive_primary_model_residency": True,
    "gemma_proof_generation": False,
    "direct_architecture_cutover": True,
}


class CheckpointCompatibilityError(ValueError):
    """A persisted architecture cannot be executed by this runtime."""


def current_capability_manifest() -> dict[str, Any]:
    """Return the authoritative, content-addressed runtime capability set."""
    from autoresearch.prefill.creative_decomposition import MOVE_REGISTRY_VERSION
    from autoresearch.prefill.evidence_planner import PLANNER_VERSION
    from autoresearch.prefill.host_compiler import HOST_COMPILER_VERSION
    from autoresearch.prefill.research_contract import CONTRACT_VERSION
    from autoresearch.prefill.stepwise_proof import PROOF_SEARCH_VERSION
    from autoresearch.prefill.strategy_tournament import TOURNAMENT_VERSION
    from autoresearch.prefill.math_ir import (
        MATH_IR_VERSION,
        REGISTRY_VERSION,
        registry_hash,
    )
    from autoresearch.prefill.typed_transport import (
        TRANSPORT_VERSION,
        registry_hash as transport_registry_hash,
    )
    from autoresearch.prefill.definition_resolution import PROTOCOL_VERSION
    from autoresearch.prefill.candidate_representation import (
        MAPPER_CAPABILITY_VERSION,
        mapper_capability_hash,
    )

    return {
        "typed_transport_version": TRANSPORT_VERSION,
        "typed_transport_registry_hash": transport_registry_hash(),
        "math_ir_version": MATH_IR_VERSION,
        "math_registry_version": REGISTRY_VERSION,
        "math_registry_hash": registry_hash(),
        "host_compiler_version": HOST_COMPILER_VERSION,
        "move_registry_version": MOVE_REGISTRY_VERSION,
        "evidence_planner_version": PLANNER_VERSION,
        "strategy_tournament_version": TOURNAMENT_VERSION,
        "research_contract_version": CONTRACT_VERSION,
        "proof_search_version": PROOF_SEARCH_VERSION,
        "atomic_definition_version": PROTOCOL_VERSION,
        "candidate_mapper_version": MAPPER_CAPABILITY_VERSION,
        "candidate_mapper_hash": mapper_capability_hash(),
        "capability_flags": dict(REQUIRED_TYPED_CAPABILITIES),
    }


def _manifest_default(name: str):
    return current_capability_manifest()[name]


class ProofState(str, Enum):
    STRATEGY_TOURNAMENT = "STRATEGY_TOURNAMENT"
    RESEARCH_CONTRACT_GATE = "RESEARCH_CONTRACT_GATE"
    # Legacy names are deserialization markers only. Architecture 7 has no
    # transition edges from them and dispatch rejects them.
    NEEDS_STRATEGY = "NEEDS_STRATEGY"
    GENERATOR = "GENERATOR"
    CRITIC = "CRITIC"
    DEFINITION_AUDITOR = "DEFINITION_AUDITOR"
    COUNTEREXAMPLE_WORKER = "COUNTEREXAMPLE_WORKER"
    SYNTHESIS = "SYNTHESIS"
    REFRAME = "REFRAME"
    DEFINITION_RESOLUTION = "DEFINITION_RESOLUTION"
    PARENT_STATEMENT_UNDERSPECIFIED = "PARENT_STATEMENT_UNDERSPECIFIED"
    DECOMPOSER = "DECOMPOSER"
    DECOMPOSITION_EXPLORATION = "DECOMPOSITION_EXPLORATION"
    CANDIDATE_PREFILTER = "CANDIDATE_PREFILTER"
    CANDIDATE_FORMALIZATION = "CANDIDATE_FORMALIZATION"
    CANDIDATE_REPRESENTATION_ANALYSIS = "CANDIDATE_REPRESENTATION_ANALYSIS"
    REDUCTION_CERTIFICATION = "REDUCTION_CERTIFICATION"
    MATH_IR_TRANSLATION = "MATH_IR_TRANSLATION"
    HOST_TYPED_IR_GATE = "HOST_TYPED_IR_GATE"
    LEAN_ELABORATION_GATE = "LEAN_ELABORATION_GATE"
    PROOF_SEARCH = "PROOF_SEARCH"
    # Source compatibility only; persisted values use the v2 state names.
    FORMALIZER = "MATH_IR_TRANSLATION"
    PROVER = "PROOF_SEARCH"
    ADVERSARIAL_REVIEW = "ADVERSARIAL_REVIEW"
    JUDGE = "JUDGE"
    COMMIT = "COMMIT"
    APPROACH_FAILED = "APPROACH_FAILED"
    PREMISE_AUDIT = "PREMISE_AUDIT"
    PREMISE_INVALIDATED = "PREMISE_INVALIDATED"
    REPAIRABLE_DEFINITION_GAP = "REPAIRABLE_DEFINITION_GAP"
    DECOMPOSITION_STAGNATED = "DECOMPOSITION_STAGNATED"
    MATHEMATICAL_STAGNATION = "MATHEMATICAL_STAGNATION"
    BLOCKED = "BLOCKED"
    IDLE = "IDLE"


class BlockedEventType(str, Enum):
    OPERATOR_UNBLOCK = "OPERATOR_UNBLOCK"
    HOST_GATE_DEFECTS_BACKJUMP = "HOST_GATE_DEFECTS_BACKJUMP"
    TARGET_BRANCH_CHANGE = "TARGET_BRANCH_CHANGE"
    VALIDATED_EVIDENCE_BACKJUMP = "VALIDATED_EVIDENCE_BACKJUMP"
    NEW_STRATEGY_TRIGGER = "NEW_STRATEGY_TRIGGER"


class PremiseAuditOutcomeType(str, Enum):
    """Host-owned semantic outcomes emitted from premise-audit evidence."""

    APPROACH_FAILED = "APPROACH_FAILED"
    PREMISE_SUSPECTED = "PREMISE_SUSPECTED"
    PREMISE_INVALIDATED = "PREMISE_INVALIDATED"
    PARENT_STATEMENT_UNDERSPECIFIED = "PARENT_STATEMENT_UNDERSPECIFIED"
    REPAIRABLE_DEFINITION_GAP = "REPAIRABLE_DEFINITION_GAP"


class DefinitionAuditOutcomeType(str, Enum):
    """Host-owned semantic outcomes from the Definition Auditor."""

    COMPLETE = "COMPLETE"
    MISSING_DEFINITION = "MISSING_DEFINITION"
    REFRAME_REQUIRED = "REFRAME_REQUIRED"
    PARENT_UNDERSPECIFIED = "PARENT_UNDERSPECIFIED"


@dataclass(frozen=True)
class BlockedExitEvent:
    event_id: str
    event_type: str
    reason: str
    target_state: str
    reset_role: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    schema_version: int = 1

    @property
    def typed_event(self) -> BlockedEventType:
        return BlockedEventType(self.event_type)


ROLE_ORDER = (
    ProofState.DEFINITION_AUDITOR,
    ProofState.COUNTEREXAMPLE_WORKER,
    ProofState.DECOMPOSER,
    ProofState.MATH_IR_TRANSLATION,
    ProofState.HOST_TYPED_IR_GATE,
    ProofState.LEAN_ELABORATION_GATE,
    ProofState.PROOF_SEARCH,
    ProofState.ADVERSARIAL_REVIEW,
    ProofState.JUDGE,
    ProofState.COMMIT,
)
ROLE_ARTIFACT_KEYS = {
    ProofState.STRATEGY_TOURNAMENT: "strategy_tournament",
    ProofState.RESEARCH_CONTRACT_GATE: "research_contract",
    ProofState.DEFINITION_AUDITOR: "definition_auditor",
    ProofState.COUNTEREXAMPLE_WORKER: "counterexample_worker",
    ProofState.DECOMPOSER: "decomposer",
    ProofState.MATH_IR_TRANSLATION: "math_ir_translator",
    ProofState.PROOF_SEARCH: "proof_search",
    ProofState.ADVERSARIAL_REVIEW: "adversarial_proponent",
    ProofState.JUDGE: "judge",
}
ALLOWED_TRANSITIONS = {
    ProofState.STRATEGY_TOURNAMENT: {
        ProofState.RESEARCH_CONTRACT_GATE, ProofState.DECOMPOSER,
        ProofState.DECOMPOSITION_EXPLORATION,
        ProofState.DEFINITION_RESOLUTION,
        ProofState.BLOCKED,
    },
    ProofState.RESEARCH_CONTRACT_GATE: {
        ProofState.DECOMPOSER, ProofState.MATH_IR_TRANSLATION,
        ProofState.HOST_TYPED_IR_GATE, ProofState.STRATEGY_TOURNAMENT,
        ProofState.PROOF_SEARCH, ProofState.BLOCKED,
    },
    # Audit-only legacy states deliberately have no executable outgoing edge.
    ProofState.NEEDS_STRATEGY: set(),
    ProofState.GENERATOR: set(),
    ProofState.CRITIC: set(),
    ProofState.DEFINITION_AUDITOR: {
        ProofState.DEFINITION_AUDITOR, ProofState.COUNTEREXAMPLE_WORKER,
        ProofState.DEFINITION_RESOLUTION, ProofState.SYNTHESIS,
        ProofState.DECOMPOSER,
        ProofState.BLOCKED,
    },
    ProofState.COUNTEREXAMPLE_WORKER: {
        ProofState.COUNTEREXAMPLE_WORKER, ProofState.DECOMPOSER,
        ProofState.SYNTHESIS,
        ProofState.PREMISE_AUDIT, ProofState.BLOCKED,
    },
    ProofState.SYNTHESIS: {
        ProofState.SYNTHESIS, ProofState.DEFINITION_RESOLUTION,
        ProofState.DECOMPOSER, ProofState.COUNTEREXAMPLE_WORKER,
        ProofState.BLOCKED,
    },
    ProofState.REFRAME: set(),
    ProofState.DEFINITION_RESOLUTION: {
        ProofState.STRATEGY_TOURNAMENT,
        ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
        ProofState.MATHEMATICAL_STAGNATION,
        ProofState.PREMISE_AUDIT,
        ProofState.BLOCKED,
    },
    ProofState.PARENT_STATEMENT_UNDERSPECIFIED: {
        ProofState.STRATEGY_TOURNAMENT, ProofState.PREMISE_AUDIT,
        ProofState.BLOCKED,
    },
    ProofState.DECOMPOSER: {
        ProofState.DECOMPOSER, ProofState.MATH_IR_TRANSLATION,
        ProofState.DECOMPOSITION_EXPLORATION,
        ProofState.SYNTHESIS, ProofState.DEFINITION_RESOLUTION,
        ProofState.DECOMPOSITION_STAGNATED, ProofState.MATHEMATICAL_STAGNATION,
        ProofState.BLOCKED,
    },
    ProofState.DECOMPOSITION_EXPLORATION: {
        ProofState.CANDIDATE_PREFILTER, ProofState.DECOMPOSER,
        ProofState.STRATEGY_TOURNAMENT, ProofState.BLOCKED,
    },
    ProofState.CANDIDATE_PREFILTER: {
        ProofState.CANDIDATE_FORMALIZATION, ProofState.DECOMPOSER,
        ProofState.STRATEGY_TOURNAMENT, ProofState.BLOCKED,
    },
    ProofState.CANDIDATE_FORMALIZATION: {
        ProofState.CANDIDATE_FORMALIZATION,
        ProofState.CANDIDATE_REPRESENTATION_ANALYSIS,
        ProofState.REDUCTION_CERTIFICATION, ProofState.DECOMPOSER,
        ProofState.STRATEGY_TOURNAMENT, ProofState.BLOCKED,
    },
    ProofState.CANDIDATE_REPRESENTATION_ANALYSIS: {
        ProofState.CANDIDATE_REPRESENTATION_ANALYSIS,
        ProofState.CANDIDATE_FORMALIZATION,
        ProofState.DEFINITION_RESOLUTION,
        ProofState.DECOMPOSER, ProofState.STRATEGY_TOURNAMENT,
        ProofState.BLOCKED,
    },
    ProofState.REDUCTION_CERTIFICATION: {
        ProofState.REDUCTION_CERTIFICATION,
        ProofState.CANDIDATE_FORMALIZATION, ProofState.PROOF_SEARCH,
        ProofState.ADVERSARIAL_REVIEW, ProofState.JUDGE, ProofState.COMMIT,
        ProofState.DECOMPOSER, ProofState.BLOCKED,
    },
    ProofState.MATH_IR_TRANSLATION: {
        ProofState.MATH_IR_TRANSLATION, ProofState.DECOMPOSER,
        ProofState.HOST_TYPED_IR_GATE,
        # Read-only v1 replay can traverse its recorded downstream edge.
        ProofState.PROOF_SEARCH,
        ProofState.BLOCKED,
    },
    ProofState.HOST_TYPED_IR_GATE: {
        ProofState.LEAN_ELABORATION_GATE, ProofState.DECOMPOSER,
        ProofState.MATH_IR_TRANSLATION, ProofState.SYNTHESIS,
        ProofState.STRATEGY_TOURNAMENT,
        ProofState.BLOCKED,
    },
    ProofState.LEAN_ELABORATION_GATE: {
        ProofState.PROOF_SEARCH, ProofState.DECOMPOSER,
        ProofState.MATH_IR_TRANSLATION, ProofState.STRATEGY_TOURNAMENT,
        ProofState.BLOCKED,
    },
    ProofState.PROOF_SEARCH: {
        ProofState.PROOF_SEARCH, ProofState.DECOMPOSER,
        ProofState.MATH_IR_TRANSLATION,
        ProofState.ADVERSARIAL_REVIEW, ProofState.BLOCKED,
    },
    ProofState.ADVERSARIAL_REVIEW: {
        ProofState.ADVERSARIAL_REVIEW, ProofState.DECOMPOSER,
        ProofState.MATH_IR_TRANSLATION, ProofState.HOST_TYPED_IR_GATE,
        ProofState.LEAN_ELABORATION_GATE, ProofState.PROOF_SEARCH,
        ProofState.JUDGE,
        ProofState.BLOCKED,
    },
    ProofState.JUDGE: {
        ProofState.JUDGE, ProofState.DEFINITION_AUDITOR,
        ProofState.COUNTEREXAMPLE_WORKER, ProofState.DECOMPOSER,
        ProofState.MATH_IR_TRANSLATION, ProofState.HOST_TYPED_IR_GATE,
        ProofState.LEAN_ELABORATION_GATE, ProofState.PROOF_SEARCH,
        ProofState.ADVERSARIAL_REVIEW, ProofState.COMMIT,
        ProofState.APPROACH_FAILED, ProofState.PREMISE_AUDIT,
        ProofState.BLOCKED,
    },
    ProofState.COMMIT: {ProofState.IDLE, ProofState.COMMIT},
    ProofState.APPROACH_FAILED: {
        ProofState.STRATEGY_TOURNAMENT, ProofState.BLOCKED,
    },
    ProofState.PREMISE_AUDIT: {
        ProofState.PREMISE_AUDIT,
        ProofState.APPROACH_FAILED,
        ProofState.PREMISE_INVALIDATED,
        ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
        ProofState.REPAIRABLE_DEFINITION_GAP,
        ProofState.BLOCKED,
    },
    ProofState.PREMISE_INVALIDATED: {
        ProofState.STRATEGY_TOURNAMENT, ProofState.BLOCKED,
    },
    ProofState.REPAIRABLE_DEFINITION_GAP: {
        ProofState.DEFINITION_RESOLUTION, ProofState.BLOCKED,
    },
    ProofState.DECOMPOSITION_STAGNATED: {
        ProofState.SYNTHESIS, ProofState.DEFINITION_RESOLUTION,
        ProofState.DECOMPOSER,
        ProofState.STRATEGY_TOURNAMENT, ProofState.BLOCKED,
    },
    ProofState.MATHEMATICAL_STAGNATION: {
        ProofState.DEFINITION_RESOLUTION, ProofState.STRATEGY_TOURNAMENT,
        ProofState.BLOCKED,
    },
    ProofState.BLOCKED: {
        ProofState.BLOCKED, ProofState.STRATEGY_TOURNAMENT,
        ProofState.RESEARCH_CONTRACT_GATE,
        ProofState.DECOMPOSITION_EXPLORATION,
        ProofState.CANDIDATE_PREFILTER,
        ProofState.CANDIDATE_FORMALIZATION,
        ProofState.CANDIDATE_REPRESENTATION_ANALYSIS,
        ProofState.REDUCTION_CERTIFICATION,
        *ROLE_ORDER[:-1],
    },
    ProofState.IDLE: {
        ProofState.IDLE, ProofState.STRATEGY_TOURNAMENT,
        ProofState.BLOCKED,
    },
}


@dataclass
class ArtifactRef:
    role: str
    sha256: str
    schema_version: int
    dependencies: list[str]
    path: str
    source_run_id: str
    validated_at: float
    target_context_hash: str = ""
    target_obligation_id: str = ""
    parent_statement_hash: str = ""
    strategy_plan_hash: str = ""
    environment_hash: str = ""
    candidate_sha256: str = ""
    ledger_id: str = ""
    ledger_version: int = 0
    reusable: bool = False
    validation_status: str = ""


@dataclass(frozen=True)
class HostGateDefect:
    code: str
    source_role: str
    message: str
    hard_invalid: bool = True


HOST_GATE_DEFECT_RULES = (
    (
        "UNVERIFIED_COUNTEREXAMPLE",
        "counterexample_worker",
        "claimed counterexample has no verified evidence",
        False,
    ),
    (
        "REDUCTION_CONCLUSION_MISMATCH",
        "formalizer",
        "reduction theorem conclusion differs from exact parent proposition",
        True,
    ),
    (
        "PARENT_SIGNATURE_INVALID",
        "formalizer",
        "parent signature failed:",
        True,
    ),
    (
        "CHILD_SIGNATURE_INVALID",
        "formalizer",
        "child L1 signature failed:",
        True,
    ),
    (
        "CYCLIC_OR_EQUIVALENT_CHILD",
        "decomposer",
        "child L1 rejected:",
        True,
    ),
    (
        "REDUCTION_SIGNATURE_INVALID",
        "formalizer",
        "reduction theorem signature failed or changed",
        True,
    ),
    (
        "REDUCTION_PROOF_INVALID",
        "prover",
        "complete reduction proof failed or targets another theorem",
        True,
    ),
)


def classify_host_gate_defects(errors: list[str]) -> list[HostGateDefect]:
    """Classify exact host errors without collapsing their producer roles."""
    defects = []
    for error in errors:
        normalized = " ".join(str(error).split())
        matches = [
            HostGateDefect(code, role, normalized, hard)
            for code, role, marker, hard in HOST_GATE_DEFECT_RULES
            if marker in normalized
        ]
        if len(matches) != 1:
            raise ValueError(f"unclassified or ambiguous host-gate defect: {error}")
        defects.append(matches[0])
    return defects


def earliest_invalid_role(defects: list[HostGateDefect]) -> ProofState:
    hard_roles = {
        state_for_role(defect.source_role)
        for defect in defects
        if defect.hard_invalid
    }
    for state in ROLE_ORDER:
        if state in hard_roles:
            return state
    raise ValueError("host-gate recovery requires at least one hard defect")


@dataclass
class OrchestrationCheckpoint:
    state: str = ProofState.STRATEGY_TOURNAMENT.value
    target_obligation_id: str = ""
    candidate_sha256: str = ""
    strategy_sha256: str = ""
    parent_statement_sha256: str = ""
    parent_signature_sha256: str = ""
    root_goal_sha256: str = ""
    current_role: str = "strategy_tournament"
    validated_artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    retry_counters: dict[str, int] = field(default_factory=dict)
    decomposition_iteration: int = 0
    synthesis_iteration: int = 0
    viewpoint: str = ""
    viewpoints_tried: list[str] = field(default_factory=list)
    semantic_rejection: dict[str, Any] = field(default_factory=dict)
    decomposition_proposals: list[dict[str, Any]] = field(default_factory=list)
    novel_proposals: int = 0
    mathematical_retries: int = 0
    formalizer_substate: str = ""
    lean_contract_id: str = ""
    lean_contract_version: int = 0
    lean_symbol_table_id: str = ""
    lean_symbol_table_version: int = 0
    formalizer_unit_hashes: dict[str, str] = field(default_factory=dict)
    last_transition_reason: str = "new-checkpoint"
    resume_origin: str = ""
    strategy_reused: bool = False
    source_run_ids: list[str] = field(default_factory=list)
    ledger_id: str = ""
    ledger_version: int = 0
    orchestration_id: str = ""
    commit_key: str = ""
    committed: bool = False
    blocked_reason: str = ""
    last_failure_fingerprint: str = ""
    identical_failure_count: int = 0
    last_blocked_event_id: str = ""
    invalidated_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    advisory_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    reuse_map: dict[str, str] = field(default_factory=dict)
    recovery_events: list[dict[str, Any]] = field(default_factory=list)
    architecture_version: int = ARCHITECTURE_VERSION
    typed_transport_version: int = field(
        default_factory=lambda: _manifest_default("typed_transport_version"),
    )
    typed_transport_registry_hash: str = field(
        default_factory=lambda: _manifest_default("typed_transport_registry_hash"),
    )
    math_ir_version: int = field(
        default_factory=lambda: _manifest_default("math_ir_version"),
    )
    math_registry_version: int = field(
        default_factory=lambda: _manifest_default("math_registry_version"),
    )
    math_registry_hash: str = field(
        default_factory=lambda: _manifest_default("math_registry_hash"),
    )
    candidate_mapper_version: int = field(
        default_factory=lambda: _manifest_default("candidate_mapper_version"),
    )
    candidate_mapper_hash: str = field(
        default_factory=lambda: _manifest_default("candidate_mapper_hash"),
    )
    host_compiler_version: int = field(
        default_factory=lambda: _manifest_default("host_compiler_version"),
    )
    move_registry_version: int = field(
        default_factory=lambda: _manifest_default("move_registry_version"),
    )
    evidence_planner_version: int = field(
        default_factory=lambda: _manifest_default("evidence_planner_version"),
    )
    strategy_tournament_version: int = field(
        default_factory=lambda: _manifest_default("strategy_tournament_version"),
    )
    research_contract_version: int = field(
        default_factory=lambda: _manifest_default("research_contract_version"),
    )
    proof_search_version: int = field(
        default_factory=lambda: _manifest_default("proof_search_version"),
    )
    atomic_definition_version: int = field(
        default_factory=lambda: _manifest_default("atomic_definition_version"),
    )
    capability_flags: dict[str, bool] = field(
        default_factory=lambda: dict(REQUIRED_TYPED_CAPABILITIES),
    )
    adapter_status: str = ""
    typed_ir_hash: str = ""
    lean_declaration_hash: str = ""
    proposition_hash: str = ""
    elaborated_theorem_id: str = ""
    active_gate: str = ""
    migration_event: str = ""
    migration_snapshot: str = ""
    scratchpad_refs: list[dict[str, Any]] = field(default_factory=list)
    candidate_set_hash: str = ""
    candidate_hashes: list[str] = field(default_factory=list)
    candidate_count: int = 0
    ranking_hash: str = ""
    ranked_candidate_ids: list[str] = field(default_factory=list)
    exploration_contract_id: str = ""
    exploration_contract_hash: str = ""
    exploration_candidate_refs: list[dict[str, Any]] = field(default_factory=list)
    exploration_rejections: dict[str, list[str]] = field(default_factory=dict)
    exploration_selected_candidate_ids: list[str] = field(default_factory=list)
    exploration_current_index: int = 0
    exploration_current_candidate_id: str = ""
    exploration_formalization_status: str = ""
    exploration_reduction_status: str = ""
    exploration_exhaustion_hash: str = ""
    representation_report_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    representation_current_status: str = ""
    representation_missing_primitive_ids: list[str] = field(default_factory=list)
    representation_source_resolution: str = ""
    representation_retry_state: str = ""
    representation_retry_fingerprints: list[str] = field(default_factory=list)
    representation_exhaustion_hash: str = ""
    selected_move_id: str = ""
    evidence_gap_graph_hash: str = ""
    proof_plan_hash: str = ""
    proof_plan_id: str = ""
    executable_plan_node_id: str = ""
    plan_score_explanation: dict[str, int] = field(default_factory=dict)
    theorem_card_ids: list[str] = field(default_factory=list)
    theorem_card_index_hash: str = ""
    stagnation_reason: str = ""
    semantic_stagnation_count: int = 0
    protocol_error_count: int = 0
    public_assumptions_hash: str = ""
    child_public_assumptions_hash: str = ""
    reduction_public_assumptions_hash: str = ""
    strategy_event_id: str = ""
    strategy_event_type: str = ""
    strategy_plan_ids: list[str] = field(default_factory=list)
    strategy_plan_hashes: list[str] = field(default_factory=list)
    feasible_strategy_plan_ids: list[str] = field(default_factory=list)
    pareto_plan_ids: list[str] = field(default_factory=list)
    selected_strategy_plan_id: str = ""
    selected_strategy_plan_hash: str = ""
    strategy_tournament_hash: str = ""
    research_contract_id: str = ""
    research_contract_hash: str = ""
    research_contract_rejection_codes: list[str] = field(default_factory=list)
    branches_killed: int = 0
    branch_history: dict[str, dict[str, Any]] = field(default_factory=dict)
    lean_actions_attempted: int = 0
    lean_actions_accepted: int = 0
    subgoals_closed: int = 0
    subgoals_remaining: int = 0
    new_elaborated_definitions: int = 0
    new_elaborated_lemmas: int = 0
    accepted_children: int = 0
    verified_counterexamples: int = 0
    definitions_added: int = 0
    existing_definitions_resolved: int = 0
    lemmas_proved: int = 0
    progress_vector: dict[str, int] = field(default_factory=lambda: {
        "definitions_added": 0,
        "existing_definitions_resolved": 0,
        "lemmas_proved": 0,
        "accepted_children": 0,
        "subgoals_closed": 0,
        "verified_counterexamples": 0,
    })
    progress_fingerprint: str = ""
    forbidden_semantic_fingerprints: list[str] = field(default_factory=list)
    current_definition_gap_id: str = ""
    definition_candidate_count: int = 0
    lean_definition_status: str = ""
    definition_store_hash: str = ""
    definition_environment_hash: str = ""
    definition_store_hash_delta: str = ""
    definition_environment_hash_delta: str = ""
    definition_query_hash: str = ""
    definition_source_statuses: dict[str, str] = field(default_factory=dict)
    definition_property_statuses: dict[str, dict[str, str]] = field(default_factory=dict)
    definition_branch_hashes: list[str] = field(default_factory=list)
    definition_exhaustion_hash: str = ""
    definition_interface_hash: str = ""
    definition_backjump_target: str = ""
    definition_audit_outcome: str = ""
    definition_audit_fingerprint: str = ""
    counterexample_objective: dict[str, Any] = field(default_factory=dict)
    premise_outcome_type: str = ""
    premise_outcome_fingerprint: str = ""
    premise_outcome_owner: str = ""
    premise_decision: str = ""
    premise_confidence: float = 0.0
    premise_evidence: dict[str, Any] = field(default_factory=dict)
    premise_backjump_target: str = ""
    premise_invalidated_artifacts: list[str] = field(default_factory=list)
    consumed_premise_fingerprints: list[str] = field(default_factory=list)
    proof_tokens_consumed: int = 0
    proof_search_state_path: str = ""
    strategy_provider: str = "cursor-sdk"
    strategy_provider_configured: bool = False
    strategy_model_id: str = ""
    strategy_run_status: str = "CONFIGURATION_REQUIRED"
    strategy_agent_id: str = ""
    strategy_run_id: str = ""
    strategy_prompt_hash: str = ""
    strategy_evidence_hash: str = ""
    strategy_memo_hash: str = ""
    strategy_intent_hash: str = ""
    strategy_intent_run_id: str = ""
    strategy_intent_status: str = ""
    strategy_selection_provenance: dict[str, Any] = field(default_factory=dict)
    strategy_latency_ms: int = 0
    target_context_hash: str = ""
    target_environment_hash: str = ""
    target_strategy_plan_hash: str = ""
    target_statement: str = ""
    target_evidence: dict[str, Any] = field(default_factory=dict)
    target_gap_ids: list[str] = field(default_factory=list)
    scratchpad_math_fingerprints: list[str] = field(default_factory=list)
    residency_phase: str = "GEMMA_SERVING"
    active_model: str = "gemma"
    oprover_candidate_count: int = 0
    oprover_verified_count: int = 0
    critic_advisory_state: str = "PENDING"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    @property
    def proof_state(self) -> ProofState:
        return ProofState(self.state)

    def transition(
        self,
        next_state: ProofState,
        reason: str,
        *,
        source_run_id: str = "",
        resume_origin: str = "",
        strategy_reused: bool | None = None,
        blocked_exit_event: BlockedExitEvent | None = None,
    ) -> None:
        current = self.proof_state
        if current in {
            ProofState.NEEDS_STRATEGY,
            ProofState.GENERATOR,
            ProofState.CRITIC,
        }:
            raise ValueError(
                f"legacy architecture state is audit-only: {current.value}",
            )
        if (
            current == ProofState.BLOCKED
            and next_state != ProofState.BLOCKED
            and blocked_exit_event is None
        ):
            raise ValueError(
                "BLOCKED can be exited only by an explicit typed event",
            )
        if current == ProofState.BLOCKED and next_state == ProofState.BLOCKED:
            return
        if next_state not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(
                f"illegal proof transition {current.value}->{next_state.value}",
            )
        self.state = next_state.value
        self.current_role = next_state.value.lower()
        self.last_transition_reason = str(reason)
        self.resume_origin = str(resume_origin)
        if blocked_exit_event is not None:
            self.last_blocked_event_id = blocked_exit_event.event_id
            self.blocked_reason = ""
        if strategy_reused is not None:
            self.strategy_reused = (
                verified_reuse_provenance(self)["strategy_reused"]
                if strategy_reused else False
            )
        if source_run_id and source_run_id not in self.source_run_ids:
            self.source_run_ids.append(source_run_id)
        self.updated_at = time.time()

    def retry(self, role: ProofState, reason: str, limit: int) -> bool:
        if self.proof_state == ProofState.BLOCKED:
            return False
        artifact = self.validated_artifacts.get(
            ROLE_ARTIFACT_KEYS.get(role, ""),
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "state": role.value,
                    "artifact": artifact.sha256 if artifact else "",
                    "error": " ".join(str(reason).split()),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        ).hexdigest()
        if fingerprint == self.last_failure_fingerprint:
            self.identical_failure_count += 1
        else:
            self.last_failure_fingerprint = fingerprint
            self.identical_failure_count = 1
        if self.identical_failure_count >= 2:
            self.blocked_reason = (
                f"{role.value} identical state/artifact/error cycle: {reason}"
            )
            self.transition(ProofState.BLOCKED, self.blocked_reason)
            return False
        count = self.retry_counters.get(role.value, 0) + 1
        self.retry_counters[role.value] = count
        if count > limit:
            self.blocked_reason = (
                f"{role.value} retry budget exhausted ({limit}): {reason}"
            )
            self.transition(ProofState.BLOCKED, self.blocked_reason)
            return False
        self.transition(role, reason, resume_origin=role.value)
        return True

    def adapter_blocked(self, reason: str, *, status: str) -> None:
        """Record an out-of-machine adapter/infra stop without proof retries."""
        if status not in {
            "ADAPTER_BLOCKED", "INFRASTRUCTURE_BLOCKED", "INTEGRATION_BLOCKED",
        }:
            raise ValueError("invalid adapter status")
        self.adapter_status = status
        self.blocked_reason = str(reason)
        self.last_transition_reason = f"{status}:{reason}"
        self.updated_at = time.time()

    def clear_adapter_blocked(self, reason: str = "adapter-recovered") -> None:
        """Resume a quiescent adapter stop without consuming math retries."""
        self.adapter_status = ""
        self.blocked_reason = ""
        self.last_transition_reason = str(reason)
        self.last_failure_fingerprint = ""
        self.identical_failure_count = 0
        self.updated_at = time.time()
    def begin_decomposition_iteration(
        self,
        viewpoint: str,
        reason: str,
    ) -> None:
        """Advance mathematical search without consuming protocol retries."""
        viewpoint = str(viewpoint).strip()
        if not viewpoint:
            raise ValueError("decomposition viewpoint must be non-empty")
        self.decomposition_iteration += 1
        self.viewpoint = viewpoint
        if viewpoint not in self.viewpoints_tried:
            self.viewpoints_tried.append(viewpoint)
        self.transition(
            ProofState.DECOMPOSER,
            reason,
            resume_origin=ProofState.DECOMPOSER.value,
            strategy_reused=True,
        )


def route_definition_audit_outcome(
    checkpoint: OrchestrationCheckpoint,
    *,
    outcome: DefinitionAuditOutcomeType | str,
    artifact_hash: str,
    source_run_id: str,
    missing_definition_ids: Iterable[str] = (),
    counterexample_objective: Mapping[str, Any] | None = None,
) -> bool:
    """Persist one audit outcome and route only through typed legal edges."""
    typed_outcome = DefinitionAuditOutcomeType(outcome)
    missing = tuple(sorted({str(item) for item in missing_definition_ids if item}))
    objective = dict(counterexample_objective or {})
    if objective and not {
        "objective_type", "evidence_request",
    }.issubset(objective):
        raise ValueError(
            "counterexample objective requires objective_type and evidence_request",
        )
    fingerprint = hashlib.sha256(json.dumps({
        "outcome": typed_outcome.value,
        "artifact_hash": str(artifact_hash),
        "source_run_id": str(source_run_id),
        "missing_definition_ids": missing,
        "counterexample_objective": objective,
        "target_obligation_id": checkpoint.target_obligation_id,
        "candidate_sha256": checkpoint.candidate_sha256,
        "ledger_id": checkpoint.ledger_id,
        "ledger_version": checkpoint.ledger_version,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if checkpoint.definition_audit_fingerprint == fingerprint:
        return False

    checkpoint.definition_audit_outcome = typed_outcome.value
    checkpoint.definition_audit_fingerprint = fingerprint
    checkpoint.counterexample_objective = objective
    checkpoint.premise_outcome_owner = "definition_auditor"
    checkpoint.premise_decision = typed_outcome.value
    checkpoint.premise_confidence = 1.0
    checkpoint.premise_evidence = {
        "definition_auditor_artifact_hash": str(artifact_hash),
        "source_run_id": str(source_run_id),
        "missing_definition_ids": list(missing),
    }

    if typed_outcome in {
        DefinitionAuditOutcomeType.REFRAME_REQUIRED,
        DefinitionAuditOutcomeType.PARENT_UNDERSPECIFIED,
    }:
        checkpoint.premise_outcome_type = (
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED.value
            if typed_outcome == DefinitionAuditOutcomeType.PARENT_UNDERSPECIFIED
            else typed_outcome.value
        )
        target = ProofState.SYNTHESIS
    elif objective:
        target = ProofState.COUNTEREXAMPLE_WORKER
    elif missing or typed_outcome == DefinitionAuditOutcomeType.MISSING_DEFINITION:
        target = ProofState.DEFINITION_RESOLUTION
    else:
        target = ProofState.DECOMPOSER

    # Architecture-8 strategy state reaches semantic advisory ownership via
    # Decomposer; Counterexample is reachable only through Synthesis.
    if checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT and target in {
        ProofState.SYNTHESIS, ProofState.COUNTEREXAMPLE_WORKER,
    }:
        checkpoint.transition(
            ProofState.DECOMPOSER,
            "definition-audit-route:semantic-owner",
            source_run_id=source_run_id,
            strategy_reused=True,
        )
    if checkpoint.proof_state == ProofState.DECOMPOSER and target == (
        ProofState.COUNTEREXAMPLE_WORKER
    ):
        checkpoint.transition(
            ProofState.SYNTHESIS,
            "definition-audit-route:explicit-counterexample-objective",
            source_run_id=source_run_id,
            strategy_reused=True,
        )
    if checkpoint.proof_state != target:
        checkpoint.transition(
            target,
            f"definition-audit-outcome:{typed_outcome.value}",
            source_run_id=source_run_id,
            strategy_reused=True,
        )
    checkpoint.recovery_events.append({
        "event_type": "TYPED_DEFINITION_AUDIT_OUTCOME",
        "event_id": fingerprint,
        "outcome": typed_outcome.value,
        "artifact_hash": str(artifact_hash),
        "source_run_id": str(source_run_id),
        "target_state": target.value,
        "missing_definition_ids": list(missing),
        "counterexample_objective": objective,
        "created_at": time.time(),
    })
    checkpoint.clear_adapter_blocked("typed-definition-audit-routed")
    return True


def checkpoint_compatibility_errors(
    checkpoint: OrchestrationCheckpoint,
) -> list[str]:
    """Compare persisted executable capabilities with the runtime exactly."""
    if checkpoint.architecture_version < TYPED_ARCHITECTURE_MIN_VERSION:
        return ["LEGACY_ARCHITECTURE_EXECUTION_DISABLED"]
    expected = current_capability_manifest()
    errors = []
    if checkpoint.architecture_version != ARCHITECTURE_VERSION:
        errors.append(
            "ARCHITECTURE_VERSION_MISMATCH:"
            f"{checkpoint.architecture_version}!={ARCHITECTURE_VERSION}",
        )
    if checkpoint.schema_version != SCHEMA_VERSION:
        errors.append(
            f"SCHEMA_VERSION_MISMATCH:{checkpoint.schema_version}!={SCHEMA_VERSION}",
        )
    for name, value in expected.items():
        actual = getattr(checkpoint, name)
        if actual != value:
            errors.append(f"{name.upper()}_MISMATCH")
    return errors


def require_typed_dispatch(checkpoint: OrchestrationCheckpoint) -> None:
    """Fail closed unless this checkpoint has the complete typed capability."""
    errors = checkpoint_compatibility_errors(checkpoint)
    if checkpoint.proof_state in {
        ProofState.NEEDS_STRATEGY,
        ProofState.GENERATOR,
        ProofState.CRITIC,
    }:
        errors.append("LEGACY_STRATEGY_GENERATOR_EXECUTION_DISABLED")
    if errors:
        raise CheckpointCompatibilityError(";".join(errors))


def integration_block_incompatible_checkpoint(
    checkpoint: OrchestrationCheckpoint,
) -> bool:
    """Persistable fail-closed state used at every runtime dispatch boundary."""
    errors = checkpoint_compatibility_errors(checkpoint)
    if not errors:
        return False
    checkpoint.adapter_blocked(
        "checkpoint capability incompatibility: " + ";".join(errors),
        status="INTEGRATION_BLOCKED",
    )
    return True


def apply_blocked_exit_event(
    checkpoint: OrchestrationCheckpoint,
    event: BlockedExitEvent,
) -> None:
    if checkpoint.proof_state != ProofState.BLOCKED:
        raise ValueError("blocked exit event requires BLOCKED checkpoint")
    if not event.event_id or not event.reason.strip():
        raise ValueError("blocked exit event requires ID and reason")
    event_type = event.typed_event
    target = ProofState(event.target_state)
    reset_role = ProofState(event.reset_role)
    if reset_role != target:
        raise ValueError("blocked event may reset only its target role")
    if target not in ALLOWED_TRANSITIONS[ProofState.BLOCKED]:
        raise ValueError("blocked event target is not permitted")
    if (
        event_type == BlockedEventType.NEW_STRATEGY_TRIGGER
        and target != ProofState.STRATEGY_TOURNAMENT
    ):
        raise ValueError(
            "new strategy trigger must target STRATEGY_TOURNAMENT",
        )
    required_metadata = {
        BlockedEventType.HOST_GATE_DEFECTS_BACKJUMP: (
            "defects",
            "invalidated_artifact_hashes",
            "reuse_map",
        ),
        BlockedEventType.TARGET_BRANCH_CHANGE: ("target_or_branch",),
        BlockedEventType.VALIDATED_EVIDENCE_BACKJUMP: ("evidence_sha256",),
        BlockedEventType.NEW_STRATEGY_TRIGGER: ("strategy_trigger",),
    }.get(event_type, ())
    if any(not event.metadata.get(key) for key in required_metadata):
        raise ValueError(f"{event_type.value} missing required metadata")
    if event_type == BlockedEventType.HOST_GATE_DEFECTS_BACKJUMP:
        defects = classify_host_gate_defects([
            str(item["message"]) for item in event.metadata["defects"]
        ])
        if target != earliest_invalid_role(defects):
            raise ValueError("host-gate backjump target is not earliest invalid role")
        invalidated = event.metadata["invalidated_artifact_hashes"]
        for role, digest in invalidated.items():
            ref = checkpoint.validated_artifacts.get(role)
            if ref is None or ref.sha256 != digest:
                raise ValueError(f"invalidated artifact hash mismatch for {role}")
        for role in invalidated:
            ref = checkpoint.validated_artifacts.pop(role)
            checkpoint.invalidated_artifacts[ref.sha256] = {
                **asdict(ref),
                "reason_codes": [
                    defect.code for defect in defects
                    if defect.hard_invalid
                ],
                "event_id": event.event_id,
                "audit_only": True,
            }
        counterexample = checkpoint.validated_artifacts.get(
            "counterexample_worker",
        )
        if counterexample is not None:
            checkpoint.advisory_artifacts[counterexample.sha256] = {
                **asdict(counterexample),
                "verified": False,
                "premise": False,
                "public": False,
                "certificate_gate": False,
                "event_id": event.event_id,
            }
        checkpoint.reuse_map = {
            str(role): str(digest)
            for role, digest in event.metadata["reuse_map"].items()
        }
        checkpoint.recovery_events.append({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "defect_codes": [defect.code for defect in defects],
            "target_state": target.value,
            "invalidated_artifact_hashes": dict(invalidated),
            "reuse_map": dict(checkpoint.reuse_map),
            "created_at": event.created_at,
        })
    checkpoint.retry_counters[target.value] = 0
    checkpoint.last_failure_fingerprint = ""
    checkpoint.identical_failure_count = 0
    checkpoint.transition(
        target,
        f"{event_type.value}:{event.reason}",
        resume_origin=ProofState.BLOCKED.value,
        strategy_reused=target != ProofState.STRATEGY_TOURNAMENT,
        blocked_exit_event=event,
    )


def append_blocked_event_journal(
    path: Path,
    event: BlockedExitEvent,
    *,
    before_state: str,
    after_state: str,
) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = {
        "kind": "blocked_exit_event",
        "before_state": before_state,
        "after_state": after_state,
        **asdict(event),
    }
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_checkpoint_change_journal(
    path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    checkpoint_sha256: str,
) -> None:
    """Durably record every attempted checkpoint replacement before commit."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    blocked = bool(
        checkpoint.proof_state == ProofState.BLOCKED
        or checkpoint.adapter_status
    )
    record = {
        "kind": "blocked_transition" if blocked else "checkpoint_change",
        "checkpoint_sha256": checkpoint_sha256,
        "state": checkpoint.state,
        "current_role": checkpoint.current_role,
        "adapter_status": checkpoint.adapter_status,
        "reason": (
            checkpoint.blocked_reason
            if blocked else checkpoint.last_transition_reason
        ),
        "architecture_version": checkpoint.architecture_version,
        "schema_version": checkpoint.schema_version,
        "typed_transport_version": checkpoint.typed_transport_version,
        "host_compiler_version": checkpoint.host_compiler_version,
        "premise_outcome_type": checkpoint.premise_outcome_type,
        "premise_outcome_fingerprint": checkpoint.premise_outcome_fingerprint,
        "premise_outcome_owner": checkpoint.premise_outcome_owner,
        "premise_decision": checkpoint.premise_decision,
        "premise_confidence": checkpoint.premise_confidence,
        "premise_evidence": dict(checkpoint.premise_evidence),
        "premise_backjump_target": checkpoint.premise_backjump_target,
        "premise_invalidated_artifacts": list(
            checkpoint.premise_invalidated_artifacts
        ),
        "ledger_version": checkpoint.ledger_version,
        "updated_at": checkpoint.updated_at,
    }
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _serialize(checkpoint: OrchestrationCheckpoint) -> dict:
    return asdict(checkpoint)


def save_checkpoint(path: Path, checkpoint: OrchestrationCheckpoint) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    checkpoint.strategy_reused = verified_reuse_provenance(
        checkpoint,
    )["strategy_reused"]
    checkpoint.updated_at = time.time()
    if checkpoint.decomposition_proposals:
        persist_decomposition_novelty_manifest(path, checkpoint)
    serialized = json.dumps(
        _serialize(checkpoint), ensure_ascii=False, indent=2,
    )
    checkpoint_hash = hashlib.sha256(serialized.encode()).hexdigest()
    append_checkpoint_change_journal(
        path.with_name("proof_orchestration.journal.jsonl"),
        checkpoint,
        checkpoint_sha256=checkpoint_hash,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def decomposition_rejection_reason_code(reason: str) -> str:
    """Encode repeated rejection prose without losing its semantic class."""
    text = " ".join(str(reason).lower().split())
    for marker, code in (
        ("semantic-signature-duplicate", "SEMANTIC_DUPLICATE"),
        ("structural-signature-duplicate", "STRUCTURAL_DUPLICATE"),
        ("bidirectionally entails ancestor", "CYCLIC_EQUIVALENT"),
        ("not strictly simpler", "NOT_STRICTLY_SIMPLER"),
        ("structural delta", "NO_STRUCTURAL_DELTA"),
        ("disconnected child", "DISCONNECTED_CHILD"),
        ("reduction", "FAILED_REDUCTION"),
        ("definition", "DEFINITION_GAP"),
    ):
        if marker in text:
            return code
    return "OTHER"


def decomposition_novelty_manifest(
    checkpoint: OrchestrationCheckpoint,
) -> dict[str, Any]:
    """Build the complete deterministic audit index referenced by the prompt."""
    return {
        "schema_version": 1,
        "records": [{
            "proposal_sha256": str(item.get("proposal_sha256", "")),
            "semantic_hash": str(item.get("semantic_hash", "")),
            "structural_signature": str(
                item.get("structural_signature", ""),
            ),
            "rejection_reasons": [
                str(reason) for reason in item.get("rejection_reasons", [])
            ],
            "rejection_reason_codes": [
                decomposition_rejection_reason_code(reason)
                for reason in item.get("rejection_reasons", [])
            ],
            "viewpoint": str(item.get("viewpoint", "")),
            "decomposition_iteration": int(
                item.get("decomposition_iteration", 0),
            ),
            "source_run_id": str(item.get("source_run_id", "")),
        } for item in checkpoint.decomposition_proposals],
        "viewpoints_tried": list(checkpoint.viewpoints_tried),
    }


def persist_decomposition_novelty_manifest(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
) -> tuple[str, Path]:
    """Persist a content-addressed exact history index for audit/retrieval."""
    payload = decomposition_novelty_manifest(checkpoint)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    archive_dir = (
        Path(checkpoint_path).expanduser().with_suffix(".semantic-proposals")
        / "manifests"
    )
    archive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = archive_dir / f"{digest}.json"
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return digest, path


def compact_decomposition_novelty_ledger(
    checkpoint: OrchestrationCheckpoint,
    *,
    max_entries: int = 2,
) -> dict[str, Any]:
    """Return a fixed-size model view backed by an exact host-side manifest."""
    proposals = checkpoint.decomposition_proposals
    retained = proposals[-max_entries:]
    manifest = decomposition_novelty_manifest(checkpoint)
    manifest_sha256 = hashlib.sha256(json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    reason_counts: dict[str, int] = {}
    for item in proposals:
        for reason in item.get("rejection_reasons", []):
            code = decomposition_rejection_reason_code(reason)
            reason_counts[code] = reason_counts.get(code, 0) + 1
    return {
        "manifest": f"sha256:{manifest_sha256}",
        "count": len(proposals),
        "novel": checkpoint.novel_proposals,
        "reasons": dict(sorted(reason_counts.items())),
        "viewpoints": {
            "manifest": "sha256:" + hashlib.sha256(json.dumps(
                checkpoint.viewpoints_tried,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()).hexdigest(),
            "count": len(checkpoint.viewpoints_tried),
            "base_ids": [
                viewpoint for viewpoint in checkpoint.viewpoints_tried
                if not viewpoint.startswith("synthesized_host_failures_")
            ][-7:],
            "recent_ids": checkpoint.viewpoints_tried[-2:],
        },
        "active_viewpoint": checkpoint.viewpoint,
        "duplicate_gate": "semantic_hash+structural_signature",
        "recent": [{
            "i": item.get("decomposition_iteration", 0),
            "v": item.get("viewpoint", ""),
            "p": str(item.get("proposal_sha256", ""))[:16],
            "s": str(item.get("semantic_hash", ""))[:16],
            "d": str(item.get("structural_signature", ""))[:16],
            "r": [
                decomposition_rejection_reason_code(reason)
                for reason in item.get("rejection_reasons", [])
            ],
        } for item in retained],
    }


def archive_decomposition_rejection(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    proposal: dict[str, Any],
    rejection_reasons: list[str],
    semantic_hash: str,
    structural_signature: str,
    source_run_id: str,
) -> dict[str, Any]:
    """Persist one rejected mathematical proposal as immutable audit evidence."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    encoded = json.dumps(
        proposal,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    proposal_hash = hashlib.sha256(encoded).hexdigest()
    archive_dir = checkpoint_path.with_suffix(".semantic-proposals")
    archive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive_path = archive_dir / f"{proposal_hash}.json"
    if not archive_path.exists():
        temporary = archive_path.with_name(
            f".{archive_path.name}.{os.getpid()}.tmp",
        )
        try:
            temporary.write_bytes(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    record = {
        "proposal_sha256": proposal_hash,
        "semantic_hash": str(semantic_hash),
        "structural_signature": str(structural_signature),
        "rejection_reasons": [str(item) for item in rejection_reasons],
        "viewpoint": checkpoint.viewpoint,
        "decomposition_iteration": checkpoint.decomposition_iteration,
        "source_run_id": str(source_run_id),
        "path": str(archive_path),
        "audit_only": True,
        "created_at": time.time(),
    }
    is_novel = not any(
        item.get("semantic_hash") == semantic_hash
        for item in checkpoint.decomposition_proposals
    )
    checkpoint.decomposition_proposals.append(record)
    if is_novel:
        checkpoint.novel_proposals += 1
    checkpoint.semantic_rejection = dict(record)
    checkpoint.updated_at = time.time()
    save_checkpoint(checkpoint_path, checkpoint)
    return record


def load_checkpoint(path: Path) -> OrchestrationCheckpoint | None:
    path = Path(path).expanduser()
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("orchestration checkpoint must be an object")
    legacy_schema = int(raw.get("schema_version", 1))
    legacy_architecture = int(raw.get("architecture_version", 1))
    legacy_state = str(raw.get("state", ProofState.NEEDS_STRATEGY.value))
    capability_fields = set(current_capability_manifest())
    missing_capability_fields = (
        capability_fields.difference(raw)
        if (
            legacy_schema >= SCHEMA_VERSION
            and int(raw.get("architecture_version", 1)) >= ARCHITECTURE_VERSION
        )
        else set()
    )
    if legacy_schema < SCHEMA_VERSION:
        artifacts = raw.setdefault("validated_artifacts", {})
        invalidated = raw.setdefault("invalidated_artifacts", {})
        for role in (
            "strategy",
            "generator",
            "critic",
            "synthesis",
            "decomposer",
            "formalizer_parent_signature",
            "formalizer_child_signature",
            "formalizer_reduction_signature",
            "formalizer",
            "prover",
            "adversarial_proponent",
            "judge",
        ):
            reference = artifacts.pop(role, None)
            if reference:
                invalidated[str(reference["sha256"])] = {
                    **reference,
                    "audit_only": True,
                    "reason_codes": ["LEGACY_MODEL_JSON_OR_LEAN"],
                    "migration_event": STRATEGY_TOURNAMENT_MIGRATION_EVENT,
                }
        if legacy_architecture < ARCHITECTURE_VERSION:
            target = (
                ProofState.DEFINITION_RESOLUTION
                if legacy_state == ProofState.REFRAME.value
                else ProofState.STRATEGY_TOURNAMENT
            )
            raw["state"] = target.value
            raw["current_role"] = target.value.lower()
            raw["resume_origin"] = legacy_state
            raw["last_transition_reason"] = (
                "strategy-tournament-v1:legacy-strategy-generator-audit-only"
            )
            raw["blocked_reason"] = ""
        raw["migration_event"] = (
            CANDIDATE_REPRESENTATION_MIGRATION_EVENT
            if legacy_schema >= 12 else STRATEGY_TOURNAMENT_MIGRATION_EVENT
        )
        raw["architecture_version"] = ARCHITECTURE_VERSION
        raw.update(current_capability_manifest())
        raw["adapter_status"] = ""
        raw["formalizer_substate"] = ""
        raw["lean_contract_id"] = ""
        raw["lean_contract_version"] = 0
        raw["formalizer_unit_hashes"] = {}
        raw["retry_counters"] = {
            key: value for key, value in raw.get("retry_counters", {}).items()
            if key not in {"FORMALIZER", "PROVER"}
        }
        removed_counter = "protocol_" + "attempt"
        if isinstance(raw.get("semantic_rejection"), dict):
            raw["semantic_rejection"].pop(removed_counter, None)
        for proposal in raw.get("decomposition_proposals", []):
            if isinstance(proposal, dict):
                proposal.pop(removed_counter, None)
        raw.setdefault("recovery_events", []).append({
            "event_type": "OPERATOR_MIGRATION",
            "event_id": raw["migration_event"],
            "from_schema": legacy_schema,
            "to_schema": SCHEMA_VERSION,
            "from_state": legacy_state,
            "target_state": raw["state"],
            "created_at": time.time(),
        })
    # Migration defaults for pre-state-machine or early schema documents.
    if legacy_schema < SCHEMA_VERSION:
        raw["schema_version"] = SCHEMA_VERSION
    else:
        raw.setdefault("schema_version", SCHEMA_VERSION)
    raw.setdefault("validated_artifacts", {})
    raw.setdefault("retry_counters", {})
    raw.setdefault("source_run_ids", [])
    raw.setdefault("state", ProofState.STRATEGY_TOURNAMENT.value)
    known = set(OrchestrationCheckpoint.__dataclass_fields__)
    raw = {key: value for key, value in raw.items() if key in known}
    raw["validated_artifacts"] = {
        role: ArtifactRef(**value)
        for role, value in raw["validated_artifacts"].items()
    }
    checkpoint = OrchestrationCheckpoint(**raw)
    checkpoint.proof_state
    if missing_capability_fields:
        checkpoint.adapter_blocked(
            "checkpoint capability incompatibility: "
            "MISSING_CAPABILITY_FIELDS:"
            + ",".join(sorted(missing_capability_fields)),
            status="INTEGRATION_BLOCKED",
        )
    else:
        integration_block_incompatible_checkpoint(checkpoint)
    return checkpoint


def binding_mismatch(
    checkpoint: OrchestrationCheckpoint,
    *,
    target_obligation_id: str,
    candidate_sha256: str,
    parent_statement_sha256: str,
    parent_signature_sha256: str,
    root_goal_sha256: str,
    ledger_id: str,
    ledger_version: int,
) -> str:
    expected = {
        "target_obligation_id": target_obligation_id,
        "candidate_sha256": candidate_sha256,
        "parent_statement_sha256": parent_statement_sha256,
        "parent_signature_sha256": parent_signature_sha256,
        "root_goal_sha256": root_goal_sha256,
        "ledger_id": ledger_id,
    }
    for field_name, value in expected.items():
        saved = str(getattr(checkpoint, field_name))
        if saved and saved != str(value):
            return f"{field_name}-mismatch"
    if checkpoint.ledger_version and checkpoint.ledger_version != ledger_version:
        return "ledger-version-mismatch"
    return ""


def persist_validated_artifact(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    role: str,
    payload: dict[str, Any],
    dependencies: list[str],
    source_run_id: str,
    artifact_schema_version: int = 1,
    save: bool = True,
) -> ArtifactRef:
    if checkpoint.target_context_hash:
        from autoresearch.prefill.target_context import require_binding

        require_binding(
            checkpoint,
            target_obligation_id=str(
                payload.get("target_obligation_id", checkpoint.target_obligation_id)
            ),
            parent_statement_hash=str(
                payload.get(
                    "parent_statement_hash",
                    payload.get(
                        "parent_statement_sha256",
                        checkpoint.parent_statement_sha256,
                    ),
                )
            ),
            context_hash=str(
                payload.get("target_context_hash", checkpoint.target_context_hash)
            ),
            strategy_plan_hash=str(
                payload.get(
                    "strategy_plan_hash", checkpoint.target_strategy_plan_hash,
                )
            ),
            environment_hash=str(
                payload.get(
                    "environment_hash", checkpoint.target_environment_hash,
                )
            ),
        )
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    artifact_dir = checkpoint_path.with_suffix(".artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    artifact_path = artifact_dir / f"{digest}.json"
    if not artifact_path.exists():
        temporary = artifact_path.with_name(
            f".{artifact_path.name}.{os.getpid()}.tmp",
        )
        try:
            temporary.write_bytes(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, artifact_path)
        finally:
            temporary.unlink(missing_ok=True)
    os.chmod(artifact_path, 0o600)
    ref = ArtifactRef(
        role=role,
        sha256=digest,
        schema_version=artifact_schema_version,
        dependencies=list(dependencies),
        path=str(artifact_path),
        source_run_id=source_run_id,
        validated_at=time.time(),
        target_context_hash=checkpoint.target_context_hash,
        target_obligation_id=checkpoint.target_obligation_id,
        parent_statement_hash=checkpoint.parent_statement_sha256,
        strategy_plan_hash=checkpoint.target_strategy_plan_hash,
        environment_hash=checkpoint.target_environment_hash,
        candidate_sha256=checkpoint.candidate_sha256,
        ledger_id=checkpoint.ledger_id,
        ledger_version=checkpoint.ledger_version,
        reusable=True,
        validation_status="validated",
    )
    checkpoint.validated_artifacts[role] = ref
    if save:
        save_checkpoint(checkpoint_path, checkpoint)
    return ref


def verified_reuse_provenance(
    checkpoint: OrchestrationCheckpoint,
) -> dict[str, Any]:
    """Derive reuse only from current, target-bound, content-verified refs.

    Historical transcript/cache strings and checkpoint booleans are
    intentionally ignored.  Invalid references are reported but never count
    as reuse.
    """
    references = checkpoint.validated_artifacts
    reference_hashes = {ref.sha256 for ref in references.values()}
    binding_hashes = {
        value for value in (
            checkpoint.candidate_sha256,
            checkpoint.strategy_sha256,
            checkpoint.parent_statement_sha256,
            checkpoint.parent_signature_sha256,
            checkpoint.root_goal_sha256,
            checkpoint.target_context_hash,
            checkpoint.target_strategy_plan_hash,
            checkpoint.target_environment_hash,
        ) if value
    }
    role_reused: dict[str, bool] = {}
    verified_artifacts: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, list[str]] = {}
    for role, ref in sorted(references.items()):
        reasons: list[str] = []
        if ref.role != role:
            reasons.append("role-mismatch")
        if not ref.reusable or ref.validation_status != "validated":
            reasons.append("not-marked-reusable-validated")
        expected_pairs = (
            ("target-obligation", ref.target_obligation_id,
             checkpoint.target_obligation_id),
            ("target-context", ref.target_context_hash,
             checkpoint.target_context_hash),
            ("parent-statement", ref.parent_statement_hash,
             checkpoint.parent_statement_sha256),
            ("strategy-plan", ref.strategy_plan_hash,
             checkpoint.target_strategy_plan_hash),
            ("environment", ref.environment_hash,
             checkpoint.target_environment_hash),
            ("candidate", ref.candidate_sha256,
             checkpoint.candidate_sha256),
            ("ledger-id", ref.ledger_id, checkpoint.ledger_id),
        )
        for label, actual, expected in expected_pairs:
            if expected and actual != expected:
                reasons.append(f"{label}-mismatch")
        if checkpoint.ledger_version and (
            int(ref.ledger_version) != int(checkpoint.ledger_version)
        ):
            reasons.append("ledger-version-mismatch")
        if ref.schema_version < 1:
            reasons.append("schema-version-invalid")
        if any(
            dependency not in reference_hashes
            and dependency not in binding_hashes
            for dependency in ref.dependencies
            if dependency
        ):
            reasons.append("dependency-outside-current-dag")
        encoded = b""
        try:
            encoded = Path(ref.path).expanduser().read_bytes()
        except OSError:
            reasons.append("artifact-unavailable")
        if encoded and hashlib.sha256(encoded).hexdigest() != ref.sha256:
            reasons.append("artifact-hash-mismatch")
        if not reasons:
            try:
                payload = json.loads(encoded)
            except (TypeError, ValueError, json.JSONDecodeError):
                reasons.append("artifact-json-invalid")
            else:
                if not isinstance(payload, dict):
                    reasons.append("artifact-schema-invalid")
                else:
                    payload_role = payload.get("role")
                    if payload_role and payload_role != role:
                        reasons.append("payload-role-mismatch")
                    bindings = payload.get("bindings")
                    if isinstance(bindings, dict):
                        payload_expectations = {
                            "target_obligation_id": (
                                checkpoint.target_obligation_id
                            ),
                            "candidate_sha256": checkpoint.candidate_sha256,
                            "strategy_sha256": checkpoint.strategy_sha256,
                            "parent_statement_sha256": (
                                checkpoint.parent_statement_sha256
                            ),
                            "parent_signature_sha256": (
                                checkpoint.parent_signature_sha256
                            ),
                            "root_goal_sha256": checkpoint.root_goal_sha256,
                            "ledger_id": checkpoint.ledger_id,
                            "ledger_version": checkpoint.ledger_version,
                            "environment_sha256": (
                                checkpoint.target_environment_hash
                            ),
                            "target_context_hash": (
                                checkpoint.target_context_hash
                            ),
                            "strategy_plan_hash": (
                                checkpoint.target_strategy_plan_hash
                            ),
                        }
                        for name, expected in payload_expectations.items():
                            if not expected:
                                continue
                            actual = bindings.get(name)
                            if name == "ledger_version":
                                actual = int(actual or 0)
                                expected = int(expected)
                            if actual != expected:
                                reasons.append(f"payload-{name}-mismatch")
        reused = not reasons
        role_reused[role] = reused
        if reused:
            verified_artifacts[role] = asdict(ref)
        else:
            diagnostics[role] = reasons
    strategy_role = (
        "strategy_tournament"
        if "strategy_tournament" in role_reused else "strategy"
    )
    return {
        "strategy_reused": role_reused.get(strategy_role, False),
        "generator_reused": role_reused.get("generator", False),
        "critic_reused": role_reused.get("critic", False),
        "role_reused": role_reused,
        "reused_artifacts": verified_artifacts,
        "diagnostics": diagnostics,
    }


_APPROACH_DEPENDENT_ROLES = frozenset({
    "strategy",
    "strategy_tournament",
    "research_contract",
    "generator",
    "critic",
    "synthesis",
    "decomposer",
    "math_ir_translator",
    "formalizer",
    "proof_search",
    "prover",
    "adversarial_proponent",
    "judge",
})


def premise_outcome_fingerprint(
    *,
    target_obligation_id: str,
    outcome_type: str,
    decision: str,
    evidence: dict[str, Any],
    query_hash: str = "",
    environment_hash: str = "",
) -> str:
    """Bind one semantic outcome to its target, evidence, and environment."""
    payload = {
        "target_obligation_id": str(target_obligation_id),
        "outcome_type": PremiseAuditOutcomeType(outcome_type).value,
        "decision": str(decision),
        "evidence": evidence,
        "query_hash": str(query_hash),
        "environment_hash": str(environment_hash),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def reconcile_checkpoint_ledger_version(
    checkpoint: OrchestrationCheckpoint,
    authoritative_ledger_version: int,
) -> bool:
    """Repair advisory checkpoint version without inventing ledger commits."""
    authoritative = int(authoritative_ledger_version)
    if authoritative < 0:
        raise ValueError("ledger version cannot be negative")
    if checkpoint.ledger_version == authoritative:
        return False
    previous = checkpoint.ledger_version
    checkpoint.ledger_version = authoritative
    checkpoint.recovery_events.append({
        "event_type": "LEDGER_VERSION_RECONCILED",
        "event_id": hashlib.sha256(
            f"{checkpoint.ledger_id}:{previous}:{authoritative}".encode()
        ).hexdigest(),
        "checkpoint_version_before": previous,
        "authoritative_ledger_version": authoritative,
        "checkpoint_version_after": authoritative,
        "created_at": time.time(),
    })
    checkpoint.last_transition_reason = (
        f"ledger-authoritative-version-repair:{previous}->{authoritative}"
    )
    checkpoint.updated_at = time.time()
    return True


def apply_typed_premise_outcome(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    outcome_type: PremiseAuditOutcomeType,
    decision: str,
    owner: str,
    confidence: float,
    evidence: dict[str, Any],
    source_run_id: str,
    backjump_target: str = "",
    query_hash: str = "",
    environment_hash: str = "",
) -> bool:
    """Persist a typed audit decision, then route through semantic states.

    Returns ``False`` for an idempotent replay. A repairable definition gap is
    admitted only when both query and environment fingerprints are present and
    the resulting fingerprint has not already been consumed.
    """
    if checkpoint.proof_state != ProofState.PREMISE_AUDIT:
        if (
            checkpoint.premise_outcome_fingerprint
            and checkpoint.premise_outcome_fingerprint
            in checkpoint.consumed_premise_fingerprints
        ):
            return False
        raise ValueError(
            "typed premise outcome requires an active PREMISE_AUDIT"
        )
    outcome = PremiseAuditOutcomeType(outcome_type)
    if outcome == PremiseAuditOutcomeType.REPAIRABLE_DEFINITION_GAP and (
        not query_hash or not environment_hash
    ):
        raise ValueError(
            "REPAIRABLE_DEFINITION_GAP requires query and environment hashes"
        )
    fingerprint = premise_outcome_fingerprint(
        target_obligation_id=checkpoint.target_obligation_id,
        outcome_type=outcome.value,
        decision=decision,
        evidence=evidence,
        query_hash=query_hash,
        environment_hash=environment_hash,
    )
    if fingerprint in checkpoint.consumed_premise_fingerprints:
        return False

    artifact = {
        "schema_version": 1,
        "outcome_type": outcome.value,
        "decision": str(decision),
        "owner": str(owner),
        "confidence": float(confidence),
        "evidence": evidence,
        "backjump_target": str(backjump_target),
        "query_hash": str(query_hash),
        "environment_hash": str(environment_hash),
        "fingerprint": fingerprint,
    }
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="premise_outcome",
        payload=artifact,
        dependencies=[
            ref.sha256 for role, ref in sorted(
                checkpoint.validated_artifacts.items()
            ) if role != "premise_outcome"
        ],
        source_run_id=source_run_id,
    )
    checkpoint.premise_outcome_type = outcome.value
    checkpoint.premise_outcome_fingerprint = fingerprint
    checkpoint.premise_outcome_owner = str(owner)
    checkpoint.premise_decision = str(decision)
    checkpoint.premise_confidence = float(confidence)
    checkpoint.premise_evidence = dict(evidence)
    checkpoint.premise_backjump_target = str(backjump_target)

    invalidated = []
    if outcome in {
        PremiseAuditOutcomeType.APPROACH_FAILED,
        PremiseAuditOutcomeType.PREMISE_INVALIDATED,
        PremiseAuditOutcomeType.PARENT_STATEMENT_UNDERSPECIFIED,
    }:
        for role in sorted(_APPROACH_DEPENDENT_ROLES):
            ref = checkpoint.validated_artifacts.pop(role, None)
            if ref is None:
                continue
            invalidated.append(ref.sha256)
            checkpoint.invalidated_artifacts[ref.sha256] = {
                **asdict(ref),
                "audit_only": True,
                "reason_codes": [outcome.value],
                "premise_outcome_fingerprint": fingerprint,
            }
    checkpoint.premise_invalidated_artifacts = invalidated
    checkpoint.consumed_premise_fingerprints.append(fingerprint)
    checkpoint.recovery_events.append({
        "event_type": "TYPED_PREMISE_OUTCOME",
        "event_id": fingerprint,
        "outcome_type": outcome.value,
        "owner": owner,
        "decision": decision,
        "confidence": float(confidence),
        "backjump_target": backjump_target,
        "invalidated_artifact_hashes": invalidated,
        "source_run_id": source_run_id,
        "created_at": time.time(),
    })
    # The decision and its evidence must be durable while PREMIS_AUDIT is still
    # the current state. Only then may the semantic transition be committed.
    save_checkpoint(checkpoint_path, checkpoint)

    if outcome == PremiseAuditOutcomeType.PREMISE_SUSPECTED:
        checkpoint.last_transition_reason = (
            "premise-suspected:await-independent-auditor-and-proponent"
        )
        save_checkpoint(checkpoint_path, checkpoint)
        return True
    intermediate = {
        PremiseAuditOutcomeType.APPROACH_FAILED: ProofState.APPROACH_FAILED,
        PremiseAuditOutcomeType.PREMISE_INVALIDATED: (
            ProofState.PREMISE_INVALIDATED
        ),
        PremiseAuditOutcomeType.PARENT_STATEMENT_UNDERSPECIFIED: (
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED
        ),
        PremiseAuditOutcomeType.REPAIRABLE_DEFINITION_GAP: (
            ProofState.REPAIRABLE_DEFINITION_GAP
        ),
    }[outcome]
    checkpoint.transition(
        intermediate,
        f"typed-premise-outcome:{outcome.value}",
        source_run_id=source_run_id,
        strategy_reused=False,
    )
    save_checkpoint(checkpoint_path, checkpoint)
    destination = (
        ProofState.DEFINITION_RESOLUTION
        if outcome == PremiseAuditOutcomeType.REPAIRABLE_DEFINITION_GAP
        else ProofState.STRATEGY_TOURNAMENT
    )
    checkpoint.transition(
        destination,
        (
            "repairable-definition-gap:new-query-environment"
            if outcome == PremiseAuditOutcomeType.REPAIRABLE_DEFINITION_GAP
            else f"{outcome.value.lower()}:typed-backjump-new-route"
        ),
        source_run_id=source_run_id,
        strategy_reused=False,
    )
    if backjump_target:
        checkpoint.target_obligation_id = backjump_target
    save_checkpoint(checkpoint_path, checkpoint)
    return True


def load_validated_artifacts(
    checkpoint: OrchestrationCheckpoint,
) -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    dependency_roles = {
        "definition_auditor": (),
        "counterexample_worker": ("definition_auditor",),
        "decomposer": ("definition_auditor",),
        "math_ir_translator": ("decomposer",),
        "proof_search": ("math_ir_translator",),
        "adversarial_proponent": (
            "decomposer", "math_ir_translator", "proof_search",
        ),
    }
    for state in ROLE_ORDER:
        role = ROLE_ARTIFACT_KEYS.get(state)
        if role is None or role not in checkpoint.validated_artifacts:
            break
        ref = checkpoint.validated_artifacts[role]
        if checkpoint.target_context_hash and (
            ref.target_context_hash != checkpoint.target_context_hash
            or ref.target_obligation_id != checkpoint.target_obligation_id
            or ref.parent_statement_hash != checkpoint.parent_statement_sha256
            or ref.strategy_plan_hash != checkpoint.target_strategy_plan_hash
            or ref.environment_hash != checkpoint.target_environment_hash
        ):
            raise ValueError(f"{role} artifact target context mismatch")
        dependencies_valid = (
            len(ref.dependencies) == 1
            if role == "judge"
            else ref.dependencies == [
                hashes[name] for name in dependency_roles[role]
            ]
        )
        if ref.schema_version != 1 or not dependencies_valid:
            raise ValueError(f"{role} artifact dependency mismatch")
        path = Path(ref.path)
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != ref.sha256:
            raise ValueError(f"{role} artifact hash mismatch")
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise ValueError(f"{role} artifact schema mismatch")
        loaded[role] = payload
        hashes[role] = ref.sha256
    return loaded


def state_for_role(role: str) -> ProofState:
    return {
        "definition_auditor": ProofState.DEFINITION_AUDITOR,
        "counterexample_worker": ProofState.COUNTEREXAMPLE_WORKER,
        "decomposer": ProofState.DECOMPOSER,
        "formalizer": ProofState.MATH_IR_TRANSLATION,
        "math_ir_translator": ProofState.MATH_IR_TRANSLATION,
        "prover": ProofState.PROOF_SEARCH,
        "proof_search": ProofState.PROOF_SEARCH,
        "adversarial_proponent": ProofState.ADVERSARIAL_REVIEW,
        "judge": ProofState.JUDGE,
    }[role]


def classify_failure(role: str, error: str) -> ProofState:
    text = str(error).lower()
    current = state_for_role(role)
    if any(marker in text for marker in (
        "missing definition", "unknown constant", "unknown type",
        "unknown symbol", "unknown operator",
    )):
        return ProofState.DEFINITION_RESOLUTION
    if role in {"formalizer", "math_ir_translator"}:
        return ProofState.MATH_IR_TRANSLATION
    if role in {"prover", "proof_search"}:
        if "type mismatch" in text or "signature" in text:
            return ProofState.MATH_IR_TRANSLATION
        return ProofState.PROOF_SEARCH
    if role == "judge":
        for marker, state in (
            ("definition", ProofState.DEFINITION_AUDITOR),
            ("counterexample", ProofState.COUNTEREXAMPLE_WORKER),
            ("decomposition", ProofState.DECOMPOSER),
            ("formal", ProofState.MATH_IR_TRANSLATION),
            ("proof", ProofState.PROOF_SEARCH),
            ("defense", ProofState.ADVERSARIAL_REVIEW),
        ):
            if marker in text:
                return state
    if "premise" in text:
        return ProofState.PREMISE_AUDIT
    if "approach_failed" in text or "mathematical approach" in text:
        return ProofState.APPROACH_FAILED
    return current
