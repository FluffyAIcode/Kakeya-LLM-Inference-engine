#!/usr/bin/env python3
"""Interactive Generator/Critic REPL with real-time token streaming."""
from __future__ import annotations

import argparse
import ast
import atexit
import difflib
import hashlib
import json
import os
import re
import select
import signal
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

from scripts.agent_gan_inference_demo import (
    _agent_cache_gate,
    _infer,
    build_critic_context,
    decode_complete_response,
)
from scripts.benchmark_prefill_architecture import (
    _ensure_services,
    _json_request,
)
from inference_engine.bench.prefill_fleet_report import summarize_stages
from autoresearch.prefill.prepare import (
    REPORT_PROVENANCE_SCHEMA_VERSION,
    ResumeValidationError,
    critic_artifact_payload,
    load_critic_artifact,
)
from autoresearch.prefill.lean_gate import (
    LeanSignatureResult,
    lean_proof_contract_ref,
    lean_symbol_semantic_hash,
    lean_signature_contract_ref,
    lean_theorem_signature_hash,
    lint_lean_model_prompt,
    normalize_lean_signature,
    normalize_registered_latex_identifiers,
    register_lean_symbol_table,
    resolve_lean_contract,
    resolve_lean_symbol_table,
    validate_lean_proof,
    validate_lean_signature,
)
from autoresearch.prefill.live_status import AtomicLiveStatus
from autoresearch.prefill.host_compiler import run_host_gates
from autoresearch.prefill.architecture_v9 import (
    run_architecture_v9_entry,
    run_host_definition_gate,
)
from autoresearch.prefill.cursor_strategy import CursorStrategyAdapter
from autoresearch.prefill.strategy_tournament import (
    StrategyEvent,
    build_decomposition_exploration_plan,
)
from autoresearch.prefill.decomposition_exploration import (
    gate_decomposition_exploration,
    generate_private_candidate_refs,
    next_formalization_candidate,
    prefilter_private_candidates,
    rank_private_candidates,
)
from autoresearch.prefill.candidate_representation import (
    RepresentationOutcome,
    analyze_private_candidate_representation,
    persist_representation_gap_report,
)
from autoresearch.prefill.stepwise_proof import (
    ActionSelection,
    LeanExecutionContext,
    ProofGoal,
    attempt_step,
    enumerate_applicable_actions,
    lean_step_executor,
    new_search_state,
    persist_search_state,
)
from autoresearch.prefill.creative_decomposition import (
    assert_no_scratchpad_content,
    build_candidate_set,
    persist_private_scratchpad,
    rank_candidates,
    rank_short_choice,
    synthesis_manifest,
    synthesis_trigger,
)
from autoresearch.prefill.evidence_planner import (
    build_evidence_gap_graph,
    first_executable_node,
    generate_proof_plans,
    host_evidence_context,
    validate_proof_plan,
)
from autoresearch.prefill.math_ir import (
    GateStatus,
)
from autoresearch.prefill.orchestration_state import (
    DefinitionAuditOutcomeType,
    OrchestrationCheckpoint,
    PremiseAuditOutcomeType,
    ProofState,
    apply_typed_premise_outcome,
    archive_decomposition_rejection,
    binding_mismatch,
    classify_failure,
    compact_decomposition_novelty_ledger,
    load_checkpoint as load_orchestration_checkpoint,
    load_validated_artifacts,
    persist_validated_artifact,
    reconcile_checkpoint_ledger_version,
    require_typed_dispatch,
    route_definition_audit_outcome,
    save_checkpoint as save_orchestration_checkpoint,
    sha256_text,
    state_for_role,
)
from autoresearch.prefill.definition_registry import (
    build_definition_choice_registry,
    serialize_definition_audit,
)
from autoresearch.prefill.theorem_cards import (
    build_theorem_card_index,
    pinned_environment_hash,
    search_theorem_cards,
)
from autoresearch.prefill.target_context import mathematical_state_fingerprint
from autoresearch.prefill.resume_certificate import (
    ResumeCertificateError,
    consume_resume_certificate,
    current_runtime_binding,
    resume_requires_certificate,
)
from autoresearch.prefill.semantic_decompose import (
    SemanticResponseIncomplete,
    SemanticUnitTooLarge,
    admit_token_ids,
    build_proof_step_interface,
    downstream_output_cap,
    json_syntax_diagnostic,
    lint_structured_prompt,
    repair_json_backslashes,
    scan_single_artifact_object,
    scan_structured_artifact_prefix,
    serialize_proof_step_interface,
    structured_role_minimum_output_tokens,
    structured_transport_complete,
    structured_output_cap,
)
from autoresearch.prefill.typed_transport import (
    AdapterError,
    DecodedRoleFields,
    ROLE_TRANSPORT_REGISTRY,
    decode_role_fields,
    host_artifact,
    transport_prompt,
    typed_transport_complete,
)


class TimestampedTee:
    """Mirror Terminal output to a line-timestamped, immediately flushed log."""

    def __init__(self, terminal, log_path: Path, timestamp_fn=None) -> None:
        self.terminal = terminal
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        self.timestamp_fn = timestamp_fn or (
            lambda: datetime.now().astimezone().isoformat(timespec="milliseconds")
        )
        self._line_start = True
        self._lock = threading.RLock()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def fileno(self):
        return self.terminal.fileno()

    def isatty(self):
        return self.terminal.isatty()

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self.terminal.write(text)
            if self.log.closed:
                return len(text)
            for part in text.splitlines(keepends=True):
                if self._line_start:
                    self.log.write(f"[{self.timestamp_fn()}] ")
                self.log.write(part)
                self._line_start = part.endswith(("\n", "\r"))
            self.log.flush()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self.terminal.flush()
            if not self.log.closed:
                self.log.flush()

    def log_only(self, event: str) -> None:
        with self._lock:
            if self.log.closed:
                return
            if not self._line_start:
                self.log.write("\n")
            self.log.write(f"[{self.timestamp_fn()}] {event.rstrip()}\n")
            self.log.flush()
            self._line_start = True

    def close_log(self) -> None:
        with self._lock:
            if sys.stdout is self:
                sys.stdout = self.terminal
            if sys.stderr is self:
                sys.stderr = self.terminal
            if not self.log.closed:
                self.log.flush()
                self.log.close()


def install_signal_protection() -> None:
    def ignore_sigterm(signum, _frame):
        print(
            f"\n[protected] ignored external signal {signum}. "
            "Type /quit to approve shutdown.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, ignore_sigterm)


def _telemetry_request(url: str, **kwargs):
    try:
        return _json_request(url, timeout=2, **kwargs)
    except Exception as exc:
        print(
            f"[telemetry-warning] {type(exc).__name__}: {exc}; "
            "inference will continue",
            flush=True,
        )
        return None


def dispatch_certified_architecture9_role(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    project_root: Path,
    interface_strategy_adapter: CursorStrategyAdapter | None = None,
) -> tuple[OrchestrationCheckpoint, str]:
    """Dispatch a consumed certificate to its host-owned role."""
    if checkpoint.proof_state != ProofState.DEFINITION_RESOLUTION:
        return checkpoint, ""
    return run_host_definition_gate(
        checkpoint_path,
        checkpoint,
        project_root=project_root,
        interface_strategy_adapter=interface_strategy_adapter,
    )


_RUNTIME_ARTIFACT = re.compile(
    r"^\s*(?:generator>|critic>|prompt>|\[(?:metrics|allens|error|"
    r"telemetry-warning|protected|supervisor)\]|Traceback\b)",
    re.IGNORECASE,
)


def is_runtime_artifact_prompt(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and bool(_RUNTIME_ARTIFACT.match(lines[0]))


class ReplPhase(str, Enum):
    WAITING_FOR_GOAL = "waiting_for_goal"
    READY = "ready"
    RUNNING = "running"


@dataclass(frozen=True)
class ReplCommand:
    action: str
    payload: str = ""


@dataclass
class ReplCheckpoint:
    research_goal: str
    previous_generator: str = ""
    previous_critic: str = ""
    last_run_id: str = ""
    schema_version: int = 1


@dataclass
class CriticIssueBatch:
    issue_id: str
    issues: list[str]
    status: str = "pending"
    consumed_by_run: str = ""
    schema_version: int = 1


@dataclass
class ProofObligation:
    obligation_id: str
    statement: str
    status: str = "UNRESOLVED"
    parent_id: str = ""
    last_run_id: str = ""
    last_evidence: str = ""
    formal_status: str = "UNFORMALIZED"
    lean_signature: str = ""
    lean_signature_hash: str = ""
    formalization_error: str = ""
    invalidation_kind: str = ""
    quarantine_reason: str = ""
    quarantine_root_id: str = ""
    quarantine_run_id: str = ""
    quarantine_prior_status: str = ""
    quarantine_confidence: float = 0.0
    quarantine_evidence_type: str = ""
    quarantine_evidence_source: str = ""
    quarantine_auditor_run_id: str = ""
    quarantine_proponent_run_id: str = ""
    quarantine_reversible_status: str = ""
    premise_review_status: str = ""
    temporary_quarantine_reason: str = ""
    temporary_quarantine_root_id: str = ""
    temporary_quarantine_run_id: str = ""
    invalidation_prior_status: str = ""
    premise_audit_confidence: float = 0.0
    premise_audit_evidence_type: str = ""
    premise_audit_evidence_source: str = ""
    premise_auditor_run_id: str = ""
    premise_proponent_run_id: str = ""
    premise_review_reason: str = ""
    decomposition_certificate_hash: str = ""
    reduction_theorem_hash: str = ""
    reduction_theorem_status: str = ""
    decomposition_role_run_ids: dict = field(default_factory=dict)
    dependency_labels: list[str] = field(default_factory=list)
    dependency_ids: list[str] = field(default_factory=list)
    certificate_reversible_status: str = ""
    public_assumptions: list[str] = field(default_factory=list)
    source_card_ids: list[str] = field(default_factory=list)
    root_candidate_ids: list[str] = field(default_factory=list)
    proposition_hash: str = ""
    root_bootstrap_certificate_hash: str = ""


@dataclass
class NoGoLesson:
    claim_hash: str
    refuted_premise: str
    evidence: str
    source_obligation_id: str
    run_id: str
    confidence: float = 0.0
    evidence_type: str = ""
    evidence_source: str = ""
    auditor_run_id: str = ""
    proponent_run_id: str = ""
    reversible_status: str = "ACTIVE"


@dataclass(frozen=True)
class PremiseSuspicion:
    obligation_id: str
    premise: str
    evidence_type: str
    evidence_artifact: dict
    critic_evidence: str
    claim_schema: dict = field(default_factory=dict)
    claim_hash: str = ""
    target_lean_signature_hash: str = ""


@dataclass(frozen=True)
class PremiseAudit:
    obligation_id: str
    status: str
    evidence_type: str
    evidence_source: str
    confidence: float
    artifact: dict
    analysis: str
    run_id: str = ""


@dataclass(frozen=True)
class PremiseDefense:
    obligation_id: str
    status: str
    correction: str
    failure_reason: str
    evidence: str
    run_id: str = ""


@dataclass(frozen=True)
class PremiseReview:
    status: str
    verified: bool
    confidence: float = 0.0
    evidence_type: str = ""
    evidence_source: str = ""
    auditor_run_id: str = ""
    proponent_run_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class DefinitionAudit:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    definitions: list[dict]
    missing_definitions: list[dict]
    audit_outcome: str = ""


@dataclass(frozen=True)
class CounterexampleReport:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    status: str
    cases: list[dict]


@dataclass(frozen=True)
class DecompositionProposal:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    parent_statement: str
    child: dict
    public_assumptions: list[str]
    reduction_contract: dict


@dataclass(frozen=True)
class FormalizationBundle:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    parent_signature_source: str
    parent_signature_hash: str
    parent_newly_formalized: bool
    child: dict
    reduction_theorem_source: str
    reduction_signature_hash: str


@dataclass(frozen=True)
class ProofAttempt:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    status: str
    reduction_theorem_source: str


@dataclass(frozen=True)
class DefenseReport:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    status: str
    issues: list[str]
    repairs: list[str]


@dataclass(frozen=True)
class JudgeDecision:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    decision: str
    reason: str


@dataclass
class DecompositionCertificateResult:
    verified: bool
    errors: list[str]
    artifacts: dict
    artifact_hashes: dict
    transcripts: dict
    role_run_ids: dict
    validation: dict
    created: list[ProofObligation] = field(default_factory=list)
    certificate_hash: str = ""


def build_resumed_report_provenance(
    checkpoint: OrchestrationCheckpoint,
    critic_payload: dict,
    stages: list[dict],
) -> dict:
    """Describe a partial run without claiming unexecuted benchmark stages."""
    critic_ref = checkpoint.validated_artifacts["critic"]
    bindings = critic_payload["bindings"]
    return {
        "schema_version": REPORT_PROVENANCE_SCHEMA_VERSION,
        "mode": "resumed",
        "resumed_from_state": checkpoint.state,
        "resumed_from_role": checkpoint.current_role,
        "strategy_reused": True,
        "generator_reused": True,
        "critic_reused": True,
        "bindings": {
            name: bindings[name]
            for name in (
                "target_obligation_id",
                "candidate_sha256",
                "strategy_sha256",
                "parent_statement_sha256",
                "parent_signature_sha256",
                "root_goal_sha256",
                "ledger_id",
                "ledger_version",
            )
        },
        "reused_artifacts": {
            "strategy": {
                "sha256": bindings["strategy_sha256"],
                "source_run_id": critic_payload["source_run_id"],
            },
            "generator": {
                "sha256": bindings["generator_output_sha256"],
                "source_run_id": critic_payload["source_run_id"],
            },
            "critic": asdict(critic_ref),
        },
        "newly_executed_stages": [
            str(stage.get("name", ""))
            for stage in stages
        ],
    }


def build_architecture7_report_provenance(
    checkpoint_before: OrchestrationCheckpoint,
    checkpoint_after: OrchestrationCheckpoint,
    stages: list[dict],
    *,
    ledger_sha256: str,
    environment_sha256: str,
) -> dict:
    """Report typed continuation with exact stage and checkpoint ownership."""
    before_artifacts = {
        role: asdict(reference)
        for role, reference in checkpoint_before.validated_artifacts.items()
    }
    after_artifacts = {
        role: asdict(reference)
        for role, reference in checkpoint_after.validated_artifacts.items()
    }
    produced_artifacts = {
        role: reference
        for role, reference in after_artifacts.items()
        if (
            role not in before_artifacts
            or before_artifacts[role]["sha256"] != reference["sha256"]
        )
    }
    source_runs = {
        role: reference["source_run_id"]
        for role, reference in after_artifacts.items()
    }
    return {
        "schema_version": REPORT_PROVENANCE_SCHEMA_VERSION,
        "mode": "typed_partial_resume_v2",
        "checkpoint_before": {
            "state": checkpoint_before.state,
            "role": checkpoint_before.current_role,
        },
        "checkpoint_after": {
            "state": checkpoint_after.state,
            "role": checkpoint_after.current_role,
        },
        "bindings": {
            "target_obligation_id": checkpoint_after.target_obligation_id,
            "candidate_sha256": checkpoint_after.candidate_sha256,
            "strategy_sha256": (
                checkpoint_after.strategy_sha256
                or checkpoint_after.candidate_sha256
            ),
            "parent_statement_sha256": (
                checkpoint_after.parent_statement_sha256
            ),
            "parent_signature_sha256": (
                checkpoint_after.parent_signature_sha256
            ),
            "root_goal_sha256": checkpoint_after.root_goal_sha256,
            "ledger_id": checkpoint_after.ledger_id,
            "ledger_version": checkpoint_after.ledger_version,
            "ledger_sha256": ledger_sha256,
            "environment_sha256": environment_sha256,
            "research_contract_id": checkpoint_after.research_contract_id,
            "research_contract_hash": checkpoint_after.research_contract_hash,
        },
        "source_runs": source_runs,
        "reused_artifacts": before_artifacts,
        "produced_artifacts": produced_artifacts,
        "reused_stages": list(before_artifacts),
        "newly_executed_stages": [
            str(stage.get("name", "")) for stage in stages
        ],
    }


@dataclass
class ProofObligationLedger:
    ledger_id: str
    obligations: list[ProofObligation]
    version: int = 1
    schema_version: int = 1
    no_go_lessons: list[NoGoLesson] | None = None
    backjump_target_id: str = ""

    def __post_init__(self) -> None:
        if self.no_go_lessons is None:
            self.no_go_lessons = []


def save_proof_ledger(path: Path, ledger: ProofObligationLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = asdict(ledger)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def save_decomposition_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_decomposition_manifest(path: Path) -> dict:
    """Read both archived v1 and current manifests without admitting artifacts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decomposition manifest must be a JSON object")
    return payload


def load_proof_ledger(path: Path) -> ProofObligationLedger | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    obligations = [
        ProofObligation(**item) for item in raw.pop("obligations", [])
    ]
    no_go_lessons = [
        NoGoLesson(**item) for item in (raw.pop("no_go_lessons", []) or [])
    ]
    ledger = ProofObligationLedger(
        obligations=obligations,
        no_go_lessons=no_go_lessons,
        **raw,
    )
    obligation_ids = {
        item.obligation_id for item in ledger.obligations
    }
    if (
        ledger.schema_version != 1
        or not ledger.ledger_id
        or not ledger.obligations
        or len(obligation_ids) != len(ledger.obligations)
        or len({
            lesson.claim_hash for lesson in ledger.no_go_lessons
        }) != len(ledger.no_go_lessons)
        or any(
            item.parent_id
            and (
                item.parent_id not in obligation_ids
                or item.parent_id == item.obligation_id
            )
            for item in ledger.obligations
        )
    ):
        raise ValueError("invalid proof obligation ledger")
    return ledger


def pending_obligations(
    ledger: ProofObligationLedger | None,
) -> list[ProofObligation]:
    if ledger is None:
        return []
    by_id = {
        item.obligation_id: item for item in ledger.obligations
    }

    def has_invalid_ancestor(item: ProofObligation) -> bool:
        cursor = item.parent_id
        visited = set()
        while cursor and cursor not in visited:
            visited.add(cursor)
            ancestor = by_id.get(cursor)
            if ancestor is None:
                break
            if (
                ancestor.status == "QUARANTINED"
                or (
                    ancestor.status == "DISPROVED"
                    and ancestor.invalidation_kind in {
                        "PREMISE",
                        "PREMISE_INVALIDATED",
                    }
                )
            ):
                return True
            cursor = ancestor.parent_id
        return False

    eligible = {
        item.obligation_id
        for item in ledger.obligations
        if item.status == "UNRESOLVED" and not has_invalid_ancestor(item)
    }
    unresolved_parent_ids = {
        item.parent_id
        for item in ledger.obligations
        if item.obligation_id in eligible and item.parent_id
    }
    return [
        item for item in ledger.obligations
        if (
            item.obligation_id in eligible
            and item.obligation_id not in unresolved_parent_ids
        )
    ]


def quarantine_terminal_interface_target(
    ledger: ProofObligationLedger,
    *,
    target_id: str,
    exhaustion_hash: str,
    source_run_id: str,
) -> bool:
    """Atomically prepare one root target for durable terminal quarantine."""
    target = next(
        (item for item in ledger.obligations if item.obligation_id == target_id),
        None,
    )
    if target is None:
        raise ValueError("interface exhaustion target is absent from ledger")
    reason = "TARGET_INTERFACE_EXHAUSTED:" + exhaustion_hash
    if (
        target.status == "QUARANTINED"
        and target.quarantine_reason == reason
        and target.quarantine_reversible_status == "ACTIVE"
    ):
        return False
    if target.status != "UNRESOLVED":
        raise ValueError("only an unresolved target may be quarantined")
    target.quarantine_prior_status = target.status
    target.status = "QUARANTINED"
    target.quarantine_reason = reason
    target.quarantine_root_id = target_id
    target.quarantine_run_id = source_run_id
    target.quarantine_confidence = 1.0
    target.quarantine_evidence_type = "CONTENT_ADDRESSED_EXHAUSTION"
    target.quarantine_evidence_source = exhaustion_hash
    target.quarantine_auditor_run_id = source_run_id
    target.quarantine_proponent_run_id = ""
    target.quarantine_reversible_status = "ACTIVE"
    ledger.backjump_target_id = "ROOT_UNAVAILABLE"
    ledger.version += 1
    return True


def format_proof_ledger(
    ledger: ProofObligationLedger,
    obligations: list[ProofObligation] | None = None,
) -> str:
    selected = obligations if obligations is not None else pending_obligations(ledger)
    ancestry = []
    by_id = {
        item.obligation_id: item
        for item in ledger.obligations
    }
    if len(selected) == 1:
        cursor = selected[0].parent_id
        visited = set()
        while cursor and cursor not in visited:
            visited.add(cursor)
            ancestor = by_id.get(cursor)
            if ancestor is None:
                break
            ancestry.append(ancestor)
            cursor = ancestor.parent_id
        ancestry.reverse()
    ancestry_text = "\n".join(
        f"- {item.obligation_id}: {item.statement}"
        for item in ancestry
    )
    relevant_ids = {
        item.obligation_id for item in (*ancestry, *selected)
    }

    def lesson_is_relevant(lesson: NoGoLesson) -> bool:
        cursor = lesson.source_obligation_id
        visited = set()
        while cursor and cursor not in visited:
            if cursor in relevant_ids:
                return True
            visited.add(cursor)
            source = by_id.get(cursor)
            if source is None:
                break
            cursor = source.parent_id
        return False

    items = "\n".join(
        f"- {item.obligation_id}"
        f"{f' (parent={item.parent_id})' if item.parent_id else ''}: "
        f"{item.statement}"
        f"{f' [lean_signature_hash={item.lean_signature_hash}]' if item.lean_signature_hash else ''}"
        for item in selected
    )
    no_go_text = "\n".join(
        f"- {lesson.claim_hash}: {lesson.refuted_premise}\n"
        f"  Evidence: {lesson.evidence}\n"
        f"  Verification: type={lesson.evidence_type or '(legacy)'} "
        f"source={lesson.evidence_source or '(legacy)'} "
        f"confidence={lesson.confidence:.2f} "
        f"auditor={lesson.auditor_run_id or '(legacy)'} "
        f"proponent={lesson.proponent_run_id or '(legacy)'}"
        for lesson in ledger.no_go_lessons
        if (
            lesson.reversible_status == "ACTIVE"
            and lesson_is_relevant(lesson)
        )
    )
    return (
        f"PROOF OBLIGATION LEDGER id={ledger.ledger_id} "
        f"version={ledger.version}\n"
        f"BACKJUMP TARGET: {ledger.backjump_target_id or '(none)'}\n"
        "NO-GO PREMISES (never assume, rename, or propose these):\n"
        f"{no_go_text or '(none)'}\n"
        f"COMPLETE ANCESTOR CHAIN:\n{ancestry_text or '(root target)'}\n"
        f"CURRENT TARGET:\n{items}\n"
        "Generator requirement: emit `### ISSUE_RESPONSE <ID>` for every "
        "pending ID, with `Correction:`, `Derivation:`, and `Remaining gap:`. "
        "Critic requirement: emit `### ISSUE_VERDICT <ID>` for every pending "
        "ID, with `Status: PROVED|DISPROVED|UNRESOLVED`, `Evidence:`, and "
        "`Missing lemma:`. A DISPROVED verdict must also emit "
        "`Invalidation: APPROACH|PREMISE_SUSPECTED`. A premise suspicion must "
        "also emit `Premise refuted:`, `Evidence type: "
        "FINITE_COUNTEREXAMPLE|SYMBOLIC_CONTRADICTION|LEAN_PROOF|"
        "PINNED_THEOREM`, and one-line JSON `Evidence artifact:` containing "
        "exactly `claim`. Arithmetic claim fields are `schema_version:1`, "
        "`quantifier:FOR_ALL`, `variables`, `domain`, `lhs`, the claimed "
        "`relation`, and `rhs`; do not include a witness. Lean claims must "
        "bind `contract:NEGATION_OF_TARGET_SIGNATURE` and the exact host "
        "`lean_signature_hash`. Suspicion "
        "starts independent audit and defense; it never directly invalidates "
        "the premise. Use APPROACH when only the attempted derivation fails. "
        "For an UNRESOLVED verdict, request exactly one frontier step. Do not "
        "emit Lean; only the certified Formalizer and Prover may introduce Lean."
    )


_ISSUE_RESPONSE = re.compile(
    r"^### ISSUE_RESPONSE\s+(\S+)",
    re.MULTILINE,
)
_ISSUE_VERDICT = re.compile(
    r"^### ISSUE_VERDICT(?:[ \t]+(\S+))?[ \t]*$"
    r"(?P<body>.*?)(?=^### ISSUE_VERDICT(?:[ \t]+\S+)?[ \t]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)

_PREMISE_AUDIT = re.compile(
    r"^### PREMISE_AUDIT\s+(\S+)\s*$"
    r"(?P<body>.*?)(?=^### |\Z)",
    re.MULTILINE | re.DOTALL,
)
_PREMISE_DEFENSE = re.compile(
    r"^### PREMISE_DEFENSE\s+(\S+)\s*$"
    r"(?P<body>.*?)(?=^### |\Z)",
    re.MULTILINE | re.DOTALL,
)
_VERIFIABLE_EVIDENCE_TYPES = {
    "FINITE_COUNTEREXAMPLE",
    "SYMBOLIC_CONTRADICTION",
    "LEAN_PROOF",
    "PINNED_THEOREM",
}


def _structured_field(body: str, name: str) -> str:
    match = re.search(
        rf"^\*{{0,2}}{re.escape(name)}:\*{{0,2}}[ \t]*(.*)$",
        body,
        re.MULTILINE,
    )
    if match is None:
        return ""
    value = match.group(1).strip()
    if value:
        return value
    for line in body[match.end():].splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("### ") or re.match(
            r"^\*{0,2}[^:]+:\*{0,2}(?:\s|$)",
            candidate,
        ):
            return ""
        return candidate
    return ""


def _json_artifact(value: str) -> dict:
    try:
        artifact = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        if not isinstance(value, str):
            return {}
        repaired = repair_json_backslashes(value)
        try:
            artifact = json.loads(repaired)
        except json.JSONDecodeError:
            return {}
    return artifact if isinstance(artifact, dict) else {}


def _normalize_arithmetic_expression(
    expression: object,
    variables: set[str],
) -> tuple[str, set[str]]:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("claim expression must be non-empty text")
    tree = ast.parse(expression, mode="eval")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown claim variable: {node.id}")
            names.add(node.id)
        elif isinstance(node, ast.Constant):
            if (
                not isinstance(node.value, (int, float))
                or isinstance(node.value, bool)
            ):
                raise ValueError("claim constants must be finite numbers")
        elif isinstance(node, (
            ast.Expression,
            ast.Load,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
        )):
            continue
        else:
            raise ValueError("unsafe node in claim expression")
    return ast.unparse(tree.body), names


def _normalize_claim_schema(
    artifact: dict,
    evidence_type: str,
    *,
    target_lean_signature_hash: str = "",
) -> tuple[dict, str]:
    if set(artifact) != {"claim"} or not isinstance(artifact["claim"], dict):
        raise ValueError("Critic artifact must contain exactly one claim object")
    claim = artifact["claim"]
    if evidence_type in {
        "FINITE_COUNTEREXAMPLE",
        "SYMBOLIC_CONTRADICTION",
    }:
        required = {
            "schema_version",
            "quantifier",
            "variables",
            "domain",
            "lhs",
            "relation",
            "rhs",
        }
        if set(claim) != required:
            raise ValueError("arithmetic claim schema fields are not exact")
        variables_raw = claim["variables"]
        if (
            not isinstance(variables_raw, list)
            or not variables_raw
            or any(
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
                for name in variables_raw
            )
            or len(set(variables_raw)) != len(variables_raw)
        ):
            raise ValueError("quantified variables must be unique identifiers")
        variables = set(variables_raw)
        if (
            claim["schema_version"] != 1
            or claim["quantifier"] != "FOR_ALL"
            or claim["domain"] not in {"INTEGER", "RATIONAL", "REAL"}
            or claim["relation"] not in {"==", "!=", "<", "<=", ">", ">="}
        ):
            raise ValueError("unsupported universal arithmetic claim")
        lhs, lhs_names = _normalize_arithmetic_expression(
            claim["lhs"],
            variables,
        )
        rhs, rhs_names = _normalize_arithmetic_expression(
            claim["rhs"],
            variables,
        )
        if lhs_names | rhs_names != variables:
            raise ValueError("every quantified variable must occur in the claim")
        normalized = {
            "schema_version": 1,
            "quantifier": "FOR_ALL",
            "variables": sorted(variables),
            "domain": claim["domain"],
            "lhs": lhs,
            "relation": claim["relation"],
            "rhs": rhs,
        }
    elif evidence_type == "LEAN_PROOF":
        required = {
            "schema_version",
            "contract",
            "lean_signature_hash",
        }
        if set(claim) != required:
            raise ValueError("Lean claim schema fields are not exact")
        if (
            claim["schema_version"] != 1
            or claim["contract"] != "NEGATION_OF_TARGET_SIGNATURE"
            or not target_lean_signature_hash
            or claim["lean_signature_hash"] != target_lean_signature_hash
        ):
            raise ValueError("Lean claim is not bound to the target signature")
        normalized = {
            "schema_version": 1,
            "contract": "NEGATION_OF_TARGET_SIGNATURE",
            "lean_signature_hash": target_lean_signature_hash,
        }
    else:
        if evidence_type != "PINNED_THEOREM":
            raise ValueError("unsupported evidence type")
        if not isinstance(claim, dict) or not claim:
            raise ValueError("pinned theorem claim must be explicit")
        normalized = json.loads(
            json.dumps(claim, ensure_ascii=False, sort_keys=True),
        )
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return normalized, hashlib.sha256(encoded.encode()).hexdigest()


def extract_premise_suspicions(
    critic_text: str,
    allowed_ids: set[str],
    target_lean_signature_hashes: dict[str, str] | None = None,
) -> dict[str, PremiseSuspicion]:
    target_lean_signature_hashes = target_lean_signature_hashes or {}
    suspicions = {}
    for match in _ISSUE_VERDICT.finditer(critic_text):
        obligation_id = _resolve_model_obligation_id(
            match.group(1),
            allowed_ids,
        )
        if not obligation_id:
            continue
        body = match.group("body")
        invalidation = _structured_field(body, "Invalidation")
        if invalidation not in {"PREMISE_SUSPECTED", "PREMISE"}:
            continue
        premise = _structured_field(body, "Premise refuted")
        evidence_type = _structured_field(body, "Evidence type").upper()
        artifact = _json_artifact(
            _structured_field(body, "Evidence artifact"),
        )
        evidence = _structured_field(body, "Evidence")
        try:
            claim_schema, claim_hash = _normalize_claim_schema(
                artifact,
                evidence_type,
                target_lean_signature_hash=target_lean_signature_hashes.get(
                    obligation_id,
                    "",
                ),
            )
        except ValueError:
            claim_schema, claim_hash = {}, ""
        if (
            not _canonical_claim(premise)
            or evidence_type not in _VERIFIABLE_EVIDENCE_TYPES
            or not artifact
            or not claim_schema
            or not claim_hash
            or len(evidence) < 40
        ):
            continue
        suspicions[obligation_id] = PremiseSuspicion(
            obligation_id,
            premise,
            evidence_type,
            {"claim": claim_schema},
            evidence,
            claim_schema,
            claim_hash,
            target_lean_signature_hashes.get(obligation_id, ""),
        )
    return suspicions


def parse_premise_audit(
    text: str,
    obligation_id: str,
    run_id: str = "",
) -> PremiseAudit | None:
    try:
        scanned = scan_structured_artifact_prefix(text, "PREMISE_AUDIT")
    except ValueError:
        scanned = None
    if scanned is not None and not text[scanned.end:].strip():
        try:
            payload = json.loads(scanned.json_text)
            if set(payload) != {
                "status",
                "evidence_type",
                "evidence_source",
                "confidence",
                "artifact",
                "analysis",
            }:
                return None
            audit = PremiseAudit(
                obligation_id,
                str(payload["status"]),
                str(payload["evidence_type"]).upper(),
                str(payload["evidence_source"]),
                float(payload["confidence"]),
                payload["artifact"],
                str(payload["analysis"]),
                run_id,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if (
            audit.status not in {"CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"}
            or audit.evidence_type not in _VERIFIABLE_EVIDENCE_TYPES
            or not audit.evidence_source
            or not 0.0 <= audit.confidence <= 1.0
            or not audit.analysis
        ):
            return None
        return audit
    for match in _PREMISE_AUDIT.finditer(text):
        if match.group(1) != obligation_id:
            continue
        body = match.group("body")
        status = _structured_field(body, "Status")
        evidence_type = _structured_field(body, "Evidence type").upper()
        evidence_source = _structured_field(body, "Evidence source")
        analysis = _structured_field(body, "Analysis")
        artifact = _json_artifact(_structured_field(body, "Artifact"))
        try:
            confidence = float(_structured_field(body, "Confidence"))
        except ValueError:
            return None
        if (
            status not in {"CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"}
            or evidence_type not in _VERIFIABLE_EVIDENCE_TYPES
            or not evidence_source
            or not 0.0 <= confidence <= 1.0
            or not analysis
        ):
            return None
        return PremiseAudit(
            obligation_id,
            status,
            evidence_type,
            evidence_source,
            confidence,
            artifact,
            analysis,
            run_id,
        )
    return None


def parse_premise_defense(
    text: str,
    obligation_id: str,
    run_id: str = "",
) -> PremiseDefense | None:
    try:
        scanned = scan_structured_artifact_prefix(text, "PREMISE_DEFENSE")
    except ValueError:
        scanned = None
    if scanned is not None and not text[scanned.end:].strip():
        try:
            payload = json.loads(scanned.json_text)
            if set(payload) != {
                "status",
                "correction",
                "failure_reason",
                "evidence",
            }:
                return None
            defense = PremiseDefense(
                obligation_id,
                str(payload["status"]),
                str(payload["correction"]),
                str(payload["failure_reason"]),
                str(payload["evidence"]),
                run_id,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if (
            defense.status not in {"RESCUED", "NOT_RESCUED", "INCONCLUSIVE"}
            or not defense.evidence
            or (defense.status == "RESCUED" and not defense.correction)
            or (
                defense.status == "NOT_RESCUED"
                and not defense.failure_reason
            )
        ):
            return None
        return defense
    for match in _PREMISE_DEFENSE.finditer(text):
        if match.group(1) != obligation_id:
            continue
        body = match.group("body")
        status = _structured_field(body, "Status")
        correction = _structured_field(body, "Correction")
        failure_reason = _structured_field(body, "Failure reason")
        evidence = _structured_field(body, "Evidence")
        if (
            status not in {"RESCUED", "NOT_RESCUED", "INCONCLUSIVE"}
            or not evidence
            or (status == "RESCUED" and not correction)
            or (status == "NOT_RESCUED" and not failure_reason)
        ):
            return None
        return PremiseDefense(
            obligation_id,
            status,
            correction,
            failure_reason,
            evidence,
            run_id,
        )
    return None


def build_premise_auditor_messages(
    goal: str,
    suspicion: PremiseSuspicion,
) -> list[dict[str, str]]:
    package = json.dumps(asdict(suspicion), ensure_ascii=False, sort_keys=True)
    system_content = (
        "You are an isolated Premise Auditor. Independently attack the named "
        "premise using counterexamples, exact definitions and quantifiers, "
        "theorem conflicts, and verifiable Lean, finite, or symbolic evidence. "
        "Do not trust the Critic conclusion. Return exactly "
        "`### PREMISE_AUDIT`, newline, then `Artifact:` and one "
        "compact/minified JSON object with exactly status, evidence_type, "
        "evidence_source, confidence, artifact, and analysis. Status must be "
        "CONFIRMED, NOT_CONFIRMED, or INCONCLUSIVE. For arithmetic, artifact "
        "must contain exactly the host claim_hash, unchanged claim, and a "
        "witness binding every quantified variable. For Lean, preserve exact "
        "claim/signature hashes and negation contract. Emit no Markdown fence, "
        "prose, second Artifact, or trailing token. End immediately after the "
        "matching final }."
    )
    lint_structured_prompt(system_content)
    return [{
        "role": "system",
        "content": system_content,
    }, {
        "role": "user",
        "content": (
            f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
            f"HOST-PACKAGED CRITIC SUSPICION:\n{package}"
        ),
        "_artifact_contract_role": "premise_auditor",
    }]


def build_premise_proponent_messages(
    goal: str,
    suspicion: PremiseSuspicion,
    auditor_text: str,
) -> list[dict[str, str]]:
    package = json.dumps(asdict(suspicion), ensure_ascii=False, sort_keys=True)
    system_content = (
        "You are an isolated Adversarial Proponent. Attempt to rescue the "
        "premise by finding the exact domain, topology, or quantifier "
        "correction, or by refuting the Auditor artifact. Return exactly "
        "`### PREMISE_DEFENSE`, newline, then `Artifact:` and one "
        "compact/minified JSON object with exactly status, correction, "
        "failure_reason, and evidence. Status must be RESCUED, NOT_RESCUED, "
        "or INCONCLUSIVE. Emit no Markdown fence, prose, second Artifact, or "
        "trailing token. End immediately after the matching final }."
    )
    lint_structured_prompt(system_content)
    return [{
        "role": "system",
        "content": system_content,
    }, {
        "role": "user",
        "content": (
            f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
            f"HOST-PACKAGED CRITIC SUSPICION:\n{package}\n\n"
            f"COMPLETE ISOLATED AUDITOR OUTPUT:\n{auditor_text}"
        ),
        "_artifact_contract_role": "premise_proponent",
    }]


def run_isolated_premise_review(
    goal: str,
    suspicion: PremiseSuspicion,
    run_role,
) -> tuple[PremiseAudit | None, PremiseDefense | None, dict]:
    transcripts = {"auditor": "", "proponent": ""}
    auditor_run_id = ""
    try:
        auditor_text, auditor_run_id = run_role(
            "premise_auditor",
            build_premise_auditor_messages(goal, suspicion),
        )
        transcripts["auditor"] = auditor_text
    except Exception as exc:
        transcripts["auditor"] = (
            f"AUDITOR EXECUTION FAILED: {type(exc).__name__}: {exc}"
        )
    audit = parse_premise_audit(
        transcripts["auditor"],
        suspicion.obligation_id,
        auditor_run_id,
    )
    proponent_run_id = ""
    try:
        proponent_text, proponent_run_id = run_role(
            "premise_proponent",
            build_premise_proponent_messages(
                goal,
                suspicion,
                transcripts["auditor"],
            ),
        )
        transcripts["proponent"] = proponent_text
    except Exception as exc:
        transcripts["proponent"] = (
            f"PROPONENT EXECUTION FAILED: {type(exc).__name__}: {exc}"
        )
    defense = parse_premise_defense(
        transcripts["proponent"],
        suspicion.obligation_id,
        proponent_run_id,
    )
    transcripts["auditor_run_id"] = auditor_run_id
    transcripts["proponent_run_id"] = proponent_run_id
    return audit, defense, transcripts


def _safe_arithmetic(expression: str, substitutions: dict) -> float:
    allowed_names = {
        str(key): float(value)
        for key, value in substitutions.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if len(allowed_names) != len(substitutions):
        raise ValueError("substitutions must be finite numbers")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in allowed_names:
            return allowed_names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op,
            (ast.UAdd, ast.USub),
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if abs(right) > 12:
                raise ValueError("exponent is too large")
            return left ** right
        raise ValueError("unsafe arithmetic expression")

    result = evaluate(tree)
    if not (-1e100 < result < 1e100):
        raise ValueError("non-finite arithmetic result")
    return result


def validate_evidence_artifact(
    audit: PremiseAudit,
    *,
    suspicion: PremiseSuspicion,
    project_root: Path,
    lean_validator=validate_lean_proof,
) -> tuple[bool, str]:
    artifact = audit.artifact
    if audit.evidence_type != suspicion.evidence_type:
        return False, "Auditor evidence type differs from the bound Critic claim"
    if audit.evidence_type in {
        "FINITE_COUNTEREXAMPLE",
        "SYMBOLIC_CONTRADICTION",
    }:
        try:
            if set(artifact) != {"claim_hash", "claim", "witness"}:
                raise ValueError("Auditor artifact fields are not exact")
            normalized, claim_hash = _normalize_claim_schema(
                {"claim": artifact["claim"]},
                audit.evidence_type,
            )
            if (
                normalized != suspicion.claim_schema
                or claim_hash != suspicion.claim_hash
                or artifact["claim_hash"] != suspicion.claim_hash
            ):
                raise ValueError("Auditor claim or claim hash was tampered")
            witness = artifact["witness"]
            variables = set(normalized["variables"])
            if not isinstance(witness, dict) or set(witness) != variables:
                raise ValueError("witness must bind every quantified variable")
            if normalized["domain"] == "INTEGER" and any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in witness.values()
            ):
                raise ValueError("integer claim requires integer witnesses")
            left = _safe_arithmetic(normalized["lhs"], witness)
            right = _safe_arithmetic(normalized["rhs"], witness)
            relation = normalized["relation"]
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return False, f"uncheckable arithmetic artifact: {exc}"
        tolerance = 1e-9 * max(1.0, abs(left), abs(right))
        relations = {
            "==": abs(left - right) <= tolerance,
            "!=": abs(left - right) > tolerance,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }
        claimed_relation_holds = relations.get(relation)
        if claimed_relation_holds is None:
            return False, "unsupported claimed relation"
        return (
            not claimed_relation_holds,
            "host found counterexample: claimed relation evaluated "
            f"{left} {relation} {right} as {claimed_relation_holds}",
        )
    if audit.evidence_type == "LEAN_PROOF":
        try:
            if set(artifact) != {
                "claim_hash",
                "claim",
                "lean_signature_hash",
                "contract",
                "source",
            }:
                raise ValueError("Lean Auditor artifact fields are not exact")
            normalized, claim_hash = _normalize_claim_schema(
                {"claim": artifact["claim"]},
                "LEAN_PROOF",
                target_lean_signature_hash=(
                    suspicion.target_lean_signature_hash
                ),
            )
            if (
                normalized != suspicion.claim_schema
                or claim_hash != suspicion.claim_hash
                or artifact["claim_hash"] != suspicion.claim_hash
                or artifact["lean_signature_hash"]
                != suspicion.target_lean_signature_hash
                or artifact["contract"] != "NEGATION_OF_TARGET_SIGNATURE"
            ):
                raise ValueError("Lean proof contract or hash was tampered")
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"uncheckable Lean artifact: {exc}"
        return False, (
            "target signature hash is bound, but the stored Lean signature "
            "cannot be safely transformed into an exact negation wrapper; "
            "the complete proof is recorded but not accepted"
        )
    return False, (
        "pinned theorem references are recorded but no trusted local theorem "
        "registry validates their exact assumptions"
    )


def decide_premise_review(
    audit: PremiseAudit | None,
    defense: PremiseDefense | None,
    *,
    project_root: Path,
    lean_validator=validate_lean_proof,
    suspicion: PremiseSuspicion | None = None,
) -> PremiseReview:
    if audit is None or defense is None:
        return PremiseReview("INCONCLUSIVE", False, reason="missing role output")
    common = {
        "confidence": audit.confidence,
        "evidence_type": audit.evidence_type,
        "evidence_source": audit.evidence_source,
        "auditor_run_id": audit.run_id,
        "proponent_run_id": defense.run_id,
    }
    if audit.status == "NOT_CONFIRMED":
        return PremiseReview(
            "NOT_CONFIRMED",
            False,
            reason=audit.analysis,
            **common,
        )
    if defense.status == "RESCUED":
        return PremiseReview(
            "RESCUED",
            False,
            reason=defense.correction,
            **common,
        )
    if (
        audit.status != "CONFIRMED"
        or defense.status != "NOT_RESCUED"
        or audit.confidence < 0.8
    ):
        return PremiseReview(
            "INCONCLUSIVE",
            False,
            reason="role agreement or confidence threshold not met",
            **common,
        )
    if suspicion is None:
        return PremiseReview(
            "INCONCLUSIVE",
            False,
            reason="host-bound Critic claim schema is missing",
            **common,
        )
    if suspicion is not None and (
        audit.obligation_id != suspicion.obligation_id
        or defense.obligation_id != suspicion.obligation_id
    ):
        return PremiseReview(
            "INCONCLUSIVE",
            False,
            reason="artifact is not bound to the exact suspected premise",
            **common,
        )
    verified, reason = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=project_root,
        lean_validator=lean_validator,
    )
    return PremiseReview(
        "PREMISE_INVALIDATED" if verified else "INCONCLUSIVE",
        verified,
        reason=reason,
        **common,
    )


_CERTIFIED_ARTIFACT_TYPES = {
    "DEFINITION_AUDIT": (DefinitionAudit, "definition_auditor"),
    "COUNTEREXAMPLE_REPORT": (
        CounterexampleReport,
        "counterexample_worker",
    ),
    "DECOMPOSITION_PROPOSAL": (
        DecompositionProposal,
        "decomposer",
    ),
    "FORMALIZATION_BUNDLE": (
        FormalizationBundle,
        "formalizer",
    ),
    "PROOF_ATTEMPT": (ProofAttempt, "prover"),
    "DEFENSE_REPORT": (DefenseReport, "adversarial_proponent"),
    "JUDGE_DECISION": (JudgeDecision, "judge"),
}

def _structured_transport_semantically_complete(text: str, role: str) -> bool:
    """Use typed field transport for v2 roles; retain legacy read-only closure."""
    if role in {"decomposer_scratchpad", "synthesis_scratchpad"}:
        # Private reasoning is never parsed or accepted at an artifact
        # boundary. It completes only at the inference adapter's EOS.
        return False
    if str(text).lstrip().startswith("### "):
        return structured_transport_complete(text, role)
    if role in ROLE_TRANSPORT_REGISTRY:
        return typed_transport_complete(text, role)
    return structured_transport_complete(text, role)


def _validated_structured_prompt(prompt: str) -> str:
    lint_structured_prompt(prompt)
    return prompt


def _canonical_json_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _host_owned_assumption_contract(
    public_assumptions: list[str],
    move,
) -> tuple[list[str], str, dict | None]:
    """Inject immutable assumptions and classify explicit restriction moves."""
    canonical = list(public_assumptions)
    if (
        any(not isinstance(item, str) for item in canonical)
        or canonical != public_assumptions
    ):
        raise ValueError("public assumptions must be an ordered string list")
    assumptions_hash = _canonical_json_hash(canonical)
    restriction_move = None
    if move.move_id in {"RESTRICT_DOMAIN", "CASE_SPLIT"}:
        restriction_move = {
            "move_id": move.move_id,
            "move_version": move.version,
            "provenance": "host_move_registry",
            "provenance_hash": move.content_hash,
            "antecedent_ids": list(move.precondition_ids),
            "typing_fact_ids": [],
            "content_hash": _canonical_json_hash({
                "move_id": move.move_id,
                "move_version": move.version,
                "precondition_ids": move.precondition_ids,
                "move_hash": move.content_hash,
            }),
        }
    return canonical, assumptions_hash, restriction_move


def _scan_decomposer_artifact(text: str):
    return _scan_certified_artifact(text, "DECOMPOSITION_PROPOSAL")


def _scan_certified_artifact(text: str, heading: str):
    match = re.match(
        rf"^\s*### {re.escape(heading)}\s*\n(?P<body>.*)\Z",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing {heading}")
    body = match.group("body")
    return scan_single_artifact_object(
        text if "Artifact:" in body else body,
        marker="Artifact:" if "Artifact:" in body else None,
    )


def _normalized_signature_field(
    value: object,
    *,
    label_required: bool = False,
) -> tuple[dict, object]:
    if not isinstance(value, dict):
        raise ValueError("signature field must be an object")
    required = {"kind", "name", "binders", "proposition", "source"}
    if label_required:
        required.add("label")
    if set(value) != required:
        raise ValueError(
            "signature object fields must be exactly "
            + ", ".join(sorted(required)),
        )
    expected = {
        key: str(value[key])
        for key in ("kind", "name", "binders", "proposition")
    }
    normalized = normalize_lean_signature(
        str(value["source"]),
        expected=expected,
    )
    return value, normalized


def _normalize_formalizer_payload(payload: dict) -> dict:
    """Adapt the structured v3 wire shape to the durable bundle shape."""
    new_fields = {
        "parent_signature",
        "parent_newly_formalized",
        "child_signature",
        "reduction_signature",
    }
    if not new_fields.intersection(payload):
        return payload
    host_fields = {
        "target_obligation_id",
        "parent_statement_hash",
        "root_goal_hash",
        "producer_role",
        "producer_run_id",
        "upstream_artifact_hashes",
    }
    emitted_bindings = {
        key: payload[key]
        for key in host_fields
        if key in payload
    }
    model_fields = {
        key: value
        for key, value in payload.items()
        if key not in host_fields
    }
    if set(model_fields) != new_fields:
        raise ValueError(
            "Formalizer must output exactly parent_signature, "
            "parent_newly_formalized, child_signature, and "
            "reduction_signature",
        )
    _parent_value, parent = _normalized_signature_field(
        model_fields["parent_signature"],
    )
    child_value, child = _normalized_signature_field(
        model_fields["child_signature"],
        label_required=True,
    )
    _reduction_value, reduction = _normalized_signature_field(
        model_fields["reduction_signature"],
    )
    return {**emitted_bindings,
        "parent_signature_source": parent.source,
        "parent_signature_hash": parent.declaration_hash,
        "parent_newly_formalized": model_fields["parent_newly_formalized"],
        "child": {
            "label": child_value["label"],
            "lean_signature": child.source,
            "lean_signature_hash": child.declaration_hash,
        },
        "reduction_theorem_source": reduction.source,
        "reduction_signature_hash": reduction.declaration_hash,
    }


def parse_certified_artifact(
    text: str,
    heading: str,
    *,
    target_obligation_id: str,
    parent_statement_hash: str,
    root_goal_hash: str,
    producer_run_id: str,
    upstream_artifact_hashes: list[str],
):
    artifact_type, producer_role = _CERTIFIED_ARTIFACT_TYPES[heading]
    match = re.search(
        rf"^### {re.escape(heading)}\s*$"
        r"(?P<body>.*?)(?=^### |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return None, f"missing {heading}"
    body = match.group("body")
    try:
        artifact_text = _scan_certified_artifact(text, heading).json_text
    except ValueError as exc:
        return None, f"malformed {heading} Artifact JSON: {exc}"
    if heading == "DEFINITION_AUDIT":
        try:
            payload = json.loads(artifact_text)
        except json.JSONDecodeError as exc:
            return None, (
                f"malformed {heading} Artifact JSON: "
                f"{json_syntax_diagnostic(artifact_text, exc)}"
            )
    else:
        payload = _json_artifact(artifact_text)
    if not isinstance(payload, dict) or not payload:
        return None, f"malformed {heading} Artifact JSON"
    if heading == "FORMALIZATION_BUNDLE":
        # Read-only compatibility for verbose pre-v2 Formalizer artifacts.
        payload.pop("math_ir", None)
        if isinstance(payload.get("child"), dict):
            payload["child"] = {
                key: value
                for key, value in payload["child"].items()
                if key != "statement"
            }
        try:
            payload = _normalize_formalizer_payload(payload)
        except AdapterError as exc:
            checkpoint.adapter_blocked(str(exc), status=exc.status.value)
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return DecompositionCertificateResult(
                False, [str(exc)], artifacts, hashes, transcripts, role_run_ids,
                {"host_gates_passed": False, "failure_status": exc.status.value},
            )
        except ValueError as exc:
            return None, f"invalid {heading} fields: {exc}"
    host_bindings = {
        "target_obligation_id": target_obligation_id,
        "parent_statement_hash": parent_statement_hash,
        "root_goal_hash": root_goal_hash,
        "producer_role": producer_role,
        "producer_run_id": producer_run_id,
        "upstream_artifact_hashes": upstream_artifact_hashes,
    }
    if any(
        key in payload and payload[key] != value
        for key, value in host_bindings.items()
    ):
        return None, f"tampered {heading} bindings"
    payload = {**host_bindings, **payload}
    try:
        artifact = artifact_type(**payload)
    except (TypeError, ValueError) as exc:
        return None, f"invalid {heading} fields: {exc}"
    bindings = (
        artifact.target_obligation_id == target_obligation_id
        and artifact.parent_statement_hash == parent_statement_hash
        and artifact.root_goal_hash == root_goal_hash
        and artifact.producer_role == producer_role
        and artifact.producer_run_id == producer_run_id
        and artifact.upstream_artifact_hashes == upstream_artifact_hashes
    )
    if not bindings:
        return None, f"tampered {heading} bindings"
    if (
        isinstance(artifact, CounterexampleReport)
        and artifact.status not in {
            "COUNTEREXAMPLE_FOUND",
            "NO_COUNTEREXAMPLE",
            "INCONCLUSIVE",
        }
    ) or (
        isinstance(artifact, ProofAttempt)
        and artifact.status not in {"PROVED", "FAILED", "INCONCLUSIVE"}
    ) or (
        isinstance(artifact, DefenseReport)
        and artifact.status not in {
            "DEFENDED",
            "REJECTED",
            "INCONCLUSIVE",
        }
    ) or (
        isinstance(artifact, JudgeDecision)
        and artifact.decision not in {
            "ACCEPT",
            "REJECT",
            "INCONCLUSIVE",
        }
    ):
        return None, f"invalid {heading} status"
    return artifact, ""


def _certified_role_messages(
    role: str,
    heading: str,
    package: dict,
) -> list[dict[str, str]]:
    behavior = {
        "definition_auditor": (
            "Inventory every symbol, domain, quantifier, topology, convergence "
            "notion, and dependency. Missing definitions become precise "
            "definition obligations."
        ),
        "counterexample_worker": (
            "Attack the exact typed claim with finite, limiting, boundary, "
            "and theorem-conflict cases. Unsupported citations are untrusted."
        ),
        "decomposer": (
            "Act as a local mathematical strategist for the exact viewpoint and "
            "decomposition_iteration in the package. Return one child object and "
            "one complete parent-reduction contract. The child must make a "
            "genuinely different structural delta and must not repeat any "
            "semantic or structural signature in decomposition_novelty_ledger. "
            "Never emit children, child lists, dependency edges, plans, or "
            "multiple obligations. If several missing definitions are "
            "inseparable, put every definition into one bundled DEFINITION "
            "child and list every source definition label. The reduction must "
            "state how that exact child and the exact public assumptions imply "
            "the exact parent. The child must have a host-checkable structural "
            "delta, be strictly simpler and reachable, and never restate the "
            "parent. The reduction may not assume or prove the parent circularly."
        ),
        "formalizer": (
            "Emit only exact parent, singular-child, and reduction Lean "
            "signature objects. The host computes their hashes. Never replace a bound parent or "
            "restate the natural-language parent/child."
        ),
        "prover": (
            "Produce one complete Lean proof of the exact reduction theorem. "
            "No sorry, admit, axioms, unsafe commands, or placeholders."
        ),
        "adversarial_proponent": (
            "Attack circularity, missing definitions, disconnected children, "
            "and insufficient reduction; repairs are advisory only."
        ),
        "judge": (
            "Decide ACCEPT, REJECT, or INCONCLUSIVE using only the host-verified "
            "manifest. You cannot override a failed host gate."
        ),
    }[role]
    field_contract = {
        "definition_auditor": (
            "Use exactly definitions and missing_definitions, both arrays of "
            "complete objects."
        ),
        "counterexample_worker": (
            "Use exactly status and cases. Status must be "
            "COUNTEREXAMPLE_FOUND, NO_COUNTEREXAMPLE, or INCONCLUSIVE."
        ),
        "decomposer": (
            "Use exactly parent_statement, child, public_assumptions, and "
            "reduction_contract, following the host package contract. "
            f'Required child field: "kind":'
            f'"{_decomposer_contract(package)["required_child_kind"]}". '
            'Required child field: "source_definition_labels":'
            f'{json.dumps(_decomposer_contract(package)["required_source_definition_labels"], separators=(",", ":"))}.'
        ),
        "formalizer": (
            "Use exactly parent_signature, parent_newly_formalized, "
            "child_signature, and reduction_signature. Each signature is a "
            "separate JSON object with exactly kind, name, binders, "
            "proposition, and source; child_signature additionally has label. "
            "Copy contract names exactly; source must be one theorem or lemma "
            "ending at `:= by`, with no proof body."
        ),
        "prover": (
            "Use exactly status and reduction_theorem_source. Status must be "
            "PROVED, FAILED, or INCONCLUSIVE."
        ),
        "adversarial_proponent": (
            "Use exactly status, issues, and repairs. Status must be DEFENDED, "
            "REJECTED, or INCONCLUSIVE."
        ),
        "judge": (
            "Use exactly decision and reason. Decision must be ACCEPT, REJECT, "
            "or INCONCLUSIVE."
        ),
    }[role]
    if role == "decomposer":
        model_package = _decomposer_model_package(package)
    elif role == "formalizer":
        model_package = _formalizer_model_package(package)
    elif role == "prover":
        model_package = _prover_model_package(package)
    elif role == "adversarial_proponent":
        model_package = _adversarial_review_model_package(package)
    elif role == "judge":
        model_package = _judge_model_package(package)
    else:
        model_package = package
    user_message = {
        "role": "user",
        "content": json.dumps(
            model_package,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "_artifact_contract_role": role,
    }
    if role in {
        "decomposer",
        "formalizer",
        "prover",
        "adversarial_proponent",
        "judge",
    }:
        # Host-only metadata is intentionally outside chat content.
        user_message["_host_package"] = package
    system_content = (
        f"You are the isolated {role}. {behavior} Return exactly `### "
        f"{heading}`, newline, then `Artifact:` and one compact/minified JSON "
        f"object. {field_contract} Host bindings are attached automatically; "
        "if emitted, they must match the package. Emit no Markdown fence, "
        "prose, second Artifact, or trailing token. End immediately after the "
        "matching final }."
    )
    lint_structured_prompt(system_content)
    messages = [{
        "role": "system",
        "content": system_content,
    }, user_message]
    if role in {"formalizer", "prover"}:
        lint_lean_model_prompt(messages)
    return messages


def _formalizer_schema() -> str:
    return (
        '{"parent_signature":{"kind":"theorem","name":"...",'
        '"binders":"...","proposition":"...","source":"..."},'
        '"parent_newly_formalized":true,'
        '"child_signature":{"label":"L1","kind":"theorem","name":"...",'
        '"binders":"...","proposition":"...","source":"..."},'
        '"reduction_signature":{"kind":"theorem","name":"...",'
        '"binders":"...","proposition":"...","source":"..."}}'
    )


def _formalizer_signature_contract(package: dict) -> dict:
    upstream = package["validated_upstream_artifacts"]["decomposer"]
    parent_name = (
        f"parent_{package['parent_statement_hash'][:12]}"
    )
    if package.get("parent_formal_status") != "UNFORMALIZED":
        parent_name = normalize_lean_signature(
            package["parent_lean_signature"],
        ).name
    child_label = str(upstream["child"].get("label", "L1"))
    child_name = f"child_{child_label}_{upstream['child_hash'][:12]}"
    reduction_name = (
        f"reduction_{upstream['reduction_contract_hash'][:12]}"
    )
    return {
        **lean_signature_contract_ref(),
        "names": {
            "parent_signature": parent_name,
            "child_signature": child_name,
            "reduction_signature": reduction_name,
        },
    }


def _formalizer_model_package(package: dict) -> dict:
    compact = {
        "parent_statement_ref": package["parent_statement_hash"],
        "parent_formal_status": package["parent_formal_status"],
        "lean_signature_contract": _formalizer_signature_contract(package),
        "math_ir": {
            unit: _formalizer_math_ir(unit, package, {})
            for unit in ("PARENT_SIGNATURE", "CHILD_SIGNATURE")
        },
    }
    if package["parent_formal_status"] != "UNFORMALIZED":
        compact["parent_lean_signature"] = package["parent_lean_signature"]
        compact["parent_lean_signature_hash"] = package[
            "parent_lean_signature_hash"
        ]
    return compact


def _prover_model_package(package: dict) -> dict:
    formalization = package["validated_upstream_artifacts"]["formalizer"]
    return {
        **lean_proof_contract_ref(),
        "target_statement_ref": package["parent_statement_hash"],
        "reduction_signature": {
            "source": formalization["reduction_theorem_source"],
            "signature_hash": formalization["reduction_signature_hash"],
        },
    }


def _adversarial_review_model_package(package: dict) -> dict:
    """Build one lossless content-addressed proof-review transport view."""
    upstream = package["validated_upstream_artifacts"]
    decomposition = upstream["decomposer"]
    formalization = upstream["formalizer"]
    proof = upstream["prover"]
    artifact_hashes = package["validated_artifact_hashes"]
    host_gates = {
        "validation": package["host_gate_results"]["validation"],
        "errors": package["host_gate_results"]["errors"],
    }
    semantic_units: list[str] = []
    semantic_unit_indexes: dict[str, int] = {}
    hash_table: list[str] = []
    hash_indexes: dict[str, int] = {}

    def retain(value: str) -> int:
        text = str(value)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in semantic_unit_indexes:
            semantic_unit_indexes[digest] = len(semantic_units)
            semantic_units.append(text)
        return semantic_unit_indexes[digest]

    def href(value: str) -> int:
        digest = str(value)
        if digest not in hash_indexes:
            hash_indexes[digest] = len(hash_table)
            hash_table.append(digest)
        return hash_indexes[digest]

    def compact_hashes(value):
        if isinstance(value, dict):
            return {
                key: compact_hashes(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [compact_hashes(item) for item in value]
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value.lower())
        ):
            return {"h": href(value)}
        return value

    child = decomposition["child"]
    formal_child = formalization["child"]
    compact = {
        "binding": {
            "target_id": package["target_obligation_id"],
            "target_h": href(package["target_statement_hash"]),
            "parent_h": href(package["parent_statement_hash"]),
            "root_goal_h": href(package["root_goal_hash"]),
            "formal_status": package["parent_formal_status"],
            "parent_ref": retain(package["parent_statement"]),
        },
        "artifact_h": {
            role: href(artifact_hashes[role])
            for role in ("decomposer", "formalizer", "prover")
        },
        "parent_signature": {
            "hash_h": href(formalization["parent_signature_hash"]),
            "source_ref": retain(formalization["parent_signature_source"]),
            "newly_formalized": formalization["parent_newly_formalized"],
        },
        "child": {
            "hash_h": href(_canonical_json_hash(child)),
            "label": child["label"],
            "kind": child["kind"],
            "source_definition_labels": child["source_definition_labels"],
            "statement_ref": retain(child["statement"]),
            "lean_signature_h": href(formal_child["lean_signature_hash"]),
            "lean_signature_ref": retain(formal_child["lean_signature"]),
        },
        "public_assumptions": {
            "hash_h": href(_canonical_json_hash(
                decomposition["public_assumptions"],
            )),
            "items": decomposition["public_assumptions"],
        },
        "reduction": {
            "hash_h": href(
                _canonical_json_hash(decomposition["reduction_contract"]),
            ),
            "child_label": decomposition["reduction_contract"]["child_label"],
            "derivation_ref": retain(
                decomposition["reduction_contract"]["derivation"],
            ),
            "signature_h": href(formalization["reduction_signature_hash"]),
            "signature_ref": retain(
                formalization["reduction_theorem_source"],
            ),
            "proof_status": proof["status"],
            "proof_ref": retain(proof["reduction_theorem_source"]),
        },
        "host_gates": {
            **compact_hashes(host_gates),
        },
        "semantic_units": semantic_units,
        "hashes": hash_table,
    }
    return compact


def _judge_model_package(package: dict) -> dict:
    """Keep full adjudication evidence without repeated role telemetry."""
    defense = package["defense_evidence"]
    compact = {
        "target": {
            "obligation_id": package["target_obligation_id"],
            "statement_hash": package["parent_statement_hash"],
            "root_goal_hash": package["root_goal_hash"],
        },
        "parent_statement": package["parent_statement"],
        "retained_child_statement": package["retained_child_statement"],
        "artifact_hashes": package["artifact_hashes"],
        "host_gates": {
            "validation": package["validation"],
            "errors": package["errors"],
        },
        "adversarial_review": defense,
        "judge_manifest_hash": package["upstream_artifact_hashes"][0],
    }
    compact["judge_view_hash"] = _canonical_json_hash(compact)
    return compact


def _message_host_package(messages: list[dict]) -> dict:
    return messages[-1].get(
        "_host_package",
        json.loads(messages[-1]["content"]),
    )


def _message_artifact_contract(messages: list[dict], fallback_role: str) -> str:
    return str(messages[-1].get("_artifact_contract_role", fallback_role))


def _required_certified_upstream(role: str) -> set[str]:
    return {
        "definition_auditor": set(),
        "counterexample_worker": {"definition_auditor"},
        # Counterexample prose is advisory. It is never a decomposition premise.
        "decomposer": {"definition_auditor"},
        "formalizer": {"decomposer"},
        "prover": {"formalizer"},
        "adversarial_proponent": {
            "decomposer",
            "formalizer",
            "prover",
        },
    }[role]


def _artifact_dependencies_for_role(role: str, hashes: dict[str, str]) -> list[str]:
    roles = {
        "definition_auditor": (),
        "counterexample_worker": ("definition_auditor",),
        "decomposer": ("definition_auditor",),
        "formalizer": ("decomposer",),
        "prover": ("formalizer",),
        "adversarial_proponent": ("decomposer", "formalizer", "prover"),
    }[role]
    return [hashes[item] for item in roles]


def _certified_upstream_view(artifact, *, consumer_role: str = "") -> dict:
    if isinstance(artifact, DefinitionAudit):
        if consumer_role == "decomposer":
            return {
                "missing_definitions": artifact.missing_definitions,
            }
        return {
            "definitions": artifact.definitions,
            "missing_definitions": artifact.missing_definitions,
        }
    if isinstance(artifact, CounterexampleReport):
        if consumer_role == "decomposer":
            compact_cases = []
            for case in artifact.cases:
                if not isinstance(case, dict):
                    continue
                compact = {
                    key: case[key]
                    for key in (
                        "case_id",
                        "evidence_type",
                        "evidence_source",
                        "claim",
                        "witness",
                        "mathematical_contradiction",
                    )
                    if key in case
                }
                if (
                    "mathematical_contradiction" not in compact
                    and "description" in case
                ):
                    compact["description"] = case["description"]
                compact_cases.append(compact)
            return {
                "status": artifact.status,
                "cases": compact_cases,
            }
        return {
            "status": artifact.status,
            "cases": artifact.cases,
        }
    if isinstance(artifact, DecompositionProposal):
        if consumer_role == "formalizer":
            return {
                "child": artifact.child,
                "public_assumptions": artifact.public_assumptions,
                "reduction": {
                    "child_label": artifact.reduction_contract["child_label"],
                    "derivation": artifact.reduction_contract["derivation"],
                },
            }
        return {
            "parent_statement": artifact.parent_statement,
            "child": artifact.child,
            "public_assumptions": artifact.public_assumptions,
            "reduction_contract": artifact.reduction_contract,
        }
    if isinstance(artifact, FormalizationBundle):
        return {
            "parent_signature_source": artifact.parent_signature_source,
            "parent_signature_hash": artifact.parent_signature_hash,
            "parent_newly_formalized": artifact.parent_newly_formalized,
            "child": artifact.child,
            "reduction_theorem_source": artifact.reduction_theorem_source,
            "reduction_signature_hash": artifact.reduction_signature_hash,
        }
    if isinstance(artifact, ProofAttempt):
        return {
            "status": artifact.status,
            "reduction_theorem_source": artifact.reduction_theorem_source,
        }
    return asdict(artifact)


def _formalizer_upstream_view(
    proposal: DecompositionProposal,
    proposal_hash: str,
) -> dict:
    """Losslessly bind the singular decomposition without repeated prose."""
    child = proposal.child
    assumptions = proposal.public_assumptions
    reduction = proposal.reduction_contract
    return {
        "artifact_hash": proposal_hash,
        "parent_statement_hash": proposal.parent_statement_hash,
        "child_hash": _canonical_json_hash(child),
        "child": child,
        "public_assumptions_hash": _canonical_json_hash(assumptions),
        "public_assumptions": assumptions,
        "reduction_contract_hash": _canonical_json_hash(reduction),
        "reduction": {
            "child_label": reduction["child_label"],
            "derivation": reduction["derivation"],
        },
    }


def _validate_decomposition_shape(
    proposal: DecompositionProposal,
) -> list[str]:
    errors: list[str] = []
    child = proposal.child
    if not isinstance(child, dict) or not str(child.get("label", "")):
        errors.append("child must be one object with a non-empty label")
        return errors
    if not str(child.get("statement", "")).strip():
        errors.append("child statement must be non-empty")
    required_child = {
        "label",
        "statement",
        "kind",
        "source_definition_labels",
    }
    allowed_child = required_child | {
        "move_id",
        "result_status",
        "requires_parent_case_split",
        "proves_parent",
        "theorem_card_ids",
        "public_assumptions",
        "public_assumptions_hash",
        "restriction_move",
    }
    if not required_child.issubset(child) or not set(child).issubset(allowed_child):
        errors.append(
            "child must contain the required typed fields and only registered "
            "Host metadata",
        )
    if child.get("kind") not in {"DEFINITION", "LEMMA"}:
        errors.append("child kind must be DEFINITION or LEMMA")
    if not isinstance(child.get("source_definition_labels"), list):
        errors.append("source_definition_labels must be a list")
    if "proves_parent" in child and child["proves_parent"] is not False:
        errors.append("case child cannot claim to prove the parent")
    if "requires_parent_case_split" in child and not isinstance(
        child["requires_parent_case_split"], bool,
    ):
        errors.append("requires_parent_case_split must be boolean")
    if "theorem_card_ids" in child and not isinstance(
        child["theorem_card_ids"], list,
    ):
        errors.append("theorem_card_ids must be a list")
    contract = proposal.reduction_contract
    required = {
        "child_label",
        "parent_statement",
        "public_assumptions",
        "derivation",
    }
    allowed_contract = required | {
        "public_assumptions_hash",
        "restriction_move",
    }
    if (
        not isinstance(contract, dict)
        or not required.issubset(contract)
        or not set(contract).issubset(allowed_contract)
    ):
        errors.append(
            "reduction_contract must contain Host-owned child, parent, "
            "public-assumption, derivation, and optional restriction metadata",
        )
        return errors
    if contract["child_label"] != child["label"]:
        errors.append("reduction contract does not bind the exact child")
    if contract["parent_statement"] != proposal.parent_statement:
        errors.append("reduction contract does not bind the exact parent")
    if contract["public_assumptions"] != proposal.public_assumptions:
        errors.append("reduction contract changes public assumptions")
    expected_assumptions_hash = _canonical_json_hash(proposal.public_assumptions)
    for owner, container in (("child", child), ("reduction", contract)):
        if (
            "public_assumptions" in container
            and container["public_assumptions"] != proposal.public_assumptions
        ):
            errors.append(f"{owner} changes Host-owned public assumptions")
        if (
            "public_assumptions_hash" in container
            and container["public_assumptions_hash"] != expected_assumptions_hash
        ):
            errors.append(f"{owner} public assumptions hash mismatch")
    if len(str(contract["derivation"]).strip()) < 20:
        errors.append("reduction contract derivation is incomplete")
    derivation = " ".join(str(contract["derivation"]).lower().split())
    if any(marker in derivation for marker in (
        "assume the parent",
        "assuming the parent",
        "assume the density-singularity gap lemma",
        "the parent directly implies itself",
    )):
        errors.append("reduction contract circularly assumes the parent")
    return errors


def _validate_decomposition_progress(
    ledger: ProofObligationLedger,
    parent: ProofObligation,
    proposal: DecompositionProposal,
) -> list[str]:
    """Require a deterministic strict child before it can be persisted."""
    child_statement = str(proposal.child.get("statement", ""))
    reason = _frontier_rejection_reason(
        ledger,
        parent.obligation_id,
        child_statement,
    )
    errors = [f"child {proposal.child.get('label', '')} rejected: {reason}"] if reason else []
    if not _child_has_structural_delta(parent.statement, child_statement):
        errors.append("child lacks a machine-checkable structural delta")
    if _normalize_obligation_statement(child_statement) == (
        _normalize_obligation_statement(parent.statement)
    ):
        errors.append("child restates the exact parent")
    return errors


def _validate_definition_child_selection(
    definition_audit: DefinitionAudit,
    proposal: DecompositionProposal,
) -> list[str]:
    required_labels = [
        str(item.get("obligation_label", ""))
        for item in definition_audit.missing_definitions
        if isinstance(item, dict)
    ]
    if "" in required_labels:
        return ["missing definitions must have non-empty obligation labels"]
    if not required_labels:
        return []
    child = proposal.child
    if not isinstance(child, dict) or child.get("kind") != "DEFINITION":
        return ["missing definitions require a DEFINITION bundle child"]
    raw_source_labels = child.get("source_definition_labels")
    if not isinstance(raw_source_labels, list):
        return ["source_definition_labels must be a list"]
    source_labels = [str(label) for label in raw_source_labels]
    if (
        any(not label for label in source_labels)
        or len(source_labels) != len(set(source_labels))
        or set(source_labels) != set(required_labels)
    ):
        return [
            "definition bundle must reference every missing definition label; "
            "source_definition_labels must equal exactly "
            f"{sorted(set(required_labels))}",
        ]
    return []


def _decomposer_payload_result(text: str) -> tuple[dict, str]:
    try:
        scanned = _scan_decomposer_artifact(text)
    except ValueError as exc:
        return {}, f"malformed DECOMPOSITION_PROPOSAL Artifact JSON: {exc}"
    try:
        payload = json.loads(scanned.json_text)
    except json.JSONDecodeError:
        repaired = repair_json_backslashes(scanned.json_text)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as exc:
            return {}, (
                "malformed DECOMPOSITION_PROPOSAL Artifact JSON: "
                f"{exc.msg} at line {exc.lineno} column {exc.colno}"
            )
    if not isinstance(payload, dict):
        return {}, "DECOMPOSITION_PROPOSAL Artifact must be one JSON object"
    return payload, ""


def _decomposer_payload(text: str) -> dict:
    return _decomposer_payload_result(text)[0]


def _decomposer_protocol_errors(text: str) -> list[str]:
    payload, diagnostic = _decomposer_payload_result(text)
    if diagnostic:
        return [diagnostic]
    errors = []
    if "children" in payload:
        children = payload.get("children")
        count = len(children) if isinstance(children, list) else "non-list"
        errors.append(
            f"field 'children' is forbidden (received {count}); emit exactly "
            "one 'child' object",
        )
    if not isinstance(payload.get("child"), dict):
        errors.append("field 'child' must be exactly one object")
    if "dependency_edges" in payload:
        errors.append("field 'dependency_edges' is forbidden for one child")
    if "reduction_labels" in payload:
        errors.append("field 'reduction_labels' is forbidden; bind child_label")
    contract = payload.get("reduction_contract")
    if not isinstance(contract, dict):
        errors.append("field 'reduction_contract' must be one complete object")
    allowed = {
        "parent_statement",
        "child",
        "public_assumptions",
        "reduction_contract",
        # Read-only compatibility: older producers emitted host bindings.
        "target_obligation_id",
        "parent_statement_hash",
        "root_goal_hash",
        "producer_role",
        "producer_run_id",
        "upstream_artifact_hashes",
    }
    extras = sorted(set(payload) - allowed)
    if extras:
        errors.append(f"unexpected decomposer fields: {', '.join(extras)}")
    return errors


def _decomposer_semantically_complete(
    text: str,
    package: dict,
    producer_run_id: str,
) -> bool:
    """Accept an early stop only after complete schema and host validation."""
    try:
        _scan_decomposer_artifact(text)
    except ValueError:
        return False
    if _decomposer_protocol_errors(text):
        return False
    parsed, error = parse_certified_artifact(
        text,
        "DECOMPOSITION_PROPOSAL",
        target_obligation_id=package["target_obligation_id"],
        parent_statement_hash=package["parent_statement_hash"],
        root_goal_hash=package["root_goal_hash"],
        producer_run_id=producer_run_id,
        upstream_artifact_hashes=package["upstream_artifact_hashes"],
    )
    return bool(
        not error
        and parsed is not None
        and parsed.parent_statement == package["parent_statement"]
        and not _validate_decomposition_shape(parsed)
        and _decomposer_child_matches_contract(parsed.child, package)
    )


def _validate_formalization_shape(
    bundle: FormalizationBundle,
    package: dict,
) -> list[str]:
    errors: list[str] = []
    child = bundle.child
    expected_child = package["validated_upstream_artifacts"]["decomposer"][
        "child"
    ]
    if not isinstance(child, dict) or set(child) != {
        "label",
        "lean_signature",
        "lean_signature_hash",
    }:
        errors.append(
            "child must contain exactly label, lean_signature, and "
            "lean_signature_hash",
        )
        return errors
    if child.get("label") != expected_child.get("label"):
        errors.append("formalized child label differs from exact child")
    for field_name in (
        "parent_signature_source",
        "parent_signature_hash",
        "reduction_theorem_source",
        "reduction_signature_hash",
    ):
        if not str(getattr(bundle, field_name, "")).strip():
            errors.append(f"{field_name} must be non-empty")
    if not str(child.get("lean_signature", "")).strip():
        errors.append("child lean_signature must be non-empty")
    if not str(child.get("lean_signature_hash", "")).strip():
        errors.append("child lean_signature_hash must be non-empty")
    try:
        contract = _formalizer_signature_contract(package)
        actual_names = {
            "parent_signature": normalize_lean_signature(
                bundle.parent_signature_source,
            ).name,
            "child_signature": normalize_lean_signature(
                str(child.get("lean_signature", "")),
            ).name,
            "reduction_signature": normalize_lean_signature(
                bundle.reduction_theorem_source,
            ).name,
        }
        for field, required_name in contract["names"].items():
            if actual_names[field] != required_name:
                errors.append(
                    f"{field} name changed: expected immutable "
                    f"{required_name}, got {actual_names[field]}",
                )
    except ValueError as exc:
        errors.append(f"signature contract failed: {exc}")
    return errors


def _validate_formalization_elaboration(
    bundle: FormalizationBundle,
    package: dict,
    *,
    project_root: Path,
    signature_validator,
) -> list[str]:
    """Elaborate all signatures before a Prover may see the bundle."""
    sources = {
        "parent": bundle.parent_signature_source,
        "child": str(bundle.child.get("lean_signature", "")),
        "reduction": bundle.reduction_theorem_source,
    }
    for label, source in sources.items():
        if re.search(r"\.\.\.|:=\s*\.\.\.|<[^>]*hash[^>]*>", source, re.I):
            return [f"{label} Lean signature contains a placeholder"]
    results = {
        label: signature_validator(source, project_root=project_root)
        for label, source in sources.items()
    }
    errors = []
    for label, result in results.items():
        if result.ok:
            continue
        failure_kind = (
            "contract/syntax validation failed"
            if result.status == "CONTRACT_FAILED"
            else (
                "Lean typecheck failed"
                if result.status == "TYPECHECK_FAILED"
                else f"Lean {result.status.lower()}"
            )
        )
        errors.append(f"{label} {failure_kind}: {result.error}")
    expected_hashes = {
        "parent": bundle.parent_signature_hash,
        "child": str(bundle.child.get("lean_signature_hash", "")),
        "reduction": bundle.reduction_signature_hash,
    }
    for label, result in results.items():
        if result.ok and result.signature_hash != expected_hashes[label]:
            errors.append(
                f"{label} proposition/hash mismatch: normalized declaration "
                "hash differs from the recorded hash",
            )
    if not errors:
        parent_signature = " ".join(
            _signature_only_for_comparison(sources["parent"]).split()
        )
        reduction_signature = " ".join(
            _signature_only_for_comparison(sources["reduction"]).split()
        )
        parent_conclusion = (
            parent_signature.rsplit(" : ", 1)[-1]
            if " : " in parent_signature else ""
        )
        reduction_conclusion = (
            reduction_signature.rsplit(" : ", 1)[-1]
            if " : " in reduction_signature else ""
        )
        if not parent_conclusion or reduction_conclusion != parent_conclusion:
            errors.append(
                "reduction theorem conclusion differs from exact parent proposition",
            )
    return errors


def _signature_only_for_comparison(source: str) -> str:
    return re.split(r"\s*:=\s*by\b", str(source), maxsplit=1)[0].strip()


def _circular_reduction_proof(source: str) -> bool:
    signature = _signature_only_for_comparison(source)
    conclusion_match = re.search(r"\)\s*:\s*(.+)$", signature)
    if conclusion_match is None:
        return False
    conclusion = " ".join(conclusion_match.group(1).split())
    assumptions = re.findall(r"\(\s*\w+\s*:\s*([^()]+)\)", signature)
    return any(" ".join(item.split()) == conclusion for item in assumptions)


def _formalizer_semantically_complete(
    text: str,
    package: dict,
    producer_run_id: str,
) -> bool:
    """Stop on one closed, strict, schema-valid Formalizer artifact."""
    parsed, error = parse_certified_artifact(
        text,
        "FORMALIZATION_BUNDLE",
        target_obligation_id=package["target_obligation_id"],
        parent_statement_hash=package["parent_statement_hash"],
        root_goal_hash=package["root_goal_hash"],
        producer_run_id=producer_run_id,
        upstream_artifact_hashes=package["upstream_artifact_hashes"],
    )
    return bool(
        not error
        and parsed is not None
        and not _validate_formalization_shape(parsed, package)
    )


def _formalizer_repair_messages(
    package: dict,
    *,
    validation_errors: list[str],
) -> list[dict[str, str]]:
    compact = _formalizer_model_package(package)
    compact["validation_errors"] = [
        re.sub(
            r"\\([A-Za-z]+|[{}])",
            lambda match: f"LATEX_COMMAND_{match.group(1)}",
            str(error),
        )
        for error in validation_errors
    ]
    user_message = {
        "role": "user",
        "content": json.dumps(compact, ensure_ascii=False, sort_keys=True),
        "_host_package": package,
    }
    return [{
        "role": "system",
        "content": _validated_structured_prompt((
            "Fresh Formalizer repair; never splice prior JSON. Return exactly "
            "`### FORMALIZATION_BUNDLE`, newline, `Artifact:`, and one compact "
            "JSON object with exactly parent_signature, "
            "parent_newly_formalized, child_signature, reduction_signature. "
            "Each signature is a separate object with exactly kind, name, "
            "binders, proposition, source; child_signature also has label. "
            "Copy lean_signature_contract ID, version, and names. Source must "
            "match its fields and end at `:= by` with no body. Examples show "
            "syntax only; supply complete package mathematics, never "
            "placeholders. No prose, fences, host bindings, second artifact, or "
            "trailing token; end at the final }."
        )),
    }, user_message]


FORMALIZER_UNIT_SPECS = (
    ("PARENT_SIGNATURE", "formalizer_parent_signature"),
    ("CHILD_SIGNATURE", "formalizer_child_signature"),
    ("REDUCTION_SIGNATURE", "formalizer_reduction_signature"),
)
FORMALIZER_UNIT_HEADROOM_TOKENS = 128


def _formalizer_unit_dependencies(
    unit: str,
    package: dict,
    unit_hashes: dict[str, str],
) -> list[str]:
    upstream = package["validated_upstream_artifacts"]["decomposer"]
    if unit == "PARENT_SIGNATURE":
        return [
            upstream["artifact_hash"],
            package["parent_statement_hash"],
            package["root_goal_hash"],
        ]
    if unit == "CHILD_SIGNATURE":
        return [upstream["artifact_hash"], upstream["child_hash"]]
    return [
        upstream["artifact_hash"],
        unit_hashes["PARENT_SIGNATURE"],
        unit_hashes["CHILD_SIGNATURE"],
        upstream["reduction_contract_hash"],
        upstream["public_assumptions_hash"],
    ]


def _formalizer_symbol_table(package: dict):
    audit = package.get("definition_audit", {})
    table = register_lean_symbol_table(
        list(audit.get("definitions", [])),
        missing_definitions=list(audit.get("missing_definitions", [])),
        parent_statement_hash=package["parent_statement_hash"],
    )
    return table


def _safe_reduction_description(package: dict, table) -> str:
    derivation = str(
        package["validated_upstream_artifacts"]["decomposer"][
            "reduction"
        ].get("derivation", ""),
    )
    safe = normalize_registered_latex_identifiers(derivation, table)
    safe = safe.replace("$", "").replace("{", "").replace("}", "")
    if "\\" in safe:
        raise ValueError("reduction description retains a backslash")
    return " ".join(safe.split())


def _formalizer_math_ir(
    unit: str,
    package: dict,
    validated_units: dict[str, dict],
) -> dict:
    table = _formalizer_symbol_table(package)
    upstream = package["validated_upstream_artifacts"]["decomposer"]
    if unit == "PARENT_SIGNATURE":
        description = (
            "For fixed epsilon and genus p, there exists a critical density "
            "rho_c such that any complex sequence z with density rho greater "
            "than rho_c cannot have its reciprocal series converge locally to "
            "a pole of integer coefficient m at s0 within radius delta unless "
            "the analytic function f has growth order greater than p."
        )
        source_ref = package["parent_statement_hash"]
    elif unit == "CHILD_SIGNATURE":
        missing_by_label = {
            str(item.get("obligation_label", "")): str(
                item.get("required_type", ""),
            )
            for item in package["definition_audit"].get(
                "missing_definitions",
                [],
            )
        }
        labels = upstream["child"].get("source_definition_labels", [])
        description = (
            "Bundled definition obligation: "
            + "; ".join(
                missing_by_label[str(label)]
                for label in labels
                if str(label) in missing_by_label
            )
        )
        source_ref = upstream["child_hash"]
    else:
        description = _safe_reduction_description(package, table)
        source_ref = upstream["reduction_contract_hash"]
    safe_symbols = [
        {"name": symbol.name, "type": symbol.lean_type}
        for symbol in table.symbols
    ]
    semantic_payload = {
        "unit": unit,
        "description": description,
        "symbols": safe_symbols,
        "source_ref": source_ref,
    }
    if unit == "REDUCTION_SIGNATURE":
        semantic_payload["parent_signature"] = {
            "source": validated_units["PARENT_SIGNATURE"]["source"],
            "signature_hash": validated_units["PARENT_SIGNATURE"][
                "signature_hash"
            ],
        }
        semantic_payload["child_signature"] = {
            "source": validated_units["CHILD_SIGNATURE"]["source"],
            "signature_hash": validated_units["CHILD_SIGNATURE"][
                "signature_hash"
            ],
        }
        semantic_payload["public_assumptions"] = upstream["public_assumptions"]
    encoded = json.dumps(
        semantic_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if "\\" in encoded:
        raise ValueError("Lean-safe Math IR contains a raw backslash")
    return {
        **semantic_payload,
        "symbol_table_id": table.symbol_table_id,
        "symbol_table_version": table.version,
        "proposition_semantic_hash": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _formalizer_unit_messages(
    unit: str,
    package: dict,
    validated_units: dict[str, dict],
    *,
    validation_errors: list[str] | None = None,
) -> list[dict[str, str]]:
    contract = _formalizer_signature_contract(package)
    name_key = {
        "PARENT_SIGNATURE": "parent_signature",
        "CHILD_SIGNATURE": "child_signature",
        "REDUCTION_SIGNATURE": "reduction_signature",
    }[unit]
    math_ir = _formalizer_math_ir(unit, package, validated_units)
    model_package = {
        **lean_signature_contract_ref(),
        "unit": unit,
        "required_name": contract["names"][name_key],
        "math_ir": math_ir,
    }
    if validation_errors:
        model_package["validation_errors"] = [
            re.sub(
                r"\\([A-Za-z]+|[{}])",
                lambda match: f"LATEX_COMMAND_{match.group(1)}",
                str(error),
            )
            for error in validation_errors
        ]
    system_content = (
        "Formalize exactly one Lean signature unit. Return exactly "
        "`### LEAN_SIGNATURE_UNIT`, newline, `Artifact:`, and one compact JSON "
        "object with exactly contract_id, contract_version, unit, kind, name, "
        "binders, proposition, source. Copy contract/unit/name; binders and "
        "proposition are strings; binders is exact parenthesized Lean binder "
        "text copied verbatim in source; kind is theorem or lemma; source ends "
        "`:= by`. "
        "Emit no prose, fence, proof body, second artifact, or trailing token."
    )
    lint_structured_prompt(system_content)
    messages = [{
        "role": "system",
        "content": system_content,
    }, {
        "role": "user",
        "content": json.dumps(
            model_package,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "_artifact_contract_role": {
            "PARENT_SIGNATURE": "formalizer_parent_signature",
            "CHILD_SIGNATURE": "formalizer_child_signature",
            "REDUCTION_SIGNATURE": "formalizer_reduction_signature",
        }[unit],
        "_host_package": package,
    }]
    lint_lean_model_prompt(messages)
    return messages


def _parse_formalizer_unit(
    text: str,
    *,
    unit: str,
    package: dict,
    project_root: Path,
    signature_validator,
    dependencies: list[str],
) -> tuple[dict | None, str]:
    try:
        scanned = _scan_certified_artifact(text, "LEAN_SIGNATURE_UNIT")
        payload = _json_artifact(scanned.json_text)
    except ValueError as exc:
        return None, f"malformed LEAN_SIGNATURE_UNIT: {exc}"
    required = {
        "contract_id",
        "contract_version",
        "unit",
        "kind",
        "name",
        "binders",
        "proposition",
        "source",
    }
    if set(payload) != required:
        return None, "Lean signature unit fields are not exact"
    try:
        resolve_lean_contract(
            str(payload["contract_id"]),
            int(payload["contract_version"]),
            signature_only=True,
        )
        contract = _formalizer_signature_contract(package)
        name_key = {
            "PARENT_SIGNATURE": "parent_signature",
            "CHILD_SIGNATURE": "child_signature",
            "REDUCTION_SIGNATURE": "reduction_signature",
        }[unit]
        if payload["unit"] != unit:
            raise ValueError("Lean signature unit name mismatch")
        if payload["name"] != contract["names"][name_key]:
            raise ValueError("immutable Lean declaration name mismatch")
        if not isinstance(payload["binders"], str) or not isinstance(
            payload["proposition"],
            str,
        ):
            raise ValueError("binders and proposition must be strings")
        table = _formalizer_symbol_table(package)
        binders_before_hash = lean_symbol_semantic_hash(
            payload["binders"],
            table,
        )
        proposition_before_hash = lean_symbol_semantic_hash(
            payload["proposition"],
            table,
        )
        normalized_binders = normalize_registered_latex_identifiers(
            payload["binders"],
            table,
        )
        normalized_proposition = normalize_registered_latex_identifiers(
            payload["proposition"],
            table,
        )
        normalized_source_text = normalize_registered_latex_identifiers(
            str(payload["source"]),
            table,
        )
        if (
            binders_before_hash
            != lean_symbol_semantic_hash(normalized_binders, table)
            or proposition_before_hash
            != lean_symbol_semantic_hash(normalized_proposition, table)
        ):
            raise ValueError("symbol normalization changed binder/proposition hash")
        normalized = normalize_lean_signature(
            normalized_source_text,
            expected={
                "kind": str(payload["kind"]),
                "name": str(payload["name"]),
                "binders": normalized_binders,
                "proposition": normalized_proposition,
            },
        )
        allowed_binders = {symbol.name for symbol in table.symbols}
        declared_binders = set(re.findall(
            r"[\(\{]\s*([A-Za-z_][A-Za-z0-9_']*)\s*:",
            normalized.binders,
        ))
        unregistered = declared_binders - allowed_binders
        if unregistered:
            raise ValueError(
                "unregistered Lean binder identifier(s): "
                + ", ".join(sorted(unregistered)),
            )
    except (TypeError, ValueError) as exc:
        return None, f"Lean contract validation failed: {exc}"
    elaborated = signature_validator(
        normalized.source,
        project_root=project_root,
    )
    if not elaborated.ok:
        return None, (
            f"Lean {elaborated.status.lower()} at {unit}: {elaborated.error}"
        )
    if elaborated.signature_hash != normalized.declaration_hash:
        return None, f"proposition/hash mismatch at {unit}"
    return {
        "schema_version": 1,
        "unit": unit,
        "contract_id": payload["contract_id"],
        "contract_version": int(payload["contract_version"]),
        "kind": normalized.kind,
        "name": normalized.name,
        "binders": normalized.binders,
        "proposition": normalized.proposition,
        "proposition_hash": normalized.proposition_hash,
        "source": normalized.source,
        "signature_hash": normalized.declaration_hash,
        "symbol_table_id": table.symbol_table_id,
        "symbol_table_version": table.version,
        "binder_semantic_hash": binders_before_hash,
        "proposition_semantic_hash": proposition_before_hash,
        "dependencies": dependencies,
    }, ""


def _load_formalizer_unit(
    checkpoint: OrchestrationCheckpoint,
    *,
    unit: str,
    dependencies: list[str],
    symbol_table,
) -> dict | None:
    role = f"formalizer_{unit.lower()}"
    ref = checkpoint.validated_artifacts.get(role)
    if ref is None:
        return None
    if ref.dependencies != dependencies or ref.schema_version != 1:
        return None
    encoded = Path(ref.path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != ref.sha256:
        return None
    payload = json.loads(encoded)
    if (
        not isinstance(payload, dict)
        or payload.get("unit") != unit
        or payload.get("dependencies") != dependencies
        or payload.get("symbol_table_id") != symbol_table.symbol_table_id
        or payload.get("symbol_table_version") != symbol_table.version
    ):
        return None
    resolve_lean_contract(
        str(payload.get("contract_id", "")),
        int(payload.get("contract_version", 0)),
        signature_only=True,
    )
    return payload


def _assemble_formalizer_units(
    units: dict[str, dict],
    *,
    package: dict,
    producer_run_id: str,
) -> FormalizationBundle:
    parent = units["PARENT_SIGNATURE"]
    child = units["CHILD_SIGNATURE"]
    reduction = units["REDUCTION_SIGNATURE"]
    parent_core = _signature_only_for_comparison(parent["source"])
    reduction_core = _signature_only_for_comparison(reduction["source"])
    parent_conclusion = parent_core.rsplit(" : ", 1)[-1]
    reduction_conclusion = reduction_core.rsplit(" : ", 1)[-1]
    if not parent_conclusion or reduction_conclusion != parent_conclusion:
        raise ValueError(
            "reduction theorem conclusion differs from exact parent proposition",
        )
    return FormalizationBundle(
        target_obligation_id=package["target_obligation_id"],
        parent_statement_hash=package["parent_statement_hash"],
        root_goal_hash=package["root_goal_hash"],
        producer_role="formalizer",
        producer_run_id=producer_run_id,
        upstream_artifact_hashes=package["upstream_artifact_hashes"],
        parent_signature_source=parent["source"],
        parent_signature_hash=parent["signature_hash"],
        parent_newly_formalized=(
            package["parent_formal_status"] == "UNFORMALIZED"
        ),
        child={
            "label": package["validated_upstream_artifacts"]["decomposer"][
                "child"
            ]["label"],
            "lean_signature": child["source"],
            "lean_signature_hash": child["signature_hash"],
        },
        reduction_theorem_source=reduction["source"],
        reduction_signature_hash=reduction["signature_hash"],
    )


def _run_split_formalizer(
    run_role,
    *,
    package: dict,
    project_root: Path,
    signature_validator,
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    expected_run_id: str,
) -> tuple[FormalizationBundle | None, dict[str, str], str]:
    contract_ref = lean_signature_contract_ref()
    checkpoint.lean_contract_id = str(contract_ref["contract_id"])
    checkpoint.lean_contract_version = int(contract_ref["version"])
    symbol_table = _formalizer_symbol_table(package)
    checkpoint.lean_symbol_table_id = symbol_table.symbol_table_id
    checkpoint.lean_symbol_table_version = symbol_table.version
    units: dict[str, dict] = {}
    unit_hashes = dict(checkpoint.formalizer_unit_hashes)
    transcripts: dict[str, str] = {}
    last_run_id = expected_run_id
    for unit, role_name in FORMALIZER_UNIT_SPECS:
        dependencies = _formalizer_unit_dependencies(
            unit,
            package,
            unit_hashes,
        )
        loaded = _load_formalizer_unit(
            checkpoint,
            unit=unit,
            dependencies=dependencies,
            symbol_table=symbol_table,
        )
        if loaded is not None:
            units[unit] = loaded
            unit_hashes[unit] = checkpoint.validated_artifacts[
                f"formalizer_{unit.lower()}"
            ].sha256
            checkpoint.formalizer_unit_hashes = dict(unit_hashes)
            continue
        checkpoint.formalizer_substate = unit
        checkpoint.current_role = "formalizer"
        checkpoint.resume_origin = unit
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        messages = _formalizer_unit_messages(unit, package, units)
        unit_error = ""
        for attempt in range(2):
            attempt_run_id = (
                f"{expected_run_id}:{unit.lower()}"
                + (":repair-1" if attempt else "")
            )
            try:
                text, actual_run_id = run_role(
                    role_name,
                    messages,
                    attempt_run_id,
                )
            except Exception as exc:
                unit_error = f"{type(exc).__name__}: {exc}"
                text = str(getattr(exc, "partial_text", ""))
                if text:
                    transcripts[
                        f"{unit.lower()}_partial_attempt_{attempt + 1}"
                    ] = text
            else:
                transcripts[
                    unit.lower() if not attempt
                    else f"{unit.lower()}_repair_1"
                ] = text
                if actual_run_id != attempt_run_id:
                    unit_error = "Formalizer unit run ID mismatch"
                else:
                    parsed, unit_error = _parse_formalizer_unit(
                        text,
                        unit=unit,
                        package=package,
                        project_root=project_root,
                        signature_validator=signature_validator,
                        dependencies=dependencies,
                    )
                    if parsed is not None:
                        ref = persist_validated_artifact(
                            checkpoint_path,
                            checkpoint,
                            role=f"formalizer_{unit.lower()}",
                            payload=parsed,
                            dependencies=dependencies,
                            source_run_id=actual_run_id,
                        )
                        units[unit] = parsed
                        unit_hashes[unit] = ref.sha256
                        checkpoint.formalizer_unit_hashes = dict(unit_hashes)
                        save_orchestration_checkpoint(
                            checkpoint_path,
                            checkpoint,
                        )
                        last_run_id = actual_run_id
                        break
            if attempt == 0:
                messages = _formalizer_unit_messages(
                    unit,
                    package,
                    units,
                    validation_errors=[unit_error],
                )
        if unit not in units:
            failure = f"{unit}: {unit_error}"
            checkpoint.adapter_blocked(failure, status="ADAPTER_BLOCKED")
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return None, transcripts, failure
    checkpoint.formalizer_substate = "ASSEMBLE"
    checkpoint.formalizer_unit_hashes = dict(unit_hashes)
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    try:
        bundle = _assemble_formalizer_units(
            units,
            package=package,
            producer_run_id=f"{last_run_id}:assemble",
        )
    except ValueError as exc:
        failure = f"ASSEMBLE: {exc}"
        checkpoint.blocked_reason = f"INTEGRATION_BLOCKED:{failure}"
        checkpoint.transition(ProofState.BLOCKED, checkpoint.blocked_reason)
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return None, transcripts, failure
    return bundle, transcripts, ""


def _validate_defense_shape(defense: DefenseReport) -> list[str]:
    errors = []
    if defense.status not in {"DEFENDED", "REJECTED", "INCONCLUSIVE"}:
        errors.append("status must be DEFENDED, REJECTED, or INCONCLUSIVE")
    for field_name in ("issues", "repairs"):
        value = getattr(defense, field_name)
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            errors.append(f"{field_name} must be a list of non-empty strings")
    return errors


def _defense_semantically_complete(
    text: str,
    package: dict,
    producer_run_id: str,
) -> bool:
    """Stop only on one closed, strict, host-bound defense artifact."""
    parsed, error = parse_certified_artifact(
        text,
        "DEFENSE_REPORT",
        target_obligation_id=package["target_obligation_id"],
        parent_statement_hash=package["parent_statement_hash"],
        root_goal_hash=package["root_goal_hash"],
        producer_run_id=producer_run_id,
        upstream_artifact_hashes=package["upstream_artifact_hashes"],
    )
    return bool(
        not error
        and parsed is not None
        and not _validate_defense_shape(parsed)
    )


def _defense_repair_messages(
    package: dict,
    *,
    validation_errors: list[str],
) -> list[dict[str, str]]:
    compact = _adversarial_review_model_package(package)
    compact["validation_errors"] = list(validation_errors)
    return [{
        "role": "system",
        "content": _validated_structured_prompt((
            "Fresh Adversarial Proponent protocol repair; do not continue or "
            "splice prior JSON. Return exactly `### DEFENSE_REPORT`, then "
            "one-line `Artifact:` and one compact/minified JSON object with "
            "exactly status, issues, and repairs. Status must be one of "
            "DEFENDED, REJECTED, or INCONCLUSIVE; issues and repairs must be "
            "arrays of complete strings. Emit no prose, second artifact, "
            "trailing text, host bindings, examples, or placeholders. Do not "
            "invent or host-fill mathematics. End immediately after the "
            "matching final }."
        )),
    }, {
        "role": "user",
        "content": json.dumps(compact, ensure_ascii=False, sort_keys=True),
        "_host_package": package,
    }]


def _decomposer_contract(package: dict) -> dict:
    definition_audit = package.get(
        "validated_upstream_artifacts",
        {},
    ).get("definition_auditor", {})
    missing = definition_audit.get("missing_definitions", [])
    required_labels = [
        str(item.get("obligation_label", ""))
        for item in missing
        if isinstance(item, dict)
    ]
    return {
        "required_parent_hash": package["parent_statement_hash"],
        "required_child_kind": (
            "DEFINITION" if required_labels else "LEMMA"
        ),
        "required_source_definition_labels": required_labels,
    }


def _decomposer_model_package(package: dict) -> dict:
    """Remove hash/prose duplication while preserving semantic search state."""
    contract = _decomposer_contract(package)
    novelty = dict(package.get("decomposition_novelty_ledger", {}))
    tried_viewpoints = list(novelty.pop("viewpoints", []))
    active_viewpoint = str(
        novelty.pop("active_viewpoint", package.get("viewpoint", "")),
    )
    return {
        "target_obligation_id": package["target_obligation_id"],
        "parent_statement": package["parent_statement"],
        "viewpoint": active_viewpoint,
        "immutable_bindings": {
            "parent_sha256": package["parent_statement_hash"],
            "root_goal_sha256": package["root_goal_hash"],
            "definition_audit_sha256": package[
                "upstream_artifact_hashes"
            ][0],
            "producer_run_id": package["producer_run_id"],
            "ancestor_hashes": package.get("ancestor_hashes", []),
        },
        "required_definitions": package.get(
            "validated_upstream_artifacts",
            {},
        ).get("definition_auditor", {}).get("missing_definitions", []),
        "reduction_constraints": {
            "single_child": True,
            "strictly_simpler": True,
            "non_circular_child_to_exact_parent": True,
            "required_child_kind": contract["required_child_kind"],
            "required_source_definition_labels": contract[
                "required_source_definition_labels"
            ],
        },
        "semantic_search": {
            "iteration": package.get("decomposition_iteration", 1),
            "tried_viewpoint_ids": tried_viewpoints,
            "novelty": novelty,
        },
    }


def _decomposition_ancestor_hashes(
    ledger: ProofObligationLedger,
    parent: ProofObligation,
) -> dict:
    """Bind the exact ancestor chain without repeating ancestor statements."""
    by_id = {item.obligation_id: item for item in ledger.obligations}
    records = []
    cursor = parent.parent_id
    seen = set()
    while cursor and cursor not in seen:
        seen.add(cursor)
        ancestor = by_id.get(cursor)
        if ancestor is None:
            break
        records.append({
            "obligation_id_sha256": hashlib.sha256(
                ancestor.obligation_id.encode(),
            ).hexdigest(),
            "statement_sha256": hashlib.sha256(
                ancestor.statement.encode(),
            ).hexdigest(),
            "alpha_signature_sha256": hashlib.sha256(
                _canonical_claim(ancestor.statement).encode(),
            ).hexdigest(),
        })
        cursor = ancestor.parent_id
    manifest = _canonical_json_hash(records)
    return {
        "manifest": f"sha256:{manifest}",
        "count": len(records),
        "archive": "proof_ledger",
    }


DECOMPOSER_VIEWPOINTS = (
    "definitions",
    "domain_topology",
    "quantifiers",
    "local_global_bridge",
    "constructive_witness",
    "reduction_direction",
    "boundary_cases",
)


def _decomposition_semantic_hash(proposal: DecompositionProposal) -> str:
    """Hash mathematical content after deterministic alpha normalization."""
    child = proposal.child
    reduction = proposal.reduction_contract
    return _canonical_json_hash({
        "child_kind": child.get("kind", ""),
        "child_statement": _canonical_claim(child.get("statement", "")),
        "source_definition_labels": sorted(
            str(item) for item in child.get("source_definition_labels", [])
        ),
        "public_assumptions": [
            _canonical_claim(item) for item in proposal.public_assumptions
        ],
        "derivation": _canonical_claim(reduction.get("derivation", "")),
    })


def _decomposition_structural_signature(
    proposal: DecompositionProposal,
) -> str:
    child = proposal.child
    premise, conclusion = _claim_structure(child.get("statement", ""))
    return _canonical_json_hash({
        "kind": child.get("kind", ""),
        "premise_terms": sorted(premise),
        "conclusion_terms": sorted(conclusion),
        "concepts": sorted(_semantic_concepts(child.get("statement", ""))),
        "definition_labels": sorted(
            str(item) for item in child.get("source_definition_labels", [])
        ),
    })


def _is_decomposition_semantic_rejection(errors: list[str]) -> bool:
    text = " ".join(errors).lower()
    return any(marker in text for marker in (
        "child l1 rejected",
        "strictly simpler",
        "structural delta",
        "disconnected child",
        "reduction theorem conclusion differs",
        "reduction proof failed",
        "reduction proof targets another",
        "reduction contract circular",
        "adversarial defense found a blocking defect",
    ))


def _select_decomposer_viewpoint(
    checkpoint: OrchestrationCheckpoint,
    definition_audit: DefinitionAudit,
) -> str:
    """Select the next unresolved mathematical perspective deterministically."""
    ordered = list(DECOMPOSER_VIEWPOINTS)
    if not definition_audit.missing_definitions:
        ordered.remove("definitions")
        ordered.append("definitions")
    latest_reasons = " ".join(
        checkpoint.semantic_rejection.get("rejection_reasons", []),
    ).lower()
    priorities = []
    for markers, viewpoint in (
        (("topology", "domain", "neighborhood"), "domain_topology"),
        (("quantifier", "forall", "exists"), "quantifiers"),
        (("local", "global"), "local_global_bridge"),
        (("witness", "construct", "explicit"), "constructive_witness"),
        (("reduction", "circular", "entails"), "reduction_direction"),
        (("boundary", "counterexample"), "boundary_cases"),
    ):
        if any(marker in latest_reasons for marker in markers):
            priorities.append(viewpoint)
    ordered = priorities + [item for item in ordered if item not in priorities]
    for viewpoint in ordered:
        if viewpoint not in checkpoint.viewpoints_tried:
            return viewpoint
    ledger = compact_decomposition_novelty_ledger(checkpoint)
    digest = _canonical_json_hash({
        "history": ledger["manifest"],
        "reasons": ledger["reasons"],
    })[:12]
    return f"synthesized_host_failures_{digest}"


def _decomposer_schema(package: dict) -> str:
    contract = _decomposer_contract(package)
    child_kind = contract["required_child_kind"]
    statement_description = (
        "<model-authored bundled definition obligation>"
        if child_kind == "DEFINITION"
        else "<model-authored lemma obligation>"
    )
    schema = {
        "parent_statement": "<copy exact host parent_statement>",
        "child": {
            "label": "L1",
            "statement": statement_description,
            "kind": child_kind,
            "source_definition_labels": contract[
                "required_source_definition_labels"
            ],
        },
        "public_assumptions": [],
        "reduction_contract": {
            "child_label": "L1",
            "parent_statement": "<copy exact host parent_statement>",
            "public_assumptions": [],
            "derivation": "<complete child-to-parent argument>",
        },
    }
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


def _decomposer_child_matches_contract(child: dict, package: dict) -> bool:
    contract = _decomposer_contract(package)
    labels = child.get("source_definition_labels")
    return bool(
        child.get("kind") == contract["required_child_kind"]
        and isinstance(labels, list)
        and labels == contract["required_source_definition_labels"]
    )


def _decomposer_repair_messages(
    package: dict,
    *,
    validation_errors: list[str],
    rejected_artifact: dict | None,
) -> list[dict[str, str]]:
    full_contract = _decomposer_contract(package)
    repair_contract = {
        "required_child_kind": full_contract["required_child_kind"],
        "required_source_definition_labels": full_contract[
            "required_source_definition_labels"
        ],
    }
    repair_package = {
        "target_obligation_id": package["target_obligation_id"],
        "parent_statement": package["parent_statement"],
        "parent_statement_hash": package["parent_statement_hash"],
        "root_goal_hash": package["root_goal_hash"],
        "producer_role": "decomposer",
        "producer_run_id": package["producer_run_id"],
        "upstream_artifact_hashes": package["upstream_artifact_hashes"],
        "validated_upstream_artifacts": package[
            "validated_upstream_artifacts"
        ],
        "validation_errors": validation_errors,
        "repair_contract": repair_contract,
    }
    if rejected_artifact:
        rejected_children = rejected_artifact.get("children")
        if not isinstance(rejected_children, list):
            rejected_child = rejected_artifact.get("child")
            rejected_children = (
                [rejected_child] if isinstance(rejected_child, dict) else []
            )
        repair_package["rejected_obligations"] = [
            {
                key: child[key]
                for key in (
                    "label",
                    "statement",
                    "kind",
                    "source_definition_labels",
                )
                if key in child
            }
            for child in rejected_children
            if isinstance(child, dict)
        ]
        repair_package["rejected_public_assumptions"] = rejected_artifact.get(
            "public_assumptions",
            [],
        )
        rejected_contract = rejected_artifact.get("reduction_contract")
        if isinstance(rejected_contract, dict):
            repair_package["rejected_reduction_derivation"] = (
                rejected_contract.get("derivation", "")
            )
    return [{
        "role": "system",
        "content": _validated_structured_prompt((
            "Protocol repair. Return a fresh complete response; never continue "
            "or splice prior JSON. Output exactly `### DECOMPOSITION_PROPOSAL` "
            "then one-line `Artifact:` and one compact/minified JSON object. "
            "Use exactly the fields parent_statement, child, "
            "public_assumptions, and reduction_contract. Child must contain "
            "exactly label, statement, kind, and source_definition_labels; "
            f'its required concrete kind field is "kind":'
            f'"{full_contract["required_child_kind"]}". '
            'Its required concrete labels field is '
            f'"source_definition_labels":'
            f'{json.dumps(full_contract["required_source_definition_labels"], separators=(",", ":"))}. '
            "reduction_contract must contain exactly child_label, "
            "parent_statement, public_assumptions, and derivation. The host "
            "parent_statement and "
            "parent_statement_hash in the repair package are immutable; copy "
            "the exact parent_statement into both parent locations without "
            "correction or paraphrase. The child statement and derivation are "
            "model-authored; the host constrains only the audited metadata. "
            "Produce a strictly simpler, reachable child with a concrete "
            "structural delta; do not restate the parent. The reduction_contract "
            "must derive the exact parent from the exact child and public "
            "assumptions without assuming the parent or using it circularly. "
            "Preserve every substantive obligation from a rejected complete "
            "artifact inside the one bundled child; never select or discard one. "
            "Copy required_child_kind and required_source_definition_labels "
            "exactly from repair_contract. Emit no examples, placeholders, "
            "second Artifact, or trailing prose. End immediately after the "
            "matching final }."
        )),
    }, {
        "role": "user",
        "content": json.dumps(
            repair_package,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }]


def _validate_counterexample_report(
    report: CounterexampleReport,
    *,
    project_root: Path,
) -> list[dict]:
    validations = []
    for case in report.cases:
        if not isinstance(case, dict):
            validations.append({"verified": False, "error": "malformed case"})
            continue
        evidence_type = str(case.get("evidence_type", ""))
        if evidence_type in {
            "FINITE_COUNTEREXAMPLE",
            "SYMBOLIC_CONTRADICTION",
        }:
            try:
                claim, claim_hash = _normalize_claim_schema(
                    {"claim": case["claim"]},
                    evidence_type,
                )
                suspicion = PremiseSuspicion(
                    report.target_obligation_id,
                    "Host-bound counterexample case",
                    evidence_type,
                    {"claim": claim},
                    "Counterexample worker artifact",
                    claim,
                    claim_hash,
                )
                audit = PremiseAudit(
                    report.target_obligation_id,
                    "CONFIRMED",
                    evidence_type,
                    str(case.get("evidence_source", "")),
                    1.0,
                    {
                        "claim_hash": claim_hash,
                        "claim": claim,
                        "witness": case["witness"],
                    },
                    "Deterministic counterexample validation.",
                )
                verified, error = validate_evidence_artifact(
                    audit,
                    suspicion=suspicion,
                    project_root=project_root,
                )
            except (KeyError, TypeError, ValueError) as exc:
                verified, error = False, str(exc)
        elif evidence_type == "PINNED_THEOREM":
            verified, error = (
                False,
                "unsupported theorem citation has no trusted local registry",
            )
        else:
            verified, error = False, "unsupported evidence type"
        validations.append({
            "evidence_type": evidence_type,
            "verified": verified,
            "error": error,
        })
    return validations


def _validate_decomposition_certificate(
    ledger: ProofObligationLedger,
    parent: ProofObligation,
    definition_audit: DefinitionAudit,
    counterexamples: CounterexampleReport,
    proposal: DecompositionProposal,
    formalization: FormalizationBundle,
    proof: ProofAttempt,
    defense: DefenseReport | None,
    *,
    project_root: Path,
    signature_validator=validate_lean_signature,
    proof_validator=validate_lean_proof,
) -> tuple[dict, list[str]]:
    errors = _validate_decomposition_shape(proposal)
    validation = {
        "graph_valid": not errors,
        "parent_signature_valid": False,
        "children_valid": False,
        "reduction_signature_valid": False,
        "reduction_proof_valid": False,
        "defense_nonblocking": (
            None if defense is None else defense.status in {
                "DEFENDED",
                "INCONCLUSIVE",
            }
        ),
    }
    validation["definition_inventory_nonempty"] = bool(
        definition_audit.definitions
        or definition_audit.missing_definitions
    )
    if not validation["definition_inventory_nonempty"]:
        errors.append("definition inventory is empty")
    validation["counterexample_cases"] = _validate_counterexample_report(
        counterexamples,
        project_root=project_root,
    )
    verified_counterexample = (
        counterexamples.status == "COUNTEREXAMPLE_FOUND"
        and any(
            item["verified"]
            for item in validation["counterexample_cases"]
        )
    )
    validation["verified_parent_counterexample"] = verified_counterexample
    validation["counterexample_advisory_only"] = not verified_counterexample
    validation["counterexample_is_public_premise"] = False
    validation["counterexample_is_certificate_gate"] = verified_counterexample
    if (
        counterexamples.status == "COUNTEREXAMPLE_FOUND"
        and not verified_counterexample
    ):
        validation["counterexample_advisory_reason"] = (
            "claimed counterexample has no verified evidence"
        )
    elif verified_counterexample:
        errors.append(
            "verified counterexample refutes the parent; decomposition is "
            "forbidden and premise review is required",
        )
    proposal_label = str(proposal.child.get("label", ""))
    errors.extend(
        _validate_definition_child_selection(
            definition_audit,
            proposal,
        ),
    )
    errors.extend(_validate_decomposition_progress(ledger, parent, proposal))
    formal_child = formalization.child
    parent_signature_text = " ".join(
        formalization.parent_signature_source.split(),
    ).split(" := by", 1)[0]
    reduction_signature_text = " ".join(
        formalization.reduction_theorem_source.split(),
    ).split(" := by", 1)[0]
    parent_conclusion = (
        parent_signature_text.rsplit(" : ", 1)[-1]
        if " : " in parent_signature_text else ""
    )
    reduction_conclusion = (
        reduction_signature_text.rsplit(" : ", 1)[-1]
        if " : " in reduction_signature_text else ""
    )
    if str(formal_child.get("label", "")) != proposal_label:
        errors.append("formalized child label differs from proposal")
    if not parent_conclusion or reduction_conclusion != parent_conclusion:
        errors.append(
            "reduction theorem conclusion differs from exact parent proposition",
        )
    if formalization.parent_signature_hash != parent.lean_signature_hash and (
        parent.formal_status != "UNFORMALIZED"
    ):
        errors.append("existing parent signature hash mismatch")
    parent_result = signature_validator(
        formalization.parent_signature_source,
        project_root=project_root,
    )
    if not parent_result.ok:
        errors.append(f"parent signature failed: {parent_result.error}")
    elif parent.formal_status == "UNFORMALIZED":
        if not formalization.parent_newly_formalized:
            errors.append("new parent signature was not declared")
        elif parent_result.signature_hash != formalization.parent_signature_hash:
            errors.append("new parent signature hash mismatch")
        else:
            validation["parent_signature_valid"] = True
    elif (
        formalization.parent_newly_formalized
        or parent_result.signature_hash != parent.lean_signature_hash
        or formalization.parent_signature_source != parent.lean_signature
    ):
        errors.append("existing parent signature cannot be replaced")
    else:
        validation["parent_signature_valid"] = True
    source = str(formal_child.get("lean_signature", ""))
    child_result = signature_validator(source, project_root=project_root)
    if not child_result.ok:
        errors.append(
            f"child {proposal_label} signature failed: {child_result.error}",
        )
    elif formal_child.get("lean_signature_hash") != child_result.signature_hash:
        errors.append(f"child {proposal_label} signature hash mismatch")
    validation["children_valid"] = child_result.ok
    reduction_signature = signature_validator(
        formalization.reduction_theorem_source,
        project_root=project_root,
    )
    if (
        not reduction_signature.ok
        or reduction_signature.signature_hash
        != formalization.reduction_signature_hash
    ):
        errors.append("reduction theorem signature failed or changed")
    else:
        validation["reduction_signature_valid"] = True
    proof_result = proof_validator(
        proof.reduction_theorem_source,
        project_root=project_root,
    )
    if (
        proof.status != "PROVED"
        or not proof_result.ok
        or proof_result.status != "PROVED"
        or lean_theorem_signature_hash(proof.reduction_theorem_source)
        != formalization.reduction_signature_hash
    ):
        errors.append("complete reduction proof failed or targets another theorem")
    else:
        validation["reduction_proof_valid"] = True
    if defense is not None and defense.status == "REJECTED":
        errors.append("adversarial defense found a blocking defect")
    validation["child_signature_hashes"] = {
        proposal_label: child_result.signature_hash,
    }
    validation["reduction_proof_hash"] = (
        proof_result.signature_hash if proof_result.ok else ""
    )
    validation["host_gates_passed"] = not errors
    return validation, errors


def _typed_package_text(package: dict, *, role: str) -> str:
    """Render host-owned references and a notation-free semantic hint."""
    sections = [
        ("PARENT CLAIM REF", package.get("parent_claim_ref", "")),
        ("TARGET ID", package.get("target_obligation_id", "")),
        ("TARGET STATEMENT", package.get("target_statement", "")),
        ("SELECTED STRATEGY PLAN ID", package.get("strategy_plan_id", "")),
        ("VERIFIED TARGET EVIDENCE", package.get("verified_target_evidence", "")),
        ("CURRENT TARGET GAPS", package.get("current_target_gaps", "")),
        ("PLAIN SEMANTIC SUMMARY", package.get("plain_semantic_summary", "")),
        ("VIEWPOINT", package.get("viewpoint", "")),
    ]
    if role == "definition_auditor":
        registry = package["definition_registry"]
        for heading, key in (
            ("REGISTERED SYMBOL IDS", "symbols"),
            ("REGISTERED DOMAIN IDS", "domains"),
            ("REGISTERED TOPOLOGY IDS", "topologies"),
            ("REGISTERED DEFINITION IDS", "definitions"),
        ):
            sections.append((
                heading,
                "\n".join(
                    f"{item_id} {item['content_ref']} {item['label']}"
                    if key == "definitions"
                    else f"{item_id} {content_ref}"
                    for item_id, item in registry[key].items()
                    for content_ref in (
                        (item["content_ref"] if key == "definitions" else item),
                    )
                ),
            ))
    if role == "decomposer":
        missing = package.get("validated_upstream_artifacts", {}).get(
            "definition_auditor", {},
        ).get("missing_definitions", [])
        required_kind = "DEFINITION" if missing else "LEMMA"
        sections.append(("REQUIRED CHILD KIND", required_kind))
        sections.append((
            "REGISTERED SOURCE DEFINITION IDS",
            " ".join(
                str(item.get("obligation_label", ""))
                for item in missing if isinstance(item, dict)
            ) or "none",
        ))
        sections.append((
            "REGISTERED HOST MOVE CHOICES",
            "\n".join(
                f"{item['choice_code']} {item['summary_id']} "
                f"{item['candidate_hash']}"
                for item in package.get("candidate_choices", ())
            ) or "none",
        ))
    return "\n\n".join(
        f"{heading}\n{value}" for heading, value in sections if str(value).strip()
    )


def _typed_role_messages(role: str, package: dict) -> list[dict[str, str]]:
    behavior = {
        "definition_auditor": (
            "Select only Host-registered symbol, domain, topology, and definition "
            "IDs for the exact target reference. Mark unresolved registered "
            "definitions with missing_definition_id and MISSING_DEFINITION. If "
            "the registry cannot express the audit, return REFRAME_REQUIRED. "
            "Never return mathematical notation, free symbol/domain text, JSON, "
            "source, or explanations. The Host owns labels and artifact fields."
        ),
        "decomposer": (
            "Select exactly one Host-registered move code. The Host owns every "
            "typed IR operation, operand binding, scope, type, and payload. "
            "Return the exact parent claim reference and REQUIRED CHILD KIND. "
            "Never return DSL, expressions, claims, propositions, explanations, "
            "notation, or source text. This runtime has no decode-time token "
            "allowlist, so the one-character move code is validated fail-closed."
        ),
        "synthesis": (
            "Select exactly one Host-scoped single-character choice code. The "
            "Host maps that code to an immutable candidate ID and hash, then "
            "orders the remaining alternatives deterministically. Return only "
            "the choice code and one registered reason code. Counterexample "
            "evidence marked advisory cannot be a premise. Do not return prose, "
            "mathematics, JSON, Lean, DSL, or candidate IDs."
        ),
        "proof_action_selector": (
            "Select one Host-enumerated action ID for the current elaborated "
            "goal ID, plus only its registered operand and substitution IDs. "
            "Never emit Lean, JSON, DSL, prose, or an entire proof."
        ),
        "adversarial_proponent": (
            "Review the host-elaborated proposition and proof evidence."
        ),
        "judge": "Decide only from host gate and review evidence.",
    }[role]
    user = {
        "role": "user",
        "content": _typed_package_text(package, role=role),
        "_artifact_contract_role": role,
        "_host_package": package,
    }
    registered_choices = package.get("registered_output_choices", {})
    return [{
        "role": "system",
        "content": (
            f"{behavior}\n\n"
            f"{transport_prompt(role, registered_choices=registered_choices)}"
        ),
    }, user]


def _run_typed_definition_auditor(
    parent: ProofObligation,
    root_goal: str,
    run_role,
    *,
    orchestration_id: str,
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
) -> tuple[DefinitionAudit | None, str, str, str]:
    """Run the production Definition Auditor without model-authored JSON."""
    statement_hash = hashlib.sha256(parent.statement.encode()).hexdigest()
    goal_hash = hashlib.sha256(root_goal.encode()).hexdigest()
    target_ref = f"claim:{statement_hash}"
    registry = build_definition_choice_registry(target_ref, parent.statement)
    expected_run_id = f"{orchestration_id}:definition_auditor:typed-v1"
    package = {
        "target_obligation_id": parent.obligation_id,
        "parent_statement_hash": statement_hash,
        "root_goal_hash": goal_hash,
        "parent_claim_ref": target_ref,
        "producer_role": "definition_auditor",
        "producer_run_id": expected_run_id,
        "upstream_artifact_hashes": [],
        "registered_output_choices": registry.registered_output_choices,
        "definition_registry": {
            "symbols": dict(registry.symbols),
            "domains": dict(registry.domains),
            "topologies": dict(registry.topologies),
            "definitions": {
                key: {
                    "content_ref": value.content_ref,
                    "label": value.label,
                }
                for key, value in registry.definitions.items()
            },
            "registry_hash": registry.registry_hash,
        },
    }
    text, actual_run_id = run_role(
        "definition_auditor",
        _typed_role_messages("definition_auditor", package),
        expected_run_id,
    )
    if actual_run_id != expected_run_id:
        raise AdapterError(
            "RUN_ID_MISMATCH",
            "adapter returned another run ID",
            role="definition_auditor",
        )
    semantic_reframe = False
    try:
        decoded = decode_role_fields(
            text,
            "definition_auditor",
            registered_choices=registry.registered_output_choices,
        )
    except AdapterError as exc:
        if exc.code != "INVALID_CHOICE":
            raise
        semantic_reframe = True
        values = {
            "target_ref": target_ref,
            "symbol_id": (),
            "domain_id": (),
            "topology_id": (),
            "definition_id": (),
            "missing_definition_id": (),
            "audit_outcome": "REFRAME_REQUIRED",
        }
        canonical = json.dumps(
            {
                "role": "definition_auditor",
                "values": values,
                "semantic_route": "unknown_registered_choice",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        decoded = DecodedRoleFields(
            "definition_auditor",
            values,
            hashlib.sha256(canonical).hexdigest(),
        )
    payload, envelope = serialize_definition_audit(
        decoded,
        registry,
        target_obligation_id=parent.obligation_id,
        parent_statement_hash=statement_hash,
        root_goal_hash=goal_hash,
        producer_run_id=expected_run_id,
    )
    artifact = DefinitionAudit(**payload)
    ref = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload=payload,
        dependencies=[],
        source_run_id=expected_run_id,
    )
    route_definition_audit_outcome(
        checkpoint,
        outcome=str(payload["audit_outcome"]),
        artifact_hash=ref.sha256,
        source_run_id=expected_run_id,
        missing_definition_ids=(
            str(item["definition_id"])
            for item in payload["missing_definitions"]
        ),
        counterexample_objective=checkpoint.counterexample_objective,
    )
    checkpoint.recovery_events.append({
        "event_type": (
            "DEFINITION_REGISTRY_REFRAME"
            if semantic_reframe else "DEFINITION_AUDIT_HOST_SERIALIZED"
        ),
        "event_id": envelope["content_hash"],
        "target_state": checkpoint.state,
        "transport_hash": decoded.transport_hash,
        "registry_hash": registry.registry_hash,
        "artifact_hash": ref.sha256,
        "created_at": time.time(),
    })
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    return artifact, ref.sha256, text, expected_run_id


def _start_decomposition_exploration(
    checkpoint: OrchestrationCheckpoint,
    *,
    checkpoint_path: Path,
    run_role,
    orchestration_id: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Generate private candidates and persist only Host-owned references."""
    plan = build_decomposition_exploration_plan(
        target_ref=checkpoint.target_obligation_id,
        parent_obligation_ref=checkpoint.target_obligation_id,
        parent_complexity=12,
        environment_hash=checkpoint.target_environment_hash,
        registered_definition_ids=(),
        theorem_card_ids=checkpoint.theorem_card_ids,
        dependency_ids=artifact_hashes.values(),
        evidence_refs=artifact_hashes.values(),
        known_no_go_refs=checkpoint.forbidden_semantic_fingerprints,
        candidate_budget=9,
    )
    contract = gate_decomposition_exploration(
        plan,
        target_obligation_id=checkpoint.target_obligation_id,
        target_context_hash=checkpoint.target_context_hash,
        proposition_hash=checkpoint.proposition_hash,
        evidence_refs=artifact_hashes.values(),
        no_go_refs=checkpoint.forbidden_semantic_fingerprints,
        theorem_card_ids=checkpoint.theorem_card_ids,
        candidate_budget=9,
    )
    checkpoint.transition(
        ProofState.DECOMPOSITION_EXPLORATION,
        "no-registered-move:private-candidate-exploration",
        strategy_reused=True,
    )

    def run_candidate(short_id: str, prompt: str) -> str:
        run_id = f"{orchestration_id}:exploration:{short_id}:v1"
        text, actual_run_id = run_role(
            "decomposer_scratchpad",
            [
                {
                    "role": "system",
                    "content": (
                        "Write one private untrusted mathematical memo in prose "
                        "or LaTeX. Never emit JSON, Lean, a DSL, hidden "
                        "assumptions, secrets, or an authoritative artifact."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            run_id,
        )
        if actual_run_id != run_id:
            raise RuntimeError("exploration candidate run ID mismatch")
        return text

    candidates = generate_private_candidate_refs(
        contract,
        memo_dir=checkpoint_path.with_suffix(".exploration_memos"),
        run_candidate=run_candidate,
    )
    checkpoint.transition(
        ProofState.CANDIDATE_PREFILTER,
        "private-candidates-generated",
        strategy_reused=True,
    )
    filtered = prefilter_private_candidates(
        candidates,
        target_obligation_id=checkpoint.target_obligation_id,
        no_go_refs=contract.no_go_refs,
    )
    ranking = rank_private_candidates(filtered.survivors, top_k=3)
    checkpoint.exploration_contract_id = contract.contract_id
    checkpoint.exploration_contract_hash = contract.content_hash
    checkpoint.exploration_candidate_refs = [
        {
            key: value for key, value in asdict(candidate).items()
            if key != "memo_path"
        }
        for candidate in candidates
    ]
    checkpoint.exploration_rejections = {
        candidate_id: list(reasons)
        for candidate_id, reasons in filtered.rejected
    }
    checkpoint.candidate_set_hash = filtered.candidate_set_hash
    checkpoint.candidate_count = len(candidates)
    checkpoint.candidate_hashes = [
        candidate.semantic_fingerprint for candidate in candidates
    ]
    checkpoint.ranking_hash = ranking.ranking_hash
    checkpoint.ranked_candidate_ids = list(ranking.ranked_candidate_ids)
    checkpoint.exploration_selected_candidate_ids = list(
        ranking.selected_candidate_ids
    )
    checkpoint.exploration_current_index = 0
    checkpoint.exploration_current_candidate_id = (
        ranking.selected_candidate_ids[0]
        if ranking.selected_candidate_ids else ""
    )
    checkpoint.exploration_formalization_status = (
        "PENDING" if ranking.selected_candidate_ids else "EXHAUSTED"
    )
    checkpoint.recovery_events.append({
        "event_type": "DECOMPOSITION_EXPLORATION_CANDIDATES_READY",
        "event_id": filtered.candidate_set_hash,
        "contract_id": contract.contract_id,
        "candidate_count": len(candidates),
        "survivor_count": len(filtered.survivors),
        "selected_candidate_ids": list(ranking.selected_candidate_ids),
        "rejected_reason_codes": sorted({
            reason for _, reasons in filtered.rejected for reason in reasons
        }),
        "private_memo_text_exposed": False,
        "ledger_mutated": False,
        "created_at": time.time(),
    })
    if ranking.selected_candidate_ids:
        checkpoint.transition(
            ProofState.CANDIDATE_FORMALIZATION,
            "top-k-private-candidates-selected",
            strategy_reused=True,
        )
    else:
        checkpoint.transition(
            ProofState.DECOMPOSER,
            "decomposition-exploration-empty-backjump",
            strategy_reused=True,
        )
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    return {
        "exploration_contract_id": contract.contract_id,
        "candidate_set_hash": filtered.candidate_set_hash,
        "candidate_count": len(candidates),
        "survivor_count": len(filtered.survivors),
        "selected_candidate_ids": list(ranking.selected_candidate_ids),
    }


def _run_typed_ir_v2(
    ledger: ProofObligationLedger,
    parent: ProofObligation,
    root_goal: str,
    run_role,
    *,
    project_root: Path,
    orchestration_id: str,
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    signature_validator,
    proof_validator,
    artifacts: dict,
    hashes: dict,
    role_run_ids: dict,
) -> DecompositionCertificateResult:
    """Execute the v2 path with adapter and deterministic gates off-budget."""
    statement_hash = hashlib.sha256(parent.statement.encode()).hexdigest()
    goal_hash = hashlib.sha256(root_goal.encode()).hexdigest()
    transcripts: dict[str, str] = {}
    errors: list[str] = []
    if "definition_auditor" not in artifacts:
        checkpoint.blocked_reason = "typed IR migration requires definition evidence"
        checkpoint.transition(ProofState.BLOCKED, checkpoint.blocked_reason)
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, [checkpoint.blocked_reason], artifacts, hashes,
            transcripts, role_run_ids, {
                "host_gates_passed": False,
                "failure_status": GateStatus.SEMANTIC_BACKJUMP.value,
            },
        )
    if checkpoint.representation_exhaustion_hash:
        return DecompositionCertificateResult(
            False,
            ["CANDIDATE_REPRESENTATION_EXHAUSTED"],
            artifacts,
            hashes,
            transcripts,
            role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": "REPRESENTATION_STAGNATION",
                "representation_exhaustion_hash": (
                    checkpoint.representation_exhaustion_hash
                ),
                "scratchpad_rerun": False,
                "strategy_rerun": False,
            },
        )
    if checkpoint.exploration_exhaustion_hash:
        return DecompositionCertificateResult(
            False,
            ["DECOMPOSITION_EXPLORATION_EXHAUSTED"],
            artifacts,
            hashes,
            transcripts,
            role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": "MATHEMATICAL_STAGNATION",
                "exploration_exhaustion_hash": (
                    checkpoint.exploration_exhaustion_hash
                ),
                "scratchpad_rerun": False,
                "strategy_rerun": False,
            },
        )
    if checkpoint.proof_state == ProofState.CANDIDATE_REPRESENTATION_ANALYSIS:
        index = checkpoint.exploration_current_index
        formalization_queue = checkpoint.ranked_candidate_ids
        candidate_id, next_candidate_id, queue_exhausted = (
            next_formalization_candidate(
                formalization_queue,
                current_index=index,
            )
        )
        candidate = next(
            item for item in checkpoint.exploration_candidate_refs
            if item.get("candidate_id") == candidate_id
        )
        memo_hash = str(candidate.get("memo_sha256", ""))
        memo_path = checkpoint_path.with_suffix(
            ".exploration_memos",
        ) / f"{memo_hash}.private"
        memo_bytes = memo_path.read_bytes()
        if hashlib.sha256(memo_bytes).hexdigest() != memo_hash:
            raise RuntimeError("PRIVATE_CANDIDATE_MEMO_HASH_MISMATCH")
        duplicate_ids = tuple(
            str(item.get("candidate_id", ""))
            for item in checkpoint.exploration_candidate_refs
            if (
                item.get("candidate_id") != candidate_id
                and item.get("semantic_fingerprint")
                == candidate.get("semantic_fingerprint")
            )
        )
        report = analyze_private_candidate_representation(
            candidate_id=candidate_id,
            candidate_hash=memo_hash,
            category=str(candidate.get("category", "")),
            memo_text=memo_bytes.decode("utf-8"),
            target_obligation_id=checkpoint.target_obligation_id,
            target_context_hash=checkpoint.target_context_hash,
            environment_hash=checkpoint.target_environment_hash,
            duplicate_candidate_ids=duplicate_ids,
        )
        report_path = persist_representation_gap_report(
            report,
            checkpoint_path.with_suffix(".representation_reports"),
        )
        checkpoint.representation_report_refs[candidate_id] = {
            "report_id": report.report_id,
            "sha256": report.content_hash,
            "path": str(report_path),
            "mapper_registry_hash": report.mapper_registry_hash,
            "environment_hash": report.environment_hash,
            "audit_only": True,
        }
        checkpoint.representation_current_status = report.outcome
        checkpoint.representation_missing_primitive_ids = list(
            report.missing_primitive_ids
        )
        checkpoint.representation_source_resolution = report.outcome
        checkpoint.representation_retry_state = (
            "ELIGIBLE_AFTER_HASH_CHANGE"
            if report.retry_allowed else "NOT_ALLOWED"
        )
        checkpoint.recovery_events.append({
            "event_type": "CANDIDATE_REPRESENTATION_ANALYZED",
            "event_id": report.content_hash,
            "candidate_id": candidate_id,
            "candidate_index": index,
            "report_id": report.report_id,
            "outcome": report.outcome,
            "missing_primitive_ids": list(report.missing_primitive_ids),
            "semantic_issue_ids": list(report.semantic_issue_ids),
            "hidden_assumption_ids": list(report.hidden_assumption_ids),
            "private_text_exposed": False,
            "lean_invoked": False,
            "ledger_mutated": False,
            "created_at": time.time(),
        })
        if report.outcome in {
            RepresentationOutcome.MAPPER_EXTENSION_REQUIRED.value,
            RepresentationOutcome.REGISTRY_RESOLUTION.value,
        }:
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return DecompositionCertificateResult(
                False,
                [report.outcome],
                artifacts,
                hashes,
                transcripts,
                role_run_ids,
                {
                    "host_gates_passed": False,
                    "failure_status": "ENGINEERING_REPRESENTATION_GAP",
                    "candidate_id": candidate_id,
                    "representation_report_hash": report.content_hash,
                    "math_budget_charged": False,
                    "ledger_mutated": False,
                },
            )
        if report.outcome == RepresentationOutcome.DEFINITION_RESOLUTION.value:
            checkpoint.transition(
                ProofState.DEFINITION_RESOLUTION,
                "candidate-representation:autonomous-definition-resolution",
                strategy_reused=True,
            )
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return DecompositionCertificateResult(
                False,
                ["AUTONOMOUS_DEFINITION_RESOLUTION_REQUIRED"],
                artifacts,
                hashes,
                transcripts,
                role_run_ids,
                {
                    "host_gates_passed": False,
                    "failure_status": report.outcome,
                    "candidate_id": candidate_id,
                    "representation_report_hash": report.content_hash,
                    "ledger_mutated": False,
                },
            )
        checkpoint.exploration_rejections[candidate_id] = [
            "UNMAPPABLE_TYPED_CANDIDATE_INTENT",
            report.outcome,
            *report.semantic_issue_ids,
            *report.hidden_assumption_ids,
        ]
        checkpoint.exploration_formalization_status = (
            "REJECTED_" + report.outcome
        )
        if not queue_exhausted:
            checkpoint.exploration_current_index = index + 1
            checkpoint.exploration_current_candidate_id = (
                next_candidate_id
            )
            checkpoint.exploration_formalization_status = "PENDING"
            checkpoint.transition(
                ProofState.CANDIDATE_FORMALIZATION,
                "candidate-representation-rejected:try-next-ranked-candidate",
                strategy_reused=True,
            )
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return DecompositionCertificateResult(
                False,
                ["CANDIDATE_REPRESENTATION_REJECTED_TRY_NEXT"],
                artifacts,
                hashes,
                transcripts,
                role_run_ids,
                {
                    "host_gates_passed": False,
                    "failure_status": GateStatus.SEMANTIC_BACKJUMP.value,
                    "candidate_id": candidate_id,
                    "next_candidate_id": next_candidate_id,
                    "strategy_rerun": False,
                },
            )
        exhaustion = _canonical_json_hash({
            "schema_version": 1,
            "contract_id": checkpoint.exploration_contract_id,
            "candidate_set_hash": checkpoint.candidate_set_hash,
            "formalization_queue_ids": formalization_queue,
            "representation_report_hashes": sorted(
                item["sha256"]
                for item in checkpoint.representation_report_refs.values()
            ),
            "rejections": checkpoint.exploration_rejections,
            "ledger_mutated": False,
        })
        checkpoint.representation_exhaustion_hash = exhaustion
        checkpoint.exploration_current_candidate_id = ""
        checkpoint.exploration_formalization_status = (
            "REPRESENTATION_EXHAUSTED"
        )
        checkpoint.representation_current_status = (
            RepresentationOutcome.REPRESENTATION_EXHAUSTED.value
        )
        checkpoint.recovery_events.append({
            "event_type": "CANDIDATE_REPRESENTATION_EXHAUSTED",
            "event_id": exhaustion,
            "contract_id": checkpoint.exploration_contract_id,
            "candidate_set_hash": checkpoint.candidate_set_hash,
            "analyzed_candidate_ids": list(formalization_queue),
            "representation_report_hashes": sorted(
                item["sha256"]
                for item in checkpoint.representation_report_refs.values()
            ),
            "ledger_mutated": False,
            "oprover_invoked": False,
            "target_state": ProofState.DECOMPOSER.value,
            "created_at": time.time(),
        })
        checkpoint.transition(
            ProofState.DECOMPOSER,
            "candidate-representation-exhausted:typed-backjump",
            strategy_reused=True,
        )
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False,
            ["CANDIDATE_REPRESENTATION_EXHAUSTED"],
            artifacts,
            hashes,
            transcripts,
            role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": "REPRESENTATION_STAGNATION",
                "representation_exhaustion_hash": exhaustion,
            },
        )
    if (
        checkpoint.proof_state == ProofState.CANDIDATE_FORMALIZATION
        and checkpoint.ranked_candidate_ids
    ):
        index = checkpoint.exploration_current_index
        candidate_id, _, _ = next_formalization_candidate(
            checkpoint.ranked_candidate_ids,
            current_index=index,
        )
        checkpoint.exploration_current_candidate_id = candidate_id
        checkpoint.exploration_formalization_status = (
            "UNMAPPABLE_TYPED_CANDIDATE_INTENT"
        )
        checkpoint.representation_current_status = "PENDING_STATIC_ANALYSIS"
        checkpoint.representation_missing_primitive_ids = []
        checkpoint.representation_source_resolution = ""
        checkpoint.representation_retry_state = "NOT_ANALYZED"
        checkpoint.transition(
            ProofState.CANDIDATE_REPRESENTATION_ANALYSIS,
            "candidate-unmappable:representation-analysis",
            strategy_reused=True,
        )
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False,
            ["CANDIDATE_REPRESENTATION_ANALYSIS_REQUIRED"],
            artifacts,
            hashes,
            transcripts,
            role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": "REPRESENTATION_ANALYSIS_PENDING",
                "candidate_id": candidate_id,
                "lean_invoked": False,
                "ledger_mutated": False,
            },
        )
    missing = artifacts["definition_auditor"].missing_definitions
    required_kind = "DEFINITION" if missing else "LEMMA"
    parent_claim_ref = f"claim:{statement_hash}"
    dependency_ids = (hashes["definition_auditor"],)
    theorem_index = build_theorem_card_index(project_root)
    current_card_ids = set(checkpoint.theorem_card_ids)
    theorem_cards = tuple(
        card for card in theorem_index if card.card_id in current_card_ids
    )
    trigger = synthesis_trigger(
        novel_rejections=checkpoint.semantic_stagnation_count,
        repeated_no_move=sum(
            event.get("event_type") == "NO_REGISTERED_DECOMPOSITION_MOVE"
            for event in checkpoint.recovery_events[-4:]
        ),
        evidence_roles=(
            *artifacts.keys(), "critic", "definition_auditor", "theorem_cards",
        ),
    )
    synthesis_mode = (
        checkpoint.proof_state in {ProofState.SYNTHESIS, ProofState.REFRAME}
        or (trigger.invoke and not checkpoint.ranking_hash)
    )
    candidate_registry = build_candidate_set(
        target_ref=parent_claim_ref,
        viewpoint=checkpoint.viewpoint or "definitions",
        dependency_ids=dependency_ids,
        theorem_cards=theorem_cards,
        ancestor_hashes=(
            *_decomposition_ancestor_hashes(ledger, parent),
            *(
                str(event["novelty_hash"])
                for event in checkpoint.recovery_events
                if (
                    event.get("event_type")
                    == "HOST_TYPED_MOVE_SEMANTIC_REJECTION"
                    and event.get("novelty_hash")
                )
            ),
        ),
        satisfied_precondition_ids=host_evidence_context(
            artifacts, artifact_hashes=hashes,
        ).satisfied_precondition_ids,
        satisfied_theorem_hypothesis_ids=host_evidence_context(
            artifacts, artifact_hashes=hashes,
        ).satisfied_theorem_hypothesis_ids,
        available_dependency_artifact_ids=hashes.values(),
        required_dependency_artifact_ids=dependency_ids,
    )
    empty_fingerprint = mathematical_state_fingerprint(
        checkpoint,
        failure_code="EMPTY_CANDIDATE_SET",
        move_ids=(
            item[0] for item in candidate_registry.ineligible
        ),
    )
    if not candidate_registry.candidates:
        if empty_fingerprint in checkpoint.scratchpad_math_fingerprints:
            if checkpoint.exploration_exhaustion_hash:
                return DecompositionCertificateResult(
                    False,
                    ["DECOMPOSITION_EXPLORATION_EXHAUSTED"],
                    artifacts,
                    hashes,
                    transcripts,
                    role_run_ids,
                    {
                        "host_gates_passed": False,
                        "failure_status": "MATHEMATICAL_STAGNATION",
                        "exploration_exhaustion_hash": (
                            checkpoint.exploration_exhaustion_hash
                        ),
                    },
                )
        else:
            checkpoint.scratchpad_math_fingerprints.append(empty_fingerprint)
    scratch_role = (
        "synthesis_scratchpad" if synthesis_mode else "decomposer_scratchpad"
    )
    scratch_run_id = f"{orchestration_id}:{scratch_role}:v3"
    scratch_messages = [{
        "role": "system",
        "content": (
            "Reason privately in mathematical prose or LaTeX. This transcript "
            "is untrusted, audit-only, and is never parsed or used by a gate. "
            "Compare only the supplied registered gaps, evidence, theorem cards, "
            "and moves. Do not invent prerequisites or mathematical facts. "
            "Do not include secrets."
        ),
    }, {
        "role": "user",
        "content": (
            f"Target reference: {parent_claim_ref}. Viewpoint: "
            f"{checkpoint.viewpoint or 'local_holomorphicity'}. Cards: "
            + ", ".join(card.card_id for card in theorem_cards)
        ),
    }]
    try:
        scratch_text, scratch_actual_run_id = run_role(
            scratch_role, scratch_messages, scratch_run_id,
        )
        transcripts[scratch_role] = scratch_text
        if scratch_actual_run_id != scratch_run_id:
            raise RuntimeError("scratchpad run ID mismatch")
        scratch_ref = persist_private_scratchpad(
            checkpoint_path.with_suffix(".scratchpads"),
            role=scratch_role,
            transcript=scratch_text,
            token_count=max(1, len(scratch_text.split())),
        )
        checkpoint.scratchpad_refs.append({
            "role": scratch_ref.role,
            "sha256": scratch_ref.sha256,
            "token_count": scratch_ref.token_count,
            "audit_only": True,
            "authoritative": False,
            "public": False,
        })
    except Exception as exc:
        checkpoint.adapter_blocked(
            f"private-scratchpad-inference:{type(exc).__name__}:{exc}",
            status="INFRASTRUCTURE_BLOCKED",
        )
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, [checkpoint.blocked_reason], artifacts, hashes,
            transcripts, role_run_ids, {
                "host_gates_passed": False,
                "failure_status": "INFRASTRUCTURE_BLOCKED",
            },
        )
    if not candidate_registry.candidates:
        exploration = _start_decomposition_exploration(
            checkpoint,
            checkpoint_path=checkpoint_path,
            run_role=run_role,
            orchestration_id=orchestration_id,
            artifact_hashes=hashes,
        )
        return DecompositionCertificateResult(
            False,
            ["DECOMPOSITION_EXPLORATION_STARTED"],
            artifacts,
            hashes,
            transcripts,
            role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": GateStatus.SEMANTIC_BACKJUMP.value,
                "candidate_registry_hash": candidate_registry.content_hash,
                **exploration,
            },
        )
    checkpoint.candidate_set_hash = candidate_registry.content_hash
    checkpoint.candidate_hashes = [
        candidate.candidate_hash for candidate in candidate_registry.candidates
    ]
    checkpoint.candidate_count = len(candidate_registry.candidates)
    checkpoint.theorem_card_ids = [
        card.card_id for card in theorem_cards
    ]
    checkpoint.theorem_card_index_hash = _canonical_json_hash(
        [card.content_hash for card in theorem_index],
    )
    evidence_context = host_evidence_context(artifacts, artifact_hashes=hashes)
    evidence_graph = build_evidence_gap_graph(
        artifacts=artifacts,
        artifact_hashes=hashes,
        advisory_artifacts=checkpoint.advisory_artifacts,
        theorem_cards=theorem_cards,
        candidate_set=candidate_registry,
        failure_reason_codes=tuple(
            str(code)
            for item in checkpoint.invalidated_artifacts.values()
            for code in item.get("reason_codes", ())
        ),
    )
    proof_plans = generate_proof_plans(
        evidence_graph,
        unresolved_gap_ids=evidence_context.unresolved_gap_ids,
    )
    for proof_plan in proof_plans:
        validate_proof_plan(evidence_graph, proof_plan)
    checkpoint.evidence_gap_graph_hash = evidence_graph.content_hash
    if proof_plans:
        checkpoint.proof_plan_id = proof_plans[0].plan_id
        checkpoint.proof_plan_hash = proof_plans[0].content_hash
        checkpoint.executable_plan_node_id = first_executable_node(
            proof_plans[0],
        ).graph_node_id
        checkpoint.plan_score_explanation = dict(
            proof_plans[0].score_explanation,
        )
    else:
        checkpoint.proof_plan_id = ""
        checkpoint.proof_plan_hash = ""
        checkpoint.executable_plan_node_id = ""
        checkpoint.plan_score_explanation = {}
    short_choice_map = candidate_registry.short_choice_map
    if synthesis_mode:
        if checkpoint.proof_state != ProofState.SYNTHESIS:
            checkpoint.transition(
                ProofState.SYNTHESIS,
                f"synthesis-trigger:{trigger.reason}",
                strategy_reused=True,
            )
        checkpoint.synthesis_iteration += 1
        synthesis_package = {
            "target_obligation_id": parent.obligation_id,
            "target_statement": parent.statement,
            "strategy_plan_id": checkpoint.selected_strategy_plan_id,
            "verified_target_evidence": json.dumps(
                checkpoint.target_evidence,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "current_target_gaps": " ".join(
                checkpoint.target_gap_ids
            ) or "none",
            "registered_output_choices": {
                "choice_code": candidate_registry.short_choice_codes,
                "reason_code": (
                    "DIRECT_LOCAL_CONTRADICTION",
                    "SUPPORTED_BY_THEOREM_CARDS",
                    "STRICTEST_REDUCTION",
                    "NOVEL_VIEWPOINT",
                    "LOWEST_COMPLEXITY",
                    "HIGHEST_GAP_COVERAGE",
                    "SHALLOWEST_DEPENDENCY_DEPTH",
                ),
            },
            "candidate_choices": tuple({
                "choice_code": code,
                "move_id": item.move.move_id,
                "metric_parent": item.move.metric.parent,
                "metric_child": item.move.metric.child,
                "theorem_card_ids": item.theorem_card_ids,
                "requires_parent_case_split": item.move.requires_parent_case_split,
            } for code, item in short_choice_map.items()),
            "short_choice_map_hash": candidate_registry.short_choice_map_hash,
            "advisory_counterexample": True,
            "counterexample_may_be_premise": False,
        }
        synthesis_run_id = f"{orchestration_id}:synthesis:v3"
        try:
            synthesis_text, synthesis_actual_id = run_role(
                "synthesis",
                _typed_role_messages("synthesis", synthesis_package),
                synthesis_run_id,
            )
            transcripts["synthesis"] = synthesis_text
            if synthesis_actual_id != synthesis_run_id:
                raise AdapterError(
                    "RUN_ID_MISMATCH", "synthesis returned another run ID",
                    role="synthesis",
                )
            synthesis_fields = decode_role_fields(
                synthesis_text,
                "synthesis",
                registered_choices=synthesis_package[
                    "registered_output_choices"
                ],
            ).values
            selected_choice_code = str(synthesis_fields["choice_code"])
            ranking = rank_short_choice(
                candidate_registry,
                choice_code=selected_choice_code,
                reason_code=str(synthesis_fields["reason_code"]),
                candidate_set_hash=candidate_registry.content_hash,
                short_choice_map_hash=candidate_registry.short_choice_map_hash,
            )
        except ValueError as exc:
            checkpoint.recovery_events.append({
                "event_type": "INVALID_SYNTHESIS_RATIONALE",
                "reason": str(exc),
                "candidate_set_hash": candidate_registry.content_hash,
                "short_choice_map_hash": candidate_registry.short_choice_map_hash,
                "created_at": time.time(),
            })
            checkpoint.adapter_blocked(
                f"invalid Host-checked synthesis rationale: {exc}",
                status="INTEGRATION_BLOCKED",
            )
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return DecompositionCertificateResult(
                False, [str(exc)], artifacts, hashes, transcripts, role_run_ids,
                {"host_gates_passed": False, "failure_status": "INTEGRATION_BLOCKED"},
            )
    else:
        prior_choice = next((
            code for code, candidate in short_choice_map.items()
            if candidate.move.move_id == checkpoint.selected_move_id
        ), "")
        if checkpoint.ranking_hash and prior_choice:
            selected_choice_code = prior_choice
            ranking = rank_short_choice(
                candidate_registry,
                choice_code=selected_choice_code,
                reason_code="LOWEST_COMPLEXITY",
                candidate_set_hash=candidate_registry.content_hash,
                short_choice_map_hash=candidate_registry.short_choice_map_hash,
            )
        else:
            ranking = rank_candidates(candidate_registry)
            selected_choice_code = next(
                code for code, candidate in short_choice_map.items()
                if candidate.candidate_id == ranking.selected_candidate_id
            )
    checkpoint.ranking_hash = ranking.ranking_hash
    checkpoint.ranked_candidate_ids = list(ranking.ordered_candidate_ids)
    ranked_selected = candidate_registry.resolve(ranking.selected_candidate_id)
    checkpoint.selected_move_id = ranked_selected.move.move_id
    checkpoint.viewpoint = (
        "local_holomorphicity_special_case"
        if ranked_selected.move.move_id == "SINGULARITY_CONTRADICTION"
        else checkpoint.viewpoint
    )
    checkpoint.clear_adapter_blocked("candidate-ranking-complete")
    if checkpoint.proof_state == ProofState.SYNTHESIS:
        checkpoint.transition(
            ProofState.DECOMPOSER,
            "synthesis-ranking-complete",
            strategy_reused=True,
        )
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    package = {
        "target_obligation_id": parent.obligation_id,
        "parent_claim_ref": parent_claim_ref,
        "plain_semantic_summary": (
            "derive a smaller density and local convergence obligation from "
            "the registered definitions"
        ),
        "root_goal_hash": goal_hash,
        "viewpoint": checkpoint.viewpoint or "definitions",
        "registered_output_choices": {
            "parent_claim_ref": (parent_claim_ref,),
            "child_kind": (required_kind,),
            "definition_id": tuple(
                str(item.get("obligation_label", ""))
                for item in missing if isinstance(item, dict)
            ),
            "move_id": (selected_choice_code,),
            "dependency_id": dependency_ids,
        },
        "candidate_choices": tuple({
            "choice_code": code,
            "summary_id": item.move.structural_delta,
            "candidate_hash": item.candidate_hash,
            "move_id": item.move.move_id,
            "result_status": item.move.result_status,
            "requires_parent_case_split": item.move.requires_parent_case_split,
            "theorem_card_ids": item.theorem_card_ids,
        } for code, item in short_choice_map.items()),
        "validated_upstream_artifacts": {
            "definition_auditor": _certified_upstream_view(
                artifacts["definition_auditor"], consumer_role="decomposer",
            ),
        },
    }
    expected_run_id = f"{orchestration_id}:decomposer:v2"
    try:
        text, actual_run_id = run_role(
            "decomposer",
            _typed_role_messages("decomposer", package),
            expected_run_id,
        )
        transcripts["decomposer"] = text
        if actual_run_id != expected_run_id:
            raise AdapterError(
                "RUN_ID_MISMATCH", "adapter returned another run ID",
                role="decomposer",
            )
        decoded = decode_role_fields(
            text,
            "decomposer",
            registered_choices=package["registered_output_choices"],
        )
    except AdapterError as exc:
        checkpoint.adapter_blocked(str(exc), status=exc.status.value)
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, [str(exc)], artifacts, hashes, transcripts, role_run_ids,
            {"host_gates_passed": False, "failure_status": exc.status.value},
        )
    except Exception as exc:
        checkpoint.adapter_blocked(
            f"{type(exc).__name__}: {exc}",
            status="INFRASTRUCTURE_BLOCKED",
        )
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, [checkpoint.blocked_reason], artifacts, hashes,
            transcripts, role_run_ids, {
                "host_gates_passed": False,
                "failure_status": "INFRASTRUCTURE_BLOCKED",
            },
        )
    fields = decoded.values
    selected_candidate = candidate_registry.resolve_short_code(
        str(fields["move_id"]),
        candidate_set_hash=candidate_registry.content_hash,
        short_choice_map_hash=candidate_registry.short_choice_map_hash,
    )
    source_labels = list(fields["definition_id"])
    # The host owns enum construction; the model supplies only a field value.
    child_kind = str(fields["child_kind"]).strip().upper()
    typed_ir_steps = selected_candidate.typed_payload
    typed_child_ref = selected_candidate.typed_ir_hash
    child_statement = (
        f"Host typed proposition reference {typed_child_ref}; "
        f"status={selected_candidate.move.result_status}; "
        "does_not_prove_parent=true; "
        f"requires_parent_case_split="
        f"{str(selected_candidate.move.requires_parent_case_split).lower()}"
    )
    outline_steps = list(fields["outline_step_id"])
    # Public assumptions are immutable Host contract bytes.  They never cross
    # the model transport and are injected into both consumers from one value.
    (
        canonical_public_assumptions,
        assumptions_hash,
        restriction_move,
    ) = _host_owned_assumption_contract(
        parent.public_assumptions,
        selected_candidate.move,
    )
    child_contract = {
        "label": "L1",
        "statement": child_statement,
        "kind": child_kind,
        "source_definition_labels": source_labels,
        "move_id": selected_candidate.move.move_id,
        "result_status": selected_candidate.move.result_status,
        "requires_parent_case_split": (
            selected_candidate.move.requires_parent_case_split
        ),
        "proves_parent": False,
        "theorem_card_ids": list(selected_candidate.theorem_card_ids),
        "public_assumptions": canonical_public_assumptions,
        "public_assumptions_hash": assumptions_hash,
    }
    reduction_contract = {
        "child_label": "L1",
        "parent_statement": parent.statement,
        "public_assumptions": canonical_public_assumptions,
        "public_assumptions_hash": assumptions_hash,
        "derivation": (
            "host typed IR special-case derivation; separate parent "
            "case-split reduction required"
            + (": " + " ".join(outline_steps) if outline_steps else "")
        ),
    }
    if restriction_move is not None:
        child_contract["restriction_move"] = restriction_move
        reduction_contract["restriction_move"] = restriction_move
    proposal = DecompositionProposal(
        parent.obligation_id,
        statement_hash,
        goal_hash,
        "decomposer",
        actual_run_id,
        [hashes["definition_auditor"]],
        parent.statement,
        child_contract,
        canonical_public_assumptions,
        reduction_contract,
    )
    checkpoint.public_assumptions_hash = assumptions_hash
    checkpoint.child_public_assumptions_hash = assumptions_hash
    checkpoint.reduction_public_assumptions_hash = assumptions_hash
    semantic_errors = [
        *_validate_decomposition_shape(proposal),
        *_validate_definition_child_selection(
            artifacts["definition_auditor"], proposal,
        ),
    ]
    if semantic_errors:
        checkpoint.begin_decomposition_iteration(
            _select_decomposer_viewpoint(
                checkpoint, artifacts["definition_auditor"],
            ),
            "typed-decomposer-semantic-rejection:" + "; ".join(semantic_errors),
        )
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, semantic_errors, artifacts, hashes, transcripts, role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": GateStatus.MATHEMATICAL_REJECTION.value,
            },
        )
    proposal_payload = asdict(proposal)
    proposal_ref = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="decomposer",
        payload=proposal_payload,
        dependencies=[hashes["definition_auditor"]],
        source_run_id=actual_run_id,
        artifact_schema_version=2,
    )
    artifacts["decomposer"] = proposal
    hashes["decomposer"] = proposal_ref.sha256
    role_run_ids["decomposer"] = actual_run_id
    envelope = host_artifact(
        decoded,
        host_bindings={
            "target_obligation_id": parent.obligation_id,
            "parent_statement_hash": statement_hash,
            "root_goal_hash": goal_hash,
            "producer_run_id": actual_run_id,
            "candidate_registry_hash": candidate_registry.content_hash,
            "selected_candidate_hash": selected_candidate.candidate_hash,
            "selected_move_id": selected_candidate.move.move_id,
            "candidate_set_hash": candidate_registry.content_hash,
            "ranking_hash": ranking.ranking_hash,
            "short_choice_map_hash": candidate_registry.short_choice_map_hash,
            "selected_choice_code": selected_choice_code,
            "theorem_card_ids": list(selected_candidate.theorem_card_ids),
            "result_status": selected_candidate.move.result_status,
            "requires_parent_case_split": (
                selected_candidate.move.requires_parent_case_split
            ),
            "proves_parent": False,
        },
        dependencies=[proposal_ref.sha256],
    )
    assert_no_scratchpad_content(envelope, scratch_ref)
    synthesis_payload = synthesis_manifest(
        evidence_hashes={
            key: value for key, value in hashes.items()
            if key in {"critic", "definition_auditor", "counterexample_worker"}
        },
        rejected_reason_codes=(
            checkpoint.semantic_rejection.get("rejection_reason_codes", [])
            if isinstance(checkpoint.semantic_rejection, dict) else ()
        ),
        theorem_card_ids=(card.card_id for card in theorem_cards),
        scratchpad_ref=scratch_ref,
        ranking=ranking,
        short_choice_map_hash=candidate_registry.short_choice_map_hash,
        selected_choice_code=selected_choice_code,
        counterexample_verified=False,
    )
    assert_no_scratchpad_content(synthesis_payload, scratch_ref)
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="synthesis",
        payload=synthesis_payload,
        dependencies=list(dependency_ids),
        source_run_id=(
            synthesis_run_id if synthesis_mode else "host-deterministic-ranking"
        ),
        artifact_schema_version=1,
    )
    ir_ref = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="math_ir_translator",
        payload=envelope,
        dependencies=[proposal_ref.sha256],
        source_run_id=actual_run_id,
        artifact_schema_version=2,
    )
    checkpoint.transition(
        ProofState.MATH_IR_TRANSLATION,
        "host-selected-candidate-assembled",
        source_run_id=actual_run_id,
        strategy_reused=True,
    )
    checkpoint.transition(
        ProofState.HOST_TYPED_IR_GATE,
        "typed-math-ir-host-envelope-persisted",
        source_run_id=actual_run_id,
        strategy_reused=True,
    )
    checkpoint.active_gate = "HOST_TYPED_IR_GATE"
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    gate_result = run_host_gates(
        typed_ir_steps,
        project_root=project_root,
        cache_dir=checkpoint_path.with_suffix(".host-gates"),
        lean_validator=signature_validator,
    )
    gate_payload = {
        "schema_version": 2,
        "input_artifact_hash": ir_ref.sha256,
        "ok": gate_result.ok,
        "evidence": [asdict(item) for item in gate_result.evidence],
        "compilation": (
            asdict(gate_result.compilation) if gate_result.compilation else {}
        ),
    }
    gate_ref = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="host_typed_ir_gate",
        payload=gate_payload,
        dependencies=[ir_ref.sha256],
        source_run_id="host",
        artifact_schema_version=2,
    )
    if not gate_result.ok:
        errors = [gate_result.evidence[-1].message]
        checkpoint.active_gate = gate_result.evidence[-1].stage
        if gate_result.failure_status == GateStatus.SEMANTIC_BACKJUMP.value:
            for role in ("decomposer", "math_ir_translator", "host_typed_ir_gate"):
                stale = checkpoint.validated_artifacts.pop(role, None)
                if stale:
                    checkpoint.invalidated_artifacts[stale.sha256] = {
                        **asdict(stale),
                        "audit_only": True,
                        "reason_codes": [gate_result.evidence[-1].code],
                    }
            checkpoint.recovery_events.append({
                "event_type": "HOST_TYPED_MOVE_SEMANTIC_REJECTION",
                "event_id": (
                    "host-typed-move-semantic-rejection-"
                    + selected_candidate.novelty_hash[:16]
                ),
                "candidate_hash": selected_candidate.candidate_hash,
                "novelty_hash": selected_candidate.novelty_hash,
                "move_id": selected_candidate.move.move_id,
                "reason_code": gate_result.evidence[-1].code,
                "created_at": time.time(),
            })
            checkpoint.ranking_hash = ""
            checkpoint.ranked_candidate_ids = []
            checkpoint.selected_move_id = ""
            checkpoint.transition(
                ProofState.SYNTHESIS,
                f"typed-evidence-backjump:{gate_result.evidence[-1].code}",
                strategy_reused=True,
            )
        else:
            checkpoint.blocked_reason = errors[0]
            checkpoint.transition(ProofState.BLOCKED, errors[0])
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, errors, artifacts, hashes, transcripts, role_run_ids,
            {
                "host_gates_passed": False,
                "failure_status": gate_result.failure_status,
                "gate_evidence_hash": gate_ref.sha256,
            },
        )
    compilation = gate_result.compilation
    assert compilation is not None
    checkpoint.typed_ir_hash = compilation.math_ir_hash
    checkpoint.lean_declaration_hash = compilation.declaration_hash
    checkpoint.proposition_hash = compilation.proposition_hash
    checkpoint.elaborated_theorem_id = compilation.theorem_id
    checkpoint.active_gate = "LEAN_ELABORATION_GATE"
    checkpoint.transition(
        ProofState.LEAN_ELABORATION_GATE,
        "host-typed-ir-gate-passed",
        strategy_reused=True,
    )
    quarantined_parent = any(
        str(review.get("status", "")).upper() == "QUARANTINED"
        and parent.obligation_id in {
            str(item)
            for key in ("plan_ids", "evidence_ids")
            for item in review.get(key, ())
        }
        for review in checkpoint.branch_history.values()
    )
    contract_target_ref = parent.obligation_id
    contract_parent_ref = parent.parent_id or "ROOT"
    if quarantined_parent:
        contract_target_ref = (
            parent.obligation_id
            + ":typed-reframe:"
            + compilation.proposition_hash[:20]
        )
        contract_parent_ref = parent.obligation_id
        reframe_event_id = (
            "typed-reframe-backjump:" + compilation.proposition_hash[:20]
        )
        if not any(
            item.get("event_id") == reframe_event_id
            for item in checkpoint.recovery_events
        ):
            checkpoint.recovery_events.append({
                "event_type": "TYPED_REFRAME_BACKJUMP",
                "event_id": reframe_event_id,
                "quarantined_parent_ref": parent.obligation_id,
                "replacement_target_ref": contract_target_ref,
                "proposition_hash": compilation.proposition_hash,
                "theorem_id": compilation.theorem_id,
                "created_at": time.time(),
            })
        checkpoint.target_obligation_id = contract_target_ref
    checkpoint.transition(
        ProofState.STRATEGY_TOURNAMENT,
        "lean-elaboration-passed:rerun-tournament-with-proposition-hash",
        strategy_reused=False,
    )
    checkpoint = run_architecture_v9_entry(
        checkpoint_path,
        checkpoint,
        project_root=project_root,
        target_ref=contract_target_ref,
        parent_obligation_ref=contract_parent_ref,
        parent_complexity=max(5, len(parent.statement.split())),
        event_type=StrategyEvent.TARGET_CHANGE,
        event_id=(
            "TARGET_CHANGE:"
            + hashlib.sha256(
                (
                    parent.obligation_id + compilation.proposition_hash
                ).encode(),
            ).hexdigest()[:20]
        ),
        elaborated_theorem_id=compilation.theorem_id,
        proposition_hash=compilation.proposition_hash,
    )
    if not checkpoint.research_contract_id:
        return DecompositionCertificateResult(
            False,
            [
                "RESEARCH_CONTRACT_REJECTED:"
                + ",".join(checkpoint.research_contract_rejection_codes)
            ],
            artifacts,
            hashes,
            transcripts,
            role_run_ids,
            {
                "host_gates_passed": True,
                "failure_status": GateStatus.SEMANTIC_BACKJUMP.value,
                "route_state": checkpoint.state,
            },
        )
    checkpoint.active_gate = "PROOF_SEARCH"
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    contract_ref = checkpoint.validated_artifacts.get("research_contract")
    if contract_ref is None:
        raise RuntimeError("proof search requires a persisted ResearchContract")
    contract_payload = json.loads(Path(contract_ref.path).read_text())
    from autoresearch.prefill.research_contract import ResearchContract
    contract_payload.pop("schema_version", None)
    contract = ResearchContract(**contract_payload)
    proof_state_path = checkpoint_path.with_name("stepwise_proof_state.json")
    checkpoint.proof_search_state_path = str(proof_state_path)
    search = new_search_state(contract, [ProofGoal(
        "G1",
        compilation.proposition_hash,
        (),
        f"proposition:{compilation.proposition_hash}",
    )], proof_budget=16)
    executor = lean_step_executor(LeanExecutionContext(
        project_root,
        compilation.declaration_source,
        ("KakeyaLeanGate.Prelude",),
    ))
    operand_sources = {
        f"TC{index}": card.theorem_name
        for index, card in enumerate(theorem_cards, 1)
    }
    theorem_operands = {
        card.card_id: f"TC{index}"
        for index, card in enumerate(theorem_cards, 1)
    }
    proof_actual_run_id = ""
    for step_index in range(1, 33):
        if search.status != "SEARCHING":
            break
        actions = enumerate_applicable_actions(
            search,
            local_context_ids=(),
            theorem_card_to_operand_id=theorem_operands,
        )
        goal_id = search.open_goals[0].goal_id
        action_ids = tuple(action.action_id for action in actions)
        operand_ids = tuple(sorted({
            operand for action in actions for operand in action.operand_ids
        }))
        proof_package = {
            "target_obligation_id": parent.obligation_id,
            "parent_claim_ref": f"proposition:{compilation.proposition_hash}",
            "plain_semantic_summary": "select one registered action",
            "registered_output_choices": {
                "goal_id": (goal_id,),
                "action_id": action_ids,
                "operand_id": operand_ids,
                "substitution_id": (),
            },
        }
        proof_run_id = (
            f"{orchestration_id}:proof_action_selector:{step_index}"
        )
        try:
            proof_text, proof_actual_run_id = run_role(
                "proof_action_selector",
                _typed_role_messages(
                    "proof_action_selector", proof_package,
                ),
                proof_run_id,
            )
            transcripts[f"proof_action_{step_index}"] = proof_text
            if proof_actual_run_id != proof_run_id:
                raise AdapterError(
                    "RUN_ID_MISMATCH", "adapter returned another run ID",
                    role="proof_action_selector",
                )
            fields = decode_role_fields(
                proof_text,
                "proof_action_selector",
                registered_choices=proof_package[
                    "registered_output_choices"
                ],
            ).values
            selected_action = next(
                action for action in actions
                if action.action_id == fields["action_id"]
            )
            selection = ActionSelection(
                str(fields["goal_id"]),
                str(fields["action_id"]),
                tuple(fields.get("operand_id", ())),
                (),
                (),
            )
            checkpoint.lean_actions_attempted += 1
            result = attempt_step(
                search,
                selection,
                actions,
                operand_sources=operand_sources,
                substitution_sources={},
                lean_executor=executor,
            )
        except AdapterError as exc:
            checkpoint.adapter_blocked(str(exc), status=exc.status.value)
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            return DecompositionCertificateResult(
                False, [str(exc)], artifacts, hashes, transcripts, role_run_ids,
                {"host_gates_passed": True, "failure_status": exc.status.value},
            )
        except Exception as exc:
            checkpoint.protocol_error_count += 1
            checkpoint.last_transition_reason = (
                f"proof-action-host-error:{type(exc).__name__}"
            )
            save_orchestration_checkpoint(checkpoint_path, checkpoint)
            continue
        if result.accepted:
            checkpoint.lean_actions_accepted += 1
        checkpoint.subgoals_closed = max(
            0, checkpoint.lean_actions_accepted - len(search.open_goals),
        )
        checkpoint.subgoals_remaining = len(search.open_goals)
        persist_search_state(proof_state_path, search)
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
    if search.status != "PROVED":
        errors = [
            f"stepwise proof search stopped with {search.status}; "
            f"remaining_subgoals={len(search.open_goals)}"
        ]
        checkpoint.mathematical_retries += search.semantic_failures
        checkpoint.last_transition_reason = errors[0]
        save_orchestration_checkpoint(checkpoint_path, checkpoint)
        return DecompositionCertificateResult(
            False, errors, artifacts, hashes, transcripts, role_run_ids,
            {
                "host_gates_passed": True,
                "failure_status": GateStatus.MATHEMATICAL_REJECTION.value,
            },
        )
    proof_source = "\n".join((
        compilation.declaration_source,
        *(f"  {step.rendered_ast}" for step in search.accepted_steps),
        "",
    ))
    proof_result = proof_validator(proof_source, project_root=project_root)
    if not proof_result.ok:
        raise RuntimeError(
            "accepted stepwise proof failed final validation: "
            + proof_result.error,
        )
    formalization = FormalizationBundle(
        parent.obligation_id, statement_hash, goal_hash, "host_compiler", "host",
        [proposal_ref.sha256], compilation.declaration_source,
        compilation.declaration_hash, parent.formal_status == "UNFORMALIZED",
        {
            "label": "L1",
            "lean_signature": compilation.declaration_source,
            "lean_signature_hash": compilation.declaration_hash,
        },
        compilation.declaration_source, compilation.declaration_hash,
    )
    proof = ProofAttempt(
        parent.obligation_id, statement_hash, goal_hash, "proof_search",
        proof_actual_run_id, [gate_ref.sha256], "PROVED", proof_source,
    )
    artifacts["formalizer"] = formalization
    artifacts["prover"] = proof
    hashes["formalizer"] = _canonical_json_hash(asdict(formalization))
    hashes["prover"] = _canonical_json_hash(asdict(proof))
    role_run_ids["formalizer"] = "host"
    role_run_ids["prover"] = proof_actual_run_id
    checkpoint.transition(
        ProofState.ADVERSARIAL_REVIEW,
        "typed-proof-plan-host-rendered-and-elaborated",
        strategy_reused=True,
    )
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    return DecompositionCertificateResult(
        False,
        ["adversarial review pending"],
        artifacts,
        hashes,
        transcripts,
        role_run_ids,
        {
            "host_gates_passed": True,
            "typed_ir_hash": compilation.math_ir_hash,
            "proposition_hash": compilation.proposition_hash,
            "gate_evidence_hash": gate_ref.sha256,
        },
    )


def retain_contract_provenance_after_artifact_failure(
    checkpoint: OrchestrationCheckpoint,
) -> None:
    """Invalidate executable role artifacts without erasing accepted strategy."""
    prior = checkpoint.validated_artifacts
    preserved = {
        role: reference
        for role, reference in prior.items()
        if role in {"strategy_tournament", "research_contract"}
    }
    definition = prior.get("definition_auditor")
    tournament = prior.get("strategy_tournament")
    if (
        definition is not None
        and tournament is not None
        and definition.sha256 in tournament.dependencies
    ):
        preserved["contract_definition_auditor"] = replace(
            definition,
            role="contract_definition_auditor",
            strategy_plan_hash=checkpoint.target_strategy_plan_hash,
        )
    checkpoint.validated_artifacts = preserved


def run_certified_decomposition(
    ledger: ProofObligationLedger,
    target_id: str,
    root_goal: str,
    run_role,
    *,
    project_root: Path,
    orchestration_id: str,
    signature_validator=validate_lean_signature,
    proof_validator=validate_lean_proof,
    checkpoint_path: Path | None = None,
    candidate_sha256: str = "",
    protocol_retry_limit: int = 2,
) -> DecompositionCertificateResult:
    parent = next(
        item for item in ledger.obligations
        if item.obligation_id == target_id
    )
    statement_hash = hashlib.sha256(parent.statement.encode()).hexdigest()
    goal_hash = hashlib.sha256(root_goal.encode()).hexdigest()
    artifacts = {}
    hashes = {}
    transcripts = {}
    role_run_ids = {}
    errors = []
    role_specs = [
        ("definition_auditor", "DEFINITION_AUDIT"),
        ("counterexample_worker", "COUNTEREXAMPLE_REPORT"),
        ("decomposer", "DECOMPOSITION_PROPOSAL"),
        ("formalizer", "FORMALIZATION_BUNDLE"),
        ("prover", "PROOF_ATTEMPT"),
        ("adversarial_proponent", "DEFENSE_REPORT"),
    ]
    artifact_types = {
        "definition_auditor": DefinitionAudit,
        "counterexample_worker": CounterexampleReport,
        "decomposer": DecompositionProposal,
        "formalizer": FormalizationBundle,
        "prover": ProofAttempt,
        "adversarial_proponent": DefenseReport,
        "judge": JudgeDecision,
    }
    orchestration_checkpoint = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser()
        orchestration_checkpoint = load_orchestration_checkpoint(
            checkpoint_path,
        )
        adapter_binding_matches = bool(
            orchestration_checkpoint is not None
            and not binding_mismatch(
                orchestration_checkpoint,
                target_obligation_id=target_id,
                candidate_sha256=candidate_sha256,
                parent_statement_sha256=statement_hash,
                parent_signature_sha256=parent.lean_signature_hash,
                root_goal_sha256=goal_hash,
                ledger_id=ledger.ledger_id,
                ledger_version=ledger.version,
            )
        )
        legacy_definition_adapter_block = bool(
            orchestration_checkpoint is not None
            and orchestration_checkpoint.proof_state
            == ProofState.DEFINITION_AUDITOR
            and orchestration_checkpoint.adapter_status == "ADAPTER_BLOCKED"
            and "DEFINITION_AUDIT Artifact JSON" in (
                orchestration_checkpoint.blocked_reason
            )
            and adapter_binding_matches
        )
        if legacy_definition_adapter_block:
            orchestration_checkpoint.clear_adapter_blocked(
                "typed-definition-auditor-migration",
            )
            orchestration_checkpoint.recovery_events.append({
                "event_type": "LEGACY_DEFINITION_OUTPUT_AUDIT_ONLY",
                "event_id": "definition-auditor-typed-transport-v1",
                "target_state": ProofState.DEFINITION_AUDITOR.value,
                "created_at": time.time(),
            })
            save_orchestration_checkpoint(
                checkpoint_path, orchestration_checkpoint,
            )
        if (
            orchestration_checkpoint is not None
            and (
                orchestration_checkpoint.proof_state == ProofState.BLOCKED
                or (
                    orchestration_checkpoint.adapter_status == "ADAPTER_BLOCKED"
                    and adapter_binding_matches
                )
            )
        ):
            return DecompositionCertificateResult(
                verified=False,
                errors=[
                    orchestration_checkpoint.blocked_reason
                    or "orchestration is BLOCKED",
                ],
                artifacts={},
                artifact_hashes={},
                transcripts={},
                role_run_ids={},
                validation={
                    "host_gates_passed": False,
                    "blocked": True,
                    "failure_status": (
                        orchestration_checkpoint.adapter_status
                        or ProofState.BLOCKED.value
                    ),
                },
            )
        if (
            orchestration_checkpoint is not None
            and orchestration_checkpoint.adapter_status
            == "INFRASTRUCTURE_BLOCKED"
            and adapter_binding_matches
        ):
            orchestration_checkpoint.clear_adapter_blocked(
                "quiescent-infrastructure-retry",
            )
            orchestration_checkpoint.recovery_events.append({
                "event_type": "QUIESCENT_INFRASTRUCTURE_RETRY",
                "event_id": (
                    "quiescent-infrastructure-retry-"
                    + hashlib.sha256(
                        orchestration_checkpoint.last_transition_reason.encode(),
                    ).hexdigest()[:16]
                ),
                "target_state": orchestration_checkpoint.state,
                "created_at": time.time(),
            })
            save_orchestration_checkpoint(
                checkpoint_path, orchestration_checkpoint,
            )
        mismatch = ""
        if orchestration_checkpoint is not None:
            mismatch = binding_mismatch(
                orchestration_checkpoint,
                target_obligation_id=target_id,
                candidate_sha256=candidate_sha256,
                parent_statement_sha256=statement_hash,
                parent_signature_sha256=parent.lean_signature_hash,
                root_goal_sha256=goal_hash,
                ledger_id=ledger.ledger_id,
                ledger_version=ledger.version,
            )
        if orchestration_checkpoint is not None and mismatch:
            orchestration_checkpoint.adapter_blocked(
                "CANONICAL_BINDING_OWNERSHIP_REQUIRED:" + mismatch,
                status="INTEGRATION_BLOCKED",
            )
            orchestration_checkpoint.recovery_events.append({
                "event_type": "CANONICAL_BINDING_MISMATCH_REJECTED",
                "event_id": (
                    "canonical-binding-mismatch:"
                    + hashlib.sha256(
                        (
                            mismatch
                            + orchestration_checkpoint.target_obligation_id
                            + orchestration_checkpoint.proposition_hash
                        ).encode()
                    ).hexdigest()[:20]
                ),
                "reason": mismatch,
                "preserved_artifact_hashes": sorted(
                    reference.sha256
                    for reference
                    in orchestration_checkpoint.validated_artifacts.values()
                ),
                "preserved_research_contract_id": (
                    orchestration_checkpoint.research_contract_id
                ),
                "created_at": time.time(),
            })
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
            return DecompositionCertificateResult(
                verified=False,
                errors=[
                    "CANONICAL_BINDING_OWNERSHIP_REQUIRED:" + mismatch,
                ],
                artifacts={},
                artifact_hashes={},
                transcripts={},
                role_run_ids={},
                validation={
                    "host_gates_passed": False,
                    "blocked": True,
                    "failure_status": "INTEGRATION_BLOCKED",
                    "preserved_research_contract_id": (
                        orchestration_checkpoint.research_contract_id
                    ),
                },
            )
        if orchestration_checkpoint is None:
            previous_checkpoint = orchestration_checkpoint
            orchestration_checkpoint = OrchestrationCheckpoint(
                state=ProofState.DEFINITION_AUDITOR.value,
                target_obligation_id=target_id,
                candidate_sha256=candidate_sha256,
                strategy_sha256=candidate_sha256,
                parent_statement_sha256=statement_hash,
                parent_signature_sha256=parent.lean_signature_hash,
                root_goal_sha256=goal_hash,
                current_role="definition_auditor",
                last_transition_reason=(
                    f"resume-invalidated:{mismatch}" if mismatch
                    else "certified-decomposition-requested"
                ),
                ledger_id=ledger.ledger_id,
                ledger_version=ledger.version,
                orchestration_id=orchestration_id,
            )
            if previous_checkpoint is not None:
                for field_name in (
                    "migration_event",
                    "migration_snapshot",
                    "strategy_provider",
                    "strategy_provider_configured",
                    "strategy_model_id",
                    "strategy_run_status",
                    "strategy_agent_id",
                    "strategy_run_id",
                    "strategy_prompt_hash",
                    "strategy_evidence_hash",
                    "strategy_memo_hash",
                    "strategy_latency_ms",
                    "residency_phase",
                    "active_model",
                    "oprover_candidate_count",
                    "oprover_verified_count",
                    "critic_advisory_state",
                ):
                    setattr(
                        orchestration_checkpoint,
                        field_name,
                        getattr(previous_checkpoint, field_name),
                    )
                orchestration_checkpoint.recovery_events = [
                    *previous_checkpoint.recovery_events,
                    {
                        "event_type": "BINDING_MISMATCH_RESTART",
                        "event_id": (
                            "binding-mismatch-restart:"
                            + hashlib.sha256(mismatch.encode()).hexdigest()[:20]
                        ),
                        "reason": mismatch,
                        "audit_only_previous_artifact_hashes": sorted(
                            reference.sha256
                            for reference
                            in previous_checkpoint.validated_artifacts.values()
                        ),
                        "created_at": time.time(),
                    },
                ]
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
        else:
            if (
                orchestration_checkpoint.proof_state
                == ProofState.GENERATOR
            ):
                orchestration_checkpoint.transition(
                    ProofState.CRITIC,
                    "generator-and-critic-transcripts-bound-by-current-turn",
                    strategy_reused=True,
                )
            if orchestration_checkpoint.proof_state == ProofState.CRITIC:
                orchestration_checkpoint.transition(
                    ProofState.DEFINITION_AUDITOR,
                    "critic-requested-certified-decomposition",
                    strategy_reused=True,
                )
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
            try:
                persisted = load_validated_artifacts(
                    orchestration_checkpoint,
                )
                for role, payload in persisted.items():
                    artifact = artifact_types[role](**payload)
                    artifacts[role] = artifact
                    hashes[role] = _canonical_json_hash(payload)
                    role_run_ids[role] = (
                        orchestration_checkpoint.validated_artifacts[
                            role
                        ].source_run_id
                    )
                if persisted:
                    definition_artifact = artifacts.get("definition_auditor")
                    if (
                        definition_artifact is not None
                        and orchestration_checkpoint.proof_state not in {
                            ProofState.DECOMPOSITION_EXPLORATION,
                            ProofState.CANDIDATE_PREFILTER,
                            ProofState.CANDIDATE_FORMALIZATION,
                            ProofState.CANDIDATE_REPRESENTATION_ANALYSIS,
                            ProofState.REDUCTION_CERTIFICATION,
                        }
                    ):
                        definition_ref = (
                            orchestration_checkpoint.validated_artifacts[
                                "definition_auditor"
                            ]
                        )
                        route_definition_audit_outcome(
                            orchestration_checkpoint,
                            outcome=(
                                definition_artifact.audit_outcome
                                or (
                                    DefinitionAuditOutcomeType.MISSING_DEFINITION
                                    if definition_artifact.missing_definitions
                                    else DefinitionAuditOutcomeType.COMPLETE
                                )
                            ),
                            artifact_hash=definition_ref.sha256,
                            source_run_id=definition_ref.source_run_id,
                            missing_definition_ids=(
                                str(item.get("definition_id", ""))
                                for item in (
                                    definition_artifact.missing_definitions
                                )
                            ),
                            counterexample_objective=(
                                orchestration_checkpoint.counterexample_objective
                            ),
                        )
                    orchestration_checkpoint.strategy_reused = True
                    if not orchestration_checkpoint.resume_origin:
                        orchestration_checkpoint.resume_origin = (
                            orchestration_checkpoint.state
                        )
                    orchestration_checkpoint.last_transition_reason = (
                        "loaded-validated-upstream-artifacts"
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
                    print(
                        "[orchestration-resumed] "
                        f"state={orchestration_checkpoint.state} "
                        f"artifacts={','.join(persisted)} "
                        "strategy_reused=true",
                        flush=True,
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                retain_contract_provenance_after_artifact_failure(
                    orchestration_checkpoint,
                )
                orchestration_checkpoint.state = (
                    ProofState.DEFINITION_AUDITOR.value
                )
                orchestration_checkpoint.current_role = "definition_auditor"
                orchestration_checkpoint.last_transition_reason = (
                    f"artifact-resume-invalidated:{type(exc).__name__}"
                )
                orchestration_checkpoint.recovery_events.append({
                    "event_type": "EXECUTABLE_ARTIFACT_RESUME_INVALIDATED",
                    "event_id": hashlib.sha256(
                        (
                            type(exc).__name__
                            + orchestration_checkpoint.target_obligation_id
                            + orchestration_checkpoint.research_contract_id
                        ).encode()
                    ).hexdigest(),
                    "reason": type(exc).__name__,
                    "preserved_artifact_roles": sorted(
                        orchestration_checkpoint.validated_artifacts
                    ),
                    "research_contract_id": (
                        orchestration_checkpoint.research_contract_id
                    ),
                    "created_at": time.time(),
                })
                save_orchestration_checkpoint(
                    checkpoint_path,
                    orchestration_checkpoint,
                )
                artifacts.clear()
                hashes.clear()
                role_run_ids.clear()
    if (
        orchestration_checkpoint is not None
        and checkpoint_path is not None
        and "definition_auditor" not in artifacts
    ):
        try:
            artifact, digest, transcript, role_run_id = (
                _run_typed_definition_auditor(
                    parent,
                    root_goal,
                    run_role,
                    orchestration_id=orchestration_id,
                    checkpoint_path=checkpoint_path,
                    checkpoint=orchestration_checkpoint,
                )
            )
        except Exception as exc:
            partial_text = str(getattr(exc, "partial_text", ""))
            if partial_text:
                transcripts["definition_auditor_partial_attempt_1"] = partial_text
            failure_text = f"definition_auditor typed transport failed: {exc}"
            infrastructure_failure = "There is no Stream(" in str(exc)
            orchestration_checkpoint.adapter_blocked(
                failure_text,
                status=(
                    "INFRASTRUCTURE_BLOCKED"
                    if infrastructure_failure
                    else exc.status.value
                    if isinstance(exc, AdapterError)
                    else (
                        "ADAPTER_BLOCKED"
                    )
                ),
            )
            save_orchestration_checkpoint(
                checkpoint_path, orchestration_checkpoint,
            )
            return DecompositionCertificateResult(
                verified=False,
                errors=[failure_text],
                artifacts={},
                artifact_hashes={},
                transcripts=transcripts,
                role_run_ids={},
                validation={
                    "host_gates_passed": False,
                    "blocked": True,
                    "failure_status": orchestration_checkpoint.adapter_status,
                },
            )
        artifacts["definition_auditor"] = artifact
        hashes["definition_auditor"] = digest
        transcripts["definition_auditor"] = transcript
        role_run_ids["definition_auditor"] = role_run_id
    if orchestration_checkpoint is not None and checkpoint_path is not None:
        # migration_event is audit provenance only. Executable dispatch is
        # authorized exclusively by the complete architecture capability set.
        require_typed_dispatch(orchestration_checkpoint)
        save_orchestration_checkpoint(checkpoint_path, orchestration_checkpoint)
        return _run_typed_ir_v2(
            ledger,
            parent,
            root_goal,
            run_role,
            project_root=project_root,
            orchestration_id=orchestration_id,
            checkpoint_path=checkpoint_path,
            checkpoint=orchestration_checkpoint,
            signature_validator=signature_validator,
            proof_validator=proof_validator,
            artifacts=artifacts,
            hashes=hashes,
            role_run_ids=role_run_ids,
        )
    for role, heading in role_specs:
        if role in artifacts:
            continue
        expected_run_id = f"{orchestration_id}:{role}"
        upstream = _artifact_dependencies_for_role(role, hashes)
        if orchestration_checkpoint is not None:
            desired_state = state_for_role(role)
            if orchestration_checkpoint.proof_state != desired_state:
                orchestration_checkpoint.transition(
                    desired_state,
                    f"upstream-valid:{role}",
                    resume_origin=orchestration_checkpoint.state,
                    strategy_reused=True,
                )
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
        package = {
            "target_obligation_id": target_id,
            "parent_statement": parent.statement,
            "parent_statement_hash": statement_hash,
            "target_statement_hash": statement_hash,
            "root_goal_hash": goal_hash,
            "producer_role": role,
            "producer_run_id": expected_run_id,
            "upstream_artifact_hashes": upstream,
            "validated_upstream_artifacts": {
                name: _certified_upstream_view(
                    value,
                    consumer_role=role,
                )
                for name, value in artifacts.items()
                if name in _required_certified_upstream(role)
            },
        }
        if role == "formalizer":
            package["validated_upstream_artifacts"] = {
                "decomposer": _formalizer_upstream_view(
                    artifacts["decomposer"],
                    hashes["decomposer"],
                ),
            }
            package["definition_audit"] = _certified_upstream_view(
                artifacts["definition_auditor"],
            )
        if role == "decomposer":
            if (
                orchestration_checkpoint is not None
                and not orchestration_checkpoint.viewpoint
            ):
                viewpoint = _select_decomposer_viewpoint(
                    orchestration_checkpoint,
                    artifacts["definition_auditor"],
                )
                orchestration_checkpoint.begin_decomposition_iteration(
                    viewpoint,
                    "decomposer-search-started",
                )
                save_orchestration_checkpoint(
                    checkpoint_path,
                    orchestration_checkpoint,
                )
            package["decomposer_contract"] = _decomposer_contract(package)
            package["ancestor_hashes"] = _decomposition_ancestor_hashes(
                ledger,
                parent,
            )
            package["decomposition_iteration"] = (
                orchestration_checkpoint.decomposition_iteration
                if orchestration_checkpoint is not None else 1
            )
            package["viewpoint"] = (
                orchestration_checkpoint.viewpoint
                if orchestration_checkpoint is not None else "unassigned"
            )
            package["decomposition_novelty_ledger"] = (
                compact_decomposition_novelty_ledger(orchestration_checkpoint)
                if orchestration_checkpoint is not None else {
                    "proposal_count": 0,
                    "novel_proposals": 0,
                    "recent": [],
                }
            )
        if role in {"formalizer", "prover", "adversarial_proponent"}:
            package["parent_formal_status"] = parent.formal_status
            if parent.formal_status != "UNFORMALIZED":
                package.update({
                    "parent_lean_signature": parent.lean_signature,
                    "parent_lean_signature_hash": parent.lean_signature_hash,
                })
        if (
            role == "formalizer"
            and orchestration_checkpoint is not None
            and checkpoint_path is not None
            and getattr(run_role, "_supports_split_formalizer", False)
        ):
            artifact, unit_transcripts, split_error = _run_split_formalizer(
                run_role,
                package=package,
                project_root=project_root,
                signature_validator=signature_validator,
                checkpoint_path=checkpoint_path,
                checkpoint=orchestration_checkpoint,
                expected_run_id=expected_run_id,
            )
            transcripts.update(unit_transcripts)
            if artifact is None:
                errors.append(f"formalizer split-unit failure: {split_error}")
                break
            artifacts[role] = artifact
            hashes[role] = _canonical_json_hash(asdict(artifact))
            role_run_ids[role] = artifact.producer_run_id
            persist_validated_artifact(
                checkpoint_path,
                orchestration_checkpoint,
                role=role,
                payload=asdict(artifact),
                dependencies=upstream,
                source_run_id=artifact.producer_run_id,
            )
            orchestration_checkpoint.transition(
                ProofState.PROVER,
                "formalizer-split-units-assembled",
                source_run_id=artifact.producer_run_id,
                strategy_reused=True,
            )
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
            continue
        if role == "adversarial_proponent":
            host_validation, host_errors = _validate_decomposition_certificate(
                ledger,
                parent,
                artifacts["definition_auditor"],
                artifacts["counterexample_worker"],
                artifacts["decomposer"],
                artifacts["formalizer"],
                artifacts["prover"],
                None,
                project_root=project_root,
                signature_validator=signature_validator,
                proof_validator=proof_validator,
            )
            package["validated_artifact_hashes"] = dict(hashes)
            package["host_gate_results"] = {
                "validation": host_validation,
                "errors": host_errors,
            }
        messages = _certified_role_messages(role, heading, package)
        attempts = (
            2
            if role in {"decomposer", "formalizer", "adversarial_proponent"}
            else 1
        )
        artifact = None
        for attempt in range(attempts):
            attempt_run_id = (
                expected_run_id
                if attempt == 0
                else f"{expected_run_id}:protocol-repair-1"
            )
            try:
                text, actual_run_id = run_role(
                    role,
                    messages,
                    attempt_run_id,
                )
            except Exception as exc:
                partial_text = str(getattr(exc, "partial_text", ""))
                if partial_text:
                    transcripts[
                        f"{role}_partial_attempt_{attempt + 1}"
                    ] = partial_text
                failure = f"{type(exc).__name__}: {exc}"
                if (
                    role in {
                        "decomposer",
                        "formalizer",
                        "adversarial_proponent",
                    }
                    and attempt == 0
                    and isinstance(exc, SemanticResponseIncomplete)
                ):
                    if role == "decomposer":
                        messages = _decomposer_repair_messages(
                            package,
                            validation_errors=[failure],
                            rejected_artifact=None,
                        )
                    elif role == "formalizer":
                        messages = _formalizer_repair_messages(
                            package,
                            validation_errors=[failure],
                        )
                    else:
                        messages = _defense_repair_messages(
                            package,
                            validation_errors=[failure],
                        )
                    continue
                prefix = (
                    "decomposer protocol repair failed"
                    if role == "decomposer" and attempt
                    else (
                        "formalizer protocol repair failed "
                        f"(attempt={attempt_run_id})"
                        if role == "formalizer" and attempt
                        else (
                            "adversarial_proponent protocol repair failed "
                            f"(attempt={attempt_run_id})"
                            if role == "adversarial_proponent" and attempt
                            else f"{role} failed"
                        )
                    )
                )
                failure_text = f"{prefix}: {failure}"
                errors.append(failure_text)
                if orchestration_checkpoint is not None:
                    orchestration_checkpoint.adapter_blocked(
                        failure_text,
                        status="ADAPTER_BLOCKED",
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
                break
            transcript_key = (
                role if attempt == 0 else f"{role}_protocol_repair_1"
            )
            transcripts[transcript_key] = text
            if actual_run_id != attempt_run_id:
                errors.append(f"{role} run ID mismatch")
                break
            parsed, error = parse_certified_artifact(
                text,
                heading,
                target_obligation_id=target_id,
                parent_statement_hash=statement_hash,
                root_goal_hash=goal_hash,
                producer_run_id=attempt_run_id,
                upstream_artifact_hashes=upstream,
            )
            protocol_errors = []
            semantic_errors = []
            if role == "decomposer":
                protocol_errors.extend(_decomposer_protocol_errors(text))
            if not error and role == "decomposer":
                protocol_errors.extend(_validate_decomposition_shape(parsed))
                semantic_errors.extend(
                    _validate_definition_child_selection(
                        artifacts["definition_auditor"],
                        parsed,
                    ),
                )
                semantic_errors.extend(
                    _validate_decomposition_progress(ledger, parent, parsed),
                )
                semantic_hash = _decomposition_semantic_hash(parsed)
                structural_signature = _decomposition_structural_signature(
                    parsed,
                )
                prior_semantic = {
                    item.get("semantic_hash")
                    for item in orchestration_checkpoint.decomposition_proposals
                } if orchestration_checkpoint is not None else set()
                prior_structural = {
                    item.get("structural_signature")
                    for item in orchestration_checkpoint.decomposition_proposals
                } if orchestration_checkpoint is not None else set()
                if semantic_hash in prior_semantic:
                    semantic_errors.append(
                        "semantic-signature-duplicate after alpha normalization",
                    )
                if structural_signature in prior_structural:
                    semantic_errors.append(
                        "structural-signature-duplicate; no genuine delta",
                    )
            if not error and role == "formalizer":
                protocol_errors.extend(
                    _validate_formalization_shape(parsed, package),
                )
                if not protocol_errors:
                    protocol_errors.extend(
                        _validate_formalization_elaboration(
                            parsed,
                            package,
                            project_root=project_root,
                            signature_validator=signature_validator,
                        ),
                    )
            if not error and role == "prover":
                proof_result = proof_validator(
                    parsed.reduction_theorem_source,
                    project_root=project_root,
                )
                formalization = artifacts["formalizer"]
                if (
                    parsed.status != "PROVED"
                    or not proof_result.ok
                    or proof_result.status != "PROVED"
                ):
                    protocol_errors.append(
                        "complete reduction proof failed or is not Lean-elaborated",
                    )
                elif (
                    lean_theorem_signature_hash(parsed.reduction_theorem_source)
                    != formalization.reduction_signature_hash
                ):
                    protocol_errors.append(
                        "complete reduction proof targets another theorem",
                    )
                elif _circular_reduction_proof(
                    parsed.reduction_theorem_source,
                ):
                    protocol_errors.append(
                        "complete reduction proof circularly assumes its conclusion",
                    )
            if not error and role == "adversarial_proponent":
                protocol_errors.extend(_validate_defense_shape(parsed))
            if error and not protocol_errors:
                protocol_errors.append(error)
            if semantic_errors and not protocol_errors:
                failure_text = (
                    "decomposer semantic rejection: "
                    + "; ".join(dict.fromkeys(semantic_errors))
                )
                errors.append(failure_text)
                transcripts["decomposer_semantic_rejection"] = text
                if orchestration_checkpoint is not None:
                    archive_decomposition_rejection(
                        checkpoint_path,
                        orchestration_checkpoint,
                        proposal=asdict(parsed),
                        rejection_reasons=list(dict.fromkeys(semantic_errors)),
                        semantic_hash=semantic_hash,
                        structural_signature=structural_signature,
                        source_run_id=actual_run_id,
                    )
                    for stale_role in (
                        "decomposer",
                        "formalizer_parent_signature",
                        "formalizer_child_signature",
                        "formalizer_reduction_signature",
                        "formalizer",
                        "prover",
                        "adversarial_proponent",
                        "judge",
                    ):
                        stale = orchestration_checkpoint.validated_artifacts.pop(
                            stale_role,
                            None,
                        )
                        if stale is not None:
                            orchestration_checkpoint.invalidated_artifacts[
                                stale.sha256
                            ] = {
                                **asdict(stale),
                                "reason_codes": ["SEMANTIC_DECOMPOSITION_REJECTION"],
                                "audit_only": True,
                            }
                    next_viewpoint = _select_decomposer_viewpoint(
                        orchestration_checkpoint,
                        artifacts["definition_auditor"],
                    )
                    orchestration_checkpoint.begin_decomposition_iteration(
                        next_viewpoint,
                        failure_text,
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
                    print(
                        "[decomposer-search] "
                        f"semantic_iteration="
                        f"{orchestration_checkpoint.decomposition_iteration} "
                        f"viewpoint={next_viewpoint} "
                        "strategy_reused=true",
                        flush=True,
                    )
                break
            if protocol_errors:
                if (
                    role in {
                        "decomposer",
                        "formalizer",
                        "adversarial_proponent",
                    }
                    and attempt == 0
                ):
                    if role == "decomposer":
                        messages = _decomposer_repair_messages(
                            package,
                            validation_errors=protocol_errors,
                            rejected_artifact=_decomposer_payload(text) or None,
                        )
                    elif role == "formalizer":
                        messages = _formalizer_repair_messages(
                            package,
                            validation_errors=protocol_errors,
                        )
                    else:
                        messages = _defense_repair_messages(
                            package,
                            validation_errors=protocol_errors,
                        )
                    continue
                prefix = (
                    "decomposer protocol repair failed"
                    if role == "decomposer" and attempt
                    else (
                        "formalizer protocol repair failed "
                        f"(attempt={attempt_run_id})"
                        if role == "formalizer" and attempt
                        else (
                            "adversarial_proponent protocol repair failed "
                            f"(attempt={attempt_run_id})"
                            if role == "adversarial_proponent" and attempt
                            else role
                        )
                    )
                )
                errors.append(
                    f"{prefix}: " + "; ".join(protocol_errors),
                )
                if orchestration_checkpoint is not None:
                    failure_text = errors[-1]
                    orchestration_checkpoint.adapter_blocked(
                        failure_text,
                        status="ADAPTER_BLOCKED",
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
                break
            if role == "prover" and parsed.status != "PROVED":
                failure_text = (
                    f"prover mathematical proof failure: {parsed.status}"
                )
                errors.append(failure_text)
                if orchestration_checkpoint is not None:
                    orchestration_checkpoint.mathematical_retries += 1
                    orchestration_checkpoint.retry(
                        ProofState.PROVER,
                        failure_text,
                        protocol_retry_limit,
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
                break
            artifact = parsed
            role_run_ids[role] = actual_run_id
            break
        if artifact is None:
            break
        artifacts[role] = artifact
        hashes[role] = _canonical_json_hash(asdict(artifact))
        if orchestration_checkpoint is not None:
            persist_validated_artifact(
                checkpoint_path,
                orchestration_checkpoint,
                role=role,
                payload=asdict(artifact),
                dependencies=upstream,
                source_run_id=role_run_ids[role],
            )
            next_index = [spec[0] for spec in role_specs].index(role) + 1
            next_state = (
                state_for_role(role_specs[next_index][0])
                if next_index < len(role_specs)
                else ProofState.JUDGE
            )
            orchestration_checkpoint.transition(
                next_state,
                f"{role}-artifact-validated",
                source_run_id=role_run_ids[role],
                strategy_reused=True,
            )
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
    validation = {"host_gates_passed": False}
    if not errors and len(artifacts) == 6:
        validation, errors = _validate_decomposition_certificate(
            ledger,
            parent,
            artifacts["definition_auditor"],
            artifacts["counterexample_worker"],
            artifacts["decomposer"],
            artifacts["formalizer"],
            artifacts["prover"],
            artifacts["adversarial_proponent"],
            project_root=project_root,
            signature_validator=signature_validator,
            proof_validator=proof_validator,
        )
        if errors and orchestration_checkpoint is not None:
            failure_text = "host-validation-rejected: " + "; ".join(errors)
            if _is_decomposition_semantic_rejection(errors):
                proposal = artifacts["decomposer"]
                semantic_hash = _decomposition_semantic_hash(proposal)
                structural_signature = _decomposition_structural_signature(
                    proposal,
                )
                if not any(
                    item.get("proposal_sha256")
                    == _canonical_json_hash(asdict(proposal))
                    for item in orchestration_checkpoint.decomposition_proposals
                ):
                    archive_decomposition_rejection(
                        checkpoint_path,
                        orchestration_checkpoint,
                        proposal=asdict(proposal),
                        rejection_reasons=errors,
                        semantic_hash=semantic_hash,
                        structural_signature=structural_signature,
                        source_run_id=role_run_ids["decomposer"],
                    )
                for stale_role in (
                    "decomposer",
                    "formalizer_parent_signature",
                    "formalizer_child_signature",
                    "formalizer_reduction_signature",
                    "formalizer",
                    "prover",
                    "adversarial_proponent",
                    "judge",
                ):
                    stale = orchestration_checkpoint.validated_artifacts.pop(
                        stale_role,
                        None,
                    )
                    if stale is not None:
                        orchestration_checkpoint.invalidated_artifacts[
                            stale.sha256
                        ] = {
                            **asdict(stale),
                            "reason_codes": [
                                "SEMANTIC_DECOMPOSITION_REJECTION",
                            ],
                            "audit_only": True,
                        }
                next_viewpoint = _select_decomposer_viewpoint(
                    orchestration_checkpoint,
                    artifacts["definition_auditor"],
                )
                orchestration_checkpoint.begin_decomposition_iteration(
                    next_viewpoint,
                    failure_text,
                )
            else:
                orchestration_checkpoint.adapter_blocked(
                    failure_text,
                    status="ADAPTER_BLOCKED",
                )
            save_orchestration_checkpoint(
                checkpoint_path,
                orchestration_checkpoint,
            )
    judge_manifest = {
        "target_obligation_id": target_id,
        "parent_statement": parent.statement,
        "parent_statement_hash": statement_hash,
        "root_goal_hash": goal_hash,
        "artifact_hashes": hashes,
        "validation": validation,
        "errors": errors,
        "retained_child_statement": (
            artifacts["decomposer"].child.get("statement", "")
            if "decomposer" in artifacts else ""
        ),
    }
    manifest_hash = _canonical_json_hash(judge_manifest)
    if len(artifacts) == 6 and validation.get("host_gates_passed"):
        role = "judge"
        heading = "JUDGE_DECISION"
        expected_run_id = f"{orchestration_id}:{role}"
        package = {
            **judge_manifest,
            "producer_role": role,
            "producer_run_id": expected_run_id,
            "upstream_artifact_hashes": [manifest_hash],
            "defense_evidence": {
                "artifact_hash": hashes["adversarial_proponent"],
                "status": artifacts["adversarial_proponent"].status,
                "issues": artifacts["adversarial_proponent"].issues,
                "repairs": artifacts["adversarial_proponent"].repairs,
            },
        }
        try:
            text, actual_run_id = run_role(
                role,
                _certified_role_messages(role, heading, package),
                expected_run_id,
            )
            transcripts[role] = text
            role_run_ids[role] = actual_run_id
            judge, error = parse_certified_artifact(
                text,
                heading,
                target_obligation_id=target_id,
                parent_statement_hash=statement_hash,
                root_goal_hash=goal_hash,
                producer_run_id=expected_run_id,
                upstream_artifact_hashes=[manifest_hash],
            )
            if error:
                errors.append(error)
                if orchestration_checkpoint is not None:
                    orchestration_checkpoint.retry(
                        ProofState.JUDGE,
                        error,
                        protocol_retry_limit,
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
            else:
                artifacts[role] = judge
                hashes[role] = _canonical_json_hash(asdict(judge))
                if orchestration_checkpoint is not None:
                    persist_validated_artifact(
                        checkpoint_path,
                        orchestration_checkpoint,
                        role=role,
                        payload=asdict(judge),
                        dependencies=[manifest_hash],
                        source_run_id=actual_run_id,
                    )
                if judge.decision != "ACCEPT":
                    errors.append(f"Judge decision was {judge.decision}")
                    if orchestration_checkpoint is not None:
                        orchestration_checkpoint.retry(
                            classify_failure(role, errors[-1]),
                            errors[-1],
                            protocol_retry_limit,
                        )
                        save_orchestration_checkpoint(
                            checkpoint_path,
                            orchestration_checkpoint,
                        )
                elif orchestration_checkpoint is not None:
                    orchestration_checkpoint.transition(
                        ProofState.COMMIT,
                        "judge-accepted-host-verified-certificate",
                        source_run_id=actual_run_id,
                        strategy_reused=True,
                    )
                    save_orchestration_checkpoint(
                        checkpoint_path,
                        orchestration_checkpoint,
                    )
        except Exception as exc:
            errors.append(f"judge failed: {type(exc).__name__}: {exc}")
            if orchestration_checkpoint is not None:
                orchestration_checkpoint.retry(
                    ProofState.JUDGE,
                    errors[-1],
                    protocol_retry_limit,
                )
                save_orchestration_checkpoint(
                    checkpoint_path,
                    orchestration_checkpoint,
                )
    verified = bool(validation.get("host_gates_passed")) and not errors
    certificate_hash = _canonical_json_hash({
        "orchestration_id": orchestration_id,
        "artifact_hashes": hashes,
        "validation": validation,
    })
    if (
        verified
        and orchestration_checkpoint is not None
        and orchestration_checkpoint.proof_state == ProofState.COMMIT
    ):
        orchestration_checkpoint.commit_key = certificate_hash
        save_orchestration_checkpoint(
            checkpoint_path,
            orchestration_checkpoint,
        )
    return DecompositionCertificateResult(
        verified,
        errors,
        artifacts,
        hashes,
        transcripts,
        role_run_ids,
        validation,
        certificate_hash=certificate_hash,
    )


def persist_verified_decomposition(
    ledger: ProofObligationLedger,
    target_id: str,
    result: DecompositionCertificateResult,
    run_id: str,
) -> list[ProofObligation]:
    if not result.verified:
        return []
    parent = next(
        item for item in ledger.obligations
        if item.obligation_id == target_id
    )
    proposal: DecompositionProposal = result.artifacts["decomposer"]
    formalization: FormalizationBundle = result.artifacts["formalizer"]
    if parent.formal_status == "UNFORMALIZED":
        parent.formal_status = "FORMALIZED"
        parent.lean_signature = formalization.parent_signature_source
        parent.lean_signature_hash = formalization.parent_signature_hash
    child = proposal.child
    formal = formalization.child
    label = str(child["label"])
    child_id = (
        f"{target_id}-"
        + hashlib.sha256(
            f"{result.certificate_hash}:{label}".encode(),
        ).hexdigest()[:10]
    )
    item = ProofObligation(
        obligation_id=child_id,
        statement=str(child["statement"]),
        parent_id=target_id,
        last_run_id=run_id,
        last_evidence="Persisted from verified decomposition certificate.",
        formal_status="FORMALIZED",
        lean_signature=str(formal["lean_signature"]),
        lean_signature_hash=result.validation[
            "child_signature_hashes"
        ][label],
        decomposition_certificate_hash=result.certificate_hash,
        reduction_theorem_hash=result.validation[
            "reduction_proof_hash"
        ],
        reduction_theorem_status="PROVED",
        decomposition_role_run_ids=dict(result.role_run_ids),
        dependency_labels=[],
        dependency_ids=[],
        certificate_reversible_status="ACTIVE",
    )
    existing = next(
        (
            obligation
            for obligation in ledger.obligations
            if obligation.obligation_id == child_id
        ),
        None,
    )
    if existing is not None:
        if existing.decomposition_certificate_hash != result.certificate_hash:
            raise ValueError("child ID collision with another certificate")
        result.created = [existing]
        return [existing]
    ledger.obligations.append(item)
    created = [item]
    parent.decomposition_certificate_hash = result.certificate_hash
    parent.reduction_theorem_hash = result.validation["reduction_proof_hash"]
    parent.reduction_theorem_status = "PROVED"
    parent.decomposition_role_run_ids = dict(result.role_run_ids)
    parent.certificate_reversible_status = "ACTIVE"
    ledger.version += 1
    result.created = created
    return created


def generator_issue_coverage(
    text: str,
    obligations: list[ProofObligation],
) -> tuple[set[str], set[str]]:
    required = {item.obligation_id for item in obligations}
    covered = {
        resolved
        for model_id in _ISSUE_RESPONSE.findall(text)
        if (resolved := _resolve_model_obligation_id(model_id, required))
    }
    return covered, required - covered


def _normalized_obligation_id(value: str) -> str:
    return "-".join(
        component
        for component in re.split(r"[^a-z0-9]+", value.casefold())
        if component
    )


def _anchored_obligation_similarity(model_id: str, target_id: str) -> float:
    model = _normalized_obligation_id(model_id)
    target = _normalized_obligation_id(target_id)
    if not model or not target:
        return 0.0
    if model == target:
        return 1.0
    target_components = target.split("-")
    model_components = model.split("-")
    if len(target_components) < 2:
        return 0.0
    anchor = target_components[:2]
    suffixes = [
        "-".join(model_components[index:])
        for index in range(len(model_components) - 1)
        if model_components[index:index + 2] == anchor
    ]
    if not suffixes:
        return 0.0
    return max(
        difflib.SequenceMatcher(None, suffix, target).ratio()
        for suffix in suffixes
    )


def _resolve_model_obligation_id(
    model_id: str | None,
    allowed_ids: set[str],
) -> str:
    if not model_id:
        return next(iter(allowed_ids)) if len(allowed_ids) == 1 else ""
    if model_id in allowed_ids:
        return model_id
    normalized_model = _normalized_obligation_id(model_id)
    casefold_matches = [
        target
        for target in allowed_ids
        if _normalized_obligation_id(target) == normalized_model
    ]
    if len(casefold_matches) == 1:
        return casefold_matches[0]
    if not allowed_ids:
        return ""
    ranked = sorted(
        (
            (_anchored_obligation_similarity(model_id, target), target)
            for target in allowed_ids
        ),
        reverse=True,
    )
    best_score, best_target = ranked[0]
    if len(allowed_ids) == 1:
        return best_target if best_score >= 0.88 else ""
    second_score = ranked[1][0]
    if best_score >= 0.95 and best_score - second_score >= 0.04:
        return best_target
    return ""


def _descendant_ids(
    ledger: ProofObligationLedger,
    root_id: str,
) -> set[str]:
    descendants = {root_id}
    changed = True
    while changed:
        changed = False
        for item in ledger.obligations:
            if (
                item.obligation_id not in descendants
                and item.parent_id in descendants
            ):
                descendants.add(item.obligation_id)
                changed = True
    return descendants - {root_id}


def _mark_premise_suspected(
    ledger: ProofObligationLedger,
    target: ProofObligation,
    run_id: str,
) -> None:
    target.premise_review_status = "SUSPECTED"
    by_id = {
        item.obligation_id: item for item in ledger.obligations
    }
    for descendant_id in _descendant_ids(ledger, target.obligation_id):
        descendant = by_id[descendant_id]
        descendant.temporary_quarantine_reason = (
            f"Premise {target.obligation_id} awaits independent review."
        )
        descendant.temporary_quarantine_root_id = target.obligation_id
        descendant.temporary_quarantine_run_id = run_id


def _clear_temporary_quarantine(
    ledger: ProofObligationLedger,
    root_id: str,
) -> None:
    for item in ledger.obligations:
        if item.temporary_quarantine_root_id != root_id:
            continue
        item.temporary_quarantine_reason = ""
        item.temporary_quarantine_root_id = ""
        item.temporary_quarantine_run_id = ""


def _reverse_premise_invalidation(
    ledger: ProofObligationLedger,
    root_id: str,
    review: PremiseReview,
) -> None:
    by_id = {
        item.obligation_id: item for item in ledger.obligations
    }
    root = by_id.get(root_id)
    if root is None:
        return
    if root.invalidation_kind in {"PREMISE", "PREMISE_INVALIDATED"}:
        root.status = root.invalidation_prior_status or "UNRESOLVED"
        root.invalidation_kind = ""
        root.invalidation_prior_status = ""
    root.premise_review_status = review.status
    for descendant_id in _descendant_ids(ledger, root_id):
        descendant = by_id[descendant_id]
        if (
            descendant.status == "QUARANTINED"
            and descendant.quarantine_root_id == root_id
        ):
            descendant.status = (
                descendant.quarantine_prior_status or "UNRESOLVED"
            )
        if descendant.quarantine_root_id == root_id:
            descendant.quarantine_reason = ""
            descendant.quarantine_root_id = ""
            descendant.quarantine_run_id = ""
            descendant.quarantine_prior_status = ""
            descendant.quarantine_confidence = 0.0
            descendant.quarantine_evidence_type = ""
            descendant.quarantine_evidence_source = ""
            descendant.quarantine_auditor_run_id = ""
            descendant.quarantine_proponent_run_id = ""
            descendant.quarantine_reversible_status = "REVERSED"
    _clear_temporary_quarantine(ledger, root_id)
    for lesson in ledger.no_go_lessons:
        if (
            lesson.source_obligation_id == root_id
            and lesson.reversible_status == "ACTIVE"
        ):
            lesson.reversible_status = "REVERSED"
    ledger.backjump_target_id = ""


def _upgrade_premise_invalidation(
    ledger: ProofObligationLedger,
    target: ProofObligation,
    refuted_premise: str,
    evidence: str,
    run_id: str,
    review: PremiseReview | None,
    bound_claim_hash: str,
) -> None:
    if review is None or not review.verified or not bound_claim_hash:
        return
    if target.invalidation_kind not in {"PREMISE", "PREMISE_INVALIDATED"}:
        target.invalidation_prior_status = target.status
    target.status = "DISPROVED"
    target.invalidation_kind = "PREMISE_INVALIDATED"
    target.premise_review_status = "PREMISE_INVALIDATED"
    by_id = {
        item.obligation_id: item for item in ledger.obligations
    }
    for descendant_id in _descendant_ids(ledger, target.obligation_id):
        descendant = by_id[descendant_id]
        if (
            descendant.status == "QUARANTINED"
            and descendant.quarantine_root_id == target.obligation_id
            and descendant.quarantine_reversible_status == "ACTIVE"
        ):
            continue
        descendant.quarantine_prior_status = descendant.status
        descendant.status = "QUARANTINED"
        descendant.quarantine_reason = (
            f"Verified premise invalidation at {target.obligation_id}."
        )
        descendant.quarantine_root_id = target.obligation_id
        descendant.quarantine_run_id = run_id
        descendant.quarantine_confidence = review.confidence
        descendant.quarantine_evidence_type = review.evidence_type
        descendant.quarantine_evidence_source = review.evidence_source
        descendant.quarantine_auditor_run_id = review.auditor_run_id
        descendant.quarantine_proponent_run_id = review.proponent_run_id
        descendant.quarantine_reversible_status = "ACTIVE"
    _clear_temporary_quarantine(ledger, target.obligation_id)
    claim_hash = bound_claim_hash
    existing = next(
        (
            lesson for lesson in ledger.no_go_lessons
            if lesson.claim_hash == claim_hash
        ),
        None,
    )
    if existing is None:
        ledger.no_go_lessons.append(NoGoLesson(
            claim_hash=claim_hash,
            refuted_premise=refuted_premise,
            evidence=evidence,
            source_obligation_id=target.obligation_id,
            run_id=run_id,
            confidence=review.confidence,
            evidence_type=review.evidence_type,
            evidence_source=review.evidence_source,
            auditor_run_id=review.auditor_run_id,
            proponent_run_id=review.proponent_run_id,
            reversible_status="ACTIVE",
        ))
    else:
        existing.refuted_premise = refuted_premise
        existing.evidence = evidence
        existing.source_obligation_id = target.obligation_id
        existing.run_id = run_id
        existing.confidence = review.confidence
        existing.evidence_type = review.evidence_type
        existing.evidence_source = review.evidence_source
        existing.auditor_run_id = review.auditor_run_id
        existing.proponent_run_id = review.proponent_run_id
        existing.reversible_status = "ACTIVE"


def apply_critic_verdicts(
    ledger: ProofObligationLedger,
    critic_text: str,
    run_id: str,
    obligation_ids: set[str] | None = None,
    id_repairs: list[tuple[str, str]] | None = None,
    premise_reviews: dict[str, PremiseReview] | None = None,
) -> dict[str, str]:
    pending_ids = (
        obligation_ids
        if obligation_ids is not None
        else {
            item.obligation_id for item in pending_obligations(ledger)
        }
    )
    premise_reviews = premise_reviews or {}
    suspicions = extract_premise_suspicions(
        critic_text,
        pending_ids,
        {
            item.obligation_id: item.lean_signature_hash
            for item in ledger.obligations
            if item.obligation_id in pending_ids
        },
    )
    verdicts: dict[str, tuple[str, str, str, str]] = {}
    for match in _ISSUE_VERDICT.finditer(critic_text):
        model_id = match.group(1)
        obligation_id = _resolve_model_obligation_id(
            model_id,
            pending_ids,
        )
        if not obligation_id:
            continue
        model_id = model_id or obligation_id
        if model_id != obligation_id and id_repairs is not None:
            id_repairs.append((model_id, obligation_id))
        body = match.group("body")
        status_match = re.search(
            r"^\*{0,2}Status:\*{0,2}\s*"
            r"(PROVED|DISPROVED|UNRESOLVED)\s*$",
            body,
            re.MULTILINE,
        )
        evidence_match = re.search(
            r"^\*{0,2}Evidence:\*{0,2}\s*(.+)$",
            body,
            re.MULTILINE,
        )
        missing_match = re.search(
            r"^\*{0,2}Missing lemma:\*{0,2}\s*(.*)$",
            body,
            re.MULTILINE,
        )
        if not status_match or not evidence_match:
            continue
        status = status_match.group(1)
        evidence = evidence_match.group(1).strip()
        missing = (
            missing_match.group(1).strip().lower()
            if missing_match is not None else ""
        )
        if status == "UNRESOLVED" and not missing:
            continue
        invalidation_match = re.search(
            r"^\*{0,2}Invalidation:\*{0,2}\s*"
            r"(APPROACH|PREMISE_SUSPECTED|PREMISE)\s*$",
            body,
            re.MULTILINE,
        )
        premise_match = re.search(
            r"^\*{0,2}Premise refuted:\*{0,2}[ \t]*(.*)$",
            body,
            re.MULTILINE,
        )
        invalidation_kind = (
            invalidation_match.group(1) if invalidation_match else "APPROACH"
        )
        refuted_premise = (
            premise_match.group(1).strip() if premise_match else ""
        )
        if status in {"PROVED", "DISPROVED"} and (
            len(evidence) < 40
            or missing not in {"", "none", "(none)"}
        ):
            status = "UNRESOLVED"
            evidence = (
                "Closure rejected: proof/counterexample evidence was too "
                "short or a missing lemma remained."
            )
            invalidation_kind = ""
            refuted_premise = ""
        if status == "DISPROVED" and invalidation_kind in {
            "PREMISE",
            "PREMISE_SUSPECTED",
        }:
            suspicion = suspicions.get(obligation_id)
            review = premise_reviews.get(obligation_id)
            if suspicion is None:
                status = "UNRESOLVED"
                evidence = (
                    "Premise suspicion rejected: explicit premise, evidence "
                    "type, concrete JSON artifact, and substantial evidence "
                    "are required."
                )
                invalidation_kind = ""
                refuted_premise = ""
            elif (
                review is not None
                and review.status == "PREMISE_INVALIDATED"
                and review.verified
            ):
                invalidation_kind = "PREMISE_INVALIDATED"
                refuted_premise = suspicion.premise
            elif (
                review is not None
                and review.status in {
                    "NOT_CONFIRMED",
                    "RESCUED",
                    "INCONCLUSIVE",
                }
            ):
                status = "DISPROVED"
                invalidation_kind = "APPROACH_FAILED"
                refuted_premise = suspicion.premise
            else:
                status = "UNRESOLVED"
                invalidation_kind = "PREMISE_SUSPECTED"
                refuted_premise = suspicion.premise
        elif status == "DISPROVED":
            invalidation_kind = "APPROACH_FAILED"
        verdicts[obligation_id] = (
            status,
            evidence,
            invalidation_kind,
            refuted_premise,
        )
    applied: dict[str, str] = {}
    by_id = {
        item.obligation_id: item for item in ledger.obligations
    }
    for item in ledger.obligations:
        if (
            item.obligation_id not in pending_ids
            or item.status == "QUARANTINED"
        ):
            continue
        status, evidence, invalidation_kind, refuted_premise = verdicts.get(
            item.obligation_id,
            (
                "UNRESOLVED",
                "Critic supplied no structurally valid verdict.",
                "",
                "",
            ),
        )
        review = premise_reviews.get(item.obligation_id)
        if review is not None:
            item.premise_review_status = review.status
            item.premise_audit_confidence = review.confidence
            item.premise_audit_evidence_type = review.evidence_type
            item.premise_audit_evidence_source = review.evidence_source
            item.premise_auditor_run_id = review.auditor_run_id
            item.premise_proponent_run_id = review.proponent_run_id
            item.premise_review_reason = review.reason
        if review is not None and review.status in {"NOT_CONFIRMED", "RESCUED"}:
            _reverse_premise_invalidation(ledger, item.obligation_id, review)
        elif review is not None and review.status == "INCONCLUSIVE":
            _clear_temporary_quarantine(ledger, item.obligation_id)
        if invalidation_kind == "PREMISE_INVALIDATED":
            bound_suspicion = suspicions.get(item.obligation_id)
            _upgrade_premise_invalidation(
                ledger,
                item,
                refuted_premise,
                evidence,
                run_id,
                review,
                bound_suspicion.claim_hash if bound_suspicion else "",
            )
            status = item.status
        else:
            item.status = status
        item.last_evidence = evidence
        item.last_run_id = run_id
        item.invalidation_kind = invalidation_kind
        applied[item.obligation_id] = status
        if invalidation_kind == "PREMISE_SUSPECTED":
            _mark_premise_suspected(ledger, item, run_id)
        if invalidation_kind != "PREMISE_INVALIDATED":
            continue
        cursor = item.parent_id
        visited = set()
        ledger.backjump_target_id = ""

        def has_invalid_ancestor(candidate: ProofObligation) -> bool:
            ancestor_id = candidate.parent_id
            ancestor_visited = set()
            while ancestor_id and ancestor_id not in ancestor_visited:
                ancestor_visited.add(ancestor_id)
                ancestor = by_id.get(ancestor_id)
                if ancestor is None:
                    break
                if (
                    ancestor.status == "QUARANTINED"
                    or (
                        ancestor.status == "DISPROVED"
                        and ancestor.invalidation_kind in {
                            "PREMISE",
                            "PREMISE_INVALIDATED",
                        }
                    )
                ):
                    return True
                ancestor_id = ancestor.parent_id
            return False

        while cursor and cursor not in visited:
            visited.add(cursor)
            ancestor = by_id.get(cursor)
            if ancestor is None:
                break
            if (
                ancestor.status == "UNRESOLVED"
                and not has_invalid_ancestor(ancestor)
            ):
                ledger.backjump_target_id = ancestor.obligation_id
                break
            cursor = ancestor.parent_id
    ledger.version += 1
    return applied


def _normalize_obligation_statement(statement: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", statement.lower()))


def _lemma_signature(statement: str) -> str:
    matches = re.findall(
        r"[\"“]([^\"”]{3,120}?lemma)[\"”]",
        statement,
        re.IGNORECASE,
    )
    if not matches:
        matches = re.findall(
            r"\b([A-Za-z][A-Za-z -]{2,100}\s+Lemma)\b",
            statement,
        )
    return (
        _normalize_obligation_statement(matches[0])
        if matches else ""
    )


def _obligation_terms(statement: str) -> set[str]:
    ignored = {
        "a", "an", "and", "as", "be", "for", "if", "in", "is", "of",
        "on", "or", "that", "the", "then", "there", "to", "where", "with",
        "lemma", "proof", "formal", "given",
    }
    return {
        term
        for term in _normalize_obligation_statement(statement).split()
        if term not in ignored
    }


def _canonical_claim(statement: str) -> str:
    text = _normalize_obligation_statement(statement)
    replacements = (
        (r"\blocal accumulation rate\b|\blocal density\b", "local_density"),
        (
            r"\bglobal exponent of convergence\b|\bglobal growth order\b|"
            r"\bglobal order\b|\bgrowth order\b|\bglobal growth\b",
            "global_order",
        ),
        (
            r"\blower bound\b|\bimposes a lower bound\b|\bmust satisfy\b|"
            r"\bforces\b|\bforce\b",
            "implies_bound",
        ),
        (r"\bcritical density\b|\bdensity threshold\b", "density_threshold"),
        (r"\baccumulation\b|\bclump(?:ing)?\b", "concentration"),
        (r"\bsingularity\b|\bpole\b", "singularity"),
        (r"\bsequence of zeros\b|\bzero sequence\b", "zero_sequence"),
        (r"\bfunction\b|\bfunctional relationship\b", "mapping"),
        (r"\bequivalence\b|\bsaturation\b|\bgap\b", "relation"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    # Alpha-normalize mathematical variable names while preserving operators
    # and semantic nouns. This makes rho/delta/lambda renamings comparable.
    text = re.sub(
        r"\b(?:rho|delta|lambda|epsilon|phi|sigma|p|m|s|z|r|n)\d*\b",
        "var",
        text,
    )
    return " ".join(text.split())


def _claim_structure(statement: str) -> tuple[set[str], set[str]]:
    canonical = _canonical_claim(statement)
    markers = (
        " without implies_bound ",
        " then ",
        " implies_bound ",
        " such that ",
        " implies ",
    )
    split_at = -1
    marker_size = 0
    for marker in markers:
        position = canonical.find(marker)
        if position >= 0 and (split_at < 0 or position < split_at):
            split_at = position
            marker_size = len(marker)
    if split_at < 0:
        terms = set(canonical.split())
        return terms, terms
    premise = set(canonical[:split_at].split())
    conclusion = set(canonical[split_at + marker_size:].split())
    return premise, conclusion


def _semantic_concepts(statement: str) -> set[str]:
    text = _canonical_claim(statement)
    concepts = set()
    checks = {
        "density": ("density", "concentration"),
        "global_order": ("global_order",),
        "singularity": ("singularity", "residue"),
        "genus_or_order": ("genus", " order "),
        "convergence": ("converge", "diverge", "limit"),
        "zero_sequence": ("zero_sequence", "zeros"),
        "threshold_or_bound": ("threshold", "bound", "above", "below"),
        "local_global_relation": ("local", "global"),
    }
    padded = f" {text} "
    for concept, needles in checks.items():
        if all(needle in padded for needle in needles) if (
            concept == "local_global_relation"
        ) else any(needle in padded for needle in needles):
            concepts.add(concept)
    return concepts


def _semantic_equivalence(left: str, right: str) -> tuple[bool, float]:
    left_canonical = _canonical_claim(left)
    right_canonical = _canonical_claim(right)
    left_terms = set(left_canonical.split())
    right_terms = set(right_canonical.split())
    union = left_terms | right_terms
    jaccard = (
        len(left_terms & right_terms) / len(union)
        if union else 1.0
    )
    sequence = difflib.SequenceMatcher(
        None,
        left_canonical,
        right_canonical,
    ).ratio()
    left_premise, left_conclusion = _claim_structure(left)
    right_premise, right_conclusion = _claim_structure(right)

    def overlap(first: set[str], second: set[str]) -> float:
        denominator = max(1, min(len(first), len(second)))
        return len(first & second) / denominator

    structural = min(
        overlap(left_premise, right_premise),
        overlap(left_conclusion, right_conclusion),
    )
    left_concepts = _semantic_concepts(left)
    right_concepts = _semantic_concepts(right)
    concept_overlap = overlap(left_concepts, right_concepts)
    score = max(jaccard, sequence, structural, concept_overlap)
    equivalent = (
        (jaccard >= 0.52 and structural >= 0.60)
        or sequence >= 0.68
        or structural >= 0.78
        or (
            min(len(left_concepts), len(right_concepts)) >= 4
            and concept_overlap >= 0.80
        )
    )
    return equivalent, score


def _child_has_structural_delta(parent: str, child: str) -> bool:
    parent_premise, parent_conclusion = _claim_structure(parent)
    child_premise, child_conclusion = _claim_structure(child)
    new_premise = child_premise - parent_premise
    new_conclusion = child_conclusion - parent_conclusion
    concrete = {
        "compact", "counterexample", "exists", "fixed", "forall", "limsup",
        "neighborhood", "explicit", "boundary", "constant", "inequality",
        "converges", "diverges", "residue", "genus",
    }
    return bool(
        len(new_premise) >= 3
        or len(new_conclusion) >= 3
        or (child_premise | child_conclusion) & concrete
    )


def _frontier_rejection_reason(
    ledger: ProofObligationLedger,
    parent_id: str,
    statement: str,
) -> str:
    normalized = _normalize_obligation_statement(statement)
    terms = _obligation_terms(statement)
    signature = _lemma_signature(statement)
    parent_by_id = {
        item.obligation_id: item.parent_id
        for item in ledger.obligations
    }
    ancestors = set()
    cursor = parent_id
    while cursor and cursor not in ancestors:
        ancestors.add(cursor)
        cursor = parent_by_id.get(cursor, "")
    for lesson in ledger.no_go_lessons:
        if lesson.reversible_status != "ACTIVE":
            continue
        if hashlib.sha256(_canonical_claim(statement).encode()).hexdigest() == (
            lesson.claim_hash
        ):
            return "repeats quarantined no-go premise"
        equivalent, score = _semantic_equivalence(
            lesson.refuted_premise,
            statement,
        )
        if equivalent:
            return (
                "semantically repeats quarantined no-go premise "
                f"(score={score:.2f})"
            )
    for item in ledger.obligations:
        existing_normalized = _normalize_obligation_statement(item.statement)
        if normalized == existing_normalized:
            return f"duplicates existing obligation {item.obligation_id}"
        existing_signature = _lemma_signature(item.statement)
        if signature and signature == existing_signature:
            return f"repeats existing lemma {item.obligation_id}"
        if item.obligation_id in ancestors:
            equivalent, score = _semantic_equivalence(
                item.statement,
                statement,
            )
            if equivalent:
                return (
                    f"bidirectionally entails ancestor {item.obligation_id} "
                    f"after variable normalization (score={score:.2f})"
                )
    if len(terms) < 6:
        return "frontier is too vague to be a falsifiable smaller obligation"
    parent = next(
        item for item in ledger.obligations
        if item.obligation_id == parent_id
    )
    if not _child_has_structural_delta(parent.statement, statement):
        return (
            "does not declare a new assumption, narrower domain, or "
            "falsifiable conclusion"
        )
    for item in ledger.obligations:
        existing_terms = _obligation_terms(item.statement)
        union = terms | existing_terms
        similarity = len(terms & existing_terms) / len(union) if union else 0.0
        threshold = 0.70 if item.obligation_id in ancestors else 0.85
        if similarity >= threshold:
            return (
                f"semantically cycles to {item.obligation_id} "
                f"(similarity={similarity:.2f})"
            )
    return ""


def audit_ledger_semantic_duplicates(
    ledger: ProofObligationLedger,
) -> list[tuple[str, str, float]]:
    by_id = {
        item.obligation_id: item
        for item in ledger.obligations
    }
    rejected: list[tuple[str, str, float]] = []
    rejected_ids = set()
    for item in ledger.obligations:
        if item.status != "UNRESOLVED" or not item.parent_id:
            continue
        cursor = item.parent_id
        visited = set()
        duplicate_of = ""
        duplicate_score = 0.0
        while cursor and cursor not in visited:
            visited.add(cursor)
            ancestor = by_id.get(cursor)
            if ancestor is None:
                break
            equivalent, score = _semantic_equivalence(
                ancestor.statement,
                item.statement,
            )
            if equivalent:
                duplicate_of = ancestor.obligation_id
                duplicate_score = score
                break
            cursor = ancestor.parent_id
        if duplicate_of or item.parent_id in rejected_ids:
            duplicate_of = duplicate_of or item.parent_id
            item.status = "REJECTED_DUPLICATE"
            item.last_evidence = (
                f"Semantic duplicate/cyclic descendant of {duplicate_of}."
            )
            rejected_ids.add(item.obligation_id)
            rejected.append(
                (item.obligation_id, duplicate_of, duplicate_score),
            )
    if rejected:
        ledger.version += 1
    return rejected


def certified_decomposition_requested(
    critic_text: str,
    generator_text: str,
    target_ids: set[str],
) -> bool:
    for match in _ISSUE_VERDICT.finditer(critic_text):
        target_id = _resolve_model_obligation_id(
            match.group(1),
            target_ids,
        )
        if not target_id:
            continue
        body = match.group("body")
        if (
            _structured_field(body, "Status") == "UNRESOLVED"
            and _normalize_obligation_statement(
                _structured_field(body, "Missing lemma"),
            ) not in {"", "none", "no missing lemma"}
        ):
            return True
    for match in _ISSUE_RESPONSE.finditer(generator_text):
        target_id = _resolve_model_obligation_id(
            match.group(1),
            target_ids,
        )
        if not target_id:
            continue
        body_start = match.end()
        next_match = _ISSUE_RESPONSE.search(generator_text, body_start)
        body = generator_text[
            body_start:next_match.start() if next_match else None
        ]
        remaining = _structured_field(body, "Remaining gap")
        if _normalize_obligation_statement(remaining) not in {
            "",
            "none",
            "no remaining gap",
        }:
            return True
    return False


def create_child_obligations(
    ledger: ProofObligationLedger,
    critic_text: str,
    run_id: str,
    parent_ids: set[str],
    rejections: list[str] | None = None,
    lean_signatures: dict[
        str,
        LeanSignatureResult | list[LeanSignatureResult],
    ] | None = None,
    certified_only: bool = True,
) -> list[ProofObligation]:
    if certified_only:
        if rejections is not None:
            rejections.append(
                "free-form Missing lemma child creation is disabled; "
                "a verified decomposition certificate is required",
            )
        return []
    created: list[ProofObligation] = []
    available_signatures = {
        key: list(value) if isinstance(value, list) else [value]
        for key, value in (lean_signatures or {}).items()
    }

    def add_child(parent_id: str, statement: str, evidence: str) -> None:
        statement = statement.strip().strip("`")
        normalized = _normalize_obligation_statement(statement)
        if not normalized or normalized in {"none", "no missing lemma"}:
            return
        parent = next(
            item for item in ledger.obligations
            if item.obligation_id == parent_id
        )
        by_id = {
            item.obligation_id: item for item in ledger.obligations
        }
        cursor = parent
        visited = set()
        parent_is_sound = parent.status == "UNRESOLVED"
        while cursor.parent_id and cursor.parent_id not in visited:
            visited.add(cursor.parent_id)
            cursor = by_id.get(cursor.parent_id)
            if cursor is None:
                break
            if (
                cursor.status == "QUARANTINED"
                or (
                    cursor.status == "DISPROVED"
                    and cursor.invalidation_kind in {
                        "PREMISE",
                        "PREMISE_INVALIDATED",
                    }
                )
            ):
                parent_is_sound = False
                break
        if not parent_is_sound:
            if rejections is not None:
                rejections.append(
                    f"{statement} :: parent is not a sound unresolved obligation",
                )
            return
        rejection = _frontier_rejection_reason(
            ledger,
            parent_id,
            statement,
        )
        if rejection:
            if rejections is not None:
                rejections.append(f"{statement} :: {rejection}")
            return
        lean_results = available_signatures.get(parent_id, [])
        lean_result = lean_results.pop(0) if lean_results else None
        if lean_result is None:
            if rejections is not None:
                rejections.append(
                    f"{statement} :: missing Lean theorem signature",
                )
            return
        if not lean_result.ok:
            if rejections is not None:
                rejections.append(
                    f"{statement} :: {lean_result.error}",
                )
            return
        if any(
            item.lean_signature_hash
            and item.lean_signature_hash == lean_result.signature_hash
            for item in ledger.obligations
        ):
            if rejections is not None:
                rejections.append(
                    f"{statement} :: Lean signature duplicates an ancestor",
                )
            return
        suffix = hashlib.sha256(
            f"{parent_id}:{normalized}".encode(),
        ).hexdigest()[:10]
        child = ProofObligation(
            obligation_id=f"{parent_id}-{suffix}",
            statement=statement,
            parent_id=parent_id,
            last_run_id=run_id,
            last_evidence=evidence,
            formal_status="FORMALIZED",
            lean_signature=lean_result.source,
            lean_signature_hash=lean_result.signature_hash,
        )
        ledger.obligations.append(child)
        created.append(child)

    for match in _ISSUE_VERDICT.finditer(critic_text):
        parent_id = _resolve_model_obligation_id(
            match.group(1),
            parent_ids,
        )
        if not parent_id:
            continue
        body = match.group("body")
        status_match = re.search(
            r"^\*{0,2}Status:\*{0,2}\s*"
            r"(PROVED|DISPROVED|UNRESOLVED)\s*$",
            body,
            re.MULTILINE,
        )
        missing_match = re.search(
            r"^\*{0,2}Missing lemma:\*{0,2}\s*(.+)$",
            body,
            re.MULTILINE,
        )
        if (
            status_match is None
            or status_match.group(1) != "UNRESOLVED"
            or missing_match is None
        ):
            continue
        add_child(
            parent_id,
            missing_match.group(1),
            "Created from Critic ISSUE_VERDICT missing lemma.",
        )
    for line in critic_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [
            cell.strip().strip("*").strip()
            for cell in line.strip().strip("|").split("|")
        ]
        if len(cells) < 4 or cells[1].upper() != "UNRESOLVED":
            continue
        model_leaf_id, _, evidence, missing = cells[:4]
        parent_id = next(
            (
                parent
                for parent in sorted(parent_ids, key=len, reverse=True)
                if (
                    model_leaf_id == parent
                    or model_leaf_id.startswith(f"{parent}-")
                )
            ),
            "",
        )
        if not parent_id:
            continue
        add_child(
            parent_id,
            missing,
            f"Created from Critic leaf table: {evidence}",
        )
    if created:
        ledger.version += 1
    return created


def build_autoresearch_verdict(
    candidate,
    ledger: ProofObligationLedger,
    applied_verdicts: dict[str, str],
    created: list[ProofObligation],
    rejected_frontiers: list[str] | None = None,
) -> dict:
    target_id = str(candidate.TARGET_OBLIGATION_ID)
    target = next(
        (item for item in ledger.obligations if item.obligation_id == target_id),
        None,
    )
    status = applied_verdicts.get(target_id, "UNRESOLVED")
    invalidation_kind = target.invalidation_kind if target is not None else ""
    target_children = [
        item for item in created
        if (
            item.parent_id == target_id
            and item.decomposition_certificate_hash
            and item.reduction_theorem_status == "PROVED"
        )
    ]
    if status == "PROVED":
        outcome = "SUPPORTED"
        evidence = target.last_evidence if target is not None else (
            "The targeted proof obligation was closed by the Critic."
        )
        frontier = (
            "Integrate the proved leaf into its parent proof obligation and "
            "audit every dependency."
        )
    elif (
        status == "DISPROVED"
        and invalidation_kind in {"PREMISE", "PREMISE_INVALIDATED"}
    ):
        outcome = "FALSIFIED"
        evidence = target.last_evidence if target is not None else (
            "The targeted premise was disproved by the Critic."
        )
        backjump = next(
            (
                item for item in ledger.obligations
                if item.obligation_id == ledger.backjump_target_id
            ),
            None,
        )
        frontier = (
            "Backjump to the nearest sound unresolved obligation "
            f"{backjump.obligation_id}: {backjump.statement}"
            if backjump is not None else
            "Restart from a sound unresolved root without assuming the "
            "refuted premise or any canonical restatement of it."
        )
    elif status == "DISPROVED":
        outcome = "FALSIFIED"
        evidence = target.last_evidence if target is not None else (
            "The targeted hypothesis was disproved by the Critic."
        )
        frontier = (
            "Exclude the falsified approach and construct a distinct "
            "hypothesis for the same proof obligation."
        )
    elif target_children:
        outcome = "DECOMPOSED"
        evidence = (
            "The Critic kept the target unresolved and isolated a concrete "
            f"missing lemma: {target_children[0].statement}"
        )
        frontier = target_children[0].statement
    else:
        outcome = "INCONCLUSIVE"
        evidence = (
            f"Rejected cyclic or invalid frontier: {rejected_frontiers[0]}"
            if rejected_frontiers else (
                target.last_evidence if target is not None else
                "The Critic supplied no structurally valid target verdict."
            )
        )
        frontier = (
            target.statement if target is not None else
            "Construct a concrete smaller proof obligation."
        )
    return {
        "candidate_id": str(candidate.CANDIDATE_ID),
        "target_obligation_id": target_id,
        "outcome": outcome,
        "evidence": evidence,
        "new_frontier": frontier,
        "created_obligation_ids": [
            item.obligation_id for item in target_children
        ],
        "invalidation_kind": invalidation_kind,
        "backjump_target_id": ledger.backjump_target_id,
        "no_go_lesson_hashes": [
            lesson.claim_hash
            for lesson in ledger.no_go_lessons
            if lesson.reversible_status == "ACTIVE"
        ],
    }


def save_critic_issue_batch(path: Path, batch: CriticIssueBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(batch), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_pending_critic_issues(path: Path) -> CriticIssueBatch | None:
    if not path.exists():
        return None
    batch = CriticIssueBatch(
        **json.loads(path.read_text(encoding="utf-8")),
    )
    if (
        batch.schema_version != 1
        or not batch.issue_id
        or not batch.issues
        or any(not str(issue).strip() for issue in batch.issues)
    ):
        raise ValueError("invalid Critic issue inbox")
    return batch if batch.status == "pending" else None


def format_critic_issue_injection(batch: CriticIssueBatch) -> str:
    items = "\n".join(
        f"{index}. {issue.strip()}"
        for index, issue in enumerate(batch.issues, start=1)
    )
    return (
        "\n\nEXTERNAL RIGOROUS MATHEMATICAL ISSUES "
        f"(id={batch.issue_id}):\n{items}\n"
        "The Generator must address every issue explicitly. The Critic must "
        "verify every correction and keep unresolved items in the frontier."
    )


def consume_critic_issue_batch(
    path: Path,
    batch: CriticIssueBatch,
    run_id: str,
) -> None:
    batch.status = "consumed"
    batch.consumed_by_run = run_id
    save_critic_issue_batch(path, batch)


def parse_repl_command(raw: str, phase: ReplPhase) -> ReplCommand:
    text = raw.strip()
    lower = text.lower()
    if lower in {"/quit", "/exit"}:
        return ReplCommand("quit")
    if lower.startswith("/new"):
        goal = text[4:].strip()
        if not goal:
            raise ValueError("usage: /new <goal>")
        return ReplCommand("new", goal)
    if phase == ReplPhase.WAITING_FOR_GOAL:
        if text.startswith("/"):
            raise ValueError("set a goal with /new <goal>")
        if not text:
            raise ValueError("research goal must be non-empty")
        return ReplCommand("new", text)
    if phase == ReplPhase.RUNNING:
        raise ValueError("inference is running; input is disabled")
    if lower == "/continue":
        return ReplCommand("continue")
    if lower.startswith("/steer"):
        steering = text[6:].strip()
        if not steering:
            raise ValueError("usage: /steer <text>")
        return ReplCommand("steer", steering)
    raise ValueError(
        "command rejected; use /continue, /steer <text>, "
        "/new <goal>, or /quit",
    )


def save_checkpoint(path: Path, checkpoint: ReplCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_checkpoint(path: Path) -> ReplCheckpoint | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = ReplCheckpoint(**raw)
    if checkpoint.schema_version != 1 or not checkpoint.research_goal:
        raise ValueError("invalid Agent GAN checkpoint")
    return checkpoint


_TIMESTAMP_PREFIX = re.compile(r"^\[[^\]]+\]\s?")


def recover_checkpoint_from_log(
    log_path: Path,
    run_id: str,
) -> ReplCheckpoint:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    goal = ""
    active = False
    section = ""
    generator: list[str] = []
    critic: list[str] = []
    for raw_line in lines:
        line = _TIMESTAMP_PREFIX.sub("", raw_line, count=1)
        if line.startswith("[goal] anchored:"):
            goal = line.split(":", 1)[1].strip()
        if line.startswith("[goal] reset:"):
            goal = line.split(":", 1)[1].strip()
        if f"[inference-start]" in line and f"run={run_id}" in line:
            active = True
            section = ""
            generator = []
            critic = []
            continue
        if not active:
            continue
        if line.startswith("generator>"):
            section = "generator"
            generator.append(line.removeprefix("generator>").lstrip())
            continue
        if line.startswith("[allens] Critic Prefill:"):
            section = ""
            continue
        if line.startswith("critic>"):
            section = "critic"
            critic.append(line.removeprefix("critic>").lstrip())
            continue
        if line.startswith((
            "premise_auditor>",
            "adversarial_proponent>",
            "[premise-",
        )):
            section = ""
            continue
        if line.startswith("[metrics]"):
            break
        if section == "generator":
            generator.append(line)
        elif section == "critic":
            critic.append(line)
    generator_text = "\n".join(generator).strip()
    critic_text = "\n".join(critic).strip()
    if not goal or not generator_text or not critic_text:
        raise ValueError(f"complete run {run_id!r} not found in transcript")
    return ReplCheckpoint(
        research_goal=goal,
        previous_generator=generator_text,
        previous_critic=critic_text,
        last_run_id=run_id,
    )


def enforce_prefill_token_budget(
    stage: str,
    token_ids,
    max_tokens: int,
) -> None:
    token_count = len(token_ids)
    if token_count > max_tokens:
        raise ValueError(
            f"{stage} Prefill token budget exceeded without truncation: "
            f"{token_count} > {max_tokens}",
        )


def build_generator_messages(
    goal: str,
    *,
    steering: str = "",
    previous_generator: str = "",
    previous_critic: str = "",
    proof_ledger: str = "",
    target_obligation_id: str = "",
    proof_step_interface: str = "",
) -> list[dict[str, str]]:
    if proof_step_interface:
        if not target_obligation_id:
            raise ValueError(
                "proof-step Generator requires an exact target obligation ID",
            )
        return [
            {
                "role": "system",
                "content": (
                    "Resolve exactly one host-bound proof step. Emit exactly "
                    f"one `### ISSUE_RESPONSE {target_obligation_id}` with "
                    "Correction, Derivation, and Remaining gap. Use that exact "
                    "ID. Use at most three concise Derivation steps and keep "
                    "the complete response within 450 tokens. Do not emit a "
                    "multi-level plan or Lean."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"EXACT PROOF_STEP_INTERFACE:\n{proof_step_interface}"
                    + (
                        f"\n\nCURRENT ONE-STEP STRATEGY:\n{steering}"
                        if steering else ""
                    )
                ),
            },
        ]
    if target_obligation_id:
        previous_generator = extract_obligation_history(
            previous_generator,
            target_obligation_id,
        )
        previous_critic = extract_obligation_history(
            previous_critic,
            target_obligation_id,
        )
    feedback = ""
    if previous_generator or previous_critic:
        feedback = (
            "\n\nComplete previous Generator response:\n"
            f"{previous_generator}\n\nComplete previous Critic correction:\n"
            f"{previous_critic}\n\nApply the Critic's Next Adversarial Step "
            "while remaining anchored to the immutable goal."
        )
    steering_text = (
        f"\n\nCurrent human steering (subordinate to the goal):\n{steering}"
        if steering else ""
    )
    ledger_text = (
        f"\n\n{proof_ledger}" if proof_ledger else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "Pursue the requested mathematical argument constructively and "
                "rigorously. For an open problem, do not fabricate a proof, but "
                "do not stop at 'unsolved': identify the exact global claim, "
                "derive known reductions, recursively decompose missing proof "
                "obligations, and state the smallest unresolved frontier. "
                "Distinguish unknown from impossible."
            ),
        },
        {
            "role": "user",
            "content": (
                f"IMMUTABLE RESEARCH GOAL:\n{goal}"
                f"{feedback}{steering_text}{ledger_text}"
            ),
        },
    ]


_OBLIGATION_HISTORY_SECTION = re.compile(
    r"^### (?:ISSUE_RESPONSE|ISSUE_VERDICT)\s+(\S+)\s*$"
    r"(?P<body>.*?)(?=^### |\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_obligation_history(text: str, obligation_id: str) -> str:
    sections = [
        match.group(0).strip()
        for match in _OBLIGATION_HISTORY_SECTION.finditer(text)
        if match.group(1) == obligation_id
    ]
    return "\n\n".join(sections)


def build_critic_messages(
    goal: str,
    generator_response: str,
    *,
    steering: str = "",
    proof_ledger: str = "",
    stop_reason: str,
    complete: bool,
    proof_step_interface: str = "",
) -> list[dict[str, str]]:
    if proof_step_interface:
        return [
            {
                "role": "system",
                "content": (
                    "Audit exactly one certified proof step. Read the complete "
                    "Generator response and exact ProofStepInterface. Emit one "
                    "`### ISSUE_VERDICT <exact target ID>` with `Status: "
                    "PROVED|DISPROVED|UNRESOLVED`, `Evidence:`, and `Missing "
                    "lemma:`. For DISPROVED also emit `Invalidation: "
                    "APPROACH|PREMISE_SUSPECTED`; a suspicion must include "
                    "`Premise refuted:`, `Evidence type:`, and one-line JSON "
                    "`Evidence artifact:`. Request at most one frontier step. "
                    "Do not emit Lean, plans, scores, summaries, or blanket "
                    "approval; certified workers own decomposition and proof."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"EXACT PROOF_STEP_INTERFACE:\n{proof_step_interface}\n\n"
                    f"CURRENT CRITIC DIRECTIVE:\n{steering or '(none)'}\n\n"
                    f"COMPLETE GENERATOR RESPONSE:\n{generator_response}\n\n"
                    f"Completion: {stop_reason}; complete={complete}"
                ),
            },
        ]
    ledger_text = (
        f"\n\n{proof_ledger}" if proof_ledger else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "Act as a recursive adversarial proof analyst. Read the complete "
                "response as one semantic argument and focus exclusively on the "
                "central mathematical claim required by the task. Ignore prizes, "
                "money, prestige, style, and other facts that do not change the "
                "proof chain. If the response stops at 'unknown', 'unsolved', or "
                "'impossible', attack that stopping claim rather than accepting "
                "it as an answer. Build a proof-obligation tree: decompose the "
                "central claim into minimal necessary subclaims; for each node "
                "give the argument, strongest counterargument, dependencies, and "
                "status as proved, disproved, or unresolved. Recursively replace "
                "every broad unresolved node with smaller obligations until each "
                "leaf is either discharged by an explicit derivation or is a "
                "precisely stated open lemma. Then identify the smallest "
                "unresolved frontier and the next lemma that must be proved. "
                "Never output a numeric score or blanket approval. Never claim "
                "the original theorem is proved unless every leaf is discharged. "
                "Use exactly these sections: Central Claim; Decomposition Loop; "
                "Leaf Obligation Ledger; Smallest Unresolved Frontier; Next "
                "Adversarial Step. Do not sample, summarize, simplify, or use a "
                "fallback review."
                " Begin with `Goal Alignment: ALIGNED` or `Goal Alignment: "
                "DRIFTED`. If drifted, discard the off-topic branch and restore "
                "the proof-obligation frontier for the immutable goal. When a "
                "PROOF OBLIGATION LEDGER is present, adjudicate every pending "
                "ID with the exact ISSUE_VERDICT format before any new frontier. "
                "You may emit only `Invalidation: APPROACH` or "
                "`Invalidation: PREMISE_SUSPECTED`; never claim that one Critic "
                "response permanently invalidates a premise. A suspicion must "
                "name the premise, evidence type, and a concrete one-line JSON "
                "artifact checkable by an independent worker."
            ),
        },
        {
            "role": "user",
            "content": (
                (
                    f"EXACT PROOF_STEP_INTERFACE:\n{proof_step_interface}\n\n"
                    if proof_step_interface else
                    f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
                )
                +
                f"Current steering:\n{steering or '(none)'}\n\n"
                f"Complete response:\n{generator_response}\n\n"
                f"Completion: {stop_reason}; complete={complete}"
                f"{ledger_text}"
            ),
        },
    ]


class TokenPrinter:
    def __init__(self, tokenizer, label: str, progress_callback=None) -> None:
        self.tokenizer = tokenizer
        self.last = ""
        self.progress_callback = progress_callback
        print(f"{label}> ", end="", flush=True)

    def __call__(self, token_ids) -> None:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(text[len(self.last):], end="", flush=True)
        self.last = text
        if self.progress_callback is not None:
            self.progress_callback(len(token_ids))

    def finish(self) -> None:
        print(flush=True)


class PrefillHeartbeat:
    def __init__(
        self,
        label: str,
        interval_s: float = 30.0,
        stats_provider=None,
        progress_callback=None,
        status_interval_s: float = 5.0,
    ) -> None:
        self.label = label
        self.interval_s = interval_s
        self.stats_provider = stats_provider
        self.stop = threading.Event()
        self.started = 0.0
        self.thread = None
        self.progress_callback = progress_callback
        self.status_interval_s = min(status_interval_s, interval_s)

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=1)

    def _run(self):
        next_print = self.interval_s
        while not self.stop.wait(self.status_interval_s):
            elapsed = time.perf_counter() - self.started
            progress = ""
            total = 0
            computed = 0
            if self.stats_provider is not None:
                stats = self.stats_provider()
                total = int(stats.get("remote_job_tokens_total", 0))
                computed = int(stats.get("remote_job_tokens_computed", 0))
                if total > 0:
                    percent = min(100.0, computed / total * 100.0)
                    eta = (
                        elapsed * (total - computed) / computed
                        if computed > 0 else 0.0
                    )
                    progress = (
                        f" · {computed}/{total} tokens ({percent:.1f}%)"
                        + (f" · ETA {eta:.0f}s" if computed > 0 else "")
                    )
            if self.progress_callback is not None:
                self.progress_callback(computed, total)
            if elapsed < next_print:
                continue
            next_print += self.interval_s
            print(
                f"[allens] {self.label} Prefill: {elapsed:.0f}s{progress}",
                flush=True,
            )


def _stage(
    name: str,
    warm: dict,
    actual: dict,
    text: str,
    extra_metrics=None,
) -> dict:
    delta = actual["delta"]
    warm_route = warm.get("route_provenance")
    actual_route = actual.get("route_provenance")
    stage = {
        **actual,
        "name": f"agent_{name}",
        "agent": name,
        "round": 1,
        "hit_source": "primary_hot" if delta["local_hits"] else "unknown",
        "ok": _agent_cache_gate(
            warm["delta"],
            delta,
            warm_route=warm_route,
            actual_route=actual_route,
        ) and actual["complete"],
        "warmup_prefix_tokens": warm["prefix_tokens"],
        "warmup_tokens_reused": (
            warm["delta"]["tokens_reused"]
            if warm["delta"]["remote_jobs"] == 0 else 0
        ),
        "warmup_wall_s": warm["e2e_s"],
        "warmup_remote_jobs": warm["delta"]["remote_jobs"],
        "route_provenance": {
            "warm": warm_route,
            "actual": actual_route,
        },
        "output_chars": len(text),
        "output_hash": hashlib.sha256(text.encode()).hexdigest(),
    }
    stage.update(extra_metrics or {})
    return stage


def _gate_failure(name: str, warm: dict, actual: dict) -> RuntimeError:
    keys = (
        "local_hits",
        "remote_hits",
        "remote_jobs",
        "tokens_reused",
        "tokens_computed",
        "fallbacks",
        "remote_job_failures",
    )
    compact = lambda delta: {key: delta.get(key, 0) for key in keys}
    return RuntimeError(
        f"{name} KV gate failed: "
        f"warm={compact(warm['delta'])} actual={compact(actual['delta'])}",
    )


def _run_isolated_certified_role(
    role_name,
    messages,
    expected_run_id,
    *,
    tokenizer,
    args,
    client,
    eos_ids,
    get_stats,
    live_status,
    active_obligation_id,
    target_ids,
    telemetry_state,
    isolated_role_stages,
):
    role_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    unit_headroom = (
        FORMALIZER_UNIT_HEADROOM_TOKENS
        if str(role_name).startswith("formalizer_")
        and str(role_name).endswith("_signature")
        else 0
    )
    admit_token_ids(
        role_name,
        role_ids,
        configured_prefill_tokens=args.max_prefill_tokens,
        max_retained_tokens=args.max_retained_tokens,
    )
    role_output_cap = structured_output_cap(
        role=role_name,
        max_retained_tokens=args.max_retained_tokens,
        retained_input_tokens=len(role_ids),
        minimum_output_tokens=structured_role_minimum_output_tokens(role_name),
        configured_output_tokens=(args.max_response_tokens or None),
        control_reserve_tokens=64 + unit_headroom,
    )
    print(
        "[structured-budget] "
        f"role={role_name} retained_input={len(role_ids)} "
        f"minimum_complete_schema="
        f"{structured_role_minimum_output_tokens(role_name)} "
        f"output_cap={role_output_cap} "
        f"headroom={unit_headroom} "
        f"max_retained={args.max_retained_tokens}",
        flush=True,
    )
    print(
        f"[allens] {role_name} Prefill: {len(role_ids)} tokens...",
        flush=True,
    )
    role_key = str(role_name).lower().replace(" ", "_")
    live_status.emit(
        phase=f"{role_key}_prefill",
        role=role_key,
        state="prefill",
        progress_current=0,
        progress_total=len(role_ids),
        progress_unit="tokens",
        active_obligation_id=active_obligation_id,
        worker="allens",
        source="agent_gan_repl",
        force=True,
    )
    with PrefillHeartbeat(
        role_name,
        stats_provider=get_stats,
        progress_callback=lambda current, total: live_status.emit(
            phase=f"{role_key}_prefill",
            role=role_key,
            state="prefill",
            progress_current=current,
            progress_total=total or len(role_ids),
            progress_unit="tokens",
            active_obligation_id=active_obligation_id,
            worker="allens",
            source="agent_gan_repl",
        ),
    ):
        _, role_warm = _infer(
            client,
            eos_ids,
            role_ids,
            1,
            get_stats,
            route_binding={
                "target_id": active_obligation_id,
                "run_id": expected_run_id,
            },
            route_phase="warm",
            max_retained_tokens=args.max_retained_tokens,
        )
    live_status.emit(
        phase=f"{role_key}_decode",
        role=role_key,
        state="decode",
        progress_current=0,
        progress_total=role_output_cap,
        progress_unit="tokens",
        active_obligation_id=active_obligation_id,
        worker="primary",
        source="agent_gan_repl",
        hit_source="primary_hot",
        force=True,
    )
    role_printer = TokenPrinter(
        tokenizer,
        role_name,
        progress_callback=lambda current: live_status.emit(
            phase=f"{role_key}_decode",
            role=role_key,
            state="decode",
            progress_current=current,
            progress_total=role_output_cap,
            progress_unit="tokens",
            active_obligation_id=active_obligation_id,
            worker="primary",
            source="agent_gan_repl",
            hit_source="primary_hot",
        ),
    )
    contract_role = _message_artifact_contract(messages, role_name)
    if str(role_name).endswith("_scratchpad"):
        # Private scratchpads are untrusted prose terminated only by model EOS.
        # They are never fed to any structured adapter or proof gate.
        semantic_complete = lambda _generated: False
    else:
        semantic_complete = lambda generated: (
            _structured_transport_semantically_complete(
                tokenizer.decode(generated, skip_special_tokens=True),
                contract_role,
            )
        )
    role_tokens, role_actual = _infer(
        client,
        eos_ids,
        role_ids,
        args.output_tokens,
        get_stats,
        on_token=role_printer,
        max_response_tokens=role_output_cap,
        semantic_progress=lambda chunk: bool(
            tokenizer.decode(chunk, skip_special_tokens=True).strip()
        ),
        semantic_complete=semantic_complete,
        route_binding={
            "target_id": active_obligation_id,
            "run_id": expected_run_id,
        },
        route_phase="actual",
        max_retained_tokens=args.max_retained_tokens,
    )
    role_printer.finish()
    try:
        role_text = decode_complete_response(
            tokenizer,
            role_name,
            role_tokens,
            role_actual,
        )
    except SemanticResponseIncomplete as exc:
        exc.partial_text = tokenizer.decode(
            role_tokens,
            skip_special_tokens=True,
        )
        raise
    role_stage = _stage(
        role_name,
        role_warm,
        role_actual,
        role_text,
        extra_metrics={
            "isolated_role_session": True,
            "explicit_text_handoff_only": True,
        },
    )
    if not role_stage["ok"] and not telemetry_state["degraded"]:
        raise _gate_failure(role_name, role_warm, role_actual)
    isolated_role_stages.append(role_stage)
    live_status.emit(
        phase=f"{role_key}_complete",
        role=role_key,
        state="review",
        progress_current=1,
        progress_total=1,
        progress_unit="role",
        active_obligation_id=active_obligation_id,
        source="agent_gan_repl",
        force=True,
    )
    return (
        role_text,
        expected_run_id or (
            f"local:{role_name}:{next(iter(target_ids))}"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ssh", default="allens")
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key-file", default="~/.kakeya/network_api_key")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=6144,
        help="Hard per-stage Prefill budget; over-budget input is rejected.",
    )
    parser.add_argument(
        "--max-retained-tokens",
        type=int,
        default=2052,
        help="Hard sink+window retained-KV capacity for every model call.",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=0,
        help="Optional client response cap; 0 means generate until model EOS.",
    )
    parser.add_argument("--skip-ensure", action="store_true")
    parser.add_argument(
        "--log-file",
        default="~/.kakeya/logs/agent_gan_repl.log",
        help="Timestamped local transcript log.",
    )
    parser.add_argument(
        "--state-file",
        default="~/.kakeya/agent_gan_state.json",
        help="Private resumable Generator/Critic checkpoint.",
    )
    parser.add_argument(
        "--critic-inbox",
        default="~/.kakeya/agent_gan_critic_inbox.json",
        help="Private pending rigorous-math issues for the next turn.",
    )
    parser.add_argument(
        "--proof-ledger",
        default="~/.kakeya/agent_gan_proof_ledger.json",
        help="Private persistent mathematical proof obligations.",
    )
    parser.add_argument(
        "--premise-review-dir",
        default="~/.kakeya/premise_reviews",
        help="Private durable Auditor/Proponent transcripts and artifacts.",
    )
    parser.add_argument(
        "--decomposition-review-dir",
        default="~/.kakeya/decomposition_reviews",
        help="Private atomic seven-role decomposition manifests.",
    )
    parser.add_argument(
        "--candidate-file",
        default="",
        help="AutoResearch candidate strategy applied to this experiment.",
    )
    parser.add_argument("--recover-run", default="")
    parser.add_argument(
        "--recover-log",
        default="~/.kakeya/logs/agent_gan_repl.log",
    )
    parser.add_argument("--auto-continue", action="store_true")
    parser.add_argument(
        "--no-auto-loop",
        action="store_false",
        dest="auto_loop",
        help="Pause after each successful turn instead of continuing.",
    )
    parser.set_defaults(auto_loop=True)
    parser.add_argument(
        "--auto-loop-boundary-wait-s",
        type=float,
        default=0.5,
        help="Boundary window for an explicit command before auto-continue.",
    )
    args = parser.parse_args()
    if args.output_tokens <= 0:
        raise SystemExit("output-tokens must be > 0")
    if args.max_prefill_tokens <= 0:
        raise SystemExit("max-prefill-tokens must be > 0")
    if args.max_retained_tokens <= 0:
        raise SystemExit("max-retained-tokens must be > 0")
    if args.auto_loop_boundary_wait_s < 0:
        raise SystemExit("auto-loop-boundary-wait-s must be >= 0")
    research_candidate = None
    if args.candidate_file:
        from autoresearch.prefill.prepare import _load_candidate
        research_candidate = _load_candidate(
            Path(args.candidate_file).expanduser(),
        )
    transcript = TimestampedTee(
        sys.stdout,
        Path(args.log_file).expanduser(),
    )
    sys.stdout = transcript
    sys.stderr = transcript
    atexit.register(transcript.close_log)
    install_signal_protection()
    transcript.log_only(
        f"[session-start] pid={os.getpid()} "
        f"log={transcript.log_path}",
    )

    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    if not args.skip_ensure:
        print("[startup] ensuring Primary and allens services...", flush=True)
        _ensure_services(args.worker_ssh)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_ids = _resolve_eos_token_ids(tokenizer)
    api_key = Path(args.api_key_file).expanduser().read_text().strip()

    telemetry_state = {"degraded": False, "last_stats": {}}

    def get_stats():
        stats = _telemetry_request(f"{args.dashboard}/v1/network/prefill")
        if stats is None:
            telemetry_state["degraded"] = True
            return dict(telemetry_state["last_stats"])
        telemetry_state["last_stats"] = stats
        return stats

    state_path = Path(args.state_file).expanduser()
    critic_inbox_path = Path(args.critic_inbox).expanduser()
    proof_ledger_path = Path(args.proof_ledger).expanduser()
    premise_review_dir = Path(args.premise_review_dir).expanduser()
    decomposition_review_dir = Path(
        args.decomposition_review_dir,
    ).expanduser()
    orchestration_state_path = Path(os.environ.get(
        "KAKEYA_ORCHESTRATION_STATE_PATH",
        str(Path.home() / ".kakeya/autoresearch/proof_orchestration.json"),
    )).expanduser()
    orchestration_candidate_sha256 = os.environ.get(
        "KAKEYA_CANDIDATE_SHA256",
        "",
    )
    live_status_path = Path(os.environ.get(
        "KAKEYA_LIVE_STATUS_PATH",
        str(Path.home() / ".kakeya/proof_live_status.json"),
    )).expanduser()
    supervisor_pid = int(os.environ.get("KAKEYA_SUPERVISOR_PID", os.getppid()))
    supervisor_iteration = int(
        os.environ.get("KAKEYA_SUPERVISOR_ITERATION", "0"),
    )
    live_status = AtomicLiveStatus(
        live_status_path,
        supervisor_pid=supervisor_pid,
        iteration=supervisor_iteration,
    )
    active_obligation_id = ""
    if args.recover_run:
        recovered = recover_checkpoint_from_log(
            Path(args.recover_log).expanduser(),
            args.recover_run,
        )
        save_checkpoint(state_path, recovered)
        print(
            f"[state-recovered] run={recovered.last_run_id} "
            f"generator_chars={len(recovered.previous_generator)} "
            f"critic_chars={len(recovered.previous_critic)}",
            flush=True,
        )
    checkpoint = load_checkpoint(state_path)
    research_goal = checkpoint.research_goal if checkpoint else ""
    previous_generator = checkpoint.previous_generator if checkpoint else ""
    previous_critic = checkpoint.previous_critic if checkpoint else ""
    phase = ReplPhase.READY if checkpoint else ReplPhase.WAITING_FOR_GOAL
    auto_loop_active = bool(args.auto_loop)
    pending_input = (
        "/continue"
        if checkpoint and (args.auto_continue or auto_loop_active)
        else ""
    )
    print(
        "Kakeya Agent GAN REPL ready.\n"
        "Commands: /new <goal>, /continue, /steer <text>, /quit.\n"
        "Raw text is accepted only as the initial goal; inference output can "
        "never become implicit steering.\n"
        f"Auto-loop is {'enabled' if auto_loop_active else 'disabled'}; "
        "successful turns continue automatically and exceptions pause it.\n"
        "Each turn runs allens Prefill → Primary hot Generator → "
        "allens Prefill → Primary hot Critic.",
        flush=True,
    )
    print(f"[log] {transcript.log_path}", flush=True)
    print(f"[state] {state_path} phase={phase.value}", flush=True)
    print(f"[critic-inbox] {critic_inbox_path}", flush=True)
    print(f"[proof-ledger] {proof_ledger_path}", flush=True)
    if checkpoint:
        print(
            f"[state-restored] run={checkpoint.last_run_id or '(none)'} "
            f"goal={hashlib.sha256(research_goal.encode()).hexdigest()}",
            flush=True,
        )
    with Client(args.address) as client:
        while True:
            if pending_input:
                raw_input = pending_input
                pending_input = ""
                print(f"\nprompt> {raw_input}", flush=True)
            elif auto_loop_active and phase == ReplPhase.READY:
                print("\nprompt> ", end="", flush=True)
                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    args.auto_loop_boundary_wait_s,
                )
                if readable:
                    raw_input = input()
                else:
                    raw_input = "/continue"
                    print(raw_input, flush=True)
            else:
                try:
                    raw_input = input("\nprompt> ")
                except EOFError:
                    print("\n[bye]")
                    break
            transcript.log_only(f"[input] {raw_input or '(empty)'}")
            try:
                command = parse_repl_command(raw_input, phase)
            except ValueError as exc:
                print(f"[input-rejected] {exc}", flush=True)
                continue
            if command.action == "quit":
                print("[bye]")
                break
            if command.action == "new":
                research_goal = command.payload
                previous_generator = ""
                previous_critic = ""
                save_checkpoint(
                    state_path,
                    ReplCheckpoint(research_goal=research_goal),
                )
                phase = ReplPhase.READY
                auto_loop_active = bool(args.auto_loop)
                print(f"[goal] reset: {research_goal}", flush=True)
            steering = command.payload if command.action == "steer" else ""
            generator_steering = steering
            critic_strategy = ""
            if research_candidate is not None:
                generator_steering = "\n\n".join(filter(None, (
                    steering,
                    str(research_candidate.GENERATOR_DIRECTIVE),
                )))
                critic_strategy = str(research_candidate.CRITIC_DIRECTIVE)
            if command.action in {"continue", "steer"} and args.auto_loop:
                auto_loop_active = True
            phase = ReplPhase.RUNNING
            live_status.emit(
                phase="proof_turn_queued",
                role="orchestrator",
                state="queued",
                source="agent_gan_repl",
                force=True,
            )
            critic_issue_batch = load_pending_critic_issues(
                critic_inbox_path,
            )
            critic_issue_injection = (
                format_critic_issue_injection(critic_issue_batch)
                if critic_issue_batch is not None else ""
            )
            proof_ledger = load_proof_ledger(proof_ledger_path)
            semantic_rejections = (
                audit_ledger_semantic_duplicates(proof_ledger)
                if proof_ledger is not None else []
            )
            if proof_ledger is not None and semantic_rejections:
                save_proof_ledger(proof_ledger_path, proof_ledger)
                for obligation_id, ancestor_id, score in semantic_rejections:
                    print(
                        "[proof-obligation-retro-rejected] "
                        f"id={obligation_id} duplicate_of={ancestor_id} "
                        f"score={score:.2f}",
                        flush=True,
                    )
            turn_obligations = pending_obligations(proof_ledger)
            if research_candidate is not None and turn_obligations:
                target_id = str(research_candidate.TARGET_OBLIGATION_ID)
                turn_obligations = [
                    item for item in turn_obligations
                    if item.obligation_id == target_id
                ]
                if not turn_obligations:
                    raise ValueError(
                        f"candidate target is not an unresolved leaf: {target_id}",
                    )
            elif len(turn_obligations) > 1:
                turn_obligations = turn_obligations[:1]
            active_obligation_id = (
                turn_obligations[0].obligation_id
                if turn_obligations else ""
            )
            proof_step_interface_text = ""
            if proof_ledger is not None and len(turn_obligations) == 1:
                target = turn_obligations[0]
                by_id = {
                    item.obligation_id: item
                    for item in proof_ledger.obligations
                }
                parent = by_id.get(target.parent_id)
                interface = build_proof_step_interface(
                    root_goal_hash=hashlib.sha256(
                        research_goal.encode(),
                    ).hexdigest(),
                    target=asdict(target),
                    parent=asdict(parent) if parent is not None else None,
                    active_no_go_lessons=[
                        asdict(lesson)
                        for lesson in proof_ledger.no_go_lessons
                        if lesson.reversible_status == "ACTIVE"
                    ],
                    archive_manifest={
                        "ledger_id": proof_ledger.ledger_id,
                        "ledger_version": proof_ledger.version,
                        "ledger_sha256": _canonical_json_hash(
                            asdict(proof_ledger),
                        ),
                    },
                )
                proof_step_interface_text = (
                    serialize_proof_step_interface(interface)
                )
            proof_ledger_text = (
                format_proof_ledger(proof_ledger, turn_obligations)
                if turn_obligations else ""
            )
            if proof_ledger is not None:
                print(
                    f"[proof-ledger-loaded] id={proof_ledger.ledger_id} "
                    f"version={proof_ledger.version} "
                    f"pending={len(turn_obligations)}",
                    flush=True,
                )
                for item in turn_obligations:
                    print(
                        f"[proof-obligation-pending] "
                        f"id={item.obligation_id} {item.statement}",
                        flush=True,
                    )
            if critic_issue_batch is not None:
                print(
                    f"[critic-issue-injection] id="
                    f"{critic_issue_batch.issue_id} "
                    f"count={len(critic_issue_batch.issues)}",
                    flush=True,
                )
                for index, issue in enumerate(
                    critic_issue_batch.issues,
                    start=1,
                ):
                    print(
                        f"[critic-issue-{index}] {issue}",
                        flush=True,
                    )
            run_nonce = uuid.uuid4().hex
            telemetry_state["degraded"] = False
            resume_checkpoint = load_orchestration_checkpoint(
                orchestration_state_path,
            )
            architecture9 = bool(
                resume_checkpoint is not None
                and resume_checkpoint.architecture_version >= 9
            )
            architecture9_fresh_entry = bool(
                architecture9
                and resume_checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT
            )
            if architecture9:
                if proof_ledger is None:
                    raise RuntimeError(
                        "ARCHITECTURE9_PROOF_LEDGER_REQUIRED",
                    )
                if not turn_obligations:
                    resume_checkpoint.transition(
                        ProofState.IDLE,
                        "architecture9-no-unresolved-obligations",
                        strategy_reused=True,
                    )
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        resume_checkpoint,
                    )
                    live_status.emit(
                        phase="architecture9_idle",
                        role="orchestrator",
                        state="idle",
                        source="agent_gan_repl",
                        force=True,
                    )
                    phase = ReplPhase.READY
                    auto_loop_active = False
                    continue
                host_target = turn_obligations[0].obligation_id
                if resume_checkpoint.target_obligation_id != host_target:
                    resume_checkpoint.recovery_events.append({
                        "event_type": "ARCHITECTURE9_HOST_RETARGET",
                        "event_id": (
                            "architecture9-host-retarget:"
                            + hashlib.sha256(host_target.encode()).hexdigest()[:20]
                        ),
                        "audit_only_previous_target": (
                            resume_checkpoint.target_obligation_id
                        ),
                        "target_obligation_id": host_target,
                        "created_at": time.time(),
                    })
                    resume_checkpoint.target_obligation_id = host_target
                    resume_checkpoint.last_transition_reason = (
                        "architecture9-host-retarget-unresolved-leaf"
                    )
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        resume_checkpoint,
                    )
                if (
                    resume_checkpoint.proof_state
                    == ProofState.STRATEGY_TOURNAMENT
                ):
                    resume_checkpoint = run_architecture_v9_entry(
                        orchestration_state_path,
                        resume_checkpoint,
                        project_root=Path(__file__).resolve().parents[1],
                        target_ref=host_target,
                        parent_obligation_ref=(
                            turn_obligations[0].parent_id or "ROOT"
                        ),
                        parent_complexity=max(
                            5,
                            len(turn_obligations[0].statement.split()),
                        ),
                        event_type=StrategyEvent.INITIAL_BRANCH,
                        event_id=(
                            "INITIAL_BRANCH:"
                            + hashlib.sha256(
                                (
                                    host_target
                                    + resume_checkpoint.migration_snapshot
                                ).encode()
                            ).hexdigest()[:20]
                        ),
                    )
            run = _telemetry_request(
                f"{args.dashboard}/v1/network/benchmarks",
                api_key=api_key,
                method="POST",
                body={
                    "kind": "agent_gan_interactive",
                    "config": {
                        "model_id": "gemma-4-26B-A4B-it-mlx-4bit",
                        "topology": "primary-decode-allens-prefill",
                        "agents": ([
                            "cursor_strategy",
                            "gemma_critic",
                            "premise_auditor",
                            "definition_auditor",
                            "counterexample_worker",
                            "decomposer",
                            "math_ir_translator",
                            "host_typed_ir_gate",
                            "lean_elaboration_gate",
                            "proof_search",
                            "adversarial_proponent",
                            "judge",
                        ] if architecture9 else [
                            "generator",
                            "critic",
                            "premise_auditor",
                            "definition_auditor",
                            "counterexample_worker",
                            "decomposer",
                            "math_ir_translator",
                            "host_typed_ir_gate",
                            "lean_elaboration_gate",
                            "proof_search",
                            "adversarial_proponent",
                            "judge",
                        ]),
                        "rounds": 1,
                        "output_tokens": args.output_tokens,
                        "goal_anchor": hashlib.sha256(
                            research_goal.encode(),
                        ).hexdigest(),
                        "feedback_applied": bool(previous_critic),
                        "critic_issue_id": (
                            critic_issue_batch.issue_id
                            if critic_issue_batch is not None else ""
                        ),
                        "critic_issue_count": (
                            len(critic_issue_batch.issues)
                            if critic_issue_batch is not None else 0
                        ),
                        "proof_ledger_id": (
                            proof_ledger.ledger_id
                            if proof_ledger is not None else ""
                        ),
                        "proof_ledger_version": (
                            proof_ledger.version
                            if proof_ledger is not None else 0
                        ),
                        "proof_obligations_pending": len(turn_obligations),
                        "autoresearch_candidate_id": (
                            str(research_candidate.CANDIDATE_ID)
                            if research_candidate is not None else ""
                        ),
                        "autoresearch_target_obligation": (
                            str(research_candidate.TARGET_OBLIGATION_ID)
                            if research_candidate is not None else ""
                        ),
                    },
                },
            )
            remote_run = run is not None
            run_id = run["id"] if remote_run else f"local_{run_nonce[:16]}"
            live_status.set_context(run_id=run_id)
            started_at = datetime.now().astimezone().isoformat(
                timespec="milliseconds",
            )
            print(
                f"[inference-start] time={started_at} run={run_id} "
                f"goal={hashlib.sha256(research_goal.encode()).hexdigest()}",
                flush=True,
            )
            try:
                resumed_architecture9_role = resume_requires_certificate(
                    resume_checkpoint,
                    fresh_architecture_entry=architecture9_fresh_entry,
                )
                resume_preconditions_hold = bool(
                    proof_ledger is not None
                    and resume_checkpoint is not None
                    and any(
                        item.obligation_id
                        == resume_checkpoint.target_obligation_id
                        for item in turn_obligations
                    )
                    and (
                        not orchestration_candidate_sha256
                        or resume_checkpoint.candidate_sha256
                        == orchestration_candidate_sha256
                    )
                )
                resume_certificate_hash = ""
                if resumed_architecture9_role:
                    if not resume_preconditions_hold:
                        raise ResumeCertificateError(
                            "ARCHITECTURE9_RESUME_PRECONDITION_MISMATCH"
                        )
                    lease_id = os.environ.get("KAKEYA_RESUME_LEASE_ID", "")
                    if not lease_id:
                        raise ResumeCertificateError(
                            "ARCHITECTURE9_CERTIFIED_RESUME_REQUIRED"
                        )
                    resume_certificate_hash = consume_resume_certificate(
                        orchestration_state_path,
                        resume_checkpoint,
                        asdict(proof_ledger),
                        runtime_binding=current_runtime_binding(
                            Path(__file__).resolve().parents[1],
                            tokenizer_id=args.tokenizer_id,
                            residency_path=Path.home()
                            / ".kakeya/oprover-residency.json",
                        ),
                        intended_next_role=resume_checkpoint.current_role,
                        owner_pid=os.getpid(),
                        lease_id=lease_id,
                    )
                    print(
                        "[resume-certificate-consumed] "
                        f"sha256={resume_certificate_hash} "
                        f"role={resume_checkpoint.current_role}",
                        flush=True,
                    )
                    if (
                        resume_checkpoint.proof_state
                        == ProofState.DEFINITION_RESOLUTION
                    ):
                        resume_checkpoint, definition_outcome = (
                            dispatch_certified_architecture9_role(
                                orchestration_state_path,
                                resume_checkpoint,
                                project_root=Path(__file__).resolve().parents[1],
                                interface_strategy_adapter=(
                                    CursorStrategyAdapter()
                                ),
                            )
                        )
                        print(
                            "[definition-resolution-start] "
                            f"outcome={definition_outcome or 'NO_OPEN_QUERY'} "
                            f"state={resume_checkpoint.proof_state.value}",
                            flush=True,
                        )
                        if (
                            definition_outcome
                            == ProofState.PARENT_STATEMENT_UNDERSPECIFIED.value
                            and resume_checkpoint.definition_exhaustion_hash
                        ):
                            source_run_id = (
                                resume_checkpoint.source_run_ids[-1]
                                if resume_checkpoint.source_run_ids
                                else "host:target-interface-exhaustion"
                            )
                            if quarantine_terminal_interface_target(
                                proof_ledger,
                                target_id=resume_checkpoint.target_obligation_id,
                                exhaustion_hash=(
                                    resume_checkpoint.definition_exhaustion_hash
                                ),
                                source_run_id=source_run_id,
                            ):
                                save_proof_ledger(
                                    proof_ledger_path,
                                    proof_ledger,
                                )
                                resume_checkpoint.ledger_version = (
                                    proof_ledger.version
                                )
                                save_orchestration_checkpoint(
                                    orchestration_state_path,
                                    resume_checkpoint,
                                )
                            print(
                                "[typed-interface-terminal-quarantine] "
                                f"target={resume_checkpoint.target_obligation_id} "
                                "backjump=ROOT_UNAVAILABLE "
                                f"ledger_version={proof_ledger.version}",
                                flush=True,
                            )
                        phase = ReplPhase.READY
                        auto_loop_active = False
                        continue
                resume_certified = bool(
                    resume_preconditions_hold
                    and (
                        not architecture9
                        or not resumed_architecture9_role
                        or resume_certificate_hash
                    )
                )
                if resume_certified:
                    architecture7 = resume_checkpoint.architecture_version >= 7
                    critic_payload = {}
                    if not architecture7:
                        critic_ref = resume_checkpoint.validated_artifacts.get(
                            "critic",
                        )
                        critic_payload = load_critic_artifact(
                            asdict(critic_ref) if critic_ref is not None else {},
                            expected_bindings={
                            "target_obligation_id": (
                                resume_checkpoint.target_obligation_id
                            ),
                            "candidate_sha256": (
                                resume_checkpoint.candidate_sha256
                            ),
                            "strategy_sha256": (
                                resume_checkpoint.strategy_sha256
                            ),
                            "parent_statement_sha256": (
                                resume_checkpoint.parent_statement_sha256
                            ),
                            "parent_signature_sha256": (
                                resume_checkpoint.parent_signature_sha256
                            ),
                            "root_goal_sha256": (
                                resume_checkpoint.root_goal_sha256
                            ),
                            "ledger_id": resume_checkpoint.ledger_id,
                            "ledger_version": resume_checkpoint.ledger_version,
                            },
                        )
                    decomposition_target = (
                        resume_checkpoint.target_obligation_id
                    )
                    canonical_root_goal = next(
                        item.statement for item in proof_ledger.obligations
                        if item.obligation_id == decomposition_target
                        and not item.parent_id
                        and item.formal_status == "FORMALIZED"
                        and (
                            item.proposition_hash
                            or item.lean_signature_hash
                        ) == resume_checkpoint.proposition_hash
                    )
                    target_ids = {decomposition_target}
                    isolated_role_stages = []

                    def direct_review_role(
                        role_name,
                        messages,
                        expected_run_id="",
                    ):
                        return _run_isolated_certified_role(
                            role_name,
                            messages,
                            expected_run_id,
                            tokenizer=tokenizer,
                            args=args,
                            client=client,
                            eos_ids=eos_ids,
                            get_stats=get_stats,
                            live_status=live_status,
                            active_obligation_id=active_obligation_id,
                            target_ids=target_ids,
                            telemetry_state=telemetry_state,
                            isolated_role_stages=isolated_role_stages,
                        )

                    direct_review_role._supports_split_formalizer = True

                    orchestration_id = (
                        resume_checkpoint.orchestration_id
                        or (
                            f"{run_id}:decomposition:"
                            + hashlib.sha256(
                                decomposition_target.encode(),
                            ).hexdigest()[:12]
                        )
                    )
                    print(
                        "[orchestration-direct-resume] "
                        f"state={resume_checkpoint.state} "
                        f"target={decomposition_target} "
                        "generator_reused=true critic_reused=true "
                        "strategy_reused=true",
                        flush=True,
                    )
                    certificate = run_certified_decomposition(
                        proof_ledger,
                        decomposition_target,
                        canonical_root_goal,
                        direct_review_role,
                        project_root=Path(__file__).resolve().parents[1],
                        orchestration_id=orchestration_id,
                        checkpoint_path=orchestration_state_path,
                        candidate_sha256=orchestration_candidate_sha256,
                    )
                    created_obligations = persist_verified_decomposition(
                        proof_ledger,
                        decomposition_target,
                        certificate,
                        run_id,
                    )
                    manifest_path = decomposition_review_dir / (
                        hashlib.sha256(
                            orchestration_id.encode(),
                        ).hexdigest()[:20]
                        + ".json"
                    )
                    save_decomposition_manifest(manifest_path, {
                        "schema_version": 3,
                        "decomposition_contract": "single_child_resumable_v3",
                        "orchestration_id": orchestration_id,
                        "target_obligation_id": decomposition_target,
                        "verified": certificate.verified,
                        "certificate_hash": certificate.certificate_hash,
                        "errors": certificate.errors,
                        "artifact_hashes": certificate.artifact_hashes,
                        "artifacts": {
                            role: asdict(artifact)
                            for role, artifact in certificate.artifacts.items()
                        },
                        "validation": certificate.validation,
                        "role_run_ids": certificate.role_run_ids,
                        "transcripts": certificate.transcripts,
                        "created_obligation_ids": [
                            item.obligation_id
                            for item in created_obligations
                        ],
                        "resumed": True,
                        "resume_origin": resume_checkpoint.state,
                        "strategy_reused": not architecture7,
                        "generator_reused": not architecture7,
                        "critic_reused": not architecture7,
                    })
                    save_proof_ledger(proof_ledger_path, proof_ledger)
                    committed_checkpoint = load_orchestration_checkpoint(
                        orchestration_state_path,
                    )
                    if (
                        committed_checkpoint is not None
                        and committed_checkpoint.proof_state
                        == ProofState.COMMIT
                    ):
                        committed_checkpoint.committed = True
                        committed_checkpoint.ledger_version = (
                            proof_ledger.version
                        )
                        committed_checkpoint.transition(
                            ProofState.IDLE,
                            "direct-resume-ledger-commit-complete",
                            source_run_id=run_id,
                            strategy_reused=True,
                        )
                        save_orchestration_checkpoint(
                            orchestration_state_path,
                            committed_checkpoint,
                        )
                    if research_candidate is not None:
                        print(
                            "[autoresearch-verdict] "
                            + json.dumps(
                                build_autoresearch_verdict(
                                    research_candidate,
                                    proof_ledger,
                                    {decomposition_target: "UNRESOLVED"},
                                    created_obligations,
                                    certificate.errors,
                                ),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if remote_run:
                        final_checkpoint = (
                            load_orchestration_checkpoint(
                                orchestration_state_path,
                            )
                            or resume_checkpoint
                        )
                        provenance = (
                            build_architecture7_report_provenance(
                                resume_checkpoint,
                                final_checkpoint,
                                isolated_role_stages,
                                ledger_sha256=_canonical_json_hash(
                                    asdict(proof_ledger),
                                ),
                                environment_sha256=pinned_environment_hash(
                                    Path(__file__).resolve().parents[1],
                                ),
                            )
                            if architecture7 else
                            build_resumed_report_provenance(
                                resume_checkpoint,
                                critic_payload,
                                isolated_role_stages,
                            )
                        )
                        print(
                            "[report-provenance] "
                            + json.dumps(
                                provenance,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        _telemetry_request(
                            f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                            api_key=api_key,
                            method="PATCH",
                            body={
                                "stages": isolated_role_stages,
                                "provenance": provenance,
                                "status": "completed",
                                "finished_at": time.time(),
                            },
                        )
                    print(
                        "[inference-complete] "
                        f"time={datetime.now().astimezone().isoformat(timespec='milliseconds')} "
                        f"run={run_id} resumed=true "
                        "generator_reused=true critic_reused=true",
                        flush=True,
                    )
                    save_checkpoint(
                        state_path,
                        ReplCheckpoint(
                            research_goal=research_goal,
                            previous_generator=previous_generator,
                            previous_critic=previous_critic,
                            last_run_id=run_id,
                        ),
                    )
                    phase = ReplPhase.READY
                    continue
                generator_messages = build_generator_messages(
                    research_goal,
                    steering="\n\n".join(filter(None, (
                        generator_steering,
                        critic_issue_injection,
                    ))),
                    previous_generator=previous_generator,
                    previous_critic=previous_critic,
                    proof_ledger=proof_ledger_text,
                    target_obligation_id=(
                        turn_obligations[0].obligation_id
                        if research_candidate is not None
                        and len(turn_obligations) == 1
                        else ""
                    ),
                    proof_step_interface=proof_step_interface_text,
                )
                generator_ids = tokenizer.apply_chat_template(
                    generator_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )
                admit_token_ids(
                    "Generator",
                    generator_ids,
                    configured_prefill_tokens=args.max_prefill_tokens,
                    max_retained_tokens=args.max_retained_tokens,
                )
                critic_fixed_messages = build_critic_messages(
                    research_goal,
                    "",
                    steering="\n\n".join(filter(None, (
                        steering,
                        critic_strategy,
                        critic_issue_injection,
                    ))),
                    proof_ledger=(
                        "" if proof_step_interface_text else proof_ledger_text
                    ),
                    stop_reason="eos",
                    complete=True,
                    proof_step_interface=proof_step_interface_text,
                )
                critic_fixed_ids = tokenizer.apply_chat_template(
                    critic_fixed_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )
                generator_cap = min(
                    downstream_output_cap(
                        max_retained_tokens=args.max_retained_tokens,
                        fixed_downstream_tokens=len(generator_ids),
                        configured_output_tokens=(
                            args.max_response_tokens or None
                        ),
                    ),
                    downstream_output_cap(
                        max_retained_tokens=args.max_retained_tokens,
                        fixed_downstream_tokens=len(critic_fixed_ids),
                        configured_output_tokens=(
                            args.max_response_tokens or None
                        ),
                        control_reserve_tokens=384,
                    ),
                )
                print(
                    f"[allens] Generator Prefill: {len(generator_ids)} tokens...",
                    flush=True,
                )
                live_status.emit(
                    phase="generator_prefill",
                    role="generator",
                    state="prefill",
                    progress_current=0,
                    progress_total=len(generator_ids),
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    worker="allens",
                    source="agent_gan_repl",
                    force=True,
                )
                with PrefillHeartbeat(
                    "Generator",
                    stats_provider=get_stats,
                    progress_callback=lambda current, total: live_status.emit(
                        phase="generator_prefill",
                        role="generator",
                        state="prefill",
                        progress_current=current,
                        progress_total=total or len(generator_ids),
                        progress_unit="tokens",
                        active_obligation_id=active_obligation_id,
                        worker="allens",
                        source="agent_gan_repl",
                    ),
                ):
                    _, generator_warm = _infer(
                        client, eos_ids, generator_ids, 1, get_stats,
                        max_retained_tokens=args.max_retained_tokens,
                    )
                live_status.emit(
                    phase="generator_decode",
                    role="generator",
                    state="decode",
                    progress_current=0,
                    progress_total=generator_cap,
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    worker="primary",
                    source="agent_gan_repl",
                    hit_source="primary_hot",
                    force=True,
                )
                generator_printer = TokenPrinter(
                    tokenizer,
                    "generator",
                    progress_callback=lambda current: live_status.emit(
                        phase="generator_decode",
                        role="generator",
                        state="decode",
                        progress_current=current,
                        progress_total=generator_cap,
                        progress_unit="tokens",
                        active_obligation_id=active_obligation_id,
                        worker="primary",
                        source="agent_gan_repl",
                        hit_source="primary_hot",
                    ),
                )
                generator_tokens, generator_actual = _infer(
                    client,
                    eos_ids,
                    generator_ids,
                    args.output_tokens,
                    get_stats,
                    on_token=generator_printer,
                    max_response_tokens=generator_cap,
                    semantic_progress=lambda chunk: bool(
                        tokenizer.decode(
                            chunk,
                            skip_special_tokens=True,
                        ).strip()
                    ),
                    max_retained_tokens=args.max_retained_tokens,
                )
                generator_printer.finish()
                generator_text = decode_complete_response(
                    tokenizer,
                    "Generator",
                    generator_tokens,
                    generator_actual,
                )
                orchestration_checkpoint = load_orchestration_checkpoint(
                    orchestration_state_path,
                )
                if (
                    orchestration_checkpoint is not None
                    and orchestration_checkpoint.proof_state
                    == ProofState.GENERATOR
                ):
                    orchestration_checkpoint.transition(
                        ProofState.CRITIC,
                        "generator-output-validated",
                        source_run_id=run_id,
                        strategy_reused=True,
                    )
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        orchestration_checkpoint,
                    )
                live_status.emit(
                    phase="generator_complete",
                    role="generator",
                    state="review",
                    progress_current=len(generator_tokens),
                    progress_total=len(generator_tokens),
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    source="agent_gan_repl",
                    hit_source="primary_hot",
                    force=True,
                )
                covered_issues, missing_issues = generator_issue_coverage(
                    generator_text,
                    turn_obligations,
                )
                if turn_obligations:
                    print(
                        f"[generator-issue-coverage] "
                        f"covered={len(covered_issues)}/"
                        f"{len(turn_obligations)} "
                        f"missing={','.join(sorted(missing_issues)) or '(none)'}",
                        flush=True,
                    )
                generator_stage = _stage(
                    "generator",
                    generator_warm,
                    generator_actual,
                    generator_text,
                )
                if not generator_stage["ok"] and not telemetry_state["degraded"]:
                    raise _gate_failure(
                        "Generator",
                        generator_warm,
                        generator_actual,
                    )
                if remote_run:
                    _telemetry_request(
                        f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                        api_key=api_key,
                        method="PATCH",
                        body={"stages": [generator_stage]},
                    )

                critic_context, context_metrics = build_critic_context(
                    tokenizer,
                    generator_text,
                    protocol="goal_anchored_recursive_gan_v3",
                )
                if (
                    critic_context != generator_text
                    or context_metrics["critic_omitted_tokens"] != 0
                    or context_metrics["review_scope"] != "full"
                ):
                    raise RuntimeError("Critic full-context invariant violated")
                critic_messages = build_critic_messages(
                    research_goal,
                    critic_context,
                    steering="\n\n".join(filter(None, (
                        steering,
                        critic_strategy,
                        critic_issue_injection,
                    ))),
                    proof_ledger=(
                        "" if proof_step_interface_text else proof_ledger_text
                    ) + (
                        "\nGENERATOR COVERAGE FAILURE: missing "
                        + ", ".join(sorted(missing_issues))
                        if missing_issues else ""
                    ),
                    stop_reason=generator_actual["stop_reason"],
                    complete=generator_actual["complete"],
                    proof_step_interface=proof_step_interface_text,
                )
                critic_ids = tokenizer.apply_chat_template(
                    critic_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )
                admit_token_ids(
                    "Critic",
                    critic_ids,
                    configured_prefill_tokens=args.max_prefill_tokens,
                    max_retained_tokens=args.max_retained_tokens,
                )
                critic_output_cap = downstream_output_cap(
                    max_retained_tokens=args.max_retained_tokens,
                    fixed_downstream_tokens=len(critic_ids),
                    configured_output_tokens=(
                        args.max_response_tokens or None
                    ),
                )
                if critic_output_cap < 320:
                    raise SemanticUnitTooLarge(
                        "Critic structured response reserve",
                        len(critic_ids) + 320,
                        args.max_retained_tokens,
                    )
                print(
                    f"[allens] Critic Prefill: {len(critic_ids)} tokens...",
                    flush=True,
                )
                live_status.emit(
                    phase="critic_prefill",
                    role="critic",
                    state="prefill",
                    progress_current=0,
                    progress_total=len(critic_ids),
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    worker="allens",
                    source="agent_gan_repl",
                    force=True,
                )
                with PrefillHeartbeat(
                    "Critic",
                    stats_provider=get_stats,
                    progress_callback=lambda current, total: live_status.emit(
                        phase="critic_prefill",
                        role="critic",
                        state="prefill",
                        progress_current=current,
                        progress_total=total or len(critic_ids),
                        progress_unit="tokens",
                        active_obligation_id=active_obligation_id,
                        worker="allens",
                        source="agent_gan_repl",
                    ),
                ):
                    _, critic_warm = _infer(
                        client, eos_ids, critic_ids, 1, get_stats,
                        max_retained_tokens=args.max_retained_tokens,
                    )
                live_status.emit(
                    phase="critic_decode",
                    role="critic",
                    state="decode",
                    progress_current=0,
                    progress_total=critic_output_cap,
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    worker="primary",
                    source="agent_gan_repl",
                    hit_source="primary_hot",
                    force=True,
                )
                critic_printer = TokenPrinter(
                    tokenizer,
                    "critic",
                    progress_callback=lambda current: live_status.emit(
                        phase="critic_decode",
                        role="critic",
                        state="decode",
                        progress_current=current,
                        progress_total=critic_output_cap,
                        progress_unit="tokens",
                        active_obligation_id=active_obligation_id,
                        worker="primary",
                        source="agent_gan_repl",
                        hit_source="primary_hot",
                    ),
                )
                critic_tokens, critic_actual = _infer(
                    client,
                    eos_ids,
                    critic_ids,
                    args.output_tokens,
                    get_stats,
                    on_token=critic_printer,
                    max_response_tokens=critic_output_cap,
                    semantic_progress=lambda chunk: bool(
                        tokenizer.decode(
                            chunk,
                            skip_special_tokens=True,
                        ).strip()
                    ),
                    max_retained_tokens=args.max_retained_tokens,
                )
                critic_printer.finish()
                critic_text = decode_complete_response(
                    tokenizer,
                    "Critic",
                    critic_tokens,
                    critic_actual,
                )
                live_status.emit(
                    phase="critic_complete",
                    role="critic",
                    state="review",
                    progress_current=len(critic_tokens),
                    progress_total=len(critic_tokens),
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    source="agent_gan_repl",
                    hit_source="primary_hot",
                    force=True,
                )
                critic_stage = _stage(
                    "critic",
                    critic_warm,
                    critic_actual,
                    critic_text,
                    extra_metrics={
                        **context_metrics,
                        "proof_ledger_id": (
                            proof_ledger.ledger_id
                            if proof_ledger is not None else ""
                        ),
                        "proof_obligations_total": len(turn_obligations),
                        "proof_obligations_covered": len(covered_issues),
                        "proof_obligations_unresolved": (
                            len(pending_obligations(proof_ledger))
                            if proof_ledger is not None else 0
                        ),
                    },
                )
                if not critic_stage["ok"] and not telemetry_state["degraded"]:
                    raise _gate_failure("Critic", critic_warm, critic_actual)
                orchestration_checkpoint = load_orchestration_checkpoint(
                    orchestration_state_path,
                )
                if (
                    orchestration_checkpoint is not None
                    and orchestration_checkpoint.proof_state == ProofState.CRITIC
                ):
                    generator_output_sha256 = str(
                        generator_stage.get("output_hash", ""),
                    )
                    critic_payload = critic_artifact_payload(
                        critic_stage,
                        source_run_id=run_id,
                        target_obligation_id=(
                            orchestration_checkpoint.target_obligation_id
                        ),
                        candidate_sha256=(
                            orchestration_checkpoint.candidate_sha256
                        ),
                        strategy_sha256=(
                            orchestration_checkpoint.strategy_sha256
                        ),
                        parent_statement_sha256=(
                            orchestration_checkpoint.parent_statement_sha256
                        ),
                        parent_signature_sha256=(
                            orchestration_checkpoint.parent_signature_sha256
                        ),
                        root_goal_sha256=(
                            orchestration_checkpoint.root_goal_sha256
                        ),
                        ledger_id=orchestration_checkpoint.ledger_id,
                        ledger_version=orchestration_checkpoint.ledger_version,
                        generator_output_sha256=generator_output_sha256,
                    )
                    critic_ref = persist_validated_artifact(
                        orchestration_state_path,
                        orchestration_checkpoint,
                        role="critic",
                        payload=critic_payload,
                        dependencies=[
                            orchestration_checkpoint.candidate_sha256,
                            orchestration_checkpoint.parent_statement_sha256,
                            orchestration_checkpoint.parent_signature_sha256,
                            orchestration_checkpoint.root_goal_sha256,
                            generator_output_sha256,
                        ],
                        source_run_id=run_id,
                    )
                    print(
                        "[critic-artifact-validated] "
                        f"sha256={critic_ref.sha256} run={run_id}",
                        flush=True,
                    )
                applied_verdicts = {}
                created_obligations = []
                id_repairs = []
                rejected_frontiers = []
                isolated_role_stages = []
                premise_reviews = {}
                if proof_ledger is not None and turn_obligations:
                    target_ids = {
                        item.obligation_id
                        for item in turn_obligations
                    }
                    suspicions = extract_premise_suspicions(
                        critic_text,
                        target_ids,
                        {
                            item.obligation_id: item.lean_signature_hash
                            for item in turn_obligations
                        },
                    )
                    if suspicions:
                        by_obligation_id = {
                            item.obligation_id: item
                            for item in proof_ledger.obligations
                        }
                        for suspicion in suspicions.values():
                            suspected_target = by_obligation_id[
                                suspicion.obligation_id
                            ]
                            suspected_target.invalidation_kind = (
                                "PREMISE_SUSPECTED"
                            )
                            suspected_target.last_evidence = (
                                suspicion.critic_evidence
                            )
                            suspected_target.last_run_id = run_id
                            _mark_premise_suspected(
                                proof_ledger,
                                suspected_target,
                                run_id,
                            )
                        proof_ledger.version += 1
                        save_proof_ledger(
                            proof_ledger_path,
                            proof_ledger,
                        )
                        print(
                            "[premise-suspicion-checkpoint] "
                            f"count={len(suspicions)} run={run_id}",
                            flush=True,
                        )

                    def run_review_role(
                        role_name,
                        messages,
                        expected_run_id="",
                    ):
                        role_ids = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=False,
                            enable_thinking=False,
                        )
                        admit_token_ids(
                            role_name,
                            role_ids,
                            configured_prefill_tokens=args.max_prefill_tokens,
                            max_retained_tokens=args.max_retained_tokens,
                        )
                        minimum_role_output = (
                            structured_role_minimum_output_tokens(role_name)
                        )
                        unit_headroom = (
                            FORMALIZER_UNIT_HEADROOM_TOKENS
                            if str(role_name).startswith("formalizer_")
                            and str(role_name).endswith("_signature")
                            else 0
                        )
                        role_output_cap = structured_output_cap(
                            role=role_name,
                            max_retained_tokens=args.max_retained_tokens,
                            retained_input_tokens=len(role_ids),
                            minimum_output_tokens=minimum_role_output,
                            configured_output_tokens=(
                                args.max_response_tokens or None
                            ),
                            control_reserve_tokens=64 + unit_headroom,
                        )
                        print(
                            "[structured-budget] "
                            f"role={role_name} retained_input={len(role_ids)} "
                            f"minimum_complete_schema={minimum_role_output} "
                            f"output_cap={role_output_cap} "
                            f"headroom={unit_headroom} "
                            f"max_retained={args.max_retained_tokens}",
                            flush=True,
                        )
                        print(
                            f"[allens] {role_name} Prefill: "
                            f"{len(role_ids)} tokens...",
                            flush=True,
                        )
                        role_key = str(role_name).lower().replace(" ", "_")
                        live_status.emit(
                            phase=f"{role_key}_prefill",
                            role=role_key,
                            state="prefill",
                            progress_current=0,
                            progress_total=len(role_ids),
                            progress_unit="tokens",
                            active_obligation_id=active_obligation_id,
                            worker="allens",
                            source="agent_gan_repl",
                            force=True,
                        )
                        with PrefillHeartbeat(
                            role_name,
                            stats_provider=get_stats,
                            progress_callback=lambda current, total: (
                                live_status.emit(
                                    phase=f"{role_key}_prefill",
                                    role=role_key,
                                    state="prefill",
                                    progress_current=current,
                                    progress_total=total or len(role_ids),
                                    progress_unit="tokens",
                                    active_obligation_id=active_obligation_id,
                                    worker="allens",
                                    source="agent_gan_repl",
                                )
                            ),
                        ):
                            _, role_warm = _infer(
                                client,
                                eos_ids,
                                role_ids,
                                1,
                                get_stats,
                                client_label=f"agent-gan-{role_name}-warm",
                                max_retained_tokens=args.max_retained_tokens,
                            )
                        live_status.emit(
                            phase=f"{role_key}_decode",
                            role=role_key,
                            state="decode",
                            progress_current=0,
                            progress_total=role_output_cap,
                            progress_unit="tokens",
                            active_obligation_id=active_obligation_id,
                            worker="primary",
                            source="agent_gan_repl",
                            hit_source="primary_hot",
                            force=True,
                        )
                        role_printer = TokenPrinter(
                            tokenizer,
                            role_name,
                            progress_callback=lambda current: live_status.emit(
                                phase=f"{role_key}_decode",
                                role=role_key,
                                state="decode",
                                progress_current=current,
                                progress_total=role_output_cap,
                                progress_unit="tokens",
                                active_obligation_id=active_obligation_id,
                                worker="primary",
                                source="agent_gan_repl",
                                hit_source="primary_hot",
                            ),
                        )
                        contract_role = _message_artifact_contract(
                            messages,
                            role_name,
                        )
                        semantic_complete = lambda generated: (
                            _structured_transport_semantically_complete(
                                tokenizer.decode(
                                    generated,
                                    skip_special_tokens=True,
                                ),
                                contract_role,
                            )
                        )
                        role_tokens, role_actual = _infer(
                            client,
                            eos_ids,
                            role_ids,
                            args.output_tokens,
                            get_stats,
                            on_token=role_printer,
                            max_response_tokens=role_output_cap,
                            semantic_progress=lambda chunk: bool(
                                tokenizer.decode(
                                    chunk,
                                    skip_special_tokens=True,
                                ).strip()
                            ),
                            semantic_complete=semantic_complete,
                            client_label=f"agent-gan-{role_name}",
                            max_retained_tokens=args.max_retained_tokens,
                        )
                        role_printer.finish()
                        try:
                            role_text = decode_complete_response(
                                tokenizer,
                                role_name,
                                role_tokens,
                                role_actual,
                            )
                        except SemanticResponseIncomplete as exc:
                            exc.partial_text = tokenizer.decode(
                                role_tokens,
                                skip_special_tokens=True,
                            )
                            raise
                        role_stage = _stage(
                            role_name,
                            role_warm,
                            role_actual,
                            role_text,
                            extra_metrics={
                                "isolated_role_session": True,
                                "explicit_text_handoff_only": True,
                            },
                        )
                        if (
                            not role_stage["ok"]
                            and not telemetry_state["degraded"]
                        ):
                            raise _gate_failure(
                                role_name,
                                role_warm,
                                role_actual,
                            )
                        isolated_role_stages.append(role_stage)
                        live_status.emit(
                            phase=f"{role_key}_complete",
                            role=role_key,
                            state="review",
                            progress_current=1,
                            progress_total=1,
                            progress_unit="role",
                            active_obligation_id=active_obligation_id,
                            source="agent_gan_repl",
                            force=True,
                        )
                        return (
                            role_text,
                            expected_run_id or (
                                f"{run_id}:{role_name}:"
                                f"{next(iter(target_ids))}"
                            ),
                        )

                    run_review_role._supports_split_formalizer = True

                    for current_suspicion in suspicions.values():
                        print(
                            "[premise-suspected] "
                            f"id={current_suspicion.obligation_id} "
                            f"type={current_suspicion.evidence_type}",
                            flush=True,
                        )
                        audit, defense, transcripts = (
                            run_isolated_premise_review(
                                research_goal,
                                current_suspicion,
                                run_review_role,
                            )
                        )
                        review = decide_premise_review(
                            audit,
                            defense,
                            project_root=Path(__file__).resolve().parents[1],
                            suspicion=current_suspicion,
                        )
                        if (
                            missing_issues
                            and review.status == "PREMISE_INVALIDATED"
                        ):
                            review = PremiseReview(
                                "INCONCLUSIVE",
                                False,
                                confidence=review.confidence,
                                evidence_type=review.evidence_type,
                                evidence_source=review.evidence_source,
                                auditor_run_id=review.auditor_run_id,
                                proponent_run_id=review.proponent_run_id,
                                reason=(
                                    "Generator issue coverage was incomplete; "
                                    "permanent invalidation is forbidden."
                                ),
                            )
                        premise_reviews[
                            current_suspicion.obligation_id
                        ] = review
                        artifact_payload = {
                            "schema_version": 1,
                            "benchmark_run_id": run_id,
                            "suspicion": asdict(current_suspicion),
                            "audit": asdict(audit) if audit else None,
                            "defense": asdict(defense) if defense else None,
                            "decision": asdict(review),
                            "transcripts": transcripts,
                        }
                        premise_review_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        review_key = hashlib.sha256(
                            current_suspicion.obligation_id.encode(),
                        ).hexdigest()[:16]
                        artifact_path = premise_review_dir / (
                            f"{run_id}-{review_key}.json"
                        )
                        temporary_artifact = artifact_path.with_suffix(
                            ".json.tmp",
                        )
                        temporary_artifact.write_text(
                            json.dumps(
                                artifact_payload,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        os.chmod(temporary_artifact, 0o600)
                        temporary_artifact.replace(artifact_path)
                        print(
                            "[premise-review] "
                            f"id={current_suspicion.obligation_id} "
                            f"status={review.status} "
                            f"verified={review.verified} "
                            f"artifact={artifact_path}",
                            flush=True,
                        )
                    applied_verdicts = apply_critic_verdicts(
                        proof_ledger,
                        critic_text,
                        run_id,
                        {
                            item.obligation_id
                            for item in turn_obligations
                        },
                        id_repairs,
                        premise_reviews,
                    )
                    # The ledger commit is authoritative. Persist the complete
                    # Critic/Auditor/Proponent decision before moving the
                    # orchestration checkpoint out of PREMISE_AUDIT.
                    save_proof_ledger(proof_ledger_path, proof_ledger)
                    typed_checkpoint = load_orchestration_checkpoint(
                        orchestration_state_path,
                    )
                    if (
                        typed_checkpoint is not None
                        and typed_checkpoint.proof_state
                        == ProofState.PREMISE_AUDIT
                    ):
                        reconcile_checkpoint_ledger_version(
                            typed_checkpoint,
                            proof_ledger.version,
                        )
                        typed_target = next(
                            (
                                item for item in turn_obligations
                                if item.obligation_id
                                == typed_checkpoint.target_obligation_id
                            ),
                            None,
                        )
                        typed_kind = (
                            typed_target.invalidation_kind
                            if typed_target is not None else ""
                        )
                        outcome_type = {
                            "APPROACH_FAILED": (
                                PremiseAuditOutcomeType.APPROACH_FAILED
                            ),
                            "PREMISE_SUSPECTED": (
                                PremiseAuditOutcomeType.PREMISE_SUSPECTED
                            ),
                            "PREMISE_INVALIDATED": (
                                PremiseAuditOutcomeType.PREMISE_INVALIDATED
                            ),
                        }.get(typed_kind)
                        if outcome_type is not None and typed_target is not None:
                            review = premise_reviews.get(
                                typed_target.obligation_id
                            )
                            apply_typed_premise_outcome(
                                orchestration_state_path,
                                typed_checkpoint,
                                outcome_type=outcome_type,
                                decision=(
                                    review.status
                                    if review is not None
                                    else typed_kind
                                ),
                                owner=(
                                    "adversarial_proponent"
                                    if review is not None
                                    and review.proponent_run_id
                                    else (
                                        "premise_auditor"
                                        if review is not None
                                        and review.auditor_run_id
                                        else "critic"
                                    )
                                ),
                                confidence=(
                                    review.confidence
                                    if review is not None else 1.0
                                ),
                                evidence={
                                    "critic_evidence": (
                                        typed_target.last_evidence
                                    ),
                                    "critic_invalidation": typed_kind,
                                    "premise_review": (
                                        asdict(review)
                                        if review is not None else None
                                    ),
                                },
                                source_run_id=run_id,
                                backjump_target=(
                                    proof_ledger.backjump_target_id
                                ),
                            )
                    if missing_issues:
                        rejected_frontiers.append(
                            "Generator coverage incomplete; child creation "
                            f"forbidden for {','.join(sorted(missing_issues))}",
                        )
                    elif (
                        certified_decomposition_requested(
                            critic_text,
                            generator_text,
                            target_ids,
                        )
                        and any(
                            applied_verdicts.get(target_id) == "UNRESOLVED"
                            for target_id in target_ids
                        )
                        and not any(
                            item.invalidation_kind == "PREMISE_SUSPECTED"
                            for item in turn_obligations
                        )
                    ):
                        decomposition_target = next(
                            target_id
                            for target_id in target_ids
                            if applied_verdicts.get(target_id) == "UNRESOLVED"
                        )
                        orchestration_id = (
                            f"{run_id}:decomposition:"
                            + hashlib.sha256(
                                decomposition_target.encode(),
                            ).hexdigest()[:12]
                        )
                        certificate = run_certified_decomposition(
                            proof_ledger,
                            decomposition_target,
                            research_goal,
                            run_review_role,
                            project_root=Path(__file__).resolve().parents[1],
                            orchestration_id=orchestration_id,
                            checkpoint_path=orchestration_state_path,
                            candidate_sha256=orchestration_candidate_sha256,
                        )
                        created_obligations = persist_verified_decomposition(
                            proof_ledger,
                            decomposition_target,
                            certificate,
                            run_id,
                        )
                        manifest_path = decomposition_review_dir / (
                            hashlib.sha256(
                                orchestration_id.encode(),
                            ).hexdigest()[:20]
                            + ".json"
                        )
                        manifest_payload = {
                            "schema_version": 2,
                            "decomposition_contract": "single_child_v2",
                            "orchestration_id": orchestration_id,
                            "target_obligation_id": decomposition_target,
                            "verified": certificate.verified,
                            "certificate_hash": certificate.certificate_hash,
                            "errors": certificate.errors,
                            "artifact_hashes": certificate.artifact_hashes,
                            "artifacts": {
                                role: asdict(artifact)
                                for role, artifact in (
                                    certificate.artifacts.items()
                                )
                            },
                            "validation": certificate.validation,
                            "role_run_ids": certificate.role_run_ids,
                            "transcripts": certificate.transcripts,
                            "created_obligation_ids": [
                                item.obligation_id
                                for item in created_obligations
                            ],
                        }
                        save_decomposition_manifest(
                            manifest_path,
                            manifest_payload,
                        )
                        print(
                            "[decomposition-review] "
                            f"target={decomposition_target} "
                            f"verified={certificate.verified} "
                            f"created={len(created_obligations)} "
                            f"manifest={manifest_path}",
                            flush=True,
                        )
                        rejected_frontiers.extend(certificate.errors)
                    else:
                        rejected_frontiers.append(
                            "free-form frontier retained for audit; no child "
                            "persisted without a certified decomposition",
                        )
                    for model_id, target_id in id_repairs:
                        print(
                            "[critic-id-repaired] "
                            f"model_id={model_id} target_id={target_id}",
                            flush=True,
                        )
                    for rejection in rejected_frontiers:
                        print(
                            f"[proof-obligation-rejected] {rejection}",
                            flush=True,
                        )
                    for obligation_id, status in applied_verdicts.items():
                        print(
                            f"[critic-verdict] id={obligation_id} "
                            f"status={status}",
                            flush=True,
                        )
                    for item in created_obligations:
                        print(
                            f"[proof-obligation-created] "
                            f"id={item.obligation_id} "
                            f"parent={item.parent_id} "
                            f"{item.statement}",
                            flush=True,
                        )
                    if research_candidate is not None:
                        print(
                            "[autoresearch-verdict] "
                            + json.dumps(
                                build_autoresearch_verdict(
                                    research_candidate,
                                    proof_ledger,
                                    applied_verdicts,
                                    created_obligations,
                                    rejected_frontiers,
                                ),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    print(
                        f"[proof-ledger-result] id={proof_ledger.ledger_id} "
                        f"version={proof_ledger.version} "
                        f"unresolved={len(pending_obligations(proof_ledger))}",
                        flush=True,
                    )
                previous_generator = generator_text
                previous_critic = critic_text
                live_status.emit(
                    phase="host_validation",
                    role="host_validation",
                    state="review",
                    active_obligation_id=active_obligation_id,
                    source="agent_gan_repl",
                    force=True,
                )
                completed = None
                if remote_run:
                    completed = _telemetry_request(
                        f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                        api_key=api_key,
                        method="PATCH",
                        body={
                            "stages": [
                                critic_stage,
                                *isolated_role_stages,
                            ],
                            "status": "completed",
                            "finished_at": time.time(),
                        },
                    )
                summary = (
                    completed["summary"]
                    if completed is not None
                    else summarize_stages([
                        generator_stage,
                        critic_stage,
                        *isolated_role_stages,
                    ])
                )
                print(
                    "[metrics] "
                    f"KV hit={summary['workload_kv_token_hit_rate']:.1%} "
                    f"decode={summary['aggregate_decode_tok_s']:.2f} tok/s "
                    f"latency={summary['generation_latency_ms_p50']:.2f} ms/token "
                    f"e2e={summary['aggregate_e2e_tok_s']:.2f} tok/s "
                    f"run={run_id}",
                    flush=True,
                )
                print(
                    f"[inference-complete] time="
                    f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} "
                    f"run={run_id}",
                    flush=True,
                )
                live_status.emit(
                    phase="proof_turn_completed",
                    role="orchestrator",
                    state="completed",
                    progress_current=1,
                    progress_total=1,
                    progress_unit="turn",
                    active_obligation_id=active_obligation_id,
                    source="agent_gan_repl",
                    force=True,
                )
                save_checkpoint(
                    state_path,
                    ReplCheckpoint(
                        research_goal=research_goal,
                        previous_generator=previous_generator,
                        previous_critic=previous_critic,
                        last_run_id=run_id,
                    ),
                )
                if proof_ledger is not None and turn_obligations:
                    save_proof_ledger(proof_ledger_path, proof_ledger)
                    orchestration_checkpoint = (
                        load_orchestration_checkpoint(
                            orchestration_state_path,
                        )
                    )
                    if (
                        orchestration_checkpoint is not None
                        and orchestration_checkpoint.proof_state
                        == ProofState.COMMIT
                    ):
                        orchestration_checkpoint.committed = True
                        orchestration_checkpoint.commit_key = (
                            orchestration_checkpoint.orchestration_id
                        )
                        orchestration_checkpoint.ledger_version = (
                            proof_ledger.version
                        )
                        orchestration_checkpoint.transition(
                            ProofState.IDLE,
                            "ledger-and-turn-checkpoint-committed",
                            source_run_id=run_id,
                            strategy_reused=True,
                        )
                        save_orchestration_checkpoint(
                            orchestration_state_path,
                            orchestration_checkpoint,
                        )
                    elif (
                        orchestration_checkpoint is not None
                        and orchestration_checkpoint.proof_state
                        == ProofState.CRITIC
                    ):
                        orchestration_checkpoint.transition(
                            ProofState.IDLE,
                            "turn-committed-without-certified-child",
                            source_run_id=run_id,
                            strategy_reused=True,
                        )
                        orchestration_checkpoint.ledger_version = (
                            proof_ledger.version
                        )
                        save_orchestration_checkpoint(
                            orchestration_state_path,
                            orchestration_checkpoint,
                        )
                    for item in proof_ledger.obligations:
                        if item.obligation_id not in applied_verdicts:
                            continue
                        event = (
                            "proof-obligation-carried"
                            if item.status == "UNRESOLVED"
                            else "proof-obligation-closed"
                        )
                        print(
                            f"[{event}] id={item.obligation_id} "
                            f"status={item.status} run={run_id}",
                            flush=True,
                        )
                    print(
                        f"[proof-ledger-checkpoint] "
                        f"id={proof_ledger.ledger_id} "
                        f"version={proof_ledger.version}",
                        flush=True,
                    )
                if critic_issue_batch is not None:
                    consume_critic_issue_batch(
                        critic_inbox_path,
                        critic_issue_batch,
                        run_id,
                    )
                    print(
                        f"[critic-issues-consumed] id="
                        f"{critic_issue_batch.issue_id} run={run_id}",
                        flush=True,
                    )
                phase = ReplPhase.READY
                if auto_loop_active:
                    print(
                        "[auto-loop] successful turn complete; "
                        "/continue queued",
                        flush=True,
                    )
            except Exception as exc:
                orchestration_checkpoint = load_orchestration_checkpoint(
                    orchestration_state_path,
                )
                if (
                    isinstance(exc, ResumeValidationError)
                    and orchestration_checkpoint is not None
                ):
                    orchestration_checkpoint.state = exc.route_state
                    orchestration_checkpoint.current_role = (
                        exc.route_state.lower()
                    )
                    orchestration_checkpoint.resume_origin = (
                        orchestration_checkpoint.state
                    )
                    orchestration_checkpoint.last_transition_reason = (
                        f"resume-validation-failed:{exc}"
                    )
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        orchestration_checkpoint,
                    )
                if (
                    orchestration_checkpoint is not None
                    and not isinstance(exc, ResumeValidationError)
                    and orchestration_checkpoint.proof_state
                    in {ProofState.GENERATOR, ProofState.CRITIC}
                ):
                    orchestration_checkpoint.retry(
                        orchestration_checkpoint.proof_state,
                        f"{type(exc).__name__}: {exc}",
                        2,
                    )
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        orchestration_checkpoint,
                    )
                live_status.emit(
                    phase="proof_turn_failed",
                    role="orchestrator",
                    state="failed",
                    active_obligation_id=active_obligation_id,
                    source="agent_gan_repl",
                    force=True,
                )
                if remote_run:
                    _telemetry_request(
                        f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                        api_key=api_key,
                        method="PATCH",
                        body={"status": "failed", "finished_at": time.time()},
                    )
                print(
                    f"[inference-failed] time="
                    f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} "
                    f"run={run_id} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                phase = ReplPhase.READY
                auto_loop_active = False
                print(
                    "[auto-loop-paused] inference exception; checkpoint "
                    "preserved. Use /continue after remediation.",
                    flush=True,
                )
    live_status.emit(
        phase="orchestrator_exit",
        role="orchestrator",
        state="idle",
        active_obligation_id=active_obligation_id,
        source="agent_gan_repl",
        force=True,
    )
    transcript.log_only("[session-end]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
