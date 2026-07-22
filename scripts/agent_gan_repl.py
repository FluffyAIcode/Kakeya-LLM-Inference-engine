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
from dataclasses import asdict, dataclass, field
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
from autoresearch.prefill.lean_gate import (
    LeanSignatureResult,
    lean_theorem_signature_hash,
    validate_lean_proof,
    validate_lean_signature,
)
from autoresearch.prefill.semantic_decompose import (
    SemanticUnitTooLarge,
    admit_token_ids,
    build_proof_step_interface,
    downstream_output_cap,
    serialize_proof_step_interface,
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
    children: list[dict]
    dependency_edges: list[list[str]]
    public_assumptions: list[str]
    reduction_labels: list[str]
    reduction_contract: str


@dataclass(frozen=True)
class FormalizationBundle:
    target_obligation_id: str
    parent_statement_hash: str
    root_goal_hash: str
    producer_role: str
    producer_run_id: str
    upstream_artifact_hashes: list[str]
    math_ir: dict
    parent_signature_source: str
    parent_signature_hash: str
    parent_newly_formalized: bool
    children: list[dict]
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
    r"^### ISSUE_VERDICT\s+(\S+)\s*$"
    r"(?P<body>.*?)(?=^### ISSUE_VERDICT\s+|\Z)",
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
    return match.group(1).strip() if match else ""


def _json_artifact(value: str) -> dict:
    try:
        artifact = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        if not isinstance(value, str):
            return {}
        repaired = re.sub(
            r'\\(?!(?:["\\/]|u[0-9a-fA-F]{4}))',
            r"\\\\",
            value,
        )
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
    return [{
        "role": "system",
        "content": (
            "You are an isolated Premise Auditor. Independently attack the "
            "named premise using counterexamples, exact definitions and "
            "quantifiers, theorem conflicts, and verifiable Lean, finite, or "
            "symbolic evidence. Do not trust the Critic conclusion. Return "
            "`### PREMISE_AUDIT <ID>` with `Status: "
            "CONFIRMED|NOT_CONFIRMED|INCONCLUSIVE`, `Evidence type:`, "
            "`Evidence source:`, `Confidence:` in [0,1], one-line JSON "
            "`Artifact:`, and one-line `Analysis:`. For arithmetic, Artifact "
            "must contain exactly the host `claim_hash`, unchanged `claim`, "
            "and a `witness` binding every quantified variable. The witness "
            "must make the Critic's claimed relation false. For Lean, preserve "
            "the exact claim/signature hashes and negation contract."
        ),
    }, {
        "role": "user",
        "content": (
            f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
            f"HOST-PACKAGED CRITIC SUSPICION:\n{package}"
        ),
    }]


def build_premise_proponent_messages(
    goal: str,
    suspicion: PremiseSuspicion,
    auditor_text: str,
) -> list[dict[str, str]]:
    package = json.dumps(asdict(suspicion), ensure_ascii=False, sort_keys=True)
    return [{
        "role": "system",
        "content": (
            "You are an isolated Adversarial Proponent. Attempt to rescue the "
            "premise by finding the exact domain, topology, or quantifier "
            "correction, or by refuting the Auditor artifact. Return `### "
            "PREMISE_DEFENSE <ID>` with `Status: "
            "RESCUED|NOT_RESCUED|INCONCLUSIVE`, `Correction:`, `Failure "
            "reason:`, and one-line `Evidence:`."
        ),
    }, {
        "role": "user",
        "content": (
            f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
            f"HOST-PACKAGED CRITIC SUSPICION:\n{package}\n\n"
            f"COMPLETE ISOLATED AUDITOR OUTPUT:\n{auditor_text}"
        ),
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
            "adversarial_proponent",
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


def _canonical_json_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


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
    payload = _json_artifact(_structured_field(match.group("body"), "Artifact"))
    if not payload:
        return None, f"malformed {heading} Artifact JSON"
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
            "Propose labeled child propositions, public assumptions, acyclic "
            "dependency edges, and an explicit conjunction-to-parent reduction."
        ),
        "formalizer": (
            "Emit typed Math IR, exact parent and child Lean signatures, and a "
            "reduction theorem signature scaffold. Never replace a bound parent."
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
            "Decide ACCEPT|REJECT|INCONCLUSIVE using only the host-verified "
            "manifest. You cannot override a failed host gate."
        ),
    }[role]
    artifact_schema = {
        "definition_auditor": (
            '{"definitions":[{"symbol":"...","type":"...","scope":"..."}],'
            '"missing_definitions":[{"obligation_label":"L1","symbol":"...",'
            '"required_type":"..."}]}'
        ),
        "counterexample_worker": (
            '{"status":"COUNTEREXAMPLE_FOUND|NO_COUNTEREXAMPLE|INCONCLUSIVE",'
            '"cases":[]}'
        ),
        "decomposer": (
            '{"parent_statement":"<exact>","children":[{"label":"L1",'
            '"statement":"...","kind":"DEFINITION|LEMMA"}],'
            '"dependency_edges":[],"public_assumptions":[],'
            '"reduction_labels":["L1"],"reduction_contract":"L1 and A imply P"}'
        ),
        "formalizer": (
            '{"math_ir":{"parent_signature_hash":"...","parent_proposition_hash":'
            '"...","child_labels":["L1"],"public_assumptions":[],'
            '"reduction_labels":["L1"]},"parent_signature_source":"...",'
            '"parent_signature_hash":"...","parent_newly_formalized":true,'
            '"children":[{"label":"L1","statement":"...",'
            '"lean_signature":"...","lean_signature_hash":"..."}],'
            '"reduction_theorem_source":"...","reduction_signature_hash":"..."}'
        ),
        "prover": (
            '{"status":"PROVED|FAILED|INCONCLUSIVE",'
            '"reduction_theorem_source":"..."}'
        ),
        "adversarial_proponent": (
            '{"status":"DEFENDED|REJECTED|INCONCLUSIVE",'
            '"issues":[],"repairs":[]}'
        ),
        "judge": (
            '{"decision":"ACCEPT|REJECT|INCONCLUSIVE","reason":"..."}'
        ),
    }[role]
    return [{
        "role": "system",
        "content": (
            f"You are the isolated {role}. {behavior} Return exactly `### "
            f"{heading}` followed by one-line `Artifact:` JSON. Emit only the "
            f"role fields in this schema: {artifact_schema} Host bindings are "
            "attached automatically; if emitted, they must match the package."
        ),
    }, {
        "role": "user",
        "content": json.dumps(package, ensure_ascii=False, sort_keys=True),
    }]


def _required_certified_upstream(role: str) -> set[str]:
    return {
        "definition_auditor": set(),
        "counterexample_worker": {"definition_auditor"},
        "decomposer": {
            "definition_auditor",
            "counterexample_worker",
        },
        "formalizer": {"decomposer"},
        "prover": {"formalizer"},
        "adversarial_proponent": {
            "decomposer",
            "formalizer",
            "prover",
        },
    }[role]


def _validate_dependency_graph(
    proposal: DecompositionProposal,
) -> list[str]:
    errors = []
    labels = [
        str(child.get("label", ""))
        for child in proposal.children
        if isinstance(child, dict)
    ]
    if (
        not labels
        or any(not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        return ["child labels must be non-empty and unique"]
    if len(labels) != 1:
        return ["one-step decomposition requires exactly one child"]
    if set(proposal.reduction_labels) != set(labels):
        errors.append("every child must be reachable in the reduction contract")
    graph = {label: set() for label in labels}
    for edge in proposal.dependency_edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or edge[0] not in graph
            or edge[1] not in graph
        ):
            errors.append("dependency edge references an invalid child label")
            continue
        if edge[0] == edge[1]:
            errors.append("child dependency graph must be acyclic")
            continue
        graph[edge[0]].add(edge[1])
    visiting = set()
    visited = set()

    def visit(label: str) -> bool:
        if label in visiting:
            return False
        if label in visited:
            return True
        visiting.add(label)
        if any(not visit(dependency) for dependency in graph[label]):
            return False
        visiting.remove(label)
        visited.add(label)
        return True

    if any(not visit(label) for label in labels):
        errors.append("child dependency graph must be acyclic")
    return errors


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
    defense: DefenseReport,
    *,
    project_root: Path,
    signature_validator=validate_lean_signature,
    proof_validator=validate_lean_proof,
) -> tuple[dict, list[str]]:
    errors = _validate_dependency_graph(proposal)
    validation = {
        "graph_valid": not errors,
        "parent_signature_valid": False,
        "children_valid": False,
        "reduction_signature_valid": False,
        "reduction_proof_valid": False,
        "defense_nonblocking": defense.status in {
            "DEFENDED",
            "INCONCLUSIVE",
        },
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
    if (
        counterexamples.status == "COUNTEREXAMPLE_FOUND"
        and not verified_counterexample
    ):
        errors.append("claimed counterexample has no verified evidence")
    elif verified_counterexample:
        errors.append(
            "verified counterexample refutes the parent; decomposition is "
            "forbidden and premise review is required",
        )
    proposal_labels = {
        str(child.get("label", "")): child
        for child in proposal.children
        if isinstance(child, dict)
    }
    definition_children = {
        label
        for label, child in proposal_labels.items()
        if child.get("kind") == "DEFINITION"
    }
    required_definition_labels = {
        str(item.get("obligation_label", ""))
        for item in definition_audit.missing_definitions
        if isinstance(item, dict)
    }
    if (
        "" in required_definition_labels
        or not required_definition_labels.issubset(definition_children)
    ):
        errors.append(
            "every missing definition must become a labeled definition child",
        )
    formal_children = {
        str(child.get("label", "")): child
        for child in formalization.children
        if isinstance(child, dict)
    }
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
    parent_proposition_hash = (
        hashlib.sha256(parent_conclusion.encode()).hexdigest()
        if parent_conclusion else ""
    )
    if set(formal_children) != set(proposal_labels):
        errors.append("formalized child labels differ from proposal")
    if (
        formalization.math_ir.get("parent_signature_hash")
        != formalization.parent_signature_hash
        or formalization.math_ir.get("parent_proposition_hash")
        != parent_proposition_hash
        or set(formalization.math_ir.get("child_labels", []))
        != set(proposal_labels)
        or formalization.math_ir.get("public_assumptions")
        != proposal.public_assumptions
        or set(formalization.math_ir.get("reduction_labels", []))
        != set(proposal.reduction_labels)
    ):
        errors.append("typed Math IR does not bind the exact reduction contract")
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
    child_results = {}
    proposed_items = list(proposal_labels.items())
    for index, (left_label, left_child) in enumerate(proposed_items):
        for right_label, right_child in proposed_items[index + 1:]:
            equivalent, _score = _semantic_equivalence(
                str(left_child.get("statement", "")),
                str(right_child.get("statement", "")),
            )
            if equivalent:
                errors.append(
                    f"children {left_label} and {right_label} are redundant",
                )
    for label, child in formal_children.items():
        source = str(child.get("lean_signature", ""))
        result = signature_validator(source, project_root=project_root)
        child_results[label] = result
        if not result.ok:
            errors.append(f"child {label} signature failed: {result.error}")
        elif child.get("lean_signature_hash") != result.signature_hash:
            errors.append(f"child {label} signature hash mismatch")
        if str(child.get("statement", "")) != str(
            proposal_labels.get(label, {}).get("statement", ""),
        ):
            errors.append(f"child {label} statement changed during formalization")
        reason = _frontier_rejection_reason(
            ledger,
            parent.obligation_id,
            str(child.get("statement", "")),
        )
        if reason:
            errors.append(f"child {label} rejected: {reason}")
    validation["children_valid"] = bool(child_results) and all(
        result.ok for result in child_results.values()
    )
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
    if defense.status == "REJECTED":
        errors.append("adversarial defense found a blocking defect")
    validation["child_signature_hashes"] = {
        label: result.signature_hash
        for label, result in child_results.items()
        if result.ok
    }
    validation["reduction_proof_hash"] = (
        proof_result.signature_hash if proof_result.ok else ""
    )
    validation["host_gates_passed"] = not errors
    return validation, errors


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
    for role, heading in role_specs:
        expected_run_id = f"{orchestration_id}:{role}"
        upstream = list(hashes.values())
        package = {
            "target_obligation_id": target_id,
            "parent_statement": parent.statement,
            "parent_statement_hash": statement_hash,
            "root_goal": root_goal,
            "root_goal_hash": goal_hash,
            "producer_role": role,
            "producer_run_id": expected_run_id,
            "upstream_artifact_hashes": upstream,
            "validated_upstream_artifacts": {
                name: asdict(value)
                for name, value in artifacts.items()
                if name in _required_certified_upstream(role)
            },
            "parent_formal_status": parent.formal_status,
            "parent_lean_signature": parent.lean_signature,
            "parent_lean_signature_hash": parent.lean_signature_hash,
        }
        try:
            text, actual_run_id = run_role(
                role,
                _certified_role_messages(role, heading, package),
                expected_run_id,
            )
        except Exception as exc:
            errors.append(f"{role} failed: {type(exc).__name__}: {exc}")
            break
        transcripts[role] = text
        role_run_ids[role] = actual_run_id
        if actual_run_id != expected_run_id:
            errors.append(f"{role} run ID mismatch")
            break
        artifact, error = parse_certified_artifact(
            text,
            heading,
            target_obligation_id=target_id,
            parent_statement_hash=statement_hash,
            root_goal_hash=goal_hash,
            producer_run_id=expected_run_id,
            upstream_artifact_hashes=upstream,
        )
        if error:
            errors.append(error)
            break
        artifacts[role] = artifact
        hashes[role] = _canonical_json_hash(asdict(artifact))
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
    judge_manifest = {
        "target_obligation_id": target_id,
        "parent_statement": parent.statement,
        "parent_statement_hash": statement_hash,
        "root_goal_hash": goal_hash,
        "artifact_hashes": hashes,
        "validation": validation,
        "errors": errors,
        "retained_child_statements": [
            child.get("statement", "")
            for child in (
                artifacts.get("decomposer").children
                if artifacts.get("decomposer") else []
            )
        ],
    }
    manifest_hash = _canonical_json_hash(judge_manifest)
    if len(artifacts) == 6:
        role = "judge"
        heading = "JUDGE_DECISION"
        expected_run_id = f"{orchestration_id}:{role}"
        package = {
            **judge_manifest,
            "producer_role": role,
            "producer_run_id": expected_run_id,
            "upstream_artifact_hashes": [manifest_hash],
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
            else:
                artifacts[role] = judge
                hashes[role] = _canonical_json_hash(asdict(judge))
                if judge.decision != "ACCEPT":
                    errors.append(f"Judge decision was {judge.decision}")
        except Exception as exc:
            errors.append(f"judge failed: {type(exc).__name__}: {exc}")
    verified = bool(validation.get("host_gates_passed")) and not errors
    certificate_hash = _canonical_json_hash({
        "orchestration_id": orchestration_id,
        "artifact_hashes": hashes,
        "validation": validation,
    })
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
    formal_by_label = {
        str(child["label"]): child
        for child in formalization.children
    }
    if parent.formal_status == "UNFORMALIZED":
        parent.formal_status = "FORMALIZED"
        parent.lean_signature = formalization.parent_signature_source
        parent.lean_signature_hash = formalization.parent_signature_hash
    label_to_id = {
        str(child["label"]): (
            f"{target_id}-"
            + hashlib.sha256(
                f"{result.certificate_hash}:{child['label']}".encode(),
            ).hexdigest()[:10]
        )
        for child in proposal.children
    }
    dependencies = {
        label: [] for label in label_to_id
    }
    for source, dependency in proposal.dependency_edges:
        dependencies[source].append(label_to_id[dependency])
    created = []
    for child in proposal.children:
        label = str(child["label"])
        formal = formal_by_label[label]
        item = ProofObligation(
            obligation_id=label_to_id[label],
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
            dependency_labels=[
                edge[1]
                for edge in proposal.dependency_edges
                if edge[0] == label
            ],
            dependency_ids=dependencies[label],
            certificate_reversible_status="ACTIVE",
        )
        ledger.obligations.append(item)
        created.append(item)
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
    model_id: str,
    allowed_ids: set[str],
) -> str:
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
        if not status_match or not evidence_match or missing_match is None:
            continue
        status = status_match.group(1)
        evidence = evidence_match.group(1).strip()
        missing = missing_match.group(1).strip().lower()
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
    def __init__(self, tokenizer, label: str) -> None:
        self.tokenizer = tokenizer
        self.last = ""
        print(f"{label}> ", end="", flush=True)

    def __call__(self, token_ids) -> None:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(text[len(self.last):], end="", flush=True)
        self.last = text

    def finish(self) -> None:
        print(flush=True)


class PrefillHeartbeat:
    def __init__(
        self,
        label: str,
        interval_s: float = 30.0,
        stats_provider=None,
    ) -> None:
        self.label = label
        self.interval_s = interval_s
        self.stats_provider = stats_provider
        self.stop = threading.Event()
        self.started = 0.0
        self.thread = None

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=1)

    def _run(self):
        while not self.stop.wait(self.interval_s):
            elapsed = time.perf_counter() - self.started
            progress = ""
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
    stage = {
        **actual,
        "name": f"agent_{name}",
        "agent": name,
        "round": 1,
        "hit_source": "primary_hot" if delta["local_hits"] else "unknown",
        "ok": _agent_cache_gate(warm["delta"], delta) and actual["complete"],
        "warmup_prefix_tokens": warm["prefix_tokens"],
        "warmup_tokens_reused": (
            warm["delta"]["tokens_reused"]
            if warm["delta"]["remote_jobs"] == 0 else 0
        ),
        "warmup_wall_s": warm["e2e_s"],
        "warmup_remote_jobs": warm["delta"]["remote_jobs"],
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
            run = _telemetry_request(
                f"{args.dashboard}/v1/network/benchmarks",
                api_key=api_key,
                method="POST",
                body={
                    "kind": "agent_gan_interactive",
                    "config": {
                        "model_id": "gemma-4-26B-A4B-it-mlx-4bit",
                        "topology": "primary-decode-allens-prefill",
                        "agents": [
                            "generator",
                            "critic",
                            "premise_auditor",
                            "definition_auditor",
                            "counterexample_worker",
                            "decomposer",
                            "formalizer",
                            "prover",
                            "adversarial_proponent",
                            "judge",
                        ],
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
            started_at = datetime.now().astimezone().isoformat(
                timespec="milliseconds",
            )
            print(
                f"[inference-start] time={started_at} run={run_id} "
                f"goal={hashlib.sha256(research_goal.encode()).hexdigest()}",
                flush=True,
            )
            try:
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
                with PrefillHeartbeat("Generator", stats_provider=get_stats):
                    _, generator_warm = _infer(
                        client, eos_ids, generator_ids, 1, get_stats,
                        max_retained_tokens=args.max_retained_tokens,
                    )
                generator_printer = TokenPrinter(tokenizer, "generator")
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
                with PrefillHeartbeat("Critic", stats_provider=get_stats):
                    _, critic_warm = _infer(
                        client, eos_ids, critic_ids, 1, get_stats,
                        max_retained_tokens=args.max_retained_tokens,
                    )
                critic_printer = TokenPrinter(tokenizer, "critic")
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
                        role_output_cap = downstream_output_cap(
                            max_retained_tokens=args.max_retained_tokens,
                            fixed_downstream_tokens=len(role_ids),
                            configured_output_tokens=(
                                args.max_response_tokens or None
                            ),
                            control_reserve_tokens=64,
                        )
                        print(
                            f"[allens] {role_name} Prefill: "
                            f"{len(role_ids)} tokens...",
                            flush=True,
                        )
                        with PrefillHeartbeat(
                            role_name,
                            stats_provider=get_stats,
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
                        role_printer = TokenPrinter(tokenizer, role_name)
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
                            client_label=f"agent-gan-{role_name}",
                            max_retained_tokens=args.max_retained_tokens,
                        )
                        role_printer.finish()
                        role_text = decode_complete_response(
                            tokenizer,
                            role_name,
                            role_tokens,
                            role_actual,
                        )
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
                        return (
                            role_text,
                            expected_run_id or (
                                f"{run_id}:{role_name}:"
                                f"{next(iter(target_ids))}"
                            ),
                        )

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
                            "schema_version": 1,
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
                previous_generator = generator_text
                previous_critic = critic_text
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
    transcript.log_only("[session-end]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
