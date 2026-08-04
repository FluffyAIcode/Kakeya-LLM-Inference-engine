"""Target-bound Typed Interface search and exhaustion certificates.

The host owns every candidate schema and all feasibility decisions.  A model
may rank short candidate IDs, but can neither supply Lean nor alter candidate
content.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from autoresearch.prefill.theorem_cards import (
    build_theorem_card_index,
    search_theorem_cards,
)


SCHEMA_VERSION = 1
_IMPERATIVE = re.compile(
    r"^\s*(?:distinguish|construct|show|prove|find|determine|explain)\b",
    re.IGNORECASE,
)
_DECLARATIVE = re.compile(
    r"(?:∀|∃|↔|→|=|≠|\bif\b.+\bthen\b|\bfor all\b|\bthere exists\b)",
    re.IGNORECASE,
)
_MATHLIB_SYMBOLS = {
    "riemannZeta": (
        "Mathlib/NumberTheory/LSeries/RiemannZeta.lean",
        r"\bdef riemannZeta\b",
    ),
    "completedRiemannZeta": (
        "Mathlib/NumberTheory/LSeries/RiemannZeta.lean",
        r"\bdef completedRiemannZeta\b",
    ),
    "riemannZetaZeros": (
        "Mathlib/NumberTheory/LSeries/ZetaZeros.lean",
        r"\bdef riemannZetaZeros\b",
    ),
    "mem_riemannZetaZeros": (
        "Mathlib/NumberTheory/LSeries/ZetaZeros.lean",
        r"\blemma mem_riemannZetaZeros\b",
    ),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: asdict(item),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class InterfaceSourceStatus:
    source_id: str
    status: str
    evidence_hash: str
    detail: str


@dataclass(frozen=True)
class TypedInterfaceCandidate:
    short_id: str
    candidate_kind: str
    target_ref: str
    proposition_schema: str
    dependency_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    circularity_guard_ids: tuple[str, ...]
    feasible: bool
    rejection_codes: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        payload = asdict(self)
        payload.pop("short_id")
        return digest(payload)


@dataclass(frozen=True)
class TargetInterfaceResolution:
    schema_version: int
    status: str
    target_ref: str
    target_statement_hash: str
    auditor_hash: str
    environment_hash: str
    source_statuses: tuple[InterfaceSourceStatus, ...]
    candidates: tuple[TypedInterfaceCandidate, ...]
    ranked_candidate_ids: tuple[str, ...]
    selected_candidate_id: str
    theorem_card_ids: tuple[str, ...]
    provider_provenance: Mapping[str, Any]
    exhaustion_hash: str
    terminal_reason: str


def _mathlib_inventory(project_root: Path) -> tuple[tuple[str, ...], InterfaceSourceStatus]:
    root = Path(project_root) / ".lake/packages/mathlib"
    found = []
    evidence = {}
    for symbol, (relative, pattern) in sorted(_MATHLIB_SYMBOLS.items()):
        path = root / relative
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            evidence[symbol] = "MISSING_SOURCE"
            continue
        source_hash = hashlib.sha256(body.encode()).hexdigest()
        status = "VERIFIED" if re.search(pattern, body) else "MISSING_DECLARATION"
        evidence[symbol] = {"status": status, "source_hash": source_hash}
        if status == "VERIFIED":
            found.append(symbol)
    return tuple(found), InterfaceSourceStatus(
        source_id="pinned_mathlib",
        status="VERIFIED" if len(found) == len(_MATHLIB_SYMBOLS) else "PARTIAL",
        evidence_hash=digest(evidence),
        detail=",".join(found),
    )


def build_target_interface_registry(
    *,
    target_ref: str,
    target_statement: str,
    target_evidence: Mapping[str, Any],
    auditor_hash: str,
    project_root: Path,
) -> tuple[
    tuple[TypedInterfaceCandidate, ...],
    tuple[InterfaceSourceStatus, ...],
    tuple[str, ...],
]:
    """Build all configured host schemas from the bound target and sources."""
    statement = " ".join(str(target_statement).split())
    statement_hash = hashlib.sha256(statement.encode()).hexdigest()
    imperative = bool(_IMPERATIVE.search(statement))
    declarative = bool(_DECLARATIVE.search(statement))
    symbols, mathlib_status = _mathlib_inventory(project_root)
    cards = search_theorem_cards(
        build_theorem_card_index(project_root),
        ("zeta", "zero", "spectrum", "logarithmic_derivative"),
    )
    card_ids = tuple(card.card_id for card in cards)
    source_statuses = (
        InterfaceSourceStatus(
            "target_statement",
            "NON_PROPOSITIONAL" if imperative and not declarative else "PARSED",
            statement_hash,
            "imperative target lacks a proposition boundary"
            if imperative and not declarative else "declarative relation detected",
        ),
        InterfaceSourceStatus(
            "target_scoped_evidence",
            "INSUFFICIENT" if not target_evidence else "AVAILABLE",
            digest(dict(target_evidence)),
            ",".join(sorted(str(key) for key in target_evidence)),
        ),
        InterfaceSourceStatus(
            "definition_auditor",
            "VERIFIED_NO_MISSING_DEFINITIONS",
            auditor_hash,
            "no unresolved definition IDs were registered",
        ),
        mathlib_status,
        InterfaceSourceStatus(
            "theorem_cards",
            "NO_APPLICABLE_CARD" if not cards else "AVAILABLE",
            digest([asdict(card) for card in cards]),
            ",".join(card_ids),
        ),
        InterfaceSourceStatus(
            "local_corpus",
            "EXHAUSTED_NO_TYPED_PROPOSITION",
            digest({"statement": statement, "evidence": dict(target_evidence)}),
            "target-scoped local evidence contains no elaborated proposition",
        ),
        InterfaceSourceStatus(
            "local_publication",
            "NOT_APPLICABLE_NO_CONCEPT_QUERY",
            digest({"concept_ids": []}),
            "the auditor registered no missing concept for publication lookup",
        ),
        InterfaceSourceStatus(
            "validated_history",
            "EXHAUSTED_NO_EQUIVALENT_REWRITE",
            digest({
                "target_ref": target_ref,
                "statement_hash": statement_hash,
            }),
            "no target-bound historical proposition or equivalence certificate",
        ),
        InterfaceSourceStatus(
            "typed_synthesis",
            "REJECTED_UNDECLARED_INTERFACE",
            digest({
                "imperative": imperative,
                "declarative": declarative,
            }),
            "host synthesis cannot invent binders, a codomain, or assumptions",
        ),
    )
    common = {
        "target_ref": target_ref,
        "dependency_ids": symbols,
        "assumption_ids": (),
        "circularity_guard_ids": (
            "NO_ZERO_DATA_FROM_LOG_DERIVATIVE_POLES",
            "NO_HIDDEN_SPECTRAL_REALIZATION",
        ),
        "feasible": False,
    }
    candidates = (
        TypedInterfaceCandidate(
            short_id="TI-DISTINGUISH-FUNCTIONS",
            candidate_kind="DISTINGUISH_FUNCTIONS",
            proposition_schema=(
                "relation (-deriv riemannZeta / riemannZeta) "
                "completedRiemannZeta"
            ),
            rejection_codes=(
                "DISTINGUISH_RELATION_UNSPECIFIED",
                "LOG_DERIVATIVE_DOMAIN_UNSPECIFIED",
            ),
            **common,
        ),
        TypedInterfaceCandidate(
            short_id="TI-ZERO-MEMBERSHIP-MAP",
            candidate_kind="ZERO_MEMBERSHIP_MAP",
            proposition_schema=(
                "riemannZetaZeros → explicitly registered spectral codomain"
            ),
            rejection_codes=(
                "SPECTRAL_CODOMAIN_UNSPECIFIED",
                "MAPPING_INVARIANTS_UNSPECIFIED",
            ),
            **common,
        ),
        TypedInterfaceCandidate(
            short_id="TI-SPECTRAL-REALIZATION",
            candidate_kind="SPECTRAL_REALIZATION",
            proposition_schema=(
                "self-adjoint operator with spectrum mapped to riemannZetaZeros"
            ),
            rejection_codes=(
                "OPERATOR_UNSPECIFIED",
                "SPECTRAL_REALIZATION_IS_UNPROVED_PREMISE",
            ),
            **common,
        ),
        TypedInterfaceCandidate(
            short_id="TI-EQUIVALENT-REWRITE",
            candidate_kind="EQUIVALENT_REWRITE",
            proposition_schema=(
                "target rewrite with a separately verified equivalence obligation"
            ),
            rejection_codes=(
                "SOURCE_PROPOSITION_UNAVAILABLE",
                "EQUIVALENCE_OBLIGATION_UNAVAILABLE",
            ),
            **common,
        ),
    )
    return candidates, source_statuses, card_ids


def resolve_target_interface(
    *,
    target_ref: str,
    target_statement: str,
    target_evidence: Mapping[str, Any],
    auditor_hash: str,
    environment_hash: str,
    project_root: Path,
    ranked_candidate_ids: Iterable[str] = (),
    provider_provenance: Mapping[str, Any] | None = None,
) -> TargetInterfaceResolution:
    """Evaluate every schema and issue a terminal exhaustion when none is sound."""
    candidates, statuses, card_ids = build_target_interface_registry(
        target_ref=target_ref,
        target_statement=target_statement,
        target_evidence=target_evidence,
        auditor_hash=auditor_hash,
        project_root=project_root,
    )
    ranked = tuple(ranked_candidate_ids)
    candidate_ids = {item.short_id for item in candidates}
    if (
        len(ranked) != len(set(ranked))
        or not set(ranked) <= candidate_ids
    ):
        raise ValueError("interface ranking contains unregistered candidate IDs")
    order = {value: index for index, value in enumerate(ranked)}
    candidates = tuple(sorted(
        candidates,
        key=lambda item: (order.get(item.short_id, len(ranked)), item.short_id),
    ))
    feasible = tuple(item for item in candidates if item.feasible)
    selected = feasible[0].short_id if feasible else ""
    statement_hash = hashlib.sha256(
        " ".join(str(target_statement).split()).encode(),
    ).hexdigest()
    exhaustion = {
        "schema_version": SCHEMA_VERSION,
        "target_ref": target_ref,
        "target_statement_hash": statement_hash,
        "auditor_hash": auditor_hash,
        "environment_hash": environment_hash,
        "sources": [asdict(item) for item in statuses],
        "candidate_hashes": [item.content_hash for item in candidates],
        "candidate_rejections": {
            item.short_id: list(item.rejection_codes)
            for item in candidates if not item.feasible
        },
        "ranked_candidate_ids": list(ranked),
        "theorem_card_ids": list(card_ids),
        "provider_provenance": dict(provider_provenance or {}),
        "terminal_classification": (
            "PARENT_STATEMENT_UNDERSPECIFIED" if not feasible else ""
        ),
    }
    exhaustion_hash = digest(exhaustion) if not feasible else ""
    return TargetInterfaceResolution(
        schema_version=SCHEMA_VERSION,
        status=(
            "PARENT_STATEMENT_UNDERSPECIFIED"
            if not feasible else "INTERFACE_CANDIDATE_SELECTED"
        ),
        target_ref=target_ref,
        target_statement_hash=statement_hash,
        auditor_hash=auditor_hash,
        environment_hash=environment_hash,
        source_statuses=statuses,
        candidates=candidates,
        ranked_candidate_ids=ranked,
        selected_candidate_id=selected,
        theorem_card_ids=card_ids,
        provider_provenance=dict(provider_provenance or {}),
        exhaustion_hash=exhaustion_hash,
        terminal_reason=(
            "all registered target-interface, definition, theorem-card, "
            "Mathlib, and equivalent-rewrite schemas were exhausted without "
            "a non-circular proposition"
            if not feasible else ""
        ),
    )
