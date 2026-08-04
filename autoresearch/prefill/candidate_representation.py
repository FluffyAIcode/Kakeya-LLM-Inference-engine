"""Host-owned representation analysis for private exploration candidates."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


REPRESENTATION_ANALYSIS_VERSION = 1
MAPPER_CAPABILITY_VERSION = 2


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


class RepresentationOutcome(str, Enum):
    MAPPER_EXTENSION_REQUIRED = "MAPPER_EXTENSION_REQUIRED"
    REGISTRY_RESOLUTION = "REGISTRY_RESOLUTION"
    DEFINITION_RESOLUTION = "DEFINITION_RESOLUTION"
    SEMANTIC_REJECTION = "SEMANTIC_REJECTION"
    REPRESENTATION_EXHAUSTED = "REPRESENTATION_EXHAUSTED"
    MAPPED = "MAPPED"


@dataclass(frozen=True)
class PrimitiveCapability:
    primitive_id: str
    source_kind: str
    source_ref: str
    trusted: bool


PRIMITIVE_CATALOG = {
    item.primitive_id: item for item in (
        PrimitiveCapability("COMPLEX_SCALAR", "MATHLIB", "Complex", True),
        PrimitiveCapability("REAL_SCALAR", "MATHLIB", "Real", True),
        PrimitiveCapability("UNIVERSAL_QUANTIFIER", "LEAN_CORE", "forall", True),
        PrimitiveCapability("EXISTENTIAL_QUANTIFIER", "LEAN_CORE", "Exists", True),
        PrimitiveCapability("SEQUENCE_NAT_INDEXED", "MATHLIB", "Nat → α", True),
        PrimitiveCapability("FINITE_SUM", "MATHLIB", "Finset.sum", True),
        PrimitiveCapability("INTERVAL_INTEGRAL", "MATHLIB", "intervalIntegral", True),
        PrimitiveCapability(
            "FILTER_LIMIT",
            "HOST_REGISTRY",
            "convergesComplexSequence",
            True,
        ),
        PrimitiveCapability(
            "CONTINUOUS_MAP",
            "HOST_REGISTRY",
            "continuousComplexMap",
            True,
        ),
        PrimitiveCapability(
            "HOLOMORPHIC_MAP",
            "HOST_REGISTRY",
            "holomorphicComplexMapOn",
            True,
        ),
        PrimitiveCapability(
            "BOUNDED_COMPLEX_MAP_ON",
            "HOST_REGISTRY",
            "boundedComplexMapOn",
            True,
        ),
        PrimitiveCapability("FINITE_MATRIX", "MATHLIB", "Matrix", True),
        PrimitiveCapability("LINEAR_MAP", "MATHLIB", "LinearMap", True),
        PrimitiveCapability(
            "RIEMANN_HYPOTHESIS_TARGET",
            "HOST_REGISTRY",
            "RiemannHypothesis",
            True,
        ),
        PrimitiveCapability(
            "RIEMANN_ZETA_FUNCTION",
            "HOST_REGISTRY",
            "riemannZeta",
            True,
        ),
        PrimitiveCapability("ZETA_ZERO_SEQUENCE", "NEW_DEFINITION", "", False),
        PrimitiveCapability("ZERO_COUNTING_FUNCTION", "NEW_DEFINITION", "", False),
        PrimitiveCapability("LOCAL_ZERO_DENSITY", "NEW_DEFINITION", "", False),
        PrimitiveCapability("PAIR_CORRELATION_PREDICATE", "NEW_DEFINITION", "", False),
        PrimitiveCapability("GUE_SPACING_LIMIT", "NEW_DEFINITION", "", False),
        PrimitiveCapability("SPECTRAL_MEASURE", "NEW_DEFINITION", "", False),
        PrimitiveCapability("ESSENTIAL_SPECTRUM", "NEW_DEFINITION", "", False),
        PrimitiveCapability("SPECTRAL_GAP", "NEW_DEFINITION", "", False),
        PrimitiveCapability("DIRICHLET_L_FUNCTION", "NEW_DEFINITION", "", False),
        PrimitiveCapability("SELBERG_CLASS", "NEW_DEFINITION", "", False),
        PrimitiveCapability("RANK_ONE_PERTURBATION", "NEW_DEFINITION", "", False),
        PrimitiveCapability("SECOND_ORDER_CORRELATION", "NEW_DEFINITION", "", False),
    )
}


def mapper_capability_hash() -> str:
    return _digest({
        "version": MAPPER_CAPABILITY_VERSION,
        "capabilities": [
            asdict(PRIMITIVE_CATALOG[key]) for key in sorted(PRIMITIVE_CATALOG)
        ],
    })


@dataclass(frozen=True)
class RepresentationGapReport:
    report_id: str
    candidate_id: str
    candidate_hash: str
    category: str
    target_obligation_id: str
    target_context_hash: str
    mapper_registry_hash: str
    environment_hash: str
    safe_characterization_id: str
    intended_relation_id: str
    required_primitive_ids: tuple[str, ...]
    missing_primitive_ids: tuple[str, ...]
    mathlib_source_ids: tuple[str, ...]
    host_registry_source_ids: tuple[str, ...]
    new_definition_ids: tuple[str, ...]
    hidden_assumption_ids: tuple[str, ...]
    semantic_issue_ids: tuple[str, ...]
    duplicate_candidate_ids: tuple[str, ...]
    mapper_failure_codes: tuple[str, ...]
    outcome: str
    retry_allowed: bool
    private_memo_consumed: bool
    private_text_exposed: bool
    ledger_mutation_allowed: bool
    content_hash: str
    schema_version: int = REPRESENTATION_ANALYSIS_VERSION


def _primitive_ids(text: str) -> tuple[str, ...]:
    tests = (
        ("COMPLEX_SCALAR", (r"\bcomplex\b", r"\\mathbb\{c\}")),
        ("REAL_SCALAR", (r"\breal\b", r"\\mathbb\{r\}")),
        ("UNIVERSAL_QUANTIFIER", (r"\bfor all\b", r"\bany\b")),
        ("EXISTENTIAL_QUANTIFIER", (r"\bthere exists\b",)),
        ("SEQUENCE_NAT_INDEXED", (r"\bsequence\b", r"_n\b", r"_j\b", r"_k\b")),
        ("FINITE_SUM", (r"\\sum",)),
        ("INTERVAL_INTEGRAL", (r"\\int",)),
        ("FILTER_LIMIT", (r"\\lim", r"\blimit\b", r"\\limsup")),
        ("CONTINUOUS_MAP", (r"\bcontinuous\b", r"\bc\^1\b")),
        ("HOLOMORPHIC_MAP", (r"\bholomorphic\b", r"\banalytic\b")),
        ("BOUNDED_COMPLEX_MAP_ON", (r"\bbounded\b",)),
        ("FINITE_MATRIX", (r"\bmatrix\b", r"\\mathcal\{u\}\(n\)")),
        ("LINEAR_MAP", (r"\boperator\b",)),
        ("RIEMANN_HYPOTHESIS_TARGET", (r"\briemann hypothesis\b", r"\brh-c0\b")),
        ("RIEMANN_ZETA_FUNCTION", (r"\bzeta\b", r"\\zeta")),
        ("ZETA_ZERO_SEQUENCE", (r"\bzeros?\b", r"\\gamma_", r"\\rho_")),
        ("ZERO_COUNTING_FUNCTION", (r"\bn\(t\)\b", r"zero-counting")),
        ("LOCAL_ZERO_DENSITY", (r"\blocal density\b", r"density fluctuation")),
        ("PAIR_CORRELATION_PREDICATE", (r"pair correlation",)),
        ("GUE_SPACING_LIMIT", (r"\bgue\b", r"gaussian unitary")),
        ("SPECTRAL_MEASURE", (r"spectral measure",)),
        ("ESSENTIAL_SPECTRUM", (r"essential spectrum",)),
        ("SPECTRAL_GAP", (r"spectral gap",)),
        ("DIRICHLET_L_FUNCTION", (
            r"dirichlet(?:\s|\$|\\)*l(?:\s|\$|-)*function",
            r"(?:\b|\$)l(?:\s|\$|-)*function",
        )),
        ("SELBERG_CLASS", (r"selberg class",)),
        ("RANK_ONE_PERTURBATION", (r"rank-1", r"rank-one")),
        ("SECOND_ORDER_CORRELATION", (r"second-order correlation", r"r_2")),
    )
    found = []
    for primitive_id, patterns in tests:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            found.append(primitive_id)
    zeta_context = bool(re.search(
        r"(riemann hypothesis|\\zeta|\bzeta\b|critical line|non-trivial zeros)",
        text,
        flags=re.IGNORECASE,
    ))
    if not zeta_context:
        found = [
            item for item in found
            if item not in {
                "ZETA_ZERO_SEQUENCE",
                "ZERO_COUNTING_FUNCTION",
                "LOCAL_ZERO_DENSITY",
                "PAIR_CORRELATION_PREDICATE",
                "GUE_SPACING_LIMIT",
                "SECOND_ORDER_CORRELATION",
            }
        ]
    return tuple(found)


def _semantic_issues(text: str, category: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lowered = text.casefold()
    zeta_connected = any(token in lowered for token in (
        "riemann hypothesis", "zeta", "critical line", "non-trivial zeros",
    ))
    semantic = []
    hidden = []
    if not zeta_connected:
        semantic.append("DISCONNECTED_FROM_TARGET")
    if category == "TOY_MODEL_ANALOGUE":
        semantic.append("NO_CHILD_TO_PARENT_RELATION")
    if "montgomery pair correlation conjecture holds" in lowered:
        hidden.append("ASSUME_MONTGOMERY_PAIR_CORRELATION")
    if re.search(
        r"for any .*l.*function satisfying .*riemann hypothesis",
        lowered,
    ):
        hidden.append("ASSUME_GENERALIZED_RH")
    if re.search(r"\bgue\b|gaussian unitary", lowered):
        hidden.append("ASSUME_GUE_STATISTICS")
    if "eigenvalues" in lowered and "zeros" in lowered:
        hidden.append("ASSUME_SPECTRAL_ZERO_CORRESPONDENCE")
    if "functional equation" in lowered and "despite" in lowered:
        hidden.append("ASSUME_UNREGISTERED_FUNCTIONAL_EQUATION")
    if re.search(r"\\rho_j\s*=\s*\\frac\{1\}\{2\}", lowered):
        semantic.append("PARENT_ASSUMED_IN_CANDIDATE")
    if "equivalent to the assertion that no zeros lie outside" in lowered:
        semantic.append("UNSUPPORTED_EQUIVALENCE_TO_PARENT")
    if "specific parameterization of rh-c0" in lowered:
        hidden.append("UNDECLARED_TARGET_PARAMETERIZATION")
    if "specific operator class rh-c0" in lowered:
        hidden.append("UNDECLARED_TARGET_OPERATOR_CLASS")
    return tuple(sorted(set(semantic))), tuple(sorted(set(hidden)))


def analyze_private_candidate_representation(
    *,
    candidate_id: str,
    candidate_hash: str,
    category: str,
    memo_text: str,
    target_obligation_id: str,
    target_context_hash: str,
    environment_hash: str,
    duplicate_candidate_ids: Iterable[str] = (),
) -> RepresentationGapReport:
    """Consume private prose and emit only constrained, content-addressed IDs."""
    primitives = _primitive_ids(memo_text)
    missing = tuple(
        item for item in primitives if not PRIMITIVE_CATALOG[item].trusted
    )
    mathlib = tuple(
        item for item in primitives
        if PRIMITIVE_CATALOG[item].source_kind in {"MATHLIB", "LEAN_CORE"}
    )
    host = tuple(
        item for item in primitives
        if PRIMITIVE_CATALOG[item].source_kind == "HOST_REGISTRY"
    )
    definitions = tuple(
        item for item in primitives
        if PRIMITIVE_CATALOG[item].source_kind == "NEW_DEFINITION"
    )
    semantic, hidden = _semantic_issues(memo_text, category)
    duplicates = tuple(sorted(set(duplicate_candidate_ids)))
    if duplicates:
        semantic = tuple(sorted((*semantic, "SEMANTIC_DUPLICATE")))
    mapper_failures = []
    if missing:
        mapper_failures.append("MISSING_REGISTERED_PRIMITIVE")
    if not primitives:
        mapper_failures.append("UNSUPPORTED_SEMANTIC_PATTERN")
    if semantic or hidden:
        outcome = RepresentationOutcome.SEMANTIC_REJECTION.value
    elif definitions:
        outcome = RepresentationOutcome.DEFINITION_RESOLUTION.value
    elif mathlib:
        outcome = RepresentationOutcome.MAPPER_EXTENSION_REQUIRED.value
    elif host:
        outcome = RepresentationOutcome.REGISTRY_RESOLUTION.value
    else:
        outcome = RepresentationOutcome.REPRESENTATION_EXHAUSTED.value
    characterization = (
        "ZETA_ZERO_STATISTICS"
        if "ZETA_ZERO_SEQUENCE" in primitives else
        "OPERATOR_SPECTRAL_PERTURBATION"
        if "SPECTRAL_GAP" in primitives else
        "GENERIC_UNSUPPORTED_CLAIM"
    )
    relation = {
        "DEFINITION": "PROPOSED_AUXILIARY_DEFINITION",
        "LOCAL_LEMMA": "PROPOSED_LOCAL_LEMMA",
        "CASE_SPLIT": "PROPOSED_CASE_PARTITION",
        "SUFFICIENT_CONDITION": "PROPOSED_SUFFICIENT_CONDITION",
        "EQUIVALENT_CRITERION": "PROPOSED_EQUIVALENCE",
        "OBSTRUCTION_OR_COUNTEREXAMPLE": "PROPOSED_OBSTRUCTION",
        "SPECIAL_CASE": "PROPOSED_SPECIAL_CASE",
        "BRIDGE_THEOREM": "PROPOSED_BRIDGE",
        "TOY_MODEL_ANALOGUE": "ADVISORY_TOY_ANALOGUE",
    }.get(category, "UNKNOWN_RELATION")
    body = {
        "schema_version": REPRESENTATION_ANALYSIS_VERSION,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "category": category,
        "target_obligation_id": target_obligation_id,
        "target_context_hash": target_context_hash,
        "mapper_registry_hash": mapper_capability_hash(),
        "environment_hash": environment_hash,
        "safe_characterization_id": characterization,
        "intended_relation_id": relation,
        "required_primitive_ids": primitives,
        "missing_primitive_ids": missing,
        "mathlib_source_ids": mathlib,
        "host_registry_source_ids": host,
        "new_definition_ids": definitions,
        "hidden_assumption_ids": hidden,
        "semantic_issue_ids": semantic,
        "duplicate_candidate_ids": duplicates,
        "mapper_failure_codes": tuple(mapper_failures),
        "outcome": outcome,
        "retry_allowed": outcome in {
            RepresentationOutcome.MAPPER_EXTENSION_REQUIRED.value,
            RepresentationOutcome.REGISTRY_RESOLUTION.value,
            RepresentationOutcome.DEFINITION_RESOLUTION.value,
        },
        "private_memo_consumed": True,
        "private_text_exposed": False,
        "ledger_mutation_allowed": False,
    }
    content_hash = _digest(body)
    return RepresentationGapReport(
        report_id="RGR-" + content_hash[:20],
        content_hash=content_hash,
        **{key: value for key, value in body.items() if key != "schema_version"},
    )


def persist_representation_gap_report(
    report: RepresentationGapReport,
    directory: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"{report.content_hash}.json"
    encoded = json.dumps(asdict(report), sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("REPRESENTATION_GAP_REPORT_HASH_COLLISION")
        return path
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return path


def representation_retry_fingerprint(
    report: RepresentationGapReport,
    *,
    mapper_registry_hash: str,
    environment_hash: str,
) -> str:
    return _digest({
        "candidate_id": report.candidate_id,
        "candidate_hash": report.candidate_hash,
        "prior_report_hash": report.content_hash,
        "mapper_registry_hash": mapper_registry_hash,
        "environment_hash": environment_hash,
    })


def authorize_representation_retry(
    report: RepresentationGapReport,
    *,
    mapper_registry_hash: str,
    environment_hash: str,
    consumed_fingerprints: Iterable[str],
) -> tuple[bool, str, str]:
    """Authorize one retry only after a trusted capability binding changes."""
    if not report.retry_allowed:
        return False, "", "REPRESENTATION_RETRY_NOT_ALLOWED"
    if (
        mapper_registry_hash == report.mapper_registry_hash
        and environment_hash == report.environment_hash
    ):
        return False, "", "REPRESENTATION_RETRY_HASH_UNCHANGED"
    fingerprint = representation_retry_fingerprint(
        report,
        mapper_registry_hash=mapper_registry_hash,
        environment_hash=environment_hash,
    )
    if fingerprint in set(consumed_fingerprints):
        return False, fingerprint, "REPRESENTATION_RETRY_ALREADY_CONSUMED"
    return True, fingerprint, "REPRESENTATION_RETRY_AUTHORIZED"
