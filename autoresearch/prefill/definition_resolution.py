"""Host-owned Autonomous Definition Resolution Protocol.

The protocol resolves exactly one concept per transaction.  Every input,
retrieval result, verification decision, branch, and fallback is
content-addressed; models may supply only search tags and rankings of short
candidate IDs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from autoresearch.prefill.lean_gate import _run_lean


PROTOCOL_VERSION = 1
MIGRATION_EVENT = "autonomous_definition_resolution_v1"
STORE_SCHEMA_VERSION = 1
_FORBIDDEN = re.compile(
    r"\b(?:axiom|opaque|unsafe|sorry|admit|placeholder|todo)\b|\.\.\.|…",
    re.IGNORECASE,
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=lambda item: asdict(item),
    ).encode()).hexdigest()


def _stable_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({" ".join(str(item).split()) for item in values if str(item).strip()}))


class PropertyStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DefinitionQuery:
    concept_id: str
    use_sites: tuple[str, ...]
    parent_hash: str
    typed_ir_hash: str
    auditor_hash: str
    critic_evidence_hashes: tuple[str, ...]
    counterexample_evidence_hashes: tuple[str, ...]
    domain: str
    codomain: str
    required_properties: tuple[str, ...]
    hard_properties: tuple[str, ...]
    theorem_dependencies: tuple[str, ...]
    environment_hash: str
    prior_failure_hashes: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class SourceProvenance:
    source_id: str
    source_kind: str
    status: str
    locator: str
    environment_hash: str
    evidence_hash: str
    detail: str = ""


@dataclass(frozen=True)
class TypedDefinitionCandidate:
    short_id: str
    concept_id: str
    branch_id: str
    declaration_name: str
    domain: str
    codomain: str
    binders: tuple[str, ...]
    typed_expression: str
    dependencies: tuple[str, ...]
    provenance: SourceProvenance
    required_properties: tuple[str, ...]
    claimed_properties: tuple[str, ...]
    existing_reference: str = ""
    declaration_kind: str = "def"

    @property
    def content_hash(self) -> str:
        payload = asdict(self)
        payload.pop("short_id", None)
        return digest(payload)


@dataclass(frozen=True)
class PropertyEvidence:
    property_id: str
    status: str
    evidence_kind: str
    evidence_ref: str
    detail: str = ""


@dataclass(frozen=True)
class DefinitionInterpretationBranch:
    branch_id: str
    query_hash: str
    candidate_hash: str
    rewritten_parent_hash: str
    rewritten_typed_ir_hash: str
    assumptions: tuple[str, ...]
    theorem_dependencies: tuple[str, ...]
    falsification_criteria: tuple[str, ...]
    success_criteria: tuple[str, ...]
    equivalence_obligation_hash: str


@dataclass(frozen=True)
class DefinitionInterface:
    interface_id: str
    query_hash: str
    concept_id: str
    operator_type: str
    minimum_properties: tuple[str, ...]
    conditional_parent_hash: str
    missing_axiom_obligations: tuple[str, ...]
    viable: bool


@dataclass(frozen=True)
class ResolutionResult:
    status: str
    concept_id: str
    query_hash: str
    source_statuses: tuple[SourceProvenance, ...]
    candidate_hashes: tuple[str, ...]
    branch_hashes: tuple[str, ...]
    property_statuses: Mapping[str, Mapping[str, str]]
    exhaustion_hash: str
    interface_hash: str
    committed_candidate_hash: str
    store_hash_before: str
    store_hash_after: str
    environment_hash_before: str
    environment_hash_after: str
    reason: str


SourceAdapter = Callable[
    [DefinitionQuery, Path, Mapping[str, Any]],
    tuple[list[Mapping[str, Any]], SourceProvenance],
]


@dataclass
class DefinitionSourceRegistry:
    adapters: dict[str, SourceAdapter] = field(default_factory=dict)

    def register(self, source_id: str, adapter: SourceAdapter) -> None:
        if not source_id or source_id in self.adapters:
            raise ValueError("definition source IDs must be unique and non-empty")
        self.adapters[source_id] = adapter

    def retrieve(
        self,
        query: DefinitionQuery,
        project_root: Path,
        context: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]], tuple[SourceProvenance, ...]]:
        raw: list[Mapping[str, Any]] = []
        statuses = []
        for source_id in sorted(self.adapters):
            candidates, status = self.adapters[source_id](query, project_root, context)
            if status.source_id != source_id:
                raise ValueError("source adapter returned mismatched provenance")
            raw.extend(candidates)
            statuses.append(status)
        return raw, tuple(statuses)


def build_definition_query(
    gap: Mapping[str, Any],
    *,
    parent_hash: str,
    typed_ir_hash: str,
    auditor_hash: str,
    critic_evidence_hashes: Iterable[str],
    counterexample_evidence_hashes: Iterable[str],
    theorem_dependencies: Iterable[str],
    environment_hash: str,
    prior_failure_hashes: Iterable[str] = (),
) -> DefinitionQuery:
    concept = str(gap.get("definition_id") or gap.get("concept_id") or "").strip()
    if not concept:
        raise ValueError("definition query requires exactly one concept ID")
    uses = gap.get("use_sites") or gap.get("usage_sites") or gap.get("symbol_ids") or ()
    required = gap.get("required_properties") or gap.get("properties") or ()
    hard = gap.get("hard_properties") or required
    domain = str(gap.get("domain") or gap.get("required_domain") or "")
    codomain = str(
        gap.get("codomain") or gap.get("required_codomain")
        or gap.get("required_type_id") or "",
    )
    return DefinitionQuery(
        concept_id=concept,
        use_sites=_stable_tuple(uses),
        parent_hash=str(parent_hash),
        typed_ir_hash=str(typed_ir_hash),
        auditor_hash=str(auditor_hash),
        critic_evidence_hashes=_stable_tuple(critic_evidence_hashes),
        counterexample_evidence_hashes=_stable_tuple(counterexample_evidence_hashes),
        domain=" ".join(domain.split()),
        codomain=" ".join(codomain.split()),
        required_properties=_stable_tuple(required),
        hard_properties=_stable_tuple(hard),
        theorem_dependencies=_stable_tuple(theorem_dependencies),
        environment_hash=str(environment_hash),
        prior_failure_hashes=_stable_tuple(prior_failure_hashes),
    )


def empty_store(base_environment_hash: str) -> dict[str, Any]:
    store = {
        "schema_version": STORE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "environment_base_hash": base_environment_hash,
        "queries": {},
        "sources": {},
        "candidates": {},
        "properties": {},
        "branches": {},
        "exhaustions": {},
        "interfaces": {},
        "commits": {},
        "historical_audit": {},
    }
    store["store_hash"] = store_hash(store)
    store["environment_hash"] = resolution_environment_hash(store)
    return store


def store_hash(store: Mapping[str, Any]) -> str:
    return digest({
        key: store.get(key, {})
        for key in (
            "schema_version", "protocol_version", "queries", "sources",
            "candidates", "properties", "branches", "exhaustions",
            "interfaces", "commits", "historical_audit",
        )
    })


def resolution_environment_hash(store: Mapping[str, Any]) -> str:
    return digest({
        "base": store.get("environment_base_hash", ""),
        "commits": store.get("commits", {}),
    })


def load_resolution_store(path: Path, base_environment_hash: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        return empty_store(base_environment_hash)
    store = json.loads(path.read_text(encoding="utf-8"))
    if store.get("schema_version") != STORE_SCHEMA_VERSION:
        raise ValueError("definition resolution store schema mismatch")
    if store.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("definition resolution protocol mismatch")
    if store.get("store_hash") != store_hash(store):
        raise ValueError("definition resolution store hash mismatch")
    if store.get("environment_hash") != resolution_environment_hash(store):
        raise ValueError("definition resolution environment hash mismatch")
    return store


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _status(
    source_id: str,
    kind: str,
    status: str,
    query: DefinitionQuery,
    *,
    locator: str = "",
    evidence: object = (),
    detail: str = "",
) -> SourceProvenance:
    return SourceProvenance(
        source_id, kind, status, locator, query.environment_hash,
        digest(evidence), detail,
    )


def _mathlib_source(query, project_root, context):
    matches = []
    for item in context.get("mathlib_declarations", ()):
        if query.concept_id.lower().replace("def_", "") in str(item.get("tags", "")).lower():
            matches.append(item)
    return matches, _status(
        "mathlib", "LOCAL_MATHLIB", "QUERIED", query,
        locator=str(project_root), evidence=matches,
        detail=f"{len(matches)} typed declaration matches",
    )


def _theorem_cards_source(query, project_root, context):
    matches = [
        item for item in context.get("definition_cards", ())
        if query.concept_id in set(map(str, item.get("concept_ids", ())))
    ]
    return matches, _status(
        "pinned_cards", "PINNED_CARD", "QUERIED", query,
        locator="content-addressed-card-index", evidence=matches,
        detail=f"{len(matches)} cards matched",
    )


def _local_corpus_source(query, project_root, context):
    root = Path(os.environ.get("KAKEYA_OPROOFS_ROOT", "")).expanduser()
    if not str(os.environ.get("KAKEYA_OPROOFS_ROOT", "")).strip() or not root.exists():
        return [], _status(
            "oproofs", "LOCAL_PROOF_CORPUS", "UNAVAILABLE", query,
            detail="KAKEYA_OPROOFS_ROOT is not configured or does not exist",
        )
    cards = context.get("oproofs_cards", ())
    matches = [
        item for item in cards
        if query.concept_id in set(map(str, item.get("concept_ids", ())))
    ]
    return matches, _status(
        "oproofs", "LOCAL_PROOF_CORPUS", "QUERIED", query,
        locator=str(root), evidence=matches,
    )


def _local_publication_source(query, project_root, context):
    matches = [
        item for item in context.get("publication_cards", ())
        if query.concept_id in set(map(str, item.get("concept_ids", ())))
    ]
    return matches, _status(
        "publications", "LOCAL_PUBLICATION_CARD", "QUERIED", query,
        locator="local-definition-cards", evidence=matches,
        detail=f"{len(matches)} cards matched",
    )


def _historical_source(query, project_root, context):
    matches = [
        item for item in context.get("validated_history", ())
        if item.get("semantic_validation") == "VERIFIED"
        and str(item.get("concept_id")) == query.concept_id
    ]
    return matches, _status(
        "validated_history", "VALIDATED_HISTORICAL", "QUERIED", query,
        locator="resolution-store:historical_audit", evidence=matches,
        detail=f"{len(matches)} semantically validated definitions",
    )


def _typed_synthesis_source(query, project_root, context):
    operators = context.get("typed_operator_synthesis", ())
    matches = [
        item for item in operators
        if str(item.get("concept_id")) == query.concept_id
        and (not query.codomain or str(item.get("codomain")) == query.codomain)
    ]
    return matches, _status(
        "typed_synthesis", "HOST_TYPED_SYNTHESIS", "QUERIED", query,
        locator="registered-types-and-operators", evidence=matches,
        detail=f"{len(matches)} inhabitants constructed",
    )


def default_source_registry() -> DefinitionSourceRegistry:
    registry = DefinitionSourceRegistry()
    registry.register("mathlib", _mathlib_source)
    registry.register("oproofs", _local_corpus_source)
    registry.register("pinned_cards", _theorem_cards_source)
    registry.register("publications", _local_publication_source)
    registry.register("typed_synthesis", _typed_synthesis_source)
    registry.register("validated_history", _historical_source)
    return registry


def normalize_candidates(
    query: DefinitionQuery,
    raw_candidates: Iterable[Mapping[str, Any]],
    source_statuses: Iterable[SourceProvenance],
) -> tuple[TypedDefinitionCandidate, ...]:
    provenance = {item.source_id: item for item in source_statuses}
    normalized = []
    for raw in raw_candidates:
        source_id = str(raw.get("source_id", ""))
        if source_id not in provenance:
            raise ValueError("candidate has no queried source provenance")
        expression = " ".join(str(raw.get("typed_expression", "")).split())
        reference = " ".join(str(raw.get("existing_reference", "")).split())
        if not expression and not reference:
            continue
        candidate = TypedDefinitionCandidate(
            short_id="",
            concept_id=query.concept_id,
            branch_id=str(raw.get("branch_id", "")),
            declaration_name=str(raw.get("declaration_name", "")).strip(),
            domain=" ".join(str(raw.get("domain", query.domain)).split()),
            codomain=" ".join(str(raw.get("codomain", query.codomain)).split()),
            binders=_stable_tuple(raw.get("binders", ())),
            typed_expression=expression,
            dependencies=_stable_tuple(raw.get("dependencies", ())),
            provenance=provenance[source_id],
            required_properties=query.required_properties,
            claimed_properties=_stable_tuple(raw.get("claimed_properties", ())),
            existing_reference=reference,
            declaration_kind=str(raw.get("declaration_kind", "def")),
        )
        normalized.append(candidate)
    dedup = {item.content_hash: item for item in normalized}
    ordered = [dedup[key] for key in sorted(dedup)]
    return tuple(
        replace(item, short_id=f"C{index + 1}")
        for index, item in enumerate(ordered)
    )


def lean_source(candidate: TypedDefinitionCandidate) -> str:
    if candidate.existing_reference:
        return f"#check {candidate.existing_reference}"
    if candidate.declaration_kind not in {"def", "abbrev"}:
        raise ValueError("candidate declaration kind must be def or abbrev")
    if not candidate.declaration_name:
        raise ValueError("generated declaration requires a name")
    binders = " ".join(candidate.binders)
    result_type = f" : {candidate.codomain}" if candidate.codomain else ""
    return (
        f"{candidate.declaration_kind} {candidate.declaration_name}"
        f"{(' ' + binders) if binders else ''}{result_type} := "
        f"{candidate.typed_expression}"
    )


def compile_candidate(
    candidate: TypedDefinitionCandidate,
    *,
    project_root: Path,
    environment_hash: str,
) -> tuple[bool, str, str]:
    source = lean_source(candidate)
    if _FORBIDDEN.search(source) or re.search(r"(^|\n)\s*(?:variable|noncomputable section)\b", source):
        return False, "HIDDEN_ASSUMPTION_OR_PLACEHOLDER", digest(source)
    content = (
        "import Mathlib\n\nset_option autoImplicit false\n\n"
        f"-- environment:{environment_hash}\n{source}\n"
    )
    run = _run_lean(content, project_root=project_root, timeout_s=120.0)
    if run.timed_out:
        return False, "LEAN_TIMEOUT", digest(content)
    if run.returncode:
        return False, "LEAN_REJECTED:" + run.output[-800:], digest(content)
    return True, "ELABORATED", digest(content)


def verify_properties(
    query: DefinitionQuery,
    candidate: TypedDefinitionCandidate,
    *,
    evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[PropertyEvidence, ...]:
    results = []
    for property_id in query.required_properties:
        record = evidence.get(property_id, {})
        candidate_hashes = set(map(str, record.get("candidate_hashes", ())))
        applies = not candidate_hashes or candidate.content_hash in candidate_hashes
        status = str(record.get("status", "UNKNOWN")).upper() if applies else "UNKNOWN"
        if status not in {item.value for item in PropertyStatus}:
            status = PropertyStatus.UNKNOWN.value
        evidence_ref = str(record.get("evidence_ref", ""))
        evidence_kind = str(record.get("evidence_kind", "NONE"))
        if status == PropertyStatus.VERIFIED.value and not evidence_ref:
            status = PropertyStatus.UNKNOWN.value
        results.append(PropertyEvidence(
            property_id, status, evidence_kind, evidence_ref,
            str(record.get("detail", "")),
        ))
    return tuple(results)


def _branch(query: DefinitionQuery, candidate: TypedDefinitionCandidate) -> DefinitionInterpretationBranch:
    candidate_hash = candidate.content_hash
    branch_id = "B-" + candidate_hash[:16]
    rewritten_parent = digest({
        "parent": query.parent_hash, "concept": query.concept_id,
        "candidate": candidate_hash,
    })
    rewritten_ir = digest({
        "typed_ir": query.typed_ir_hash, "candidate": candidate_hash,
    })
    equivalence = digest({
        "original_parent": query.parent_hash,
        "rewritten_parent": rewritten_parent,
        "obligation": "semantic_equivalence_or_explicit_reduction",
    })
    return DefinitionInterpretationBranch(
        branch_id, query.fingerprint, candidate_hash, rewritten_parent,
        rewritten_ir, (), query.theorem_dependencies,
        tuple(f"refute:{item}" for item in query.hard_properties),
        tuple(f"verify:{item}" for item in query.hard_properties),
        equivalence,
    )


def synthesize_interface(query: DefinitionQuery) -> DefinitionInterface:
    operator_type = (
        f"{query.domain} → {query.codomain}"
        if query.domain and query.codomain else query.codomain or query.domain
    )
    # Auditor ontology IDs constrain search but are not executable Lean types.
    # A list of use-site symbols alone cannot justify inventing a signature.
    viable = bool(
        operator_type
        and not re.search(r"(^|[ →])(TYPE|DOM)_[A-Z0-9_]+", operator_type)
    )
    obligations = tuple(f"AXIOM_OBLIGATION:{item}" for item in query.hard_properties)
    interface_id = digest({
        "query": query.fingerprint,
        "operator_type": operator_type,
        "minimum_properties": query.hard_properties,
        "uses": query.use_sites,
    })
    return DefinitionInterface(
        interface_id, query.fingerprint, query.concept_id,
        operator_type or "USAGE_DERIVED_UNKNOWN_TYPE",
        query.hard_properties,
        digest({"parent": query.parent_hash, "interface": interface_id}),
        obligations, viable,
    )


def resolve_one_concept(
    *,
    query: DefinitionQuery,
    store_path: Path,
    project_root: Path,
    source_registry: DefinitionSourceRegistry | None = None,
    source_context: Mapping[str, Any] | None = None,
    property_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    ranked_short_ids: Iterable[str] = (),
) -> ResolutionResult:
    """Run one crash-safe, idempotent definition-resolution transaction."""
    store = load_resolution_store(store_path, query.environment_hash)
    before = store_hash(store)
    before_environment = resolution_environment_hash(store)
    query_hash = query.fingerprint
    previous = store["queries"].get(query_hash)
    if previous and previous.get("terminal"):
        return ResolutionResult(
            "IDENTICAL_QUERY_EXHAUSTED" if previous["status"] == "EXHAUSTED" else "IDEMPOTENT_REPLAY",
            query.concept_id, query_hash, (), tuple(previous.get("candidate_hashes", ())),
            tuple(previous.get("branch_hashes", ())),
            previous.get("property_statuses", {}),
            str(previous.get("exhaustion_hash", "")),
            str(previous.get("interface_hash", "")),
            str(previous.get("committed_candidate_hash", "")),
            before, before, before_environment, before_environment,
            "identical terminal query cannot rerun without semantic input change",
        )
    registry = source_registry or default_source_registry()
    context = dict(source_context or {})
    raw, statuses = registry.retrieve(query, project_root, context)
    candidates = normalize_candidates(query, raw, statuses)
    rankings = tuple(ranked_short_ids)
    if rankings and (
        len(set(rankings)) != len(rankings)
        or not set(rankings) <= {item.short_id for item in candidates}
    ):
        raise ValueError("model ranking may contain scoped short candidate IDs only")
    candidate_order = {item: index for index, item in enumerate(rankings)}
    candidates = tuple(sorted(
        candidates, key=lambda item: (candidate_order.get(item.short_id, len(rankings)), item.content_hash),
    ))
    property_map: dict[str, dict[str, str]] = {}
    survivors = []
    rejections = {}
    for candidate in candidates:
        compiled, compile_status, compilation_hash = compile_candidate(
            candidate, project_root=project_root,
            environment_hash=query.environment_hash,
        )
        evidence = verify_properties(
            query, candidate, evidence=property_evidence or {},
        )
        statuses_for_candidate = {item.property_id: item.status for item in evidence}
        property_map[candidate.content_hash] = statuses_for_candidate
        hard_ok = all(
            statuses_for_candidate.get(item) == PropertyStatus.VERIFIED.value
            for item in query.hard_properties
        )
        if compiled and hard_ok:
            survivors.append(candidate)
        else:
            rejections[candidate.content_hash] = {
                "compile_status": compile_status,
                "compilation_hash": compilation_hash,
                "unmet_hard_properties": [
                    item for item in query.hard_properties
                    if statuses_for_candidate.get(item) != PropertyStatus.VERIFIED.value
                ],
            }
        store["candidates"][candidate.content_hash] = asdict(candidate)
        store["properties"][candidate.content_hash] = [asdict(item) for item in evidence]
    branches = tuple(_branch(query, item) for item in survivors)
    branch_hashes = tuple(digest(asdict(item)) for item in branches)
    for branch_hash, branch in zip(branch_hashes, branches):
        store["branches"][branch_hash] = asdict(branch)
    exhaustion_hash = ""
    interface_hash = ""
    committed_hash = ""
    reason = ""
    if len(survivors) > 1:
        status = "PARENT_STATEMENT_UNDERSPECIFIED"
        reason = "multiple hard-feasible interpretations require branch tournament"
    elif len(survivors) == 1:
        candidate = survivors[0]
        committed_hash = candidate.content_hash
        store["commits"][query.concept_id] = {
            "query_hash": query_hash,
            "candidate_hash": committed_hash,
            "provenance": asdict(candidate.provenance),
            "properties": property_map[committed_hash],
            "lean_source_hash": digest(lean_source(candidate)),
            "committed_at": time.time(),
        }
        status = "COMMITTED"
        reason = "Lean elaborated, hard properties verified, provenance complete"
    else:
        interface = synthesize_interface(query)
        interface_hash = digest(asdict(interface))
        store["interfaces"][interface_hash] = asdict(interface)
        exhaustion = {
            "schema_version": 1,
            "query_hash": query_hash,
            "concept_id": query.concept_id,
            "parent_hash": query.parent_hash,
            "environment_hash": query.environment_hash,
            "sources": [asdict(item) for item in statuses],
            "candidate_hashes": [item.content_hash for item in candidates],
            "rejections": rejections,
            "unmet_properties": list(query.hard_properties),
            "interface_hash": interface_hash,
        }
        exhaustion_hash = digest(exhaustion)
        store["exhaustions"][exhaustion_hash] = exhaustion
        status = "INTERFACE_REQUIRED" if interface.viable else "PARENT_STATEMENT_UNDERSPECIFIED"
        reason = (
            "concrete retrieval exhausted; conditional theorem requires explicit interface axioms"
            if interface.viable
            else "no concrete definition, viable typed interface, or equivalent rewrite"
        )
    store["sources"][query_hash] = [asdict(item) for item in statuses]
    query_record = {
        **asdict(query),
        "query_hash": query_hash,
        "status": "EXHAUSTED" if not survivors else status,
        "terminal": True,
        "candidate_hashes": [item.content_hash for item in candidates],
        "branch_hashes": list(branch_hashes),
        "property_statuses": property_map,
        "exhaustion_hash": exhaustion_hash,
        "interface_hash": interface_hash,
        "committed_candidate_hash": committed_hash,
        "completed_at": time.time(),
    }
    store["queries"][query_hash] = query_record
    store["store_hash"] = store_hash(store)
    store["environment_hash"] = resolution_environment_hash(store)
    _atomic_write(store_path, store)
    return ResolutionResult(
        status, query.concept_id, query_hash, statuses,
        tuple(item.content_hash for item in candidates), branch_hashes,
        property_map, exhaustion_hash, interface_hash, committed_hash,
        before, store["store_hash"], before_environment,
        store["environment_hash"], reason,
    )


def migrate_legacy_registry(
    legacy: Mapping[str, Any],
    *,
    base_environment_hash: str,
) -> dict[str, Any]:
    """Import legacy definitions as audit-only pending semantic validation."""
    store = empty_store(base_environment_hash)
    for name, record in sorted(legacy.get("definitions", {}).items()):
        concept = str(record.get("gap_id", ""))
        obligations = {
            "DEF_POLE_NEIGHBORHOOD": ("full_vs_punctured_neighborhood",),
            "DEF_FUNCTION_BINDING": ("partial_sum_index_binding",),
            "DEF_GROWTH_ORDER": ("growth_order_semantics",),
        }.get(concept, ("semantic_equivalence_to_actual_use",))
        key = digest({"name": name, "record": record})
        store["historical_audit"][key] = {
            "concept_id": concept,
            "declaration_name": str(name),
            "legacy_record": dict(record),
            "semantic_validation": "PENDING",
            "property_obligations": list(obligations),
            "audit_only": True,
        }
    store["store_hash"] = store_hash(store)
    store["environment_hash"] = resolution_environment_hash(store)
    return store
