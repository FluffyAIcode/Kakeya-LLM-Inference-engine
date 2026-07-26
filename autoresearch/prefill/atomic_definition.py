"""Host-owned atomic definition transactions and mathematical progress."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from autoresearch.prefill.lean_gate import _run_lean


CAPABILITY_VERSION = 1
MIGRATION_EVENT = "atomic_define_one_concept_progress_v1"
PROGRESS_FIELDS = (
    "definitions_added",
    "existing_definitions_resolved",
    "lemmas_proved",
    "accepted_children",
    "subgoals_closed",
    "verified_counterexamples",
)
_PLACEHOLDER = re.compile(r"\b(?:True|sorry|admit|placeholder|todo)\b|\.\.\.|…")


@dataclass(frozen=True)
class ProgressVector:
    definitions_added: int = 0
    existing_definitions_resolved: int = 0
    lemmas_proved: int = 0
    accepted_children: int = 0
    subgoals_closed: int = 0
    verified_counterexamples: int = 0

    @property
    def total(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class TypedDefinitionCandidate:
    short_id: str
    gap_id: str
    declaration_name: str
    lean_source: str
    dependency_ids: tuple[str, ...]
    semantic_bindings: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class GapDecision:
    gap_id: str
    classification: str
    dependencies: tuple[str, ...]
    candidates: tuple[TypedDefinitionCandidate, ...] = ()
    existing_reference: str = ""
    typed_entry: str = ""
    reason: str = ""


@dataclass(frozen=True)
class DefineOneResult:
    status: str
    gap_id: str
    classification: str
    candidate_count: int
    candidate_id: str
    lean_status: str
    lean_source: str
    artifact_hash: str
    registry_hash_before: str
    registry_hash_after: str
    environment_hash_before: str
    environment_hash_after: str
    progress: ProgressVector
    reason: str = ""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=lambda item: asdict(item),
    ).encode()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def empty_definition_registry(environment_hash: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CAPABILITY_VERSION,
        "definitions": {},
        "resolved_gaps": {},
        "environment_base_hash": environment_hash,
    }
    payload["registry_hash"] = registry_hash(payload)
    payload["environment_hash"] = environment_registry_hash(payload)
    return payload


def registry_hash(registry: Mapping[str, Any]) -> str:
    return _digest({
        "schema_version": registry.get("schema_version", CAPABILITY_VERSION),
        "definitions": registry.get("definitions", {}),
        "resolved_gaps": registry.get("resolved_gaps", {}),
    })


def environment_registry_hash(registry: Mapping[str, Any]) -> str:
    return _digest({
        "base": registry.get("environment_base_hash", ""),
        "registry": registry_hash(registry),
    })


def load_definition_registry(path: Path, environment_hash: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        return empty_definition_registry(environment_hash)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CAPABILITY_VERSION:
        raise ValueError("definition registry schema mismatch")
    if raw.get("registry_hash") != registry_hash(raw):
        raise ValueError("definition registry hash mismatch")
    if raw.get("environment_hash") != environment_registry_hash(raw):
        raise ValueError("definition environment hash mismatch")
    return raw


def classify_gap(
    gap: Mapping[str, Any],
    *,
    dependency_graph: Mapping[str, Iterable[str]],
    theorem_card_ids: Iterable[str],
) -> GapDecision:
    """Classify an audited gap without permitting model-authored definitions."""
    gap_id = str(gap.get("definition_id", "")).strip()
    dependencies = tuple(sorted(str(item) for item in dependency_graph.get(
        gap_id, (),
    )))
    if gap_id == "DEF_EPSILON":
        return GapDecision(
            gap_id, "TYPED_RESTRICTION", dependencies,
            typed_entry="epsilon : ℝ; assumption 0 < epsilon",
            reason="positivity is a binder assumption, not a definition",
        )
    if gap_id == "DEF_GENUS":
        return GapDecision(
            gap_id, "TYPED_RESTRICTION", dependencies,
            typed_entry="genus : ℕ",
            reason="genus is a typed Nat binder, not a standalone definition",
        )
    if gap_id == "DEF_SERIES_CONVERGENCE":
        return GapDecision(
            gap_id, "EXISTING_REFERENCE", dependencies,
            existing_reference="Mathlib:LocallyUniformly",
            reason="use the imported Mathlib convergence notion",
        )
    candidates = _candidates_for_gap(gap_id, dependencies)
    if not candidates:
        return GapDecision(
            gap_id, "NO_TYPED_DEFINITION_CANDIDATE", dependencies,
            reason="no registered constructor/operator can inhabit the audited type",
        )
    return GapDecision(gap_id, "MISSING_CONCEPT", dependencies, candidates)


def _candidate(
    short_id: str,
    gap_id: str,
    name: str,
    source: str,
    dependencies: tuple[str, ...],
    bindings: tuple[str, ...],
) -> TypedDefinitionCandidate:
    return TypedDefinitionCandidate(
        short_id, gap_id, name, source, dependencies, bindings,
    )


def _candidates_for_gap(
    gap_id: str,
    dependencies: tuple[str, ...],
) -> tuple[TypedDefinitionCandidate, ...]:
    specs: dict[str, tuple[tuple[str, str, str, tuple[str, ...]], ...]] = {
        "DEF_POLE_NEIGHBORHOOD": (
            ("A", "kakeyaPoleNeighborhood",
             "def kakeyaPoleNeighborhood (a : ℂ) (ε : ℝ) : Set ℂ := Metric.ball a ε",
             ("Metric.ball", "Set", "Complex")),
            ("B", "kakeyaPuncturedPoleNeighborhood",
             "def kakeyaPuncturedPoleNeighborhood (a : ℂ) (ε : ℝ) : Set ℂ := Metric.ball a ε \\ {a}",
             ("Metric.ball", "Set.diff", "Set.singleton", "Complex")),
        ),
        "DEF_FUNCTION_BINDING": (
            ("A", "kakeyaPartialSum",
             "def kakeyaPartialSum (term : ℕ → ℂ → ℂ) (n : ℕ) (z : ℂ) : ℂ := ∑ k ∈ Finset.range n, term k z",
             ("Finset.sum", "Finset.range", "Complex")),
            ("B", "kakeyaTermFamily",
             "abbrev kakeyaTermFamily := ℕ → ℂ → ℂ",
             ("Nat", "Complex", "Function")),
        ),
        "DEF_GROWTH_ORDER": (
            ("A", "kakeyaHasGrowthOrder",
             "def kakeyaHasGrowthOrder (f : ℂ → ℂ) (ρ : ℝ) : Prop := ∃ C : ℝ, 0 < C ∧ ∀ z, ‖f z‖ ≤ Real.exp (C * ‖z‖ ^ ρ)",
             ("Exists", "Norm.norm", "Real.exp", "Complex")),
            ("B", "kakeyaGrowthBound",
             "def kakeyaGrowthBound (f : ℂ → ℂ) (p : ℕ) : Prop := ∃ C : ℝ, 0 < C ∧ ∀ z, ‖f z‖ ≤ Real.exp (C * ‖z‖ ^ p)",
             ("Exists", "Norm.norm", "Real.exp", "Complex")),
        ),
        "DEF_CRITICAL_DENSITY": (
            ("A", "kakeyaCriticalDensity",
             "def kakeyaCriticalDensity (p : ℕ) : ℝ := (p + 1 : ℕ)",
             ("Nat.cast", "Nat.add")),
        ),
    }
    return tuple(
        _candidate(short_id, gap_id, name, source, dependencies, bindings)
        for short_id, name, source, bindings in specs.get(gap_id, ())
    )


def dependency_graph_for_audit(
    missing: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Derive typed dependencies from symbol/type evidence in the audit."""
    ids = {
        str(item.get("definition_id", "")): item for item in missing
        if item.get("definition_id")
    }
    graph: dict[str, tuple[str, ...]] = {}
    symbol_owner: dict[str, str] = {}
    for gap_id, item in ids.items():
        for symbol in item.get("symbol_ids", ()):
            symbol_owner.setdefault(str(symbol), gap_id)
    for gap_id, item in ids.items():
        dependencies: set[str] = set()
        for symbol in item.get("symbol_ids", ()):
            owner = symbol_owner.get(str(symbol))
            if owner and owner != gap_id:
                dependencies.add(owner)
        # Typed environment evidence refines ambiguous shared-symbol ownership.
        required_type = str(item.get("required_type_id", ""))
        if required_type == "TYPE_CANONICAL_PRODUCT_BINDING":
            dependencies.discard("DEF_SEQUENCE_DENSITY")
            dependencies.add("DEF_POLE_NEIGHBORHOOD")
        elif required_type == "TYPE_CONVERGENCE_MODE":
            dependencies.add("DEF_FUNCTION_BINDING")
            dependencies.discard("DEF_SEQUENCE_DENSITY")
        elif required_type == "TYPE_ENTIRE_FUNCTION_ORDER":
            dependencies.add("DEF_FUNCTION_BINDING")
            dependencies.discard("DEF_GENUS")
        elif required_type == "TYPE_SEQUENCE_DENSITY":
            dependencies.add("DEF_FUNCTION_BINDING")
        elif required_type == "TYPE_DENSITY_THRESHOLD":
            dependencies.add("DEF_SEQUENCE_DENSITY")
            dependencies.discard("DEF_FUNCTION_BINDING")
        graph[gap_id] = tuple(sorted(item for item in dependencies if item in ids))
    return graph


def first_dependency_closed_gap(
    missing: Iterable[Mapping[str, Any]],
    *,
    dependency_graph: Mapping[str, Iterable[str]],
    resolved_gap_ids: Iterable[str],
) -> Mapping[str, Any] | None:
    resolved = set(resolved_gap_ids)
    candidates = [
        item for item in missing
        if str(item.get("definition_id", "")) not in resolved
        and set(dependency_graph.get(str(item.get("definition_id", "")), ()))
        <= resolved
    ]
    if not candidates:
        return None
    order = {
        "DEF_EPSILON": 0,
        "DEF_GENUS": 1,
        "DEF_POLE_NEIGHBORHOOD": 2,
        "DEF_FUNCTION_BINDING": 3,
        "DEF_SERIES_CONVERGENCE": 4,
        "DEF_GROWTH_ORDER": 5,
        "DEF_SEQUENCE_DENSITY": 6,
        "DEF_CRITICAL_DENSITY": 7,
    }
    return min(candidates, key=lambda item: (
        order.get(str(item.get("definition_id", "")), 100),
        str(item.get("definition_id", "")),
    ))


def _validate_candidate(
    candidate: TypedDefinitionCandidate,
    project_root: Path,
) -> tuple[bool, str]:
    if _PLACEHOLDER.search(candidate.lean_source):
        return False, "PLACEHOLDER_OR_TRIVIAL_DEFINITION"
    if re.search(
        r"(^|\n)\s*(?:axiom|variable|opaque|unsafe|noncomputable section)\b",
        candidate.lean_source,
    ):
        return False, "HIDDEN_ASSUMPTION_OR_FORBIDDEN_COMMAND"
    if len(re.findall(r"\b(?:def|abbrev)\b", candidate.lean_source)) != 1:
        return False, "EXPECTED_EXACTLY_ONE_DEFINITION"
    if not candidate.semantic_bindings:
        return False, "UNBOUND_SEMANTICS"
    content = (
        "import Mathlib\n\n"
        "set_option autoImplicit false\n\n"
        + candidate.lean_source
        + "\n"
    )
    run = _run_lean(content, project_root=project_root, timeout_s=120.0)
    if run.timed_out:
        return False, "LEAN_DEFINITION_TIMEOUT"
    if run.returncode != 0:
        return False, "LEAN_DEFINITION_INVALID:" + run.output[-1000:]
    return True, "ELABORATED"


def validate_typed_definition_candidate(
    candidate: TypedDefinitionCandidate,
    *,
    project_root: Path,
) -> tuple[bool, str]:
    return _validate_candidate(candidate, project_root)


def define_one_concept(
    *,
    gap: Mapping[str, Any],
    definition_auditor_hash: str,
    dependency_graph: Mapping[str, Iterable[str]],
    registry_path: Path,
    project_root: Path,
    theorem_card_ids: Iterable[str],
    current_environment_hash: str,
    selected_candidate_id: str = "",
) -> DefineOneResult:
    """Resolve exactly one audited gap in one crash-safe idempotent commit."""
    gap_id = str(gap.get("definition_id", ""))
    registry = load_definition_registry(registry_path, current_environment_hash)
    before_registry = registry_hash(registry)
    before_environment = environment_registry_hash(registry)
    previous = registry["resolved_gaps"].get(gap_id)
    if previous is not None:
        return DefineOneResult(
            "IDEMPOTENT_REPLAY", gap_id, previous["classification"],
            int(previous.get("candidate_count", 0)),
            str(previous.get("candidate_id", "")),
            str(previous.get("lean_status", "")),
            str(previous.get("lean_source", "")),
            str(previous["artifact_hash"]),
            before_registry, before_registry, before_environment,
            before_environment, ProgressVector(),
            "gap already resolved by the same content-addressed transaction",
        )
    decision = classify_gap(
        gap,
        dependency_graph=dependency_graph,
        theorem_card_ids=theorem_card_ids,
    )
    if any(
        dependency not in registry["resolved_gaps"]
        for dependency in decision.dependencies
    ):
        return DefineOneResult(
            "DEPENDENCY_OPEN", gap_id, decision.classification,
            len(decision.candidates), "", "NOT_RUN", "", "",
            before_registry, before_registry, before_environment,
            before_environment, ProgressVector(),
            "typed dependencies are unresolved",
        )
    candidate: TypedDefinitionCandidate | None = None
    lean_status = "NOT_REQUIRED"
    lean_source = ""
    progress = ProgressVector(existing_definitions_resolved=1)
    if decision.classification == "NO_TYPED_DEFINITION_CANDIDATE":
        return DefineOneResult(
            "NO_TYPED_DEFINITION_CANDIDATE", gap_id, decision.classification,
            0, "", "NOT_RUN", "", _digest({
                "gap_id": gap_id, "audit": definition_auditor_hash,
                "reason": decision.reason,
            }), before_registry, before_registry, before_environment,
            before_environment, ProgressVector(), decision.reason,
        )
    if decision.classification == "MISSING_CONCEPT":
        candidate_map = {item.short_id: item for item in decision.candidates}
        candidate = (
            candidate_map.get(selected_candidate_id)
            if selected_candidate_id else decision.candidates[0]
        )
        if candidate is None:
            raise ValueError("candidate selection must be a registered short ID")
        valid, lean_status = _validate_candidate(candidate, project_root)
        lean_source = candidate.lean_source
        if not valid:
            return DefineOneResult(
                "LEAN_DEFINITION_REJECTED", gap_id, decision.classification,
                len(decision.candidates), candidate.short_id, lean_status,
                lean_source, candidate.content_hash, before_registry,
                before_registry, before_environment, before_environment,
                ProgressVector(), lean_status,
            )
        progress = ProgressVector(definitions_added=1)
    artifact = {
        "schema_version": CAPABILITY_VERSION,
        "action": "DEFINE_ONE_CONCEPT",
        "gap_id": gap_id,
        "classification": decision.classification,
        "definition_auditor_hash": definition_auditor_hash,
        "dependencies": list(decision.dependencies),
        "candidate_count": len(decision.candidates),
        "candidate_id": candidate.short_id if candidate else "",
        "candidate_hashes": [
            item.content_hash for item in decision.candidates
        ],
        "lean_source": lean_source,
        "lean_status": lean_status,
        "existing_reference": decision.existing_reference,
        "typed_entry": decision.typed_entry,
        "semantic_bindings": (
            list(candidate.semantic_bindings) if candidate else []
        ),
        "environment_hash_before": before_environment,
        "created_at": time.time(),
    }
    artifact_hash = _digest({
        key: value for key, value in artifact.items() if key != "created_at"
    })
    artifact["artifact_hash"] = artifact_hash
    registry["resolved_gaps"][gap_id] = artifact
    if candidate is not None:
        registry["definitions"][candidate.declaration_name] = {
            "gap_id": gap_id,
            "candidate_hash": candidate.content_hash,
            "lean_source": candidate.lean_source,
            "artifact_hash": artifact_hash,
        }
    registry["registry_hash"] = registry_hash(registry)
    registry["environment_hash"] = environment_registry_hash(registry)
    _atomic_json(registry_path, registry)
    after_registry = registry["registry_hash"]
    after_environment = registry["environment_hash"]
    if before_registry == after_registry or before_environment == after_environment:
        raise RuntimeError("atomic definition commit did not change environment")
    return DefineOneResult(
        "RESOLVED", gap_id, decision.classification,
        len(decision.candidates), candidate.short_id if candidate else "",
        lean_status, lean_source, artifact_hash, before_registry,
        after_registry, before_environment, after_environment, progress,
        decision.reason,
    )


def update_stagnation(
    *,
    previous_fingerprint: str,
    previous_count: int,
    obligation_id: str,
    environment_hash: str,
    move_class: str,
    progress: ProgressVector,
    completed_semantic_iteration: bool,
    infrastructure_failure: bool = False,
) -> tuple[str, int, bool]:
    fingerprint = _digest({
        "obligation_id": obligation_id,
        "environment_hash": environment_hash,
        "move_class": move_class,
    })
    if infrastructure_failure or not completed_semantic_iteration:
        return previous_fingerprint, previous_count, False
    if progress.total:
        return fingerprint, 0, False
    count = previous_count + 1 if fingerprint == previous_fingerprint else 1
    return fingerprint, count, count >= 3


def record_semantic_iteration(
    checkpoint: Any,
    progress: ProgressVector,
    *,
    move_class: str,
    completed: bool = True,
    infrastructure_failure: bool = False,
) -> bool:
    """Apply the hard zero-delta invariant to one completed semantic iteration."""
    checkpoint.progress_vector = asdict(progress)
    fingerprint, count, stagnant = update_stagnation(
        previous_fingerprint=checkpoint.progress_fingerprint,
        previous_count=checkpoint.semantic_stagnation_count,
        obligation_id=checkpoint.target_obligation_id,
        environment_hash=checkpoint.definition_environment_hash,
        move_class=move_class,
        progress=progress,
        completed_semantic_iteration=completed,
        infrastructure_failure=infrastructure_failure,
    )
    checkpoint.progress_fingerprint = fingerprint
    checkpoint.semantic_stagnation_count = count
    if progress.total:
        checkpoint.stagnation_reason = ""
    elif stagnant:
        checkpoint.stagnation_reason = (
            "three completed zero-delta semantic iterations:" + fingerprint
        )
        if fingerprint not in checkpoint.forbidden_semantic_fingerprints:
            checkpoint.forbidden_semantic_fingerprints.append(fingerprint)
    return stagnant


def verified_progress_vector(
    *,
    definitions_added: int = 0,
    existing_definitions_resolved: int = 0,
    lemmas_proved: int = 0,
    accepted_children: int = 0,
    subgoals_closed: int = 0,
    verified_counterexamples: int = 0,
    lean_source: str = "",
) -> ProgressVector:
    """Reject registration notes and proposition ``True`` as progress."""
    source = " ".join(str(lean_source).split())
    if re.search(r"\b(?:theorem|lemma)\b[^:]*:\s*\(?True\)?\s*:=", source):
        lemmas_proved = 0
        subgoals_closed = 0
    return ProgressVector(
        max(0, definitions_added),
        max(0, existing_definitions_resolved),
        max(0, lemmas_proved),
        max(0, accepted_children),
        max(0, subgoals_closed),
        max(0, verified_counterexamples),
    )
