"""Host-owned creative decomposition, private reasoning, and synthesis policy."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from autoresearch.prefill.math_ir import (
    TypedIRCandidate,
    TypedIRError,
    build_decomposer_candidate_registry,
    parse_math_ir,
    validate_math_ir,
)
from autoresearch.prefill.theorem_cards import TheoremCard


MOVE_REGISTRY_VERSION = 5
MIN_CANDIDATES = 3
MIGRATION_EVENT = "creative_decomposition_synthesis_moves_v3"
SHORT_CHOICE_CODES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
RANK_REASON_CODES = (
    "DIRECT_LOCAL_CONTRADICTION",
    "SUPPORTED_BY_THEOREM_CARDS",
    "STRICTEST_REDUCTION",
    "NOVEL_VIEWPOINT",
    "LOWEST_COMPLEXITY",
    "HIGHEST_GAP_COVERAGE",
    "SHALLOWEST_DEPENDENCY_DEPTH",
)
PREFILTER_REJECTION_CODES = (
    "ANCESTOR_EQUIVALENT",
    "DISCONNECTED_TASK",
    "UNSUPPORTED_SYMBOL",
    "UNVERIFIED_PREMISE",
    "NOT_STRICTLY_SIMPLER",
    "DUPLICATE_CANDIDATE",
    "UNMET_PRECONDITION",
    "UNMET_THEOREM_HYPOTHESIS",
    "MISSING_DEPENDENCY_ARTIFACT",
)


@dataclass(frozen=True)
class ComplexityMetric:
    parent: int
    child: int

    @property
    def strictly_simpler(self) -> bool:
        return self.child < self.parent


@dataclass(frozen=True)
class HostMoveSpec:
    move_id: str
    version: int
    operand_kinds: tuple[str, ...]
    precondition_ids: tuple[str, ...]
    structural_delta: str
    assumptions_added: tuple[str, ...]
    assumptions_removed: tuple[str, ...]
    expected_reduction_path: tuple[str, ...]
    theorem_card_tags: tuple[str, ...]
    metric: ComplexityMetric
    result_status: str
    requires_parent_case_split: bool = False
    resolves_gap_ids: tuple[str, ...] = ()

    @property
    def content_hash(self) -> str:
        return _digest(asdict(self))


MOVE_REGISTRY: dict[str, HostMoveSpec] = {
    spec.move_id: spec for spec in (
        HostMoveSpec(
            "CASE_SPLIT", 3, ("parent_claim", "case_predicate"),
            ("partition_exhaustive",), "partition parent into exhaustive cases",
            (), (), ("prove_each_case", "assemble_parent"),
            ("identity_principle",), ComplexityMetric(12, 9),
            "PARENT_REDUCTION",
        ),
        HostMoveSpec(
            "RESTRICT_DOMAIN", 3, ("domain", "center", "radius"),
            ("positive_radius", "subdomain_of_parent"),
            "replace global domain by an open disk", (), (),
            ("prove_local_lemma", "transport_to_parent"),
            ("locally_uniform_limit",), ComplexityMetric(12, 7), "CHILD_LEMMA",
        ),
        HostMoveSpec(
            "REMOVE_IRRELEVANT_ASSUMPTION", 3, ("parent_claim", "assumption_id"),
            ("assumption_not_in_dependency_closure",),
            "remove one dependency-irrelevant assumption", (),
            ("irrelevant_assumption",), ("prove_reduced_claim", "weaken"),
            (), ComplexityMetric(12, 10), "CHILD_LEMMA",
        ),
        HostMoveSpec(
            "HOLOMORPHIC_EXTENSION", 3,
            ("term_family", "sum", "disk", "pole_set"),
            ("poles_outside_disk", "local_uniform_convergence", "term_holomorphicity"),
            "derive holomorphicity of the local sum", (),
            ("global_growth_conditions",),
            ("apply_locally_uniform_limit_card", "obtain_holomorphic_sum"),
            ("locally_uniform_limit", "holomorphic_sum"),
            ComplexityMetric(12, 5), "CASE_LEMMA", True,
        ),
        HostMoveSpec(
            "SINGULARITY_CONTRADICTION", 3,
            ("term_family", "sum", "pole_set", "center", "radius", "residue"),
            (
                "poles_outside_disk", "local_uniform_convergence",
                "term_holomorphicity", "sum_equals_residue_over_difference",
                "nonzero_residue", "positive_radius",
            ),
            "contrast holomorphic local sum with a nonzero simple pole", (),
            ("global_growth_conditions", "unrelated_poles"),
            (
                "derive_holomorphic_sum", "show_simple_pole_nonholomorphic",
                "derive_local_contradiction",
            ),
            (
                "locally_uniform_limit", "holomorphic_sum",
                "removable_singularity", "identity_principle",
            ),
            ComplexityMetric(12, 3), "SPECIAL_CASE_LEMMA", True,
        ),
        HostMoveSpec(
            "DENSITY_LOWER_BOUND", 3, ("sequence", "threshold"),
            ("density_defined",), "isolate density threshold", (), (),
            ("prove_density_bound",), (), ComplexityMetric(12, 8), "CHILD_LEMMA",
        ),
        HostMoveSpec(
            "LOCAL_CONVERGENCE_OBLIGATION", 3, ("sequence", "center", "radius"),
            ("positive_radius",), "isolate local convergence", (), (),
            ("prove_local_convergence",), ("locally_uniform_limit",),
            ComplexityMetric(12, 7), "CHILD_LEMMA",
        ),
        HostMoveSpec(
            "GROWTH_CONSTRAINT_OBLIGATION", 3, ("function", "degree"),
            ("growth_notation_defined",), "isolate growth constraint", (), (),
            ("prove_growth_bound",), (), ComplexityMetric(12, 8), "CHILD_LEMMA",
        ),
        HostMoveSpec(
            "LOCAL_TO_GROWTH_BRIDGE", 3, ("local_claim", "growth_claim"),
            ("both_claims_scoped",), "bridge local and global viewpoints", (), (),
            ("prove_bridge",), (), ComplexityMetric(12, 9), "BRIDGE_LEMMA",
        ),
    )
}


@dataclass(frozen=True)
class MoveCandidate:
    candidate_id: str
    choice_id: str
    move: HostMoveSpec
    typed_payload: tuple[str, ...]
    typed_ir_hash: str
    novelty_hash: str
    theorem_card_ids: tuple[str, ...]
    connected: bool = True
    uses_verified_premises_only: bool = True
    satisfied_precondition_ids: tuple[str, ...] = ()
    dependency_depth: int = 0
    gap_coverage: int = 0

    @property
    def candidate_hash(self) -> str:
        return _digest({
            "candidate_id": self.candidate_id,
            "choice_id": self.choice_id,
            "move_hash": self.move.content_hash,
            "typed_ir_hash": self.typed_ir_hash,
            "novelty_hash": self.novelty_hash,
            "theorem_card_ids": self.theorem_card_ids,
        })


@dataclass(frozen=True)
class CandidateSet:
    target_ref: str
    viewpoint: str
    candidates: tuple[MoveCandidate, ...]
    rejected: tuple[tuple[str, str], ...]
    content_hash: str
    ineligible: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def choices(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)

    def resolve(self, candidate_id: str) -> MoveCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise TypedIRError(
            "INVALID_DECOMPOSITION_CHOICE",
            f"candidate {candidate_id!r} is outside set {self.content_hash}",
        )

    @property
    def short_choice_map(self) -> Mapping[str, MoveCandidate]:
        if len(self.candidates) > len(SHORT_CHOICE_CODES):
            raise ValueError("TOO_MANY_SHORT_CHOICE_CANDIDATES")
        return {
            SHORT_CHOICE_CODES[index]: candidate
            for index, candidate in enumerate(self.candidates)
        }

    @property
    def short_choice_codes(self) -> tuple[str, ...]:
        return tuple(self.short_choice_map)

    @property
    def short_choice_map_hash(self) -> str:
        return _digest({
            "candidate_set_hash": self.content_hash,
            "mapping": {
                code: {
                    "candidate_id": candidate.candidate_id,
                    "candidate_hash": candidate.candidate_hash,
                }
                for code, candidate in self.short_choice_map.items()
            },
        })

    def resolve_short_code(
        self,
        code: str,
        *,
        candidate_set_hash: str,
        short_choice_map_hash: str,
    ) -> MoveCandidate:
        if candidate_set_hash != self.content_hash:
            raise TypedIRError(
                "STALE_CANDIDATE_SET_HASH",
                f"candidate set {candidate_set_hash!r} is not current",
            )
        if short_choice_map_hash != self.short_choice_map_hash:
            raise TypedIRError(
                "STALE_SHORT_CHOICE_MAP_HASH",
                f"short map {short_choice_map_hash!r} is not current",
            )
        try:
            return self.short_choice_map[code]
        except KeyError as exc:
            raise TypedIRError(
                "INVALID_SHORT_CHOICE_CODE",
                f"choice code {code!r} is outside set {self.content_hash}",
            ) from exc


@dataclass(frozen=True)
class CandidateRanking:
    ordered_candidate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    selected_candidate_id: str
    ranking_hash: str


@dataclass(frozen=True)
class ScratchpadRef:
    role: str
    sha256: str
    path: str
    token_count: int
    audit_only: bool = True
    authoritative: bool = False
    parse_contract: str = "NEVER_PARSE"
    public: bool = False


@dataclass(frozen=True)
class SynthesisTrigger:
    invoke: bool
    reason: str


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _cards_for_move(
    move: HostMoveSpec,
    cards: Iterable[TheoremCard],
) -> tuple[str, ...]:
    wanted = set(move.theorem_card_tags)
    return tuple(sorted(
        card.card_id for card in cards
        if wanted.intersection(card.applicability_tags)
    ))


def _eligibility_reason_codes(
    move: HostMoveSpec,
    *,
    satisfied_precondition_ids: set[str],
    available_dependency_artifact_ids: set[str],
    required_dependency_artifact_ids: set[str],
    cards: tuple[TheoremCard, ...],
    satisfied_theorem_hypothesis_ids: set[str],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if set(move.precondition_ids).difference(satisfied_precondition_ids):
        reasons.append("UNMET_PRECONDITION")
    matching_cards = tuple(
        card for card in cards
        if set(move.theorem_card_tags).intersection(card.applicability_tags)
    )
    if any(
        set(card.required_hypotheses).difference(
            satisfied_theorem_hypothesis_ids,
        )
        for card in matching_cards
    ):
        reasons.append("UNMET_THEOREM_HYPOTHESIS")
    if required_dependency_artifact_ids.difference(
        available_dependency_artifact_ids,
    ):
        reasons.append("MISSING_DEPENDENCY_ARTIFACT")
    return tuple(dict.fromkeys(reasons))


def build_candidate_set(
    *,
    target_ref: str,
    viewpoint: str,
    dependency_ids: Iterable[str] = (),
    theorem_cards: Iterable[TheoremCard] = (),
    ancestor_hashes: Iterable[str] = (),
    disconnected_move_ids: Iterable[str] = (),
    unverified_premise_move_ids: Iterable[str] = (),
    satisfied_precondition_ids: Iterable[str] = (),
    satisfied_theorem_hypothesis_ids: Iterable[str] = (),
    available_dependency_artifact_ids: Iterable[str] = (),
    required_dependency_artifact_ids: Iterable[str] = (),
    minimum: int = MIN_CANDIDATES,
) -> CandidateSet:
    """Generate and prefilter a deterministic finite Host-owned candidate set."""
    dependencies = tuple(dependency_ids)
    satisfied = set(satisfied_precondition_ids)
    raw = build_decomposer_candidate_registry(
        target_ref=target_ref,
        viewpoint=viewpoint,
        dependency_ids=dependencies,
    )
    raw_candidates = list(raw.candidates)
    ancestors = set(ancestor_hashes)
    disconnected = set(disconnected_move_ids)
    unverified = set(unverified_premise_move_ids)
    cards = tuple(theorem_cards)
    satisfied_hypotheses = set(satisfied_theorem_hypothesis_ids)
    available_dependencies = set(available_dependency_artifact_ids)
    required_dependencies = set(required_dependency_artifact_ids)
    accepted: list[MoveCandidate] = []
    rejected: list[tuple[str, str]] = []
    ineligible: list[tuple[str, tuple[str, ...]]] = []
    seen_novelty: set[str] = set()
    for item in raw_candidates:
        move = MOVE_REGISTRY.get(item.transformation_id)
        if move is None:
            rejected.append((item.choice_id, "UNSUPPORTED_SYMBOL"))
            continue
        eligibility_reasons = _eligibility_reason_codes(
            move,
            satisfied_precondition_ids=satisfied,
            available_dependency_artifact_ids=available_dependencies,
            required_dependency_artifact_ids=required_dependencies,
            cards=cards,
            satisfied_theorem_hypothesis_ids=satisfied_hypotheses,
        )
        if eligibility_reasons:
            ineligible.append((item.transformation_id, eligibility_reasons))
            continue
        novelty_hash = _digest({
            "move": move.content_hash,
            "typed_ir": item.typed_ir_hash,
            "target": target_ref,
            "viewpoint": viewpoint,
        })
        if item.typed_ir_hash in ancestors or novelty_hash in ancestors:
            rejected.append((item.choice_id, "ANCESTOR_EQUIVALENT"))
        elif item.transformation_id in disconnected:
            rejected.append((item.choice_id, "DISCONNECTED_TASK"))
        elif item.transformation_id in unverified:
            rejected.append((item.choice_id, "UNVERIFIED_PREMISE"))
        elif not move.metric.strictly_simpler:
            rejected.append((item.choice_id, "NOT_STRICTLY_SIMPLER"))
        elif novelty_hash in seen_novelty:
            rejected.append((item.choice_id, "DUPLICATE_CANDIDATE"))
        else:
            seen_novelty.add(novelty_hash)
            accepted.append(MoveCandidate(
                candidate_id=f"C{len(accepted) + 1}_{item.transformation_id}",
                choice_id=item.choice_id,
                move=move,
                typed_payload=item.typed_payload,
                typed_ir_hash=item.typed_ir_hash,
                novelty_hash=novelty_hash,
                theorem_card_ids=_cards_for_move(move, cards),
                connected=True,
                uses_verified_premises_only=True,
                satisfied_precondition_ids=tuple(sorted(
                    set(move.precondition_ids).intersection(satisfied),
                )),
                dependency_depth=len(required_dependencies),
                gap_coverage=max(1, len(move.resolves_gap_ids)),
            ))
    precondition_eligible_count = len(accepted) + len(rejected)
    if raw.candidates and precondition_eligible_count < minimum and not ineligible:
        raise ValueError(
            f"INSUFFICIENT_DISTINCT_CANDIDATES:{len(accepted)}<{minimum}"
        )
    content_hash = _digest({
        "registry_version": MOVE_REGISTRY_VERSION,
        "target_ref": target_ref,
        "viewpoint": viewpoint,
        "candidates": [item.candidate_hash for item in accepted],
        "rejected": rejected,
        "ineligible": ineligible,
    })
    return CandidateSet(
        target_ref, viewpoint, tuple(accepted), tuple(rejected), content_hash,
        tuple(ineligible),
    )


def rank_candidates(
    candidate_set: CandidateSet,
    *,
    ordered_candidate_ids: Iterable[str] | None = None,
    reason_codes: Iterable[str] = (),
) -> CandidateRanking:
    """Validate constrained ranking IDs or apply deterministic Host ranking."""
    by_id = {item.candidate_id: item for item in candidate_set.candidates}
    if ordered_candidate_ids is None:
        ordered = tuple(item.candidate_id for item in sorted(
            candidate_set.candidates,
            key=lambda item: (
                not item.move.requires_parent_case_split,
                item.move.metric.child,
                -len(item.theorem_card_ids),
                item.candidate_id,
            ),
        ))
        reasons = (
            "LOWEST_COMPLEXITY",
            "HIGHEST_GAP_COVERAGE",
            "SHALLOWEST_DEPENDENCY_DEPTH",
        )
    else:
        ordered = tuple(ordered_candidate_ids)
        reasons = tuple(reason_codes)
        if (
            len(ordered) != len(by_id)
            or set(ordered) != set(by_id)
            or len(ordered) != len(set(ordered))
        ):
            raise ValueError("INVALID_CONSTRAINED_CANDIDATE_RANKING")
        if not reasons or any(code not in RANK_REASON_CODES for code in reasons):
            raise ValueError("INVALID_CONSTRAINED_RANK_REASON")
    if not ordered:
        raise ValueError("NO_VALID_CANDIDATE")
    selected = ordered[0]
    selected_candidate = by_id[selected]
    for reason in reasons:
        if reason in {"LOWEST_COMPLEXITY", "STRICTEST_REDUCTION"} and (
            selected_candidate.move.metric.child
            != min(item.move.metric.child for item in by_id.values())
        ):
            raise ValueError(f"INVALID_RANK_REASON_METRIC:{reason}")
        if reason == "SUPPORTED_BY_THEOREM_CARDS" and (
            not selected_candidate.theorem_card_ids
            or len(selected_candidate.theorem_card_ids)
            != max(len(item.theorem_card_ids) for item in by_id.values())
        ):
            raise ValueError(f"INVALID_RANK_REASON_METRIC:{reason}")
        if reason == "HIGHEST_GAP_COVERAGE" and (
            selected_candidate.gap_coverage
            != max(item.gap_coverage for item in by_id.values())
        ):
            raise ValueError(f"INVALID_RANK_REASON_METRIC:{reason}")
        if reason == "SHALLOWEST_DEPENDENCY_DEPTH" and (
            selected_candidate.dependency_depth
            != min(item.dependency_depth for item in by_id.values())
        ):
            raise ValueError(f"INVALID_RANK_REASON_METRIC:{reason}")
        if reason == "DIRECT_LOCAL_CONTRADICTION" and (
            selected_candidate.move.result_status != "SPECIAL_CASE_LEMMA"
        ):
            raise ValueError(f"INVALID_RANK_REASON_METRIC:{reason}")
    ranking_hash = _digest({
        "candidate_set_hash": candidate_set.content_hash,
        "ordered": ordered,
        "reason_codes": reasons,
        "selected": selected,
    })
    return CandidateRanking(ordered, reasons, selected, ranking_hash)


def rank_short_choice(
    candidate_set: CandidateSet,
    *,
    choice_code: str,
    reason_code: str,
    candidate_set_hash: str,
    short_choice_map_hash: str,
) -> CandidateRanking:
    """Map one scoped short code to an immutable candidate and Host ranking."""
    selected = candidate_set.resolve_short_code(
        choice_code,
        candidate_set_hash=candidate_set_hash,
        short_choice_map_hash=short_choice_map_hash,
    )
    deterministic = rank_candidates(candidate_set)
    ordered = (
        selected.candidate_id,
        *(
            candidate_id
            for candidate_id in deterministic.ordered_candidate_ids
            if candidate_id != selected.candidate_id
        ),
    )
    return rank_candidates(
        candidate_set,
        ordered_candidate_ids=ordered,
        reason_codes=(reason_code,),
    )


def persist_private_scratchpad(
    directory: Path,
    *,
    role: str,
    transcript: str,
    token_count: int,
) -> ScratchpadRef:
    """Persist untrusted prose privately, never as a validated role artifact."""
    encoded = str(transcript).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"{digest}.txt"
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return ScratchpadRef(role, digest, str(path), int(token_count))


def run_private_scratchpad(
    inference: Callable[[str], tuple[str, int]],
    *,
    prompt: str,
    directory: Path,
    role: str = "decomposer_scratchpad",
) -> ScratchpadRef:
    transcript, token_count = inference(prompt)
    return persist_private_scratchpad(
        directory, role=role, transcript=transcript, token_count=token_count,
    )


def assert_no_scratchpad_content(
    artifact: Mapping[str, object],
    scratchpad: ScratchpadRef,
) -> None:
    encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
    private_text = Path(scratchpad.path).read_text(encoding="utf-8")

    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for key, item in value.items():
                yield str(key)
                yield from strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)

    if private_text and any(private_text in value for value in strings(artifact)):
        raise ValueError("SCRATCHPAD_TEXT_CROSSED_ARTIFACT_GATE")
    if scratchpad.path in encoded:
        raise ValueError("SCRATCHPAD_PATH_CROSSED_ARTIFACT_GATE")


def synthesis_trigger(
    *,
    novel_rejections: int,
    repeated_no_move: int,
    evidence_roles: Iterable[str],
    stagnation_threshold: int = 3,
) -> SynthesisTrigger:
    roles = set(evidence_roles)
    if novel_rejections >= stagnation_threshold:
        return SynthesisTrigger(True, "NOVEL_SEMANTIC_REJECTIONS")
    if repeated_no_move >= 2:
        return SynthesisTrigger(True, "REPEATED_NO_MOVE")
    if len(roles.intersection({
        "critic", "definition_auditor", "counterexample_worker", "theorem_cards",
    })) >= 3:
        return SynthesisTrigger(True, "MULTI_ROLE_EVIDENCE")
    return SynthesisTrigger(False, "NOT_TRIGGERED")


def synthesis_manifest(
    *,
    evidence_hashes: Mapping[str, str],
    rejected_reason_codes: Iterable[str],
    theorem_card_ids: Iterable[str],
    scratchpad_ref: ScratchpadRef,
    ranking: CandidateRanking,
    short_choice_map_hash: str,
    selected_choice_code: str,
    counterexample_verified: bool,
) -> dict[str, object]:
    """Build a public-safe synthesis artifact containing IDs and hashes only."""
    if "counterexample_worker" in evidence_hashes and not counterexample_verified:
        premise_policy = "ADVISORY_ONLY"
    else:
        premise_policy = "VERIFIED_ONLY"
    return {
        "schema_version": 1,
        "synthesis_version": 3,
        "created_at": time.time(),
        "evidence_hashes": dict(sorted(evidence_hashes.items())),
        "rejected_reason_codes": tuple(rejected_reason_codes),
        "theorem_card_ids": tuple(theorem_card_ids),
        "scratchpad_ref": scratchpad_ref.sha256,
        "scratchpad_audit_only": True,
        "ranking_hash": ranking.ranking_hash,
        "short_choice_map_hash": short_choice_map_hash,
        "selected_choice_code": selected_choice_code,
        "ranked_candidate_ids": ranking.ordered_candidate_ids,
        "selected_candidate_id": ranking.selected_candidate_id,
        "counterexample_premise_policy": premise_policy,
    }
