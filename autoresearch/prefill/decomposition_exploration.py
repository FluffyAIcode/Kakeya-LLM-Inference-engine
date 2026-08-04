"""Private, bounded candidate exploration before authoritative proof gates."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from autoresearch.prefill.strategy_tournament import (
    PlanClass,
    StrategyPlan,
)


EXPLORATION_VERSION = 1
MIN_EXPLORATION_CANDIDATES = 8
MAX_EXPLORATION_CANDIDATES = 16
DEFAULT_TOP_K = 3
EXPLORATION_CATEGORIES = (
    "DEFINITION",
    "LOCAL_LEMMA",
    "CASE_SPLIT",
    "SUFFICIENT_CONDITION",
    "EQUIVALENT_CRITERION",
    "OBSTRUCTION_OR_COUNTEREXAMPLE",
    "SPECIAL_CASE",
    "BRIDGE_THEOREM",
    "TOY_MODEL_ANALOGUE",
)
PREFILTER_REJECTION_CODES = (
    "MISSING_METADATA",
    "EXACT_DUPLICATE",
    "ALPHA_DUPLICATE",
    "SEMANTIC_DUPLICATE",
    "KNOWN_NO_GO",
    "PARENT_RESTATEMENT",
    "STRONGER_THAN_PARENT",
    "HIDDEN_ASSUMPTION",
    "MISSING_FALSIFIER",
    "DISCONNECTED_FROM_TARGET",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class DecompositionExplorationContract:
    contract_id: str
    plan_id: str
    plan_hash: str
    target_obligation_id: str
    target_context_hash: str
    proposition_hash: str
    evidence_refs: tuple[str, ...]
    no_go_refs: tuple[str, ...]
    theorem_card_ids: tuple[str, ...]
    candidate_budget: int
    required_categories: tuple[str, ...]
    ledger_mutation_allowed: bool
    proof_search_allowed: bool
    outputs_private_advisory_only: bool
    content_hash: str
    schema_version: int = EXPLORATION_VERSION


@dataclass(frozen=True)
class PrivateCandidateRef:
    candidate_id: str
    short_id: str
    category: str
    memo_sha256: str
    memo_path: str
    target_obligation_id: str
    expected_relation_id: str
    required_definition_ids: tuple[str, ...]
    reduction_sketch_id: str
    falsification_criterion_id: str
    success_criterion_id: str
    information_gain_category: str
    declared_assumption_ids: tuple[str, ...]
    target_evidence_refs: tuple[str, ...]
    alpha_fingerprint: str
    semantic_fingerprint: str
    public: bool = False
    authoritative: bool = False


@dataclass(frozen=True)
class ExplorationPrefilterResult:
    candidate_set_hash: str
    survivors: tuple[PrivateCandidateRef, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ExplorationRanking:
    ranked_candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    ranking_hash: str


@dataclass(frozen=True)
class TypedCandidateIntent:
    intent_id: str
    candidate_id: str
    target_obligation_id: str
    category: str
    registered_mapper_id: str
    required_definition_ids: tuple[str, ...]
    typed_ir: tuple[tuple[str, str], ...]
    content_hash: str


@dataclass(frozen=True)
class CandidateFormalization:
    candidate_id: str
    intent_hash: str
    lean_declaration: str
    proposition_hash: str
    elaborated: bool
    rejection_code: str


@dataclass(frozen=True)
class ReductionCertification:
    candidate_id: str
    child_proposition_hash: str
    reduction_theorem_hash: str
    reduction_proof_hash: str
    assumptions_match: bool
    strict_reduction: bool
    non_circular: bool
    critic_accepted: bool
    judge_accepted: bool
    commit_allowed: bool
    rejection_codes: tuple[str, ...]
    content_hash: str


def gate_decomposition_exploration(
    plan: StrategyPlan,
    *,
    target_obligation_id: str,
    target_context_hash: str,
    proposition_hash: str,
    evidence_refs: Iterable[str],
    no_go_refs: Iterable[str],
    theorem_card_ids: Iterable[str],
    candidate_budget: int,
) -> DecompositionExplorationContract:
    """Admit private search without granting proof search or ledger mutation."""
    if plan.plan_class != PlanClass.DECOMPOSE_TO_SUBPROBLEMS.value:
        raise ValueError("EXPLORATION_PLAN_CLASS_REQUIRED")
    if (
        not target_obligation_id
        or len(target_context_hash) != 64
        or len(proposition_hash) != 64
        or plan.target_ref != target_obligation_id
    ):
        raise ValueError("EXPLORATION_CANONICAL_TARGET_REQUIRED")
    if not MIN_EXPLORATION_CANDIDATES <= candidate_budget <= (
        MAX_EXPLORATION_CANDIDATES
    ):
        raise ValueError("EXPLORATION_CANDIDATE_BUDGET_INVALID")
    required = tuple(plan.candidate_categories or EXPLORATION_CATEGORIES)
    if len(set(required)) < MIN_EXPLORATION_CANDIDATES:
        raise ValueError("EXPLORATION_DIVERSITY_REQUIREMENT_INVALID")
    body = {
        "schema_version": EXPLORATION_VERSION,
        "plan_id": plan.plan_id,
        "plan_hash": plan.content_hash,
        "target_obligation_id": target_obligation_id,
        "target_context_hash": target_context_hash,
        "proposition_hash": proposition_hash,
        "evidence_refs": tuple(sorted(set(evidence_refs))),
        "no_go_refs": tuple(sorted(set(no_go_refs))),
        "theorem_card_ids": tuple(sorted(set(theorem_card_ids))),
        "candidate_budget": candidate_budget,
        "required_categories": required,
        "ledger_mutation_allowed": False,
        "proof_search_allowed": False,
        "outputs_private_advisory_only": True,
    }
    content_hash = _digest(body)
    return DecompositionExplorationContract(
        contract_id="DEC-" + content_hash[:20],
        content_hash=content_hash,
        **{key: value for key, value in body.items() if key != "schema_version"},
    )


def generate_private_candidate_refs(
    contract: DecompositionExplorationContract,
    *,
    memo_dir: Path,
    run_candidate: Callable[[str, str], str],
    required_definition_ids: Iterable[str] = (),
) -> tuple[PrivateCandidateRef, ...]:
    """Run independent prose-only searches; Host never parses memo text."""
    memo_dir = Path(memo_dir)
    memo_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    definitions = tuple(sorted(set(required_definition_ids)))
    refs = []
    categories = contract.required_categories[:contract.candidate_budget]
    for index, category in enumerate(categories, 1):
        short_id = f"C{index:02d}"
        prompt = (
            f"Explore one {category} subproblem for target "
            f"{contract.target_obligation_id}. Write natural mathematics or "
            "LaTeX only. Include an independently falsifiable proposal, but "
            "do not emit JSON, Lean, DSL, credentials, or authoritative claims."
        )
        text = str(run_candidate(short_id, prompt))
        encoded = text.encode()
        memo_hash = hashlib.sha256(encoded).hexdigest()
        path = memo_dir / f"{memo_hash}.private"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        normalized_tokens = re.findall(r"[a-z0-9_]+", text.lower())
        alpha_tokens = tuple(
            "_" if len(token) == 1 and token.isalpha() else token
            for token in normalized_tokens
        )
        alpha = _digest(alpha_tokens)
        semantic = _digest(tuple(sorted(
            token for token in normalized_tokens
            if token not in {
                "a", "an", "and", "for", "in", "of", "or", "the", "to",
            }
        )))
        candidate_id = "XC-" + _digest({
            "contract": contract.content_hash,
            "short_id": short_id,
            "category": category,
            "memo_sha256": memo_hash,
        })[:20]
        refs.append(PrivateCandidateRef(
            candidate_id=candidate_id,
            short_id=short_id,
            category=category,
            memo_sha256=memo_hash,
            memo_path=str(path),
            target_obligation_id=contract.target_obligation_id,
            expected_relation_id=f"REL-{category}",
            required_definition_ids=definitions,
            reduction_sketch_id=f"REDUCE-{category}",
            falsification_criterion_id=f"FALSIFY-{category}",
            success_criterion_id=f"SUCCESS-{category}",
            information_gain_category=category,
            declared_assumption_ids=(),
            target_evidence_refs=contract.evidence_refs,
            alpha_fingerprint=alpha,
            semantic_fingerprint=semantic,
        ))
    return tuple(refs)


def prefilter_private_candidates(
    candidates: Iterable[PrivateCandidateRef],
    *,
    target_obligation_id: str,
    no_go_refs: Iterable[str] = (),
    allowed_assumption_ids: Iterable[str] = (),
) -> ExplorationPrefilterResult:
    """Apply metadata-only hard constraints without Lean or memo parsing."""
    candidates = tuple(candidates)
    no_go = set(no_go_refs)
    allowed = set(allowed_assumption_ids)
    memo_hashes: set[str] = set()
    alpha_hashes: set[str] = set()
    semantic_hashes: set[str] = set()
    survivors = []
    rejected = []
    for candidate in candidates:
        reasons = []
        required = (
            candidate.candidate_id,
            candidate.short_id,
            candidate.category,
            candidate.memo_sha256,
            candidate.expected_relation_id,
            candidate.reduction_sketch_id,
            candidate.falsification_criterion_id,
            candidate.success_criterion_id,
            candidate.information_gain_category,
        )
        if not all(required):
            reasons.append("MISSING_METADATA")
        if candidate.memo_sha256 in memo_hashes:
            reasons.append("EXACT_DUPLICATE")
        if candidate.alpha_fingerprint in alpha_hashes:
            reasons.append("ALPHA_DUPLICATE")
        if candidate.semantic_fingerprint in semantic_hashes:
            reasons.append("SEMANTIC_DUPLICATE")
        if (
            candidate.memo_sha256 in no_go
            or candidate.semantic_fingerprint in no_go
        ):
            reasons.append("KNOWN_NO_GO")
        if candidate.target_obligation_id != target_obligation_id:
            reasons.append("DISCONNECTED_FROM_TARGET")
        if not candidate.falsification_criterion_id:
            reasons.append("MISSING_FALSIFIER")
        if not set(candidate.declared_assumption_ids) <= allowed:
            reasons.append("HIDDEN_ASSUMPTION")
        memo_hashes.add(candidate.memo_sha256)
        alpha_hashes.add(candidate.alpha_fingerprint)
        semantic_hashes.add(candidate.semantic_fingerprint)
        if reasons:
            rejected.append((candidate.candidate_id, tuple(dict.fromkeys(reasons))))
        else:
            survivors.append(candidate)
    fingerprint = _digest({
        "target": target_obligation_id,
        "candidate_ids": [item.candidate_id for item in candidates],
        "survivors": [item.candidate_id for item in survivors],
        "rejected": rejected,
    })
    return ExplorationPrefilterResult(
        fingerprint, tuple(survivors), tuple(rejected),
    )


def rank_private_candidates(
    survivors: Iterable[PrivateCandidateRef],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> ExplorationRanking:
    """Rank short IDs only; private memo text is unavailable here."""
    survivors = tuple(survivors)
    ordered = tuple(
        item.candidate_id for item in sorted(
            survivors,
            key=lambda item: (
                EXPLORATION_CATEGORIES.index(item.category),
                item.candidate_id,
            ),
        )
    )
    selected = ordered[:max(0, min(top_k, len(ordered)))]
    reasons = tuple("DIVERSE_INFORMATION_GAIN" for _ in ordered)
    ranking_hash = _digest({
        "ordered": ordered,
        "selected": selected,
        "reason_codes": reasons,
    })
    return ExplorationRanking(ordered, selected, reasons, ranking_hash)


def next_formalization_candidate(
    ranked_candidate_ids: Iterable[str],
    *,
    current_index: int,
) -> tuple[str, str, bool]:
    """Return current/next IDs without requesting another Strategy run."""
    queue = tuple(ranked_candidate_ids)
    if not queue or current_index < 0 or current_index >= len(queue):
        raise ValueError("FORMALIZATION_QUEUE_INDEX_INVALID")
    current = queue[current_index]
    next_id = queue[current_index + 1] if current_index + 1 < len(queue) else ""
    return current, next_id, not bool(next_id)


def build_typed_candidate_intent(
    candidate: PrivateCandidateRef,
    *,
    registered_category_mappers: dict[str, tuple[str, tuple[tuple[str, str], ...]]],
) -> TypedCandidateIntent:
    """Map Host metadata only; private memo content is never an input."""
    try:
        mapper_id, typed_ir = registered_category_mappers[candidate.category]
    except KeyError as exc:
        raise ValueError("UNMAPPABLE_TYPED_CANDIDATE_INTENT") from exc
    body = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "target_obligation_id": candidate.target_obligation_id,
        "category": candidate.category,
        "registered_mapper_id": mapper_id,
        "required_definition_ids": candidate.required_definition_ids,
        "typed_ir": typed_ir,
    }
    content_hash = _digest(body)
    return TypedCandidateIntent(
        intent_id="TCI-" + content_hash[:20],
        content_hash=content_hash,
        **{key: value for key, value in body.items() if key != "schema_version"},
    )


def formalize_candidate_statement(
    intent: TypedCandidateIntent,
    *,
    render_registered_intent: Callable[[TypedCandidateIntent], str],
    elaborate_statement: Callable[[str], bool],
) -> CandidateFormalization:
    """Render deterministic Lean and elaborate the proposition only."""
    declaration = render_registered_intent(intent)
    if (
        not declaration.startswith("theorem ")
        or ":=" in declaration
        or " by" in declaration
    ):
        raise ValueError("CANDIDATE_STATEMENT_ONLY_REQUIRED")
    proposition_hash = hashlib.sha256(declaration.encode()).hexdigest()
    elaborated = bool(elaborate_statement(declaration))
    return CandidateFormalization(
        candidate_id=intent.candidate_id,
        intent_hash=intent.content_hash,
        lean_declaration=declaration,
        proposition_hash=proposition_hash,
        elaborated=elaborated,
        rejection_code="" if elaborated else "LEAN_STATEMENT_ELABORATION_FAILED",
    )


def certify_child_reduction(
    formalization: CandidateFormalization,
    *,
    reduction_theorem_hash: str,
    reduction_proof_hash: str,
    assumptions_match: bool,
    strict_reduction: bool,
    non_circular: bool,
    critic_accepted: bool,
    judge_accepted: bool,
) -> ReductionCertification:
    """Authorize commit only after every existing proof-level gate passes."""
    reasons = []
    if not formalization.elaborated:
        reasons.append("CHILD_PROPOSITION_UNELABORATED")
    if len(reduction_theorem_hash) != 64:
        reasons.append("REDUCTION_THEOREM_UNVERIFIED")
    if len(reduction_proof_hash) != 64:
        reasons.append("REDUCTION_PROOF_UNVERIFIED")
    if not assumptions_match:
        reasons.append("PUBLIC_ASSUMPTION_MISMATCH")
    if not strict_reduction:
        reasons.append("NON_REDUCING_CHILD")
    if not non_circular:
        reasons.append("CIRCULAR_REDUCTION")
    if not critic_accepted:
        reasons.append("CRITIC_REJECTED")
    if not judge_accepted:
        reasons.append("JUDGE_REJECTED")
    body = {
        "schema_version": 1,
        "candidate_id": formalization.candidate_id,
        "child_proposition_hash": formalization.proposition_hash,
        "reduction_theorem_hash": reduction_theorem_hash,
        "reduction_proof_hash": reduction_proof_hash,
        "assumptions_match": assumptions_match,
        "strict_reduction": strict_reduction,
        "non_circular": non_circular,
        "critic_accepted": critic_accepted,
        "judge_accepted": judge_accepted,
        "commit_allowed": not reasons,
        "rejection_codes": tuple(reasons),
    }
    return ReductionCertification(
        content_hash=_digest(body),
        **{key: value for key, value in body.items() if key != "schema_version"},
    )


def exploration_exhaustion_certificate(
    contract: DecompositionExplorationContract,
    *,
    candidate_set_hash: str,
    rejected: Iterable[tuple[str, Iterable[str]]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "certificate_kind": "decomposition_exploration_exhaustion",
        "contract_id": contract.contract_id,
        "target_obligation_id": contract.target_obligation_id,
        "target_context_hash": contract.target_context_hash,
        "proposition_hash": contract.proposition_hash,
        "candidate_set_hash": candidate_set_hash,
        "rejected": [
            [candidate_id, sorted(set(reasons))]
            for candidate_id, reasons in rejected
        ],
        "ledger_mutated": False,
        "proof_search_invoked": False,
    }
    body["certificate_hash"] = _digest(body)
    return body
