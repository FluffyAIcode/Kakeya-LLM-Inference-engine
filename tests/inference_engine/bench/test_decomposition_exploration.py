from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

from autoresearch.prefill.decomposition_exploration import (
    build_typed_candidate_intent,
    certify_child_reduction,
    formalize_candidate_statement,
    EXPLORATION_CATEGORIES,
    exploration_exhaustion_certificate,
    gate_decomposition_exploration,
    generate_private_candidate_refs,
    next_formalization_candidate,
    prefilter_private_candidates,
    rank_private_candidates,
)
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    load_checkpoint,
)
from autoresearch.prefill.strategy_tournament import (
    PlanClass,
    PlanExecutionStatus,
    build_decomposition_exploration_plan,
    build_host_plans,
    evaluate_feasibility,
)
from scripts.agent_gan_repl import (
    _run_typed_ir_v2,
    _start_decomposition_exploration,
)


ROOT = hashlib.sha256(b"RiemannHypothesis").hexdigest()
CONTEXT = "c" * 64
ENVIRONMENT = "e" * 64
TARGET = "RH-C0-root"


def _plan():
    return build_decomposition_exploration_plan(
        target_ref=TARGET,
        parent_obligation_ref=TARGET,
        parent_complexity=12,
        environment_hash=ENVIRONMENT,
        registered_definition_ids=("RiemannHypothesis",),
        theorem_card_ids=("card:rh",),
        dependency_ids=("definition-audit",),
        evidence_refs=("evidence:root",),
        known_no_go_refs=("no-go:old-route",),
        candidate_budget=9,
    )


def _contract():
    plan = _plan()
    return gate_decomposition_exploration(
        plan,
        target_obligation_id=TARGET,
        target_context_hash=CONTEXT,
        proposition_hash=ROOT,
        evidence_refs=plan.evidence_refs,
        no_go_refs=plan.known_no_go_refs,
        theorem_card_ids=plan.theorem_card_ids,
        candidate_budget=plan.candidate_budget,
    )


def test_strategy_tournament_has_first_class_exploration_plan():
    plans = build_host_plans(
        target_ref=TARGET,
        parent_obligation_ref=TARGET,
        parent_complexity=12,
        environment_hash=ENVIRONMENT,
        registered_definition_ids=("RiemannHypothesis",),
        theorem_card_ids=("card:rh",),
        dependency_ids=("definition-audit",),
        evidence_refs=("evidence:root",),
        elaborated_theorem_id="KakeyaRiemannHypothesisRoot",
        proposition_hash=ROOT,
    )
    exploration = next(
        plan for plan in plans
        if plan.plan_class == PlanClass.DECOMPOSE_TO_SUBPROBLEMS.value
    )
    assert exploration.execution_status == (
        PlanExecutionStatus.EXPLORATION_ONLY.value
    )
    assert 8 <= exploration.candidate_budget <= 16
    assert len(set(exploration.candidate_categories)) >= 8
    decisions = evaluate_feasibility(
        (exploration,),
        registered_definition_ids=("RiemannHypothesis",),
        resolved_dependency_ids=("definition-audit",),
        verified_theorem_card_ids=("card:rh",),
        allowed_assumption_ids=(),
        no_go_hashes=(),
    )
    assert decisions[0].feasible


def test_exploration_contract_allows_unelaborated_private_search_only():
    contract = _contract()
    assert contract.ledger_mutation_allowed is False
    assert contract.proof_search_allowed is False
    assert contract.outputs_private_advisory_only is True
    assert contract.proposition_hash == ROOT
    with pytest.raises(ValueError, match="CANONICAL_TARGET"):
        gate_decomposition_exploration(
            _plan(),
            target_obligation_id=TARGET,
            target_context_hash="",
            proposition_hash=ROOT,
            evidence_refs=(),
            no_go_refs=(),
            theorem_card_ids=(),
            candidate_budget=9,
        )


def test_private_candidate_text_never_enters_candidate_refs(tmp_path):
    secret_texts = {}

    def run(short_id, prompt):
        text = f"private LaTeX memo {short_id}: $x_{short_id}$"
        secret_texts[short_id] = text
        assert "JSON" not in text
        return text

    refs = generate_private_candidate_refs(
        _contract(),
        memo_dir=tmp_path / "private",
        run_candidate=run,
        required_definition_ids=("RiemannHypothesis",),
    )
    assert len(refs) == 9
    assert {item.category for item in refs} == set(EXPLORATION_CATEGORIES)
    assert len({item.semantic_fingerprint for item in refs}) == 9
    for ref in refs:
        serialized = repr(ref)
        assert secret_texts[ref.short_id] not in serialized
        assert ref.public is False
        assert ref.authoritative is False
        assert (tmp_path / "private" / f"{ref.memo_sha256}.private").is_file()


def test_prefilter_is_lean_free_and_enforces_hard_metadata(tmp_path):
    refs = generate_private_candidate_refs(
        _contract(),
        memo_dir=tmp_path / "private",
        run_candidate=lambda short_id, prompt: f"memo-{short_id}",
    )
    duplicate = replace(
        refs[1],
        candidate_id="duplicate",
        memo_sha256=refs[0].memo_sha256,
        alpha_fingerprint=refs[0].alpha_fingerprint,
        semantic_fingerprint=refs[0].semantic_fingerprint,
    )
    hidden = replace(
        refs[2],
        candidate_id="hidden",
        declared_assumption_ids=("undeclared-choice",),
    )
    disconnected = replace(
        refs[3],
        candidate_id="disconnected",
        target_obligation_id="OTHER",
    )
    filtered = prefilter_private_candidates(
        (refs[0], duplicate, hidden, disconnected, *refs[4:]),
        target_obligation_id=TARGET,
        allowed_assumption_ids=(),
    )
    reasons = dict(filtered.rejected)
    assert {
        "EXACT_DUPLICATE", "ALPHA_DUPLICATE", "SEMANTIC_DUPLICATE",
    } <= set(
        reasons["duplicate"]
    )
    assert reasons["hidden"] == ("HIDDEN_ASSUMPTION",)
    assert reasons["disconnected"] == ("DISCONNECTED_FROM_TARGET",)
    assert len(filtered.survivors) == 6


def test_top_k_ranking_and_exhaustion_are_content_addressed(tmp_path):
    refs = generate_private_candidate_refs(
        _contract(),
        memo_dir=tmp_path / "private",
        run_candidate=lambda short_id, prompt: f"memo-{short_id}",
    )
    filtered = prefilter_private_candidates(
        refs,
        target_obligation_id=TARGET,
    )
    ranking = rank_private_candidates(filtered.survivors, top_k=3)
    assert len(ranking.selected_candidate_ids) == 3
    assert ranking.selected_candidate_ids == ranking.ranked_candidate_ids[:3]
    certificate = exploration_exhaustion_certificate(
        _contract(),
        candidate_set_hash=filtered.candidate_set_hash,
        rejected=((item.candidate_id, ("UNMAPPABLE",)) for item in refs),
    )
    assert certificate["ledger_mutated"] is False
    assert certificate["proof_search_invoked"] is False
    assert len(str(certificate["certificate_hash"])) == 64
    current, next_id, exhausted = next_formalization_candidate(
        ranking.ranked_candidate_ids,
        current_index=0,
    )
    assert current == ranking.ranked_candidate_ids[0]
    assert next_id == ranking.ranked_candidate_ids[1]
    assert not exhausted
    _, last_next, last_exhausted = next_formalization_candidate(
        ranking.ranked_candidate_ids,
        current_index=len(ranking.ranked_candidate_ids) - 1,
    )
    assert last_next == ""
    assert last_exhausted


def test_candidate_formalization_uses_registered_metadata_not_private_text(
    tmp_path,
):
    candidate = generate_private_candidate_refs(
        _contract(),
        memo_dir=tmp_path / "private",
        run_candidate=lambda short_id, prompt: "private claim must remain unread",
    )[1]
    with pytest.raises(ValueError, match="UNMAPPABLE"):
        build_typed_candidate_intent(
            candidate,
            registered_category_mappers={},
        )
    intent = build_typed_candidate_intent(
        candidate,
        registered_category_mappers={
            "LOCAL_LEMMA": (
                "RH_LOCAL_LEMMA_V1",
                (("target", TARGET), ("shape", "registered-local-lemma")),
            ),
        },
    )
    assert "private claim" not in repr(intent)
    formalization = formalize_candidate_statement(
        intent,
        render_registered_intent=lambda item: (
            "theorem exploration_child : True"
        ),
        elaborate_statement=lambda declaration: declaration.endswith(": True"),
    )
    assert formalization.elaborated
    assert ":=" not in formalization.lean_declaration


def test_reduction_and_judge_gates_remain_mandatory(tmp_path):
    candidate = generate_private_candidate_refs(
        _contract(),
        memo_dir=tmp_path / "private",
        run_candidate=lambda short_id, prompt: f"memo-{short_id}",
    )[1]
    intent = build_typed_candidate_intent(
        candidate,
        registered_category_mappers={
            "LOCAL_LEMMA": ("LOCAL_V1", (("shape", "local"),)),
        },
    )
    formalization = formalize_candidate_statement(
        intent,
        render_registered_intent=lambda item: "theorem child : True",
        elaborate_statement=lambda declaration: True,
    )
    rejected = certify_child_reduction(
        formalization,
        reduction_theorem_hash="",
        reduction_proof_hash="",
        assumptions_match=False,
        strict_reduction=False,
        non_circular=False,
        critic_accepted=False,
        judge_accepted=False,
    )
    assert not rejected.commit_allowed
    assert {
        "REDUCTION_THEOREM_UNVERIFIED",
        "REDUCTION_PROOF_UNVERIFIED",
        "PUBLIC_ASSUMPTION_MISMATCH",
        "NON_REDUCING_CHILD",
        "CIRCULAR_REDUCTION",
        "CRITIC_REJECTED",
        "JUDGE_REJECTED",
    } <= set(rejected.rejection_codes)
    accepted = certify_child_reduction(
        formalization,
        reduction_theorem_hash="a" * 64,
        reduction_proof_hash="b" * 64,
        assumptions_match=True,
        strict_reduction=True,
        non_circular=True,
        critic_accepted=True,
        judge_accepted=True,
    )
    assert accepted.commit_allowed


def test_clean_no_registered_move_starts_resumable_exploration(tmp_path):
    checkpoint_path = tmp_path / "orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        target_obligation_id=TARGET,
        target_context_hash=CONTEXT,
        target_environment_hash=ENVIRONMENT,
        proposition_hash=ROOT,
        root_goal_sha256=ROOT,
        parent_statement_sha256=ROOT,
        theorem_card_ids=["card:rh"],
    )
    calls = []

    def run_role(role, messages, expected_run_id):
        calls.append((role, expected_run_id, messages))
        return f"private independent memo {expected_run_id}", expected_run_id

    result = _start_decomposition_exploration(
        checkpoint,
        checkpoint_path=checkpoint_path,
        run_role=run_role,
        orchestration_id="orch-rh",
        artifact_hashes={"definition_auditor": "d" * 64},
    )
    assert result["candidate_count"] == 9
    assert result["survivor_count"] == 9
    assert len(calls) == 9
    assert checkpoint.proof_state == ProofState.CANDIDATE_FORMALIZATION
    assert checkpoint.exploration_formalization_status == "PENDING"
    assert len(checkpoint.exploration_selected_candidate_ids) == 3
    assert all(
        "private independent memo" not in repr(item)
        for item in checkpoint.exploration_candidate_refs
    )
    restored = load_checkpoint(checkpoint_path)
    assert restored.candidate_set_hash == checkpoint.candidate_set_hash
    assert restored.exploration_current_index == 0


def test_exploration_path_has_no_authoritative_model_transport():
    source = inspect.getsource(_start_decomposition_exploration)
    assert "persist_validated_artifact" not in source
    assert "persist_verified_decomposition" not in source
    assert "save_proof_ledger" not in source
    assert "json.loads" not in source
    assert "parse_math_ir" not in source
    assert "validate_lean" not in source


def test_exhaustion_replay_never_reruns_strategy_or_private_scratchpad(
    tmp_path,
):
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        target_obligation_id=TARGET,
        exploration_exhaustion_hash="x" * 64,
    )
    calls = []
    result = _run_typed_ir_v2(
        SimpleNamespace(),
        SimpleNamespace(statement="RiemannHypothesis"),
        "RiemannHypothesis",
        lambda *args: calls.append(args),
        project_root=tmp_path,
        orchestration_id="orch",
        checkpoint_path=tmp_path / "checkpoint.json",
        checkpoint=checkpoint,
        signature_validator=lambda *args: True,
        proof_validator=lambda *args: True,
        artifacts={"definition_auditor": object()},
        hashes={},
        role_run_ids={},
    )
    assert result.errors == ["DECOMPOSITION_EXPLORATION_EXHAUSTED"]
    assert result.validation["scratchpad_rerun"] is False
    assert calls == []
    assert checkpoint.proof_state == ProofState.DECOMPOSER
