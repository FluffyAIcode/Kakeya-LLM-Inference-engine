import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.prefill.architecture_v7 import run_architecture_v7_entry
from autoresearch.prefill.cursor_strategy import CursorStrategyAdapter
from autoresearch.prefill.research_contract import gate_research_contract
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    persist_validated_artifact,
    save_checkpoint,
)
from autoresearch.prefill.stepwise_proof import (
    ActionSelection,
    FeedbackCode,
    LeanExecutionContext,
    LeanStepResult,
    ProofGoal,
    attempt_step,
    beam_rank,
    enumerate_applicable_actions,
    load_search_state,
    lean_step_executor,
    new_search_state,
    persist_search_state,
    render_lean_ast,
)
from autoresearch.prefill.strategy_tournament import (
    BranchEvidence,
    BranchHistory,
    CriticReason,
    FeasibilityReason,
    PlanClass,
    PlanExecutionStatus,
    StrategyEvent,
    build_definition_resolution_plan,
    build_host_plans,
    evaluate_feasibility,
    review_branch,
    run_tournament,
    strategy_event_due,
)
from autoresearch.prefill.target_context import activate_target_context
from autoresearch.prefill.theorem_cards import pinned_environment_hash
from scripts.migrate_strategy_tournament_v1 import migrate
from scripts.migrate_research_contract_preproof_v2 import (
    MIGRATION_EVENT as PREPROOF_MIGRATION_EVENT,
    migrate as migrate_preproof,
)


ENV = "e" * 64


class _IntentSDK:
    def __init__(self, target, gap_ref=""):
        self.target = target
        self.gap_ref = gap_ref
        self.calls = 0

    def list_models(self, api_key):
        return [SimpleNamespace(id="gpt-5.6-sol")]

    def prompt(self, prompt, *, api_key, model_id, cwd):
        self.calls += 1
        if self.calls == 1:
            registered = re.search(
                r"registered host plan IDs: ([^.]*)\.",
                prompt,
            )
            selected = (
                registered.group(1).split(",")[0].strip()
                if registered else "DIRECT_PROOF"
            )
            return SimpleNamespace(
                status="finished",
                result=f"SELECTED_PLAN_ID: {selected}",
                agent_id="strategy",
                id="strategy-run",
            )
        gap = f"gap_ref {self.gap_ref};\n" if self.gap_ref else ""
        return SimpleNamespace(
            status="finished",
            result=(
                "plan_class DIRECT_PROOF;\n"
                f"target_ref {self.target};\n"
                f"{gap}"
                "move_family MOVE_DIRECT;\n"
                "evidence_ref EVIDENCE_TARGET_STATEMENT;\n"
                "falsification_criterion_id FALSIFY_DIRECT_PROOF;\n"
                "success_criterion_id SUCCESS_DIRECT_PROOF;\n"
                "abandonment_criterion_id ABANDON_DIRECT_PROOF;\n"
                "END;"
            ),
            agent_id="intent",
            id="intent-run",
        )


def _strategy_adapter(target, gap_ref=""):
    return CursorStrategyAdapter(
        api_key="key",
        model_id="gpt-5.6-sol",
        sdk=_IntentSDK(target, gap_ref),
        max_attempts=1,
    )


def _plans():
    return build_host_plans(
        target_ref="target:root",
        parent_obligation_ref="ROOT",
        parent_complexity=10,
        environment_hash=ENV,
        registered_definition_ids=("D1",),
        theorem_card_ids=("T1", "T2"),
        dependency_ids=("E1",),
        evidence_refs=("E1",),
        elaborated_theorem_id="hostTheorem",
        proposition_hash="p" * 64,
    )


def _decisions(plans=None):
    return evaluate_feasibility(
        plans or _plans(),
        registered_definition_ids=("D1",),
        resolved_dependency_ids=("E1",),
        verified_theorem_card_ids=("T1", "T2"),
        allowed_assumption_ids=(),
    )


def _contract():
    plan = _plans()[0]
    decision = gate_research_contract(
        plan,
        elaborated_theorem_id="hostTheorem",
        elaborated_proposition_hash="p" * 64,
        proof_obligation_id="O1",
        registered_definition_ids=("D1",),
        resolved_dependency_ids=("E1",),
        verified_theorem_card_ids=("T1", "T2"),
        allowed_assumption_ids=(),
        environment_hash=ENV,
        expected_plan_hash=plan.content_hash,
    )
    assert decision.accepted
    return decision.contract


def test_four_independent_plan_classes_and_event_only_trigger():
    plans = _plans()
    assert {item.plan_class for item in plans} == {
        item.value for item in PlanClass
        if item is not PlanClass.DEFINITION_RESOLUTION_PLAN
    }
    assert len({item.source_move_id for item in plans}) == 4
    assert all(
        item.execution_status == PlanExecutionStatus.EXECUTABLE.value
        for item in plans
    )
    assert not strategy_event_due(
        None, rounds_without_verified_progress=2, stagnation_threshold=3,
    )
    assert strategy_event_due(
        StrategyEvent.TARGET_CHANGE,
        rounds_without_verified_progress=0,
        stagnation_threshold=3,
    )


def test_hard_feasibility_duplicate_no_go_and_critic_cannot_override():
    plans = _plans()
    duplicate = replace(
        plans[0], plan_id="P5", content_hash="f" * 64,
    )
    decisions = evaluate_feasibility(
        (*plans, duplicate),
        registered_definition_ids=("D1",),
        resolved_dependency_ids=("E1",),
        verified_theorem_card_ids=("T1", "T2"),
        allowed_assumption_ids=(),
        no_go_hashes=(plans[1].content_hash,),
    )
    by_id = {item.plan_id: item for item in decisions}
    assert FeasibilityReason.NO_GO_ROUTE.value in by_id["P2"].reason_codes
    assert FeasibilityReason.DUPLICATE_ROUTE.value in by_id["P5"].reason_codes
    tournament = run_tournament(
        event_id="event:1",
        event_type=StrategyEvent.INITIAL_BRANCH,
        plans=plans,
        decisions=decisions[:4],
        critic_ranked_plan_ids=("P2", "P1"),
        critic_reason_codes=(CriticReason.LOWEST_RISK,),
    )
    assert "P2" not in tournament.critic_ranked_plan_ids
    assert tournament.selected_plan_id in tournament.pareto_plan_ids


def test_plan_contract_criteria_and_strict_reduction_rejections():
    plan = _plans()[0]
    rejected = gate_research_contract(
        replace(plan, target_complexity=plan.parent_complexity),
        elaborated_theorem_id="",
        elaborated_proposition_hash="",
        proof_obligation_id="",
        registered_definition_ids=(),
        resolved_dependency_ids=(),
        verified_theorem_card_ids=(),
        allowed_assumption_ids=(),
        environment_hash=ENV,
        expected_plan_hash=plan.content_hash,
    )
    assert not rejected.accepted
    assert "UNELABORATED_TARGET" in rejected.reason_codes
    assert "NON_REDUCING_TARGET" in rejected.reason_codes
    assert rejected.route_state == "DECOMPOSER"


def test_production_p4_fixture_keeps_eight_unresolved_definitions_and_is_planning_only():
    missing = tuple(f"DEF_{index}" for index in range(8))
    gaps = tuple(f"gap:definition:{item}" for item in missing)
    plans = build_host_plans(
        target_ref="production:P4",
        parent_obligation_ref="ROOT",
        parent_complexity=59,
        environment_hash=ENV,
        registered_definition_ids=(),
        unresolved_definition_ids=missing,
        definition_gap_ids=gaps,
        definition_auditor_hash="a" * 64,
        theorem_card_ids=("T1",),
        dependency_ids=("a" * 64,),
        evidence_refs=("a" * 64,),
    )
    p4 = plans[3]
    assert p4.plan_id == "P4"
    assert p4.required_definition_ids == missing
    assert p4.unresolved_definition_ids == missing
    assert p4.definition_gap_ids == gaps
    assert p4.definition_auditor_hash == "a" * 64
    assert p4.execution_status == PlanExecutionStatus.PLANNING_ONLY.value
    assert not p4.proposition_transformation_ref
    decisions = _decisions(plans)
    assert FeasibilityReason.PLANNING_ONLY.value in decisions[3].reason_codes
    assert FeasibilityReason.UNMET_DEFINITION.value in decisions[3].reason_codes


def test_full_preproof_transition_sequence_and_valid_contract(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "proof_orchestration.json"
    project_root = Path(__file__).resolve().parents[3]
    statement = "A replacement target with one registered definition."
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        target_obligation_id="replacement",
    )
    activate_target_context(
        checkpoint_path,
        checkpoint,
        target_obligation_id="replacement",
        statement=statement,
        environment_hash=pinned_environment_hash(project_root),
        strategy_plan_hash="STRATEGY_PENDING",
    )
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload={
            "definitions": [{"definition_id": "D1"}],
            "missing_definitions": [],
        },
        dependencies=[],
        source_run_id="audit",
    )
    transitions = []
    original = OrchestrationCheckpoint.transition

    def recording_transition(self, next_state, reason, **kwargs):
        transitions.append(next_state)
        return original(self, next_state, reason, **kwargs)

    monkeypatch.setattr(
        OrchestrationCheckpoint, "transition", recording_transition,
    )
    checkpoint.transition(ProofState.MATH_IR_TRANSLATION, "decomposed")
    checkpoint.transition(ProofState.HOST_TYPED_IR_GATE, "translated")
    checkpoint.transition(ProofState.LEAN_ELABORATION_GATE, "host-gated")
    checkpoint.transition(ProofState.STRATEGY_TOURNAMENT, "lean-elaborated")
    result = run_architecture_v7_entry(
        checkpoint_path,
        checkpoint,
        project_root=project_root,
        target_ref="replacement",
        target_statement=statement,
        parent_obligation_ref="quarantined-parent",
        parent_complexity=20,
        event_type=StrategyEvent.TARGET_CHANGE,
        event_id="TARGET_CHANGE:typed",
        elaborated_theorem_id="hostTheorem",
        proposition_hash="p" * 64,
        strategy_adapter=_strategy_adapter("replacement"),
    )
    assert transitions == [
        ProofState.MATH_IR_TRANSLATION,
        ProofState.HOST_TYPED_IR_GATE,
        ProofState.LEAN_ELABORATION_GATE,
        ProofState.STRATEGY_TOURNAMENT,
        ProofState.RESEARCH_CONTRACT_GATE,
        ProofState.PROOF_SEARCH,
    ]
    assert result.proof_state == ProofState.PROOF_SEARCH
    assert result.research_contract_id
    contract_payload = json.loads(Path(
        result.validated_artifacts["research_contract"].path
    ).read_text())
    assert contract_payload["definition_auditor_hash"]
    assert contract_payload["definition_gap_ids"] == []


def test_unelaborated_missing_definition_and_quarantine_route_to_decomposer(
    tmp_path,
):
    checkpoint_path = tmp_path / "proof_orchestration.json"
    target = "quarantined-parent"
    project_root = Path(__file__).resolve().parents[3]
    statement = "A quarantined target with eight missing definitions."
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.STRATEGY_TOURNAMENT.value,
        current_role="strategy_tournament",
        target_obligation_id=target,
        branch_history={
            "branch": {
                "status": "QUARANTINED",
                "evidence_ids": [target],
            },
        },
    )
    activate_target_context(
        checkpoint_path,
        checkpoint,
        target_obligation_id=target,
        statement=statement,
        environment_hash=pinned_environment_hash(project_root),
        strategy_plan_hash="STRATEGY_PENDING",
    )
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload={
            "definitions": [],
            "missing_definitions": [
                {"definition_id": f"DEF_{index}"} for index in range(8)
            ],
        },
        dependencies=[],
        source_run_id="audit",
    )
    result = run_architecture_v7_entry(
        checkpoint_path,
        checkpoint,
        project_root=project_root,
        target_ref=target,
        target_statement=statement,
        parent_obligation_ref="ROOT",
        parent_complexity=20,
        event_type=StrategyEvent.INITIAL_BRANCH,
        event_id="INITIAL_BRANCH:fixture",
        strategy_adapter=_strategy_adapter(
            target, "gap:definition:DEF_0",
        ),
    )
    assert result.proof_state == ProofState.DEFINITION_RESOLUTION
    assert result.research_contract_id == ""
    assert "research_contract" not in result.validated_artifacts
    assert "definition_auditor" in result.validated_artifacts
    assert (
        result.validated_artifacts["definition_auditor"].strategy_plan_hash
        == "STRATEGY_PENDING"
    )
    assert (
        result.validated_artifacts["strategy_tournament"].strategy_plan_hash
        == result.selected_strategy_plan_hash
    )
    assert result.research_contract_rejection_codes == []
    assert result.last_transition_reason == (
        "typed-definition-resolution-plan:MISSING_DEFINITION,"
        "UNELABORATED_TARGET,QUARANTINED_PARENT_REQUIRES_TYPED_REFRAME"
    )
    tournament = json.loads(Path(
        result.validated_artifacts["strategy_tournament"].path
    ).read_text())
    assert len(tournament["plans"]) == 1
    assert len(tournament["plans"][0]["unresolved_definition_ids"]) == 8
    assert tournament["plans"][0]["plan_kind"] == "DEFINITION_RESOLUTION_PLAN"
    assert tournament["plans"][0]["execution_status"] == "EXECUTABLE"
    assert tournament["plans"][0]["proof_search_allowed"] is False


def test_branch_kill_and_reframe_use_recorded_evidence_only():
    failures = BranchHistory("b", evidence=[
        BranchEvidence(
            f"E{i}", "FAIL", semantic_failure_delta=1,
            provenance_hash=str(i) * 64,
        )
        for i in range(3)
    ])
    assert review_branch(
        failures, stagnation_threshold=4, failure_threshold=3,
    ).status == "QUARANTINED"
    stagnant = BranchHistory("c", evidence=[
        BranchEvidence(f"E{i}", "NONE", provenance_hash=str(i) * 64)
        for i in range(4)
    ])
    assert review_branch(
        stagnant, stagnation_threshold=4, failure_threshold=3,
    ).status == "REFRAME_REQUIRED"


def test_typed_remediation_is_feasible_without_fabricated_theorem():
    plan = build_definition_resolution_plan(
        target_ref="RH-C1",
        parent_obligation_ref="ROOT",
        parent_complexity=10,
        environment_hash=ENV,
        registered_definition_ids=("DEF_A",),
        unresolved_definition_ids=("DEF_B",),
        definition_gap_ids=("gap:definition:DEF_B",),
        definition_auditor_hash="a" * 64,
        dependency_ids=("audit",),
        evidence_refs=("audit",),
    )
    decisions = evaluate_feasibility(
        (plan,),
        registered_definition_ids=("DEF_A",),
        resolved_dependency_ids=("audit",),
        verified_theorem_card_ids=(),
        allowed_assumption_ids=(),
    )
    assert plan.plan_id.startswith("DRP-")
    assert plan.plan_class == "DEFINITION_RESOLUTION_PLAN"
    assert plan.execution_status == "EXECUTABLE"
    assert plan.theorem_card_ids == ()
    assert decisions[0].feasible
    assert "NO_PROOF_SEARCH" in plan.restriction_ids


def test_proof_search_refuses_free_text_or_unelaborated_goal():
    plan = _plans()[0]
    decision = gate_research_contract(
        plan,
        elaborated_theorem_id="",
        elaborated_proposition_hash="",
        proof_obligation_id="O1",
        registered_definition_ids=("D1",),
        resolved_dependency_ids=("E1",),
        verified_theorem_card_ids=("T1",),
        allowed_assumption_ids=(),
        environment_hash=ENV,
        expected_plan_hash=plan.content_hash,
    )
    assert decision.contract is None
    with pytest.raises(AttributeError):
        new_search_state(decision.contract, [])  # type: ignore[arg-type]


def test_action_ids_mapping_deterministic_ast_feedback_budget_and_resume(tmp_path):
    contract = _contract()
    state = new_search_state(contract, [
        ProofGoal("G1", "p" * 64, ("H1",), "⊢ True"),
    ], proof_budget=2)
    actions = enumerate_applicable_actions(
        state,
        local_context_ids=("H1",),
        theorem_card_to_operand_id={"T1": "LEMMA1"},
    )
    exact = next(item for item in actions if item.kind == "EXACT")
    assert render_lean_ast(
        exact, operand_sources={"H1": "h"}, substitution_sources={},
    ) == "exact h"
    selection = ActionSelection("G1", exact.action_id, ("H1",), (), ())
    infrastructure = lambda _ast, _state: LeanStepResult(
        False, FeedbackCode.HOST_ERROR.value, (), "host:1",
    )
    attempt_step(
        state, selection, actions, operand_sources={"H1": "h"},
        substitution_sources={}, lean_executor=infrastructure,
    )
    assert state.semantic_failures == 0
    semantic = lambda _ast, _state: LeanStepResult(
        False, FeedbackCode.TYPE_MISMATCH.value, (), "lean:1",
    )
    attempt_step(
        state, selection, actions, operand_sources={"H1": "h"},
        substitution_sources={}, lean_executor=semantic,
    )
    assert state.semantic_failures == 1
    accepted = lambda _ast, _state: LeanStepResult(
        True, FeedbackCode.ACCEPTED.value, (),
    )
    attempt_step(
        state, selection, actions, operand_sources={"H1": "h"},
        substitution_sources={}, lean_executor=accepted, token_count=7,
    )
    assert state.status == "PROVED" and len(state.accepted_steps) == 1
    path = tmp_path / "proof.json"
    persist_search_state(path, state)
    resumed = load_search_state(path)
    assert resumed.accepted_steps == state.accepted_steps
    assert resumed.open_goals == state.open_goals
    assert resumed.rejected_feedback == []
    assert json.loads(path.read_text())["accepted_steps"][0]["rendered_ast"] == "exact h"


def test_beam_ranking_rewards_lean_progress_support_and_novelty():
    contract = _contract()
    state = new_search_state(contract, [
        ProofGoal("G1", "p" * 64, (), "g1"),
        ProofGoal("G2", "p" * 64, (), "g2"),
    ])
    actions = enumerate_applicable_actions(
        state,
        local_context_ids=(),
        theorem_card_to_operand_id={"T1": "L1"},
    )
    ranked = beam_rank(((state, item) for item in actions), beam_width=2)
    assert ranked[0][1].theorem_card_id == "T1"


@pytest.mark.skipif(shutil.which("lake") is None, reason="Lean unavailable")
def test_real_lean_multistep_acceptance_and_invalid_action_safe_failure():
    root = Path(__file__).resolve().parents[3]
    contract = _contract()
    state = new_search_state(contract, [
        ProofGoal("G1", "p" * 64, ("HP", "HQ"), "⊢ P ∧ Q"),
    ])
    executor = lean_step_executor(LeanExecutionContext(
        root,
        "theorem stepwiseAnd (P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q := by",
        ("Mathlib",),
    ))
    actions = enumerate_applicable_actions(
        state, local_context_ids=("HP", "HQ"),
        theorem_card_to_operand_id={}, constructor_ids=("AND",),
    )
    constructor = next(item for item in actions if item.kind == "CONSTRUCTOR")
    first = attempt_step(
        state,
        ActionSelection("G1", constructor.action_id, ("AND",), (), ()),
        actions,
        operand_sources={"AND": "And.intro", "HP": "hP", "HQ": "hQ"},
        substitution_sources={},
        lean_executor=executor,
    )
    assert first.accepted and state.open_goals
    actions = enumerate_applicable_actions(
        state, local_context_ids=("HP", "HQ"),
        theorem_card_to_operand_id={},
    )
    hp = next(item for item in actions if item.operand_ids == ("HP",))
    second = attempt_step(
        state,
        ActionSelection(state.open_goals[0].goal_id, hp.action_id, ("HP",), (), ()),
        actions,
        operand_sources={"HP": "hP", "HQ": "hQ"},
        substitution_sources={},
        lean_executor=executor,
    )
    assert second.accepted
    actions = enumerate_applicable_actions(
        state, local_context_ids=("HQ",),
        theorem_card_to_operand_id={},
    )
    hq = next(item for item in actions if item.operand_ids == ("HQ",))
    final = attempt_step(
        state,
        ActionSelection(state.open_goals[0].goal_id, hq.action_id, ("HQ",), (), ()),
        actions,
        operand_sources={"HQ": "hQ"},
        substitution_sources={},
        lean_executor=executor,
    )
    assert final.accepted and state.status == "PROVED"


def test_migration_branch_review_is_recorded_evidence_only_and_idempotent(
    tmp_path,
):
    checkpoint_path = tmp_path / "proof_orchestration.json"
    ledger_path = tmp_path / "ledger.json"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_checkpoint(checkpoint_path, OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        ledger_version=91,
    ))
    ledger_path.write_text(json.dumps({
        "version": 91,
        "obligations": [{
            "obligation_id": "density-1",
            "statement": "Density Singularity recorded claim",
            "status": "UNRESOLVED",
            "formal_status": "UNFORMALIZED",
            "last_evidence": "recorded only",
        }],
    }))
    first = migrate(checkpoint_path, ledger_path, snapshot)
    second = migrate(checkpoint_path, ledger_path, snapshot)
    assert first == second
    raw = json.loads(checkpoint_path.read_text())
    assert raw["state"] == "STRATEGY_TOURNAMENT"
    review = raw["branch_history"]["density-singularity"]
    assert review["assistant_verdict"] is None
    assert review["evidence_ids"] == ["density-1"]


def test_preproof_migration_preserves_tournament_and_quarantine_idempotently(
    tmp_path,
):
    checkpoint_path = tmp_path / "proof_orchestration.json"
    ledger_path = tmp_path / "ledger.json"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.SYNTHESIS.value,
        current_role="synthesis",
        ledger_version=91,
        strategy_plan_ids=["P1", "P2", "P3", "P4"],
        selected_strategy_plan_id="P4",
        strategy_tournament_hash="t" * 64,
        research_contract_rejection_codes=["UNELABORATED_TARGET"],
        branch_history={"branch": {"status": "QUARANTINED"}},
    )
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="strategy_tournament",
        payload={"plan_ids": checkpoint.strategy_plan_ids},
        dependencies=[],
        source_run_id="tournament",
    )
    ledger_path.write_text(json.dumps({"version": 91}))
    first = migrate_preproof(checkpoint_path, ledger_path, snapshot)
    second = migrate_preproof(checkpoint_path, ledger_path, snapshot)
    assert first == second
    raw = json.loads(checkpoint_path.read_text())
    assert raw["state"] == "DECOMPOSER"
    assert raw["research_contract_id"] == ""
    assert raw["research_contract_rejection_codes"] == []
    assert raw["validated_artifacts"]["strategy_tournament"]
    assert raw["branch_history"]["branch"]["status"] == "QUARANTINED"
    assert first["event_id"] == PREPROOF_MIGRATION_EVENT
