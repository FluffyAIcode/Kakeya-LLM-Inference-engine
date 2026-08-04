from pathlib import Path

from autoresearch.prefill.rh_strategy_seed import (
    FINITE_WARNING,
    build_rh_strategy_plans,
    rh_strategy_specs,
)
from autoresearch.prefill.strategy_tournament import (
    FeasibilityReason,
    PlanExecutionStatus,
    StrategyEvent,
    evaluate_feasibility,
    run_tournament,
)


def test_three_sourced_independent_target_bound_rh_plans():
    root = Path(__file__).resolve().parents[3]
    specs = rh_strategy_specs()
    plans = build_rh_strategy_plans(root)
    assert len(specs) == len(plans) == 3
    assert len({item.content_hash for item in specs}) == 3
    assert len({item.content_hash for item in plans}) == 3
    assert len({item.target_ref for item in plans}) == 3
    assert all(item.finite_warning == FINITE_WARNING for item in specs)
    assert all(item.relation_source for item in specs)
    assert all(item.relation_obligation for item in specs)
    assert all(item.first_subgoal for item in specs)
    assert all(item.success_criterion for item in specs)
    assert all(item.falsification_criterion for item in specs)
    assert all(item.abandonment_criterion for item in specs)


def test_only_jensen_first_subgoal_is_executable_and_selected():
    root = Path(__file__).resolve().parents[3]
    plans = build_rh_strategy_plans(root)
    assert plans[0].execution_status == PlanExecutionStatus.EXECUTABLE.value
    assert all(
        item.execution_status == PlanExecutionStatus.PLANNING_ONLY.value
        for item in plans[1:]
    )
    decisions = evaluate_feasibility(
        plans,
        registered_definition_ids=plans[0].required_definition_ids,
        resolved_dependency_ids=plans[0].dependency_ids,
        verified_theorem_card_ids=plans[0].theorem_card_ids,
        allowed_assumption_ids=(),
    )
    assert decisions[0].feasible
    assert all(not item.feasible for item in decisions[1:])
    assert all(
        FeasibilityReason.PLANNING_ONLY.value in item.reason_codes
        for item in decisions[1:]
    )
    tournament = run_tournament(
        event_id="OPERATOR_STRATEGY_REDIRECT:RH-C0",
        event_type=StrategyEvent.TARGET_CHANGE,
        plans=plans,
        decisions=decisions,
    )
    assert tournament.selected_plan_id == plans[0].plan_id
