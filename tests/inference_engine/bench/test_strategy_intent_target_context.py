from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.prefill.architecture_v9 import run_architecture_v9_entry
from autoresearch.prefill.cursor_strategy import (
    STRATEGY_INTENT_UNMAPPABLE,
    CursorStrategyAdapter,
)
from autoresearch.prefill.definition_registry import (
    build_definition_choice_registry,
)
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    persist_validated_artifact,
)
from autoresearch.prefill.strategy_tournament import StrategyEvent
from autoresearch.prefill.target_context import (
    activate_target_context,
    context_store_path,
    mathematical_state_fingerprint,
)
from autoresearch.prefill.theorem_cards import pinned_environment_hash


class IntentSDK:
    def __init__(self, intent: str | BaseException):
        self.intent = intent
        self.calls = 0

    def list_models(self, api_key):
        return [SimpleNamespace(id="gpt-5.6-sol")]

    def prompt(self, prompt, *, api_key, model_id, cwd):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                status="finished",
                result=(
                    "Try a direct distinction using the target evidence.\n"
                    "SELECTED_PLAN_ID: DIRECT_PROOF"
                ),
                agent_id="agent-strategy",
                id="run-strategy",
            )
        if isinstance(self.intent, BaseException):
            raise self.intent
        return SimpleNamespace(
            status="finished",
            result=self.intent,
            agent_id="agent-intent",
            id="run-intent",
        )


def _intent(target: str, *, evidence: str = "EVIDENCE_TARGET_STATEMENT") -> str:
    return "\n".join((
        "plan_class DIRECT_PROOF;",
        f"target_ref {target};",
        "move_family MOVE_DIRECT;",
        f"evidence_ref {evidence};",
        "falsification_criterion_id FALSIFY_DIRECT_PROOF;",
        "success_criterion_id SUCCESS_DIRECT_PROOF;",
        "abandonment_criterion_id ABANDON_DIRECT_PROOF;",
        "END;",
    ))


def _run(tmp_path: Path, sdk: IntentSDK, *, target: str = "RH-C1"):
    root = Path(__file__).resolve().parents[3]
    statement = (
        "Distinguish -zeta'(s)/zeta(s) from xi(s) and construct any "
        "zero/spectrum mapping without circularly reading zeta zeros."
    )
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.STRATEGY_TOURNAMENT.value,
        target_obligation_id=target,
    )
    return run_architecture_v9_entry(
        tmp_path / "checkpoint.json",
        checkpoint,
        project_root=root,
        target_ref=target,
        target_statement=statement,
        target_evidence={"verified": "RH-C1 concise evidence"},
        parent_obligation_ref="ROOT",
        parent_complexity=20,
        event_type=StrategyEvent.INITIAL_BRANCH,
        event_id="INITIAL_BRANCH:test",
        elaborated_theorem_id="KakeyaLeanGate.rhC1",
        proposition_hash="a" * 64,
        strategy_adapter=CursorStrategyAdapter(
            api_key="key",
            model_id="gpt-5.6-sol",
            sdk=sdk,
            max_attempts=1,
        ),
    )


def test_memo_intent_plan_transaction_persists_target_bound_plan(tmp_path):
    checkpoint = _run(tmp_path, IntentSDK(_intent("RH-C1")))
    assert checkpoint.strategy_memo_hash
    assert checkpoint.strategy_intent_hash
    assert checkpoint.strategy_intent_run_id == "run-intent"
    assert checkpoint.strategy_plan_ids
    assert checkpoint.strategy_plan_hashes
    assert all(item.startswith("SP-") for item in checkpoint.strategy_plan_ids)
    assert checkpoint.selected_strategy_plan_id in checkpoint.strategy_plan_ids
    assert checkpoint.selected_strategy_plan_hash in checkpoint.strategy_plan_hashes
    assert checkpoint.target_strategy_plan_hash == checkpoint.selected_strategy_plan_hash
    assert checkpoint.research_contract_id
    artifact = checkpoint.validated_artifacts["strategy_tournament"]
    assert artifact.target_context_hash == checkpoint.target_context_hash
    assert checkpoint.strategy_selection_provenance["memo_hash"] == (
        checkpoint.strategy_memo_hash
    )
    assert checkpoint.target_statement.startswith("Distinguish")
    assert checkpoint.target_evidence["verified"] == "RH-C1 concise evidence"


def test_unmappable_intent_routes_to_registry_evidence_expansion(tmp_path):
    checkpoint = _run(
        tmp_path,
        IntentSDK(_intent("RH-C1", evidence="STALE_EVIDENCE")),
    )
    assert checkpoint.strategy_run_status == STRATEGY_INTENT_UNMAPPABLE
    assert checkpoint.strategy_plan_ids == []
    assert checkpoint.research_contract_id == ""
    assert checkpoint.proof_state == ProofState.DEFINITION_RESOLUTION
    assert checkpoint.current_definition_gap_id == "REGISTRY_EVIDENCE_EXPANSION"
    assert "UNREGISTERED_ID:evidence_ref" in checkpoint.stagnation_reason


def test_crash_after_private_memo_is_not_hash_only_success(tmp_path):
    path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="intent crash"):
        _run(tmp_path, IntentSDK(RuntimeError("intent crash")))
    persisted = json.loads(path.read_text())
    assert persisted["strategy_plan_ids"] == []
    assert persisted["strategy_memo_hash"] == ""
    assert persisted["strategy_intent_status"] == ""


def test_cross_target_switch_archives_c2_and_cleans_c1(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = OrchestrationCheckpoint()
    environment = "env"
    c2_statement = "Use density, genus, and pole evidence."
    activate_target_context(
        path,
        checkpoint,
        target_obligation_id="RH-C2",
        statement=c2_statement,
        environment_hash=environment,
        strategy_plan_hash="plan-c2",
        gap_ids=("D1", "D8"),
    )
    persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={
            "target_obligation_id": "RH-C2",
            "parent_statement_hash": hashlib.sha256(
                c2_statement.encode()
            ).hexdigest(),
            "definitions": ["density", "genus"],
        },
        dependencies=[],
        source_run_id="c2-audit",
    )
    checkpoint.target_gap_ids = ["D1", "D8"]
    checkpoint.premise_evidence = {
        "missing_definition_ids": ["DEF_SEQUENCE_DENSITY", "DEF_GENUS"],
    }
    checkpoint.premise_outcome_type = "REFRAME_REQUIRED"
    checkpoint.strategy_run_status = "FINISHED"
    checkpoint.strategy_run_id = "stale-c2-run"
    checkpoint.strategy_intent_hash = "stale-c2-intent"
    checkpoint.strategy_selection_provenance = {"target": "RH-C2"}
    checkpoint.proof_plan_id = "stale-c2-plan"
    activate_target_context(
        path,
        checkpoint,
        target_obligation_id="RH-C1",
        statement="Distinguish zeta logarithmic-derivative from xi.",
        environment_hash=environment,
        strategy_plan_hash="plan-c1",
    )
    assert checkpoint.validated_artifacts == {}
    assert checkpoint.target_gap_ids == []
    assert checkpoint.definition_query_hash == ""
    assert checkpoint.premise_evidence == {}
    assert checkpoint.premise_outcome_type == ""
    assert checkpoint.theorem_card_ids == []
    assert checkpoint.strategy_run_status == "CONFIGURATION_REQUIRED"
    assert checkpoint.strategy_run_id == ""
    assert checkpoint.strategy_intent_hash == ""
    assert checkpoint.strategy_selection_provenance == {}
    assert checkpoint.proof_plan_id == ""
    store = json.loads(context_store_path(path).read_text())
    active = store["contexts"][store["active_context_hash"]]
    assert active["binding"]["target_obligation_id"] == "RH-C1"
    assert "density" not in json.dumps(active).lower()
    assert any(
        item["binding"]["target_obligation_id"] == "RH-C2"
        and item["audit_only"]
        for item in store["contexts"].values()
    )


def test_definition_registry_is_derived_from_exact_target():
    c2 = build_definition_choice_registry(
        "claim:c2", "A density and fixed genus pole lemma."
    )
    assert "DEF_SEQUENCE_DENSITY" in c2.definitions
    assert "DEF_GENUS" in c2.definitions
    c1 = build_definition_choice_registry(
        "claim:c1",
        "Distinguish the zeta logarithmic-derivative from completed xi and "
        "construct a non-circular zero/spectrum mapping.",
    )
    assert "DEF_ZETA_LOG_DERIVATIVE" in c1.definitions
    assert "DEF_COMPLETED_XI" in c1.definitions
    assert not any(
        word in json.dumps({
            key: value.label for key, value in c1.definitions.items()
        }).lower()
        for word in ("density", "genus")
    )


@pytest.mark.parametrize(
    "role",
    (
        "definition_auditor", "counterexample_worker", "decomposer",
        "math_ir_translator", "proof_search", "adversarial_proponent", "judge",
        "synthesis", "strategy_tournament", "research_contract",
    ),
)
def test_every_role_rejects_target_context_mismatch(tmp_path, role):
    path = tmp_path / "checkpoint.json"
    checkpoint = OrchestrationCheckpoint()
    statement = "Current target statement."
    activate_target_context(
        path,
        checkpoint,
        target_obligation_id="RH-C1",
        statement=statement,
        environment_hash="env",
        strategy_plan_hash="plan",
    )
    with pytest.raises(ValueError, match="TARGET_CONTEXT_MISMATCH"):
        persist_validated_artifact(
            path,
            checkpoint,
            role=role,
            payload={
                "target_obligation_id": "RH-C2",
                "parent_statement_hash": hashlib.sha256(
                    statement.encode()
                ).hexdigest(),
            },
            dependencies=[],
            source_run_id="stale",
        )


def test_mathematical_fingerprint_ignores_run_and_viewpoint():
    checkpoint = OrchestrationCheckpoint(
        target_obligation_id="RH-C1",
        target_context_hash="ctx",
        target_environment_hash="env",
        source_run_ids=["run-1"],
        viewpoint="first",
        candidate_count=0,
    )
    first = mathematical_state_fingerprint(
        checkpoint,
        failure_code="EMPTY_CANDIDATE_SET",
    )
    checkpoint.source_run_ids = ["run-2"]
    checkpoint.viewpoint = "changed"
    second = mathematical_state_fingerprint(
        checkpoint,
        failure_code="EMPTY_CANDIDATE_SET",
    )
    assert first == second
