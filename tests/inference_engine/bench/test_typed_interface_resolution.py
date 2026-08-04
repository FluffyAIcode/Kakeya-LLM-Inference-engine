import json
from pathlib import Path

import pytest

from autoresearch.prefill.architecture_v7 import run_host_definition_gate
from autoresearch.prefill.cursor_strategy import StrategyMemo, StrategyTelemetry
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    persist_validated_artifact,
)
from autoresearch.prefill.typed_interface_resolution import (
    build_target_interface_registry,
    resolve_target_interface,
)


ROOT = Path(__file__).resolve().parents[3]
TARGET = "RH-C1"
STATEMENT = (
    "Distinguish -zeta'(s)/zeta(s) from xi(s) and construct any "
    "zero/spectrum mapping without circularly reading zeta zeros from "
    "logarithmic-derivative poles."
)


def test_exact_target_builds_multiple_bound_host_candidates():
    candidates, statuses, cards = build_target_interface_registry(
        target_ref=TARGET,
        target_statement=STATEMENT,
        target_evidence={"EVIDENCE_TARGET_STATEMENT": STATEMENT},
        auditor_hash="a" * 64,
        project_root=ROOT,
    )
    assert [item.short_id for item in candidates] == [
        "TI-DISTINGUISH-FUNCTIONS",
        "TI-ZERO-MEMBERSHIP-MAP",
        "TI-SPECTRAL-REALIZATION",
        "TI-EQUIVALENT-REWRITE",
    ]
    assert len({item.content_hash for item in candidates}) == 4
    assert all(not item.feasible for item in candidates)
    assert all(item.circularity_guard_ids for item in candidates)
    assert next(
        item for item in statuses if item.source_id == "target_statement"
    ).status == "NON_PROPOSITIONAL"
    mathlib = next(
        item for item in statuses if item.source_id == "pinned_mathlib"
    )
    assert mathlib.status == "VERIFIED"
    assert {
        "riemannZeta",
        "completedRiemannZeta",
        "riemannZetaZeros",
        "mem_riemannZetaZeros",
    } <= set(mathlib.detail.split(","))
    assert cards == ()
    assert {item.source_id for item in statuses} == {
        "target_statement",
        "target_scoped_evidence",
        "definition_auditor",
        "pinned_mathlib",
        "theorem_cards",
        "local_corpus",
        "local_publication",
        "validated_history",
        "typed_synthesis",
    }


def test_interface_exhaustion_is_stable_and_rejects_unregistered_rankings():
    kwargs = {
        "target_ref": TARGET,
        "target_statement": STATEMENT,
        "target_evidence": {"EVIDENCE_TARGET_STATEMENT": STATEMENT},
        "auditor_hash": "a" * 64,
        "environment_hash": "e" * 64,
        "project_root": ROOT,
        "ranked_candidate_ids": ("TI-ZERO-MEMBERSHIP-MAP",),
        "provider_provenance": {
            "provider": "cursor-sdk",
            "model_id": "gpt-test",
            "run_id": "run-test",
        },
    }
    first = resolve_target_interface(**kwargs)
    second = resolve_target_interface(**kwargs)
    assert first == second
    assert first.status == "PARENT_STATEMENT_UNDERSPECIFIED"
    assert first.exhaustion_hash
    assert first.selected_candidate_id == ""
    with pytest.raises(ValueError, match="unregistered"):
        resolve_target_interface(
            **{
                **kwargs,
                "ranked_candidate_ids": ("TI-MODEL-INVENTED",),
            },
        )


class _InterfaceStrategy:
    def advise(self, *, evidence, registered_plan_ids):
        assert evidence["target_ref"] == TARGET
        assert set(registered_plan_ids) == {
            "TI-DISTINGUISH-FUNCTIONS",
            "TI-ZERO-MEMBERSHIP-MAP",
            "TI-SPECTRAL-REALIZATION",
            "TI-EQUIVALENT-REWRITE",
        }
        return (
            StrategyMemo(
                text="SELECTED_PLAN_ID: TI-ZERO-MEMBERSHIP-MAP",
                provider="cursor-sdk",
                model_id="gpt-test",
                agent_id="agent-test",
                run_id="run-test",
                prompt_hash="p" * 64,
                evidence_hash="e" * 64,
            ),
            StrategyTelemetry(
                provider="cursor-sdk",
                model_id="gpt-test",
                configured=True,
                status="FINISHED",
                agent_id="agent-test",
                run_id="run-test",
                prompt_hash="p" * 64,
                evidence_hash="e" * 64,
                memo_hash="m" * 64,
                attempts=1,
            ),
        )


def test_cursor_ranking_cannot_override_host_exhaustion(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_RESOLUTION.value,
        current_role="definition_resolution",
        target_obligation_id=TARGET,
        current_definition_gap_id="GAP_ELABORATED_TARGET_REQUIRED",
        selected_strategy_plan_id="DRP-test",
        selected_strategy_plan_hash="s" * 64,
        target_strategy_plan_hash="s" * 64,
        target_context_hash="c" * 64,
        target_environment_hash="e" * 64,
        parent_statement_sha256="p" * 64,
        target_evidence={"EVIDENCE_TARGET_STATEMENT": STATEMENT},
    )
    persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={"definitions": [], "missing_definitions": []},
        dependencies=[],
        source_run_id="definition",
    )

    returned, outcome = run_host_definition_gate(
        path,
        checkpoint,
        project_root=ROOT,
        interface_strategy_adapter=_InterfaceStrategy(),
    )

    assert returned is checkpoint
    assert outcome == "PARENT_STATEMENT_UNDERSPECIFIED"
    assert checkpoint.proof_state == ProofState.BLOCKED
    artifact = checkpoint.validated_artifacts["typed_interface_resolution"]
    payload = json.loads(Path(artifact.path).read_text())
    assert payload["ranked_candidate_ids"] == [
        "TI-ZERO-MEMBERSHIP-MAP",
    ]
    assert payload["provider_provenance"]["run_id"] == "run-test"
    assert payload["selected_candidate_id"] == ""
