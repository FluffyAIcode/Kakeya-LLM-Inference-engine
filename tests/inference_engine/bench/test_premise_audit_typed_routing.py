import json
from pathlib import Path

import pytest

from autoresearch.prefill.orchestration_state import (
    ALLOWED_TRANSITIONS,
    OrchestrationCheckpoint,
    PremiseAuditOutcomeType,
    ProofState,
    apply_typed_premise_outcome,
    persist_validated_artifact,
    reconcile_checkpoint_ledger_version,
)
from autoresearch.prefill.architecture_v7 import run_host_definition_gate
from scripts.agent_gan_repl import (
    ProofObligation,
    ProofObligationLedger,
    apply_critic_verdicts,
)
from autoresearch.prefill.supervisor import should_resume_downstream


TARGET = "density-gap"
APPROACH_EVIDENCE = (
    "The density rho, epsilon neighborhood, and relation to f are undefined; "
    "the response therefore cannot establish the claimed growth implication."
)


def _checkpoint(**kwargs):
    return OrchestrationCheckpoint(
        state=ProofState.PREMISE_AUDIT.value,
        current_role="premise_audit",
        target_obligation_id=TARGET,
        ledger_id="ledger",
        ledger_version=91,
        **kwargs,
    )


def test_exact_disproved_approach_without_missing_lemma_is_closed():
    ledger = ProofObligationLedger(
        "ledger",
        [ProofObligation(TARGET, "Density-Singularity Gap Lemma")],
        version=91,
    )
    critic = f"""### ISSUE_VERDICT
Status: DISPROVED
Evidence: {APPROACH_EVIDENCE}
Invalidation: APPROACH
Premise refuted: density rho is undefined
"""
    applied = apply_critic_verdicts(ledger, critic, "br_fixture", {TARGET})
    assert applied == {TARGET: "DISPROVED"}
    assert ledger.obligations[0].invalidation_kind == "APPROACH_FAILED"
    assert ledger.version == 92


def test_approach_failure_persists_then_routes_new_strategy(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint()
    definition = tmp_path / "definition.json"
    definition.write_text("{}")
    checkpoint.validated_artifacts = {}
    from autoresearch.prefill.orchestration_state import persist_validated_artifact

    persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={"definitions": [], "missing_definitions": []},
        dependencies=[],
        source_run_id="definition",
    )
    persist_validated_artifact(
        path,
        checkpoint,
        role="decomposer",
        payload={"route": "old"},
        dependencies=[],
        source_run_id="old-route",
    )
    changed = apply_typed_premise_outcome(
        path,
        checkpoint,
        outcome_type=PremiseAuditOutcomeType.APPROACH_FAILED,
        decision="DISPROVED",
        owner="critic",
        confidence=1.0,
        evidence={"critic_evidence": APPROACH_EVIDENCE},
        source_run_id="br_fixture",
    )
    assert changed
    assert checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT
    assert "definition_auditor" in checkpoint.validated_artifacts
    assert "decomposer" not in checkpoint.validated_artifacts
    assert checkpoint.premise_invalidated_artifacts
    journal = [
        json.loads(line)
        for line in path.with_name(
            "proof_orchestration.journal.jsonl"
        ).read_text().splitlines()
    ]
    typed = [
        item for item in journal
        if item.get("premise_outcome_type") == "APPROACH_FAILED"
    ]
    assert [item["state"] for item in typed[-3:]] == [
        "PREMISE_AUDIT", "APPROACH_FAILED", "STRATEGY_TOURNAMENT",
    ]
    for _ in range(14):
        assert not apply_typed_premise_outcome(
            path,
            checkpoint,
            outcome_type=PremiseAuditOutcomeType.APPROACH_FAILED,
            decision="DISPROVED",
            owner="critic",
            confidence=1.0,
            evidence={"critic_evidence": APPROACH_EVIDENCE},
            source_run_id="br_fixture",
        )
    assert len([
        event for event in checkpoint.recovery_events
        if event["event_type"] == "TYPED_PREMISE_OUTCOME"
    ]) == 1
    assert not should_resume_downstream(
        checkpoint,
        candidate_sha256=checkpoint.candidate_sha256,
        force_strategy=False,
        strategy_trigger_exists=False,
    )


@pytest.mark.parametrize(
    ("outcome", "decision", "expected_intermediate"),
    [
        (
            PremiseAuditOutcomeType.PARENT_STATEMENT_UNDERSPECIFIED,
            "IDENTICAL_QUERY_EXHAUSTED",
            ProofState.PARENT_STATEMENT_UNDERSPECIFIED,
        ),
        (
            PremiseAuditOutcomeType.PREMISE_INVALIDATED,
            "FAILED_RESCUE",
            ProofState.PREMISE_INVALIDATED,
        ),
    ],
)
def test_backjump_outcomes_are_typed_and_durable(
    tmp_path,
    outcome,
    decision,
    expected_intermediate,
):
    path = tmp_path / f"{outcome.value}.json"
    checkpoint = _checkpoint()
    apply_typed_premise_outcome(
        path,
        checkpoint,
        outcome_type=outcome,
        decision=decision,
        owner="adversarial_proponent",
        confidence=0.97,
        evidence={"auditor": "confirmed", "proponent": "rescue failed"},
        source_run_id="br_review",
        backjump_target="sound-ancestor",
    )
    assert expected_intermediate in ALLOWED_TRANSITIONS[ProofState.PREMISE_AUDIT]
    assert checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT
    assert checkpoint.target_obligation_id == "sound-ancestor"
    assert checkpoint.premise_backjump_target == "sound-ancestor"


def test_suspected_waits_for_two_stage_review(tmp_path):
    checkpoint = _checkpoint()
    apply_typed_premise_outcome(
        tmp_path / "checkpoint.json",
        checkpoint,
        outcome_type=PremiseAuditOutcomeType.PREMISE_SUSPECTED,
        decision="AWAITING_AUDITOR_AND_PROPONENT",
        owner="critic",
        confidence=0.6,
        evidence={"suspicion": "typed claim"},
        source_run_id="br_suspected",
    )
    assert checkpoint.proof_state == ProofState.PREMISE_AUDIT
    assert "await-independent-auditor-and-proponent" in (
        checkpoint.last_transition_reason
    )


def test_only_new_repairable_gap_routes_to_definition_resolution(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="query and environment"):
        apply_typed_premise_outcome(
            path,
            checkpoint,
            outcome_type=PremiseAuditOutcomeType.REPAIRABLE_DEFINITION_GAP,
            decision="REPAIR",
            owner="host",
            confidence=1.0,
            evidence={"gap": "rho"},
            source_run_id="host",
        )
    apply_typed_premise_outcome(
        path,
        checkpoint,
        outcome_type=PremiseAuditOutcomeType.REPAIRABLE_DEFINITION_GAP,
        decision="REPAIR",
        owner="host",
        confidence=1.0,
        evidence={"gap": "rho"},
        source_run_id="host",
        query_hash="q" * 64,
        environment_hash="e" * 64,
    )
    assert checkpoint.proof_state == ProofState.DEFINITION_RESOLUTION
    assert ProofState.DECOMPOSER not in ALLOWED_TRANSITIONS[
        ProofState.PREMISE_AUDIT
    ]


def test_ledger_version_is_authoritative_and_idempotent():
    checkpoint = _checkpoint()
    checkpoint.ledger_version = 92
    assert reconcile_checkpoint_ledger_version(checkpoint, 91)
    assert checkpoint.ledger_version == 91
    assert checkpoint.recovery_events[-1]["checkpoint_version_before"] == 92
    assert not reconcile_checkpoint_ledger_version(checkpoint, 91)


def test_exhausted_gap_cannot_rerun_from_strategy(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.STRATEGY_TOURNAMENT.value,
        current_role="strategy_tournament",
        target_obligation_id=TARGET,
        premise_outcome_type="PARENT_STATEMENT_UNDERSPECIFIED",
        definition_query_hash="q" * 64,
        definition_environment_hash="e" * 64,
        consumed_premise_fingerprints=["q" * 64],
    )
    returned, outcome = run_host_definition_gate(
        path,
        checkpoint,
        project_root=Path(__file__).resolve().parents[3],
    )
    assert returned is checkpoint
    assert outcome == ""
    assert checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT


def test_unelaborated_target_exhausts_interfaces_and_quarantines(tmp_path):
    path = tmp_path / "checkpoint.json"
    plan_hash = "a" * 64
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_RESOLUTION.value,
        current_role="definition_resolution",
        target_obligation_id=TARGET,
        current_definition_gap_id="GAP_ELABORATED_TARGET_REQUIRED",
        selected_strategy_plan_id="DRP-a",
        selected_strategy_plan_hash=plan_hash,
        target_strategy_plan_hash=plan_hash,
        target_environment_hash="e" * 64,
        target_evidence={
            "EVIDENCE_TARGET_STATEMENT": (
                "Distinguish -zeta'(s)/zeta(s) from xi(s) and construct "
                "any zero/spectrum mapping without circularly reading zeros."
            ),
        },
    )
    persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={"definitions": [], "missing_definitions": []},
        dependencies=[],
        source_run_id="definition",
    )
    auditor_hash = checkpoint.validated_artifacts["definition_auditor"].sha256

    returned, outcome = run_host_definition_gate(
        path,
        checkpoint,
        project_root=Path(__file__).resolve().parents[3],
    )

    assert returned is checkpoint
    assert outcome == "PARENT_STATEMENT_UNDERSPECIFIED"
    assert checkpoint.proof_state == ProofState.BLOCKED
    assert checkpoint.active_gate == "TARGET_TYPED_INTERFACE_GATE"
    assert checkpoint.lean_definition_status == "PARENT_STATEMENT_UNDERSPECIFIED"
    assert checkpoint.selected_move_id == "EXHAUST_TARGET_INTERFACE_REGISTRY"
    assert checkpoint.blocked_reason.startswith("MATHEMATICAL_TERMINAL_BLOCKER:")
    artifact = checkpoint.validated_artifacts[
        "interface_exhaustion_certificate"
    ]
    payload = json.loads(Path(artifact.path).read_text())
    assert payload["auditor_hash"] == auditor_hash
    assert payload["certificate_kind"] == (
        "target_interface_exhaustion_certificate"
    )
    assert payload["typed_backjump_target"] == "ROOT_UNAVAILABLE"
    assert payload["quarantine_target"] == TARGET
    assert payload["proof_search_allowed"] is False
    assert payload["oprover_allowed"] is False
    assert len(payload["candidates"]) == 4
    assert all(not item["feasible"] for item in payload["candidates"])
    assert any(
        item["status"] == "VERIFIED"
        for item in payload["source_statuses"]
        if item["source_id"] == "pinned_mathlib"
    )
    assert checkpoint.branch_history
    quarantine = next(iter(checkpoint.branch_history.values()))
    assert quarantine["status"] == "QUARANTINED"
    assert quarantine["plan_ids"] == [TARGET]
