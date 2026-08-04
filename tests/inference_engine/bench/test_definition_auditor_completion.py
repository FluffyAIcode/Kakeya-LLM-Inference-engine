import copy
import hashlib
import json
from pathlib import Path

import pytest

from autoresearch.prefill.orchestration_state import (
    ALLOWED_TRANSITIONS,
    DefinitionAuditOutcomeType,
    OrchestrationCheckpoint,
    ProofState,
    load_checkpoint,
    persist_validated_artifact,
    route_definition_audit_outcome,
)
from autoresearch.prefill.prepare import ResumeValidationError, evaluate
from scripts.agent_gan_repl import (
    ProofObligation,
    _run_typed_definition_auditor,
    build_architecture7_report_provenance,
)


RUN_ID = "br_172ab0943c602e01"
TARGET = "RH-C2-production-regression"
PARENT_HASH = "fedc0c013ff64b830adc12be3d396e96d0cd86aa977e46681baffd3537ad67b6"
ROOT_HASH = "ba31be416856d1f975f8f885f54d9babde1aada7ee98d42341c3b58cad91a061"
CANDIDATE_HASH = "e" * 64
TARGET_STATEMENT = (
    "**The Formalization of Pole-Mimicry:** Define a property "
    "$P(s, \\{z_n\\})$ such that a sequence of zeros $\\{z_n\\}$ satisfies "
    "$P$ if the resulting Weierstrass product approximates a pole at $s_0$ "
    "within a specified error $\\delta$. Then, prove the existence of "
    "$\\rho_c(\\epsilon)$ such that for $\\rho > \\rho_c$, $P$ is impossible "
    "for $z_n \\in S_\\epsilon$."
)


class Candidate:
    CANDIDATE_ID = TARGET
    TARGET_OBLIGATION_ID = TARGET
    CANDIDATE_SHA256 = CANDIDATE_HASH
    PREFILL_COMPUTE_CHUNK_TOKENS = 256
    SNAPSHOT_MODE = "final_only"
    MAX_SEGMENT_SECONDS = 300
    REQUIRE_FULL_CONTEXT = True
    ALLOW_FALLBACK = False


def _checkpoint(state=ProofState.DECOMPOSER):
    return OrchestrationCheckpoint(
        state=state.value,
        current_role=state.value.lower(),
        target_obligation_id=TARGET,
        candidate_sha256=CANDIDATE_HASH,
        strategy_sha256=CANDIDATE_HASH,
        parent_statement_sha256=PARENT_HASH,
        root_goal_sha256=ROOT_HASH,
        ledger_id="rh-rigorous-obligations-v1",
        ledger_version=93,
        target_context_hash="3" * 64,
        target_environment_hash="2" * 64,
        target_strategy_plan_hash="4" * 64,
        orchestration_id=f"{RUN_ID}:decomposition:dbb8725d6ba5",
        resume_origin=ProofState.DECOMPOSER.value,
    )


def test_exact_reframe_from_decomposer_is_persisted_and_legally_routed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint()
    persist_validated_artifact(
        path,
        checkpoint,
        role="strategy_tournament",
        payload={"schema_version": 1, "plan": "P1"},
        dependencies=[],
        source_run_id="host:INITIAL_BRANCH",
    )
    before = copy.deepcopy(checkpoint)
    calls = []

    def run_role(role, _messages, expected_run_id):
        calls.append(role)
        return (
            f"target_ref claim:{PARENT_HASH};\n"
            "audit_outcome REFRAME_REQUIRED;\nEND;",
            expected_run_id,
        )

    artifact, digest, _transcript, source_run_id = _run_typed_definition_auditor(
        ProofObligation(TARGET, TARGET_STATEMENT),
        "root goal",
        run_role,
        orchestration_id=checkpoint.orchestration_id,
        checkpoint_path=path,
        checkpoint=checkpoint,
    )
    assert calls == ["definition_auditor"]
    assert artifact.audit_outcome == "REFRAME_REQUIRED"
    assert checkpoint.proof_state == ProofState.SYNTHESIS
    assert checkpoint.current_role == "synthesis"
    assert checkpoint.mathematical_retries == 0
    assert checkpoint.adapter_status == ""
    assert checkpoint.validated_artifacts["definition_auditor"].sha256 == digest
    assert hashlib.sha256(TARGET_STATEMENT.encode()).hexdigest() == PARENT_HASH
    assert json.loads(Path(
        checkpoint.validated_artifacts["definition_auditor"].path,
    ).read_text(encoding="utf-8"))["audit_outcome"] == "REFRAME_REQUIRED"
    assert load_checkpoint(path).proof_state == ProofState.SYNTHESIS

    event_count = len([
        item for item in checkpoint.recovery_events
        if item["event_type"] == "TYPED_DEFINITION_AUDIT_OUTCOME"
    ])
    assert not route_definition_audit_outcome(
        checkpoint,
        outcome=DefinitionAuditOutcomeType.REFRAME_REQUIRED,
        artifact_hash=digest,
        source_run_id=source_run_id,
        missing_definition_ids=(
            item["definition_id"] for item in artifact.missing_definitions
        ),
    )
    assert len([
        item for item in checkpoint.recovery_events
        if item["event_type"] == "TYPED_DEFINITION_AUDIT_OUTCOME"
    ]) == event_count

    provenance = build_architecture7_report_provenance(
        before,
        checkpoint,
        [{"name": "agent_definition_auditor", "ok": True, "complete": True}],
        ledger_sha256="1" * 64,
        environment_sha256="2" * 64,
    )
    report = {
        "id": RUN_ID,
        "status": "completed",
        "stages": [{"name": "agent_definition_auditor", "ok": True, "complete": True}],
        "provenance": provenance,
    }
    result = evaluate(report, Candidate)
    assert result["evaluation_provenance"]["typed_partial"] is True
    assert result["evaluation_provenance"]["checkpoint_after_state"] == "SYNTHESIS"
    assert provenance["produced_artifacts"]["definition_auditor"]["sha256"] == digest
    assert provenance["reused_stages"] == ["strategy_tournament"]


@pytest.mark.parametrize(
    ("outcome", "missing", "expected"),
    [
        ("COMPLETE", (), ProofState.DECOMPOSER),
        ("MISSING_DEFINITION", ("DEF_DENSITY",), ProofState.DEFINITION_RESOLUTION),
        ("REFRAME_REQUIRED", (), ProofState.SYNTHESIS),
        ("PARENT_UNDERSPECIFIED", (), ProofState.SYNTHESIS),
    ],
)
def test_static_definition_audit_outcome_coverage(outcome, missing, expected):
    checkpoint = _checkpoint(ProofState.DEFINITION_AUDITOR)
    assert expected in ALLOWED_TRANSITIONS[ProofState.DEFINITION_AUDITOR]
    assert route_definition_audit_outcome(
        checkpoint,
        outcome=outcome,
        artifact_hash="a" * 64,
        source_run_id="br_audit",
        missing_definition_ids=missing,
    )
    assert checkpoint.proof_state == expected


def test_counterexample_requires_explicit_typed_objective():
    checkpoint = _checkpoint()
    route_definition_audit_outcome(
        checkpoint,
        outcome="COMPLETE",
        artifact_hash="a" * 64,
        source_run_id="br_no_objective",
    )
    assert checkpoint.proof_state == ProofState.DECOMPOSER

    checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="objective_type and evidence_request"):
        route_definition_audit_outcome(
            checkpoint,
            outcome="COMPLETE",
            artifact_hash="b" * 64,
            source_run_id="br_bad_objective",
            counterexample_objective={"objective_type": "FALSIFY"},
        )
    route_definition_audit_outcome(
        checkpoint,
        outcome="COMPLETE",
        artifact_hash="c" * 64,
        source_run_id="br_explicit_objective",
        counterexample_objective={
            "objective_type": "FALSIFY_REGISTERED_CLAIM",
            "evidence_request": "construct a verified finite witness",
        },
    )
    assert checkpoint.proof_state == ProofState.COUNTEREXAMPLE_WORKER


def test_typed_partial_provenance_missing_or_stale_fails_closed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    before = _checkpoint()
    persist_validated_artifact(
        path,
        before,
        role="strategy_tournament",
        payload={"schema_version": 1},
        dependencies=[],
        source_run_id="host:strategy",
    )
    after = copy.deepcopy(before)
    provenance = build_architecture7_report_provenance(
        before,
        after,
        [],
        ledger_sha256="1" * 64,
        environment_sha256="2" * 64,
    )
    report = {"stages": [], "provenance": provenance}
    del provenance["bindings"]["environment_sha256"]
    with pytest.raises(ResumeValidationError, match="bindings are incomplete"):
        evaluate(report, Candidate)

    provenance = build_architecture7_report_provenance(
        before,
        after,
        [],
        ledger_sha256="1" * 64,
        environment_sha256="2" * 64,
    )
    provenance["reused_artifacts"]["strategy_tournament"]["sha256"] = "0" * 64
    with pytest.raises(ResumeValidationError, match="stale artifact hash"):
        evaluate({"stages": [], "provenance": provenance}, Candidate)
