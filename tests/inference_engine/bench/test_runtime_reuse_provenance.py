from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    persist_validated_artifact,
    verified_reuse_provenance,
)


def _checkpoint():
    return OrchestrationCheckpoint(
        target_obligation_id="RH-C1",
        candidate_sha256="a" * 64,
        strategy_sha256="b" * 64,
        parent_statement_sha256="c" * 64,
        parent_signature_sha256="d" * 64,
        root_goal_sha256="e" * 64,
        ledger_id="ledger-current",
        ledger_version=95,
        target_context_hash="f" * 64,
        target_environment_hash="1" * 64,
        target_strategy_plan_hash="2" * 64,
    )


def _persist_role(path, checkpoint, role, dependencies):
    return persist_validated_artifact(
        path,
        checkpoint,
        role=role,
        payload={
            "schema_version": 1,
            "role": role,
            "target_obligation_id": checkpoint.target_obligation_id,
            "target_context_hash": checkpoint.target_context_hash,
            "strategy_plan_hash": checkpoint.target_strategy_plan_hash,
            "environment_hash": checkpoint.target_environment_hash,
        },
        dependencies=dependencies,
        source_run_id=f"br_{role}",
        save=False,
    )


def test_no_critic_and_stale_previous_critic_never_claim_reuse(tmp_path):
    checkpoint = _checkpoint()
    checkpoint.advisory_artifacts["previous_critic"] = {
        "text": "historical cache only",
    }
    strategy = _persist_role(
        tmp_path / "orchestration.json",
        checkpoint,
        "strategy_tournament",
        [],
    )
    _persist_role(
        tmp_path / "orchestration.json",
        checkpoint,
        "generator",
        [strategy.sha256],
    )

    reuse = verified_reuse_provenance(checkpoint)

    assert reuse["strategy_reused"] is True
    assert reuse["generator_reused"] is True
    assert reuse["critic_reused"] is False
    assert "previous_critic" not in reuse["role_reused"]


def test_actual_valid_critic_artifact_claims_reuse(tmp_path):
    checkpoint = _checkpoint()
    strategy = _persist_role(
        tmp_path / "orchestration.json",
        checkpoint,
        "strategy_tournament",
        [],
    )
    generator = _persist_role(
        tmp_path / "orchestration.json",
        checkpoint,
        "generator",
        [strategy.sha256],
    )
    _persist_role(
        tmp_path / "orchestration.json",
        checkpoint,
        "critic",
        [generator.sha256],
    )

    reuse = verified_reuse_provenance(checkpoint)

    assert reuse["critic_reused"] is True
    assert reuse["role_reused"]["critic"] is True
    assert reuse["diagnostics"] == {}


def test_target_mismatched_artifacts_fail_closed_with_diagnostic(tmp_path):
    checkpoint = _checkpoint()
    _persist_role(
        tmp_path / "orchestration.json",
        checkpoint,
        "critic",
        [],
    )
    checkpoint.target_obligation_id = "RH-C2"

    reuse = verified_reuse_provenance(checkpoint)

    assert reuse["critic_reused"] is False
    assert "target-obligation-mismatch" in reuse["diagnostics"]["critic"]
