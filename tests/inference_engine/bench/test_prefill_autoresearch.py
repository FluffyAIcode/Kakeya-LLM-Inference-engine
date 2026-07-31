import hashlib
import json

import pytest

from autoresearch.prefill.prepare import (
    ReportValidationError,
    ResumeValidationError,
    critic_artifact_payload,
    evaluate,
)


class Candidate:
    CANDIDATE_ID = "test"
    TARGET_OBLIGATION_ID = "RH-C1"
    PREFILL_COMPUTE_CHUNK_TOKENS = 256
    SNAPSHOT_MODE = "final_only"
    MAX_SEGMENT_SECONDS = 300
    REQUIRE_FULL_CONTEXT = True
    ALLOW_FALLBACK = False


def _report(**overrides):
    stage = {
        "name": "agent_critic",
        "ok": True,
        "complete": True,
        "prefix_tokens": 1000,
        "warmup_wall_s": 1000,
        "review_scope": "full",
        "generator_full_tokens": 900,
        "critic_context_tokens": 900,
        "critic_omitted_tokens": 0,
        "critic_protocol": "goal_anchored_recursive_gan_v3",
        "proof_obligations_total": 5,
        "proof_obligations_covered": 5,
        "proof_obligations_unresolved": 5,
        "delta": {"fallbacks": 0, "remote_job_failures": 0},
    }
    stage.update(overrides)
    return {"stages": [stage]}


def test_autoresearch_accepts_faster_full_context_candidate():
    result = evaluate(_report(), Candidate)
    assert result["accepted"]
    assert result["metric_cold_critic_prefill_s"] == 1000
    assert result["estimated_max_segment_s"] == 256
    assert all(result["constraints"].values())


def test_autoresearch_rejects_slow_segment_or_semantic_regression():
    slow = type("Slow", (), {
        "PREFILL_COMPUTE_CHUNK_TOKENS": 512,
        "SNAPSHOT_MODE": "final_only",
        "MAX_SEGMENT_SECONDS": 300,
        "CANDIDATE_ID": "slow",
        "TARGET_OBLIGATION_ID": "RH-C1",
        "REQUIRE_FULL_CONTEXT": True,
        "ALLOW_FALLBACK": False,
    })
    assert not evaluate(_report(), slow)["accepted"]
    assert not evaluate(
        _report(critic_omitted_tokens=1),
        Candidate,
    )["accepted"]


def _resumed_report(tmp_path, stages):
    candidate_sha = "a" * 64
    generator_sha = "b" * 64
    parent_sha = "c" * 64
    root_sha = "d" * 64
    critic_stage = _report()["stages"][0]
    payload = critic_artifact_payload(
        critic_stage,
        source_run_id="br_source",
        target_obligation_id=Candidate.TARGET_OBLIGATION_ID,
        candidate_sha256=candidate_sha,
        strategy_sha256=candidate_sha,
        parent_statement_sha256=parent_sha,
        parent_signature_sha256="",
        root_goal_sha256=root_sha,
        ledger_id="ledger",
        ledger_version=87,
        generator_output_sha256=generator_sha,
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact_path = tmp_path / "critic.json"
    artifact_path.write_bytes(encoded)
    critic_ref = {
        "role": "critic",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "schema_version": 1,
        "dependencies": [
            candidate_sha, parent_sha, "", root_sha, generator_sha,
        ],
        "path": str(artifact_path),
        "source_run_id": "br_source",
        "validated_at": 1,
        "reusable": True,
        "validation_status": "validated",
    }
    reused_refs = {}
    for role, sha in (
        ("strategy", candidate_sha),
        ("generator", generator_sha),
    ):
        role_path = tmp_path / f"{role}.json"
        role_path.write_text(json.dumps({"role": role, "sha256": sha}))
        role_encoded = role_path.read_bytes()
        reused_refs[role] = {
            "role": role,
            "sha256": hashlib.sha256(role_encoded).hexdigest(),
            "schema_version": 1,
            "dependencies": [],
            "path": str(role_path),
            "source_run_id": "br_source",
            "validated_at": 1,
            "reusable": True,
            "validation_status": "validated",
        }
    provenance = {
        "schema_version": 2,
        "mode": "resumed",
        "resumed_from_state": "DECOMPOSER",
        "resumed_from_role": "decomposer",
        "strategy_reused": True,
        "generator_reused": True,
        "critic_reused": True,
        "bindings": payload["bindings"],
        "reused_artifacts": {
            **reused_refs,
            "critic": critic_ref,
        },
        "newly_executed_stages": [
            stage["name"] for stage in stages
        ],
    }
    candidate = type(
        "ResumedCandidate",
        (Candidate,),
        {"CANDIDATE_SHA256": candidate_sha},
    )
    return {
        "id": "br_resumed",
        "status": "completed",
        "stages": stages,
        "provenance": provenance,
    }, candidate


@pytest.mark.parametrize("stages", [
    [],
    [
        {"name": "agent_counterexample_worker"},
        {"name": "agent_decomposer"},
        {"name": "agent_decomposer"},
    ],
])
def test_archived_resumed_report_shapes_reuse_bound_critic(tmp_path, stages):
    report, candidate = _resumed_report(tmp_path, stages)
    result = evaluate(report, candidate)
    assert result["accepted"]
    assert result["evaluation_provenance"]["critic_reused"] is True
    assert result["evaluation_provenance"]["critic_source_run_id"] == "br_source"
    assert result["evaluation_provenance"]["newly_executed_stages"] == [
        stage["name"] for stage in stages
    ]


def test_fresh_report_missing_critic_fails():
    with pytest.raises(ReportValidationError, match="physical Critic"):
        evaluate({"stages": []}, Candidate)


def test_resumed_report_invalid_provenance_fails_closed(tmp_path):
    report, candidate = _resumed_report(tmp_path, [])
    report["provenance"]["bindings"]["candidate_sha256"] = "stale"
    with pytest.raises(ResumeValidationError, match="candidate hash"):
        evaluate(report, candidate)
    assert not evaluate(
        _report(delta={"fallbacks": 1, "remote_job_failures": 1}),
        Candidate,
    )["accepted"]
