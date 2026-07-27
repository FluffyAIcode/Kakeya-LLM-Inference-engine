from __future__ import annotations

import copy
import hashlib
import hmac
import json
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    persist_validated_artifact,
)
from autoresearch.prefill.resume_certificate import (
    ResumeCertificateError,
    acquire_resume_lease,
    build_resume_certificate,
    build_resume_report_provenance,
    certificate_key_path,
    certificate_pointer_path,
    checkpoint_hash,
    consume_resume_certificate,
    create_host_key,
    ledger_hash,
    persist_resume_certificate,
    resume_lease_path,
    resume_requires_certificate,
)


TARGET = "RH-C1"
CONTEXT = "c" * 64
ENVIRONMENT = "e" * 64
PLAN_ID = "SP-f629a3b1b863760d5bd5"
PLAN_HASH = "f629a3b1b863760d5bd5a1b1e1b3e914ac3df26f61e88a69f9bcbe3b27e7a6f2"
PARENT_HASH = "p" * 64
ROOT_HASH = "r" * 64
REPORT_HASH = "9" * 64
LEASE = "lease-production"
PID = 4401
RUNTIME = {
    "runtime_revision": "1" * 64,
    "model_id": "gemma-4-26B-A4B-it-mlx-4bit",
    "model_revision": "local-4bit-v1",
    "tokenizer_revision": "gemma4-v1",
    "cache_format_version": "kakeya-prefill-v3-kl-d4-q38",
    "cache_revision_hash": "2" * 64,
    "residency_state_hash": "3" * 64,
}


def _ledger(version: int = 94):
    return {
        "ledger_id": "rh-rigorous-obligations-v1",
        "version": version,
        "obligations": [{"obligation_id": TARGET, "status": "UNRESOLVED"}],
    }


def _checkpoint(path: Path) -> OrchestrationCheckpoint:
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_RESOLUTION.value,
        current_role="definition_resolution",
        target_obligation_id=TARGET,
        parent_statement_sha256=PARENT_HASH,
        root_goal_sha256=ROOT_HASH,
        target_context_hash=CONTEXT,
        target_environment_hash=ENVIRONMENT,
        target_strategy_plan_hash=PLAN_HASH,
        strategy_provider="cursor-sdk",
        strategy_provider_configured=True,
        strategy_model_id="gpt-5.6-sol",
        strategy_run_status="FINISHED",
        strategy_run_id="run-strategy",
        strategy_prompt_hash="4" * 64,
        strategy_evidence_hash="5" * 64,
        strategy_memo_hash="6" * 64,
        strategy_intent_hash="7" * 64,
        strategy_intent_run_id="run-intent",
        strategy_intent_status="FINISHED",
        strategy_plan_ids=[PLAN_ID],
        strategy_plan_hashes=[PLAN_HASH],
        feasible_strategy_plan_ids=[PLAN_ID],
        pareto_plan_ids=[PLAN_ID],
        selected_strategy_plan_id=PLAN_ID,
        selected_strategy_plan_hash=PLAN_HASH,
        ledger_id="rh-rigorous-obligations-v1",
        ledger_version=94,
        migration_event="strategy_intent_target_context_v1",
        residency_phase="GEMMA_SERVING",
        active_model="gemma",
    )
    persist_validated_artifact(
        path,
        checkpoint,
        role="strategy_tournament",
        payload={
            "schema_version": 1,
            "plan_ids": [PLAN_ID],
            "plan_hashes": [PLAN_HASH],
            "plans": [{
                "plan_id": PLAN_ID,
                "plan_kind": "DEFINITION_RESOLUTION_PLAN",
                "proof_search_allowed": False,
                "next_gate": "RESEARCH_CONTRACT_GATE",
            }],
        },
        dependencies=[],
        source_run_id="host:strategy",
        save=False,
    )
    persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={
            "schema_version": 1,
            "target_obligation_id": TARGET,
            "parent_statement_hash": PARENT_HASH,
            "audit_outcome": "COMPLETE",
        },
        dependencies=[],
        source_run_id="run-definition-auditor",
        save=False,
    )
    checkpoint.validated_artifacts["definition_auditor"] = replace(
        checkpoint.validated_artifacts["definition_auditor"],
        strategy_plan_hash="NO_FEASIBLE_PLAN",
    )
    return checkpoint


def _issue(path: Path, checkpoint: OrchestrationCheckpoint, *, now=1000.0):
    create_host_key(certificate_key_path(path))
    lease = acquire_resume_lease(
        path,
        checkpoint,
        lease_id=LEASE,
        supervisor_pid=PID,
        supervisor_generation="test-generation-1",
        ledger_sha256=ledger_hash(_ledger()),
        environment_sha256=ENVIRONMENT,
        runtime_binding=RUNTIME,
        intended_next_role="definition_resolution",
        active_conflict=False,
        issued_at=now,
        ttl_s=60,
    )
    certificate = build_resume_certificate(
        checkpoint,
        _ledger(),
        checkpoint_before_hash="8" * 64,
        checkpoint_after_hash=checkpoint_hash(checkpoint),
        report_provenance_hash=REPORT_HASH,
        runtime_binding=RUNTIME,
        intended_next_role="definition_resolution",
        owner_pid=PID,
        lease_id=LEASE,
        lease=lease,
        nonce="nonce-1",
        issued_at=now,
        ttl_s=60,
    )
    persist_resume_certificate(path, certificate)
    return certificate


def _consume(path: Path, checkpoint: OrchestrationCheckpoint, **changes):
    values = {
        "ledger": _ledger(),
        "runtime_binding": RUNTIME,
        "intended_next_role": "definition_resolution",
        "owner_pid": PID,
        "lease_id": LEASE,
        "now": 1010.0,
    }
    values.update(changes)
    return consume_resume_certificate(
        path,
        checkpoint,
        values.pop("ledger"),
        **values,
    )


def _resign_pointer(path: Path, mutate):
    pointer = certificate_pointer_path(path)
    signed = json.loads(pointer.read_text())
    signed.pop("signature")
    mutate(signed)
    body = dict(signed)
    body.pop("certificate_hash")
    signed["certificate_hash"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    key = certificate_key_path(path).read_bytes()
    signed["signature"] = hmac.new(
        key,
        json.dumps(signed, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    pointer.write_text(json.dumps(signed, sort_keys=True, separators=(",", ":")))


def test_production_definition_resolution_resume_consumes_once(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    certificate = _issue(path, checkpoint)
    assert _consume(path, checkpoint) == certificate["certificate_hash"]
    assert not certificate_pointer_path(path).exists()
    with pytest.raises(ResumeCertificateError, match="CERTIFIED_RESUME_REQUIRED"):
        _consume(path, checkpoint)


def test_missing_selected_plan_and_report_provenance_fail_closed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    checkpoint.selected_strategy_plan_id = ""
    with pytest.raises(ResumeCertificateError, match="EVIDENCE_MISSING"):
        build_resume_certificate(
            checkpoint,
            _ledger(),
            checkpoint_before_hash="8" * 64,
            checkpoint_after_hash=checkpoint_hash(checkpoint),
            report_provenance_hash=REPORT_HASH,
            runtime_binding=RUNTIME,
            intended_next_role="definition_resolution",
            owner_pid=PID,
            lease_id=LEASE,
            lease={},
            nonce="nonce",
        )
    checkpoint.selected_strategy_plan_id = PLAN_ID
    with pytest.raises(ResumeCertificateError, match="REPORT_PROVENANCE_REQUIRED"):
        build_resume_certificate(
            checkpoint,
            _ledger(),
            checkpoint_before_hash="8" * 64,
            checkpoint_after_hash=checkpoint_hash(checkpoint),
            report_provenance_hash="",
            runtime_binding=RUNTIME,
            intended_next_role="definition_resolution",
            owner_pid=PID,
            lease_id=LEASE,
            lease={},
            nonce="nonce",
        )


def test_legacy_partial_or_audit_only_evidence_never_certifies(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    legacy = copy.deepcopy(checkpoint)
    legacy.architecture_version = 8
    with pytest.raises(ResumeCertificateError, match="ARCHITECTURE_MISMATCH"):
        build_resume_certificate(
            legacy,
            _ledger(),
            checkpoint_before_hash="8" * 64,
            checkpoint_after_hash=checkpoint_hash(legacy),
            report_provenance_hash=REPORT_HASH,
            runtime_binding=RUNTIME,
            intended_next_role="definition_resolution",
            owner_pid=PID,
            lease_id=LEASE,
            lease={},
            nonce="nonce",
        )
    persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={
            "schema_version": 1,
            "audit_only": True,
            "target_obligation_id": TARGET,
            "parent_statement_hash": PARENT_HASH,
        },
        dependencies=[],
        source_run_id="legacy-audit",
        save=False,
    )
    with pytest.raises(ResumeCertificateError, match="NOT_EXECUTABLE"):
        build_resume_certificate(
            checkpoint,
            _ledger(),
            checkpoint_before_hash="8" * 64,
            checkpoint_after_hash=checkpoint_hash(checkpoint),
            report_provenance_hash=REPORT_HASH,
            runtime_binding=RUNTIME,
            intended_next_role="definition_resolution",
            owner_pid=PID,
            lease_id=LEASE,
            lease={},
            nonce="nonce",
        )
    partial_runtime = dict(RUNTIME)
    partial_runtime.pop("cache_revision_hash")
    clean = _checkpoint(tmp_path / "clean.json")
    with pytest.raises(ResumeCertificateError, match="RUNTIME_BINDING_MISSING"):
        build_resume_certificate(
            clean,
            _ledger(),
            checkpoint_before_hash="8" * 64,
            checkpoint_after_hash=checkpoint_hash(clean),
            report_provenance_hash=REPORT_HASH,
            runtime_binding=partial_runtime,
            intended_next_role="definition_resolution",
            owner_pid=PID,
            lease_id=LEASE,
            lease={},
            nonce="nonce",
        )


def test_stale_checkpoint_and_expired_certificate_fail_closed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    _issue(path, checkpoint)
    stale = copy.deepcopy(checkpoint)
    stale.updated_at += 1
    with pytest.raises(ResumeCertificateError, match="CHECKPOINT_STALE"):
        _consume(path, stale)
    with pytest.raises(ResumeCertificateError, match="EXPIRED"):
        _consume(path, checkpoint, now=1060.0)


def test_tampered_signature_fails_closed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    _issue(path, checkpoint)
    pointer = certificate_pointer_path(path)
    signed = json.loads(pointer.read_text())
    signed["nonce"] = "tampered"
    pointer.write_text(json.dumps(signed))
    with pytest.raises(ResumeCertificateError, match="SIGNATURE_INVALID"):
        _consume(path, checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("target", "RH-C2", "TARGET_MISMATCH"),
        ("model", "wrong-model", "MODEL_MISMATCH"),
    ),
)
def test_wrong_target_or_provider_model_fails_closed(tmp_path, field, value, error):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    _issue(path, checkpoint)

    def mutate(signed):
        if field == "target":
            signed["target"]["obligation_id"] = value
        else:
            signed["provider"]["model_id"] = value

    _resign_pointer(path, mutate)
    with pytest.raises(ResumeCertificateError, match=error):
        _consume(path, checkpoint)


def test_wrong_ledger_runtime_and_replay_fail_closed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    certificate = _issue(path, checkpoint)
    with pytest.raises(ResumeCertificateError, match="LEDGER_MISMATCH"):
        _consume(path, checkpoint, ledger=_ledger(version=95))
    wrong_runtime = {**RUNTIME, "model_revision": "other"}
    with pytest.raises(ResumeCertificateError, match="RUNTIME_MISMATCH"):
        _consume(path, checkpoint, runtime_binding=wrong_runtime)
    assert _consume(path, checkpoint) == certificate["certificate_hash"]
    archive = path.with_name(
        "proof_orchestration.resume_certificates"
    ) / f"{certificate['certificate_hash']}.json"
    certificate_pointer_path(path).write_bytes(archive.read_bytes())
    with pytest.raises(ResumeCertificateError, match="REPLAYED"):
        _consume(path, checkpoint)


def test_static_architecture9_gate_covers_every_resumed_role_without_fresh_deadlock():
    for state in ProofState:
        checkpoint = OrchestrationCheckpoint(
            state=state.value,
            current_role=state.value.lower(),
        )
        assert resume_requires_certificate(
            checkpoint, fresh_architecture_entry=False
        ) is (state is not ProofState.STRATEGY_TOURNAMENT)
    fresh = OrchestrationCheckpoint(
        state=ProofState.STRATEGY_TOURNAMENT.value,
        current_role="strategy_tournament",
    )
    assert not resume_requires_certificate(
        fresh, fresh_architecture_entry=True
    )


def test_definition_resolution_certificate_forbids_proof_and_oprover(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    certificate = _issue(path, checkpoint)
    assert certificate["intended_next_role"] == "definition_resolution"
    assert certificate["execution_policy"] == {
        "definition_resolution_only": True,
        "proof_search_allowed": False,
        "oprover_allowed": False,
        "next_gate": "RESEARCH_CONTRACT_GATE",
    }
    _consume(path, checkpoint)
    lease = json.loads(resume_lease_path(path).read_text())
    assert lease["consumed"] is True


def test_resume_report_provenance_is_target_bound_and_durable(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    payload, digest, artifact_path = build_resume_report_provenance(
        path, checkpoint,
    )
    assert payload["target_obligation_id"] == TARGET
    assert payload["target_context_hash"] == CONTEXT
    assert payload["selected_plan_hash"] == PLAN_HASH
    assert len(digest) == 64
    assert artifact_path.is_file()


def test_resume_lease_conflict_and_tamper_fail_closed(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = _checkpoint(path)
    certificate = _issue(path, checkpoint)
    with pytest.raises(ResumeCertificateError, match="ACTIVE_CONFLICT"):
        acquire_resume_lease(
            path,
            checkpoint,
            lease_id="other",
            supervisor_pid=PID + 1,
            supervisor_generation="other-generation",
            ledger_sha256=ledger_hash(_ledger()),
            environment_sha256=ENVIRONMENT,
            runtime_binding=RUNTIME,
            intended_next_role="definition_resolution",
            active_conflict=False,
            issued_at=1001,
            ttl_s=60,
        )
    lease_path = resume_lease_path(path)
    lease = json.loads(lease_path.read_text())
    lease["supervisor_generation"] = "tampered"
    lease_path.write_text(json.dumps(lease))
    with pytest.raises(ResumeCertificateError, match="LEASE_TAMPERED"):
        _consume(path, checkpoint)
    assert certificate_pointer_path(path).exists()
