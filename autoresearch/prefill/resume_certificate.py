"""Host-signed, single-use certificates for Architecture-9 role resumes."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from autoresearch.prefill.orchestration_state import OrchestrationCheckpoint


CERTIFICATE_SCHEMA_VERSION = 1
CERTIFICATE_KIND = "architecture9_resume"
LEASE_KIND = "architecture9_resume_lease"
REQUIRED_RUNTIME_FIELDS = frozenset({
    "runtime_revision",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "cache_format_version",
    "cache_revision_hash",
    "residency_state_hash",
})


class ResumeCertificateError(RuntimeError):
    """The host cannot prove that a resumed Architecture-9 role is safe."""


def resume_requires_certificate(
    checkpoint: OrchestrationCheckpoint | None,
    *,
    fresh_architecture_entry: bool,
) -> bool:
    """Statically require certification for every resumed Architecture-9 role."""
    return bool(
        checkpoint is not None
        and checkpoint.architecture_version >= 9
        and not fresh_architecture_entry
        and checkpoint.current_role != "strategy_tournament"
    )


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def checkpoint_hash(checkpoint: OrchestrationCheckpoint) -> str:
    return _digest(asdict(checkpoint))


def ledger_hash(ledger: Mapping[str, Any]) -> str:
    return _digest(dict(ledger))


def build_resume_report_provenance(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
) -> tuple[dict[str, Any], str, Path]:
    """Persist provenance from verified target-bound checkpoint artifacts."""
    dag, _payloads = _artifact_dag(checkpoint)
    roles = {item["role"] for item in dag}
    if not {"strategy_tournament", "definition_auditor"} <= roles:
        raise ResumeCertificateError(
            "RESUME_REPORT_STRATEGY_AND_AUDITOR_REQUIRED"
        )
    payload = {
        "schema_version": 1,
        "mode": "architecture9_certified_resume",
        "target_obligation_id": checkpoint.target_obligation_id,
        "target_context_hash": checkpoint.target_context_hash,
        "parent_statement_hash": checkpoint.parent_statement_sha256,
        "environment_hash": checkpoint.target_environment_hash,
        "selected_plan_id": checkpoint.selected_strategy_plan_id,
        "selected_plan_hash": checkpoint.selected_strategy_plan_hash,
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "artifact_dependency_dag": dag,
    }
    digest = _digest(payload)
    payload["report_provenance_hash"] = digest
    path = Path(checkpoint_path).with_name(
        f"proof_orchestration.report_provenance.{digest}.json"
    )
    _atomic_write(path, _canonical(payload))
    return payload, digest, path


def current_runtime_binding(
    project_root: Path,
    *,
    tokenizer_id: str,
    residency_path: Path,
) -> dict[str, str]:
    """Fingerprint the pinned production runtime without trusting model output."""
    root = Path(project_root)
    source_hashes = {}
    for relative in (
        "scripts/agent_gan_repl.py",
        "autoresearch/prefill/orchestration_state.py",
        "autoresearch/prefill/architecture_v9.py",
        "autoresearch/prefill/resume_certificate.py",
    ):
        path = root / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        residency_encoded = Path(residency_path).expanduser().read_bytes()
        residency = json.loads(residency_encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeCertificateError(
            "RESUME_CERTIFICATE_RESIDENCY_STATE_REQUIRED"
        ) from exc
    gemma_pid = int(residency.get("gemma_pid", 0))
    if (
        residency.get("phase") != "GEMMA_SERVING"
        or residency.get("active_model") != "gemma"
        or int(residency.get("oprover_pid", -1)) != 0
        or int(residency.get("owner_pid", -1)) != 0
        or gemma_pid <= 0
    ):
        raise ResumeCertificateError("RESUME_CERTIFICATE_RESIDENCY_UNSAFE")
    try:
        os.kill(gemma_pid, 0)
    except OSError as exc:
        raise ResumeCertificateError(
            "RESUME_CERTIFICATE_GEMMA_OWNER_PID_STALE"
        ) from exc
    cache = {
        "model_id": os.environ.get(
            "KAKEYA_CACHE_MODEL_ID", Path(tokenizer_id).name
        ),
        "model_revision": os.environ.get(
            "KAKEYA_MODEL_REVISION", "local-4bit-v1"
        ),
        "tokenizer_revision": os.environ.get(
            "KAKEYA_TOKENIZER_REVISION", "gemma4-v1"
        ),
        "cache_format_version": os.environ.get(
            "KAKEYA_CACHE_FORMAT_VERSION", "kakeya-prefill-v3-kl-d4-q38"
        ),
    }
    return {
        "runtime_revision": _digest(source_hashes),
        **cache,
        "cache_revision_hash": _digest(cache),
        "residency_state_hash": hashlib.sha256(residency_encoded).hexdigest(),
    }


def certificate_pointer_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name(
        "proof_orchestration.resume_certificate.json"
    )


def certificate_journal_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name(
        "proof_orchestration.resume_certificates.journal.jsonl"
    )


def certificate_key_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name(".resume_certificate_hmac_key")


def resume_lease_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name("proof_orchestration.resume_lease.json")


def resume_lease_journal_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name(
        "proof_orchestration.resume_leases.journal.jsonl"
    )


def _certificate_dir(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name(
        "proof_orchestration.resume_certificates"
    )


def create_host_key(path: Path) -> None:
    """Create the host key explicitly; runtime verification never creates it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, os.urandom(32))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_key(path: Path) -> bytes:
    try:
        key = Path(path).read_bytes()
    except OSError as exc:
        raise ResumeCertificateError("RESUME_CERTIFICATE_HOST_KEY_REQUIRED") from exc
    if len(key) != 32:
        raise ResumeCertificateError("RESUME_CERTIFICATE_HOST_KEY_INVALID")
    return key


def _artifact_dag(
    checkpoint: OrchestrationCheckpoint,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not checkpoint.validated_artifacts:
        raise ResumeCertificateError("RESUME_CERTIFICATE_ARTIFACTS_REQUIRED")
    active_hashes = {
        reference.sha256 for reference in checkpoint.validated_artifacts.values()
    }
    dag = []
    payloads: dict[str, dict[str, Any]] = {}
    for role, reference in sorted(checkpoint.validated_artifacts.items()):
        if (
            reference.schema_version != 1
            or reference.target_obligation_id != checkpoint.target_obligation_id
            or reference.target_context_hash != checkpoint.target_context_hash
            or reference.parent_statement_hash
            != checkpoint.parent_statement_sha256
            or reference.strategy_plan_hash
            != checkpoint.target_strategy_plan_hash
            or reference.environment_hash
            != checkpoint.target_environment_hash
        ):
            raise ResumeCertificateError(
                f"RESUME_CERTIFICATE_ARTIFACT_BINDING_MISMATCH:{role}"
            )
        if any(dependency not in active_hashes for dependency in reference.dependencies):
            raise ResumeCertificateError(
                f"RESUME_CERTIFICATE_ARTIFACT_DAG_MISMATCH:{role}"
            )
        path = Path(reference.path)
        try:
            encoded = path.read_bytes()
            payload = json.loads(encoded)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeCertificateError(
                f"RESUME_CERTIFICATE_ARTIFACT_UNAVAILABLE:{role}"
            ) from exc
        if hashlib.sha256(encoded).hexdigest() != reference.sha256:
            raise ResumeCertificateError(
                f"RESUME_CERTIFICATE_ARTIFACT_HASH_MISMATCH:{role}"
            )
        if not isinstance(payload, dict) or payload.get("audit_only") is True:
            raise ResumeCertificateError(
                f"RESUME_CERTIFICATE_ARTIFACT_NOT_EXECUTABLE:{role}"
            )
        payloads[role] = payload
        dag.append({
            "role": role,
            "sha256": reference.sha256,
            "schema_version": reference.schema_version,
            "dependencies": list(reference.dependencies),
            "source_run_id": reference.source_run_id,
        })
    return dag, payloads


def _validate_evidence(
    checkpoint: OrchestrationCheckpoint,
    *,
    report_provenance_hash: str,
    runtime_binding: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_checkpoint = {
        "target_obligation_id": checkpoint.target_obligation_id,
        "target_context_hash": checkpoint.target_context_hash,
        "target_environment_hash": checkpoint.target_environment_hash,
        "strategy_provider": checkpoint.strategy_provider,
        "strategy_model_id": checkpoint.strategy_model_id,
        "strategy_run_id": checkpoint.strategy_run_id,
        "strategy_prompt_hash": checkpoint.strategy_prompt_hash,
        "strategy_evidence_hash": checkpoint.strategy_evidence_hash,
        "strategy_memo_hash": checkpoint.strategy_memo_hash,
        "strategy_intent_hash": checkpoint.strategy_intent_hash,
        "strategy_intent_run_id": checkpoint.strategy_intent_run_id,
        "selected_strategy_plan_id": checkpoint.selected_strategy_plan_id,
        "selected_strategy_plan_hash": checkpoint.selected_strategy_plan_hash,
        "ledger_id": checkpoint.ledger_id,
        "migration_event": checkpoint.migration_event,
    }
    missing = sorted(name for name, value in required_checkpoint.items() if not value)
    if missing:
        raise ResumeCertificateError(
            "RESUME_CERTIFICATE_EVIDENCE_MISSING:" + ",".join(missing)
        )
    if checkpoint.selected_strategy_plan_id not in checkpoint.strategy_plan_ids:
        raise ResumeCertificateError("RESUME_CERTIFICATE_SELECTED_PLAN_ID_MISMATCH")
    if checkpoint.selected_strategy_plan_hash not in checkpoint.strategy_plan_hashes:
        raise ResumeCertificateError("RESUME_CERTIFICATE_SELECTED_PLAN_HASH_MISMATCH")
    if checkpoint.target_strategy_plan_hash != checkpoint.selected_strategy_plan_hash:
        raise ResumeCertificateError("RESUME_CERTIFICATE_TARGET_PLAN_MISMATCH")
    if checkpoint.strategy_run_status != "FINISHED":
        raise ResumeCertificateError("RESUME_CERTIFICATE_PROVIDER_RUN_INCOMPLETE")
    if len(report_provenance_hash) != 64:
        raise ResumeCertificateError("RESUME_CERTIFICATE_REPORT_PROVENANCE_REQUIRED")
    missing_runtime = sorted(
        field for field in REQUIRED_RUNTIME_FIELDS if not runtime_binding.get(field)
    )
    if missing_runtime:
        raise ResumeCertificateError(
            "RESUME_CERTIFICATE_RUNTIME_BINDING_MISSING:"
            + ",".join(missing_runtime)
        )
    dag, payloads = _artifact_dag(checkpoint)
    strategy = payloads.get("strategy_tournament")
    if strategy is None:
        raise ResumeCertificateError("RESUME_CERTIFICATE_STRATEGY_ARTIFACT_REQUIRED")
    plans = dict(zip(strategy.get("plan_ids", ()), strategy.get("plan_hashes", ())))
    if plans.get(checkpoint.selected_strategy_plan_id) != (
        checkpoint.selected_strategy_plan_hash
    ):
        raise ResumeCertificateError("RESUME_CERTIFICATE_STRATEGY_ARTIFACT_MISMATCH")
    return dag, strategy


def build_resume_certificate(
    checkpoint: OrchestrationCheckpoint,
    ledger: Mapping[str, Any],
    *,
    checkpoint_before_hash: str,
    checkpoint_after_hash: str,
    report_provenance_hash: str,
    runtime_binding: Mapping[str, str],
    intended_next_role: str,
    owner_pid: int,
    lease_id: str,
    lease: Mapping[str, Any],
    nonce: str,
    issued_at: float | None = None,
    ttl_s: float = 300.0,
) -> dict[str, Any]:
    """Build and validate unsigned host evidence without inventing any field."""
    now = time.time() if issued_at is None else float(issued_at)
    if checkpoint.architecture_version != 9:
        raise ResumeCertificateError("RESUME_CERTIFICATE_ARCHITECTURE_MISMATCH")
    if not checkpoint_before_hash or not checkpoint_after_hash:
        raise ResumeCertificateError("RESUME_CERTIFICATE_CHECKPOINT_PAIR_REQUIRED")
    if checkpoint_after_hash != checkpoint_hash(checkpoint):
        raise ResumeCertificateError("RESUME_CERTIFICATE_CHECKPOINT_AFTER_MISMATCH")
    if (
        str(ledger.get("ledger_id", "")) != checkpoint.ledger_id
        or int(ledger.get("version", -1)) != checkpoint.ledger_version
    ):
        raise ResumeCertificateError("RESUME_CERTIFICATE_LEDGER_MISMATCH")
    if not intended_next_role or intended_next_role != checkpoint.current_role:
        raise ResumeCertificateError("RESUME_CERTIFICATE_INTENDED_ROLE_MISMATCH")
    dag, strategy = _validate_evidence(
        checkpoint,
        report_provenance_hash=report_provenance_hash,
        runtime_binding=runtime_binding,
    )
    if owner_pid <= 0 or not lease_id or not nonce or not lease:
        raise ResumeCertificateError("RESUME_CERTIFICATE_LEASE_REQUIRED")
    lease_body = dict(lease)
    lease_hash = str(lease_body.pop("lease_hash", ""))
    if lease_hash != _digest(lease_body):
        raise ResumeCertificateError("RESUME_CERTIFICATE_LEASE_TAMPERED")
    if (
        lease.get("lease_id") != lease_id
        or int(lease.get("supervisor_pid", 0)) != owner_pid
        or lease.get("checkpoint_sha256") != checkpoint_after_hash
        or lease.get("target_context_hash") != checkpoint.target_context_hash
        or lease.get("intended_next_role") != intended_next_role
        or lease.get("runtime_sha256") != _digest(dict(runtime_binding))
    ):
        raise ResumeCertificateError("RESUME_CERTIFICATE_LEASE_MISMATCH")
    selected_payload = next(
        (
            item for item in strategy.get("plans", ())
            if item.get("plan_id") == checkpoint.selected_strategy_plan_id
        ),
        {},
    )
    remediation = (
        selected_payload.get("plan_kind") == "DEFINITION_RESOLUTION_PLAN"
    )
    if remediation and (
        intended_next_role != "definition_resolution"
        or selected_payload.get("proof_search_allowed") is not False
        or selected_payload.get("next_gate") != "RESEARCH_CONTRACT_GATE"
    ):
        raise ResumeCertificateError(
            "RESUME_CERTIFICATE_REMEDIATION_ROUTE_INVALID"
        )
    body = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_kind": CERTIFICATE_KIND,
        "architecture_version": checkpoint.architecture_version,
        "checkpoint_schema_version": checkpoint.schema_version,
        "target": {
            "obligation_id": checkpoint.target_obligation_id,
            "context_hash": checkpoint.target_context_hash,
            "parent_statement_hash": checkpoint.parent_statement_sha256,
            "root_goal_hash": checkpoint.root_goal_sha256,
            "environment_hash": checkpoint.target_environment_hash,
        },
        "provider": {
            "provider": checkpoint.strategy_provider,
            "model_id": checkpoint.strategy_model_id,
            "run_id": checkpoint.strategy_run_id,
            "prompt_hash": checkpoint.strategy_prompt_hash,
            "evidence_hash": checkpoint.strategy_evidence_hash,
            "memo_hash": checkpoint.strategy_memo_hash,
            "intent_hash": checkpoint.strategy_intent_hash,
            "intent_run_id": checkpoint.strategy_intent_run_id,
        },
        "selected_plan": {
            "id": checkpoint.selected_strategy_plan_id,
            "hash": checkpoint.selected_strategy_plan_hash,
        },
        "ledger": {
            "id": checkpoint.ledger_id,
            "version": checkpoint.ledger_version,
            "sha256": ledger_hash(ledger),
        },
        "checkpoints": {
            "before_sha256": checkpoint_before_hash,
            "after_sha256": checkpoint_after_hash,
        },
        "migration_event": checkpoint.migration_event,
        "report_provenance_hash": report_provenance_hash,
        "artifact_dependency_dag": dag,
        "runtime": dict(runtime_binding),
        "residency": {
            "phase": checkpoint.residency_phase,
            "active_model": checkpoint.active_model,
        },
        "intended_next_role": intended_next_role,
        "lease": {
            "id": lease_id,
            "owner_pid": int(owner_pid),
            "hash": lease_hash,
            "supervisor_generation": lease["supervisor_generation"],
        },
        "execution_policy": {
            "definition_resolution_only": remediation,
            "proof_search_allowed": not remediation,
            "oprover_allowed": not remediation,
            "next_gate": (
                "RESEARCH_CONTRACT_GATE" if remediation else "PROOF_SEARCH"
            ),
        },
        "issued_at": now,
        "expires_at": now + float(ttl_s),
        "nonce": nonce,
    }
    body["certificate_hash"] = _digest(body)
    return body


def _append_journal(path: Path, record: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(record)) + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_resume_lease(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    *,
    lease_id: str,
    supervisor_pid: int,
    supervisor_generation: str,
    ledger_sha256: str,
    environment_sha256: str,
    runtime_binding: Mapping[str, str],
    intended_next_role: str,
    active_conflict: bool,
    issued_at: float | None = None,
    ttl_s: float = 300.0,
) -> dict[str, Any]:
    """Create one exclusive target-bound lease; identical replay is idempotent."""
    if active_conflict:
        raise ResumeCertificateError("RESUME_LEASE_ACTIVE_CONFLICT")
    if (
        not lease_id
        or supervisor_pid <= 0
        or not supervisor_generation
        or len(ledger_sha256) != 64
        or len(environment_sha256) != 64
        or intended_next_role != checkpoint.current_role
    ):
        raise ResumeCertificateError("RESUME_LEASE_BINDING_INCOMPLETE")
    missing_runtime = sorted(
        name for name in REQUIRED_RUNTIME_FIELDS if not runtime_binding.get(name)
    )
    if missing_runtime:
        raise ResumeCertificateError("RESUME_LEASE_RUNTIME_INCOMPLETE")
    now = time.time() if issued_at is None else float(issued_at)
    body = {
        "schema_version": 1,
        "lease_kind": LEASE_KIND,
        "lease_id": lease_id,
        "supervisor_pid": int(supervisor_pid),
        "supervisor_generation": supervisor_generation,
        "target_obligation_id": checkpoint.target_obligation_id,
        "target_context_hash": checkpoint.target_context_hash,
        "architecture_version": checkpoint.architecture_version,
        "checkpoint_schema_version": checkpoint.schema_version,
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "ledger_sha256": ledger_sha256,
        "environment_sha256": environment_sha256,
        "runtime_sha256": _digest(dict(runtime_binding)),
        "residency_state_hash": runtime_binding["residency_state_hash"],
        "intended_next_role": intended_next_role,
        "issued_at": now,
        "expires_at": now + float(ttl_s),
        "consumed": False,
    }
    body["lease_hash"] = _digest(body)
    path = resume_lease_path(checkpoint_path)
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing == body:
                return body
            if (
                not existing.get("consumed")
                and now < float(existing.get("expires_at", 0))
            ):
                raise ResumeCertificateError("RESUME_LEASE_ACTIVE_CONFLICT")
        _atomic_write(path, _canonical(body))
        _append_journal(resume_lease_journal_path(checkpoint_path), {
            "kind": "resume_lease_acquired",
            "lease_id": lease_id,
            "lease_hash": body["lease_hash"],
            "supervisor_generation": supervisor_generation,
            "issued_at": now,
        })
    return body


def _load_verified_lease(
    checkpoint_path: Path,
    *,
    lease_id: str,
    owner_pid: int,
    now: float,
) -> dict[str, Any]:
    try:
        lease = json.loads(resume_lease_path(checkpoint_path).read_text(
            encoding="utf-8"
        ))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeCertificateError("RESUME_LEASE_REQUIRED") from exc
    unhashed = dict(lease)
    lease_hash = str(unhashed.pop("lease_hash", ""))
    if lease_hash != _digest(unhashed):
        raise ResumeCertificateError("RESUME_LEASE_TAMPERED")
    if (
        lease.get("lease_id") != lease_id
        or int(lease.get("supervisor_pid", 0)) != owner_pid
    ):
        raise ResumeCertificateError("RESUME_LEASE_OWNER_MISMATCH")
    if lease.get("consumed") or now >= float(lease.get("expires_at", 0)):
        raise ResumeCertificateError("RESUME_LEASE_STALE")
    return lease


def persist_resume_certificate(
    checkpoint_path: Path,
    certificate: Mapping[str, Any],
    *,
    key_path: Path | None = None,
) -> Path:
    """Sign, journal, archive, then publish the active certificate pointer."""
    checkpoint_path = Path(checkpoint_path)
    key = _read_key(key_path or certificate_key_path(checkpoint_path))
    body = dict(certificate)
    expected_hash = body.pop("certificate_hash", "")
    if expected_hash != _digest(body):
        raise ResumeCertificateError("RESUME_CERTIFICATE_CONTENT_HASH_MISMATCH")
    body["certificate_hash"] = expected_hash
    body["signature"] = hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()
    archive = _certificate_dir(checkpoint_path) / f"{expected_hash}.json"
    pointer = certificate_pointer_path(checkpoint_path)
    journal = certificate_journal_path(checkpoint_path)
    lock_path = pointer.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        encoded = _canonical(body)
        if archive.exists() and archive.read_bytes() != encoded:
            raise ResumeCertificateError("RESUME_CERTIFICATE_ARCHIVE_COLLISION")
        if not archive.exists():
            _atomic_write(archive, encoded)
        _append_journal(journal, {
            "kind": "resume_certificate_issued",
            "certificate_hash": expected_hash,
            "checkpoint_after_sha256": body["checkpoints"]["after_sha256"],
            "issued_at": body["issued_at"],
        })
        _atomic_write(pointer, encoded)
    return archive


def _consumed_hashes(journal: Path) -> set[str]:
    if not journal.exists():
        return set()
    consumed = set()
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResumeCertificateError("RESUME_CERTIFICATE_JOURNAL_CORRUPT") from exc
        if record.get("kind") == "resume_certificate_consumed":
            consumed.add(str(record.get("certificate_hash", "")))
    return consumed


def consume_resume_certificate(
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    ledger: Mapping[str, Any],
    *,
    runtime_binding: Mapping[str, str],
    intended_next_role: str,
    owner_pid: int,
    lease_id: str,
    now: float | None = None,
    key_path: Path | None = None,
) -> str:
    """Verify and durably consume one certificate before any resumed role runs."""
    checkpoint_path = Path(checkpoint_path)
    pointer = certificate_pointer_path(checkpoint_path)
    journal = certificate_journal_path(checkpoint_path)
    lock_path = pointer.with_suffix(".lock")
    key = _read_key(key_path or certificate_key_path(checkpoint_path))
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            encoded = pointer.read_bytes()
            signed = json.loads(encoded)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeCertificateError(
                "ARCHITECTURE9_CERTIFIED_RESUME_REQUIRED"
            ) from exc
        signature = str(signed.pop("signature", ""))
        if not hmac.compare_digest(
            signature,
            hmac.new(key, _canonical(signed), hashlib.sha256).hexdigest(),
        ):
            raise ResumeCertificateError("RESUME_CERTIFICATE_SIGNATURE_INVALID")
        certificate_hash = str(signed.get("certificate_hash", ""))
        unsigned = dict(signed)
        unsigned.pop("certificate_hash", None)
        if certificate_hash != _digest(unsigned):
            raise ResumeCertificateError("RESUME_CERTIFICATE_CONTENT_HASH_MISMATCH")
        timestamp = time.time() if now is None else float(now)
        if timestamp < float(signed["issued_at"]) or timestamp >= float(
            signed["expires_at"]
        ):
            raise ResumeCertificateError("RESUME_CERTIFICATE_EXPIRED")
        if certificate_hash in _consumed_hashes(journal):
            raise ResumeCertificateError("RESUME_CERTIFICATE_REPLAYED")
        active_lease = _load_verified_lease(
            checkpoint_path,
            lease_id=lease_id,
            owner_pid=owner_pid,
            now=timestamp,
        )
        expected = {
            "architecture_version": checkpoint.architecture_version,
            "checkpoint_schema_version": checkpoint.schema_version,
            "intended_next_role": intended_next_role,
        }
        if any(signed.get(name) != value for name, value in expected.items()):
            raise ResumeCertificateError("RESUME_CERTIFICATE_DISPATCH_MISMATCH")
        if signed.get("ledger", {}).get("sha256") != ledger_hash(ledger):
            raise ResumeCertificateError("RESUME_CERTIFICATE_LEDGER_MISMATCH")
        if signed.get("runtime") != dict(runtime_binding):
            raise ResumeCertificateError("RESUME_CERTIFICATE_RUNTIME_MISMATCH")
        if signed.get("target", {}).get("obligation_id") != (
            checkpoint.target_obligation_id
        ):
            raise ResumeCertificateError("RESUME_CERTIFICATE_TARGET_MISMATCH")
        if signed.get("provider", {}).get("model_id") != checkpoint.strategy_model_id:
            raise ResumeCertificateError("RESUME_CERTIFICATE_MODEL_MISMATCH")
        if signed.get("checkpoints", {}).get("after_sha256") != checkpoint_hash(
            checkpoint
        ):
            raise ResumeCertificateError("RESUME_CERTIFICATE_CHECKPOINT_STALE")
        lease = signed.get("lease", {})
        if lease.get("id") != lease_id or int(lease.get("owner_pid", 0)) != owner_pid:
            raise ResumeCertificateError("RESUME_CERTIFICATE_LEASE_MISMATCH")
        if (
            lease.get("hash") != active_lease.get("lease_hash")
            or lease.get("supervisor_generation")
            != active_lease.get("supervisor_generation")
        ):
            raise ResumeCertificateError("RESUME_CERTIFICATE_LEASE_MISMATCH")
        policy = signed.get("execution_policy", {})
        if intended_next_role == "definition_resolution" and (
            policy.get("definition_resolution_only") is not True
            or policy.get("proof_search_allowed") is not False
            or policy.get("oprover_allowed") is not False
        ):
            raise ResumeCertificateError("RESUME_CERTIFICATE_POLICY_MISMATCH")
        _validate_evidence(
            checkpoint,
            report_provenance_hash=str(signed.get("report_provenance_hash", "")),
            runtime_binding=runtime_binding,
        )
        _append_journal(journal, {
            "kind": "resume_certificate_consumed",
            "certificate_hash": certificate_hash,
            "consumed_at": timestamp,
            "consumer_pid": owner_pid,
            "lease_id": lease_id,
            "intended_next_role": intended_next_role,
        })
        consumed_lease = dict(active_lease)
        consumed_lease.pop("lease_hash", None)
        consumed_lease["consumed"] = True
        consumed_lease["consumed_at"] = timestamp
        consumed_lease["lease_hash"] = _digest(consumed_lease)
        _atomic_write(
            resume_lease_path(checkpoint_path),
            _canonical(consumed_lease),
        )
        _append_journal(resume_lease_journal_path(checkpoint_path), {
            "kind": "resume_lease_consumed",
            "lease_id": lease_id,
            "certificate_hash": certificate_hash,
            "consumed_at": timestamp,
        })
        pointer.unlink(missing_ok=True)
        directory = os.open(pointer.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return certificate_hash
