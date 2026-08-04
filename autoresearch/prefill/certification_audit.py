"""Strict migration helpers for historical OProver certification metadata."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Mapping


STRICT = "STRICTLY_INDEPENDENTLY_VERIFIED"
SHORTCUT = "INVALID_PROOF_SHORTCUT"
INCOMPLETE = "LEGACY_INCOMPLETE"
RECOMPILE_FAILED = "RECOMPILE_FAILED"
MISMATCH = "HASH/ENV_MISMATCH"
REVOKED = "CERTIFICATION_REVOKED"

_HOLE = re.compile(r"\b(?:sorry|admit|sorryAx)\b|\bby\?", re.IGNORECASE)
_WARNING = re.compile(r"(?mi)^.*\bwarning:")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def artifact_integrity(payload: Mapping[str, object]) -> bool:
    """Validate all self-contained hashes in a retained proof artifact."""
    proof = str(payload.get("proof_body", ""))
    identity = {
        key: value for key, value in payload.items()
        if key not in {"artifact_hash", "created_at"}
    }
    return (
        bool(payload.get("artifact_hash"))
        and _sha256(_canonical(identity)) == payload.get("artifact_hash")
        and _sha256(proof) == payload.get("proof_body_hash")
        and _sha256(proof) == payload.get("candidate_hash")
        and _sha256(str(payload.get("lean_output", "")))
        == payload.get("lean_output_hash")
        and len(proof.encode()) == payload.get("proof_byte_length")
    )


def classify_candidate(
    candidate_hash: str,
    artifacts: Mapping[str, Mapping[str, object]],
    recompiles: Mapping[str, bool] | None = None,
) -> str:
    """Classify one historically verified candidate under strict policy."""
    artifact = artifacts.get(candidate_hash)
    if artifact is None:
        return INCOMPLETE
    if not artifact_integrity(artifact):
        return MISMATCH
    proof = str(artifact.get("proof_body", ""))
    if _HOLE.search(proof):
        return SHORTCUT
    if _WARNING.search(str(artifact.get("lean_output", ""))):
        return RECOMPILE_FAILED
    if recompiles is None or candidate_hash not in recompiles:
        return INCOMPLETE
    return STRICT if recompiles[candidate_hash] else RECOMPILE_FAILED


def _contains_verified_status(value: object) -> bool:
    if isinstance(value, dict):
        if (
            value.get("status") == "INDEPENDENTLY_VERIFIED"
            or value.get("proof_status") == "INDEPENDENTLY_VERIFIED"
        ):
            return True
        return any(_contains_verified_status(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_verified_status(child) for child in value)
    return False


def _candidate_fields(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key in ("verified_candidate_hash", "selected_candidate_hash"):
            if value.get(key):
                found.add(str(value[key]))
        for child in value.values():
            found.update(_candidate_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_candidate_fields(child))
    return found


def collect_verified_candidate_hashes(value: object) -> set[str]:
    """Collect hashes in records containing a historical verified status."""
    if not _contains_verified_status(value):
        return set()
    return _candidate_fields(value)


def revoke_document(
    value: object,
    outcomes: Mapping[str, str],
    *,
    report_hash: str,
    inherited_outcome: str = "",
) -> tuple[object, int]:
    """Revoke every invalid status in one JSON document."""
    count = 0
    if isinstance(value, dict):
        candidate = (
            value.get("verified_candidate_hash")
            or value.get("selected_candidate_hash")
        )
        outcome = outcomes.get(str(candidate), "") if candidate else ""
        if not outcome:
            descendant_candidates = collect_verified_candidate_hashes(value)
            descendant_outcomes = {
                outcomes[candidate]
                for candidate in descendant_candidates
                if candidate in outcomes and outcomes[candidate] != STRICT
            }
            if len(descendant_outcomes) == 1:
                outcome = descendant_outcomes.pop()
        if not outcome:
            outcome = inherited_outcome
        if (
            value.get("status") == "INDEPENDENTLY_VERIFIED"
            and outcome
            and outcome != STRICT
        ):
            value["revoked_status"] = value["status"]
            value["status"] = REVOKED
            value["strict_audit_outcome"] = outcome
            value["strict_audit_report_hash"] = report_hash
            count += 1
        if (
            value.get("proof_status") == "INDEPENDENTLY_VERIFIED"
            and outcome
            and outcome != STRICT
        ):
            value["revoked_proof_status"] = value["proof_status"]
            value["proof_status"] = REVOKED
            value["strict_audit_outcome"] = outcome
            value["strict_audit_report_hash"] = report_hash
            count += 1
        if outcome and outcome != STRICT:
            if value.get("integration_status") == "VERIFIED_PROOF_ARTIFACT":
                value["integration_status"] = REVOKED
            if value.get("classification") == "VERIFIED_SUPPORTING_LEMMA":
                value["classification"] = REVOKED
        for key, child in tuple(value.items()):
            value[key], child_count = revoke_document(
                child, outcomes, report_hash=report_hash,
                inherited_outcome=outcome,
            )
            count += child_count
    elif isinstance(value, list):
        for index, child in enumerate(value):
            value[index], child_count = revoke_document(
                child, outcomes, report_hash=report_hash,
                inherited_outcome=inherited_outcome,
            )
            count += child_count
    return value, count


def commit_revocations(
    documents: Mapping[Path, object],
    outcomes: Mapping[str, str],
    *,
    report_hash: str,
    journal_path: Path,
) -> int:
    """Stage, journal, and atomically replace each affected JSON file."""
    staged: list[tuple[Path, Path]] = []
    revoked = 0
    for path, original in documents.items():
        updated, count = revoke_document(
            original, outcomes, report_hash=report_hash,
        )
        if not count:
            continue
        temporary = path.with_name(f".{path.name}.{report_hash}.audit.tmp")
        temporary.write_bytes(_canonical(updated) + b"\n")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        staged.append((path, temporary))
        revoked += count
    journal = {
        "schema_version": 1,
        "report_hash": report_hash,
        "state": "PREPARED",
        "created_at": time.time(),
        "files": [str(path) for path, _ in staged],
        "revoked_status_count": revoked,
    }
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_bytes(_canonical(journal) + b"\n")
    for path, temporary in staged:
        os.replace(temporary, path)
    journal["state"] = "COMMITTED"
    journal_path.write_bytes(_canonical(journal) + b"\n")
    return revoked
