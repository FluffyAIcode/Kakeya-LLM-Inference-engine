import hashlib
import json

from autoresearch.prefill.certification_audit import (
    INCOMPLETE,
    REVOKED,
    SHORTCUT,
    STRICT,
    artifact_integrity,
    classify_candidate,
    collect_verified_candidate_hashes,
    commit_revocations,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()


def artifact(proof):
    digest = hashlib.sha256(proof.encode()).hexdigest()
    body = {
        "artifact_hash": "",
        "candidate_hash": digest,
        "proof_body": proof,
        "proof_body_hash": digest,
        "proof_byte_length": len(proof.encode()),
        "lean_output": "",
        "lean_output_hash": hashlib.sha256(b"").hexdigest(),
    }
    identity = {
        key: value for key, value in body.items() if key != "artifact_hash"
    }
    body["artifact_hash"] = hashlib.sha256(canonical(identity)).hexdigest()
    return body


def test_bulk_audit_classifies_bodies_and_hash_only_records():
    invalid = artifact("by\n  sorry")
    strict = artifact("by\n  exact h")
    artifacts = {
        invalid["candidate_hash"]: invalid,
        strict["candidate_hash"]: strict,
    }
    assert artifact_integrity(invalid)
    assert classify_candidate(
        invalid["candidate_hash"], artifacts, {},
    ) == SHORTCUT
    assert classify_candidate("missing", artifacts, {}) == INCOMPLETE
    assert classify_candidate(
        strict["candidate_hash"], artifacts,
        {strict["candidate_hash"]: True},
    ) == STRICT


def test_bulk_revocation_is_journaled_and_idempotent(tmp_path):
    invalid = artifact("by\n  sorry")
    candidate = invalid["candidate_hash"]
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "entries": [{
            "status": "INDEPENDENTLY_VERIFIED",
            "verified_candidate_hash": candidate,
        }],
    }))
    document = json.loads(path.read_text())
    assert collect_verified_candidate_hashes(document) == {candidate}
    journal = tmp_path / "revocations" / "journal.json"
    count = commit_revocations(
        {path: document},
        {candidate: SHORTCUT},
        report_hash="audit-hash",
        journal_path=journal,
    )
    assert count == 1
    migrated = json.loads(path.read_text())
    assert migrated["entries"][0]["status"] == REVOKED
    assert json.loads(journal.read_text())["state"] == "COMMITTED"
    assert commit_revocations(
        {path: migrated},
        {candidate: SHORTCUT},
        report_hash="audit-hash",
        journal_path=journal,
    ) == 0


def test_nested_judge_status_uses_sibling_candidate_hash():
    candidate = "a" * 64
    document = {
        "classification": "VERIFIED_SUPPORTING_LEMMA",
        "oprover": {"verified_candidate_hash": candidate},
        "lean": {"proof_status": "INDEPENDENTLY_VERIFIED"},
    }
    assert collect_verified_candidate_hashes(document) == {candidate}
