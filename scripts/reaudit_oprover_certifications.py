#!/usr/bin/env python3
"""Reaudit and revoke historical OProver certification metadata."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from autoresearch.prefill.certification_audit import (
    INCOMPLETE,
    STRICT,
    artifact_integrity,
    classify_candidate,
    collect_verified_candidate_hashes,
    commit_revocations,
    revoke_document,
)
from autoresearch.prefill.independent_reconstruction import (
    VerifiedProofArtifactError,
    VerifiedProofStore,
    build_reconstruction_package,
    recompile_verified_artifact,
)
from scripts.oprover_reconstruction_acceptance import (
    CANARIES,
    CANARY_TARGETS,
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certification-root", type=Path, required=True)
    parser.add_argument("--jensen-project", type=Path, required=True)
    parser.add_argument("--routes-project", type=Path, required=True)
    parser.add_argument("--revoke", action="store_true")
    return parser.parse_args()


def candidate_metadata(value: object, found: dict[str, dict]) -> None:
    if isinstance(value, dict):
        if value.get("status") == "INDEPENDENTLY_VERIFIED":
            candidate = (
                value.get("verified_candidate_hash")
                or value.get("selected_candidate_hash")
            )
            if candidate:
                current = found.setdefault(str(candidate), {})
                for key in (
                    "key", "target", "theorem_hash", "legacy_proposition_hash",
                    "environment_hash", "source_hash", "source_path",
                    "dependency_prefix_hash",
                ):
                    if value.get(key) and not current.get(key):
                        current[key] = value[key]
        for child in value.values():
            candidate_metadata(child, found)
    elif isinstance(value, list):
        for child in value:
            candidate_metadata(child, found)


def main() -> int:
    args = parse_args()
    root = args.certification_root.expanduser().resolve()
    documents: dict[Path, object] = {}
    artifacts: dict[str, dict] = {}
    artifact_paths: dict[str, Path] = {}
    verified: set[str] = set()
    references: dict[str, set[str]] = {}
    metadata: dict[str, dict] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        documents[path] = body
        candidate_metadata(body, metadata)
        found = collect_verified_candidate_hashes(body)
        verified.update(found)
        for candidate in found:
            references.setdefault(candidate, set()).add(str(path.relative_to(root)))
        if isinstance(body, dict) and body.get("proof_body"):
            candidate = str(body.get("candidate_hash", ""))
            artifacts[candidate] = body
            artifact_paths[candidate] = path

    recompiles: dict[str, bool] = {}
    recompile_logs: dict[str, str] = {}
    for candidate in sorted(verified):
        artifact = artifacts.get(candidate)
        if artifact is None or not artifact_integrity(artifact):
            continue
        if classify_candidate(candidate, artifacts, {}) != INCOMPLETE:
            continue
        entry_key = str(artifact.get("provenance", {}).get("entry_key", ""))
        if not entry_key.startswith("canary:"):
            continue
        canary_key = entry_key.split(":", 1)[1]
        source_text = CANARIES.get(canary_key)
        theorem_id = CANARY_TARGETS.get(canary_key)
        if source_text is None or theorem_id is None:
            continue
        with tempfile.TemporaryDirectory(prefix="oprover-strict-reaudit-") as raw:
            source = Path(raw) / str(artifact["source_path"])
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(source_text, encoding="utf-8")
            package = build_reconstruction_package(
                source_path=source,
                project_root=Path(raw),
                theorem_id=theorem_id,
                environment_hash=str(artifact["environment_hash"]),
            )
            artifact_path = artifact_paths[candidate]
            store = VerifiedProofStore(artifact_path.parents[1])
            try:
                _, lean = recompile_verified_artifact(
                    store=store,
                    artifact_reference=artifact_path,
                    package=package,
                    project_root=args.jensen_project,
                )
                recompiles[candidate] = lean.accepted and not lean.output
                recompile_logs[candidate] = lean.output
            except VerifiedProofArtifactError as exc:
                recompiles[candidate] = False
                recompile_logs[candidate] = str(exc)

    outcomes = {
        candidate: classify_candidate(candidate, artifacts, recompiles)
        for candidate in sorted(verified)
    }
    expected_revocations = 0
    for document in documents.values():
        _, count = revoke_document(
            copy.deepcopy(document), outcomes, report_hash="PENDING",
        )
        expected_revocations += count

    source_records = []
    route_queue = root / "architecture9-routes-20260801" / "queue-result.json"
    if route_queue in documents:
        queue = documents[route_queue]
        for entry in queue.get("entries", []):
            if entry.get("lean_existing_status") != "VERIFIED":
                continue
            source = args.routes_project / entry["source_path"]
            actual = sha256(source.read_bytes()) if source.is_file() else ""
            source_records.append({
                "key": entry["key"],
                "target": entry["target"],
                "classification": entry["classification"],
                "evidence_class": "EXISTING_ROUTE_SOURCE_LEAN_NOT_OPROVER",
                "expected_source_hash": entry["source_root_hash"],
                "actual_source_hash": actual,
                "source_hash_verified": actual == entry["source_root_hash"],
                "oprover_status": entry["oprover_status"],
            })

    report = {
        "schema_version": 1,
        "policy": "STRICT_NO_SORRY_ADMIT_SORRYAX_PLACEHOLDERS_OR_WARNINGS",
        "created_at": time.time(),
        "certification_root": str(root),
        "jensen_project": str(args.jensen_project.resolve()),
        "routes_project": str(args.routes_project.resolve()),
        "unique_verified_candidate_count": len(verified),
        "retained_proof_artifact_count": len(artifacts),
        "outcome_counts": dict(sorted(Counter(outcomes.values()).items())),
        "candidates": [{
            "candidate_hash": candidate,
            "artifact_path": (
                str(artifact_paths[candidate].relative_to(root))
                if candidate in artifact_paths else ""
            ),
            "theorem_id": str(artifacts.get(candidate, {}).get("theorem_id", "")),
            "historical_key": str(metadata.get(candidate, {}).get("key", "")),
            "historical_target": str(
                metadata.get(candidate, {}).get("target", ""),
            ),
            "environment_hash": str(
                artifacts.get(candidate, {}).get(
                    "environment_hash",
                    metadata.get(candidate, {}).get("environment_hash", ""),
                )
            ),
            "entry_key": str(
                artifacts.get(candidate, {}).get("provenance", {}).get(
                    "entry_key", "",
                )
            ),
            "outcome": outcomes[candidate],
            "historical_references": sorted(references.get(candidate, set())),
            "recompile_log": recompile_logs.get(candidate, ""),
        } for candidate in sorted(verified)],
        "existing_route_source_records": source_records,
        "existing_route_source_count": len(source_records),
        "existing_route_source_hashes_verified": all(
            item["source_hash_verified"] for item in source_records
        ),
        "production_ledger_modified": False,
        "expected_revoked_status_count": expected_revocations,
        "integrity_scheme": "sha256-content-addressed",
        "signature_scheme": "ssh-keygen-ed25519-detached",
    }
    identity = {key: value for key, value in report.items() if key != "report_hash"}
    report_hash = sha256(canonical(identity))
    report["report_hash"] = report_hash
    report_dir = root / "strict-reaudit-20260804"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_hash}.json"
    report_path.write_bytes(canonical(report) + b"\n")

    revoked = 0
    if args.revoke:
        journal = report_dir / f"{report_hash}.revocation-journal.json"
        revoked = commit_revocations(
            documents,
            outcomes,
            report_hash=report_hash,
            journal_path=journal,
        )
    print(json.dumps({
        "report": str(report_path),
        "report_hash": report_hash,
        "outcomes": report["outcome_counts"],
        "revoked_status_count": revoked,
        "strict_count": sum(value == STRICT for value in outcomes.values()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
