#!/usr/bin/env python3
"""Fixed evaluation harness for Karpathy-style Prefill autoresearch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


class ReportValidationError(ValueError):
    """A report/evaluator contract failure, not an infrastructure failure."""


class ResumeValidationError(ReportValidationError):
    """A resumed report could not prove safe reuse of an upstream artifact."""

    def __init__(self, message: str, *, route_state: str = "CRITIC") -> None:
        super().__init__(message)
        self.route_state = route_state


REPORT_PROVENANCE_SCHEMA_VERSION = 1
CRITIC_ARTIFACT_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def critic_artifact_payload(
    stage: dict,
    *,
    source_run_id: str,
    target_obligation_id: str,
    candidate_sha256: str,
    strategy_sha256: str,
    parent_statement_sha256: str,
    parent_signature_sha256: str,
    root_goal_sha256: str,
    ledger_id: str,
    ledger_version: int,
    generator_output_sha256: str,
) -> dict:
    """Build the content-addressed evaluation artifact for a physical Critic."""
    if stage.get("name") != "agent_critic":
        raise ReportValidationError("Critic artifact requires a physical Critic stage")
    return {
        "schema_version": CRITIC_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "critic_evaluation",
        "source_run_id": source_run_id,
        "bindings": {
            "target_obligation_id": target_obligation_id,
            "candidate_sha256": candidate_sha256,
            "strategy_sha256": strategy_sha256,
            "parent_statement_sha256": parent_statement_sha256,
            "parent_signature_sha256": parent_signature_sha256,
            "root_goal_sha256": root_goal_sha256,
            "ledger_id": ledger_id,
            "ledger_version": int(ledger_version),
            "generator_output_sha256": generator_output_sha256,
        },
        "critic_stage": dict(stage),
    }


def load_critic_artifact(
    artifact_ref: dict,
    *,
    expected_bindings: dict,
) -> dict:
    """Verify hash, schema, source identity, and all checkpoint bindings."""
    required_ref = {
        "role", "sha256", "schema_version", "dependencies", "path",
        "source_run_id",
    }
    if not isinstance(artifact_ref, dict) or not required_ref.issubset(artifact_ref):
        raise ResumeValidationError("resumed report has no complete Critic artifact ref")
    if artifact_ref["role"] != "critic":
        raise ResumeValidationError("reused artifact role is not critic")
    if int(artifact_ref["schema_version"]) != CRITIC_ARTIFACT_SCHEMA_VERSION:
        raise ResumeValidationError("reused Critic artifact schema is incompatible")
    path = Path(str(artifact_ref["path"])).expanduser()
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ResumeValidationError(
            f"reused Critic artifact is unavailable: {exc}",
        ) from exc
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != str(artifact_ref["sha256"]):
        raise ResumeValidationError("reused Critic artifact hash mismatch")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ResumeValidationError("reused Critic artifact is not JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CRITIC_ARTIFACT_SCHEMA_VERSION
        or payload.get("artifact_kind") != "critic_evaluation"
        or payload.get("source_run_id") != artifact_ref["source_run_id"]
    ):
        raise ResumeValidationError("reused Critic artifact provenance mismatch")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ResumeValidationError("reused Critic artifact has no bindings")
    for name, expected in expected_bindings.items():
        actual = bindings.get(name)
        if name == "ledger_version":
            actual, expected = int(actual or 0), int(expected or 0)
        else:
            actual, expected = str(actual or ""), str(expected or "")
        if actual != expected:
            route = "GENERATOR" if name in {
                "target_obligation_id", "candidate_sha256", "strategy_sha256",
                "parent_statement_sha256", "parent_signature_sha256",
                "root_goal_sha256",
            } else "CRITIC"
            raise ResumeValidationError(
                f"reused Critic artifact {name} mismatch",
                route_state=route,
            )
    dependencies = list(artifact_ref.get("dependencies") or [])
    expected_dependencies = [
        str(bindings.get("candidate_sha256", "")),
        str(bindings.get("parent_statement_sha256", "")),
        str(bindings.get("parent_signature_sha256", "")),
        str(bindings.get("root_goal_sha256", "")),
        str(bindings.get("generator_output_sha256", "")),
    ]
    if dependencies != expected_dependencies:
        raise ResumeValidationError("reused Critic artifact dependency mismatch")
    stage = payload.get("critic_stage")
    if not isinstance(stage, dict) or stage.get("name") != "agent_critic":
        raise ResumeValidationError("reused Critic artifact has no physical stage")
    return payload


def _report_provenance(report: dict) -> dict | None:
    provenance = report.get("provenance")
    if provenance is None:
        provenance = report.get("config", {}).get("report_provenance")
    return provenance if isinstance(provenance, dict) else None


def _critic_for_evaluation(report: dict, candidate) -> tuple[dict, dict]:
    stages = report.get("stages", [])
    if not isinstance(stages, list):
        raise ReportValidationError("report stages must be a list")
    provenance = _report_provenance(report)
    if not provenance or provenance.get("mode", "fresh") == "fresh":
        critic = next(
            (stage for stage in stages if stage.get("name") == "agent_critic"),
            None,
        )
        if critic is None:
            raise ReportValidationError("fresh report has no physical Critic stage")
        return critic, {
            "critic_reused": False,
            "critic_source_run_id": report.get("id", ""),
            "critic_artifact_sha256": "",
        }
    required = {
        "schema_version", "mode", "resumed_from_state", "resumed_from_role",
        "strategy_reused", "generator_reused", "critic_reused",
        "bindings", "reused_artifacts", "newly_executed_stages",
    }
    if (
        not required.issubset(provenance)
        or provenance.get("schema_version") != REPORT_PROVENANCE_SCHEMA_VERSION
        or provenance.get("mode") != "resumed"
    ):
        raise ResumeValidationError("resumed report provenance schema is incomplete")
    if not all(
        provenance.get(name) is True
        for name in ("strategy_reused", "generator_reused", "critic_reused")
    ):
        raise ResumeValidationError("resumed report does not declare required reuse")
    actual_stage_names = [str(stage.get("name", "")) for stage in stages]
    if list(provenance["newly_executed_stages"]) != actual_stage_names:
        raise ResumeValidationError("resumed report executed-stage provenance mismatch")
    if "agent_critic" in actual_stage_names:
        raise ResumeValidationError("resumed report ambiguously reuses and executes Critic")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, dict):
        raise ResumeValidationError("resumed report bindings are missing")
    expected = {
        "target_obligation_id": candidate.TARGET_OBLIGATION_ID,
        "candidate_sha256": str(bindings.get("candidate_sha256", "")),
        "strategy_sha256": str(bindings.get("strategy_sha256", "")),
        "parent_statement_sha256": str(
            bindings.get("parent_statement_sha256", ""),
        ),
        "parent_signature_sha256": str(
            bindings.get("parent_signature_sha256", ""),
        ),
        "root_goal_sha256": str(bindings.get("root_goal_sha256", "")),
        "ledger_id": str(bindings.get("ledger_id", "")),
        "ledger_version": int(bindings.get("ledger_version", 0)),
    }
    candidate_hash = getattr(candidate, "CANDIDATE_SHA256", "")
    if candidate_hash and candidate_hash != expected["candidate_sha256"]:
        raise ResumeValidationError(
            "resumed report candidate hash mismatch",
            route_state="GENERATOR",
        )
    critic_ref = provenance.get("reused_artifacts", {}).get("critic")
    payload = load_critic_artifact(critic_ref, expected_bindings=expected)
    return payload["critic_stage"], {
        "critic_reused": True,
        "critic_source_run_id": payload["source_run_id"],
        "critic_artifact_sha256": critic_ref["sha256"],
        "resumed_from_state": provenance["resumed_from_state"],
        "resumed_from_role": provenance["resumed_from_role"],
        "newly_executed_stages": actual_stage_names,
    }


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("prefill_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(report: dict, candidate) -> dict:
    critic, evaluation_provenance = _critic_for_evaluation(report, candidate)
    prefix_tokens = int(critic.get("prefix_tokens", 0))
    warmup_s = float(critic.get("warmup_wall_s", 0))
    measured_tps = prefix_tokens / warmup_s if warmup_s > 0 else 0.0
    estimated_max_segment_s = (
        candidate.PREFILL_COMPUTE_CHUNK_TOKENS / measured_tps
        if measured_tps > 0 else float("inf")
    )
    delta = critic.get("delta", {})
    constraints = {
        "stage_ok": bool(critic.get("ok")),
        "complete": bool(critic.get("complete")),
        "full_context": (
            critic.get("review_scope") == "full"
            and int(critic.get("critic_omitted_tokens", -1)) == 0
            and int(critic.get("critic_context_tokens", -1))
            == int(critic.get("generator_full_tokens", -2))
        ),
        "recursive_protocol": (
            critic.get("critic_protocol")
            == "goal_anchored_recursive_gan_v3"
        ),
        "no_fallback": int(delta.get("fallbacks", 0)) == 0,
        "no_job_failure": int(delta.get("remote_job_failures", 0)) == 0,
        "segment_under_budget": (
            estimated_max_segment_s <= candidate.MAX_SEGMENT_SECONDS
        ),
        "final_only_snapshot": candidate.SNAPSHOT_MODE == "final_only",
        "candidate_requires_full_context": (
            candidate.REQUIRE_FULL_CONTEXT is True
        ),
        "candidate_forbids_fallback": candidate.ALLOW_FALLBACK is False,
    }
    return {
        "accepted": all(constraints.values()),
        "metric_cold_critic_prefill_s": warmup_s,
        "measured_prefill_tps": measured_tps,
        "estimated_max_segment_s": estimated_max_segment_s,
        "compute_chunk_tokens": candidate.PREFILL_COMPUTE_CHUNK_TOKENS,
        "candidate_id": candidate.CANDIDATE_ID,
        "target_obligation_id": candidate.TARGET_OBLIGATION_ID,
        "proof_obligations_total": int(
            critic.get("proof_obligations_total", 0),
        ),
        "proof_obligations_covered": int(
            critic.get("proof_obligations_covered", 0),
        ),
        "proof_obligations_unresolved": int(
            critic.get("proof_obligations_unresolved", 0),
        ),
        "constraints": constraints,
        "evaluation_provenance": evaluation_provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).with_name("candidate.py"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).with_name("results.tsv"),
    )
    args = parser.parse_args()
    candidate = _load_candidate(args.candidate)
    result = evaluate(json.loads(args.report.read_text()), candidate)
    write_header = not args.results.exists()
    with args.results.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "timestamp",
                "accepted",
                "metric_cold_critic_prefill_s",
                "measured_prefill_tps",
                "estimated_max_segment_s",
                "compute_chunk_tokens",
            ),
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({"timestamp": time.time(), **{
            key: result[key] for key in writer.fieldnames if key != "timestamp"
        }})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
