#!/usr/bin/env python3
"""Real Karpathy-style AutoResearch supervisor around one-shot GAN experiments."""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.prepare import (
    ReportValidationError,
    ResumeValidationError,
    _load_candidate,
    evaluate,
)
from autoresearch.prefill.lean_gate import warm_lean_environment
from autoresearch.prefill.live_status import AtomicLiveStatus
from autoresearch.prefill.atomic_definition import (
    record_semantic_iteration,
    verified_progress_vector,
)
from autoresearch.prefill.architecture_v9 import (
    run_architecture_v9_entry,
    run_host_definition_gate,
)
from autoresearch.prefill.cursor_strategy import CursorStrategyAdapter
from autoresearch.prefill.strategy_tournament import StrategyEvent
from autoresearch.prefill.orchestration_state import (
    BlockedEventType,
    BlockedExitEvent,
    OrchestrationCheckpoint,
    ProofState,
    append_blocked_event_journal,
    apply_blocked_exit_event,
    load_checkpoint as load_orchestration_checkpoint,
    reconcile_checkpoint_ledger_version,
    save_checkpoint as save_orchestration_checkpoint,
)
from autoresearch.prefill.semantic_decompose import (
    SemanticResponseIncomplete,
    SemanticUnitTooLarge,
    admit_token_ids,
    build_proof_step_interface,
    downstream_output_cap,
    lint_structured_prompt,
    repair_json_backslashes,
    scan_single_artifact_object,
    structured_transport_complete,
)


BLOCKED_HEARTBEAT_INTERVAL_S = 20.0
RESUMABLE_ORCHESTRATION_STATES = frozenset({
    ProofState.STRATEGY_TOURNAMENT,
    ProofState.RESEARCH_CONTRACT_GATE,
    ProofState.DEFINITION_AUDITOR,
    ProofState.COUNTEREXAMPLE_WORKER,
    ProofState.SYNTHESIS,
    ProofState.DEFINITION_RESOLUTION,
    ProofState.DECOMPOSER,
    ProofState.MATH_IR_TRANSLATION,
    ProofState.HOST_TYPED_IR_GATE,
    ProofState.LEAN_ELABORATION_GATE,
    ProofState.PROOF_SEARCH,
    ProofState.ADVERSARIAL_REVIEW,
    ProofState.JUDGE,
    ProofState.COMMIT,
    ProofState.PREMISE_AUDIT,
    ProofState.BLOCKED,
})


REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "target_obligation_id",
    "hypothesis",
    "generator_directive",
    "critic_directive",
    "prefill_compute_chunk_tokens",
)


class StrategyPrefillBudgetExceeded(ValueError):
    def __init__(self, token_count: int, max_tokens: int) -> None:
        self.token_count = int(token_count)
        self.max_tokens = int(max_tokens)
        super().__init__(
            "Strategy Prefill token budget exceeded without truncation: "
            f"{self.token_count} > {self.max_tokens}",
        )


class CandidateNoveltyStagnation(ValueError):
    """Expected condition when neither strategy source has a novel candidate."""

    def __init__(
        self,
        *,
        candidate: dict,
        hypothesis_sha256: str,
        candidate_sha256: str,
        reasons: list[str],
        strategy_mode: str,
    ) -> None:
        self.candidate = candidate
        self.hypothesis_sha256 = hypothesis_sha256
        self.candidate_sha256 = candidate_sha256
        self.reasons = tuple(reasons)
        self.strategy_mode = strategy_mode
        super().__init__(
            "candidate novelty exhausted: " + ", ".join(self.reasons),
        )


def _json_request(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def _wait_port(host: str, port: int, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"service did not become ready: {host}:{port}")


def _candidate_snapshot(module) -> dict:
    return {
        "candidate_id": str(module.CANDIDATE_ID),
        "target_obligation_id": str(module.TARGET_OBLIGATION_ID),
        "hypothesis": str(module.HYPOTHESIS),
        "generator_directive": str(module.GENERATOR_DIRECTIVE),
        "critic_directive": str(module.CRITIC_DIRECTIVE),
        "prefill_compute_chunk_tokens": int(
            module.PREFILL_COMPUTE_CHUNK_TOKENS,
        ),
        "snapshot_mode": str(module.SNAPSHOT_MODE),
        "max_segment_seconds": float(module.MAX_SEGMENT_SECONDS),
        "require_full_context": bool(module.REQUIRE_FULL_CONTEXT),
        "allow_fallback": bool(module.ALLOW_FALLBACK),
    }


def validate_candidate(candidate: dict) -> None:
    text_fields = {
        "candidate_id",
        "target_obligation_id",
        "hypothesis",
        "generator_directive",
        "critic_directive",
    }
    missing = [
        field
        for field in REQUIRED_CANDIDATE_FIELDS
        if (
            not candidate.get(field)
            or (
                field in text_fields
                and not isinstance(candidate.get(field), str)
            )
        )
    ]
    if missing:
        raise ValueError(f"candidate missing fields: {missing}")
    if candidate["prefill_compute_chunk_tokens"] not in (64, 128, 256):
        raise ValueError("chunk tokens must be one of 64, 128, 256")
    if candidate.get("snapshot_mode", "final_only") != "final_only":
        raise ValueError("snapshot mode must remain final_only")
    if candidate.get("require_full_context", True) is not True:
        raise ValueError("candidate must require full context")
    if candidate.get("allow_fallback", False) is not False:
        raise ValueError("candidate must forbid fallback")
    plan = candidate.get("plan")
    if isinstance(plan, dict) and len(plan.get("steps", [])) > 1:
        raise ValueError("Strategy candidate must propose exactly one step")


def _select_repair_target(current: dict, ledger: dict) -> str:
    leaves = _pending_leaf_ids(ledger)
    if not leaves:
        raise ValueError("proof ledger has no unresolved leaf")
    backjump_target = str(ledger.get("backjump_target_id", ""))
    if backjump_target:
        if backjump_target in leaves:
            return backjump_target
        parents = {
            str(item.get("obligation_id", "")): str(item.get("parent_id", ""))
            for item in ledger.get("obligations", [])
        }

        def descends_from_backjump(obligation_id: str) -> bool:
            cursor = obligation_id
            visited = set()
            while cursor and cursor not in visited:
                if cursor == backjump_target:
                    return True
                visited.add(cursor)
                cursor = parents.get(cursor, "")
            return False

        backjump_leaves = [
            obligation_id
            for obligation_id in leaves
            if descends_from_backjump(obligation_id)
        ]
        if backjump_leaves:
            return backjump_leaves[0]
    current_target = str(current.get("target_obligation_id", ""))
    if current_target in leaves:
        return current_target
    parents = {
        str(item.get("obligation_id", "")): str(item.get("parent_id", ""))
        for item in ledger.get("obligations", [])
    }
    cursor = parents.get(current_target, "")
    visited = set()
    while cursor and cursor not in visited:
        if cursor in leaves:
            return cursor
        visited.add(cursor)
        cursor = parents.get(cursor, "")

    def distance_from_current(obligation_id: str) -> int:
        distance = 0
        cursor = obligation_id
        visited = set()
        while cursor and cursor not in visited:
            if cursor == current_target:
                return distance
            visited.add(cursor)
            cursor = parents.get(cursor, "")
            distance += 1
        return -1

    descendants = [
        (distance_from_current(obligation_id), obligation_id)
        for obligation_id in leaves
        if distance_from_current(obligation_id) >= 0
    ]
    if descendants:
        return max(descendants)[1]
    return leaves[0]


def repair_candidate_schema(
    candidate: dict,
    *,
    current: dict,
    ledger: dict,
) -> tuple[dict, list[str]]:
    aliases = {
        "candidate_id": ("id", "strategy_id"),
        "target_obligation_id": (
            "target", "target_id", "obligation_id",
        ),
        "hypothesis": ("strategy_hypothesis",),
        "generator_directive": (
            "generator", "generator_prompt", "generator_strategy",
        ),
        "critic_directive": (
            "critic", "critic_prompt", "critic_strategy",
        ),
        "prefill_compute_chunk_tokens": (
            "chunk_tokens", "compute_chunk_tokens",
        ),
    }
    normalized = {
        str(key).strip().lower().replace("-", "_"): value
        for key, value in candidate.items()
    }
    repaired = dict(candidate)
    changed: list[str] = []
    for field, field_aliases in aliases.items():
        if repaired.get(field):
            continue
        for alias in (field, *field_aliases):
            value = normalized.get(alias)
            if value not in (None, ""):
                repaired[field] = value
                changed.append(field)
                break
    nested_hypothesis = repaired.get("hypothesis")
    if isinstance(nested_hypothesis, dict):
        nested_target = (
            nested_hypothesis.get("target_obligation")
            or nested_hypothesis.get("target_obligation_id")
            or nested_hypothesis.get("target")
        )
        if nested_target and not repaired.get("target_obligation_id"):
            repaired["target_obligation_id"] = str(nested_target).strip()
            changed.append("target_obligation_id")
        repaired["hypothesis"] = str(
            nested_hypothesis.get("statement")
            or nested_hypothesis.get("text")
            or nested_hypothesis.get("claim")
            or ""
        ).strip()
        changed.append("hypothesis")
    for field in (
        "candidate_id",
        "target_obligation_id",
        "generator_directive",
        "critic_directive",
    ):
        value = repaired.get(field)
        if isinstance(value, dict):
            repaired[field] = str(
                value.get("statement")
                or value.get("text")
                or value.get("content")
                or ""
            ).strip()
            changed.append(field)
        elif value is not None and not isinstance(value, str):
            repaired[field] = str(value).strip()
            changed.append(field)
    target = str(repaired.get("target_obligation_id", ""))
    leaves = _pending_leaf_ids(ledger)
    if target not in leaves:
        repaired["target_obligation_id"] = _select_repair_target(
            current,
            ledger,
        )
        if "target_obligation_id" not in changed:
            changed.append("target_obligation_id")
    target = repaired["target_obligation_id"]
    statement = next(
        str(item.get("statement", ""))
        for item in ledger.get("obligations", [])
        if item.get("obligation_id") == target
    )
    hypothesis = str(repaired.get("hypothesis", "")).strip()
    repaired["hypothesis"] = hypothesis
    plan = repaired.get("plan", {})
    plan_steps = plan.get("steps", []) if isinstance(plan, dict) else []
    plan_text = " ".join(
        f"Step {index}: {str(step).strip()}"
        for index, step in enumerate(plan_steps, start=1)
        if str(step).strip()
    )
    if not repaired.get("generator_directive") and hypothesis:
        repaired["generator_directive"] = (
            f"Focus exclusively on {target}: {statement} "
            f"Construct and test this hypothesis: {hypothesis} "
            f"{plan_text}"
        )
        changed.append("generator_directive")
    if not repaired.get("critic_directive") and hypothesis:
        repaired["critic_directive"] = (
            f"Attempt to falsify the {target} hypothesis: {hypothesis} "
            "Identify the first invalid inference and one strictly smaller "
            "missing lemma."
        )
        changed.append("critic_directive")
    fixed_chunk_tokens = int(current["prefill_compute_chunk_tokens"])
    proposed_chunk_tokens = repaired.get("prefill_compute_chunk_tokens")
    try:
        proposed_chunk_tokens = int(proposed_chunk_tokens)
    except (TypeError, ValueError):
        proposed_chunk_tokens = None
    if (
        proposed_chunk_tokens is None
        or int(proposed_chunk_tokens) != fixed_chunk_tokens
    ):
        repaired["prefill_compute_chunk_tokens"] = fixed_chunk_tokens
        changed.append("prefill_compute_chunk_tokens")
    else:
        repaired["prefill_compute_chunk_tokens"] = fixed_chunk_tokens
    return repaired, changed


def render_candidate(candidate: dict) -> str:
    validate_candidate(candidate)
    return (
        '"""AutoResearch agent-editable strategy. Generated by supervisor."""\n\n'
        f"CANDIDATE_ID = {candidate['candidate_id']!r}\n"
        f"TARGET_OBLIGATION_ID = {candidate['target_obligation_id']!r}\n"
        f"HYPOTHESIS = {candidate['hypothesis']!r}\n"
        f"GENERATOR_DIRECTIVE = {candidate['generator_directive']!r}\n"
        f"CRITIC_DIRECTIVE = {candidate['critic_directive']!r}\n"
        f"PREFILL_COMPUTE_CHUNK_TOKENS = "
        f"{candidate['prefill_compute_chunk_tokens']}\n"
        'SNAPSHOT_MODE = "final_only"\n'
        f"MAX_SEGMENT_SECONDS = {candidate.get('max_segment_seconds', 300.0)!r}\n"
        "REQUIRE_FULL_CONTEXT = True\n"
        "ALLOW_FALLBACK = False\n"
    )


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            value["strategy_parse_mode"] = "json"
            return value
    for block in re.findall(
        r"```(?:json|python)?\s*(.*?)```",
        stripped,
        re.DOTALL | re.IGNORECASE,
    ):
        repaired = repair_json_backslashes(block.strip())
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            value["strategy_parse_mode"] = "json-escape-repaired"
            return value
        try:
            value = ast.literal_eval(block.strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            value["strategy_parse_mode"] = "python-literal"
            return value
    target_matches = re.findall(
        r"(?:Targeting\s+Leaf|pending\s+leaf)\*{0,2}\s*:?\s*"
        r"(?:\*\*)?`?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)`?(?:\*\*)?",
        stripped,
        re.IGNORECASE,
    )
    objective_match = re.search(
        r"Objective\*{0,2}\s*:\s*(.+)$",
        stripped,
        re.MULTILINE | re.IGNORECASE,
    )
    steps = [
        re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
        for line in stripped.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+\S", line)
        and "Objective" not in line
        and "Constraint" not in line
    ]
    objective = (
        objective_match.group(1).strip()
        if objective_match is not None
        else ""
    )
    if not objective:
        prose = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip()
            and not line.strip().startswith(("#", "```"))
        ]
        objective = " ".join(prose)
    if not objective:
        raise ValueError("strategy agent returned no usable candidate")
    digest = hashlib.sha256(stripped.encode()).hexdigest()[:12]
    return {
        "candidate_id": f"candidate-prose-{digest}",
        "target_obligation_id": (
            target_matches[-1] if target_matches else ""
        ),
        "hypothesis": objective,
        "plan": {"steps": steps},
        "strategy_parse_mode": "prose",
    }


def parse_research_verdict(output: str, candidate_id: str) -> dict:
    event_matches = re.findall(
        r"^\[autoresearch-verdict\]\s+(\{.*\})\s*$",
        output,
        re.MULTILINE,
    )
    if event_matches:
        fields = json.loads(event_matches[-1])
        if fields.get("candidate_id") != candidate_id:
            raise ValueError("research verdict candidate ID mismatch")
        if fields.get("outcome") not in {
            "SUPPORTED", "FALSIFIED", "DECOMPOSED", "INCONCLUSIVE",
        }:
            raise ValueError("invalid research verdict outcome")
        if (
            len(str(fields.get("evidence", ""))) < 40
            or len(str(fields.get("new_frontier", ""))) < 30
        ):
            raise ValueError("research verdict lacks substantive evidence/frontier")
        return {
            "outcome": fields["outcome"],
            "evidence": fields["evidence"],
            "new_frontier": fields["new_frontier"],
            "created_obligation_ids": list(
                fields.get("created_obligation_ids", []),
            ),
            "invalidation_kind": str(
                fields.get("invalidation_kind", ""),
            ),
            "backjump_target_id": str(
                fields.get("backjump_target_id", ""),
            ),
            "no_go_lesson_hashes": list(
                fields.get("no_go_lesson_hashes", []),
            ),
        }
    matches = list(re.finditer(
        r"^(?:critic>\s*)?### AUTORESEARCH_VERDICT\s*$"
        r"(?P<body>.*?)(?=^### |\Z)",
        output,
        re.MULTILINE | re.DOTALL,
    ))
    if not matches:
        raise ValueError("Critic emitted no AUTORESEARCH_VERDICT")
    body = matches[-1].group("body")
    fields = {}
    for name in ("Candidate ID", "Outcome", "Evidence", "New frontier"):
        match = re.search(
            rf"^{re.escape(name)}:\s*(.+)$",
            body,
            re.MULTILINE,
        )
        if not match:
            raise ValueError(f"research verdict missing {name}")
        fields[name] = match.group(1).strip()
    if fields["Candidate ID"] != candidate_id:
        raise ValueError("research verdict candidate ID mismatch")
    if fields["Outcome"] not in {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}:
        raise ValueError("invalid research verdict outcome")
    if len(fields["Evidence"]) < 40 or len(fields["New frontier"]) < 30:
        raise ValueError("research verdict lacks substantive evidence/frontier")
    return {
        "outcome": fields["Outcome"],
        "evidence": fields["Evidence"],
        "new_frontier": fields["New frontier"],
        "created_obligation_ids": [],
        "invalidation_kind": "",
        "backjump_target_id": "",
        "no_go_lesson_hashes": [],
    }


def _pending_leaf_ids(ledger: dict) -> list[str]:
    obligations = ledger.get("obligations", [])
    by_id = {
        str(item.get("obligation_id", "")): item
        for item in obligations
    }

    def invalidated_by_ancestor(item: dict) -> bool:
        cursor = str(item.get("parent_id", ""))
        visited = set()
        while cursor and cursor not in visited:
            visited.add(cursor)
            ancestor = by_id.get(cursor)
            if ancestor is None:
                break
            if (
                ancestor.get("status") == "QUARANTINED"
                or (
                    ancestor.get("status") == "DISPROVED"
                    and ancestor.get("invalidation_kind") in {
                        "PREMISE",
                        "PREMISE_INVALIDATED",
                    }
                )
            ):
                return True
            cursor = str(ancestor.get("parent_id", ""))
        return False

    unresolved = {
        str(item.get("obligation_id", ""))
        for item in obligations
        if (
            item.get("status") == "UNRESOLVED"
            and not invalidated_by_ancestor(item)
        )
    }
    unresolved_parents = {
        str(item.get("parent_id", ""))
        for item in obligations
        if (
            str(item.get("obligation_id", "")) in unresolved
            and item.get("parent_id")
        )
    }
    return sorted(unresolved - unresolved_parents)


def _build_legacy_strategy_research_state(
    *,
    current: dict,
    ledger: dict,
    results_text: str,
) -> dict:
    text_by_id: dict[str, str] = {}

    def intern(value) -> str:
        text = str(value or "")
        if not text:
            return ""
        text_id = hashlib.sha256(text.encode()).hexdigest()[:20]
        text_by_id[text_id] = text
        return text_id

    target_id = _select_repair_target(current, ledger)
    obligations = {
        str(item.get("obligation_id", "")): item
        for item in ledger.get("obligations", [])
    }
    ancestry = []
    cursor = target_id
    visited = set()
    while cursor and cursor not in visited:
        visited.add(cursor)
        item = obligations[cursor]
        ancestry_item = {
            "obligation_id": cursor,
            "statement_ref": intern(item.get("statement", "")),
            "status": item.get("status", ""),
            "parent_id": item.get("parent_id", ""),
            "last_run_id": item.get("last_run_id", ""),
        }
        if cursor == target_id:
            ancestry_item["last_evidence_ref"] = intern(
                item.get("last_evidence", ""),
            )
        ancestry.append(ancestry_item)
        cursor = str(item.get("parent_id", ""))
    ancestry.reverse()
    ancestry_ids = {
        item["obligation_id"] for item in ancestry
    }
    ancestry_refs = {
        item["obligation_id"]: f"a{index}"
        for index, item in enumerate(ancestry)
    }
    for item in ancestry:
        item["obligation_ref"] = ancestry_refs[item["obligation_id"]]
        parent_id = str(item.pop("parent_id", ""))
        item["parent_ref"] = ancestry_refs.get(parent_id, "")
        if item["obligation_id"] != target_id:
            item.pop("obligation_id", None)
            item.pop("last_run_id", None)

    def lesson_is_relevant(lesson: dict) -> bool:
        cursor = str(lesson.get("source_obligation_id", ""))
        visited = set()
        while cursor and cursor not in visited:
            if cursor in ancestry_ids:
                return True
            visited.add(cursor)
            source = obligations.get(cursor)
            if source is None:
                break
            cursor = str(source.get("parent_id", ""))
        return False

    relevant_lessons = [
        lesson
        for lesson in ledger.get("no_go_lessons", [])
        if (
            lesson.get("reversible_status", "ACTIVE") == "ACTIVE"
            and lesson_is_relevant(lesson)
        )
    ]
    relevant_rows = []
    if results_text.strip():
        for row in csv.DictReader(
            io.StringIO(results_text),
            delimiter="\t",
        ):
            if row.get("target_obligation_id") not in ancestry_ids:
                continue
            relevant_rows.append(row)

    def exact_record(row: dict) -> dict:
        return {
            "timestamp": row.get("timestamp", ""),
            "experiment_id": row.get("experiment_id", ""),
            "run_id": row.get("run_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "target_obligation_id": row.get("target_obligation_id", ""),
            "hypothesis_sha256": row.get("hypothesis_sha256", ""),
            "research_outcome": row.get("research_outcome", ""),
            "invalidation_kind": row.get("invalidation_kind", ""),
            "research_evidence": row.get("research_evidence", ""),
            "new_frontier": row.get("new_frontier", ""),
            "kept": row.get("kept", ""),
            "error": row.get("error", ""),
        }

    def record_hash(row: dict) -> str:
        encoded = json.dumps(
            exact_record(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()[:20]

    def is_kept(row: dict) -> bool:
        return row.get("kept") in {True, "True"}

    def is_proof_critical(row: dict) -> bool:
        return (
            is_kept(row)
            and row.get("research_outcome") in {
                "DECOMPOSED",
                "SUPPORTED",
                "FALSIFIED",
            }
        ) or row.get("invalidation_kind") == "PREMISE_INVALIDATED"

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in relevant_rows:
        key = (
            str(row.get("target_obligation_id", "")),
            str(row.get("hypothesis_sha256", "")),
        )
        grouped.setdefault(key, []).append(row)

    hypothesis_values = sorted({hypothesis for _, hypothesis in grouped})
    hypothesis_refs = {}
    for hypothesis in hypothesis_values:
        prefix_size = min(20, len(hypothesis))
        reference = hypothesis[:prefix_size]
        while (
            any(
                other != hypothesis and other.startswith(reference)
                for other in hypothesis_values
            )
            and prefix_size < len(hypothesis)
        ):
            prefix_size += 1
            reference = hypothesis[:prefix_size]
        hypothesis_refs[hypothesis] = reference
    experiment_groups = []
    latest_record_hashes = set()
    for (group_target, hypothesis_hash), rows in sorted(grouped.items()):
        latest = rows[-1]
        outcome_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}
        hashes = [record_hash(row) for row in rows]
        for row in rows:
            outcome = str(row.get("research_outcome", "") or "(none)")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            error = " ".join(str(row.get("error", "")).split())
            if error:
                fingerprint = hashlib.sha256(
                    error.encode(),
                ).hexdigest()[:20]
                error_counts[fingerprint] = error_counts.get(fingerprint, 0) + 1
        latest_record_hashes.add(hashes[-1])
        experiment_groups.append([
            ancestry_refs[group_target],
            hypothesis_refs[hypothesis_hash],
            len(rows),
            dict(sorted(outcome_counts.items())),
            dict(sorted(error_counts.items())),
            [
                rows[0].get("timestamp", ""),
            ],
            [
                latest.get("timestamp", ""),
            ],
            hashes[:-1],
            [
                hashes[-1],
                latest.get("research_outcome", ""),
                latest.get("invalidation_kind", ""),
                intern(latest.get("research_evidence", "")),
                intern(latest.get("new_frontier", "")),
                intern(latest.get("error", "")),
            ],
        ])

    critical_events = []
    for row in relevant_rows:
        if not is_proof_critical(row):
            continue
        item_hash = record_hash(row)
        event = [
            item_hash,
            ancestry_refs[str(row.get("target_obligation_id", ""))],
            hypothesis_refs[str(row.get("hypothesis_sha256", ""))],
            row.get("research_outcome", ""),
            row.get("invalidation_kind", ""),
        ]
        if item_hash not in latest_record_hashes:
            event.extend([
                intern(row.get("research_evidence", "")),
                intern(row.get("new_frontier", "")),
            ])
        critical_events.append(event)

    archive_outcomes: dict[str, int] = {}
    archive_errors: dict[str, int] = {}
    archive_hashes = []
    for row in relevant_rows:
        archive_hashes.append(record_hash(row))
        outcome = str(row.get("research_outcome", "") or "(none)")
        archive_outcomes[outcome] = archive_outcomes.get(outcome, 0) + 1
        error = " ".join(str(row.get("error", "")).split())
        if error:
            fingerprint = hashlib.sha256(error.encode()).hexdigest()[:20]
            archive_errors[fingerprint] = archive_errors.get(fingerprint, 0) + 1
    state = {
        "target_leaf_id": target_id,
        "target_ancestry": ancestry,
        "proof_critical_events": critical_events,
        "experiment_groups": experiment_groups,
        "event_view_schema": {
            "critical_event": (
                "[hash,target_ref,unique_hypothesis_hash_prefix,outcome,"
                "invalidation,(evidence_ref,frontier_ref if historical)]"
            ),
            "experiment_group": (
                "[target_ref,hypothesis_hash_prefix,count,outcomes,errors,"
                "first_ts,last_ts,prior_hashes,"
                "latest(hash,outcome,invalidation,evidence,frontier,error)]"
            ),
        },
        "archive_manifest": {
            "source": "append-only results.tsv",
            "record_count": len(relevant_rows),
            "group_count": len(experiment_groups),
            "hypothesis_set_sha256": hashlib.sha256(
                "".join(hypothesis_values).encode(),
            ).hexdigest(),
            "ordered_records_sha256": hashlib.sha256(
                "".join(archive_hashes).encode(),
            ).hexdigest(),
            "outcome_counts": dict(sorted(archive_outcomes.items())),
            "error_fingerprint_counts": dict(sorted(archive_errors.items())),
        },
        "current_candidate": {
            "candidate_id": current.get("candidate_id", ""),
            "target_obligation_id": current.get(
                "target_obligation_id",
                "",
            ),
            "hypothesis_ref": intern(current.get("hypothesis", "")),
            "prefill_compute_chunk_tokens": current.get(
                "prefill_compute_chunk_tokens",
            ),
        },
        "premise_recovery": {
            "backjump_target_id": ledger.get("backjump_target_id", ""),
            "no_go_lessons": [
                {
                    "claim_hash": lesson.get("claim_hash", ""),
                    "refuted_premise_ref": intern(
                        lesson.get("refuted_premise", ""),
                    ),
                    "evidence_ref": intern(lesson.get("evidence", "")),
                    "source_obligation_id": lesson.get(
                        "source_obligation_id",
                        "",
                    ),
                    "run_id": lesson.get("run_id", ""),
                    "confidence": lesson.get("confidence", 0.0),
                    "evidence_type": lesson.get("evidence_type", ""),
                    "evidence_source_ref": intern(
                        lesson.get("evidence_source", ""),
                    ),
                    "auditor_run_id": lesson.get("auditor_run_id", ""),
                    "proponent_run_id": lesson.get(
                        "proponent_run_id",
                        "",
                    ),
                    "reversible_status": lesson.get(
                        "reversible_status",
                        "ACTIVE",
                    ),
                }
                for lesson in relevant_lessons
            ],
        },
    }
    state["text_by_id"] = text_by_id
    return state


def build_strategy_research_state(
    *,
    current: dict,
    ledger: dict,
    results_text: str,
) -> dict:
    target_id = _select_repair_target(current, ledger)
    obligations = {
        str(item.get("obligation_id", "")): item
        for item in ledger.get("obligations", [])
    }
    target = obligations[target_id]
    parent = obligations.get(str(target.get("parent_id", "")))
    ancestry_ids = set()
    cursor = target_id
    while cursor and cursor not in ancestry_ids:
        ancestry_ids.add(cursor)
        cursor = str(obligations.get(cursor, {}).get("parent_id", ""))
    lessons = []
    for lesson in ledger.get("no_go_lessons", []):
        if lesson.get("reversible_status", "ACTIVE") != "ACTIVE":
            continue
        source = str(lesson.get("source_obligation_id", ""))
        visited = set()
        relevant = False
        while source and source not in visited:
            if source in ancestry_ids:
                relevant = True
                break
            visited.add(source)
            source = str(obligations.get(source, {}).get("parent_id", ""))
        if relevant:
            lessons.append({
                "claim_hash": lesson.get("claim_hash", ""),
                "refuted_premise": lesson.get("refuted_premise", ""),
                "evidence": lesson.get("evidence", ""),
                "evidence_type": lesson.get("evidence_type", ""),
                "confidence": lesson.get("confidence", 0.0),
            })
    record_hashes = []
    outcomes: dict[str, int] = {}
    latest_failure = {}
    if results_text.strip():
        for row in csv.DictReader(io.StringIO(results_text), delimiter="\t"):
            encoded = json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            record_hashes.append(hashlib.sha256(encoded.encode()).hexdigest())
            outcome = str(row.get("research_outcome", "") or "(none)")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            error = str(row.get("error", ""))
            if "SEMANTIC_RESPONSE_INCOMPLETE" in error:
                role_match = re.search(
                    r"SEMANTIC_RESPONSE_INCOMPLETE:\s+(\w+)\s+stopped",
                    error,
                )
                token_match = re.search(r"after\s+(\d+)\s+tokens", error)
                latest_failure = {
                    "kind": "SEMANTIC_RESPONSE_INCOMPLETE",
                    "role": (
                        role_match.group(1) if role_match else "unknown"
                    ),
                    "response_tokens": (
                        int(token_match.group(1)) if token_match else 0
                    ),
                }
    archive_manifest = {
        "source": "append-only results.tsv",
        "record_count": len(record_hashes),
        "ordered_records_sha256": hashlib.sha256(
            "".join(record_hashes).encode(),
        ).hexdigest(),
        "outcome_counts": dict(sorted(outcomes.items())),
        "latest_failure": latest_failure,
        "ledger_version": ledger.get("version", 0),
        "ledger_sha256": hashlib.sha256(
            json.dumps(
                ledger,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        ).hexdigest(),
    }
    root_goal_hash = str(ledger.get("root_goal_hash", "")) or hashlib.sha256(
        str(ledger.get("ledger_id", "")).encode(),
    ).hexdigest()
    interface = build_proof_step_interface(
        root_goal_hash=root_goal_hash,
        target=target,
        parent=parent,
        active_no_go_lessons=lessons,
        archive_manifest=archive_manifest,
    )
    return {
        "proof_step_interface": asdict(interface),
    }


STRATEGY_CONTRACT = """\
Propose exactly ONE novel falsifiable step for TARGET_LEAF_ID as the required
JSON fields; never emit a plan. Keep prefill_compute_chunk_tokens unchanged.
target_obligation_id must equal TARGET_LEAF_ID.
Respect the exact ProofStepInterface, active no-go premises, quarantine and
verified backjumps; never repeat a hypothesis or ancestor cycle.
On SEMANTIC_RESPONSE_INCOMPLETE, choose a strictly smaller question, not shorter
wording of the same step. Require retained-capacity, final-only snapshots,
complete Critic review, Primary decode-only, allens prefill-only, <=300s
segments, and no fallback, sampling, slicing, truncation, restart or cache
clearing. Archive hashes carry no mathematical meaning."""


def build_strategy_contract(program: str) -> str:
    required_markers = (
        "## Objective",
        "## Hard constraints",
        "Only a host-upgraded invalidation",
        "The concise stable `STRATEGY_CONTRACT`",
    )
    if any(marker not in program for marker in required_markers):
        raise ValueError("program is missing authoritative Strategy rules")
    return STRATEGY_CONTRACT


def build_strategy_prompt(
    *,
    program: str,
    current: dict,
    results_text: str,
    ledger: dict,
) -> str:
    research_state = build_strategy_research_state(
        current=current,
        ledger=ledger,
        results_text=results_text,
    )
    contract = build_strategy_contract(program)
    prompt = (
        "You are the AutoResearch strategy agent. Follow the authoritative "
        "human-owned Strategy contract exactly. Attack TARGET_LEAF_ID and propose "
        "exactly one falsifiable next proof step. Return JSON only with keys: "
        + ", ".join(REQUIRED_CANDIDATE_FIELDS)
        + ". prefill_compute_chunk_tokens is immutable and must equal "
        f"{current['prefill_compute_chunk_tokens']}. "
        "Do not weaken retained-capacity, final-only snapshot, or no-fallback "
        "rules. The hypothesis must encode one step, not a multi-level plan. "
        "It must not assume, rename, or propose any premise recorded in "
        "RESEARCH_STATE.proof_step_interface.active_no_go_lessons. "
        "It must either construct a concrete object or attempt a concrete "
        "counterexample for the target leaf. target_obligation_id must equal "
        "TARGET_LEAF_ID. The interface is complete for this certified step; "
        "archived prose is intentionally inactive and represented only by "
        "content hashes, never by an LLM summary."
        f"\n\nSTRATEGY_CONTRACT:\n{contract}"
        "\n\nRESEARCH_STATE:\n"
        f"{json.dumps(research_state, ensure_ascii=False, separators=(',', ':'))}"
    )
    lint_structured_prompt(prompt)
    return prompt


_STRATEGY_JSON_FENCE = re.compile(
    r"\A```json[^\S\r\n]*\r?\n(?P<object>.*)\r?\n```[^\S\r\n]*\Z",
    re.DOTALL,
)


def parse_strategy_candidate_transport(output: str) -> tuple[dict, str]:
    """Normalize one strategy object at the host transport boundary.

    A JSON-labelled Markdown envelope and literal LaTeX backslashes are
    transport defects, not proof attempts.  The host may remove only that
    exact envelope and repair only invalid JSON escape runs; the strict
    single-object scanner and later candidate gates remain authoritative.
    """
    stripped = output.strip()
    match = _STRATEGY_JSON_FENCE.fullmatch(stripped)
    parse_mode = "strict-json"
    if match is not None:
        stripped = match.group("object").strip()
        parse_mode = "host-unwrapped-json-fence"
    elif stripped.startswith("```"):
        raise ValueError("Strategy transport permits only one JSON object")
    artifact_text = scan_single_artifact_object(
        stripped,
        marker=None,
    ).json_text
    try:
        candidate = json.loads(artifact_text)
    except json.JSONDecodeError:
        repaired = repair_json_backslashes(artifact_text)
        if repaired == artifact_text:
            raise
        candidate = json.loads(repaired)
        parse_mode += "+host-repaired-json-escapes"
    if not isinstance(candidate, dict):
        raise ValueError("Strategy candidate JSON must be one object")
    return candidate, parse_mode


class StrategyPrefillHeartbeat:
    def __init__(
        self,
        dashboard: str = "http://127.0.0.1:8090",
        interval_s: float = 10.0,
        progress_callback=None,
    ) -> None:
        self.dashboard = dashboard.rstrip("/")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: dict = {}
        self._last: tuple | None = None
        self.progress_callback = progress_callback

    def __enter__(self):
        try:
            self._baseline = _json_request(
                f"{self.dashboard}/v1/network/summary",
            ).get("prefill", {})
        except Exception as exc:
            print(
                "[autoresearch] Strategy Prefill telemetry warning: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)
        self._emit()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._emit()

    def _delta(self, current: dict, name: str) -> int:
        return max(
            0,
            int(current.get(name, 0)) - int(self._baseline.get(name, 0)),
        )

    def _emit(self) -> None:
        try:
            current = _json_request(
                f"{self.dashboard}/v1/network/summary",
            ).get("prefill", {})
        except Exception:
            return
        new_remote_jobs = self._delta(current, "remote_jobs")
        if new_remote_jobs:
            # These two fields are gauges for the current job, not cumulative
            # counters. Subtracting the previous same-length job makes a retry
            # look permanently stuck at 0/0.
            total = int(current.get("remote_job_tokens_total", 0))
            computed = int(current.get("remote_job_tokens_computed", 0))
        else:
            total = 0
            computed = 0
        state = (
            computed,
            total,
            new_remote_jobs,
            self._delta(current, "remote_hits"),
            self._delta(current, "tokens_reused"),
        )
        if not total or state == self._last:
            return
        self._last = state
        if self.progress_callback is not None:
            self.progress_callback(computed, total)
        percent = min(100.0, 100.0 * computed / total)
        print(
            f"[autoresearch] Strategy Prefill: {computed}/{total} tokens "
            f"({percent:.1f}%) · remote_hits={state[3]} reused={state[4]}",
            flush=True,
        )


def propose_candidate(
    *,
    address: str,
    tokenizer_id: str,
    program: str,
    current: dict,
    results_text: str,
    ledger: dict,
    max_prefill_tokens: int = 8448,
    max_retained_tokens: int = 2052,
    live_status: AtomicLiveStatus | None = None,
    active_obligation_id: str = "",
) -> dict:
    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    prompt = build_strategy_prompt(
        program=program,
        current=current,
        results_text=results_text,
        ledger=ledger,
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    try:
        admit_token_ids(
            "Strategy ProofStepInterface",
            ids,
            configured_prefill_tokens=max_prefill_tokens,
            max_retained_tokens=max_retained_tokens,
            control_reserve_tokens=256,
        )
    except SemanticUnitTooLarge as exc:
        raise StrategyPrefillBudgetExceeded(
            exc.token_count,
            exc.max_tokens,
        ) from exc
    strategy_output_cap = downstream_output_cap(
        max_retained_tokens=max_retained_tokens,
        fixed_downstream_tokens=len(ids),
        configured_output_tokens=512,
        control_reserve_tokens=32,
    )
    generated: list[int] = []
    print(
        f"[autoresearch] Strategy Prefill: 0/{len(ids)} tokens (0.0%)",
        flush=True,
    )
    with Client(address) as client:
        with client.create_session(
            eos_token_ids=_resolve_eos_token_ids(tokenizer),
            client_label="autoresearch-strategy",
        ) as session:
            if live_status is not None:
                live_status.emit(
                    phase="strategy_prefill",
                    role="strategy",
                    state="prefill",
                    progress_current=0,
                    progress_total=len(ids),
                    progress_unit="tokens",
                    active_obligation_id=active_obligation_id,
                    worker="allens",
                    source="proof_supervisor",
                    force=True,
                )
            with StrategyPrefillHeartbeat(
                progress_callback=(
                    lambda current, total: live_status.emit(
                        phase="strategy_prefill",
                        role="strategy",
                        state="prefill",
                        progress_current=current,
                        progress_total=total,
                        progress_unit="tokens",
                        active_obligation_id=active_obligation_id,
                        worker="allens",
                        source="proof_supervisor",
                    )
                    if live_status is not None else None
                ),
            ):
                session.append(ids)
            print(
                f"[autoresearch] Strategy Prefill complete: {len(ids)} tokens",
                flush=True,
            )
            semantic_completed = False
            while len(generated) < strategy_output_cap:
                if live_status is not None:
                    live_status.emit(
                        phase="strategy_decode",
                        role="strategy",
                        state="decode",
                        progress_current=len(generated),
                        progress_total=strategy_output_cap,
                        progress_unit="tokens",
                        active_obligation_id=active_obligation_id,
                        worker="primary",
                        source="proof_supervisor",
                        hit_source="primary_hot",
                    )
                before = len(generated)
                for token in session.generate(
                    max_tokens=min(
                        64,
                        strategy_output_cap - len(generated),
                    ),
                ):
                    generated.append(int(token))
                    if structured_transport_complete(
                        tokenizer.decode(
                            generated,
                            skip_special_tokens=True,
                        ),
                        "strategy",
                    ):
                        semantic_completed = True
                        break
                print(
                    f"[autoresearch] Strategy Decode: {len(generated)} tokens "
                    "stop_reason="
                    f"{'semantic_complete' if semantic_completed else session.last_stop_reason}",
                    flush=True,
                )
                if semantic_completed:
                    break
                if session.last_stop_reason != 1:
                    break
                if len(generated) == before:
                    raise RuntimeError("strategy agent made no progress")
            if not semantic_completed and session.last_stop_reason != 2:
                raise SemanticResponseIncomplete(
                    "Strategy",
                    token_count=len(generated),
                    stop_reason=session.last_stop_reason,
                    response_cap_exhausted=(
                        len(generated) >= strategy_output_cap
                    ),
                )
    strategy_output = tokenizer.decode(generated, skip_special_tokens=True)
    print(
        f"[autoresearch] Strategy Output: {strategy_output.strip()}",
        flush=True,
    )
    candidate, parse_mode = parse_strategy_candidate_transport(strategy_output)
    print(
        f"[autoresearch] phase=strategy-parse mode={parse_mode}",
        flush=True,
    )
    candidate, repaired_fields = repair_candidate_schema(
        candidate,
        current=current,
        ledger=ledger,
    )
    if repaired_fields:
        print(
            "[autoresearch] phase=strategy-schema-repair "
            f"fields={','.join(sorted(set(repaired_fields)))} "
            f"target={candidate.get('target_obligation_id', '')}",
            flush=True,
        )
    candidate.update({
        "snapshot_mode": "final_only",
        "max_segment_seconds": 300.0,
        "require_full_context": True,
        "allow_fallback": False,
    })
    validate_candidate(candidate)
    return candidate


def check_runtime_health(
    worker_address: str,
    dashboard: str = "http://127.0.0.1:8090",
) -> dict:
    worker_host, worker_port_text = worker_address.rsplit(":", 1)
    _wait_port(worker_host, int(worker_port_text))
    _wait_port("127.0.0.1", 51051)
    _wait_port("127.0.0.1", 8090)
    summary = _json_request(
        f"{dashboard.rstrip('/')}/v1/network/summary",
    )
    if int(summary.get("online_nodes", 0)) < 1:
        raise RuntimeError("prefill fleet has no online worker")
    return summary


def _backup(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
    else:
        temporary = path.with_suffix(path.suffix + ".restore")
        temporary.write_bytes(content)
        os.chmod(temporary, 0o600)
        temporary.replace(path)


def run_gan_experiment(
    *,
    repo: Path,
    candidate_path: Path,
    state_path: Path,
    timeout_s: float,
    max_retained_tokens: int,
    live_status_path: Path,
    supervisor_pid: int,
    iteration: int,
    orchestration_state_path: Path,
    candidate_sha256: str,
) -> tuple[str, str]:
    command = [
        "bash", str(repo / "scripts/run_agent_gan_repl.sh"),
        "--skip-ensure", "--no-auto-loop",
        "--candidate-file", str(candidate_path),
        "--state-file", str(state_path),
        "--max-retained-tokens", str(max_retained_tokens),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=repo,
        env={
            **os.environ,
            "KAKEYA_LIVE_STATUS_PATH": str(live_status_path),
            "KAKEYA_SUPERVISOR_PID": str(supervisor_pid),
            "KAKEYA_SUPERVISOR_ITERATION": str(iteration),
            "KAKEYA_ORCHESTRATION_STATE_PATH": str(
                orchestration_state_path,
            ),
            "KAKEYA_CANDIDATE_SHA256": candidate_sha256,
        },
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write("/continue\n/quit\n")
    process.stdin.flush()
    process.stdin.close()
    timed_out = threading.Event()

    def terminate_on_timeout() -> None:
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout_s, terminate_on_timeout)
    timer.daemon = True
    timer.start()
    lines: list[str] = []
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = process.wait()
    finally:
        timer.cancel()
    output = "".join(lines)
    if timed_out.is_set():
        raise TimeoutError(
            f"GAN experiment exceeded {timeout_s}s: {output[-4000:]}",
        )
    if returncode != 0:
        raise RuntimeError(
            f"GAN experiment failed ({returncode}): {output[-4000:]}",
        )
    matches = re.findall(r"run=(br_[0-9a-f]+)", output)
    if not matches:
        raise RuntimeError("GAN experiment produced no benchmark run id")
    run_id = matches[-1]
    return run_id, output


def extract_gan_failure_reason(output: str) -> str:
    matches = re.findall(
        r"^\[inference-failed\].*?\berror=(.+)$",
        output,
        re.MULTILINE,
    )
    return matches[-1].strip() if matches else ""


def extract_report_provenance(output: str) -> dict | None:
    matches = re.findall(
        r"^\[report-provenance\] (.+)$",
        output,
        re.MULTILINE,
    )
    if not matches:
        return None
    payload = json.loads(matches[-1])
    if not isinstance(payload, dict):
        raise ReportValidationError("report provenance is not an object")
    return payload


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def best_kept(results: list[dict]) -> dict | None:
    kept = [row for row in results if row.get("kept") == "True"]
    if not kept:
        return None
    return min(
        kept,
        key=lambda row: (
            int(row["proof_obligations_unresolved"]),
            float(row["metric_cold_critic_prefill_s"]),
        ),
    )


def _created_ids(row: dict) -> list[str]:
    raw = row.get("created_obligation_ids", "")
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _row_made_progress(row: dict) -> bool:
    if row.get("kept") not in {True, "True"}:
        return False
    outcome = row.get("research_outcome")
    if row.get("invalidation_kind") in {
        "PREMISE",
        "PREMISE_INVALIDATED",
    }:
        return False
    if outcome in {"SUPPORTED", "FALSIFIED"}:
        return True
    if outcome == "DECOMPOSED":
        # Legacy rows predate created_obligation_ids but were already admitted
        # by the host's child-creation gate.
        return bool(_created_ids(row)) or not row.get(
            "created_obligation_ids",
        )
    return False


def strategy_trigger_reason(
    results: list[dict],
    *,
    stagnation_rounds: int,
    force: bool = False,
    trigger_file: Path | None = None,
) -> str:
    if force:
        return "manual-cli"
    if trigger_file is not None and trigger_file.exists():
        return "manual-trigger-file"
    if results and results[-1].get("research_outcome") == "STAGNATED":
        return ""
    if (
        results
        and results[-1].get("invalidation_kind") == "PREMISE_INVALIDATED"
    ):
        return "premise-invalidated"
    if (
        results
        and results[-1].get("invalidation_kind") == "APPROACH_FAILED"
    ):
        return "branch-falsified"
    if results and results[-1].get("research_outcome") == "FALSIFIED":
        return "branch-falsified"
    stagnant = 0
    for row in reversed(results):
        if _row_made_progress(row):
            break
        stagnant += 1
    if stagnant >= stagnation_rounds:
        return f"stagnation-{stagnant}"
    return ""


def infrastructure_failure_fingerprint(row: dict) -> str:
    """Return a stable fingerprint for a completed failed infrastructure run."""
    if row.get("research_outcome") != "EVALUATION_FAILED":
        return ""
    if row.get("failure_class") != "infrastructure":
        return ""
    error = " ".join(str(row.get("error", "")).lower().split())
    if not error:
        return ""
    return hashlib.sha256(error.encode()).hexdigest()


def failure_class_for_exception(exc: Exception) -> str:
    """Keep evaluator/orchestration defects out of the infrastructure circuit."""
    if "ResumeValidationError:" in str(exc):
        return "integration"
    if isinstance(
        exc,
        (
            ReportValidationError,
            ValueError,
            KeyError,
            TypeError,
            AssertionError,
        ),
    ):
        return "integration"
    return "infrastructure"


def is_resumable_checkpoint(
    checkpoint: OrchestrationCheckpoint | None,
    *,
    candidate_sha256: str = "",
) -> bool:
    """Return whether a persisted role must continue in another iteration."""
    return bool(
        checkpoint is not None
        and checkpoint.proof_state in RESUMABLE_ORCHESTRATION_STATES
        and (
            not candidate_sha256
            or checkpoint.candidate_sha256 == candidate_sha256
        )
    )


def is_nonfatal_semantic_continuation(
    row: dict,
    checkpoint: OrchestrationCheckpoint | None,
) -> bool:
    """Return whether a persisted semantic route requires another run.

    The checkpoint is authoritative.  Some completed GAN subprocesses report a
    stale ``running`` state while atomically persisting a valid semantic
    backjump, so their wrapper row can be labelled as an infrastructure
    failure.  Such rows must not consume the infrastructure circuit or trigger
    Strategy replanning.
    """
    return bool(
        row.get("supervisor_outcome") == "CONTINUE"
        and checkpoint is not None
        and checkpoint.proof_state in RESUMABLE_ORCHESTRATION_STATES
        and checkpoint.proof_state != ProofState.BLOCKED
        and not checkpoint.adapter_status
    )


def route_contract_to_subgoal_generation(
    checkpoint: OrchestrationCheckpoint,
) -> bool:
    """Require a typed elaborated subgoal before any OProver residency request."""
    if (
        checkpoint.proof_state != ProofState.PROOF_SEARCH
        or not checkpoint.research_contract_id
        or checkpoint.proof_plan_id
        or checkpoint.executable_plan_node_id
    ):
        return False
    if checkpoint.adapter_status:
        if checkpoint.blocked_reason != (
            "PROOF_ADVISOR_UNAVAILABLE:RESIDENCY_PROCESS_MANAGER_REQUIRED"
        ):
            return False
        checkpoint.clear_adapter_blocked(
            "research-contract-awaits-elaborated-subgoal",
        )
    event_id = hashlib.sha256(
        (
            checkpoint.target_obligation_id
            + checkpoint.research_contract_id
            + checkpoint.proposition_hash
        ).encode()
    ).hexdigest()
    if any(
        item.get("event_type") == "RESEARCH_CONTRACT_SUBGOAL_REQUIRED"
        and item.get("event_id") == event_id
        for item in checkpoint.recovery_events
    ):
        return False
    checkpoint.transition(
        ProofState.DECOMPOSER,
        "research-contract:generate-strictly-reducing-elaborated-subgoal",
        strategy_reused=True,
    )
    checkpoint.recovery_events.append({
        "event_type": "RESEARCH_CONTRACT_SUBGOAL_REQUIRED",
        "event_id": event_id,
        "target_obligation_id": checkpoint.target_obligation_id,
        "research_contract_id": checkpoint.research_contract_id,
        "selected_strategy_plan_id": checkpoint.selected_strategy_plan_id,
        "proposition_hash": checkpoint.proposition_hash,
        "target_state": ProofState.DECOMPOSER.value,
        "created_at": time.time(),
    })
    return True


def is_contract_bound_subgoal_resume(
    checkpoint: OrchestrationCheckpoint | None,
) -> bool:
    """Recognize a typed decomposition resume independent of wrapper candidate."""
    return bool(
        checkpoint is not None
        and checkpoint.proof_state == ProofState.DECOMPOSER
        and checkpoint.target_obligation_id
        and checkpoint.proposition_hash
        and checkpoint.target_context_hash
        and checkpoint.selected_strategy_plan_id
        and checkpoint.research_contract_id
        and not checkpoint.adapter_status
    )


def recover_contract_subgoal_duplicate_block(
    checkpoint: OrchestrationCheckpoint | None,
) -> BlockedExitEvent | None:
    """Repair only the legacy novelty block on a bound decomposition resume."""
    duplicate_reason = (
        "Strategy proposals were duplicates; reuse the current candidate "
        "and unresolved role."
    )
    if (
        checkpoint is None
        or checkpoint.proof_state != ProofState.BLOCKED
        or checkpoint.blocked_reason != duplicate_reason
        or checkpoint.proof_plan_id
        or checkpoint.executable_plan_node_id
        or not checkpoint.research_contract_id
        or not checkpoint.selected_strategy_plan_id
        or not checkpoint.target_context_hash
    ):
        return None
    evidence = next(
        (
            str(item.get("event_id", ""))
            for item in reversed(checkpoint.recovery_events)
            if item.get("event_type") == "RESEARCH_CONTRACT_SUBGOAL_REQUIRED"
            and item.get("target_obligation_id")
            == checkpoint.target_obligation_id
            and item.get("research_contract_id")
            == checkpoint.research_contract_id
        ),
        "",
    )
    if not evidence:
        return None
    event = BlockedExitEvent(
        event_id=hashlib.sha256(
            (
                "contract-subgoal-duplicate-recovery:"
                + evidence
                + checkpoint.target_context_hash
            ).encode()
        ).hexdigest(),
        event_type=BlockedEventType.VALIDATED_EVIDENCE_BACKJUMP.value,
        reason="target-bound decomposition bypasses candidate novelty",
        target_state=ProofState.DECOMPOSER.value,
        reset_role=ProofState.DECOMPOSER.value,
        metadata={"evidence_sha256": evidence},
    )
    apply_blocked_exit_event(checkpoint, event)
    return event


def should_resume_downstream(
    checkpoint: OrchestrationCheckpoint | None,
    *,
    candidate_sha256: str,
    force_strategy: bool,
    strategy_trigger_exists: bool,
) -> bool:
    """Keep a persisted role unless an explicit Strategy policy overrides it."""
    return bool(
        (
            is_resumable_checkpoint(
                checkpoint,
                candidate_sha256=candidate_sha256,
            )
            or is_contract_bound_subgoal_resume(checkpoint)
        )
        and not (
            checkpoint is not None
            and checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT
            and checkpoint.premise_outcome_type in {
                "APPROACH_FAILED",
                "PREMISE_INVALIDATED",
                "PARENT_STATEMENT_UNDERSPECIFIED",
            }
        )
        and not force_strategy
        and not strategy_trigger_exists
    )


def _normalized_hypothesis(text: str) -> str:
    return " ".join(
        re.findall(r"[^\W_]+", str(text).casefold(), flags=re.UNICODE),
    )


def _hypothesis_semantically_matches(left: str, right: str) -> bool:
    normalized_left = _normalized_hypothesis(left)
    normalized_right = _normalized_hypothesis(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    left_terms = set(normalized_left.split())
    right_terms = set(normalized_right.split())
    union = left_terms | right_terms
    jaccard = len(left_terms & right_terms) / len(union) if union else 0.0
    sequence = difflib.SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
    ).ratio()
    return jaccard >= 0.90 or sequence >= 0.94


def candidate_novelty_rejections(
    candidate: dict,
    *,
    current: dict,
    results: list[dict],
) -> tuple[list[str], str, str]:
    """Return durable hash/semantic duplicate reasons without mutating disk."""
    rendered = render_candidate(candidate).encode()
    candidate_sha256 = hashlib.sha256(rendered).hexdigest()
    hypothesis_sha256 = hashlib.sha256(
        candidate["hypothesis"].strip().lower().encode(),
    ).hexdigest()
    reasons: list[str] = []
    seen_hypotheses = {
        str(row.get("hypothesis_sha256", ""))
        for row in results
        if row.get("hypothesis_sha256")
    }
    seen_candidates = {
        str(row.get("candidate_sha256", ""))
        for row in results
        if row.get("candidate_sha256")
    }
    if hypothesis_sha256 in seen_hypotheses:
        reasons.append("hypothesis-hash-duplicate")
    if candidate_sha256 in seen_candidates:
        reasons.append("candidate-hash-duplicate")
    semantic_history = [str(current.get("hypothesis", ""))]
    semantic_history.extend(
        str(row.get("hypothesis", "") or row.get("research_hypothesis", ""))
        for row in results
    )
    if any(
        _hypothesis_semantically_matches(candidate["hypothesis"], previous)
        for previous in semantic_history
        if previous
    ):
        reasons.append("hypothesis-semantic-duplicate")
    return reasons, hypothesis_sha256, candidate_sha256


def build_host_candidate(
    current: dict,
    ledger: dict,
    *,
    target_id: str = "",
) -> dict:
    target_id = str(target_id or _select_repair_target(current, ledger))
    target = next(
        item
        for item in ledger.get("obligations", [])
        if item.get("obligation_id") == target_id
    )
    statement = str(target.get("statement", "")).strip()
    evidence = str(target.get("last_evidence", "")).strip()
    obligations = {
        str(item.get("obligation_id", "")): item
        for item in ledger.get("obligations", [])
    }
    target_ancestry = set()
    cursor = target_id
    while cursor and cursor not in target_ancestry:
        target_ancestry.add(cursor)
        cursor = str(obligations.get(cursor, {}).get("parent_id", ""))

    def lesson_is_relevant(lesson: dict) -> bool:
        source_id = str(lesson.get("source_obligation_id", ""))
        visited = set()
        while source_id and source_id not in visited:
            if source_id in target_ancestry:
                return True
            visited.add(source_id)
            source_id = str(
                obligations.get(source_id, {}).get("parent_id", ""),
            )
        return False

    no_go = "; ".join(
        str(lesson.get("refuted_premise", "")).strip()
        for lesson in ledger.get("no_go_lessons", [])
        if (
            str(lesson.get("refuted_premise", "")).strip()
            and lesson.get("reversible_status", "ACTIVE") == "ACTIVE"
            and lesson_is_relevant(lesson)
        )
    )
    no_go_directive = (
        f" Forbidden refuted premises: {no_go}."
        if no_go else ""
    )
    digest = hashlib.sha256(target_id.encode()).hexdigest()[:12]
    candidate = {
        "candidate_id": f"host-leaf-{digest}",
        "target_obligation_id": target_id,
        "hypothesis": statement,
        "generator_directive": (
            f"Resolve or falsify the exact target leaf {target_id}: "
            f"{statement} Previous Critic evidence: {evidence or '(none)'}. "
            "Provide an explicit derivation or counterexample; do not rename "
            f"the same gap as a new lemma.{no_go_directive}"
        ),
        "critic_directive": (
            f"Adversarially test target leaf {target_id}. Reject unsupported "
            "existence claims and semantic restatements. Mark PROVED or "
            "DISPROVED only with explicit evidence; otherwise identify one "
            "strictly smaller, falsifiable missing obligation."
        ),
        "prefill_compute_chunk_tokens": int(
            current["prefill_compute_chunk_tokens"],
        ),
        "snapshot_mode": "final_only",
        "max_segment_seconds": 300.0,
        "require_full_context": True,
        "allow_fallback": False,
    }
    validate_candidate(candidate)
    return candidate


def select_novel_candidate(
    proposed: dict,
    *,
    strategy_mode: str,
    current: dict,
    ledger: dict,
    results: list[dict],
) -> tuple[dict, str, str, str, bool]:
    """Select at most one host fallback, before candidate.py is mutated."""
    validate_candidate(proposed)
    rejections, hypothesis_sha256, candidate_sha256 = (
        candidate_novelty_rejections(
            proposed,
            current=current,
            results=results,
        )
    )
    used_host_fallback = False
    if strategy_mode == "gemma" and rejections:
        used_host_fallback = True
        strategy_mode = "host_strategy_deferred"
        proposed = build_host_candidate(current, ledger)
        validate_candidate(proposed)
        rejections, hypothesis_sha256, candidate_sha256 = (
            candidate_novelty_rejections(
                proposed,
                current=current,
                results=results,
            )
        )
    if rejections:
        raise CandidateNoveltyStagnation(
            candidate=proposed,
            hypothesis_sha256=hypothesis_sha256,
            candidate_sha256=candidate_sha256,
            reasons=rejections,
            strategy_mode=strategy_mode,
        )
    return (
        proposed,
        strategy_mode,
        hypothesis_sha256,
        candidate_sha256,
        used_host_fallback,
    )


def should_keep(result: dict, baseline: dict | None) -> bool:
    if not result["accepted"]:
        return False
    if int(result.get("proof_obligations_covered", 0)) != int(
        result.get("proof_obligations_total", 0),
    ):
        return False
    outcome = result.get("research_outcome")
    if outcome not in {"SUPPORTED", "FALSIFIED", "DECOMPOSED"}:
        return False
    if outcome == "DECOMPOSED" and not result.get("created_obligation_ids"):
        return False
    if baseline is None:
        return True
    return int(result["proof_obligations_unresolved"]) <= int(
        baseline["proof_obligations_unresolved"],
    )


RESULT_FIELDS = (
    "timestamp", "experiment_id", "run_id", "candidate_id",
    "target_obligation_id", "constraints_pass", "accepted", "kept",
    "metric_cold_critic_prefill_s", "baseline_metric_s",
    "proof_obligations_total", "proof_obligations_covered",
    "proof_obligations_unresolved", "compute_chunk_tokens",
    "candidate_sha256", "report_path",
    "hypothesis_sha256", "research_outcome", "research_evidence",
    "new_frontier", "created_obligation_ids", "strategy_mode",
    "invalidation_kind", "backjump_target_id", "no_go_lesson_hashes",
    "transcript_path", "error",
    "orchestration_state", "resumed_role", "resume_origin",
    "transition_reason", "retry_count", "strategy_reused",
    "generator_reused", "critic_reused", "critic_source_run_id",
    "critic_artifact_sha256", "newly_executed_stages", "failure_class",
    "supervisor_outcome", "continuation_reason",
)


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            old_fields = tuple(reader.fieldnames or ())
            old_rows = list(reader)
        if old_fields != RESULT_FIELDS:
            temporary = path.with_suffix(path.suffix + ".migrating")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=RESULT_FIELDS,
                    delimiter="\t",
                )
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({
                        field: old_row.get(field, "")
                        for field in RESULT_FIELDS
                    })
            temporary.replace(path)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


class BlockedIdleLogger:
    """Log BLOCKED transitions while suppressing unchanged poll noise."""

    def __init__(self) -> None:
        self._signature: tuple[str, str, str, str] | None = None

    def observe(self, checkpoint: OrchestrationCheckpoint) -> None:
        signature = (
            checkpoint.state,
            checkpoint.blocked_reason,
            checkpoint.target_obligation_id,
            checkpoint.last_blocked_event_id,
        )
        if self._signature == signature:
            return
        self._signature = signature
        print(
            "[autoresearch] phase=blocked-idle inference_started=false "
            f"reason={checkpoint.blocked_reason}",
            flush=True,
        )

    def transition(
        self,
        *,
        next_state: str,
        cause: str,
        event_id: str = "",
    ) -> None:
        self._signature = None


def run_iteration(args, iteration: int) -> dict:
    from scripts.agent_gan_repl import (
        audit_ledger_semantic_duplicates,
        load_proof_ledger,
        save_proof_ledger,
    )

    root = Path(__file__).resolve().parents[2]
    ar = Path(__file__).resolve().parent
    candidate_path = ar / "candidate.py"
    results_path = Path(args.results).expanduser()
    reports_dir = Path(args.reports_dir).expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file).expanduser()
    ledger_path = Path(args.proof_ledger).expanduser()
    program = (ar / "program.md").read_text()
    results = read_results(results_path)
    baseline = best_kept(results)
    current_module = _load_candidate(candidate_path)
    current = _candidate_snapshot(current_module)
    previous_candidate = candidate_path.read_bytes()
    previous_state = _backup(state_path)
    ledger_object = load_proof_ledger(ledger_path)
    live_status: AtomicLiveStatus = args._live_status
    live_status.set_context(iteration=iteration, run_id="")
    active_obligation_id = (
        _select_repair_target(current, asdict(ledger_object))
        if ledger_object is not None else current["target_obligation_id"]
    )
    live_status.emit(
        phase="iteration_boundary",
        role="supervisor",
        state="queued",
        active_obligation_id=active_obligation_id,
        source="proof_supervisor",
        force=True,
    )
    semantic_rejections = (
        audit_ledger_semantic_duplicates(ledger_object)
        if ledger_object is not None else []
    )
    if ledger_object is not None and semantic_rejections:
        save_proof_ledger(ledger_path, ledger_object)
        for obligation_id, ancestor_id, score in semantic_rejections:
            print(
                "[autoresearch] phase=semantic-retro-reject "
                f"id={obligation_id} duplicate_of={ancestor_id} "
                f"score={score:.2f}",
                flush=True,
            )
    previous_ledger = _backup(ledger_path)

    proposed = current
    gan_completed = False
    run_id = ""
    experiment_id = ""
    report_path = reports_dir / "not-started.json"
    transcript_path = reports_dir / "not-started.log"
    hypothesis_sha256 = ""
    candidate_sha256 = hashlib.sha256(previous_candidate).hexdigest()
    strategy_mode = "baseline"
    orchestration_state_path = Path(
        args.orchestration_state_file,
    ).expanduser()
    orchestration_checkpoint = load_orchestration_checkpoint(
        orchestration_state_path,
    )
    if (
        orchestration_checkpoint is not None
        and ledger_object is not None
        and reconcile_checkpoint_ledger_version(
            orchestration_checkpoint,
            ledger_object.version,
        )
    ):
        save_orchestration_checkpoint(
            orchestration_state_path,
            orchestration_checkpoint,
        )
    if (
        orchestration_checkpoint is not None
        and route_contract_to_subgoal_generation(orchestration_checkpoint)
    ):
        save_orchestration_checkpoint(
            orchestration_state_path,
            orchestration_checkpoint,
        )
        print(
            "[proof-live] stage=subgoal-generation "
            f"target={orchestration_checkpoint.target_obligation_id} "
            f"proposition={orchestration_checkpoint.target_statement} "
            f"plan={orchestration_checkpoint.selected_strategy_plan_id} "
            f"contract={orchestration_checkpoint.research_contract_id} "
            "next=DECOMPOSER reason=elaborated-subgoal-required",
            flush=True,
        )
    if (
        orchestration_checkpoint is not None
        and (
            orchestration_checkpoint.proof_state == ProofState.BLOCKED
            or orchestration_checkpoint.adapter_status in {
                "ADAPTER_BLOCKED",
                "INFRASTRUCTURE_BLOCKED",
                "INTEGRATION_BLOCKED",
            }
        )
    ):
        live_status.emit(
            phase="blocked_idle",
            role="orchestrator",
            state="idle",
            active_obligation_id=orchestration_checkpoint.target_obligation_id,
            source="proof_supervisor",
            force=True,
        )
        return {
            "iteration": iteration,
            "research_outcome": "BLOCKED",
            "orchestration_state": (
                orchestration_checkpoint.adapter_status
                or ProofState.BLOCKED.value
            ),
            "transition_reason": orchestration_checkpoint.blocked_reason,
            "failure_class": "",
            "error": "",
            "inference_started": False,
        }
    if (
        orchestration_checkpoint is not None
        and orchestration_checkpoint.proof_state == ProofState.COMMIT
        and orchestration_checkpoint.commit_key
        and ledger_object is not None
        and any(
            item.decomposition_certificate_hash
            == orchestration_checkpoint.commit_key
            for item in ledger_object.obligations
        )
    ):
        orchestration_checkpoint.committed = True
        orchestration_checkpoint.ledger_version = ledger_object.version
        orchestration_checkpoint.transition(
            ProofState.IDLE,
            "reconciled-idempotent-commit-after-crash",
            strategy_reused=True,
        )
        save_orchestration_checkpoint(
            orchestration_state_path,
            orchestration_checkpoint,
        )
    if orchestration_checkpoint is None:
        parent = next(
            (
                item for item in ledger_object.obligations
                if item.obligation_id == current["target_obligation_id"]
            ),
            None,
        ) if ledger_object is not None else None
        research_goal = ""
        if state_path.exists():
            try:
                research_goal = str(
                    json.loads(state_path.read_text()).get(
                        "research_goal",
                        "",
                    ),
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        orchestration_checkpoint = OrchestrationCheckpoint(
            state=ProofState.STRATEGY_TOURNAMENT.value,
            target_obligation_id=current["target_obligation_id"],
            candidate_sha256=hashlib.sha256(previous_candidate).hexdigest(),
            strategy_sha256=hashlib.sha256(previous_candidate).hexdigest(),
            parent_statement_sha256=(
                hashlib.sha256(parent.statement.encode()).hexdigest()
                if parent is not None else ""
            ),
            parent_signature_sha256=(
                parent.lean_signature_hash if parent is not None else ""
            ),
            root_goal_sha256=(
                hashlib.sha256(research_goal.encode()).hexdigest()
                if research_goal else ""
            ),
            current_role="strategy_tournament",
            last_transition_reason="migrated-current-ledger-and-candidate",
            resume_origin="legacy-checkpoint",
            strategy_reused=True,
            ledger_id=(
                ledger_object.ledger_id if ledger_object is not None else ""
            ),
            ledger_version=(
                ledger_object.version if ledger_object is not None else 0
            ),
        )
        save_orchestration_checkpoint(
            orchestration_state_path,
            orchestration_checkpoint,
        )
    interface_strategy_adapter = CursorStrategyAdapter()
    orchestration_checkpoint, definition_outcome = run_host_definition_gate(
        orchestration_state_path,
        orchestration_checkpoint,
        project_root=root,
        interface_strategy_adapter=(
            interface_strategy_adapter
            if interface_strategy_adapter.configured()
            else None
        ),
    )
    if definition_outcome:
        live_status.emit(
            phase="host_definition_gate",
            role="host",
            state=(
                "completed"
                if definition_outcome in {
                    "COMMITTED", "INTERFACE_REQUIRED",
                    "PARENT_STATEMENT_UNDERSPECIFIED",
                    "IDEMPOTENT_REPLAY", "IDENTICAL_QUERY_EXHAUSTED",
                } else "failed"
            ),
            active_obligation_id=orchestration_checkpoint.target_obligation_id,
            source="proof_supervisor",
            force=True,
        )
        return {
            "iteration": iteration,
            "research_outcome": definition_outcome,
            "orchestration_state": orchestration_checkpoint.state,
            "transition_reason": (
                orchestration_checkpoint.last_transition_reason
            ),
            "failure_class": "",
            "error": "",
            "inference_started": False,
            "supervisor_outcome": (
                "CONTINUE"
                if is_resumable_checkpoint(orchestration_checkpoint)
                else "ITERATION_COMPLETE"
            ),
            "continuation_reason": (
                orchestration_checkpoint.stagnation_reason
            ),
        }
    resume_downstream = should_resume_downstream(
        orchestration_checkpoint,
        candidate_sha256=hashlib.sha256(previous_candidate).hexdigest(),
        force_strategy=args.force_strategy,
        strategy_trigger_exists=Path(
            args.strategy_trigger_file,
        ).expanduser().exists(),
    )
    try:
        print(
            f"[autoresearch] iteration={iteration} "
            f"phase=runtime-health-check candidate={current['candidate_id']}",
            flush=True,
        )
        live_status.emit(
            phase="runtime_health_check",
            role="supervisor",
            state="review",
            active_obligation_id=active_obligation_id,
            source="proof_supervisor",
            force=True,
        )
        health = check_runtime_health(
            args.worker_address,
            args.dashboard,
        )
        print(
            "[autoresearch] phase=runtime-healthy "
            f"online_nodes={health.get('online_nodes', 0)} "
            f"kv_hit_rate={health.get('kv_hit_rate', 0):.1%}",
            flush=True,
        )
        ledger_data = json.loads(ledger_path.read_text())
        if state_path.exists():
            checkpoint_data = json.loads(state_path.read_text())
            research_goal = str(checkpoint_data.get("research_goal", ""))
            if research_goal:
                ledger_data["root_goal_hash"] = hashlib.sha256(
                    research_goal.encode(),
                ).hexdigest()
        trigger_file = Path(args.strategy_trigger_file).expanduser()
        trigger_reason = (
            ""
            if resume_downstream else strategy_trigger_reason(
                results,
                stagnation_rounds=args.strategy_stagnation_rounds,
                force=args.force_strategy and iteration == 0,
                trigger_file=trigger_file,
            )
        )
        if (
            not resume_downstream
            and orchestration_checkpoint is not None
            and orchestration_checkpoint.proof_state
            == ProofState.STRATEGY_TOURNAMENT
            and orchestration_checkpoint.premise_outcome_type in {
                "APPROACH_FAILED",
                "PREMISE_INVALIDATED",
                "PARENT_STATEMENT_UNDERSPECIFIED",
            }
        ):
            trigger_reason = (
                "typed-premise-"
                + orchestration_checkpoint.premise_outcome_type.lower()
            )
        architecture9_strategy = bool(
            orchestration_checkpoint is not None
            and orchestration_checkpoint.architecture_version >= 9
            and orchestration_checkpoint.proof_state
            == ProofState.STRATEGY_TOURNAMENT
        )
        if architecture9_strategy:
            strategy_mode = "cursor_strategy"
            proposed = build_host_candidate(
                current,
                ledger_data,
                target_id=orchestration_checkpoint.target_obligation_id,
            )
            print(
                "[autoresearch] phase=strategy-proposal "
                "mode=cursor_strategy fallback=disabled",
                flush=True,
            )
        elif resume_downstream:
            strategy_mode = "resumed"
            proposed = current
            hypothesis_sha256 = hashlib.sha256(
                current["hypothesis"].strip().lower().encode(),
            ).hexdigest()
            print(
                "[autoresearch] phase=orchestration-resume "
                f"state={orchestration_checkpoint.state} "
                f"role={orchestration_checkpoint.current_role} "
                f"origin={orchestration_checkpoint.resume_origin or 'checkpoint'} "
                "strategy_reused=true",
                flush=True,
            )
        elif baseline is None and iteration == 0 and not trigger_reason:
            print(
                "[autoresearch] phase=baseline using current candidate",
                flush=True,
            )
        elif trigger_reason:
            strategy_mode = "gemma"
            print(
                "[autoresearch] phase=strategy-proposal "
                f"mode=gemma trigger={trigger_reason}",
                flush=True,
            )
            try:
                proposed = propose_candidate(
                    address=args.address,
                    tokenizer_id=args.tokenizer_id,
                    program=program,
                    current=current,
                    results_text=(
                        results_path.read_text()
                        if results_path.exists() else ""
                    ),
                    ledger=ledger_data,
                    max_prefill_tokens=args.strategy_max_prefill_tokens,
                    max_retained_tokens=args.max_retained_tokens,
                    live_status=live_status,
                    active_obligation_id=active_obligation_id,
                )
                if proposed["target_obligation_id"] not in _pending_leaf_ids(
                    ledger_data,
                ):
                    raise ValueError(
                        "strategy agent targeted a non-leaf proof obligation",
                    )
            except StrategyPrefillBudgetExceeded as exc:
                strategy_mode = "host_strategy_deferred"
                proposed = build_host_candidate(current, ledger_data)
                print(
                    "[autoresearch] phase=strategy-deferred-budget "
                    f"tokens={exc.token_count} max={exc.max_tokens} "
                    f"fallback=deterministic-host",
                    flush=True,
                )
            except SemanticResponseIncomplete as exc:
                strategy_mode = "host_strategy_deferred"
                proposed = build_host_candidate(current, ledger_data)
                print(
                    "[autoresearch] phase=strategy-deferred-semantic "
                    f"reason={exc} fallback=deterministic-host",
                    flush=True,
                )
        else:
            strategy_mode = "host"
            proposed = build_host_candidate(current, ledger_data)
            print(
                "[autoresearch] phase=deterministic-candidate "
                f"target={proposed['target_obligation_id']}",
                flush=True,
            )
        if resume_downstream or architecture9_strategy:
            used_host_fallback = False
            hypothesis_sha256 = hashlib.sha256(
                proposed["hypothesis"].strip().lower().encode()
            ).hexdigest()
            candidate_sha256 = hashlib.sha256(
                render_candidate(proposed).encode()
            ).hexdigest()
        else:
            (
                proposed,
                strategy_mode,
                hypothesis_sha256,
                candidate_sha256,
                used_host_fallback,
            ) = select_novel_candidate(
                proposed,
                strategy_mode=strategy_mode,
                current=current,
                ledger=ledger_data,
                results=results,
            )
        if used_host_fallback:
            print(
                "[autoresearch] phase=strategy-deferred-repeat "
                "fallback=deterministic-host",
                flush=True,
            )
        if trigger_reason == "manual-trigger-file":
            trigger_file.unlink(missing_ok=True)
        hypothesis_novel = True
        candidate_path.write_text(render_candidate(proposed))
        print(
            f"[autoresearch] phase=candidate-written "
            f"candidate={proposed['candidate_id']} "
            f"target={proposed['target_obligation_id']} "
            f"mode={strategy_mode}",
            flush=True,
        )
        candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        if not resume_downstream and not architecture9_strategy:
            selected_parent = next(
                (
                    item for item in ledger_data.get("obligations", [])
                    if item.get("obligation_id")
                    == proposed["target_obligation_id"]
                ),
                {},
            )
            orchestration_checkpoint = OrchestrationCheckpoint(
                state=ProofState.STRATEGY_TOURNAMENT.value,
                target_obligation_id=proposed["target_obligation_id"],
                candidate_sha256=candidate_sha256,
                strategy_sha256=candidate_sha256,
                parent_statement_sha256=hashlib.sha256(
                    str(selected_parent.get("statement", "")).encode(),
                ).hexdigest(),
                parent_signature_sha256=str(
                    selected_parent.get("lean_signature_hash", ""),
                ),
                root_goal_sha256=str(
                    ledger_data.get("root_goal_hash", ""),
                ),
                current_role="strategy_tournament",
                last_transition_reason=(
                    f"strategy-trigger:{trigger_reason}"
                    if trigger_reason else "candidate-selected"
                ),
                strategy_reused=False,
                ledger_id=str(ledger_data.get("ledger_id", "")),
                ledger_version=int(ledger_data.get("version", 0)),
            )
            save_orchestration_checkpoint(
                orchestration_state_path,
                orchestration_checkpoint,
            )
        if (
            orchestration_checkpoint.proof_state
            == ProofState.STRATEGY_TOURNAMENT
        ):
            selected_parent = next(
                (
                    item for item in ledger_data.get("obligations", [])
                    if item.get("obligation_id")
                    == proposed["target_obligation_id"]
                ),
                {},
            )
            event_type = (
                StrategyEvent.INITIAL_BRANCH
                if not orchestration_checkpoint.strategy_event_id
                else StrategyEvent.TARGET_CHANGE
            )
            event_id = (
                f"{event_type.value}:"
                + hashlib.sha256(
                    (
                        proposed["target_obligation_id"]
                        + str(ledger_data.get("version", 0))
                    ).encode(),
                ).hexdigest()[:20]
            )
            orchestration_checkpoint = run_architecture_v9_entry(
                orchestration_state_path,
                orchestration_checkpoint,
                project_root=root,
                target_ref=proposed["target_obligation_id"],
                parent_obligation_ref=str(
                    selected_parent.get("parent_id", "ROOT"),
                ),
                parent_complexity=max(
                    5, len(str(selected_parent.get("statement", "")).split()),
                ),
                event_type=event_type,
                event_id=event_id,
                elaborated_theorem_id=(
                    orchestration_checkpoint.elaborated_theorem_id
                ),
                proposition_hash=orchestration_checkpoint.proposition_hash,
                target_statement=str(selected_parent.get("statement", "")),
                target_evidence={
                    "last_evidence": str(
                        selected_parent.get("last_evidence", "")
                    ),
                    "formal_status": str(
                        selected_parent.get("formal_status", "")
                    ),
                    "ledger_version": int(ledger_data.get("version", 0)),
                },
            )
            if orchestration_checkpoint.adapter_status == "INTEGRATION_BLOCKED":
                live_status.emit(
                    phase="strategy_configuration_required",
                    role="cursor_strategy",
                    state="idle",
                    active_obligation_id=(
                        orchestration_checkpoint.target_obligation_id
                    ),
                    source="proof_supervisor",
                    force=True,
                )
                return {
                    "iteration": iteration,
                    "research_outcome": "BLOCKED",
                    "orchestration_state": "INTEGRATION_BLOCKED",
                    "transition_reason": (
                        orchestration_checkpoint.blocked_reason
                    ),
                    "failure_class": "",
                    "error": "",
                    "inference_started": False,
                }
            if (
                orchestration_checkpoint.proof_state == ProofState.PROOF_SEARCH
                and orchestration_checkpoint.research_contract_id
            ):
                if route_contract_to_subgoal_generation(
                    orchestration_checkpoint,
                ):
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        orchestration_checkpoint,
                    )
                    print(
                        "[proof-live] stage=research-contract "
                        f"target={orchestration_checkpoint.target_obligation_id} "
                        f"plan={orchestration_checkpoint.selected_strategy_plan_id} "
                        f"contract={orchestration_checkpoint.research_contract_id} "
                        "accepted=true next=DECOMPOSER "
                        "reason=elaborated-subgoal-required",
                        flush=True,
                    )
                else:
                    # OProver residency is permitted only after both an accepted
                    # contract and a concrete typed proof-plan node exist.
                    orchestration_checkpoint.adapter_blocked(
                        "PROOF_ADVISOR_UNAVAILABLE:"
                        "RESIDENCY_PROCESS_MANAGER_REQUIRED",
                        status="INTEGRATION_BLOCKED",
                    )
                    save_orchestration_checkpoint(
                        orchestration_state_path,
                        orchestration_checkpoint,
                    )
                    live_status.emit(
                        phase="proof_advisor_unavailable",
                        role="oprover_advisor",
                        state="idle",
                        active_obligation_id=(
                            orchestration_checkpoint.target_obligation_id
                        ),
                        source="proof_supervisor",
                        force=True,
                    )
                    return {
                        "iteration": iteration,
                        "research_outcome": "BLOCKED",
                        "orchestration_state": "INTEGRATION_BLOCKED",
                        "transition_reason": (
                            orchestration_checkpoint.blocked_reason
                        ),
                        "failure_class": "",
                        "error": "",
                        "inference_started": False,
                    }
        experiment_id = (
            f"ar_{int(time.time())}_{iteration}_"
            f"{hashlib.sha256(candidate_path.read_bytes()).hexdigest()[:8]}"
        )
        report_path = reports_dir / f"{experiment_id}.json"
        transcript_path = reports_dir / f"{experiment_id}.log"
        print(
            f"[autoresearch] phase=gan-experiment id={experiment_id}",
            flush=True,
        )
        live_status.emit(
            phase="proof_run_queued",
            role="orchestrator",
            state="queued",
            active_obligation_id=proposed["target_obligation_id"],
            source="proof_supervisor",
            force=True,
        )
        run_id, gan_output = run_gan_experiment(
            repo=root,
            candidate_path=candidate_path,
            state_path=state_path,
            timeout_s=args.experiment_timeout_s,
            max_retained_tokens=args.max_retained_tokens,
            live_status_path=Path(args.live_status_file).expanduser(),
            supervisor_pid=os.getpid(),
            iteration=iteration,
            orchestration_state_path=orchestration_state_path,
            candidate_sha256=candidate_sha256,
        )
        gan_completed = True
        transcript_path.write_text(gan_output)
        report = _json_request(
            f"http://127.0.0.1:8090/v1/network/benchmarks/{run_id}",
        )
        if report.get("status") != "completed":
            failure_reason = extract_gan_failure_reason(gan_output)
            raise RuntimeError(
                f"GAN benchmark is not completed: {report.get('status')}"
                + (
                    f"; {failure_reason}"
                    if failure_reason else ""
                ),
            )
        transcript_provenance = extract_report_provenance(gan_output)
        if transcript_provenance is not None:
            report["provenance"] = transcript_provenance
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        candidate_module = _load_candidate(candidate_path)
        candidate_module.CANDIDATE_SHA256 = candidate_sha256
        result = evaluate(report, candidate_module)
        evaluation_provenance = result["evaluation_provenance"]
        verdict = parse_research_verdict(
            gan_output,
            proposed["candidate_id"],
        )
        result.update({
            "research_outcome": verdict["outcome"],
            "research_evidence": verdict["evidence"],
            "new_frontier": verdict["new_frontier"],
            "created_obligation_ids": verdict["created_obligation_ids"],
            "transcript_path": str(transcript_path),
            "hypothesis_novel": hypothesis_novel,
            "invalidation_kind": verdict["invalidation_kind"],
            "backjump_target_id": verdict["backjump_target_id"],
            "no_go_lesson_hashes": verdict["no_go_lesson_hashes"],
        })
        keep = should_keep(result, baseline)
        latest_orchestration = load_orchestration_checkpoint(
            orchestration_state_path,
        )
        recoverable_role_pending = is_resumable_checkpoint(
            latest_orchestration,
        )
        print(
            f"[autoresearch] phase=evaluate accepted={result['accepted']} "
            f"unresolved={result['proof_obligations_unresolved']} "
            f"outcome={verdict['outcome']} "
            f"prefill_s={result['metric_cold_critic_prefill_s']:.3f} "
            f"critic_reused={str(evaluation_provenance['critic_reused']).lower()} "
            f"critic_source={evaluation_provenance['critic_source_run_id']} "
            f"decision={'keep' if keep else 'revert'}",
            flush=True,
        )
        live_status.set_context(run_id=run_id)
        live_status.emit(
            phase="iteration_completed",
            role="supervisor",
            state="completed",
            progress_current=1,
            progress_total=1,
            progress_unit="iteration",
            active_obligation_id=proposed["target_obligation_id"],
            source="proof_supervisor",
            force=True,
        )
        row = {
            "timestamp": time.time(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "candidate_id": proposed["candidate_id"],
            "target_obligation_id": proposed["target_obligation_id"],
            "constraints_pass": result["accepted"],
            "accepted": result["accepted"],
            "kept": keep,
            "metric_cold_critic_prefill_s": result[
                "metric_cold_critic_prefill_s"
            ],
            "baseline_metric_s": (
                baseline["metric_cold_critic_prefill_s"] if baseline else ""
            ),
            "proof_obligations_total": result["proof_obligations_total"],
            "proof_obligations_covered": result["proof_obligations_covered"],
            "proof_obligations_unresolved": result[
                "proof_obligations_unresolved"
            ],
            "compute_chunk_tokens": result["compute_chunk_tokens"],
            "candidate_sha256": candidate_sha256,
            "report_path": str(report_path),
            "hypothesis_sha256": hypothesis_sha256,
            "research_outcome": verdict["outcome"],
            "research_evidence": verdict["evidence"],
            "new_frontier": verdict["new_frontier"],
            "created_obligation_ids": json.dumps(
                verdict["created_obligation_ids"],
            ),
            "strategy_mode": strategy_mode,
            "invalidation_kind": verdict["invalidation_kind"],
            "backjump_target_id": verdict["backjump_target_id"],
            "no_go_lesson_hashes": json.dumps(
                verdict["no_go_lesson_hashes"],
            ),
            "transcript_path": str(transcript_path),
            "orchestration_state": (
                latest_orchestration.state
                if latest_orchestration is not None else ""
            ),
            "resumed_role": (
                latest_orchestration.current_role
                if resume_downstream else ""
            ),
            "resume_origin": (
                latest_orchestration.resume_origin
                if resume_downstream else ""
            ),
            "transition_reason": (
                latest_orchestration.last_transition_reason
                if latest_orchestration is not None else ""
            ),
            "retry_count": (
                latest_orchestration.retry_counters.get(
                    latest_orchestration.state,
                    0,
                )
                if latest_orchestration is not None else 0
            ),
            "strategy_reused": bool(
                resume_downstream
                or (
                    latest_orchestration is not None
                    and latest_orchestration.strategy_reused
                )
            ),
            "generator_reused": bool(
                evaluation_provenance.get("critic_reused"),
            ),
            "critic_reused": bool(
                evaluation_provenance.get("critic_reused"),
            ),
            "critic_source_run_id": evaluation_provenance.get(
                "critic_source_run_id",
                "",
            ),
            "critic_artifact_sha256": evaluation_provenance.get(
                "critic_artifact_sha256",
                "",
            ),
            "newly_executed_stages": json.dumps(
                evaluation_provenance.get("newly_executed_stages", []),
            ),
            "failure_class": "",
            "supervisor_outcome": (
                "CONTINUE" if recoverable_role_pending
                else "ITERATION_COMPLETE"
            ),
            "continuation_reason": (
                latest_orchestration.last_transition_reason
                if recoverable_role_pending else ""
            ),
        }
        append_result(results_path, row)
        if not keep and not recoverable_role_pending:
            candidate_path.write_bytes(previous_candidate)
            print(
                "[autoresearch] phase=candidate-reverted "
                "completed-run-preserved",
                flush=True,
            )
        elif recoverable_role_pending and not keep:
            print(
                "[autoresearch] phase=candidate-preserved "
                f"resume_state={latest_orchestration.state} "
                "strategy_reused=true",
                flush=True,
            )
        else:
            print("[autoresearch] phase=kept", flush=True)
        return row
    except CandidateNoveltyStagnation as exc:
        candidate_path.write_bytes(previous_candidate)
        _restore(state_path, previous_state)
        _restore(ledger_path, previous_ledger)
        blocked_checkpoint = load_orchestration_checkpoint(
            orchestration_state_path,
        )
        if blocked_checkpoint is not None:
            if blocked_checkpoint.proof_state != ProofState.BLOCKED:
                blocked_checkpoint.transition(
                    ProofState.BLOCKED,
                    "duplicate-strategy-nonfatal-use-current-candidate",
                    strategy_reused=True,
                )
            blocked_checkpoint.blocked_reason = (
                "Strategy proposals were duplicates; reuse the current "
                "candidate and unresolved role."
            )
            save_orchestration_checkpoint(
                orchestration_state_path,
                blocked_checkpoint,
            )
        live_status.emit(
            phase="iteration_skipped",
            role="supervisor",
            state="completed",
            active_obligation_id=active_obligation_id,
            source="proof_supervisor",
            force=True,
        )
        row = {
            "timestamp": time.time(),
            "candidate_id": exc.candidate.get("candidate_id", ""),
            "target_obligation_id": exc.candidate.get(
                "target_obligation_id",
                "",
            ),
            "constraints_pass": False,
            "accepted": False,
            "kept": False,
            "baseline_metric_s": (
                baseline["metric_cold_critic_prefill_s"] if baseline else ""
            ),
            "compute_chunk_tokens": exc.candidate.get(
                "prefill_compute_chunk_tokens",
                "",
            ),
            "candidate_sha256": exc.candidate_sha256,
            "hypothesis_sha256": exc.hypothesis_sha256,
            "research_outcome": "STAGNATED",
            "research_evidence": (
                "Candidate rejected before experiment; all available bounded "
                "strategy proposals were duplicates."
            ),
            "strategy_mode": exc.strategy_mode,
            "invalidation_kind": "STRATEGY_STAGNATION",
            "error": f"{type(exc).__name__}: {exc}",
        }
        append_result(results_path, row)
        print(
            "[autoresearch] phase=iteration-skipped "
            "outcome=STAGNATED reason=duplicate-candidate "
            "candidate-preserved=true",
            flush=True,
        )
        return row
    except Exception as exc:
        failure_class = failure_class_for_exception(exc)
        live_status.emit(
            phase="iteration_failed",
            role="supervisor",
            state="failed",
            active_obligation_id=active_obligation_id,
            source="proof_supervisor",
            force=True,
        )
        print(
            f"[autoresearch] phase=failed error={type(exc).__name__}: {exc}",
            flush=True,
        )
        failed_orchestration = load_orchestration_checkpoint(
            orchestration_state_path,
        )
        if (
            isinstance(exc, ResumeValidationError)
            and failed_orchestration is not None
        ):
            previous_resume_state = failed_orchestration.state
            legacy_routes = {
                ProofState.NEEDS_STRATEGY.value,
                ProofState.GENERATOR.value,
                ProofState.CRITIC.value,
            }
            route_state = (
                ProofState.STRATEGY_TOURNAMENT.value
                if exc.route_state in legacy_routes
                else exc.route_state
            )
            failed_orchestration.state = route_state
            failed_orchestration.current_role = route_state.lower()
            failed_orchestration.resume_origin = previous_resume_state
            failed_orchestration.last_transition_reason = (
                f"resume-validation-failed:{exc}"
            )
            save_orchestration_checkpoint(
                orchestration_state_path,
                failed_orchestration,
            )
        failed_role_resumable = is_resumable_checkpoint(
            failed_orchestration,
            candidate_sha256=candidate_sha256,
        )
        if not failed_role_resumable:
            candidate_path.write_bytes(previous_candidate)
        if not gan_completed:
            _restore(state_path, previous_state)
            _restore(ledger_path, previous_ledger)
        if not gan_completed:
            raise
        row = {
            "timestamp": time.time(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "candidate_id": proposed.get("candidate_id", ""),
            "target_obligation_id": proposed.get("target_obligation_id", ""),
            "constraints_pass": False,
            "accepted": False,
            "kept": False,
            "baseline_metric_s": (
                baseline["metric_cold_critic_prefill_s"] if baseline else ""
            ),
            "compute_chunk_tokens": proposed.get(
                "prefill_compute_chunk_tokens",
                "",
            ),
            "candidate_sha256": candidate_sha256,
            "report_path": str(report_path),
            "hypothesis_sha256": hypothesis_sha256,
            "research_outcome": "EVALUATION_FAILED",
            "strategy_mode": strategy_mode,
            "transcript_path": str(transcript_path),
            "error": f"{type(exc).__name__}: {exc}",
            "failure_class": failure_class,
            "supervisor_outcome": (
                "CONTINUE" if failed_role_resumable else "ITERATION_COMPLETE"
            ),
            "continuation_reason": (
                failed_orchestration.last_transition_reason
                if failed_role_resumable else ""
            ),
        }
        append_result(results_path, row)
        print(
            f"[autoresearch] phase=completed-run-preserved run={run_id}",
            flush=True,
        )
        return row


def run_supervisor_iterations(args) -> int:
    """Run iterations while preserving infrastructure circuit-breaker policy."""
    last_failure_fingerprint = ""
    consecutive_infrastructure_failures = 0
    last_continuation_fingerprint = ""
    repeated_continuations = 0
    blocked_logger = BlockedIdleLogger()
    last_blocked_heartbeat_at = 0.0
    iteration = 0
    configured_iterations = getattr(args, "iterations", None)
    stop_file = Path(
        getattr(
            args,
            "stop_file",
            Path.home() / ".kakeya/autoresearch/stop_supervisor",
        ),
    ).expanduser()
    while configured_iterations is None or iteration < configured_iterations:
        if stop_file.exists():
            blocked_logger.transition(
                next_state="EXIT",
                cause="explicit-operator-stop",
            )
            print(
                "[autoresearch] phase=operator-stop "
                f"file={stop_file}",
                flush=True,
            )
            return 0
        configured_orchestration_path = getattr(
            args,
            "orchestration_state_file",
            "",
        )
        orchestration_path = (
            Path(configured_orchestration_path).expanduser()
            if configured_orchestration_path else None
        )
        checkpoint = (
            load_orchestration_checkpoint(orchestration_path)
            if orchestration_path is not None else None
        )
        event_path = Path(
            getattr(
                args,
                "operator_event_file",
                (
                    orchestration_path.with_name("operator_event.json")
                    if orchestration_path is not None
                    else Path.home() / ".kakeya/autoresearch/operator_event.json"
                ),
            ),
        ).expanduser()
        if (
            checkpoint is not None
            and checkpoint.proof_state == ProofState.BLOCKED
            and event_path.exists()
        ):
            blocked_logger.observe(checkpoint)
            raw_event = json.loads(event_path.read_text(encoding="utf-8"))
            event = BlockedExitEvent(**raw_event)
            before_state = checkpoint.state
            apply_blocked_exit_event(checkpoint, event)
            save_orchestration_checkpoint(orchestration_path, checkpoint)
            append_blocked_event_journal(
                orchestration_path.with_name(
                    "proof_orchestration.journal.jsonl",
                ),
                event,
                before_state=before_state,
                after_state=checkpoint.state,
            )
            event_path.unlink()
            blocked_logger.transition(
                next_state=checkpoint.state,
                cause=event.event_type,
                event_id=event.event_id,
            )
        recovered_event = recover_contract_subgoal_duplicate_block(checkpoint)
        if recovered_event is not None and orchestration_path is not None:
            save_orchestration_checkpoint(orchestration_path, checkpoint)
            append_blocked_event_journal(
                orchestration_path.with_name(
                    "proof_orchestration.journal.jsonl",
                ),
                recovered_event,
                before_state=ProofState.BLOCKED.value,
                after_state=checkpoint.state,
            )
            blocked_logger.transition(
                next_state=checkpoint.state,
                cause=recovered_event.event_type,
                event_id=recovered_event.event_id,
            )
            print(
                "[proof-live] stage=backjump "
                "from=BLOCKED to=DECOMPOSER "
                "reason=target-bound-candidate-novelty-bypass",
                flush=True,
            )
        if (
            checkpoint is not None
            and route_contract_to_subgoal_generation(checkpoint)
        ):
            save_orchestration_checkpoint(orchestration_path, checkpoint)
            blocked_logger.transition(
                next_state=checkpoint.state,
                cause="research-contract-subgoal-required",
            )
            print(
                "[proof-live] stage=subgoal-generation "
                f"target={checkpoint.target_obligation_id} "
                f"proposition={checkpoint.target_statement} "
                f"plan={checkpoint.selected_strategy_plan_id} "
                f"contract={checkpoint.research_contract_id} "
                "next=DECOMPOSER reason=elaborated-subgoal-required",
                flush=True,
            )
        if (
            checkpoint is not None
            and checkpoint.proof_state == ProofState.DEFINITION_AUDITOR
            and checkpoint.adapter_status == "ADAPTER_BLOCKED"
            and "DEFINITION_AUDIT Artifact JSON" in checkpoint.blocked_reason
        ):
            checkpoint.clear_adapter_blocked(
                "typed-definition-auditor-supervisor-migration",
            )
            checkpoint.recovery_events.append({
                "event_type": "LEGACY_DEFINITION_OUTPUT_AUDIT_ONLY",
                "event_id": "definition-auditor-typed-transport-v1",
                "target_state": ProofState.DEFINITION_AUDITOR.value,
                "created_at": time.time(),
            })
            save_orchestration_checkpoint(orchestration_path, checkpoint)
        if (
            checkpoint is not None
            and checkpoint.proof_state == ProofState.MATHEMATICAL_STAGNATION
        ):
            if hasattr(args, "_live_status"):
                args._live_status.emit(
                    phase="mathematical_stagnation",
                    role="orchestrator",
                    state="idle",
                    active_obligation_id=checkpoint.target_obligation_id,
                    source="proof_supervisor",
                    force=True,
                )
            print(
                "[autoresearch] phase=mathematical-stagnation "
                f"reason={checkpoint.stagnation_reason}",
                flush=True,
            )
            return 0
        if (
            checkpoint is not None
            and checkpoint.proof_state != ProofState.BLOCKED
            and checkpoint.adapter_status == "INFRASTRUCTURE_BLOCKED"
        ):
            blocked_reason = checkpoint.blocked_reason
            checkpoint.clear_adapter_blocked(
                "quiescent-infrastructure-retry",
            )
            checkpoint.recovery_events.append({
                "event_type": "QUIESCENT_INFRASTRUCTURE_RETRY",
                "event_id": (
                    "quiescent-infrastructure-retry-"
                    + hashlib.sha256(blocked_reason.encode()).hexdigest()[:16]
                ),
                "target_state": checkpoint.state,
                "created_at": time.time(),
            })
            save_orchestration_checkpoint(orchestration_path, checkpoint)
        if checkpoint is not None and (
            checkpoint.proof_state == ProofState.BLOCKED
            or checkpoint.adapter_status in {
                "ADAPTER_BLOCKED",
                "INFRASTRUCTURE_BLOCKED",
                "INTEGRATION_BLOCKED",
            }
        ):
            now = time.monotonic()
            heartbeat_interval = getattr(
                args,
                "blocked_heartbeat_interval_s",
                BLOCKED_HEARTBEAT_INTERVAL_S,
            )
            if (
                hasattr(args, "_live_status")
                and (
                    last_blocked_heartbeat_at == 0.0
                    or now - last_blocked_heartbeat_at >= heartbeat_interval
                )
            ):
                args._live_status.emit(
                    phase="blocked_idle",
                    role="orchestrator",
                    state="idle",
                    active_obligation_id=checkpoint.target_obligation_id,
                    source="proof_supervisor",
                    force=True,
                )
                last_blocked_heartbeat_at = now
            blocked_logger.observe(checkpoint)
            if getattr(args, "blocked_policy", "wait") == "exit":
                blocked_logger.transition(
                    next_state="EXIT",
                    cause="blocked-policy-exit",
                )
                return 0
            iteration += 1
            if (
                configured_iterations is None
                or iteration < configured_iterations
            ):
                time.sleep(getattr(args, "blocked_poll_interval_s", 30.0))
            continue
        blocked_logger.transition(
            next_state=checkpoint.state if checkpoint is not None else "UNKNOWN",
            cause="checkpoint-state-change",
        )
        row = run_iteration(args, iteration)
        print(json.dumps(row, indent=2, sort_keys=True))
        continuation_checkpoint = (
            load_orchestration_checkpoint(orchestration_path)
            if orchestration_path is not None else None
        )
        if (
            checkpoint is not None
            and continuation_checkpoint is not None
            and row.get("supervisor_outcome") == "ITERATION_COMPLETE"
            and not row.get("failure_class")
        ):
            lean_source = ""
            gate_ref = continuation_checkpoint.validated_artifacts.get(
                "host_typed_ir_gate",
            )
            if gate_ref is not None:
                try:
                    gate_payload = json.loads(
                        Path(gate_ref.path).read_text(encoding="utf-8"),
                    )
                    lean_source = str(
                        gate_payload.get("compilation", {}).get(
                            "declaration_source", "",
                        ),
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    lean_source = ""
            progress = verified_progress_vector(
                definitions_added=(
                    continuation_checkpoint.definitions_added
                    - checkpoint.definitions_added
                ),
                existing_definitions_resolved=(
                    continuation_checkpoint.existing_definitions_resolved
                    - checkpoint.existing_definitions_resolved
                ),
                lemmas_proved=(
                    continuation_checkpoint.new_elaborated_lemmas
                    - checkpoint.new_elaborated_lemmas
                ),
                accepted_children=(
                    continuation_checkpoint.accepted_children
                    - checkpoint.accepted_children
                ),
                subgoals_closed=(
                    continuation_checkpoint.subgoals_closed
                    - checkpoint.subgoals_closed
                ),
                verified_counterexamples=(
                    continuation_checkpoint.verified_counterexamples
                    - checkpoint.verified_counterexamples
                ),
                lean_source=lean_source,
            )
            stagnant = record_semantic_iteration(
                continuation_checkpoint,
                progress,
                move_class=(
                    continuation_checkpoint.selected_move_id
                    or continuation_checkpoint.current_role
                ),
            )
            if (
                stagnant
                and continuation_checkpoint.proof_state
                == ProofState.DECOMPOSER
            ):
                continuation_checkpoint.transition(
                    ProofState.MATHEMATICAL_STAGNATION,
                    continuation_checkpoint.stagnation_reason,
                    strategy_reused=True,
                )
            save_orchestration_checkpoint(
                orchestration_path,
                continuation_checkpoint,
            )
        semantic_continuation = is_nonfatal_semantic_continuation(
            row,
            continuation_checkpoint,
        )
        if semantic_continuation:
            save_orchestration_checkpoint(
                orchestration_path,
                continuation_checkpoint,
            )
            print(
                "[autoresearch] phase=semantic-backjump-continuation "
                f"state={continuation_checkpoint.state} "
                f"reason={continuation_checkpoint.last_transition_reason} "
                "strategy_reused=true",
                flush=True,
            )
        fingerprint = (
            "" if semantic_continuation
            else infrastructure_failure_fingerprint(row)
        )
        if fingerprint:
            if fingerprint == last_failure_fingerprint:
                consecutive_infrastructure_failures += 1
            else:
                last_failure_fingerprint = fingerprint
                consecutive_infrastructure_failures = 1
            if (
                consecutive_infrastructure_failures
                >= args.max_consecutive_infrastructure_failures
            ):
                print(
                    "[autoresearch] phase=infrastructure-circuit-open "
                    f"consecutive={consecutive_infrastructure_failures} "
                    f"fingerprint={fingerprint[:12]} "
                    f"error={row.get('error', '')}",
                    flush=True,
                )
                return 2
        else:
            last_failure_fingerprint = ""
            consecutive_infrastructure_failures = 0
        iteration += 1
        if (
            semantic_continuation
            and (
                configured_iterations is None
                or iteration < configured_iterations
            )
        ):
            continuation_fingerprint = hashlib.sha256(
                (
                    continuation_checkpoint.state
                    + "\0"
                    + continuation_checkpoint.last_transition_reason
                    + "\0"
                    + continuation_checkpoint.candidate_sha256
                ).encode()
            ).hexdigest()
            if continuation_fingerprint == last_continuation_fingerprint:
                repeated_continuations += 1
            else:
                last_continuation_fingerprint = continuation_fingerprint
                repeated_continuations = 1
            base_backoff = getattr(args, "continuation_backoff_s", 1.0)
            max_backoff = getattr(args, "continuation_max_backoff_s", 30.0)
            backoff = min(
                max_backoff,
                base_backoff * (2 ** min(repeated_continuations - 1, 8)),
            )
            time.sleep(backoff)
        elif not semantic_continuation:
            last_continuation_fingerprint = ""
            repeated_continuations = 0
    blocked_logger.transition(
        next_state="EXIT",
        cause="supervisor-iterations-complete",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="stop after N iterations; omitted means run until explicit stop",
    )
    parser.add_argument(
        "--worker-address",
        default="169.254.27.104:53051",
    )
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--strategy-max-prefill-tokens",
        type=int,
        default=8448,
    )
    parser.add_argument(
        "--max-retained-tokens",
        type=int,
        default=2052,
    )
    parser.add_argument(
        "--strategy-stagnation-rounds",
        type=int,
        default=3,
    )
    parser.add_argument("--force-strategy", action="store_true")
    parser.add_argument(
        "--strategy-trigger-file",
        default=str(
            Path.home()
            / ".kakeya/autoresearch/request_strategy"
        ),
    )
    parser.add_argument(
        "--tokenizer-id",
        default=str(
            Path.home()
            / "kakeya-models/gemma-4-26B-A4B-it-mlx-4bit"
        ),
    )
    parser.add_argument(
        "--results",
        default=str(Path.home() / ".kakeya/autoresearch/prefill/results.tsv"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(Path.home() / ".kakeya/autoresearch/prefill/reports"),
    )
    parser.add_argument(
        "--state-file",
        default=str(Path.home() / ".kakeya/agent_gan_state.json"),
    )
    parser.add_argument(
        "--proof-ledger",
        default=str(Path.home() / ".kakeya/agent_gan_proof_ledger.json"),
    )
    parser.add_argument(
        "--live-status-file",
        default=str(Path.home() / ".kakeya/proof_live_status.json"),
    )
    parser.add_argument(
        "--orchestration-state-file",
        default=str(
            Path.home()
            / ".kakeya/autoresearch/proof_orchestration.json"
        ),
    )
    parser.add_argument("--experiment-timeout-s", type=float, default=7200)
    parser.add_argument(
        "--max-consecutive-infrastructure-failures",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--blocked-policy",
        choices=("wait", "exit"),
        default="wait",
    )
    parser.add_argument("--blocked-poll-interval-s", type=float, default=30.0)
    parser.add_argument(
        "--blocked-heartbeat-interval-s",
        type=float,
        default=BLOCKED_HEARTBEAT_INTERVAL_S,
    )
    parser.add_argument(
        "--operator-event-file",
        default=str(Path.home() / ".kakeya/autoresearch/operator_event.json"),
    )
    parser.add_argument(
        "--stop-file",
        default=str(Path.home() / ".kakeya/autoresearch/stop_supervisor"),
    )
    parser.add_argument("--continuation-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--continuation-max-backoff-s",
        type=float,
        default=30.0,
    )
    args = parser.parse_args()
    os.environ["KAKEYA_ORCHESTRATION_STATE_PATH"] = str(
        Path(args.orchestration_state_file).expanduser(),
    )
    args._live_status = AtomicLiveStatus(
        Path(args.live_status_file),
        supervisor_pid=os.getpid(),
        run_id=f"supervisor-{os.getpid()}",
    )
    if args.iterations is not None and args.iterations <= 0:
        raise SystemExit("iterations must be > 0")
    if args.continuation_backoff_s < 0:
        raise SystemExit("continuation-backoff-s must be >= 0")
    if args.continuation_max_backoff_s < args.continuation_backoff_s:
        raise SystemExit(
            "continuation-max-backoff-s must be >= continuation-backoff-s",
        )
    if args.strategy_max_prefill_tokens <= 0:
        raise SystemExit("strategy-max-prefill-tokens must be > 0")
    if args.max_retained_tokens <= 0:
        raise SystemExit("max-retained-tokens must be > 0")
    if args.strategy_stagnation_rounds <= 0:
        raise SystemExit("strategy-stagnation-rounds must be > 0")
    if args.max_consecutive_infrastructure_failures <= 0:
        raise SystemExit(
            "max-consecutive-infrastructure-failures must be > 0",
        )
    lean_warmup = warm_lean_environment(
        Path(__file__).resolve().parents[2],
    )
    print(
        "[autoresearch] phase=lean-warmup "
        f"status={lean_warmup.status} "
        f"elapsed_s={lean_warmup.elapsed_s:.2f} "
        f"error={lean_warmup.error or '(none)'}",
        flush=True,
    )
    if not lean_warmup.ok:
        raise SystemExit(lean_warmup.error)
    try:
        return run_supervisor_iterations(args)
    finally:
        args._live_status.emit(
            phase="supervisor_exit",
            role="supervisor",
            state="idle",
            source="proof_supervisor",
            force=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
