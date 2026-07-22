#!/usr/bin/env python3
"""Real Karpathy-style AutoResearch supervisor around one-shot GAN experiments."""
from __future__ import annotations

import argparse
import ast
import csv
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

from autoresearch.prefill.prepare import _load_candidate, evaluate
from autoresearch.prefill.lean_gate import warm_lean_environment
from autoresearch.prefill.semantic_decompose import (
    SemanticResponseIncomplete,
    SemanticUnitTooLarge,
    admit_token_ids,
    build_proof_step_interface,
    downstream_output_cap,
)


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
        repaired = re.sub(
            r'\\(?!(?:["\\/]|u[0-9a-fA-F]{4}))',
            r"\\\\",
            block.strip(),
        )
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
    return (
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


class StrategyPrefillHeartbeat:
    def __init__(
        self,
        dashboard: str = "http://127.0.0.1:8090",
        interval_s: float = 10.0,
    ) -> None:
        self.dashboard = dashboard.rstrip("/")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: dict = {}
        self._last: tuple | None = None

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
            with StrategyPrefillHeartbeat():
                session.append(ids)
            print(
                f"[autoresearch] Strategy Prefill complete: {len(ids)} tokens",
                flush=True,
            )
            while len(generated) < strategy_output_cap:
                before = len(generated)
                generated.extend(
                    int(token)
                    for token in session.generate(
                        max_tokens=min(
                            64,
                            strategy_output_cap - len(generated),
                        ),
                    )
                )
                print(
                    f"[autoresearch] Strategy Decode: {len(generated)} tokens "
                    f"stop_reason={session.last_stop_reason}",
                    flush=True,
                )
                if session.last_stop_reason != 1:
                    break
                if len(generated) == before:
                    raise RuntimeError("strategy agent made no progress")
            if session.last_stop_reason != 2:
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
    candidate = _extract_json(strategy_output)
    parse_mode = candidate.pop("strategy_parse_mode", "unknown")
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
    error = " ".join(str(row.get("error", "")).lower().split())
    if not error:
        return ""
    return hashlib.sha256(error.encode()).hexdigest()


def build_host_candidate(current: dict, ledger: dict) -> dict:
    target_id = _select_repair_target(current, ledger)
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
    try:
        print(
            f"[autoresearch] iteration={iteration} "
            f"phase=runtime-health-check candidate={current['candidate_id']}",
            flush=True,
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
        trigger_reason = strategy_trigger_reason(
            results,
            stagnation_rounds=args.strategy_stagnation_rounds,
            force=args.force_strategy and iteration == 0,
            trigger_file=trigger_file,
        )
        if baseline is None and iteration == 0 and not trigger_reason:
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
                )
                if proposed["target_obligation_id"] not in _pending_leaf_ids(
                    ledger_data,
                ):
                    raise ValueError(
                        "strategy agent targeted a non-leaf proof obligation",
                    )
                if trigger_reason == "manual-trigger-file":
                    trigger_file.unlink(missing_ok=True)
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
        candidate_path.write_text(render_candidate(proposed))
        print(
            f"[autoresearch] phase=candidate-written "
            f"candidate={proposed['candidate_id']} "
            f"target={proposed['target_obligation_id']} "
            f"mode={strategy_mode}",
            flush=True,
        )
        validate_candidate(proposed)
        hypothesis_sha256 = hashlib.sha256(
            proposed["hypothesis"].strip().lower().encode(),
        ).hexdigest()
        seen_hypotheses = {
            row.get("hypothesis_sha256", "")
            for row in results
            if row.get("hypothesis_sha256")
        }
        hypothesis_novel = hypothesis_sha256 not in seen_hypotheses
        if strategy_mode == "gemma" and not hypothesis_novel:
            raise ValueError("strategy agent repeated a previous hypothesis")
        candidate_sha256 = hashlib.sha256(
            candidate_path.read_bytes(),
        ).hexdigest()
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
        run_id, gan_output = run_gan_experiment(
            repo=root,
            candidate_path=candidate_path,
            state_path=state_path,
            timeout_s=args.experiment_timeout_s,
            max_retained_tokens=args.max_retained_tokens,
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
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        candidate_module = _load_candidate(candidate_path)
        result = evaluate(report, candidate_module)
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
        print(
            f"[autoresearch] phase=evaluate accepted={result['accepted']} "
            f"unresolved={result['proof_obligations_unresolved']} "
            f"outcome={verdict['outcome']} "
            f"prefill_s={result['metric_cold_critic_prefill_s']:.3f} "
            f"decision={'keep' if keep else 'revert'}",
            flush=True,
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
        }
        append_result(results_path, row)
        if not keep:
            candidate_path.write_bytes(previous_candidate)
            print(
                "[autoresearch] phase=candidate-reverted "
                "completed-run-preserved",
                flush=True,
            )
        else:
            print("[autoresearch] phase=kept", flush=True)
        return row
    except Exception as exc:
        print(
            f"[autoresearch] phase=failed error={type(exc).__name__}: {exc}",
            flush=True,
        )
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
        }
        append_result(results_path, row)
        print(
            f"[autoresearch] phase=completed-run-preserved run={run_id}",
            flush=True,
        )
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
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
    parser.add_argument("--experiment-timeout-s", type=float, default=7200)
    parser.add_argument(
        "--max-consecutive-infrastructure-failures",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("iterations must be > 0")
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
    last_failure_fingerprint = ""
    consecutive_infrastructure_failures = 0
    for iteration in range(args.iterations):
        row = run_iteration(args, iteration)
        print(json.dumps(row, indent=2, sort_keys=True))
        fingerprint = infrastructure_failure_fingerprint(row)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
