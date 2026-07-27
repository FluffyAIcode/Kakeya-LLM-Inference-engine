"""Content-addressed, crash-consistent active-target namespaces."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


MIGRATION_EVENT = "strategy_intent_target_context_v1"
CONTEXT_SCHEMA_VERSION = 1


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class TargetBinding:
    target_obligation_id: str
    parent_statement_hash: str
    strategy_plan_hash: str
    environment_hash: str
    context_hash: str


@dataclass(frozen=True)
class TargetContext:
    binding: TargetBinding
    statement: str
    evidence: Mapping[str, Any]
    gap_ids: tuple[str, ...] = ()
    definition_ids: tuple[str, ...] = ()
    definition_query_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    theorem_card_ids: tuple[str, ...] = ()
    artifact_hashes: tuple[str, ...] = ()
    created_at: float = 0.0
    schema_version: int = CONTEXT_SCHEMA_VERSION


def make_binding(
    *,
    target_obligation_id: str,
    parent_statement_hash: str,
    strategy_plan_hash: str,
    environment_hash: str,
) -> TargetBinding:
    body = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "target_obligation_id": str(target_obligation_id),
        "parent_statement_hash": str(parent_statement_hash),
        "strategy_plan_hash": str(strategy_plan_hash),
        "environment_hash": str(environment_hash),
    }
    return TargetBinding(
        target_obligation_id=body["target_obligation_id"],
        parent_statement_hash=body["parent_statement_hash"],
        strategy_plan_hash=body["strategy_plan_hash"],
        environment_hash=body["environment_hash"],
        context_hash=_digest(body),
    )


def context_store_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).with_name("proof_target_contexts.json")


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "migration_event": MIGRATION_EVENT,
            "active_context_hash": "",
            "contexts": {},
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("contexts"), dict):
        raise ValueError("TARGET_CONTEXT_STORE_INVALID")
    return raw


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def activate_target_context(
    checkpoint_path: Path,
    checkpoint: Any,
    *,
    target_obligation_id: str,
    statement: str,
    environment_hash: str,
    strategy_plan_hash: str = "",
    evidence: Mapping[str, Any] | None = None,
    gap_ids: Iterable[str] = (),
    definition_ids: Iterable[str] = (),
    definition_query_ids: Iterable[str] = (),
    candidate_ids: Iterable[str] = (),
    theorem_card_ids: Iterable[str] = (),
) -> tuple[TargetContext, bool]:
    """Switch active target in one durable namespace transaction.

    Prior pointers remain only in immutable audit context snapshots. The caller
    persists the checkpoint after this store commit; replay is idempotent.
    """
    target = str(target_obligation_id).strip()
    statement = str(statement).strip()
    if not target or not statement or not environment_hash:
        raise ValueError("TARGET_CONTEXT_REQUIRES_TARGET_STATEMENT_ENVIRONMENT")
    statement_hash = hashlib.sha256(statement.encode()).hexdigest()
    binding = make_binding(
        target_obligation_id=target,
        parent_statement_hash=statement_hash,
        strategy_plan_hash=str(strategy_plan_hash),
        environment_hash=str(environment_hash),
    )
    path = context_store_path(checkpoint_path)
    store = _read_store(path)
    previous_hash = str(store.get("active_context_hash", ""))
    unchanged = (
        previous_hash == binding.context_hash
        and str(getattr(checkpoint, "target_context_hash", ""))
        == binding.context_hash
    )
    if unchanged:
        raw = store["contexts"][binding.context_hash]
        return _context_from_dict(raw), False

    now = time.time()
    if previous_hash and previous_hash in store["contexts"]:
        store["contexts"][previous_hash]["active"] = False
        store["contexts"][previous_hash]["audit_only"] = True
        store["contexts"][previous_hash]["deactivated_at"] = now

    old_refs = dict(getattr(checkpoint, "validated_artifacts", {}))
    invalidated = getattr(checkpoint, "invalidated_artifacts", {})
    for role, ref in old_refs.items():
        invalidated[ref.sha256] = {
            **asdict(ref),
            "audit_only": True,
            "reason_codes": ["TARGET_CONTEXT_SWITCH"],
            "prior_context_hash": str(getattr(checkpoint, "target_context_hash", "")),
        }
    checkpoint.validated_artifacts.clear()
    old_advisory = dict(getattr(checkpoint, "advisory_artifacts", {}))
    for digest, value in old_advisory.items():
        invalidated[str(digest)] = {
            **dict(value),
            "audit_only": True,
            "reason_codes": ["TARGET_CONTEXT_SWITCH"],
            "prior_context_hash": str(getattr(checkpoint, "target_context_hash", "")),
        }
    checkpoint.advisory_artifacts.clear()

    # Every active target-local pointer is reset together.
    for name, empty in {
        "candidate_sha256": "",
        "strategy_sha256": "",
        "strategy_plan_ids": [],
        "strategy_plan_hashes": [],
        "feasible_strategy_plan_ids": [],
        "pareto_plan_ids": [],
        "selected_strategy_plan_id": "",
        "selected_strategy_plan_hash": "",
        "strategy_tournament_hash": "",
        "strategy_event_id": "",
        "strategy_event_type": "",
        "strategy_run_status": "CONFIGURATION_REQUIRED",
        "strategy_agent_id": "",
        "strategy_run_id": "",
        "strategy_prompt_hash": "",
        "strategy_evidence_hash": "",
        "strategy_memo_hash": "",
        "strategy_intent_hash": "",
        "strategy_intent_run_id": "",
        "strategy_intent_status": "",
        "strategy_selection_provenance": {},
        "research_contract_id": "",
        "research_contract_hash": "",
        "research_contract_rejection_codes": [],
        "candidate_set_hash": "",
        "candidate_hashes": [],
        "candidate_count": 0,
        "ranking_hash": "",
        "ranked_candidate_ids": [],
        "selected_move_id": "",
        "evidence_gap_graph_hash": "",
        "proof_plan_hash": "",
        "proof_plan_id": "",
        "executable_plan_node_id": "",
        "plan_score_explanation": {},
        "theorem_card_ids": [],
        "theorem_card_index_hash": "",
        "target_gap_ids": [],
        "current_definition_gap_id": "",
        "definition_candidate_count": 0,
        "definition_query_hash": "",
        "definition_audit_outcome": "",
        "definition_audit_fingerprint": "",
        "counterexample_objective": {},
        "premise_outcome_type": "",
        "premise_outcome_fingerprint": "",
        "premise_outcome_owner": "",
        "premise_decision": "",
        "premise_confidence": 0.0,
        "premise_evidence": {},
        "premise_backjump_target": "",
        "premise_invalidated_artifacts": [],
        "consumed_premise_fingerprints": [],
        "definition_source_statuses": {},
        "definition_property_statuses": {},
        "definition_branch_hashes": [],
        "scratchpad_math_fingerprints": [],
    }.items():
        setattr(checkpoint, name, empty)

    context = TargetContext(
        binding=binding,
        statement=statement,
        evidence=dict(evidence or {}),
        gap_ids=tuple(sorted(set(map(str, gap_ids)))),
        definition_ids=tuple(sorted(set(map(str, definition_ids)))),
        definition_query_ids=tuple(sorted(set(map(str, definition_query_ids)))),
        candidate_ids=tuple(sorted(set(map(str, candidate_ids)))),
        theorem_card_ids=tuple(sorted(set(map(str, theorem_card_ids)))),
        created_at=now,
    )
    store["contexts"][binding.context_hash] = {
        **asdict(context),
        "active": True,
        "audit_only": False,
    }
    store["active_context_hash"] = binding.context_hash
    store["migration_event"] = MIGRATION_EVENT
    _atomic_write(path, store)

    checkpoint.target_obligation_id = target
    checkpoint.parent_statement_sha256 = statement_hash
    checkpoint.target_context_hash = binding.context_hash
    checkpoint.target_environment_hash = str(environment_hash)
    checkpoint.target_strategy_plan_hash = str(strategy_plan_hash)
    checkpoint.target_statement = statement
    checkpoint.target_evidence = dict(evidence or {})
    checkpoint.migration_event = MIGRATION_EVENT
    checkpoint.recovery_events.append({
        "event_type": "TARGET_CONTEXT_ACTIVATED",
        "event_id": binding.context_hash,
        "prior_context_hash": previous_hash,
        "target_obligation_id": target,
        "parent_statement_hash": statement_hash,
        "invalidated_artifact_hashes": sorted(ref.sha256 for ref in old_refs.values()),
        "created_at": now,
    })
    return context, True


def update_active_context(
    checkpoint_path: Path,
    checkpoint: Any,
    **updates: Iterable[str] | Mapping[str, Any] | str,
) -> TargetContext:
    path = context_store_path(checkpoint_path)
    store = _read_store(path)
    context_hash = str(getattr(checkpoint, "target_context_hash", ""))
    if not context_hash or store.get("active_context_hash") != context_hash:
        raise ValueError("TARGET_CONTEXT_NOT_ACTIVE")
    raw = dict(store["contexts"][context_hash])
    allowed = {
        "evidence", "gap_ids", "definition_ids", "definition_query_ids",
        "candidate_ids", "theorem_card_ids", "artifact_hashes",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError("TARGET_CONTEXT_UNKNOWN_FIELDS:" + ",".join(sorted(unknown)))
    for key, value in updates.items():
        raw[key] = dict(value) if key == "evidence" else sorted(set(map(str, value)))
    store["contexts"][context_hash] = raw
    _atomic_write(path, store)
    return _context_from_dict(raw)


def require_binding(
    checkpoint: Any,
    *,
    target_obligation_id: str,
    parent_statement_hash: str,
    context_hash: str,
    strategy_plan_hash: str = "",
    environment_hash: str = "",
) -> None:
    expected = {
        "target_obligation_id": str(getattr(checkpoint, "target_obligation_id", "")),
        "parent_statement_hash": str(getattr(checkpoint, "parent_statement_sha256", "")),
        "context_hash": str(getattr(checkpoint, "target_context_hash", "")),
        "strategy_plan_hash": str(getattr(checkpoint, "target_strategy_plan_hash", "")),
        "environment_hash": str(getattr(checkpoint, "target_environment_hash", "")),
    }
    actual = {
        "target_obligation_id": str(target_obligation_id),
        "parent_statement_hash": str(parent_statement_hash),
        "context_hash": str(context_hash),
        "strategy_plan_hash": str(strategy_plan_hash),
        "environment_hash": str(environment_hash),
    }
    for name, wanted in expected.items():
        if wanted and actual[name] != wanted:
            raise ValueError(f"TARGET_CONTEXT_MISMATCH:{name}")


def mathematical_state_fingerprint(
    checkpoint: Any,
    *,
    failure_code: str = "",
    move_ids: Iterable[str] = (),
) -> str:
    """Fingerprint mathematical state; intentionally excludes run/viewpoint."""
    return _digest({
        "target_context_hash": str(getattr(checkpoint, "target_context_hash", "")),
        "target_obligation_id": str(getattr(checkpoint, "target_obligation_id", "")),
        "parent_statement_hash": str(getattr(checkpoint, "parent_statement_sha256", "")),
        "strategy_plan_ids": sorted(getattr(checkpoint, "strategy_plan_ids", ())),
        "gap_set": sorted({
            str(getattr(checkpoint, "current_definition_gap_id", "")),
            *map(str, getattr(checkpoint, "target_gap_ids", ())),
        } - {""}),
        "evidence": getattr(checkpoint, "target_evidence", {}),
        "move_set": sorted(set(map(str, move_ids))),
        "environment_hash": str(getattr(checkpoint, "target_environment_hash", "")),
        "failure": str(failure_code),
        "candidate_hashes": sorted(getattr(checkpoint, "candidate_hashes", ())),
    })


def _context_from_dict(raw: Mapping[str, Any]) -> TargetContext:
    clean = {key: value for key, value in raw.items() if key not in {
        "active", "audit_only", "deactivated_at",
    }}
    clean["binding"] = TargetBinding(**clean["binding"])
    for key in (
        "gap_ids", "definition_ids", "definition_query_ids", "candidate_ids",
        "theorem_card_ids", "artifact_hashes",
    ):
        clean[key] = tuple(clean.get(key, ()))
    return TargetContext(**clean)
