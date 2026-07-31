"""Atomic, public-safe live execution status for proof orchestration."""
from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    load_checkpoint,
    verified_reuse_provenance,
)


SCHEMA_VERSION = 2
VALID_STATES = {
    "queued",
    "prefill",
    "decode",
    "review",
    "completed",
    "failed",
    "idle",
}
_SAFE_TEXT = re.compile(r"[^A-Za-z0-9_.:@+\- ]")
_HIGH_ENTROPY_TOKEN = re.compile(
    r"(?i)(?:\b(?:cursor|sk|key|token)_[A-Za-z0-9_\-]{20,}\b"
    r"|(?<![A-Fa-f0-9])[A-Fa-f0-9]{40,56}(?![A-Fa-f0-9]))"
)


def _safe_text(value, limit: int = 160) -> str:
    text = str(value or "")
    if "/" in text or re.search(
        r"(?i)(?:api[_ -]?key|secret|access[_ -]?token|prompt)\s*[:=]",
        text,
    ) or _HIGH_ENTROPY_TOKEN.search(text):
        return "redacted"
    return _SAFE_TEXT.sub("_", text)[:limit]


class AtomicLiveStatus:
    """Write one status document via locked, permission-restricted replace."""

    def __init__(
        self,
        path: Path,
        *,
        supervisor_pid: int,
        iteration: int = 0,
        run_id: str = "",
        min_interval_s: float = 2.0,
    ) -> None:
        self.path = Path(path).expanduser()
        self.supervisor_pid = int(supervisor_pid)
        self.iteration = int(iteration)
        self.run_id = _safe_text(run_id)
        self.min_interval_s = float(min_interval_s)
        self.writer_pid = os.getpid()
        self._sequence = 0
        self._last_write = 0.0
        self._last_signature = None
        self._lock = threading.Lock()
        self._phase_started_at = time.time()

    def set_context(
        self,
        *,
        iteration: int | None = None,
        run_id: str | None = None,
    ) -> None:
        if iteration is not None:
            self.iteration = int(iteration)
        if run_id is not None:
            self.run_id = _safe_text(run_id)

    def emit(
        self,
        *,
        phase: str,
        role: str = "",
        state: str,
        progress_current: int | None = None,
        progress_total: int | None = None,
        progress_unit: str = "",
        active_obligation_id: str = "",
        worker: str = "",
        source: str = "",
        hit_source: str = "",
        force: bool = False,
    ) -> bool:
        if state not in VALID_STATES:
            raise ValueError(f"invalid live status state: {state}")
        now = time.time()
        signature = (
            phase,
            role,
            state,
            progress_current,
            progress_total,
            progress_unit,
            active_obligation_id,
            worker,
            source,
            hit_source,
            self.iteration,
            self.run_id,
        )
        with self._lock:
            phase_changed = (
                self._last_signature is None
                or self._last_signature[:3] != signature[:3]
            )
            if phase_changed:
                self._phase_started_at = now
            if (
                not force
                and signature == self._last_signature
                and now - self._last_write < self.min_interval_s
            ):
                return False
            if (
                not force
                and not phase_changed
                and now - self._last_write < self.min_interval_s
            ):
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                os.chmod(lock_path, 0o600)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        existing = json.loads(
                            self.path.read_text(encoding="utf-8"),
                        )
                        self._sequence = max(
                            self._sequence,
                            int(existing.get("sequence", 0)),
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                    self._sequence += 1
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "supervisor_pid": self.supervisor_pid,
                        "writer_pid": self.writer_pid,
                        "run_id": self.run_id,
                        "iteration": self.iteration,
                        "phase": _safe_text(phase),
                        "role": _safe_text(role),
                        "state": state,
                        "progress": {
                            "current": (
                                max(0, int(progress_current))
                                if progress_current is not None else None
                            ),
                            "total": (
                                max(0, int(progress_total))
                                if progress_total is not None else None
                            ),
                            "unit": _safe_text(progress_unit, 32),
                        },
                        "started_at": self._phase_started_at,
                        "updated_at": now,
                        "active_obligation_id": _safe_text(
                            active_obligation_id,
                        ),
                        "worker": _safe_text(worker),
                        "source": _safe_text(source),
                        "hit_source": _safe_text(hit_source),
                        "sequence": self._sequence,
                    }
                    orchestration_path = os.environ.get(
                        "KAKEYA_ORCHESTRATION_STATE_PATH",
                        "",
                    )
                    if orchestration_path:
                        try:
                            checkpoint = load_checkpoint(
                                Path(orchestration_path).expanduser(),
                            )
                            orchestration = (
                                json.loads(
                                    Path(orchestration_path).expanduser().read_text(
                                        encoding="utf-8",
                                    ),
                                )
                                if checkpoint is not None else {}
                            )
                        except (OSError, TypeError, ValueError, json.JSONDecodeError):
                            checkpoint = None
                            orchestration = {}
                        retry_counters = orchestration.get(
                            "retry_counters",
                            {},
                        )
                        current_state = _safe_text(
                            orchestration.get("state", ""),
                        )
                        reuse = (
                            verified_reuse_provenance(checkpoint)
                            if checkpoint is not None else {
                                "strategy_reused": False,
                                "generator_reused": False,
                                "critic_reused": False,
                                "role_reused": {},
                                "reused_artifacts": {},
                                "diagnostics": {},
                            }
                        )
                        critic_ref = reuse["reused_artifacts"].get("critic", {})
                        adapter_status = _safe_text(
                            orchestration.get("adapter_status", ""),
                        )
                        blocked_category = (
                            adapter_status
                            or (
                                "PROOF_BLOCKED"
                                if current_state == "BLOCKED" else ""
                            )
                        )
                        payload.update({
                            "orchestration_state": current_state,
                            "active_role": _safe_text(
                                orchestration.get("current_role", role),
                            ),
                            "resume_origin": _safe_text(
                                orchestration.get("resume_origin", ""),
                            ),
                            "transition_reason": _safe_text(
                                orchestration.get(
                                    "last_transition_reason",
                                    "",
                                ),
                            ),
                            "retry_count": int(
                                retry_counters.get(current_state, 0),
                            ),
                            "decomposition_iteration": int(
                                orchestration.get(
                                    "decomposition_iteration",
                                    0,
                                ),
                            ),
                            "viewpoint": _safe_text(
                                orchestration.get("viewpoint", ""),
                            ),
                            "synthesis_iteration": int(
                                orchestration.get("synthesis_iteration", 0),
                            ),
                            "candidate_count": int(
                                orchestration.get(
                                    "definition_candidate_count",
                                    orchestration.get("candidate_count", 0),
                                ),
                            ),
                            "selected_move": _safe_text(
                                orchestration.get("selected_move_id", ""),
                            ),
                            "candidate_set_hash": _safe_text(
                                orchestration.get("candidate_set_hash", ""),
                            ),
                            "ranking_hash": _safe_text(
                                orchestration.get("ranking_hash", ""),
                            ),
                            "decomposition_exploration": {
                                "contract_id": _safe_text(
                                    orchestration.get(
                                        "exploration_contract_id", "",
                                    ),
                                ),
                                "generated": int(orchestration.get(
                                    "candidate_count", 0,
                                )),
                                "surviving": (
                                    int(orchestration.get("candidate_count", 0))
                                    - len(orchestration.get(
                                        "exploration_rejections", {},
                                    ))
                                ),
                                "selected_candidate_ids": [
                                    _safe_text(item, 80)
                                    for item in orchestration.get(
                                        "exploration_selected_candidate_ids",
                                        [],
                                    )
                                ],
                                "current_index": int(orchestration.get(
                                    "exploration_current_index", 0,
                                )),
                                "current_candidate_id": _safe_text(
                                    orchestration.get(
                                        "exploration_current_candidate_id", "",
                                    ),
                                    80,
                                ),
                                "formalization_status": _safe_text(
                                    orchestration.get(
                                        "exploration_formalization_status", "",
                                    ),
                                ),
                                "reduction_status": _safe_text(
                                    orchestration.get(
                                        "exploration_reduction_status", "",
                                    ),
                                ),
                                "rejected_reason_codes": sorted({
                                    _safe_text(reason, 80)
                                    for reasons in orchestration.get(
                                        "exploration_rejections", {},
                                    ).values()
                                    for reason in reasons
                                }),
                                "representation_analysis": {
                                    "status": _safe_text(orchestration.get(
                                        "representation_current_status", "",
                                    )),
                                    "missing_primitive_ids": [
                                        _safe_text(item, 80)
                                        for item in orchestration.get(
                                            "representation_missing_primitive_ids",
                                            [],
                                        )
                                    ],
                                    "source_resolution": _safe_text(
                                        orchestration.get(
                                            "representation_source_resolution",
                                            "",
                                        ),
                                    ),
                                    "retry_state": _safe_text(
                                        orchestration.get(
                                            "representation_retry_state", "",
                                        ),
                                    ),
                                    "report_count": len(orchestration.get(
                                        "representation_report_refs", {},
                                    )),
                                    "exhaustion_hash": _safe_text(
                                        orchestration.get(
                                            "representation_exhaustion_hash",
                                            "",
                                        ),
                                        80,
                                    ),
                                },
                            },
                            "theorem_card_count": len(
                                orchestration.get("theorem_card_ids", []),
                            ),
                            "stagnation_reason": _safe_text(
                                orchestration.get("stagnation_reason", ""),
                            ),
                            "definition_resolution": {
                                "action": (
                                    "RESOLVE_ONE_DEFINITION_QUERY"
                                    if orchestration.get(
                                        "current_definition_gap_id", "",
                                    ) else ""
                                ),
                                "current_gap": _safe_text(orchestration.get(
                                    "current_definition_gap_id", "",
                                )),
                                "candidate_count": int(orchestration.get(
                                    "definition_candidate_count", 0,
                                )),
                                "lean_status": _safe_text(orchestration.get(
                                    "lean_definition_status", "",
                                )),
                                "store_hash_delta": _safe_text(
                                    orchestration.get(
                                        "definition_store_hash_delta", "",
                                    ),
                                    160,
                                ),
                                "environment_hash_delta": _safe_text(
                                    orchestration.get(
                                        "definition_environment_hash_delta", "",
                                    ),
                                    160,
                                ),
                                "query_hash": _safe_text(orchestration.get(
                                    "definition_query_hash", "",
                                )),
                                "source_statuses": {
                                    _safe_text(key, 48): _safe_text(value, 48)
                                    for key, value in orchestration.get(
                                        "definition_source_statuses", {},
                                    ).items()
                                },
                                "property_statuses": {
                                    _safe_text(key, 80): {
                                        _safe_text(prop, 80): _safe_text(
                                            status, 24,
                                        )
                                        for prop, status in values.items()
                                    }
                                    for key, values in orchestration.get(
                                        "definition_property_statuses", {},
                                    ).items()
                                },
                                "branch_count": len(orchestration.get(
                                    "definition_branch_hashes", [],
                                )),
                                "exhaustion_hash": _safe_text(
                                    orchestration.get(
                                        "definition_exhaustion_hash", "",
                                    ),
                                ),
                                "interface_hash": _safe_text(
                                    orchestration.get(
                                        "definition_interface_hash", "",
                                    ),
                                ),
                                "backjump_target": _safe_text(
                                    orchestration.get(
                                        "definition_backjump_target", "",
                                    ),
                                ),
                            },
                            "mathematical_progress": {
                                key: int(value)
                                for key, value in orchestration.get(
                                    "progress_vector", {},
                                ).items()
                            },
                            "stagnation_count": int(orchestration.get(
                                "semantic_stagnation_count", 0,
                            )),
                            "semantic_rejection": bool(
                                orchestration.get("semantic_rejection"),
                            ),
                            "novel_proposals": int(
                                orchestration.get("novel_proposals", 0),
                            ),
                            "architecture_version": int(
                                orchestration.get("architecture_version", 1),
                            ),
                            "active_gate": _safe_text(
                                orchestration.get("active_gate", ""),
                            ),
                            "adapter_status": _safe_text(
                                adapter_status,
                            ),
                            "blocked_category": _safe_text(blocked_category),
                            "blocked_reason": _safe_text(
                                orchestration.get("blocked_reason", ""),
                            ),
                            "resume_role": _safe_text(
                                orchestration.get("current_role", ""),
                            ),
                            "execution_state": state,
                            "execution_phase": _safe_text(phase),
                            "typed_ir_hash": _safe_text(
                                orchestration.get("typed_ir_hash", ""),
                            ),
                            "proposition_hash": _safe_text(
                                orchestration.get("proposition_hash", ""),
                            ),
                            "elaborated_theorem_id": _safe_text(
                                orchestration.get("elaborated_theorem_id", ""),
                            ),
                            "lean_contract_id": _safe_text(
                                orchestration.get("lean_contract_id", ""),
                            ),
                            "lean_contract_version": int(
                                orchestration.get("lean_contract_version", 0),
                            ),
                            "lean_symbol_table_id": _safe_text(
                                orchestration.get("lean_symbol_table_id", ""),
                            ),
                            "lean_symbol_table_version": int(
                                orchestration.get(
                                    "lean_symbol_table_version",
                                    0,
                                ),
                            ),
                            "validated_formalizer_unit_hashes": {
                                _safe_text(key, 48): _safe_text(value, 80)
                                for key, value in orchestration.get(
                                    "formalizer_unit_hashes",
                                    {},
                                ).items()
                            },
                            "strategy_reused": bool(
                                reuse["strategy_reused"],
                            ),
                            "generator_reused": reuse["generator_reused"],
                            "critic_reused": reuse["critic_reused"],
                            "role_reused": reuse["role_reused"],
                            "reuse_diagnostics": reuse["diagnostics"],
                            "critic_artifact_sha256": _safe_text(
                                critic_ref.get("sha256", ""),
                            ),
                            "critic_source_run_id": _safe_text(
                                critic_ref.get("source_run_id", ""),
                            ),
                            "strategy_tournament": {
                                "event_type": _safe_text(orchestration.get(
                                    "strategy_event_type", "",
                                )),
                                "plans_total": len(orchestration.get(
                                    "strategy_plan_ids", [],
                                )),
                                "plans_feasible": len(orchestration.get(
                                    "feasible_strategy_plan_ids", [],
                                )),
                                "branches_killed": int(orchestration.get(
                                    "branches_killed", 0,
                                )),
                                "selected_plan_id": _safe_text(
                                    orchestration.get(
                                        "selected_strategy_plan_id", "",
                                    ),
                                ),
                            },
                            "strategy_provider": {
                                "provider": _safe_text(orchestration.get(
                                    "strategy_provider", "cursor-sdk",
                                )),
                                "configured": bool(orchestration.get(
                                    "strategy_provider_configured", False,
                                )),
                                "model_id": _safe_text(orchestration.get(
                                    "strategy_model_id", "",
                                )),
                                "run_status": _safe_text(orchestration.get(
                                    "strategy_run_status",
                                    "CONFIGURATION_REQUIRED",
                                )),
                                "run_id": _safe_text(orchestration.get(
                                    "strategy_run_id", "",
                                )),
                            },
                            "model_residency": {
                                "phase": _safe_text(orchestration.get(
                                    "residency_phase", "GEMMA_SERVING",
                                )),
                                "active_model": _safe_text(orchestration.get(
                                    "active_model", "gemma",
                                )),
                            },
                            "oprover_advisor": {
                                "candidates": int(orchestration.get(
                                    "oprover_candidate_count", 0,
                                )),
                                "verified": int(orchestration.get(
                                    "oprover_verified_count", 0,
                                )),
                            },
                            "critic_advisory_state": _safe_text(
                                orchestration.get(
                                    "critic_advisory_state", "PENDING",
                                ),
                            ),
                            "research_contract": {
                                "contract_id": _safe_text(orchestration.get(
                                    "research_contract_id", "",
                                )),
                                "accepted": bool(orchestration.get(
                                    "research_contract_id", "",
                                )),
                                "rejection_codes": [
                                    _safe_text(item, 64)
                                    for item in orchestration.get(
                                        "research_contract_rejection_codes", [],
                                    )
                                ],
                            },
                            "proof_search": {
                                "actions_attempted": int(orchestration.get(
                                    "lean_actions_attempted", 0,
                                )),
                                "actions_accepted": int(orchestration.get(
                                    "lean_actions_accepted", 0,
                                )),
                                "subgoals_closed": int(orchestration.get(
                                    "subgoals_closed", 0,
                                )),
                                "subgoals_remaining": int(orchestration.get(
                                    "subgoals_remaining", 0,
                                )),
                                "new_definitions": int(orchestration.get(
                                    "new_elaborated_definitions", 0,
                                )),
                                "new_lemmas": int(orchestration.get(
                                    "new_elaborated_lemmas", 0,
                                )),
                                "accepted_children": int(orchestration.get(
                                    "accepted_children", 0,
                                )),
                                "verified_counterexamples": int(
                                    orchestration.get(
                                        "verified_counterexamples", 0,
                                    ),
                                ),
                                "tokens_per_accepted_step": (
                                    int(orchestration.get(
                                        "proof_tokens_consumed", 0,
                                    ))
                                    / max(1, int(orchestration.get(
                                        "lean_actions_accepted", 0,
                                    )))
                                ),
                            },
                        })
                    temporary = self.path.with_name(
                        f".{self.path.name}.{self.writer_pid}."
                        f"{threading.get_ident()}.tmp",
                    )
                    try:
                        temporary.write_text(
                            json.dumps(
                                payload,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                        os.chmod(temporary, 0o600)
                        os.replace(temporary, self.path)
                        os.chmod(self.path, 0o600)
                    finally:
                        temporary.unlink(missing_ok=True)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            self._last_signature = signature
            self._last_write = now
            return True
