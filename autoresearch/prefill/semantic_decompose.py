"""Retained-KV admission and exact one-step proof interfaces."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


class SemanticUnitTooLarge(ValueError):
    status = "SEMANTIC_UNIT_TOO_LARGE"

    def __init__(self, unit: str, token_count: int, max_tokens: int) -> None:
        self.unit = unit
        self.token_count = int(token_count)
        self.max_tokens = int(max_tokens)
        super().__init__(
            f"{self.status}: {unit} requires {token_count} retained tokens; "
            f"limit is {max_tokens}",
        )


class SemanticResponseIncomplete(RuntimeError):
    """A structured role stopped before EOS; partial output is audit-only."""

    status = "SEMANTIC_RESPONSE_INCOMPLETE"

    def __init__(
        self,
        role: str,
        *,
        token_count: int,
        stop_reason: str,
        response_cap_exhausted: bool,
    ) -> None:
        self.role = role
        self.token_count = int(token_count)
        self.stop_reason = str(stop_reason)
        self.response_cap_exhausted = bool(response_cap_exhausted)
        super().__init__(
            f"{self.status}: {role} stopped before EOS after {token_count} "
            f"tokens (stop_reason={stop_reason}, "
            f"response_cap_exhausted={response_cap_exhausted})"
        )


@dataclass(frozen=True)
class ProofStepInterface:
    root_goal_hash: str
    target_obligation_id: str
    target_statement: str
    target_statement_hash: str
    target_formal_status: str = "UNFORMALIZED"
    target_lean_signature: str = ""
    target_lean_signature_hash: str = ""
    parent_interface: dict = field(default_factory=dict)
    public_assumptions: list[str] = field(default_factory=list)
    dependency_interface: dict = field(default_factory=dict)
    active_no_go_lessons: list[dict] = field(default_factory=list)
    current_target_evidence: str = ""
    archive_manifest: dict = field(default_factory=dict)
    interface_hash: str = ""


def canonical_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_proof_step_interface(
    *,
    root_goal_hash: str,
    target: dict,
    parent: dict | None,
    active_no_go_lessons: list[dict],
    archive_manifest: dict,
) -> ProofStepInterface:
    statement = str(target.get("statement", ""))
    parent_interface = {}
    if parent is not None:
        parent_interface = {
            "obligation_id": parent.get("obligation_id", ""),
            "statement_hash": hashlib.sha256(
                str(parent.get("statement", "")).encode(),
            ).hexdigest(),
            "formal_status": parent.get("formal_status", "UNFORMALIZED"),
            "lean_signature": parent.get("lean_signature", ""),
            "lean_signature_hash": parent.get("lean_signature_hash", ""),
            "certificate_hash": parent.get(
                "decomposition_certificate_hash",
                "",
            ),
            "reduction_theorem_hash": parent.get(
                "reduction_theorem_hash",
                "",
            ),
            "dependency_ids": list(parent.get("dependency_ids", [])),
        }
    payload = {
        "root_goal_hash": root_goal_hash,
        "target_obligation_id": target.get("obligation_id", ""),
        "target_statement": statement,
        "target_statement_hash": hashlib.sha256(
            statement.encode(),
        ).hexdigest(),
        "target_formal_status": target.get(
            "formal_status",
            "UNFORMALIZED",
        ),
        "target_lean_signature": target.get("lean_signature", ""),
        "target_lean_signature_hash": target.get(
            "lean_signature_hash",
            "",
        ),
        "parent_interface": parent_interface,
        "public_assumptions": list(target.get("public_assumptions", [])),
        "dependency_interface": {
            "certificate_hash": target.get(
                "decomposition_certificate_hash",
                "",
            ),
            "reduction_theorem_hash": target.get(
                "reduction_theorem_hash",
                "",
            ),
            "dependency_labels": list(target.get("dependency_labels", [])),
            "dependency_ids": list(target.get("dependency_ids", [])),
        },
        "active_no_go_lessons": active_no_go_lessons,
        "current_target_evidence": target.get("last_evidence", ""),
        "archive_manifest": archive_manifest,
    }
    return ProofStepInterface(
        **payload,
        interface_hash=canonical_hash(payload),
    )


def serialize_proof_step_interface(interface: ProofStepInterface) -> str:
    return json.dumps(
        asdict(interface),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def effective_input_limit(
    configured_prefill_tokens: int,
    max_retained_tokens: int,
    *,
    control_reserve_tokens: int = 0,
) -> int:
    if configured_prefill_tokens <= 0 or max_retained_tokens <= 0:
        raise ValueError("token limits must be > 0")
    if control_reserve_tokens < 0:
        raise ValueError("control reserve must be >= 0")
    effective = min(
        int(configured_prefill_tokens),
        int(max_retained_tokens) - int(control_reserve_tokens),
    )
    if effective <= 0:
        raise ValueError("control reserve consumes retained capacity")
    return effective


def admit_token_ids(
    unit: str,
    token_ids,
    *,
    configured_prefill_tokens: int,
    max_retained_tokens: int,
    control_reserve_tokens: int = 0,
) -> int:
    limit = effective_input_limit(
        configured_prefill_tokens,
        max_retained_tokens,
        control_reserve_tokens=control_reserve_tokens,
    )
    token_count = len(token_ids)
    if token_count > limit:
        raise SemanticUnitTooLarge(unit, token_count, limit)
    return token_count


def downstream_output_cap(
    *,
    max_retained_tokens: int,
    fixed_downstream_tokens: int,
    configured_output_tokens: int | None,
    control_reserve_tokens: int = 32,
) -> int:
    available = (
        int(max_retained_tokens)
        - int(fixed_downstream_tokens)
        - int(control_reserve_tokens)
    )
    if available <= 0:
        raise SemanticUnitTooLarge(
            "downstream fixed package",
            fixed_downstream_tokens + control_reserve_tokens,
            max_retained_tokens,
        )
    if configured_output_tokens is None or configured_output_tokens <= 0:
        return available
    return min(int(configured_output_tokens), available)
