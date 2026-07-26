"""Bounded, field-typed model transport whose values are wrapped by the host.

The wire grammar is intentionally not JSON and has no model-visible artifact
metadata.  A response is a sequence of registered field blocks:

    parent_claim_ref
    <registered content reference>
    /parent_claim_ref
    END

Fields may repeat only when registered as repeated.  Host bindings, versions,
hashes and persistence envelopes are added after this adapter succeeds.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping


TRANSPORT_VERSION = 4
MAX_TRANSPORT_CHARS = 64 * 1024
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_FORBIDDEN_VALUE = re.compile(
    r"```|(?:^|\s)(?:Artifact|schema_version)\s*:|"
    r"(?:^|\s)(?:theorem|lemma)\s+\w+|:=\s*by|"
    r"\\(?:frac|sum|forall|exists|mathbb|text|begin|end)\b|\$",
    re.IGNORECASE | re.MULTILINE,
)


class AdapterStatus(str, Enum):
    ADAPTER_BLOCKED = "ADAPTER_BLOCKED"
    INFRASTRUCTURE_BLOCKED = "INFRASTRUCTURE_BLOCKED"


class AdapterError(ValueError):
    """Transport failure outside the mathematical state machine."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        role: str,
        field: str = "",
        status: AdapterStatus = AdapterStatus.ADAPTER_BLOCKED,
    ) -> None:
        self.code = str(code)
        self.role = str(role)
        self.field = str(field)
        self.status = status
        super().__init__(f"{status.value}:{code}:{role}:{field}:{message}")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    repeated: bool = False
    required: bool = True
    max_chars: int = 4096
    choices: tuple[str, ...] = ()
    value_kind: str = "identifier"

    def __post_init__(self) -> None:
        if not _FIELD_NAME.fullmatch(self.name):
            raise ValueError(f"invalid transport field name: {self.name}")
        if self.max_chars <= 0:
            raise ValueError("field max_chars must be positive")
        if self.value_kind not in {
            "enum", "identifier", "reference", "proof_step",
        }:
            raise ValueError(f"invalid transport value kind: {self.value_kind}")
        if self.value_kind == "enum" and not self.choices:
            raise ValueError(f"enum field has no registered choices: {self.name}")


@dataclass(frozen=True)
class RoleTransport:
    role: str
    fields: tuple[FieldSpec, ...]
    version: int = TRANSPORT_VERSION

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if not self.role or len(names) != len(set(names)):
            raise ValueError(f"invalid role transport registry entry: {self.role}")


def _ref(name: str, **kwargs) -> FieldSpec:
    return FieldSpec(name, value_kind="reference", max_chars=256, **kwargs)


def _id(name: str, **kwargs) -> FieldSpec:
    return FieldSpec(name, value_kind="identifier", max_chars=64, **kwargs)


def _enum(name: str, choices: tuple[str, ...], **kwargs) -> FieldSpec:
    return FieldSpec(name, choices=choices, value_kind="enum", max_chars=64, **kwargs)


ROLE_TRANSPORT_REGISTRY: dict[str, RoleTransport] = {
    "strategy_tournament_selector": RoleTransport(
        "strategy_tournament_selector", (
        _id("plan_id", repeated=True),
        _enum("reason_code", (
            "MAXIMIZES_INFORMATION_GAIN",
            "STRONGEST_THEOREM_SUPPORT",
            "LOWEST_COMPLEXITY",
            "LOWEST_RISK",
            "STRICTEST_REDUCTION",
        ), repeated=True),
    )),
    "strategy_critic_selector": RoleTransport(
        "strategy_critic_selector", (
        _id("plan_id", repeated=True),
        _enum("reason_code", (
            "MAXIMIZES_INFORMATION_GAIN",
            "STRONGEST_THEOREM_SUPPORT",
            "LOWEST_COMPLEXITY",
            "LOWEST_RISK",
            "STRICTEST_REDUCTION",
        ), repeated=True),
    )),
    "synthesis": RoleTransport("synthesis", (
        _enum("choice_code", tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        _enum("reason_code", (
            "DIRECT_LOCAL_CONTRADICTION",
            "SUPPORTED_BY_THEOREM_CARDS",
            "STRICTEST_REDUCTION",
            "NOVEL_VIEWPOINT",
            "LOWEST_COMPLEXITY",
            "HIGHEST_GAP_COVERAGE",
            "SHALLOWEST_DEPENDENCY_DEPTH",
        )),
    )),
    "premise_suspicion": RoleTransport("premise_suspicion", (
        _ref("target_ref"),
        _ref("evidence_ref"),
    )),
    "premise_auditor": RoleTransport("premise_auditor", (
        _ref("target_ref"),
        _ref("evidence_ref", repeated=True),
        _id("analysis_code", repeated=True, required=False),
        _enum("status", ("CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE")),
    )),
    "premise_proponent": RoleTransport("premise_proponent", (
        _ref("target_ref"),
        _ref("correction_ref", required=False),
        _ref("evidence_ref"),
        _enum("status", ("RESCUED", "NOT_RESCUED", "INCONCLUSIVE")),
    )),
    "definition_auditor": RoleTransport("definition_auditor", (
        _ref("target_ref"),
        _id("symbol_id", repeated=True, required=False),
        _id("domain_id", repeated=True, required=False),
        _id("topology_id", repeated=True, required=False),
        _id("definition_id", repeated=True, required=False),
        _id("missing_definition_id", repeated=True, required=False),
        _enum("audit_outcome", (
            "COMPLETE", "MISSING_DEFINITION", "REFRAME_REQUIRED",
        )),
    )),
    "counterexample_worker": RoleTransport("counterexample_worker", (
        _ref("target_ref"),
        _ref("case_ref", repeated=True, required=False),
        _enum("status", (
            "COUNTEREXAMPLE_FOUND", "NO_COUNTEREXAMPLE", "INCONCLUSIVE",
        )),
    )),
    "decomposer": RoleTransport("decomposer", (
        _ref("parent_claim_ref"),
        _enum("child_kind", ("DEFINITION", "LEMMA", "COROLLARY")),
        _id("definition_id", repeated=True, required=False),
        _ref("dependency_id", repeated=True, required=False),
        _id("outline_step_id", repeated=True, required=False),
        _enum("move_id", ("A", "B", "C", "D")),
    )),
    "math_ir_translator": RoleTransport("math_ir_translator", (
        _ref("parent_claim_ref"),
        _enum("move_id", ("A", "B", "C", "D")),
    )),
    "proof_action_selector": RoleTransport("proof_action_selector", (
        _id("goal_id"),
        _id("action_id"),
        _id("operand_id", repeated=True, required=False),
        _id("substitution_id", repeated=True, required=False),
    )),
    "adversarial_proponent": RoleTransport("adversarial_proponent", (
        _ref("target_proposition_ref"),
        _id("issue_code", repeated=True, required=False),
        _id("repair_step_id", repeated=True, required=False),
        _enum("status", ("DEFENDED", "REJECTED", "INCONCLUSIVE")),
    )),
    "judge": RoleTransport("judge", (
        _ref("target_proposition_ref"),
        _enum("decision", ("ACCEPT", "REJECT", "INCONCLUSIVE")),
        _id("reason_code"),
    )),
}


@dataclass(frozen=True)
class DecodedRoleFields:
    role: str
    values: Mapping[str, str | tuple[str, ...]]
    transport_hash: str


def transport_prompt(
    role: str,
    *,
    registered_choices: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """Return the only model-facing framing contract."""
    contract = resolve_transport(role)
    lines = [
        "Return only bounded typed field records.",
        "Each record is one line: field_name SPACE field_value SEMICOLON.",
        "Repeat only fields marked repeatable. Finish with the exact line END;",
        "Do not emit braces, quoted keys, invented metadata, fences, or source code.",
    ]
    for field in contract.fields:
        qualifier = "required" if field.required else "optional"
        if field.repeated:
            qualifier += ", repeatable"
        choice_registry = registered_choices or {}
        choices = tuple(choice_registry.get(field.name, field.choices))
        if choices:
            qualifier += "; choose exactly " + " or ".join(choices)
        elif field.name in choice_registry:
            qualifier += "; no registered choices; omit this field"
        qualifier += f"; type {field.value_kind}"
        lines.append(f"{field.name} ({qualifier})")
    return "\n".join(lines)


def resolve_transport(role: str) -> RoleTransport:
    try:
        return ROLE_TRANSPORT_REGISTRY[str(role)]
    except KeyError as exc:
        raise AdapterError(
            "UNREGISTERED_ROLE",
            "role has no typed transport",
            role=str(role),
        ) from exc


def _validate_value(
    role: str,
    spec: FieldSpec,
    value: str,
    *,
    choices: tuple[str, ...] | None = None,
) -> str:
    if spec.name == "choice_code" and value != value.strip():
        raise AdapterError(
            "INVALID_CHOICE",
            "choice code must contain exactly one scoped character",
            role=role,
            field=spec.name,
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterError("EMPTY_FIELD", "field is empty", role=role, field=spec.name)
    if len(normalized) > spec.max_chars:
        raise AdapterError(
            "FIELD_TOO_LARGE",
            f"field exceeds {spec.max_chars} characters",
            role=role,
            field=spec.name,
        )
    if _FORBIDDEN_VALUE.search(normalized):
        raise AdapterError(
            "MODEL_OWNED_SYNTAX",
            "field contains forbidden envelope, Lean, fence, or raw LaTeX syntax",
            role=role,
            field=spec.name,
        )
    allowed = spec.choices if choices is None else choices
    if (allowed and normalized not in allowed) or (
        choices is not None and not allowed
    ):
        raise AdapterError(
            "INVALID_CHOICE",
            (
                f"expected one of {', '.join(allowed)}"
                if allowed else "field has no registered choices and must be omitted"
            ),
            role=role,
            field=spec.name,
        )
    if spec.value_kind == "identifier" and not _IDENTIFIER.fullmatch(normalized):
        raise AdapterError(
            "INVALID_IDENTIFIER", "expected one registered DSL identifier",
            role=role, field=spec.name,
        )
    if spec.value_kind == "reference" and not _REFERENCE.fullmatch(normalized):
        raise AdapterError(
            "INVALID_REFERENCE", "expected one host content reference",
            role=role, field=spec.name,
        )
    if spec.value_kind == "proof_step":
        from autoresearch.prefill.math_ir import TACTIC_REGISTRY
        parts = normalized.split()
        if (
            not parts
            or parts[0] not in TACTIC_REGISTRY
            or len(parts) - 1 != TACTIC_REGISTRY[parts[0]][0]
            or any(not _IDENTIFIER.fullmatch(item) for item in parts[1:])
        ):
            raise AdapterError(
                "INVALID_PROOF_STEP", "expected registered tactic and identifier IDs",
                role=role, field=spec.name,
            )
    return normalized


def decode_role_fields(
    text: str,
    role: str,
    *,
    registered_choices: Mapping[str, Iterable[str]] | None = None,
) -> DecodedRoleFields:
    """Decode one complete response without mutating proof state."""
    contract = resolve_transport(role)
    if not isinstance(text, str):
        raise AdapterError("NOT_TEXT", "response is not text", role=role)
    if len(text) > MAX_TRANSPORT_CHARS:
        raise AdapterError("RESPONSE_TOO_LARGE", "response exceeds cap", role=role)
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[-1]:
        lines.pop()
    specs = {field.name: field for field in contract.fields}
    decoded: dict[str, list[str]] = {}
    compact = bool(lines) and all(
        not line.strip()
        or line.strip() == "END;"
        or (
            line.strip().endswith(";")
            and len(line.strip().split(maxsplit=1)) == 2
        )
        for line in lines
    )
    if compact:
        saw_end = False
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            if line == "END;":
                saw_end = True
                if any(item.strip() for item in lines[index + 1:]):
                    raise AdapterError(
                        "TRAILING_DATA", "tokens follow END", role=role,
                    )
                break
            marker, raw_value = line[:-1].split(maxsplit=1)
            if marker not in specs:
                raise AdapterError(
                    "UNKNOWN_FIELD",
                    f"unknown field {marker!r}",
                    role=role,
                    field=marker,
                )
            spec = specs[marker]
            values = decoded.setdefault(marker, [])
            if values and not spec.repeated:
                raise AdapterError(
                    "DUPLICATE_FIELD",
                    "field is not repeatable",
                    role=role,
                    field=marker,
                )
            choices = (
                tuple(registered_choices[marker])
                if registered_choices is not None and marker in registered_choices
                else None
            )
            values.append(_validate_value(
                role, spec, raw_value, choices=choices,
            ))
        if not saw_end:
            raise AdapterError("INCOMPLETE", "missing END", role=role)
        return _finish_decoded(role, contract, decoded)
    index = 0
    saw_end = False
    while index < len(lines):
        marker = lines[index].strip()
        index += 1
        if marker == "END":
            saw_end = True
            if any(line.strip() for line in lines[index:]):
                raise AdapterError(
                    "TRAILING_DATA", "tokens follow END", role=role,
                )
            break
        if marker not in specs:
            raise AdapterError(
                "UNKNOWN_FIELD", f"unknown field {marker!r}", role=role, field=marker,
            )
        closing = f"/{marker}"
        value_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != closing:
            if lines[index].strip() == "END":
                raise AdapterError(
                    "UNCLOSED_FIELD", "END occurred inside field", role=role,
                    field=marker,
                )
            value_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise AdapterError(
                "UNCLOSED_FIELD", f"missing {closing}", role=role, field=marker,
            )
        index += 1
        spec = specs[marker]
        values = decoded.setdefault(marker, [])
        if values and not spec.repeated:
            raise AdapterError(
                "DUPLICATE_FIELD", "field is not repeatable", role=role, field=marker,
            )
        choices = (
            tuple(registered_choices[marker])
            if registered_choices is not None and marker in registered_choices
            else None
        )
        values.append(_validate_value(
            role, spec, "\n".join(value_lines), choices=choices,
        ))
    if not saw_end:
        raise AdapterError("INCOMPLETE", "missing END", role=role)
    return _finish_decoded(role, contract, decoded)


def _finish_decoded(
    role: str,
    contract: RoleTransport,
    decoded: Mapping[str, list[str]],
) -> DecodedRoleFields:
    missing = [
        field.name for field in contract.fields
        if field.required and not decoded.get(field.name)
    ]
    if missing:
        raise AdapterError(
            "MISSING_FIELDS", ", ".join(missing), role=role,
        )
    values: dict[str, str | tuple[str, ...]] = {}
    for spec in contract.fields:
        items = decoded.get(spec.name, [])
        values[spec.name] = tuple(items) if spec.repeated else (items[0] if items else "")
    canonical = json.dumps(
        {"role": role, "values": values, "transport_version": contract.version},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return DecodedRoleFields(role, values, hashlib.sha256(canonical).hexdigest())


def typed_transport_complete(text: str, role: str) -> bool:
    """Streaming stop predicate; acceptance still requires full decode."""
    if not isinstance(text, str) or len(text) > MAX_TRANSPORT_CHARS:
        return False
    if not re.search(r"(?:^|\n)END;?\s*\Z", text):
        return False
    try:
        decode_role_fields(text, role)
    except AdapterError:
        return False
    return True


def host_artifact(
    decoded: DecodedRoleFields,
    *,
    host_bindings: Mapping[str, object],
    dependencies: Iterable[str],
) -> dict[str, object]:
    """Create the JSON-persisted envelope exclusively on the host."""
    payload = {
        "schema_version": TRANSPORT_VERSION,
        "producer_role": decoded.role,
        "transport_hash": decoded.transport_hash,
        "dependencies": list(dependencies),
        "fields": dict(decoded.values),
        "bindings": dict(host_bindings),
    }
    payload["content_hash"] = hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    return payload


def registry_manifest() -> dict[str, object]:
    return {
        role: {
            "version": contract.version,
            "fields": [asdict(field) for field in contract.fields],
        }
        for role, contract in sorted(ROLE_TRANSPORT_REGISTRY.items())
    }


def registry_hash() -> str:
    """Content address the executable typed transport registry."""
    return hashlib.sha256(json.dumps(
        {
            "transport_version": TRANSPORT_VERSION,
            "roles": registry_manifest(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
