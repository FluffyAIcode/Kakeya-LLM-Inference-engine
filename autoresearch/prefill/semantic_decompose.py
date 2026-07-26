"""Retained-KV admission and exact one-step proof interfaces."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field


DEFAULT_ARTIFACT_MAX_CHARS = 64 * 1024

STRUCTURED_ROLE_MINIMUM_OUTPUT_TOKENS = {
    "premise_auditor": 256,
    "premise_proponent": 256,
    "definition_auditor": 384,
    "counterexample_worker": 256,
    "decomposer": 512,
    "decomposer_scratchpad": 512,
    "synthesis_scratchpad": 512,
    "synthesis": 256,
    "formalizer": 768,
    "formalizer_parent_signature": 256,
    "formalizer_child_signature": 256,
    "formalizer_reduction_signature": 320,
    "prover": 384,
    "adversarial_proponent": 256,
    "judge": 256,
}


@dataclass(frozen=True)
class ScannedArtifact:
    json_text: str
    start: int
    end: int


def json_syntax_diagnostic(text: str, error: json.JSONDecodeError) -> str:
    """Describe a transport-complete syntax failure without echoing full output."""
    radius = 18
    start = max(0, error.pos - radius)
    end = min(len(text), error.pos + radius)
    context = text[start:end].replace("\r", " ").replace("\n", " ")
    context = re.sub(r"[^A-Za-z0-9_.,:{}\[\]\"' $\\:+\-]", "_", context)
    return (
        "EOS-complete malformed JSON: "
        f"{error.msg} at line {error.lineno} column {error.colno}; "
        f"first syntax context {context!r}"
    )


@dataclass(frozen=True)
class StructuredArtifactContract:
    role: str
    heading: str | None
    marker: str | None
    required_fields: tuple[str, ...]
    transport_stop: bool = True


STRUCTURED_ARTIFACT_CONTRACTS = {
    "strategy": StructuredArtifactContract(
        "strategy",
        None,
        None,
        (
            "candidate_id",
            "target_obligation_id",
            "hypothesis",
            "generator_directive",
            "critic_directive",
            "prefill_compute_chunk_tokens",
        ),
    ),
    "premise_suspicion": StructuredArtifactContract(
        "premise_suspicion",
        None,
        "Evidence artifact:",
        ("claim",),
        transport_stop=False,
    ),
    "premise_auditor": StructuredArtifactContract(
        "premise_auditor",
        "PREMISE_AUDIT",
        "Artifact:",
        (
            "status",
            "evidence_type",
            "evidence_source",
            "confidence",
            "artifact",
            "analysis",
        ),
    ),
    "premise_proponent": StructuredArtifactContract(
        "premise_proponent",
        "PREMISE_DEFENSE",
        "Artifact:",
        ("status", "correction", "failure_reason", "evidence"),
    ),
    "definition_auditor": StructuredArtifactContract(
        "definition_auditor",
        "DEFINITION_AUDIT",
        "Artifact:",
        ("definitions", "missing_definitions"),
    ),
    "counterexample_worker": StructuredArtifactContract(
        "counterexample_worker",
        "COUNTEREXAMPLE_REPORT",
        "Artifact:",
        ("status", "cases"),
    ),
    "decomposer": StructuredArtifactContract(
        "decomposer",
        "DECOMPOSITION_PROPOSAL",
        "Artifact:",
        ("parent_statement", "child", "public_assumptions", "reduction_contract"),
    ),
    "formalizer": StructuredArtifactContract(
        "formalizer",
        "FORMALIZATION_BUNDLE",
        "Artifact:",
        (
            "parent_signature",
            "parent_newly_formalized",
            "child_signature",
            "reduction_signature",
        ),
    ),
    "formalizer_parent_signature": StructuredArtifactContract(
        "formalizer_parent_signature",
        "LEAN_SIGNATURE_UNIT",
        "Artifact:",
        (
            "contract_id",
            "contract_version",
            "unit",
            "kind",
            "name",
            "binders",
            "proposition",
            "source",
        ),
    ),
    "formalizer_child_signature": StructuredArtifactContract(
        "formalizer_child_signature",
        "LEAN_SIGNATURE_UNIT",
        "Artifact:",
        (
            "contract_id",
            "contract_version",
            "unit",
            "kind",
            "name",
            "binders",
            "proposition",
            "source",
        ),
    ),
    "formalizer_reduction_signature": StructuredArtifactContract(
        "formalizer_reduction_signature",
        "LEAN_SIGNATURE_UNIT",
        "Artifact:",
        (
            "contract_id",
            "contract_version",
            "unit",
            "kind",
            "name",
            "binders",
            "proposition",
            "source",
        ),
    ),
    "prover": StructuredArtifactContract(
        "prover",
        "PROOF_ATTEMPT",
        "Artifact:",
        ("status", "reduction_theorem_source"),
    ),
    "adversarial_proponent": StructuredArtifactContract(
        "adversarial_proponent",
        "DEFENSE_REPORT",
        "Artifact:",
        ("status", "issues", "repairs"),
    ),
    "judge": StructuredArtifactContract(
        "judge",
        "JUDGE_DECISION",
        "Artifact:",
        ("decision", "reason"),
    ),
}


def structured_contract(role: str) -> StructuredArtifactContract:
    try:
        return STRUCTURED_ARTIFACT_CONTRACTS[role]
    except KeyError as exc:
        raise ValueError(f"unregistered structured role: {role}") from exc


def structured_transport_complete(text: str, role: str) -> bool:
    """Return true only at the registered contract's first object boundary."""
    contract = structured_contract(role)
    if not contract.transport_stop:
        return False
    try:
        if contract.heading is None:
            scanned = scan_artifact_object_prefix(text, marker=contract.marker)
        else:
            scanned = scan_structured_artifact_prefix(text, contract.heading)
    except ValueError:
        return False
    return bool(scanned is not None and not text[scanned.end:].strip())


def lint_structured_prompt(prompt: str) -> None:
    """Fail fast on conversational or syntactically invalid JSON contracts."""
    if (
        re.search(r'"[^"\n]*\|[^"\n]*"', prompt)
        or re.search(r"\b[A-Z][A-Z_]*(?:\|[A-Z][A-Z_]*)+\b", prompt)
    ):
        raise ValueError("structured prompt contains an invalid union placeholder")
    forbidden = (
        "prose after json",
        "prose after the json",
        "continue after the final }",
        "follow the json with",
    )
    lowered = prompt.casefold()
    if any(item in lowered for item in forbidden):
        raise ValueError("structured prompt permits prose after JSON")


def scan_artifact_object_prefix(
    text: str,
    *,
    marker: str | None = "Artifact:",
    max_chars: int = DEFAULT_ARTIFACT_MAX_CHARS,
) -> ScannedArtifact | None:
    """Return the first balanced Artifact object, even if more text follows.

    This is a transport-boundary scanner, not an acceptance parser.  It is
    deliberately string/escape aware and tracks matching object/array
    delimiters, but leaves JSON/schema/host validation to the strict parser.
    """
    if not isinstance(text, str):
        raise ValueError("structured response must be text")
    if max_chars <= 0:
        raise ValueError("artifact size cap must be > 0")
    if marker is None:
        start = 0
    else:
        marker_index = text.find(marker)
        if marker_index < 0:
            return None
        start = marker_index + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return None
    if text[start] != "{":
        raise ValueError("Artifact must start with a JSON object")
    stack: list[str] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}
    for index in range(start, len(text)):
        if index - start >= max_chars:
            raise ValueError("Artifact exceeds size cap")
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != matching[char]:
                raise ValueError("unbalanced Artifact JSON")
            stack.pop()
            if not stack:
                end = index + 1
                return ScannedArtifact(text[start:end], start, end)
    if len(text) - start >= max_chars:
        raise ValueError("Artifact exceeds size cap")
    return None


def scan_structured_artifact_prefix(
    text: str,
    heading: str,
    *,
    max_chars: int = DEFAULT_ARTIFACT_MAX_CHARS,
) -> ScannedArtifact | None:
    """Locate a transport-complete object after one exact heading and marker."""
    match = re.match(
        rf"^\s*### {re.escape(heading)}[^\S\r\n]*\r?\n"
        r"[^\S\r\n]*Artifact:[^\S\r\n]*",
        text,
    )
    if match is None:
        return None
    scanned = scan_artifact_object_prefix(
        text[match.end():],
        marker=None,
        max_chars=max_chars,
    )
    if scanned is None:
        return None
    return ScannedArtifact(
        scanned.json_text,
        match.end() + scanned.start,
        match.end() + scanned.end,
    )


def scan_single_artifact_object(
    text: str,
    *,
    marker: str | None = "Artifact:",
    max_chars: int = DEFAULT_ARTIFACT_MAX_CHARS,
) -> ScannedArtifact:
    """Locate exactly one bounded top-level JSON object after ``marker``.

    The scanner is string/escape aware, so braces in prose or LaTeX strings
    do not affect balancing. Any non-whitespace after the object is rejected;
    this prevents a second artifact or trailing text from changing meaning.
    """
    if marker is not None:
        marker_index = text.find(marker)
        if marker_index < 0:
            raise ValueError("missing Artifact marker")
        if text.find(marker, marker_index + len(marker)) >= 0:
            raise ValueError("multiple Artifact markers")
    scanned = scan_artifact_object_prefix(
        text,
        marker=marker,
        max_chars=max_chars,
    )
    if scanned is None:
        candidate_start = 0
        if marker is not None:
            candidate_start = text.find(marker) + len(marker)
        candidate = text[candidate_start:].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(json_syntax_diagnostic(candidate, exc)) from exc
        raise ValueError("transport-incomplete Artifact JSON")
    if text[scanned.end:].strip():
        raise ValueError("trailing text or second Artifact is forbidden")
    return scanned


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


class StructuredResponseBudgetTooSmall(SemanticUnitTooLarge):
    status = "STRUCTURED_RESPONSE_BUDGET_TOO_SMALL"

    def __init__(
        self,
        role: str,
        *,
        retained_input_tokens: int,
        available_tokens: int,
        required_tokens: int,
        max_retained_tokens: int,
        control_reserve_tokens: int,
    ) -> None:
        self.role = role
        self.retained_input_tokens = int(retained_input_tokens)
        self.available_tokens = int(available_tokens)
        self.required_tokens = int(required_tokens)
        self.max_retained_tokens = int(max_retained_tokens)
        self.control_reserve_tokens = int(control_reserve_tokens)
        self.compaction_tokens_required = max(
            0,
            self.required_tokens - self.available_tokens,
        )
        ValueError.__init__(
            self,
            f"{self.status}: {role} retained_input={retained_input_tokens}, "
            f"output_available={available_tokens}, "
            f"minimum_complete_schema={required_tokens}, "
            f"control_reserve={control_reserve_tokens}, "
            f"max_retained={max_retained_tokens}; compact whole input fields "
            f"by at least {self.compaction_tokens_required} tokens (semantic "
            "units must not be truncated)",
        )


def repair_json_backslashes(value: str) -> str:
    """Make literal model-emitted backslash runs valid JSON losslessly."""
    repaired = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            repaired.append(value[index])
            index += 1
            continue
        end = index
        while end < len(value) and value[end] == "\\":
            end += 1
        count = end - index
        next_text = value[end:]
        valid_single_escape = (
            next_text.startswith('"')
            or next_text.startswith("/")
            or (
                next_text.startswith("u")
                and len(next_text) >= 5
                and all(char in "0123456789abcdefABCDEF" for char in next_text[1:5])
            )
        )
        if count % 2 and not valid_single_escape:
            count += 1
        repaired.append("\\" * count)
        index = end
    return "".join(repaired)


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


def structured_output_cap(
    *,
    role: str,
    max_retained_tokens: int,
    retained_input_tokens: int,
    minimum_output_tokens: int,
    configured_output_tokens: int | None,
    control_reserve_tokens: int = 32,
) -> int:
    """Budget a complete structured response from actual retained capacity."""
    available = (
        int(max_retained_tokens)
        - int(retained_input_tokens)
        - int(control_reserve_tokens)
    )
    configured = (
        available
        if configured_output_tokens is None or configured_output_tokens <= 0
        else min(available, int(configured_output_tokens))
    )
    if available < minimum_output_tokens or configured < minimum_output_tokens:
        raise StructuredResponseBudgetTooSmall(
            role,
            retained_input_tokens=retained_input_tokens,
            available_tokens=max(0, configured),
            required_tokens=minimum_output_tokens,
            max_retained_tokens=max_retained_tokens,
            control_reserve_tokens=control_reserve_tokens,
        )
    return configured


def structured_role_minimum_output_tokens(role: str) -> int:
    """Return the reserved tokens for one complete compact role artifact."""
    try:
        return STRUCTURED_ROLE_MINIMUM_OUTPUT_TOKENS[str(role)]
    except KeyError as exc:
        raise ValueError(f"unknown structured role: {role}") from exc
