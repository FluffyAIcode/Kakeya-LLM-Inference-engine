"""Crash-safe deterministic Typed IR and Lean gate pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from autoresearch.prefill.math_ir import (
    GateStatus,
    LeanCompilation,
    MathIR,
    TypedIRError,
    ValidatedMathIR,
    build_lean_declaration,
    parse_math_ir,
    registry_hash,
    validate_math_ir,
    verify_proposition_equivalence,
)


HOST_COMPILER_VERSION = 2


@dataclass(frozen=True)
class GateEvidence:
    stage: str
    status: str
    input_hash: str
    output_hash: str = ""
    code: str = ""
    message: str = ""
    owner: str = "host"
    created_at: float = 0.0


@dataclass(frozen=True)
class HostGateResult:
    ok: bool
    math_ir: MathIR | None
    validated_ir: ValidatedMathIR | None
    compilation: LeanCompilation | None
    evidence: tuple[GateEvidence, ...]
    failure_status: str = ""
    backjump_owner: str = ""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: asdict(item),
    ).encode()).hexdigest()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _classify_elaboration_failure(output: str) -> tuple[GateStatus, str, str]:
    lowered = str(output).casefold()
    if any(marker in lowered for marker in (
        "unknown identifier", "unknown constant", "unknown namespace",
        "failed to synthesize", "declaration uses 'sorry'",
    )):
        return (
            GateStatus.SEMANTIC_BACKJUMP,
            "MISSING_DEFINITION_ENVIRONMENT",
            "decomposer",
        )
    if any(marker in lowered for marker in (
        "unexpected token", "invalid syntax", "parser", "lexer",
        "type mismatch", "application type mismatch",
    )):
        return GateStatus.INTEGRATION_BLOCKED, "HOST_LEAN_GENERATION_DEFECT", "host"
    return GateStatus.INTEGRATION_BLOCKED, "LEAN_ENVIRONMENT_FAILURE", "host"


def run_host_gates(
    math_ir_lines: Iterable[str],
    *,
    project_root: Path,
    cache_dir: Path,
    lean_validator: Callable,
) -> HostGateResult:
    """Run all deterministic gates once for a content-addressed input."""
    normalized_lines = tuple(str(item).strip() for item in math_ir_lines if str(item).strip())
    input_hash = _digest({
        "compiler_version": HOST_COMPILER_VERSION,
        "registry_hash": registry_hash(),
        "lines": normalized_lines,
    })
    cache_path = Path(cache_dir).expanduser() / f"{input_hash}.json"
    evidence: list[GateEvidence] = []
    math_ir: MathIR | None = None
    validated: ValidatedMathIR | None = None
    compilation: LeanCompilation | None = None

    def passed(stage: str, output_hash: str) -> None:
        evidence.append(GateEvidence(
            stage, GateStatus.PASSED.value, input_hash, output_hash,
            created_at=time.time(),
        ))

    def failed(stage: str, error: TypedIRError) -> HostGateResult:
        evidence.append(GateEvidence(
            stage,
            error.status.value,
            input_hash,
            code=error.code,
            message=str(error),
            owner=error.owner,
            created_at=time.time(),
        ))
        _write_atomic(cache_path, {
            "schema_version": HOST_COMPILER_VERSION,
            "input_hash": input_hash,
            "registry_hash": registry_hash(),
            "ok": False,
            "evidence": [asdict(item) for item in evidence],
        })
        return HostGateResult(
            False, math_ir, validated, compilation, tuple(evidence),
            error.status.value, error.owner,
        )

    try:
        math_ir = parse_math_ir(
            normalized_lines,
            theorem_id=f"typed_{input_hash[:16]}",
        )
        passed("TYPED_TRANSPORT_VALIDATION", _digest(asdict(math_ir)))
    except TypedIRError as exc:
        return failed("TYPED_TRANSPORT_VALIDATION", exc)
    try:
        validated = validate_math_ir(math_ir)
        passed("TYPED_MATH_IR_STRUCTURAL_VALIDATION", validated.content_hash)
        passed("SYMBOL_TYPE_OPERATOR_RESOLUTION", registry_hash())
    except TypedIRError as exc:
        return failed("TYPED_MATH_IR_STRUCTURAL_VALIDATION", exc)
    try:
        compilation = build_lean_declaration(validated)
        passed("LEAN_AST_SOURCE_GENERATION", compilation.declaration_hash)
    except Exception as exc:
        return failed("LEAN_AST_SOURCE_GENERATION", TypedIRError(
            "HOST_AST_BUILDER_EXCEPTION",
            f"{type(exc).__name__}: {exc}",
            owner="host",
            status=GateStatus.INTEGRATION_BLOCKED,
        ))
    elaborated = lean_validator(
        compilation.declaration_source,
        project_root=Path(project_root),
    )
    if not elaborated.ok:
        status, code, owner = _classify_elaboration_failure(
            elaborated.error or elaborated.output,
        )
        return failed("LEAN_ELABORATION", TypedIRError(
            code,
            elaborated.error or elaborated.output,
            owner=owner,
            status=status,
        ))
    passed("LEAN_ELABORATION", elaborated.signature_hash)
    try:
        if elaborated.signature_hash != compilation.declaration_hash:
            raise TypedIRError(
                "ELABORATED_DECLARATION_HASH_MISMATCH",
                "Lean elaborated a declaration different from the host AST",
                owner="host",
                status=GateStatus.INTEGRATION_BLOCKED,
            )
        verify_proposition_equivalence(validated, compilation)
        passed("PROPOSITION_HASH_EQUIVALENCE", compilation.proposition_hash)
    except TypedIRError as exc:
        return failed("PROPOSITION_HASH_EQUIVALENCE", exc)
    payload = {
        "schema_version": HOST_COMPILER_VERSION,
        "input_hash": input_hash,
        "registry_hash": registry_hash(),
        "ok": True,
        "math_ir": asdict(math_ir),
        "compilation": asdict(compilation),
        "evidence": [asdict(item) for item in evidence],
    }
    _write_atomic(cache_path, payload)
    return HostGateResult(
        True, math_ir, validated, compilation, tuple(evidence),
    )
