"""Fail-closed Lean theorem-signature gate for proof obligations."""
from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


_SIGNATURE_BLOCK = re.compile(
    r"^### LEAN_SIGNATURE(?:\s+(\S+))?\s*$"
    r"\s*```lean\s*(?P<source>.*?)```",
    re.MULTILINE | re.DOTALL,
)
_FORBIDDEN = re.compile(
    r"(^|\s)(?:import|axiom|opaque|unsafe|macro|syntax|elab|run_cmd|run_tac|"
    r"set_option|def|abbrev|instance|structure|inductive|class|namespace|"
    r"section|variable|open|attribute)\b|#(?:eval|check|print|reduce)|"
    r"\b(?:IO|System|FilePath)\b",
    re.MULTILINE,
)


@dataclass(frozen=True)
class LeanSignatureResult:
    source: str
    signature_hash: str
    ok: bool
    status: str = "FORMALIZED"
    error: str = ""
    attempts: int = 1
    elapsed_s: float = 0.0
    output: str = ""


@dataclass(frozen=True)
class _LeanRun:
    returncode: int | None
    timed_out: bool
    elapsed_s: float
    output: str


def extract_lean_signature_blocks(text: str) -> list[tuple[str, str]]:
    return [
        ((match.group(1) or "").strip(), match.group("source").strip())
        for match in _SIGNATURE_BLOCK.finditer(text)
    ]


def _signature_only(source: str) -> str:
    match = re.search(r"\s*:=\s*by\b", source)
    return source[:match.start()].strip() if match else source.strip()


def lean_theorem_signature_hash(source: str) -> str:
    signature = " ".join(_signature_only(source).split())
    return hashlib.sha256(signature.encode()).hexdigest() if signature else ""


def _run_lean(
    content: str,
    *,
    project_root: Path,
    timeout_s: float,
) -> _LeanRun:
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".lean",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(content)
        path = Path(handle.name)
    process = None
    try:
        process = subprocess.Popen(
            ["lake", "env", "lean", str(path)],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_s)
            return _LeanRun(
                process.returncode,
                False,
                time.monotonic() - started,
                output or "",
            )
        except subprocess.TimeoutExpired as exc:
            partial = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            remainder, _ = process.communicate()
            return _LeanRun(
                None,
                True,
                time.monotonic() - started,
                partial + (remainder or ""),
            )
    except OSError as exc:
        return _LeanRun(
            None,
            False,
            time.monotonic() - started,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        path.unlink(missing_ok=True)


def warm_lean_environment(
    project_root: Path,
    *,
    timeout_s: float = 120.0,
) -> LeanSignatureResult:
    source = "theorem kakeyaLeanWarmup : True := by trivial"
    content = (
        "import KakeyaLeanGate\n\n"
        "set_option autoImplicit false\n\n"
        + source
        + "\n"
    )
    run = _run_lean(
        content,
        project_root=project_root,
        timeout_s=timeout_s,
    )
    if run.timed_out:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_TIMEOUT",
            error=f"Lean warmup timed out after {timeout_s:.1f}s",
            elapsed_s=run.elapsed_s,
            output=run.output,
        )
    if run.returncode != 0:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="ENVIRONMENT_FAILED",
            error=f"Lean warmup failed: {run.output[-2000:]}",
            elapsed_s=run.elapsed_s,
            output=run.output,
        )
    return LeanSignatureResult(
        source,
        "",
        True,
        status="ENVIRONMENT_READY",
        elapsed_s=run.elapsed_s,
        output=run.output,
    )


def validate_lean_signature(
    source: str,
    *,
    project_root: Path,
    timeout_s: float = 45.0,
    retry_timeout_s: float = 120.0,
) -> LeanSignatureResult:
    source = source.strip()
    if not source:
        return LeanSignatureResult(
            "", "", False, status="TYPECHECK_FAILED",
            error="empty Lean signature",
        )
    if len(source) > 12_000:
        return LeanSignatureResult(
            "", "", False, status="TYPECHECK_FAILED",
            error="Lean signature too large",
        )
    if _FORBIDDEN.search(source):
        return LeanSignatureResult(
            source,
            "",
            False,
            status="UNSAFE_REJECTED",
            error="forbidden Lean command in generated signature",
        )
    declarations = re.findall(r"^\s*theorem\s+([A-Za-z_][\w']*)", source, re.MULTILINE)
    if len(declarations) != 1:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_FAILED",
            error="expected exactly one theorem declaration",
        )
    if not re.search(r"\s*:=\s*by\b", source):
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_FAILED",
            error="theorem signature must end with `:= by` proof scaffold",
        )
    signature_hash = lean_theorem_signature_hash(source)
    content = (
        "import KakeyaLeanGate\n\n"
        "set_option autoImplicit false\n\n"
        + source
        + "\n"
    )
    first = _run_lean(
        content,
        project_root=project_root,
        timeout_s=timeout_s,
    )
    attempts = 1
    total_elapsed = first.elapsed_s
    output = first.output
    run = first
    if first.timed_out:
        warmup = warm_lean_environment(
            project_root,
            timeout_s=retry_timeout_s,
        )
        total_elapsed += warmup.elapsed_s
        output += warmup.output
        if not warmup.ok:
            return LeanSignatureResult(
                source,
                signature_hash,
                False,
                status=warmup.status,
                error=warmup.error,
                attempts=1,
                elapsed_s=total_elapsed,
                output=output,
            )
        run = _run_lean(
            content,
            project_root=project_root,
            timeout_s=retry_timeout_s,
        )
        attempts = 2
        total_elapsed += run.elapsed_s
        output += run.output
    if run.timed_out:
        return LeanSignatureResult(
            source,
            signature_hash,
            False,
            status="TYPECHECK_TIMEOUT",
            error=(
                f"Lean typecheck timed out after {attempts} attempts "
                f"({timeout_s:.1f}s/{retry_timeout_s:.1f}s)"
            ),
            attempts=attempts,
            elapsed_s=total_elapsed,
            output=output,
        )
    if run.returncode != 0:
        return LeanSignatureResult(
            source,
            signature_hash,
            False,
            status="TYPECHECK_FAILED",
            error=f"Lean typecheck failed: {run.output[-2000:]}",
            attempts=attempts,
            elapsed_s=total_elapsed,
            output=output,
        )
    return LeanSignatureResult(
        source,
        signature_hash,
        True,
        status="FORMALIZED",
        attempts=attempts,
        elapsed_s=total_elapsed,
        output=output,
    )


def validate_lean_proof(
    source: str,
    *,
    project_root: Path,
    timeout_s: float = 45.0,
) -> LeanSignatureResult:
    """Compile one complete theorem without sorry/admit or added axioms."""
    source = source.strip()
    if (
        not source
        or len(source) > 12_000
        or _FORBIDDEN.search(source)
        or re.search(r"\b(?:sorry|admit)\b", source)
    ):
        return LeanSignatureResult(
            source,
            "",
            False,
            status="UNSAFE_REJECTED",
            error="Lean proof is empty, unsafe, oversized, or incomplete",
        )
    declarations = re.findall(
        r"^\s*theorem\s+([A-Za-z_][\w']*)",
        source,
        re.MULTILINE,
    )
    if len(declarations) != 1 or not re.search(r"\s*:=\s*by\b", source):
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_FAILED",
            error="expected exactly one complete theorem declaration",
        )
    proof_hash = hashlib.sha256(source.encode()).hexdigest()
    run = _run_lean(
        "import KakeyaLeanGate\n\nset_option autoImplicit false\n\n"
        + source
        + "\n",
        project_root=project_root,
        timeout_s=timeout_s,
    )
    if run.timed_out:
        return LeanSignatureResult(
            source,
            proof_hash,
            False,
            status="TYPECHECK_TIMEOUT",
            error=f"Lean proof timed out after {timeout_s:.1f}s",
            elapsed_s=run.elapsed_s,
            output=run.output,
        )
    if run.returncode != 0:
        return LeanSignatureResult(
            source,
            proof_hash,
            False,
            status="TYPECHECK_FAILED",
            error=f"Lean proof failed: {run.output[-2000:]}",
            elapsed_s=run.elapsed_s,
            output=run.output,
        )
    return LeanSignatureResult(
        source,
        proof_hash,
        True,
        status="PROVED",
        elapsed_s=run.elapsed_s,
        output=run.output,
    )
