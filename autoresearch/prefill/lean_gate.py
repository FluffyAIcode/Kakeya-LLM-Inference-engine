"""Fail-closed Lean theorem-signature gate for proof obligations."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
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
    error: str = ""


def extract_lean_signature_blocks(text: str) -> list[tuple[str, str]]:
    return [
        ((match.group(1) or "").strip(), match.group("source").strip())
        for match in _SIGNATURE_BLOCK.finditer(text)
    ]


def _signature_only(source: str) -> str:
    match = re.search(r"\s*:=\s*by\b", source)
    return source[:match.start()].strip() if match else source.strip()


def validate_lean_signature(
    source: str,
    *,
    project_root: Path,
    timeout_s: float = 30.0,
) -> LeanSignatureResult:
    source = source.strip()
    if not source:
        return LeanSignatureResult("", "", False, "empty Lean signature")
    if len(source) > 12_000:
        return LeanSignatureResult("", "", False, "Lean signature too large")
    if _FORBIDDEN.search(source):
        return LeanSignatureResult(
            source,
            "",
            False,
            "forbidden Lean command in generated signature",
        )
    declarations = re.findall(r"^\s*theorem\s+([A-Za-z_][\w']*)", source, re.MULTILINE)
    if len(declarations) != 1:
        return LeanSignatureResult(
            source,
            "",
            False,
            "expected exactly one theorem declaration",
        )
    if not re.search(r"\s*:=\s*by\b", source):
        return LeanSignatureResult(
            source,
            "",
            False,
            "theorem signature must end with `:= by` proof scaffold",
        )
    signature = " ".join(_signature_only(source).split())
    signature_hash = hashlib.sha256(signature.encode()).hexdigest()
    content = (
        "import KakeyaLeanGate\n\n"
        "set_option autoImplicit false\n\n"
        + source
        + "\n"
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(content)
            path = Path(handle.name)
        completed = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LeanSignatureResult(
            source,
            signature_hash,
            False,
            f"Lean invocation failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if "path" in locals():
            path.unlink(missing_ok=True)
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout).strip()
        return LeanSignatureResult(
            source,
            signature_hash,
            False,
            f"Lean typecheck failed: {error[-2000:]}",
        )
    return LeanSignatureResult(source, signature_hash, True)
