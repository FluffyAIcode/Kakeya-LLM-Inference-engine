"""Fail-closed, independently-auditable OProver theorem reconstruction.

The target proof body is never included in the prompt or persisted package.
OProver output remains untrusted until the exact project Lean executable accepts
it in a temporary source file.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol


class ReconstructionStatus(str, Enum):
    PROVIDER_ADAPTER_FAILED = "PROVIDER/ADAPTER_FAILED"
    NO_CANDIDATE = "NO_CANDIDATE"
    LEAN_REJECTED = "LEAN_REJECTED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"


_DECLARATION = re.compile(
    r"(?m)^(?:@\[.*\]\s*)*(?:(?:private|protected|noncomputable)\s+)*"
    r"(?:theorem|lemma|def|abbrev|example|instance|structure|class|inductive)\s+"
)
_FENCE = re.compile(r"```(?:lean4?|Lean4?)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ReconstructionPackage:
    schema_version: int
    theorem_id: str
    source_path: str
    namespace: str
    imports: tuple[str, ...]
    target_header: str
    preserved_context: str
    prompt_context: str
    retrieval_refs: tuple[str, ...]
    retrieval_context: tuple[str, ...]
    source_hash: str
    environment_hash: str
    theorem_hash: str
    dependency_prefix_hash: str
    original_proof_hash: str

    def prompt_source(self) -> str:
        return f"{self.prompt_context}{self.target_header}\n"

    def public_record(self) -> dict:
        body = asdict(self)
        # The original proof hash binds the source without exposing the body.
        return body


@dataclass(frozen=True)
class ProviderCandidate:
    text: str
    stop_reason: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ReconstructionProvider(Protocol):
    def generate(
        self, *, prompt: str, seed: int, max_tokens: int,
    ) -> ProviderCandidate: ...


@dataclass(frozen=True)
class LeanResult:
    accepted: bool
    output: str
    timed_out: bool = False


@dataclass(frozen=True)
class ReconstructionAttempt:
    index: int
    seed: int
    status: str
    raw_stop_reason: str
    raw_output_hash: str
    parser_result: str
    candidate_hashes: tuple[str, ...]
    selected_candidate_hash: str
    lean_output: str
    lean_timed_out: bool
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ReconstructionResult:
    status: str
    package_hash: str
    verified_candidate_hash: str
    attempts: tuple[ReconstructionAttempt, ...]


def build_reconstruction_package(
    *,
    source_path: Path,
    project_root: Path,
    theorem_id: str,
    environment_hash: str,
    retrieval_refs: tuple[str, ...] = (),
    retrieval_context: tuple[str, ...] = (),
    prompt_context_chars: int = 48_000,
) -> ReconstructionPackage:
    """Preserve all declarations before the target and redact only its body."""
    source_path = Path(source_path).resolve()
    project_root = Path(project_root).resolve()
    source = source_path.read_text(encoding="utf-8")
    short_name = theorem_id.rsplit(".", 1)[-1]
    target = re.search(
        r"(?m)^(?P<indent>[ \t]*)(?:(?:private|protected)\s+)?"
        rf"(?P<kind>theorem|lemma)\s+(?:{re.escape(theorem_id)}|"
        rf"{re.escape(short_name)})\b",
        source,
    )
    if target is None:
        raise ValueError("RECONSTRUCTION_TARGET_NOT_FOUND")
    if target.group("indent"):
        raise ValueError("RECONSTRUCTION_TARGET_MUST_BE_TOP_LEVEL")
    assignment = source.find(":=", target.start())
    if assignment < 0:
        raise ValueError("RECONSTRUCTION_TARGET_ASSIGNMENT_NOT_FOUND")
    next_declaration = _DECLARATION.search(source, assignment + 2)
    body_end = next_declaration.start() if next_declaration else len(source)
    target_header = source[target.start():assignment + 2].rstrip()
    original_body = source[assignment + 2:body_end]
    if not original_body.strip():
        raise ValueError("RECONSTRUCTION_TARGET_BODY_EMPTY")

    prefix = source[:target.start()]
    import_lines = tuple(re.findall(r"(?m)^import\s+.+$", prefix))
    namespaces = re.findall(r"(?m)^namespace\s+([A-Za-z0-9_'.]+)\s*$", prefix)
    namespace = ".".join(namespaces)
    if prompt_context_chars < 1:
        raise ValueError("PROMPT_CONTEXT_BUDGET_MUST_BE_POSITIVE")
    if len(prefix) <= prompt_context_chars:
        prompt_context = prefix
    else:
        cutoff = len(prefix) - prompt_context_chars
        declaration = _DECLARATION.search(prefix, cutoff)
        context_start = declaration.start() if declaration else cutoff
        prompt_context = (
            "\n".join(import_lines)
            + "\n\n"
            + prefix[context_start:].lstrip()
        )
    relative = source_path.relative_to(project_root)
    dependency_prefix_hash = _sha256(prefix)
    theorem_hash = _sha256(target_header)
    package = ReconstructionPackage(
        schema_version=2,
        theorem_id=theorem_id,
        source_path=str(relative),
        namespace=namespace,
        imports=import_lines,
        target_header=target_header,
        preserved_context=prefix,
        prompt_context=prompt_context,
        retrieval_refs=tuple(retrieval_refs),
        retrieval_context=tuple(retrieval_context),
        source_hash=_sha256(source),
        environment_hash=environment_hash,
        theorem_hash=theorem_hash,
        dependency_prefix_hash=dependency_prefix_hash,
        original_proof_hash=_sha256(original_body),
    )
    if original_body.strip() in package.prompt_source():
        raise RuntimeError("TARGET_PROOF_LEAK_DETECTED")
    return package


def build_native_prompt(
    package: ReconstructionPackage,
    *,
    previous_attempt: str = "",
    compiler_feedback: str = "",
) -> str:
    """Use the prompt shape published with OProver, with bounded feedback."""
    retrieval = "\n".join(package.retrieval_context) or "(none available)"
    return (
        "**Current Task:**\n"
        "Complete the following Lean 4 code. Return a proof beginning with `by` "
        "in a Lean fence or as plain Lean.\n\n"
        f"```lean4\n{package.prompt_source()}```\n\n"
        "**Relevant retrieved declarations (not target proofs):**\n"
        f"{retrieval}\n\n"
        "Before producing the Lean 4 proof, provide a concise proof plan. "
        "Use the preserved imports, namespace, local definitions, and earlier "
        "certified dependency lemmas. Do not restate or modify the theorem.\n\n"
        f"**Previous Failed Attempt:**\n```lean4\n{previous_attempt}\n```\n\n"
        f"**Error Messages:**\n{compiler_feedback}\n"
    )


def extract_lean_candidates(text: str) -> tuple[str, ...]:
    """Extract proof terms from fenced and plain OProver responses."""
    clean = _THINK.sub("", text).strip()
    regions = [match.group(1).strip() for match in _FENCE.finditer(clean)]
    regions.append(_FENCE.sub("", clean).strip())
    options: list[str] = []
    for region in regions:
        if not region:
            continue
        if ":=" in region:
            options.append(region.rsplit(":=", 1)[1].strip())
        markers = tuple(match.start() for match in re.finditer(r"(?m)(?:^|\s)\bby\b", region))
        options.extend(region[index:].strip() for index in reversed(markers))
        if region.startswith(("exact ", "simpa", "simp", "aesop", "omega", "linarith")):
            options.append("by\n  " + region)
    normalized = []
    for option in options:
        option = option.strip()
        if option.startswith("by") and option not in normalized:
            normalized.append(option)
    return tuple(normalized)


def verify_candidate_with_project_lean(
    package: ReconstructionPackage,
    candidate: str,
    project_root: Path,
    *,
    timeout_seconds: int = 120,
) -> LeanResult:
    """Insert one candidate into one isolated theorem and run project Lean."""
    with tempfile.TemporaryDirectory(prefix="kakeya-reconstruct-") as raw:
        source = Path(raw) / "Reconstruction.lean"
        source.write_text(
            f"{package.preserved_context}{package.target_header}\n{candidate}\n",
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                ["lake", "env", "lean", str(source)],
                cwd=Path(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = str(exc.stdout or exc.stderr or "")
            return LeanResult(False, output[-8000:], True)
        return LeanResult(result.returncode == 0, result.stdout[-8000:], False)


def run_pass_at_k(
    *,
    package: ReconstructionPackage,
    provider: ReconstructionProvider,
    project_root: Path,
    k: int,
    base_seed: int = 0,
    max_tokens: int = 4096,
    lean_verify: Callable[
        [ReconstructionPackage, str, Path], LeanResult
    ] | None = None,
) -> ReconstructionResult:
    if k < 1:
        raise ValueError("PASS_AT_K_REQUIRES_POSITIVE_K")
    verifier = lean_verify or (
        lambda pkg, candidate, root: verify_candidate_with_project_lean(
            pkg, candidate, root,
        )
    )
    attempts: list[ReconstructionAttempt] = []
    previous = ""
    feedback = ""
    package_hash = _sha256(json.dumps(
        package.public_record(), sort_keys=True, separators=(",", ":"),
    ))
    for index in range(k):
        seed = base_seed + index
        prompt = build_native_prompt(
            package, previous_attempt=previous, compiler_feedback=feedback,
        )
        prompt_hash = _sha256(prompt)
        try:
            generated = provider.generate(
                prompt=prompt, seed=seed, max_tokens=max_tokens,
            )
        except Exception as exc:
            attempts.append(ReconstructionAttempt(
                index, seed, ReconstructionStatus.PROVIDER_ADAPTER_FAILED.value,
                "provider_exception", "", "provider_failed", (), "", 
                f"{type(exc).__name__}:{str(exc)[:500]}", False, prompt_hash, 0, 0,
            ))
            return ReconstructionResult(
                ReconstructionStatus.PROVIDER_ADAPTER_FAILED.value,
                package_hash, "", tuple(attempts),
            )
        options = extract_lean_candidates(generated.text)
        hashes = tuple(_sha256(item) for item in options)
        if not options:
            attempts.append(ReconstructionAttempt(
                index, seed, ReconstructionStatus.NO_CANDIDATE.value,
                generated.stop_reason, _sha256(generated.text), "no_lean_proof",
                (), "", "", False, prompt_hash, generated.prompt_tokens,
                generated.completion_tokens,
            ))
            previous = generated.text[-4000:]
            feedback = "No Lean proof beginning with `by` was extracted."
            continue
        selected_hash = ""
        last = LeanResult(False, "")
        for option, candidate_hash in zip(options, hashes):
            last = verifier(package, option, Path(project_root))
            if last.accepted:
                selected_hash = candidate_hash
                attempts.append(ReconstructionAttempt(
                    index, seed, ReconstructionStatus.INDEPENDENTLY_VERIFIED.value,
                    generated.stop_reason, _sha256(generated.text),
                    f"extracted:{len(options)}", hashes, selected_hash, "",
                    False, prompt_hash, generated.prompt_tokens,
                    generated.completion_tokens,
                ))
                return ReconstructionResult(
                    ReconstructionStatus.INDEPENDENTLY_VERIFIED.value,
                    package_hash, selected_hash, tuple(attempts),
                )
        attempts.append(ReconstructionAttempt(
            index, seed, ReconstructionStatus.LEAN_REJECTED.value,
            generated.stop_reason, _sha256(generated.text),
            f"extracted:{len(options)}", hashes, "", last.output,
            last.timed_out, prompt_hash, generated.prompt_tokens,
            generated.completion_tokens,
        ))
        previous = options[0][-4000:]
        feedback = last.output[-4000:] or "Lean rejected the candidate."
    return ReconstructionResult(
        ReconstructionStatus.SEARCH_EXHAUSTED.value,
        package_hash, "", tuple(attempts),
    )
