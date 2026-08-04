"""Fail-closed, independently-auditable OProver theorem reconstruction.

The target proof body is never included in the prompt or persisted package.
OProver output remains untrusted until the exact project Lean executable accepts
it in a temporary source file.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Protocol


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
    verified_artifact_hash: str = ""
    verified_artifact_path: str = ""


@dataclass(frozen=True)
class VerifiedProofArtifact:
    schema_version: int
    artifact_hash: str
    theorem_id: str
    theorem_statement: str
    theorem_hash: str
    source_hash: str
    source_path: str
    proof_body: str
    proof_body_hash: str
    proof_encoding: str
    proof_byte_length: int
    reconstructed_source_hash: str
    imports: tuple[str, ...]
    namespace: str
    prompt_context_hash: str
    retrieval_refs_hash: str
    retrieval_context_hash: str
    dependency_prefix_hash: str
    environment_hash: str
    toolchain_hash: str
    model_id: str
    model_revision: str
    quantization: str
    attempt_index: int
    seed: int
    candidate_hash: str
    lean_accepted: bool
    lean_output: str
    lean_output_hash: str
    lean_timed_out: bool
    package_hash: str
    created_at: float
    provenance: Mapping[str, str]


class VerifiedProofArtifactError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def _toolchain_hash(project_root: Path) -> str:
    root = Path(project_root)
    files = {}
    for name in ("lean-toolchain", "lake-manifest.json", "lakefile.lean"):
        path = root / name
        files[name] = _sha256(path.read_bytes()) if path.is_file() else ""
    return _sha256(_canonical(files))


def _atomic_write_private(
    path: Path, encoded: bytes, *, immutable: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
    )
    try:
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if immutable:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _artifact_identity(payload: Mapping[str, object]) -> str:
    identity = {
        key: value for key, value in payload.items()
        if key not in {"artifact_hash", "created_at"}
    }
    return _sha256(_canonical(identity))


class VerifiedProofStore:
    """Private, immutable content-addressed store for Lean-accepted proofs."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def persist(
        self,
        *,
        package: ReconstructionPackage,
        candidate: str,
        lean_result: LeanResult,
        project_root: Path,
        package_hash: str,
        attempt_index: int,
        seed: int,
        model_id: str,
        model_revision: str,
        quantization: str,
        provenance: Mapping[str, str] | None = None,
        created_at: float | None = None,
    ) -> VerifiedProofArtifact:
        if not lean_result.accepted or lean_result.timed_out:
            raise VerifiedProofArtifactError(
                "VERIFIED_PROOF_REQUIRES_ACCEPTED_LEAN_RESULT"
            )
        candidate = candidate.strip()
        if not candidate.startswith("by"):
            raise VerifiedProofArtifactError("VERIFIED_PROOF_BODY_INVALID")
        reconstructed = (
            f"{package.preserved_context}{package.target_header}\n"
            f"{candidate}\n"
        )
        body = {
            "schema_version": 1,
            "artifact_hash": "",
            "theorem_id": package.theorem_id,
            "theorem_statement": package.target_header,
            "theorem_hash": package.theorem_hash,
            "source_hash": package.source_hash,
            "source_path": package.source_path,
            "proof_body": candidate,
            "proof_body_hash": _sha256(candidate),
            "proof_encoding": "utf-8",
            "proof_byte_length": len(candidate.encode()),
            "reconstructed_source_hash": _sha256(reconstructed),
            "imports": list(package.imports),
            "namespace": package.namespace,
            "prompt_context_hash": _sha256(package.prompt_context),
            "retrieval_refs_hash": _sha256(_canonical(package.retrieval_refs)),
            "retrieval_context_hash": _sha256(
                _canonical(package.retrieval_context)
            ),
            "dependency_prefix_hash": package.dependency_prefix_hash,
            "environment_hash": package.environment_hash,
            "toolchain_hash": _toolchain_hash(Path(project_root)),
            "model_id": model_id,
            "model_revision": model_revision,
            "quantization": quantization,
            "attempt_index": int(attempt_index),
            "seed": int(seed),
            "candidate_hash": _sha256(candidate),
            "lean_accepted": True,
            "lean_output": lean_result.output,
            "lean_output_hash": _sha256(lean_result.output),
            "lean_timed_out": False,
            "package_hash": package_hash,
            "created_at": float(created_at if created_at is not None else time.time()),
            "provenance": dict(provenance or {}),
        }
        digest = _artifact_identity(body)
        body["artifact_hash"] = digest
        path = self.root / digest[:2] / f"{digest}.json"
        if path.exists():
            return self.load(path)
        try:
            _atomic_write_private(path, _canonical(body), immutable=True)
        except FileExistsError:
            # A concurrent writer committed the same content address first.
            pass
        loaded = self.load(path)
        if loaded.candidate_hash != _sha256(candidate):
            raise VerifiedProofArtifactError("VERIFIED_PROOF_WRITE_MISMATCH")
        return loaded

    def load(self, reference: str | Path) -> VerifiedProofArtifact:
        path = Path(reference)
        if not path.is_absolute():
            digest = str(reference)
            path = self.root / digest[:2] / f"{digest}.json"
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VerifiedProofArtifactError(
                "VERIFIED_PROOF_ARTIFACT_UNREADABLE"
            ) from exc
        expected = str(body.get("artifact_hash", ""))
        if not expected or _artifact_identity(body) != expected:
            raise VerifiedProofArtifactError(
                "VERIFIED_PROOF_ARTIFACT_HASH_MISMATCH"
            )
        proof = str(body.get("proof_body", ""))
        if (
            _sha256(proof) != body.get("proof_body_hash")
            or _sha256(proof) != body.get("candidate_hash")
            or body.get("proof_encoding") != "utf-8"
            or len(proof.encode()) != body.get("proof_byte_length")
        ):
            raise VerifiedProofArtifactError("VERIFIED_PROOF_BODY_HASH_MISMATCH")
        if not body.get("lean_accepted") or body.get("lean_timed_out"):
            raise VerifiedProofArtifactError("VERIFIED_PROOF_LEAN_STATUS_INVALID")
        return VerifiedProofArtifact(
            **{
                **body,
                "imports": tuple(body.get("imports", ())),
                "provenance": dict(body.get("provenance", {})),
            }
        )


def build_reconstruction_package(
    *,
    source_path: Path,
    project_root: Path,
    theorem_id: str,
    environment_hash: str,
    retrieval_refs: tuple[str, ...] = (),
    retrieval_context: tuple[str, ...] = (),
    prompt_context_chars: int = 12_000,
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


def recompile_verified_artifact(
    *,
    store: VerifiedProofStore,
    artifact_reference: str | Path,
    package: ReconstructionPackage,
    project_root: Path,
) -> tuple[VerifiedProofArtifact, LeanResult]:
    artifact = store.load(artifact_reference)
    checks = {
        "theorem_id": package.theorem_id,
        "theorem_statement": package.target_header,
        "theorem_hash": package.theorem_hash,
        "source_hash": package.source_hash,
        "source_path": package.source_path,
        "dependency_prefix_hash": package.dependency_prefix_hash,
        "environment_hash": package.environment_hash,
        "toolchain_hash": _toolchain_hash(Path(project_root)),
        "imports": package.imports,
        "namespace": package.namespace,
    }
    for name, expected in checks.items():
        if getattr(artifact, name) != expected:
            raise VerifiedProofArtifactError(
                f"VERIFIED_PROOF_{name.upper()}_MISMATCH"
            )
    if artifact.candidate_hash != _sha256(artifact.proof_body):
        raise VerifiedProofArtifactError("VERIFIED_PROOF_CANDIDATE_HASH_MISMATCH")
    reconstructed = (
        f"{package.preserved_context}{package.target_header}\n"
        f"{artifact.proof_body}\n"
    )
    if artifact.reconstructed_source_hash != _sha256(reconstructed):
        raise VerifiedProofArtifactError(
            "VERIFIED_PROOF_RECONSTRUCTED_SOURCE_HASH_MISMATCH"
        )
    result = verify_candidate_with_project_lean(
        package, artifact.proof_body, Path(project_root),
    )
    if not result.accepted or result.timed_out:
        raise VerifiedProofArtifactError("VERIFIED_PROOF_RECOMPILE_FAILED")
    return artifact, result


def integrate_verified_artifact(
    *,
    store: VerifiedProofStore,
    artifact_reference: str | Path,
    source_path: Path,
    project_root: Path,
    theorem_id: str,
    environment_hash: str,
) -> VerifiedProofArtifact:
    source_path = Path(source_path).resolve()
    project_root = Path(project_root).resolve()
    source_mode = source_path.stat().st_mode & 0o777
    package = build_reconstruction_package(
        source_path=source_path,
        project_root=project_root,
        theorem_id=theorem_id,
        environment_hash=environment_hash,
    )
    artifact, _ = recompile_verified_artifact(
        store=store,
        artifact_reference=artifact_reference,
        package=package,
        project_root=project_root,
    )
    source = source_path.read_text(encoding="utf-8")
    short_name = theorem_id.rsplit(".", 1)[-1]
    target = re.search(
        r"(?m)^(?P<indent>[ \t]*)(?:(?:private|protected)\s+)?"
        rf"(?P<kind>theorem|lemma)\s+(?:{re.escape(theorem_id)}|"
        rf"{re.escape(short_name)})\b",
        source,
    )
    if target is None or target.group("indent"):
        raise VerifiedProofArtifactError("VERIFIED_PROOF_INTEGRATION_TARGET_INVALID")
    assignment = source.find(":=", target.start())
    if assignment < 0:
        raise VerifiedProofArtifactError(
            "VERIFIED_PROOF_INTEGRATION_ASSIGNMENT_MISSING"
        )
    next_declaration = _DECLARATION.search(source, assignment + 2)
    body_end = next_declaration.start() if next_declaration else len(source)
    integrated = (
        source[:assignment + 2]
        + "\n"
        + artifact.proof_body.rstrip()
        + "\n\n"
        + source[body_end:].lstrip("\n")
    )
    _atomic_write_private(source_path, integrated.encode())
    os.chmod(source_path, source_mode)
    return artifact


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
    artifact_store: VerifiedProofStore | None = None,
    model_id: str = "",
    model_revision: str = "",
    quantization: str = "",
    provenance: Mapping[str, str] | None = None,
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
                verified_artifact = None
                if artifact_store is not None:
                    verified_artifact = artifact_store.persist(
                        package=package,
                        candidate=option,
                        lean_result=last,
                        project_root=Path(project_root),
                        package_hash=package_hash,
                        attempt_index=index,
                        seed=seed,
                        model_id=model_id,
                        model_revision=model_revision,
                        quantization=quantization,
                        provenance=provenance,
                    )
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
                    (
                        verified_artifact.artifact_hash
                        if verified_artifact is not None else ""
                    ),
                    (
                        str(
                            artifact_store.root
                            / verified_artifact.artifact_hash[:2]
                            / f"{verified_artifact.artifact_hash}.json"
                        )
                        if verified_artifact is not None else ""
                    ),
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
