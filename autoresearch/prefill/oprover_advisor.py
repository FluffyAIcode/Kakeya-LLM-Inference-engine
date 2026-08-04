"""Untrusted OProver candidates with isolated Lean verification."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol


PROOF_ADVISOR_UNAVAILABLE = "PROOF_ADVISOR_UNAVAILABLE"
OFFICIAL_MODEL_ID = "m-a-p/OProver-8B"
OFFICIAL_REVISION = "cd9ffd383b584d95bf00e04b88b35b05928b211c"
OFFICIAL_LICENSE = "apache-2.0"
OFFICIAL_ARCHITECTURE = "Qwen3ForCausalLM"
OFFICIAL_PARAMETER_COUNT = 8_190_735_360
OFFICIAL_SOURCE_BYTES = 16_393_509_991


class RetrievalProvider(Protocol):
    def retrieve(
        self, *, theorem_id: str, goal: str, limit: int,
    ) -> tuple[str, ...]: ...


class UnavailableOProofs:
    def retrieve(
        self, *, theorem_id: str, goal: str, limit: int,
    ) -> tuple[str, ...]:
        raise RuntimeError("OPROOFS_RETRIEVAL_UNAVAILABLE")


@dataclass(frozen=True)
class OProverConfig:
    model_path: Path
    model_id: str = OFFICIAL_MODEL_ID
    revision: str = OFFICIAL_REVISION
    tokenizer_id: str = OFFICIAL_MODEL_ID
    quantization: str = "q5"
    source_checksum_manifest: str = ""
    candidate_count: int = 4

    def validate(self) -> None:
        if self.model_id != OFFICIAL_MODEL_ID:
            raise ValueError("UNVERIFIED_OPROVER_MODEL_ID")
        if self.revision != OFFICIAL_REVISION:
            raise ValueError("UNPINNED_OPROVER_REVISION")
        if self.quantization not in {"q4", "q5"}:
            raise ValueError("OPROVER_QUANTIZATION_MUST_BE_Q4_OR_Q5")
        if self.candidate_count < 1:
            raise ValueError("OPROVER_REQUIRES_AT_LEAST_ONE_CANDIDATE")
        if not self.source_checksum_manifest.startswith("sha256:"):
            raise ValueError("OPROVER_SOURCE_MANIFEST_HASH_REQUIRED")

    def verify_model_bundle(self) -> None:
        self.validate()
        try:
            config = json.loads(
                (self.model_path / "config.json").read_text(encoding="utf-8")
            )
            manifest_path = self.model_path / "source-manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OPROVER_MODEL_BUNDLE_INVALID") from exc
        expected_bits = 5 if self.quantization == "q5" else 4
        quantization = config.get("quantization", {})
        if int(quantization.get("bits", 0) or 0) != expected_bits:
            raise RuntimeError("OPROVER_QUANTIZATION_METADATA_MISMATCH")
        if manifest.get("repo_id") != OFFICIAL_MODEL_ID:
            raise RuntimeError("OPROVER_SOURCE_MODEL_MISMATCH")
        if manifest.get("revision") != OFFICIAL_REVISION:
            raise RuntimeError("OPROVER_SOURCE_REVISION_MISMATCH")
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        if self.source_checksum_manifest != f"sha256:{digest}":
            raise RuntimeError("OPROVER_SOURCE_MANIFEST_HASH_MISMATCH")
        if not tuple(self.model_path.glob("*.safetensors")):
            raise RuntimeError("OPROVER_MODEL_WEIGHTS_MISSING")


@dataclass(frozen=True)
class ProofAdvice:
    advice_id: str
    theorem_id: str
    proposition_hash: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    quantization: str
    candidate_index: int
    verified_lean_hash: str
    action_ids: tuple[str, ...]
    retrieval_refs: tuple[str, ...]
    artifact_path: str


class OProverRuntime(Protocol):
    def generate(
        self,
        *,
        theorem_id: str,
        lean_goal: str,
        local_context: tuple[str, ...],
        retrieval_refs: tuple[str, ...],
        candidate_count: int,
    ) -> Iterable[str]: ...


class OProverProofAdvisor:
    def __init__(
        self,
        *,
        config: OProverConfig,
        runtime: OProverRuntime,
        artifact_dir: Path,
        lean_verify: Callable[[str, Path], tuple[bool, tuple[str, ...]]],
        retrieval: RetrievalProvider | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.runtime = runtime
        self.artifact_dir = Path(artifact_dir)
        self.lean_verify = lean_verify
        self.retrieval = retrieval or UnavailableOProofs()

    def advise(
        self,
        *,
        theorem_id: str,
        proposition_hash: str,
        lean_goal: str,
        local_context: tuple[str, ...],
        theorem_card_refs: tuple[str, ...] = (),
    ) -> tuple[ProofAdvice, ...]:
        if not theorem_id or not proposition_hash or not lean_goal.strip():
            raise ValueError("PROOF_ADVISOR_REFUSES_UNELABORATED_TARGET")
        if not self.config.model_path.is_dir():
            raise RuntimeError(PROOF_ADVISOR_UNAVAILABLE)
        self.config.verify_model_bundle()
        refs = list(theorem_card_refs)
        try:
            refs.extend(self.retrieval.retrieve(
                theorem_id=theorem_id,
                goal=lean_goal,
                limit=8,
            ))
        except RuntimeError as exc:
            if str(exc) != "OPROOFS_RETRIEVAL_UNAVAILABLE":
                raise
        candidates = tuple(self.runtime.generate(
            theorem_id=theorem_id,
            lean_goal=lean_goal,
            local_context=local_context,
            retrieval_refs=tuple(refs),
            candidate_count=self.config.candidate_count,
        ))
        if len(candidates) > self.config.candidate_count:
            candidates = candidates[:self.config.candidate_count]
        verified: list[ProofAdvice] = []
        for index, candidate in enumerate(candidates):
            with tempfile.TemporaryDirectory(
                prefix="kakeya-oprover-lean-",
            ) as scratch:
                accepted, action_ids = self.lean_verify(
                    candidate, Path(scratch),
                )
            if not accepted:
                continue
            lean_hash = hashlib.sha256(candidate.encode()).hexdigest()
            body = {
                "schema_version": 1,
                "theorem_id": theorem_id,
                "proposition_hash": proposition_hash,
                "model_id": self.config.model_id,
                "model_revision": self.config.revision,
                "tokenizer_id": self.config.tokenizer_id,
                "quantization": self.config.quantization,
                "candidate_index": index,
                "verified_lean_hash": lean_hash,
                "action_ids": tuple(action_ids),
                "retrieval_refs": tuple(refs),
            }
            digest = hashlib.sha256(json.dumps(
                body, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            self.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.artifact_dir / f"{digest}.json"
            advice = ProofAdvice(
                advice_id="PA-" + digest[:20],
                artifact_path=str(path),
                **{key: value for key, value in body.items()
                   if key != "schema_version"},
            )
            # Candidate source is intentionally absent; only its verified hash
            # and host-registered action IDs survive the isolation boundary.
            path.write_text(
                json.dumps(asdict(advice), sort_keys=True),
                encoding="utf-8",
            )
            os.chmod(path, 0o600)
            verified.append(advice)
        return tuple(verified)
