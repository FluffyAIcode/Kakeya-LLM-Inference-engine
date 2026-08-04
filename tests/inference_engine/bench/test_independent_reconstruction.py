from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import autoresearch.prefill.independent_reconstruction as reconstruction
from autoresearch.prefill.independent_reconstruction import (
    LeanResult,
    ProviderCandidate,
    ReconstructionStatus,
    VerifiedProofArtifactError,
    VerifiedProofStore,
    build_native_prompt,
    build_reconstruction_package,
    extract_lean_candidates,
    integrate_verified_artifact,
    recompile_verified_artifact,
    run_pass_at_k,
)


SOURCE = """import Mathlib

namespace Route

def localValue : Nat := 3

theorem certifiedDependency : localValue = 3 := by rfl

theorem target (n : Nat) (h : n = localValue) : n = 3 := by
  rw [h]
  exact certifiedDependency

theorem afterTarget : True := by trivial

end Route
"""


def package(tmp_path: Path):
    source = tmp_path / "Route.lean"
    source.write_text(SOURCE)
    return build_reconstruction_package(
        source_path=source,
        project_root=tmp_path,
        theorem_id="Route.target",
        environment_hash="environment",
        retrieval_refs=("Mathlib.Nat", "Route.certifiedDependency"),
        retrieval_context=("#check Nat.add_comm", "#check Route.certifiedDependency"),
    )


def test_package_redacts_only_target_body_and_binds_context(tmp_path):
    result = package(tmp_path)
    prompt_source = result.prompt_source()
    assert "import Mathlib" in prompt_source
    assert "namespace Route" in prompt_source
    assert "def localValue" in prompt_source
    assert "theorem certifiedDependency" in prompt_source
    assert "theorem target" in prompt_source
    assert "rw [h]" not in prompt_source
    assert "exact certifiedDependency" not in prompt_source
    assert "afterTarget" not in prompt_source
    assert result.imports == ("import Mathlib",)
    assert result.namespace == "Route"
    assert result.source_hash and result.theorem_hash
    assert result.dependency_prefix_hash and result.original_proof_hash


def test_native_prompt_preserves_context_without_proof_leak(tmp_path):
    prompt = build_native_prompt(
        package(tmp_path),
        previous_attempt="by simp",
        compiler_feedback="unknown identifier",
    )
    assert "**Current Task:**" in prompt
    assert "#check Route.certifiedDependency" in prompt
    assert "by simp" in prompt
    assert "unknown identifier" in prompt
    assert "rw [h]" not in prompt


def test_parser_accepts_fenced_plain_and_full_declaration():
    text = """plan
```lean4
theorem ignored : True := by
  trivial
```
"""
    assert extract_lean_candidates(text)[0] == "by\n  trivial"
    assert extract_lean_candidates("by\n  exact h") == ("by\n  exact h",)
    assert extract_lean_candidates("exact h") == ("by\n  exact h",)
    assert extract_lean_candidates("<think>secret</think> prose") == ()


class SequenceProvider:
    def __init__(self, outputs=(), failure=None):
        self.outputs = iter(outputs)
        self.failure = failure
        self.prompts = []
        self.seeds = []

    def generate(self, *, prompt, seed, max_tokens):
        self.prompts.append(prompt)
        self.seeds.append(seed)
        if self.failure:
            raise self.failure
        return next(self.outputs)


def test_pass_at_k_uses_deterministic_seeds_and_lean_feedback(tmp_path):
    provider = SequenceProvider((
        ProviderCandidate("```lean\nby bad\n```", "length"),
        ProviderCandidate("by exact h", "stop"),
    ))

    def verify(_package, candidate, _root):
        if candidate == "by exact h":
            return LeanResult(True, "")
        return LeanResult(False, "unknown tactic 'bad'")

    result = run_pass_at_k(
        package=package(tmp_path),
        provider=provider,
        project_root=tmp_path,
        k=3,
        base_seed=41,
        lean_verify=verify,
    )
    assert result.status == ReconstructionStatus.INDEPENDENTLY_VERIFIED.value
    assert provider.seeds == [41, 42]
    assert "unknown tactic 'bad'" in provider.prompts[1]
    assert [attempt.status for attempt in result.attempts] == [
        ReconstructionStatus.LEAN_REJECTED.value,
        ReconstructionStatus.INDEPENDENTLY_VERIFIED.value,
    ]


def test_statuses_distinguish_provider_no_candidate_and_exhaustion(tmp_path):
    failed = run_pass_at_k(
        package=package(tmp_path),
        provider=SequenceProvider(failure=RuntimeError("offline")),
        project_root=tmp_path,
        k=2,
    )
    assert failed.status == ReconstructionStatus.PROVIDER_ADAPTER_FAILED.value

    provider = SequenceProvider((
        ProviderCandidate("no proof", "length"),
        ProviderCandidate("also no proof", "length"),
    ))
    empty = run_pass_at_k(
        package=package(tmp_path), provider=provider,
        project_root=tmp_path, k=2,
    )
    assert empty.status == ReconstructionStatus.SEARCH_EXHAUSTED.value
    assert all(
        attempt.status == ReconstructionStatus.NO_CANDIDATE.value
        for attempt in empty.attempts
    )

    rejected = run_pass_at_k(
        package=package(tmp_path),
        provider=SequenceProvider((
            ProviderCandidate("by nope"), ProviderCandidate("by still_nope"),
        )),
        project_root=tmp_path,
        k=2,
        lean_verify=lambda *_: LeanResult(False, "type mismatch"),
    )
    assert rejected.status == ReconstructionStatus.SEARCH_EXHAUSTED.value
    assert all(
        attempt.status == ReconstructionStatus.LEAN_REJECTED.value
        for attempt in rejected.attempts
    )


def test_no_model_output_mutates_source_or_package(tmp_path):
    built = package(tmp_path)
    source = tmp_path / "Route.lean"
    before = source.read_bytes()
    run_pass_at_k(
        package=built,
        provider=SequenceProvider((ProviderCandidate("by malicious"),)),
        project_root=tmp_path,
        k=1,
        lean_verify=lambda *_: LeanResult(False, "rejected"),
    )
    assert source.read_bytes() == before
    assert "malicious" not in str(built.public_record())


def _verified_store(tmp_path: Path):
    built = package(tmp_path)
    store = VerifiedProofStore(tmp_path / "verified")
    lean = LeanResult(True, "")
    package_hash = hashlib.sha256(json.dumps(
        built.public_record(), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    artifact = store.persist(
        package=built,
        candidate="by\n  simpa [localValue] using h",
        lean_result=lean,
        project_root=tmp_path,
        package_hash=package_hash,
        attempt_index=2,
        seed=43,
        model_id="m-a-p/OProver-8B",
        model_revision="revision",
        quantization="q4",
        provenance={"role": "oprover"},
        created_at=123.0,
    )
    path = (
        store.root / artifact.artifact_hash[:2]
        / f"{artifact.artifact_hash}.json"
    )
    return built, store, artifact, path


def test_verified_candidate_persists_reloads_recompiles_and_integrates(
    tmp_path, monkeypatch,
):
    built, store, artifact, path = _verified_store(tmp_path)
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    reloaded = VerifiedProofStore(store.root).load(artifact.artifact_hash)
    assert reloaded == artifact
    monkeypatch.setattr(
        reconstruction, "verify_candidate_with_project_lean",
        lambda *_args, **_kwargs: LeanResult(True, ""),
    )
    checked, lean = recompile_verified_artifact(
        store=store,
        artifact_reference=path,
        package=built,
        project_root=tmp_path,
    )
    assert lean.accepted and checked.proof_body == artifact.proof_body
    route = tmp_path / "Route.lean"
    route.write_text(route.read_text().replace(
        "theorem target",
        "def destinationContext : Nat := 7\n\ntheorem target",
    ))
    integrated = integrate_verified_artifact(
        store=store,
        artifact_reference=path,
        source_path=route,
        project_root=tmp_path,
        theorem_id="Route.target",
        environment_hash="environment",
    )
    source = route.read_text()
    assert integrated.artifact_hash == artifact.artifact_hash
    assert "simpa [localValue] using h" in source
    assert "rw [h]" not in source


def test_pass_at_k_persists_only_after_lean_acceptance(tmp_path):
    store = VerifiedProofStore(tmp_path / "verified")
    provider = SequenceProvider((
        ProviderCandidate("by bad"),
        ProviderCandidate("by exact h"),
    ))

    def verify(_package, candidate, _root):
        return LeanResult(candidate == "by exact h", "rejected")

    result = run_pass_at_k(
        package=package(tmp_path),
        provider=provider,
        project_root=tmp_path,
        k=2,
        lean_verify=verify,
        artifact_store=store,
        model_id="model",
        model_revision="revision",
        quantization="q4",
    )
    assert result.verified_artifact_hash
    artifacts = list(store.root.glob("*/*.json"))
    assert len(artifacts) == 1
    assert store.load(artifacts[0]).proof_body == "by exact h"


def test_tampered_body_hash_and_environment_are_rejected(
    tmp_path, monkeypatch,
):
    built, store, artifact, path = _verified_store(tmp_path)
    body = json.loads(path.read_text())
    body["proof_body"] = "by malicious"
    path.write_text(json.dumps(body))
    with pytest.raises(
        VerifiedProofArtifactError, match="ARTIFACT_HASH_MISMATCH",
    ):
        store.load(path)

    path.unlink()
    _, store, artifact, path = _verified_store(tmp_path)
    wrong = reconstruction.ReconstructionPackage(
        **{**built.__dict__, "environment_hash": "wrong-environment"}
    )
    monkeypatch.setattr(
        reconstruction, "verify_candidate_with_project_lean",
        lambda *_args, **_kwargs: LeanResult(True, ""),
    )
    with pytest.raises(
        VerifiedProofArtifactError, match="ENVIRONMENT_HASH_MISMATCH",
    ):
        recompile_verified_artifact(
            store=store, artifact_reference=path,
            package=wrong, project_root=tmp_path,
        )


def test_atomic_write_failure_leaves_no_artifact(tmp_path, monkeypatch):
    built = package(tmp_path)
    store = VerifiedProofStore(tmp_path / "verified")
    monkeypatch.setattr(
        os, "link",
        lambda *_args: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(OSError, match="disk failure"):
        store.persist(
            package=built,
            candidate="by exact h",
            lean_result=LeanResult(True, ""),
            project_root=tmp_path,
            package_hash="package",
            attempt_index=0,
            seed=1,
            model_id="model",
            model_revision="revision",
            quantization="q4",
        )
    assert not list(store.root.glob("**/*.json"))
    assert not list(store.root.glob("**/*.tmp"))


def test_duplicate_persist_is_idempotent_and_private_prompt_not_leaked(tmp_path):
    built, store, first, path = _verified_store(tmp_path)
    second = store.persist(
        package=built,
        candidate=first.proof_body,
        lean_result=LeanResult(True, ""),
        project_root=tmp_path,
        package_hash=first.package_hash,
        attempt_index=2,
        seed=43,
        model_id="m-a-p/OProver-8B",
        model_revision="revision",
        quantization="q4",
        provenance={"role": "oprover"},
        created_at=999.0,
    )
    assert second == first
    encoded = path.read_text()
    assert "Previous Failed Attempt" not in encoded
    assert "#check Route.certifiedDependency" not in encoded
    assert "<think>" not in encoded
    assert len(list(store.root.glob("*/*.json"))) == 1


def test_rejected_candidate_cannot_be_persisted(tmp_path):
    built = package(tmp_path)
    with pytest.raises(
        VerifiedProofArtifactError, match="REQUIRES_ACCEPTED",
    ):
        VerifiedProofStore(tmp_path / "verified").persist(
            package=built,
            candidate="by bad",
            lean_result=LeanResult(False, "type mismatch"),
            project_root=tmp_path,
            package_hash="package",
            attempt_index=0,
            seed=1,
            model_id="model",
            model_revision="revision",
            quantization="q4",
        )


def test_proof_hole_candidate_is_rejected_before_verifier(tmp_path):
    calls = []

    def verify(*args):
        calls.append(args)
        return LeanResult(True, "")

    result = run_pass_at_k(
        package=package(tmp_path),
        provider=SequenceProvider((ProviderCandidate("by\n  sorry"),)),
        project_root=tmp_path,
        k=1,
        lean_verify=verify,
        artifact_store=VerifiedProofStore(tmp_path / "verified"),
    )
    assert result.status == ReconstructionStatus.SEARCH_EXHAUSTED.value
    assert result.attempts[0].status == ReconstructionStatus.LEAN_REJECTED.value
    assert result.attempts[0].lean_output == "PROOF_HOLE_TOKEN_REJECTED"
    assert calls == []
    assert not list((tmp_path / "verified").glob("**/*.json"))


def test_accepted_result_with_proof_hole_cannot_be_persisted(tmp_path):
    with pytest.raises(
        VerifiedProofArtifactError, match="BODY_CONTAINS_HOLE",
    ):
        VerifiedProofStore(tmp_path / "verified").persist(
            package=package(tmp_path),
            candidate="by exact sorryAx _ true",
            lean_result=LeanResult(True, ""),
            project_root=tmp_path,
            package_hash="package",
            attempt_index=0,
            seed=1,
            model_id="model",
            model_revision="revision",
            quantization="q4",
        )


def test_lean_sorry_warning_cannot_count_as_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reconstruction.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="warning: declaration uses 'sorry'",
        ),
    )
    result = reconstruction.verify_candidate_with_project_lean(
        package(tmp_path), "by\n  trivial", tmp_path,
    )
    assert not result.accepted
    assert "declaration uses 'sorry'" in result.output
