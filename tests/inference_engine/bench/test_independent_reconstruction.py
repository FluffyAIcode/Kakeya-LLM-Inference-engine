from __future__ import annotations

from pathlib import Path

from autoresearch.prefill.independent_reconstruction import (
    LeanResult,
    ProviderCandidate,
    ReconstructionStatus,
    build_native_prompt,
    build_reconstruction_package,
    extract_lean_candidates,
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
