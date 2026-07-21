from pathlib import Path

import autoresearch.prefill.lean_gate as lean_gate
from autoresearch.prefill.lean_gate import (
    extract_lean_signature_blocks,
    validate_lean_signature,
)


ROOT = Path(__file__).resolve().parents[3]


def test_extract_and_typecheck_minimal_mathlib_signature():
    text = """
### LEAN_SIGNATURE RH-C2-leaf
```lean
theorem local_pole_signature
    (f : ℂ → ℂ)
    (h : Continuous f) :
    Continuous f := by
  sorry
```
"""
    blocks = extract_lean_signature_blocks(text)
    assert len(blocks) == 1
    target, source = blocks[0]
    assert target == "RH-C2-leaf"
    result = validate_lean_signature(source, project_root=ROOT)
    assert result.ok, result.error
    assert result.status == "FORMALIZED"
    assert len(result.signature_hash) == 64


def test_lean_gate_rejects_unknown_type():
    result = validate_lean_signature(
        "theorem bad (x : MissingType) : True := by trivial",
        project_root=ROOT,
    )
    assert not result.ok
    assert result.status == "TYPECHECK_FAILED"
    assert "unknown" in result.error.lower()


def test_lean_gate_rejects_executable_or_axiomatic_commands():
    for source in (
        "axiom hidden : False",
        "def hidden : Nat := 1\ntheorem ok : True := by trivial",
        "theorem bad : True := by run_tac do pure ()",
        "#eval 1 + 1",
    ):
        result = validate_lean_signature(source, project_root=ROOT)
        assert not result.ok
        assert result.status == "UNSAFE_REJECTED"


def test_lean_gate_retries_timeout_after_warmup(monkeypatch):
    runs = iter((
        lean_gate._LeanRun(None, True, 45.0, "first partial"),
        lean_gate._LeanRun(0, False, 3.0, "warm"),
        lean_gate._LeanRun(0, False, 4.0, "retry"),
    ))
    monkeypatch.setattr(lean_gate, "_run_lean", lambda *args, **kwargs: next(runs))
    result = validate_lean_signature(
        "theorem retried : True := by trivial",
        project_root=ROOT,
    )
    assert result.ok
    assert result.status == "FORMALIZED"
    assert result.attempts == 2
    assert result.elapsed_s == 52.0
    assert "first partial" in result.output


def test_lean_gate_classifies_second_timeout(monkeypatch):
    runs = iter((
        lean_gate._LeanRun(None, True, 45.0, "first"),
        lean_gate._LeanRun(0, False, 2.0, "warm"),
        lean_gate._LeanRun(None, True, 120.0, "second"),
    ))
    monkeypatch.setattr(lean_gate, "_run_lean", lambda *args, **kwargs: next(runs))
    result = validate_lean_signature(
        "theorem timeout : True := by trivial",
        project_root=ROOT,
    )
    assert not result.ok
    assert result.status == "TYPECHECK_TIMEOUT"
    assert result.attempts == 2
    assert "firstwarmsecond" in result.output
