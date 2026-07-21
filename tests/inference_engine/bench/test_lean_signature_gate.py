from pathlib import Path

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
    assert len(result.signature_hash) == 64


def test_lean_gate_rejects_unknown_type():
    result = validate_lean_signature(
        "theorem bad (x : MissingType) : True := by trivial",
        project_root=ROOT,
    )
    assert not result.ok
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
