from pathlib import Path
import json

import pytest

import autoresearch.prefill.lean_gate as lean_gate
from autoresearch.prefill.lean_gate import (
    LEAN_CONTRACT_USER_REGISTRY,
    LEAN_CONTRACT_REGISTRY,
    LEAN_SIGNATURE_CONTRACT,
    LEAN_SIGNATURE_FIXTURES,
    lean_signature_contract_ref,
    lean_symbol_semantic_hash,
    lint_lean_model_prompt,
    normalize_lean_signature,
    normalize_registered_latex_identifiers,
    register_lean_symbol_table,
    resolve_lean_contract,
    resolve_lean_symbol_table,
    validate_lean_proof,
    validate_lean_signature,
)


ROOT = Path(__file__).resolve().parents[3]

LATEST_PARENT_FAILURE = (
    "**The Density-Singularity Gap Lemma:** Prove that for a given "
    "$\\epsilon$ and a fixed genus $p$, there exists a critical density."
)
LATEST_CHILD_FAILURE = (
    "theorem L1_definition : ∀ (z_n : ℕ → ℂ), True"
)
LATEST_REDUCTION_FAILURE = (
    "ba31be416856d1f975f8f885f54d9babde1aada7ee98d42341c3b58cad91a061"
)
LATEST_LATEX_PARENT_SOURCE = (
    r"theorem parent_537f9def407e "
    r"(\epsilon : \mathbb{R}_{>0}) (p : \mathbb{N}) "
    r"(\rho_c : \mathbb{R}) (\{z_n\} : \mathbb{N} \to \mathbb{C}) "
    r"(s_0 : \mathbb{C}) (m : \mathbb{C}) "
    r"(\delta : \mathbb{R}_{>0}) : \rho > \rho_c \implies "
    r"\neg (\sum_{n=1}^{\infty} \frac{1}{s-z_n} = "
    r"\frac{m}{s-s_0} \text{ in a } \delta\text{-neighborhood of } "
    r"s_0 \implies \text{growth_order}(f) \le p) := by"
)


@pytest.mark.parametrize(
    ("fixture", "source", "error"),
    [
        ("zero_declarations", "", "empty Lean declaration"),
        (
            "multiple_declarations",
            "theorem one : True := by\ntheorem two : True := by",
            "exactly one",
        ),
        (
            "prose",
            "Here is Lean:\ntheorem one : True := by",
            "only one theorem",
        ),
        (
            "fence",
            "```lean\ntheorem one : True := by\n```",
            "fences",
        ),
        ("def", "def one : Prop := True", "forbidden Lean command"),
        (
            "forbidden_command",
            "import Mathlib\ntheorem one : True := by",
            "forbidden Lean command",
        ),
        (
            "missing_scaffold",
            "theorem one : True",
            "must end with `:= by`",
        ),
        (
            "duplicate_scaffold",
            "theorem one : True := by := by",
            "duplicate",
        ),
        (
            "proof_body",
            "theorem one : True := by trivial",
            "no proof body",
        ),
        (
            "placeholder",
            "theorem one : True := by sorry",
            "placeholder",
        ),
        (
            "latex_escape",
            r"theorem one (x : \mathbb{R}) : True := by",
            "LaTeX",
        ),
        (
            "latest_parent_prose",
            LATEST_PARENT_FAILURE,
            "LaTeX",
        ),
        (
            "latest_child_missing_scaffold",
            LATEST_CHILD_FAILURE,
            "must end with `:= by`",
        ),
        (
            "latest_reduction_hash",
            LATEST_REDUCTION_FAILURE,
            "exactly one",
        ),
    ],
)
def test_signature_contract_rejects_required_fixtures(fixture, source, error):
    assert fixture in LEAN_SIGNATURE_FIXTURES
    result = validate_lean_signature(source, project_root=ROOT)
    assert not result.ok
    assert result.status == "CONTRACT_FAILED"
    assert error.casefold() in result.error.casefold()


def test_valid_multiline_binders_and_lemma_elaborate():
    source = """lemma localPoleSignature
    (P : Prop)
    (h : P) :
    P := by"""
    result = validate_lean_signature(source, project_root=ROOT)
    assert result.ok, result.error
    assert result.status == "FORMALIZED"
    assert result.declaration_name == "localPoleSignature"
    assert result.proposition == "P"
    assert result.normalized_source.endswith(":= by")
    assert len(result.signature_hash) == 64


def test_mathematical_inequality_is_not_an_angle_placeholder():
    result = validate_lean_signature(
        "theorem successorStrict (n : Nat) : n < n + 1 := by",
        project_root=ROOT,
    )
    assert result.ok, result.error


def test_lean_gate_rejects_unknown_type_after_contract_validation():
    result = validate_lean_signature(
        "theorem bad (x : MissingType) : True := by",
        project_root=ROOT,
    )
    assert not result.ok
    assert result.status == "TYPECHECK_FAILED"
    assert "unknown" in result.error.lower()


def test_scaffold_normalization_preserves_all_mathematical_fields_and_hashes():
    original = """theorem normalized
    (P : Prop)
    (h : P) :
    P"""
    before = LEAN_SIGNATURE_CONTRACT.normalize_signature(original)
    canonical = normalize_lean_signature(original + "   :=   by   ")
    assert canonical.source.endswith(" := by")
    assert canonical.name == before.name
    assert canonical.binders == before.binders
    assert canonical.proposition == before.proposition
    assert canonical.proposition_hash == before.proposition_hash
    assert canonical.declaration_hash == before.declaration_hash


@pytest.mark.parametrize(
    ("source", "expected", "error"),
    [
        (
            "theorem changed (P : Prop) : P",
            {"name": "immutable", "binders": "(P : Prop)", "proposition": "P"},
            "name changed",
        ),
        (
            "theorem immutable (P : Prop) : Not P",
            {"name": "immutable", "binders": "(P : Prop)", "proposition": "P"},
            "proposition changed",
        ),
        (
            "theorem immutable (Q : Prop) : Q",
            {"name": "immutable", "binders": "(P : Prop)", "proposition": "Q"},
            "binders changed",
        ),
    ],
)
def test_normalization_never_semantically_repairs(source, expected, error):
    with pytest.raises(ValueError, match=error):
        normalize_lean_signature(source, expected=expected)


def test_prover_uses_same_contract_but_requires_complete_body():
    incomplete = validate_lean_proof(
        "theorem proofTarget : True := by",
        project_root=ROOT,
    )
    assert not incomplete.ok
    assert incomplete.status == "CONTRACT_FAILED"
    complete = validate_lean_proof(
        "theorem proofTarget : True := by\n  trivial",
        project_root=ROOT,
    )
    assert complete.ok, complete.error
    assert complete.status == "PROVED"


def test_all_lean_producing_roles_are_registered_with_every_fixture():
    assert set(LEAN_CONTRACT_USER_REGISTRY) == {
        "formalizer",
        "prover",
        "premise_auditor",
    }
    required = set(LEAN_SIGNATURE_FIXTURES)
    assert len(required) == 27
    for role, user in LEAN_CONTRACT_USER_REGISTRY.items():
        assert user.role == role
        contract = resolve_lean_contract(
            user.contract_id,
            user.contract_version,
        )
        assert contract.content_sha256
        assert set(user.fixtures) == required
        source = (ROOT / user.file).read_text()
        assert f'"{role}"' in source
        if user.policy == "signature":
            assert "validate_lean_signature" in source
        elif user.policy == "complete_proof":
            assert "validate_lean_proof" in source
        else:
            assert "cannot be safely transformed" in source


def test_versioned_content_addressed_contract_ids_fail_closed():
    reference = lean_signature_contract_ref()
    contract = resolve_lean_contract(
        reference["contract_id"],
        reference["version"],
        signature_only=True,
    )
    assert contract.contract_id.endswith(contract.content_sha256[:16])
    assert (contract.contract_id, contract.version) in LEAN_CONTRACT_REGISTRY
    with pytest.raises(ValueError, match="unknown or stale"):
        resolve_lean_contract("missing-contract", 1)
    with pytest.raises(ValueError, match="unknown or stale"):
        resolve_lean_contract(contract.contract_id, contract.version + 1)
    with pytest.raises(ValueError, match="policy mismatch"):
        resolve_lean_contract(
            contract.contract_id,
            contract.version,
            signature_only=False,
        )


def _symbol_table():
    return register_lean_symbol_table(
        [
            {"symbol": "\\epsilon", "type": "real constant"},
            {"symbol": "\\rho", "type": "density of sequence"},
            {"symbol": "\\delta", "type": "neighborhood radius"},
            {"symbol": "p", "type": "integer (genus)"},
        ],
        parent_statement_hash="parent",
    )


def test_registered_identifier_latex_normalization_is_hash_preserving():
    table = _symbol_table()
    original = r"(\epsilon : ℝ) (\rho : ℝ) (\delta : ℝ)"
    before = lean_symbol_semantic_hash(original, table)
    normalized = normalize_registered_latex_identifiers(original, table)
    assert normalized == "(epsilon : ℝ) (rho : ℝ) (delta : ℝ)"
    assert lean_symbol_semantic_hash(normalized, table) == before
    assert resolve_lean_symbol_table(
        table.symbol_table_id,
        table.version,
    ) == table
    assert table.symbol_table_id.endswith(table.content_sha256[:16])
    with pytest.raises(ValueError, match="unknown or stale"):
        resolve_lean_symbol_table(table.symbol_table_id, table.version + 1)
    escaped = json.loads(r'{"binder":"\\\\epsilon"}')["binder"]
    with pytest.raises(ValueError, match="unapproved LaTeX"):
        normalize_registered_latex_identifiers(escaped, table)


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        (r"\sum n", r"\\sum"),
        (r"\frac{1}{x}", r"\\frac"),
        (r"\{z_n\}", r"\\\{"),
        (r"\unknown x", r"\\unknown"),
        (
            r"(\epsilon : \mathbb{R})",
            r"\\mathbb",
        ),
    ],
)
def test_semantic_and_unknown_latex_commands_fail_closed(source, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        normalize_registered_latex_identifiers(source, _symbol_table())


def test_exact_latest_latex_parent_output_rejects_semantic_commands():
    with pytest.raises(ValueError) as caught:
        normalize_registered_latex_identifiers(
            LATEST_LATEX_PARENT_SOURCE,
            _symbol_table(),
        )
    message = str(caught.value)
    for command in (r"\mathbb", r"\sum", r"\frac", r"\text", r"\{"):
        assert command in message


def test_lean_prompt_lint_rejects_raw_latex_but_accepts_unicode_ascii():
    with pytest.raises(ValueError, match="raw LaTeX"):
        lint_lean_model_prompt([
            {"role": "user", "content": r'{"binder":"\\rho"}'},
        ])
    lint_lean_model_prompt([
        {
            "role": "user",
            "content": '{"binders":["epsilon : ℝ","rho : ℝ","delta : ℝ"]}',
        },
    ])


def test_lean_gate_retries_timeout_after_warmup(monkeypatch):
    runs = iter((
        lean_gate._LeanRun(None, True, 45.0, "first partial"),
        lean_gate._LeanRun(0, False, 3.0, "warm"),
        lean_gate._LeanRun(0, False, 4.0, "retry"),
    ))
    monkeypatch.setattr(lean_gate, "_run_lean", lambda *args, **kwargs: next(runs))
    result = validate_lean_signature(
        "theorem retried : True := by",
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
        "theorem timeout : True := by",
        project_root=ROOT,
    )
    assert not result.ok
    assert result.status == "TYPECHECK_TIMEOUT"
    assert result.attempts == 2
    assert "firstwarmsecond" in result.output
