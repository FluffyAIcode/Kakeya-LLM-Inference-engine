from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.prefill.definition_resolution import (
    DefinitionSourceRegistry,
    PropertyStatus,
    build_definition_query,
    default_source_registry,
    load_resolution_store,
    migrate_legacy_registry,
    normalize_candidates,
    resolve_one_concept,
)


ROOT = Path(__file__).resolve().parents[3]


def query(**overrides):
    gap = {
        "definition_id": "DEF_TEST",
        "domain": "ℕ",
        "codomain": "ℕ",
        "use_sites": ["u2", "u1"],
        "required_properties": ["monotone"],
        "hard_properties": ["monotone"],
    }
    gap.update(overrides.pop("gap", {}))
    return build_definition_query(
        gap, parent_hash=overrides.pop("parent_hash", "p"),
        typed_ir_hash=overrides.pop("typed_ir_hash", "ir"),
        auditor_hash="audit", critic_evidence_hashes=(),
        counterexample_evidence_hashes=(), theorem_dependencies=(),
        environment_hash=overrides.pop("environment_hash", "env"),
        prior_failure_hashes=overrides.pop("prior_failure_hashes", ()),
    )


def registry(*raw):
    result = DefinitionSourceRegistry()

    def adapter(definition_query, _root, _context):
        from autoresearch.prefill.definition_resolution import SourceProvenance
        return list(raw), SourceProvenance(
            "fixture", "FIXTURE", "QUERIED", "test",
            definition_query.environment_hash, "evidence",
        )

    result.register("fixture", adapter)
    return result


def candidate(name="testDef", expression="fun n => n", branch=""):
    return {
        "source_id": "fixture",
        "declaration_name": name,
        "domain": "ℕ",
        "codomain": "ℕ → ℕ",
        "typed_expression": expression,
        "branch_id": branch,
        "claimed_properties": ["monotone"],
    }


@pytest.fixture
def lean_ok(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.prefill.definition_resolution._run_lean",
        lambda *_args, **_kwargs: SimpleNamespace(
            timed_out=False, returncode=0, output="",
        ),
    )


def evidence(status="VERIFIED"):
    return {"monotone": {
        "status": status,
        "evidence_kind": "LEAN_THEOREM_CARD",
        "evidence_ref": "sha256:proof",
    }}


def test_query_fingerprint_is_semantic_and_changes_with_evidence():
    first = query()
    reordered = query(gap={
        "definition_id": "DEF_TEST", "domain": "ℕ", "codomain": "ℕ",
        "use_sites": ["u1", "u2"], "required_properties": ["monotone"],
        "hard_properties": ["monotone"],
    })
    assert first.fingerprint == reordered.fingerprint
    assert first.fingerprint != query(parent_hash="changed").fingerprint
    assert first.fingerprint != query(environment_hash="changed").fingerprint


def test_unavailable_oproofs_is_reported_honestly(monkeypatch):
    monkeypatch.delenv("KAKEYA_OPROOFS_ROOT", raising=False)
    _, statuses = default_source_registry().retrieve(query(), ROOT, {})
    status = {item.source_id: item.status for item in statuses}
    assert status["oproofs"] == "UNAVAILABLE"


def test_candidate_normalization_requires_provenance():
    with pytest.raises(ValueError, match="provenance"):
        normalize_candidates(query(), [candidate()], ())


def test_success_compiles_verifies_and_commits(tmp_path, lean_ok):
    result = resolve_one_concept(
        query=query(), store_path=tmp_path / "resolution.json",
        project_root=ROOT, source_registry=registry(candidate()),
        property_evidence=evidence(),
    )
    assert result.status == "COMMITTED"
    assert result.committed_candidate_hash
    assert result.environment_hash_before != result.environment_hash_after
    store = load_resolution_store(tmp_path / "resolution.json", "env")
    assert store["commits"]["DEF_TEST"]["candidate_hash"] == result.committed_candidate_hash


def test_unknown_hard_property_cannot_commit(tmp_path, lean_ok):
    result = resolve_one_concept(
        query=query(), store_path=tmp_path / "resolution.json",
        project_root=ROOT, source_registry=registry(candidate()),
        property_evidence={},
    )
    assert result.status == "INTERFACE_REQUIRED"
    assert result.exhaustion_hash and result.interface_hash
    statuses = next(iter(result.property_statuses.values()))
    assert statuses["monotone"] == PropertyStatus.UNKNOWN.value


def test_ontology_type_id_does_not_fake_a_viable_interface(tmp_path, lean_ok):
    abstract = query(gap={
        "definition_id": "DEF_SEQUENCE_DENSITY",
        "domain": "",
        "codomain": "",
        "required_type_id": "TYPE_SEQUENCE_DENSITY",
        "symbol_ids": ["SYM_SEQUENCE", "SYM_SEQUENCE_DENSITY"],
    })
    result = resolve_one_concept(
        query=abstract, store_path=tmp_path / "resolution.json",
        project_root=ROOT, source_registry=registry(),
    )
    assert result.status == "PARENT_STATEMENT_UNDERSPECIFIED"
    assert result.interface_hash and result.exhaustion_hash


def test_multiple_survivors_branch_without_silent_choice(tmp_path, lean_ok):
    result = resolve_one_concept(
        query=query(), store_path=tmp_path / "resolution.json",
        project_root=ROOT,
        source_registry=registry(
            candidate("definitionOne", "fun n => n", "one"),
            candidate("definitionTwo", "fun n => n + 1", "two"),
        ),
        property_evidence=evidence(),
    )
    assert result.status == "PARENT_STATEMENT_UNDERSPECIFIED"
    assert len(result.branch_hashes) == 2
    assert not result.committed_candidate_hash
    store = load_resolution_store(tmp_path / "resolution.json", "env")
    for branch_hash in result.branch_hashes:
        assert store["branches"][branch_hash]["equivalence_obligation_hash"]


def test_identical_exhausted_query_cannot_rerun(tmp_path, lean_ok):
    path = tmp_path / "resolution.json"
    first = resolve_one_concept(
        query=query(), store_path=path, project_root=ROOT,
        source_registry=registry(), property_evidence={},
    )
    second = resolve_one_concept(
        query=query(), store_path=path, project_root=ROOT,
        source_registry=registry(candidate()), property_evidence=evidence(),
    )
    assert first.status == "INTERFACE_REQUIRED"
    assert second.status == "IDENTICAL_QUERY_EXHAUSTED"
    assert second.store_hash_before == second.store_hash_after


def test_model_can_rank_short_ids_only(tmp_path, lean_ok):
    with pytest.raises(ValueError, match="short candidate IDs"):
        resolve_one_concept(
            query=query(), store_path=tmp_path / "resolution.json",
            project_root=ROOT, source_registry=registry(candidate()),
            property_evidence=evidence(),
            ranked_short_ids=('{"lean":"axiom injected : False"}',),
        )


def test_hidden_assumption_is_rejected_before_lean(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autoresearch.prefill.definition_resolution._run_lean", pytest.fail,
    )
    bad = candidate(expression="axiom hidden : False")
    result = resolve_one_concept(
        query=query(), store_path=tmp_path / "resolution.json",
        project_root=ROOT, source_registry=registry(bad),
        property_evidence=evidence(),
    )
    assert result.status == "INTERFACE_REQUIRED"


def test_legacy_definitions_are_audit_only_with_property_obligations():
    store = migrate_legacy_registry({
        "definitions": {
            "oldNeighborhood": {
                "gap_id": "DEF_POLE_NEIGHBORHOOD",
                "lean_source": "def oldNeighborhood := 1",
            },
        },
    }, base_environment_hash="env")
    record = next(iter(store["historical_audit"].values()))
    assert record["audit_only"] is True
    assert record["semantic_validation"] == "PENDING"
    assert record["property_obligations"] == [
        "full_vs_punctured_neighborhood",
    ]
    assert not store["commits"]


def test_static_active_architecture_has_no_legacy_registry_or_reframe_loop():
    source = (ROOT / "autoresearch/prefill/architecture_v7.py").read_text()
    assert "definition_" + "registry.json" not in source
    assert "NO_TYPED_" + "DEFINITION_CANDIDATE" not in source
    assert "ProofState." + "REFRAME" not in source
