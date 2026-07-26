from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.prefill.atomic_definition import (
    ProgressVector,
    TypedDefinitionCandidate,
    classify_gap,
    define_one_concept,
    dependency_graph_for_audit,
    first_dependency_closed_gap,
    load_definition_registry,
    record_semantic_iteration,
    verified_progress_vector,
    validate_typed_definition_candidate,
)
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
)
from autoresearch.prefill.orchestration_state import (
    persist_validated_artifact,
    save_checkpoint,
)
from scripts.migrate_atomic_define_one_concept_v1 import migrate


ROOT = Path(__file__).resolve().parents[3]


def _gap(definition_id, *, required_type_id="", symbols=()):
    return {
        "definition_id": definition_id,
        "required_type_id": required_type_id,
        "symbol_ids": list(symbols),
    }


def _lean_ok(*_args, **_kwargs):
    return SimpleNamespace(timed_out=False, returncode=0, output="")


def test_binder_restrictions_are_typed_entries_not_fake_definitions():
    epsilon = classify_gap(
        _gap("DEF_EPSILON"), dependency_graph={}, theorem_card_ids=(),
    )
    genus = classify_gap(
        _gap("DEF_GENUS"), dependency_graph={}, theorem_card_ids=(),
    )
    assert epsilon.classification == "TYPED_RESTRICTION"
    assert "0 < epsilon" in epsilon.typed_entry
    assert genus.classification == "TYPED_RESTRICTION"
    assert "ℕ" in genus.typed_entry
    assert not epsilon.candidates and not genus.candidates


def test_existing_mathlib_reference_resolves_without_redefinition(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "autoresearch.prefill.atomic_definition._run_lean",
        pytest.fail,
    )
    result = define_one_concept(
        gap=_gap("DEF_SERIES_CONVERGENCE"),
        definition_auditor_hash="a" * 64,
        dependency_graph={"DEF_SERIES_CONVERGENCE": ()},
        registry_path=tmp_path / "registry.json",
        project_root=ROOT,
        theorem_card_ids=("locally_uniform_limit-card",),
        current_environment_hash="e" * 64,
    )
    assert result.status == "RESOLVED"
    assert result.classification == "EXISTING_REFERENCE"
    assert result.progress == ProgressVector(existing_definitions_resolved=1)
    assert not load_definition_registry(
        tmp_path / "registry.json", "e" * 64,
    )["definitions"]


def test_one_concept_definition_changes_hash_once_and_restart_is_idempotent(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "autoresearch.prefill.atomic_definition._run_lean", _lean_ok,
    )
    kwargs = dict(
        gap=_gap("DEF_POLE_NEIGHBORHOOD"),
        definition_auditor_hash="a" * 64,
        dependency_graph={"DEF_POLE_NEIGHBORHOOD": ()},
        registry_path=tmp_path / "registry.json",
        project_root=ROOT,
        theorem_card_ids=(),
        current_environment_hash="e" * 64,
    )
    first = define_one_concept(**kwargs)
    second = define_one_concept(**kwargs)
    assert first.status == "RESOLVED"
    assert first.progress.definitions_added == 1
    assert first.registry_hash_before != first.registry_hash_after
    assert first.environment_hash_before != first.environment_hash_after
    assert second.status == "IDEMPOTENT_REPLAY"
    assert second.registry_hash_before == second.registry_hash_after
    assert second.progress.total == 0


def test_invalid_lean_definition_is_rejected_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autoresearch.prefill.atomic_definition._run_lean",
        lambda *_args, **_kwargs: SimpleNamespace(
            timed_out=False, returncode=1, output="type mismatch",
        ),
    )
    result = define_one_concept(
        gap=_gap("DEF_POLE_NEIGHBORHOOD"),
        definition_auditor_hash="a" * 64,
        dependency_graph={"DEF_POLE_NEIGHBORHOOD": ()},
        registry_path=tmp_path / "registry.json",
        project_root=ROOT,
        theorem_card_ids=(),
        current_environment_hash="e" * 64,
    )
    assert result.status == "LEAN_DEFINITION_REJECTED"
    assert result.progress.total == 0
    assert result.registry_hash_before == result.registry_hash_after
    assert not (tmp_path / "registry.json").exists()


def test_unknown_gap_emits_mathematical_evidence_not_retry(tmp_path):
    result = define_one_concept(
        gap=_gap("DEF_UNKNOWN"),
        definition_auditor_hash="a" * 64,
        dependency_graph={"DEF_UNKNOWN": ()},
        registry_path=tmp_path / "registry.json",
        project_root=ROOT,
        theorem_card_ids=(),
        current_environment_hash="e" * 64,
    )
    assert result.status == "NO_TYPED_DEFINITION_CANDIDATE"
    assert result.artifact_hash
    assert result.progress.total == 0
    assert not (tmp_path / "registry.json").exists()


def test_dependency_order_is_derived_from_typed_evidence():
    missing = (
        _gap("DEF_EPSILON", symbols=("SYM_EPSILON",)),
        _gap("DEF_GENUS", symbols=("SYM_GENUS_P",)),
        _gap(
            "DEF_POLE_NEIGHBORHOOD",
            required_type_id="TYPE_PUNCTURED_OR_FULL_NEIGHBORHOOD",
            symbols=("SYM_POLE_LOCATION",),
        ),
        _gap(
            "DEF_FUNCTION_BINDING",
            required_type_id="TYPE_CANONICAL_PRODUCT_BINDING",
            symbols=("SYM_FUNCTION", "SYM_SEQUENCE"),
        ),
        _gap(
            "DEF_SERIES_CONVERGENCE",
            required_type_id="TYPE_CONVERGENCE_MODE",
            symbols=("SYM_SEQUENCE",),
        ),
        _gap(
            "DEF_GROWTH_ORDER",
            required_type_id="TYPE_ENTIRE_FUNCTION_ORDER",
            symbols=("SYM_FUNCTION", "SYM_GENUS_P"),
        ),
        _gap(
            "DEF_SEQUENCE_DENSITY",
            required_type_id="TYPE_SEQUENCE_DENSITY",
            symbols=("SYM_SEQUENCE",),
        ),
    )
    graph = dependency_graph_for_audit(missing)
    assert graph["DEF_FUNCTION_BINDING"] == ("DEF_POLE_NEIGHBORHOOD",)
    assert graph["DEF_SERIES_CONVERGENCE"] == ("DEF_FUNCTION_BINDING",)
    assert graph["DEF_GROWTH_ORDER"] == ("DEF_FUNCTION_BINDING",)
    assert first_dependency_closed_gap(
        missing,
        dependency_graph=graph,
        resolved_gap_ids=("DEF_EPSILON", "DEF_GENUS"),
    )["definition_id"] == "DEF_POLE_NEIGHBORHOOD"


def test_true_theorem_and_registration_note_have_zero_progress():
    progress = verified_progress_vector(
        lemmas_proved=1,
        subgoals_closed=1,
        lean_source="theorem fakeProgress : True := by trivial",
    )
    assert progress.total == 0
    assert verified_progress_vector().total == 0


def test_three_zero_delta_iterations_stagnate_and_forbid_fingerprint():
    checkpoint = OrchestrationCheckpoint(
        target_obligation_id="ROOT",
        definition_environment_hash="env",
    )
    outcomes = [
        record_semantic_iteration(
            checkpoint,
            ProgressVector(),
            move_class="DEFINE_ONE_CONCEPT:gap",
        )
        for _ in range(3)
    ]
    assert outcomes == [False, False, True]
    assert checkpoint.semantic_stagnation_count == 3
    assert checkpoint.progress_fingerprint in (
        checkpoint.forbidden_semantic_fingerprints
    )
    assert checkpoint.stagnation_reason.startswith("three completed")


def test_infrastructure_failure_does_not_count_and_progress_resets():
    checkpoint = OrchestrationCheckpoint(
        target_obligation_id="ROOT",
        definition_environment_hash="env",
    )
    record_semantic_iteration(
        checkpoint, ProgressVector(), move_class="M",
        infrastructure_failure=True,
    )
    assert checkpoint.semantic_stagnation_count == 0


def test_environment_change_can_replan_from_definition_resolution():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_RESOLUTION.value,
    )
    checkpoint.transition(
        ProofState.STRATEGY_TOURNAMENT,
        "definition-environment-changed:replan",
    )
    assert checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT
    record_semantic_iteration(checkpoint, ProgressVector(), move_class="M")
    record_semantic_iteration(
        checkpoint,
        ProgressVector(verified_counterexamples=1),
        move_class="M",
    )
    assert checkpoint.semantic_stagnation_count == 0


def test_447_repeated_legacy_actions_cannot_execute_or_make_progress():
    checkpoint = OrchestrationCheckpoint(
        target_obligation_id="ROOT",
        definition_environment_hash="env",
    )
    stopped_at = None
    for iteration in range(447):
        stagnant = record_semantic_iteration(
            checkpoint, ProgressVector(),
            move_class="REGISTER_DEFINITION_OBLIGATION",
        )
        if stagnant:
            stopped_at = iteration + 1
            break
    assert stopped_at == 3


def test_architecture_current_contains_no_executable_legacy_registration():
    production = (
        ROOT / "autoresearch/prefill/creative_decomposition.py"
    ).read_text(encoding="utf-8")
    assert "REGISTER_" + "DEFINITION_OBLIGATION" not in production


def test_candidate_selection_accepts_short_ids_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autoresearch.prefill.atomic_definition._run_lean", _lean_ok,
    )
    with pytest.raises(ValueError, match="registered short ID"):
        define_one_concept(
            gap=_gap("DEF_POLE_NEIGHBORHOOD"),
            definition_auditor_hash="a" * 64,
            dependency_graph={"DEF_POLE_NEIGHBORHOOD": ()},
            registry_path=tmp_path / "registry.json",
            project_root=ROOT,
            theorem_card_ids=(),
            current_environment_hash="e" * 64,
            selected_candidate_id='{"lean":"def injected := True"}',
        )


def test_hidden_assumption_is_rejected_before_lean_runs(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.prefill.atomic_definition._run_lean",
        pytest.fail,
    )
    candidate = TypedDefinitionCandidate(
        "A",
        "DEF_BAD",
        "bad",
        "axiom hidden : Prop\ndef bad : Prop := hidden",
        (),
        ("Prop",),
    )
    assert validate_typed_definition_candidate(
        candidate, project_root=ROOT,
    ) == (False, "HIDDEN_ASSUMPTION_OR_FORBIDDEN_COMMAND")


def test_migration_preserves_audit_and_selects_first_real_gap(tmp_path):
    home = tmp_path / "home"
    autoresearch = home / "autoresearch"
    autoresearch.mkdir(parents=True)
    checkpoint_path = autoresearch / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state="DECOMPOSER",
        selected_move_id="REGISTER_DEFINITION_OBLIGATION",
        typed_ir_hash="old",
        proposition_hash="old-true",
    )
    persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload={"missing_definitions": [
            _gap("DEF_EPSILON", symbols=("SYM_EPSILON",)),
            _gap("DEF_GENUS", symbols=("SYM_GENUS_P",)),
            _gap(
                "DEF_POLE_NEIGHBORHOOD",
                required_type_id="TYPE_PUNCTURED_OR_FULL_NEIGHBORHOOD",
                symbols=("SYM_POLE_LOCATION",),
            ),
        ]},
        dependencies=[],
        source_run_id="audit",
    )
    save_checkpoint(checkpoint_path, checkpoint)
    (home / "agent_gan_proof_ledger.json").write_text(
        json.dumps({"version": 92}), encoding="utf-8",
    )
    result = migrate(
        home=home,
        project_root=ROOT,
        supervisor_pid=999999,
        snapshot=tmp_path / "snapshot",
    )
    assert result["current_gap"] == "DEF_POLE_NEIGHBORHOOD"
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert raw["migration_event"] == "atomic_define_one_concept_progress_v1"
    assert raw["selected_move_id"] == "DEFINE_ONE_CONCEPT"
    assert raw["typed_ir_hash"] == ""
    assert "definition_auditor" in raw["validated_artifacts"]
