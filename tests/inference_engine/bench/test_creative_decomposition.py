from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.prefill.creative_decomposition import (
    MIGRATION_EVENT,
    MOVE_REGISTRY,
    assert_no_scratchpad_content,
    build_candidate_set,
    persist_private_scratchpad,
    rank_candidates,
    rank_short_choice,
    synthesis_manifest,
    synthesis_trigger,
)
from autoresearch.prefill.evidence_planner import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
    ProofPlan,
    ProofPlanNode,
    VerificationStatus,
    build_evidence_gap_graph,
    first_executable_node,
    generate_proof_plans,
    host_evidence_context,
    replan_after_failure,
    validate_proof_plan,
)
from autoresearch.prefill.host_compiler import run_host_gates
from autoresearch.prefill.lean_gate import validate_lean_signature
from autoresearch.prefill.math_ir import parse_math_ir, validate_math_ir
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
)
from autoresearch.prefill.theorem_cards import (
    build_theorem_card_index,
    search_theorem_cards,
    validate_theorem_cards,
)
from autoresearch.prefill.typed_transport import (
    AdapterError,
    decode_role_fields,
    transport_prompt,
)
from scripts.migrate_creative_decomposition_v3 import (
    MIGRATION_EVENT as ARCHITECTURE_MIGRATION_EVENT,
    atomic_snapshot,
    migrate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[3]


def _candidate_set(**kwargs):
    cards = build_theorem_card_index(ROOT)
    preconditions = {
        item
        for move in MOVE_REGISTRY.values()
        for item in move.precondition_ids
        if item != "registered_definition_gap"
    }
    hypotheses = {
        item for card in cards for item in card.required_hypotheses
    }
    return build_candidate_set(
        target_ref="claim:" + "a" * 64,
        viewpoint="local_holomorphicity",
        dependency_ids=("d" * 64,),
        theorem_cards=cards,
        satisfied_precondition_ids=preconditions,
        satisfied_theorem_hypothesis_ids=hypotheses,
        available_dependency_artifact_ids=("d" * 64,),
        required_dependency_artifact_ids=("d" * 64,),
        **kwargs,
    )


def test_versioned_move_registry_has_required_scoped_strict_moves():
    required = {
        "CASE_SPLIT", "RESTRICT_DOMAIN", "REMOVE_IRRELEVANT_ASSUMPTION",
        "HOLOMORPHIC_EXTENSION", "SINGULARITY_CONTRADICTION",
    }
    assert required <= MOVE_REGISTRY.keys()
    for move_id in required:
        move = MOVE_REGISTRY[move_id]
        assert move.version == 3
        assert move.operand_kinds
        assert move.precondition_ids
        assert move.metric.strictly_simpler
        assert len(move.content_hash) == 64


def test_host_generates_k_distinct_candidates_and_rejects_ancestor():
    first = _candidate_set()
    assert len(first.candidates) >= 3
    assert len({item.novelty_hash for item in first.candidates}) == len(
        first.candidates,
    )
    ancestor = next(
        item for item in first.candidates
        if item.move.move_id == "SINGULARITY_CONTRADICTION"
    )
    second = _candidate_set(ancestor_hashes=(ancestor.typed_ir_hash,))
    assert (
        ancestor.choice_id, "ANCESTOR_EQUIVALENT"
    ) in second.rejected
    assert all(
        item.typed_ir_hash != ancestor.typed_ir_hash
        for item in second.candidates
    )


def test_local_holomorphicity_candidate_is_honest_special_case_and_elaborates(tmp_path):
    candidates = _candidate_set()
    candidate = next(
        item for item in candidates.candidates
        if item.move.move_id == "SINGULARITY_CONTRADICTION"
    )
    assert candidate.move.result_status == "SPECIAL_CASE_LEMMA"
    assert candidate.move.requires_parent_case_split
    assert "locally_uniform_limit_holomorphic" in candidate.theorem_card_ids
    validated = validate_math_ir(parse_math_ir(candidate.typed_payload))
    assert validated.content_hash == candidate.typed_ir_hash
    assert len(validated.math_ir.premises) == 7
    result = run_host_gates(
        candidate.typed_payload,
        project_root=ROOT,
        cache_dir=tmp_path / "gates",
        lean_validator=validate_lean_signature,
    )
    assert result.ok
    assert result.compilation is not None
    assert "polesOutsideDisk" in result.compilation.declaration_source
    assert "localUniformConvergenceOnDisk" in result.compilation.declaration_source
    assert "agreesWithSimplePoleOnPuncturedDisk" in (
        result.compilation.declaration_source
    )


def test_private_scratchpad_is_unparsed_and_cannot_cross_artifact_gate(tmp_path):
    private = r"""Untrusted: use $\frac m{s-s_0}$; {"not":"transport"}; theorem X := by"""
    ref = persist_private_scratchpad(
        tmp_path / "scratchpads",
        role="decomposer_scratchpad",
        transcript=private,
        token_count=19,
    )
    assert ref.audit_only and not ref.authoritative and not ref.public
    assert ref.parse_contract == "NEVER_PARSE"
    safe = {"scratchpad_ref": ref.sha256, "selected_candidate_id": "C1"}
    assert_no_scratchpad_content(safe, ref)
    with pytest.raises(ValueError, match="SCRATCHPAD_TEXT"):
        assert_no_scratchpad_content({"payload": private}, ref)


def test_synthesis_trigger_ranking_and_counterexample_advisory(tmp_path):
    trigger = synthesis_trigger(
        novel_rejections=3,
        repeated_no_move=0,
        evidence_roles=("critic", "definition_auditor"),
    )
    assert trigger.invoke and trigger.reason == "NOVEL_SEMANTIC_REJECTIONS"
    candidates = _candidate_set()
    ranking = rank_candidates(candidates)
    selected = candidates.resolve(ranking.selected_candidate_id)
    assert selected.move.move_id == "SINGULARITY_CONTRADICTION"
    ref = persist_private_scratchpad(
        tmp_path, role="synthesis_scratchpad",
        transcript="private comparison", token_count=2,
    )
    manifest = synthesis_manifest(
        evidence_hashes={"counterexample_worker": "c" * 64},
        rejected_reason_codes=("NOT_STRICTLY_SIMPLER",),
        theorem_card_ids=("locally_uniform_limit_holomorphic",),
        scratchpad_ref=ref,
        ranking=ranking,
        short_choice_map_hash=candidates.short_choice_map_hash,
        selected_choice_code="I",
        counterexample_verified=False,
    )
    assert manifest["counterexample_premise_policy"] == "ADVISORY_ONLY"
    assert "private comparison" not in str(manifest)


def test_synthesis_transport_uses_only_scoped_short_choice_codes():
    candidates = _candidate_set()
    assert len(candidates.candidates) == 9
    assert candidates.short_choice_codes == tuple("ABCDEFGHI")
    assert candidates.short_choice_map_hash == _candidate_set().short_choice_map_hash
    registered = {"choice_code": candidates.short_choice_codes}
    decoded = decode_role_fields(
        "choice_code I;\nreason_code DIRECT_LOCAL_CONTRADICTION;\nEND;",
        "synthesis",
        registered_choices=registered,
    )
    ranking = rank_short_choice(
        candidates,
        choice_code=str(decoded.values["choice_code"]),
        reason_code=str(decoded.values["reason_code"]),
        candidate_set_hash=candidates.content_hash,
        short_choice_map_hash=candidates.short_choice_map_hash,
    )
    assert candidates.resolve(ranking.selected_candidate_id).move.move_id == (
        "SINGULARITY_CONTRADICTION"
    )
    assert candidates.resolve_short_code(
        "I",
        candidate_set_hash=candidates.content_hash,
        short_choice_map_hash=candidates.short_choice_map_hash,
    ).candidate_hash == (
        candidates.short_choice_map["I"].candidate_hash
    )

    invalid_codes = (
        "C1_DENSITY_LOWER_BOUND",
        "C1_DENSITY_LOWER_BOUN",
        "a",
        "A ",
        "AA",
        "J",
    )
    for code in invalid_codes:
        with pytest.raises(AdapterError):
            decode_role_fields(
                f"choice_code {code};\nreason_code LOWEST_COMPLEXITY;\nEND;",
                "synthesis",
                registered_choices=registered,
            )
    with pytest.raises(AdapterError, match="DUPLICATE_FIELD"):
        decode_role_fields(
            "choice_code A;\nchoice_code B;\n"
            "reason_code LOWEST_COMPLEXITY;\nEND;",
            "synthesis",
            registered_choices=registered,
        )

    prompt = transport_prompt("synthesis", registered_choices=registered)
    assert "choice_code" in prompt
    assert "C1_DENSITY_LOWER_BOUND" not in prompt
    assert "candidate_id" not in prompt
    assert "ranked_candidate_id" not in prompt
    assert "selected_candidate_id" not in prompt


def test_semantic_stagnation_changes_decomposition_without_global_strategy():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSITION_STAGNATED.value,
        migration_event=MIGRATION_EVENT,
    )
    checkpoint.transition(
        ProofState.SYNTHESIS, "novel-rejections-threshold", strategy_reused=True,
    )
    checkpoint.transition(
        ProofState.DECOMPOSER, "change-case-partition", strategy_reused=True,
    )
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.strategy_reused


def test_actual_mathlib_cards_search_elaborate_and_stale_hash_fails():
    cards = build_theorem_card_index(ROOT)
    names = {card.theorem_name for card in cards}
    assert {
        "TendstoLocallyUniformlyOn.differentiableOn",
        "Complex.differentiableOn_tsum_of_summable_norm",
        "Complex.analyticAt_of_differentiable_on_punctured_nhds_of_continuousAt",
        "Complex.differentiableOn_update_limUnder_of_bddAbove",
        "AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq",
    } <= names
    retrieved = search_theorem_cards(
        cards, ("locally_uniform_limit", "removable_singularity"),
    )
    assert retrieved
    validate_theorem_cards(retrieved, project_root=ROOT)
    with pytest.raises(ValueError, match="STALE_THEOREM_CARD_SOURCE"):
        validate_theorem_cards(
            (replace(retrieved[0], source_hash="0" * 64),),
            project_root=ROOT,
        )


def test_atomic_v87_snapshot_and_migration_follows_dependencies(tmp_path):
    checkpoint_path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        ledger_version=87,
        strategy_reused=True,
    )
    from autoresearch.prefill.orchestration_state import save_checkpoint
    save_checkpoint(checkpoint_path, checkpoint)
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"version":87}', encoding="utf-8")
    snapshot = atomic_snapshot(
        snapshot_root=tmp_path / "snapshots",
        files=(checkpoint_path, ledger),
        old_supervisor_pid=45556,
    )
    assert snapshot.is_dir()
    manifest = __import__("json").loads(
        (snapshot / "manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest["old_supervisor_pid"] == 45556
    assert len(manifest["files"]) == 2
    migrate_checkpoint(checkpoint_path, snapshot)
    from autoresearch.prefill.orchestration_state import load_checkpoint
    migrated = load_checkpoint(checkpoint_path)
    assert migrated.proof_state == ProofState.DEFINITION_AUDITOR
    assert migrated.migration_event == ARCHITECTURE_MIGRATION_EVENT
    assert migrated.migration_snapshot == str(snapshot)
    assert migrated.strategy_reused


def _missing_definition_artifact(*definition_ids):
    return {
        "definitions": [],
        "missing_definitions": [
            {"definition_id": item, "obligation_label": f"D{index + 1}"}
            for index, item in enumerate(definition_ids)
        ],
    }


def test_production_regression_density_ineligible_reason_checked_and_codes_remap():
    cards = build_theorem_card_index(ROOT)
    dependency = "d" * 64
    unfiltered = build_candidate_set(
        target_ref="claim:" + "a" * 64,
        viewpoint="regression",
        dependency_ids=(dependency,),
        theorem_cards=cards,
        satisfied_precondition_ids={
            item
            for move in MOVE_REGISTRY.values()
            for item in move.precondition_ids
            if item != "registered_definition_gap"
        },
        satisfied_theorem_hypothesis_ids={
            item for card in cards for item in card.required_hypotheses
        },
        available_dependency_artifact_ids=(dependency,),
        required_dependency_artifact_ids=(dependency,),
    )
    density = next(
        item for item in unfiltered.candidates
        if item.move.move_id == "DENSITY_LOWER_BOUND"
    )
    density_code = next(
        code for code, item in unfiltered.short_choice_map.items()
        if item == density
    )
    with pytest.raises(ValueError, match="INVALID_RANK_REASON_METRIC"):
        rank_short_choice(
            unfiltered,
            choice_code=density_code,
            reason_code="LOWEST_COMPLEXITY",
            candidate_set_hash=unfiltered.content_hash,
            short_choice_map_hash=unfiltered.short_choice_map_hash,
        )

    filtered = build_candidate_set(
        target_ref="claim:" + "a" * 64,
        viewpoint="regression",
        dependency_ids=(dependency,),
        theorem_cards=cards,
        satisfied_precondition_ids=("registered_definition_gap",),
        available_dependency_artifact_ids=(dependency,),
        required_dependency_artifact_ids=(dependency,),
    )
    assert filtered.candidates == ()
    assert any(
        move_id == "DENSITY_LOWER_BOUND"
        and "UNMET_PRECONDITION" in reasons
        for move_id, reasons in filtered.ineligible
    )
    assert filtered.short_choice_map == {}


def test_evidence_planner_provenance_advisory_exclusion_and_multiple_plans():
    artifact_hash = "a" * 64
    artifact = _missing_definition_artifact(
        "DEF_SEQUENCE_DENSITY", "DEF_CRITICAL_DENSITY",
    )
    context = host_evidence_context(
        {"definition_auditor": artifact},
        artifact_hashes={"definition_auditor": artifact_hash},
    )
    candidates = build_candidate_set(
        target_ref="claim:" + "b" * 64,
        viewpoint="recorded_gaps",
        satisfied_precondition_ids=context.satisfied_precondition_ids,
    )
    graph = build_evidence_gap_graph(
        artifacts={"definition_auditor": artifact},
        artifact_hashes={"definition_auditor": artifact_hash},
        advisory_artifacts={
            "c" * 64: {"role": "counterexample_worker", "verified": False},
        },
        theorem_cards=(),
        candidate_set=candidates,
    )
    gaps = [node for node in graph.nodes if node.kind == NodeKind.GAP.value]
    assert any(artifact_hash in node.provenance_refs for node in gaps)
    advisory_ids = {
        node.node_id for node in graph.nodes
        if node.verification_status == VerificationStatus.ADVISORY.value
    }
    assert advisory_ids
    assert not any(
        edge.kind == EdgeKind.SUPPORTS.value and edge.source_id in advisory_ids
        for edge in graph.edges
    )
    plans = generate_proof_plans(
        graph, unresolved_gap_ids=context.unresolved_gap_ids,
    )
    # Definition gaps are completed by Host DEFINE_ONE_CONCEPT before planning.
    assert plans == ()
    assert plans == generate_proof_plans(
        graph, unresolved_gap_ids=context.unresolved_gap_ids,
    )


def test_plan_dependency_cycle_case_and_restart_fail_closed():
    artifact = _missing_definition_artifact("DEF_A", "DEF_B")
    context = host_evidence_context(
        {"definition_auditor": artifact},
        artifact_hashes={"definition_auditor": "a" * 64},
    )
    candidates = build_candidate_set(
        target_ref="claim:" + "c" * 64,
        viewpoint="gaps",
        satisfied_precondition_ids=context.satisfied_precondition_ids,
    )
    graph = build_evidence_gap_graph(
        artifacts={"definition_auditor": artifact},
        artifact_hashes={"definition_auditor": "a" * 64},
        advisory_artifacts={},
        theorem_cards=(),
        candidate_set=candidates,
    )
    plans = generate_proof_plans(
        graph, unresolved_gap_ids=context.unresolved_gap_ids,
    )
    assert plans == ()


def test_theorem_hypothesis_gate_replan_and_generic_scoring_static_guard():
    cards = build_theorem_card_index(ROOT)
    preconditions = {
        item
        for move in MOVE_REGISTRY.values()
        for item in move.precondition_ids
        if item != "registered_definition_gap"
    }
    gated = build_candidate_set(
        target_ref="claim:" + "d" * 64,
        viewpoint="theorem_gate",
        theorem_cards=cards,
        satisfied_precondition_ids=preconditions,
    )
    assert any(
        "UNMET_THEOREM_HYPOTHESIS" in reasons
        for _move_id, reasons in gated.ineligible
    )

    artifact = _missing_definition_artifact("DEF_A", "DEF_B")
    context = host_evidence_context(
        {"definition_auditor": artifact},
        artifact_hashes={"definition_auditor": "a" * 64},
    )
    candidates = build_candidate_set(
        target_ref="claim:" + "e" * 64,
        viewpoint="replan",
        satisfied_precondition_ids=context.satisfied_precondition_ids,
    )
    graph = build_evidence_gap_graph(
        artifacts={"definition_auditor": artifact},
        artifact_hashes={"definition_auditor": "a" * 64},
        advisory_artifacts={},
        theorem_cards=(),
        candidate_set=candidates,
    )
    plans = generate_proof_plans(
        graph, unresolved_gap_ids=context.unresolved_gap_ids,
    )
    assert plans == ()

    import inspect
    from autoresearch.prefill import evidence_planner
    scoring_source = inspect.getsource(evidence_planner.generate_proof_plans)
    for forbidden in (
        "DENSITY_LOWER_BOUND",
        "LOCAL_CONVERGENCE_OBLIGATION",
        "HOLOMORPHIC_EXTENSION",
        "SINGULARITY_CONTRADICTION",
    ):
        assert forbidden not in scoring_source
