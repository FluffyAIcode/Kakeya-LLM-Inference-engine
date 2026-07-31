import hashlib
import json
import os
import pytest
from dataclasses import asdict
from types import SimpleNamespace

from autoresearch.prefill.live_status import AtomicLiveStatus
from autoresearch.prefill.prepare import ReportValidationError
from autoresearch.prefill.orchestration_state import (
    ARCHITECTURE_VERSION,
    BlockedEventType,
    BlockedExitEvent,
    OrchestrationCheckpoint,
    ProofState,
    apply_blocked_exit_event,
    archive_decomposition_rejection,
    classify_host_gate_defects,
    classify_failure,
    compact_decomposition_novelty_ledger,
    earliest_invalid_role,
    load_checkpoint as load_orchestration_checkpoint,
    persist_validated_artifact,
    save_checkpoint as save_orchestration_checkpoint,
)
from autoresearch.prefill.semantic_decompose import (
    SemanticUnitTooLarge,
    admit_token_ids,
    downstream_output_cap,
)
from autoresearch.prefill.supervisor import (
    BLOCKED_HEARTBEAT_INTERVAL_S,
    RESUMABLE_ORCHESTRATION_STATES,
    BenchmarkFinalizationTimeout,
    BenchmarkTerminalFailure,
    BlockedIdleLogger,
    CandidateNoveltyStagnation,
    append_result,
    best_kept,
    build_host_candidate,
    build_strategy_contract,
    build_strategy_prompt,
    build_strategy_research_state,
    check_runtime_health,
    extract_gan_failure_reason,
    failure_class_for_exception,
    infrastructure_failure_fingerprint,
    is_contract_bound_subgoal_resume,
    is_resumable_checkpoint,
    is_nonfatal_semantic_continuation,
    parse_strategy_candidate_transport,
    parse_research_verdict,
    read_results,
    reconcile_canonical_root_binding,
    repair_research_contract_artifact_dependency,
    repair_candidate_schema,
    recover_contract_subgoal_duplicate_block,
    render_candidate,
    route_contract_to_subgoal_generation,
    run_supervisor_iterations,
    select_novel_candidate,
    should_resume_downstream,
    should_keep,
    StrategyPrefillHeartbeat,
    StrategyPrefillBudgetExceeded,
    strategy_trigger_reason,
    _extract_json,
    _pending_leaf_ids,
    validate_candidate,
    wait_for_benchmark_finalization,
)
from pathlib import Path


def test_contract_requires_elaborated_subgoal_before_oprover():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.PROOF_SEARCH.value,
        current_role="proof_search",
        target_obligation_id="RH-C0-root",
        target_statement="RiemannHypothesis",
        proposition_hash="p" * 64,
        selected_strategy_plan_id="SP-root",
        research_contract_id="RC-root",
        adapter_status="INTEGRATION_BLOCKED",
        blocked_reason=(
            "PROOF_ADVISOR_UNAVAILABLE:"
            "RESIDENCY_PROCESS_MANAGER_REQUIRED"
        ),
    )
    assert route_contract_to_subgoal_generation(checkpoint)
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.adapter_status == ""
    assert checkpoint.recovery_events[-1]["event_type"] == (
        "RESEARCH_CONTRACT_SUBGOAL_REQUIRED"
    )
    assert not route_contract_to_subgoal_generation(checkpoint)


def test_contract_with_executable_subgoal_does_not_backjump():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.PROOF_SEARCH.value,
        current_role="proof_search",
        target_obligation_id="RH-C0-root",
        proposition_hash="p" * 64,
        research_contract_id="RC-root",
        proof_plan_id="PP-child",
        executable_plan_node_id="L1",
    )
    assert not route_contract_to_subgoal_generation(checkpoint)
    assert checkpoint.proof_state == ProofState.PROOF_SEARCH


def test_target_bound_decomposer_resume_ignores_wrapper_candidate_hash():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        target_obligation_id="RH-C0-root",
        proposition_hash="p" * 64,
        target_context_hash="c" * 64,
        selected_strategy_plan_id="SP-root",
        research_contract_id="RC-root",
        candidate_sha256="",
    )
    assert is_contract_bound_subgoal_resume(checkpoint)
    assert should_resume_downstream(
        checkpoint,
        candidate_sha256="different-wrapper-hash",
        force_strategy=False,
        strategy_trigger_exists=False,
    )


def test_contract_bound_definition_resolution_preserves_provenance_on_wrapper_drift():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_RESOLUTION.value,
        current_role="definition_resolution",
        target_obligation_id="RH-C0-root",
        proposition_hash="p" * 64,
        target_context_hash="c" * 64,
        selected_strategy_plan_id="SP-root",
        research_contract_id="RC-root",
        candidate_sha256="stale-wrapper-hash",
        definition_audit_outcome="COMPLETE",
    )
    assert is_contract_bound_subgoal_resume(checkpoint)
    assert should_resume_downstream(
        checkpoint,
        candidate_sha256="new-wrapper-hash",
        force_strategy=False,
        strategy_trigger_exists=False,
    )


def test_duplicate_wrapper_block_recovers_bound_decomposer():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.BLOCKED.value,
        current_role="blocked",
        target_obligation_id="RH-C0-root",
        proposition_hash="p" * 64,
        target_context_hash="c" * 64,
        selected_strategy_plan_id="SP-root",
        research_contract_id="RC-root",
        blocked_reason=(
            "Strategy proposals were duplicates; reuse the current candidate "
            "and unresolved role."
        ),
        recovery_events=[{
            "event_type": "RESEARCH_CONTRACT_SUBGOAL_REQUIRED",
            "event_id": "e" * 64,
            "target_obligation_id": "RH-C0-root",
            "research_contract_id": "RC-root",
        }],
    )
    event = recover_contract_subgoal_duplicate_block(checkpoint)
    assert event is not None
    assert event.event_type == (
        BlockedEventType.VALIDATED_EVIDENCE_BACKJUMP.value
    )
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.blocked_reason == ""
    assert recover_contract_subgoal_duplicate_block(checkpoint) is None


def test_research_contract_dependency_repairs_to_artifact_hash(tmp_path):
    checkpoint = OrchestrationCheckpoint(
        strategy_tournament_hash="tournament-content-hash",
        research_contract_id="RC-root",
        target_obligation_id="RH-C0-root",
    )
    path = tmp_path / "orchestration.json"
    tournament = persist_validated_artifact(
        path,
        checkpoint,
        role="strategy_tournament",
        payload={"schema_version": 1, "plans": []},
        dependencies=[],
        source_run_id="host:strategy",
    )
    contract = persist_validated_artifact(
        path,
        checkpoint,
        role="research_contract",
        payload={"schema_version": 1, "contract_id": "RC-root"},
        dependencies=[checkpoint.strategy_tournament_hash],
        source_run_id="host:contract",
    )
    assert contract.dependencies != [tournament.sha256]
    assert repair_research_contract_artifact_dependency(checkpoint)
    assert contract.dependencies == [tournament.sha256]
    assert not repair_research_contract_artifact_dependency(checkpoint)


def test_stale_root_goal_cache_reconciles_without_losing_contract(tmp_path):
    canonical = hashlib.sha256(b"RiemannHypothesis").hexdigest()
    state_path = tmp_path / "agent_state.json"
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "research_goal": "stale Hilbert-Polya cache",
    }))
    ledger = {
        "ledger_id": "rh",
        "version": 96,
        "obligations": [{
            "obligation_id": "RH-C0-root",
            "statement": "RiemannHypothesis",
            "status": "UNRESOLVED",
            "parent_id": "",
            "formal_status": "FORMALIZED",
            "lean_signature_hash": canonical,
            "proposition_hash": canonical,
        }],
    }
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        target_obligation_id="RH-C0-root",
        target_statement="RiemannHypothesis",
        proposition_hash=canonical,
        parent_statement_sha256=canonical,
        parent_signature_sha256=canonical,
        root_goal_sha256=hashlib.sha256(b"stale").hexdigest(),
        selected_strategy_plan_id="SP-root",
        research_contract_id="RC-root",
        research_contract_hash="contract-hash",
    )
    assert reconcile_canonical_root_binding(
        checkpoint,
        ledger,
        state_path=state_path,
    )
    assert checkpoint.root_goal_sha256 == canonical
    assert checkpoint.research_contract_id == "RC-root"
    assert checkpoint.research_contract_hash == "contract-hash"
    assert ledger["root_goal_hash"] == canonical
    assert json.loads(state_path.read_text())["research_goal"] == (
        "RiemannHypothesis"
    )
    assert not reconcile_canonical_root_binding(
        checkpoint,
        ledger,
        state_path=state_path,
    )


def test_true_canonical_proposition_mismatch_preserves_checkpoint(tmp_path):
    canonical = hashlib.sha256(b"RiemannHypothesis").hexdigest()
    state_path = tmp_path / "agent_state.json"
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "research_goal": "stale cache",
    }))
    ledger = {"obligations": [{
        "obligation_id": "RH-C0-root",
        "statement": "RiemannHypothesis",
        "parent_id": "",
        "formal_status": "FORMALIZED",
        "proposition_hash": canonical,
    }]}
    checkpoint = OrchestrationCheckpoint(
        target_obligation_id="RH-C0-root",
        proposition_hash="f" * 64,
        research_contract_id="RC-root",
        research_contract_hash="contract-hash",
    )
    before = asdict(checkpoint)
    with pytest.raises(
        ValueError,
        match="CANONICAL_ROOT_PROPOSITION_MISMATCH",
    ):
        reconcile_canonical_root_binding(
            checkpoint,
            ledger,
            state_path=state_path,
        )
    assert asdict(checkpoint) == before
    assert json.loads(state_path.read_text())["research_goal"] == "stale cache"


def test_live_status_atomic_transitions_and_permissions(tmp_path):
    path = tmp_path / "proof_live_status.json"
    status = AtomicLiveStatus(
        path,
        supervisor_pid=os.getpid(),
        iteration=7,
        run_id="br_test",
        min_interval_s=0,
    )
    assert status.emit(
        phase="generator_prefill",
        role="generator",
        state="prefill",
        progress_current=128,
        progress_total=512,
        progress_unit="tokens",
        active_obligation_id="RH-C2-child",
        worker="allens",
        source="agent_gan_repl",
        force=True,
    )
    first = json.loads(path.read_text())
    assert first["sequence"] == 1
    assert first["state"] == "prefill"
    assert first["progress"] == {
        "current": 128,
        "total": 512,
        "unit": "tokens",
    }
    status.emit(
        phase="generator_decode",
        role="generator",
        state="decode",
        progress_current=32,
        progress_total=320,
        progress_unit="tokens",
        active_obligation_id="RH-C2-child",
        worker="primary",
        source="agent_gan_repl",
        hit_source="primary_hot",
        force=True,
    )
    second = json.loads(path.read_text())
    assert second["sequence"] == 2
    assert second["state"] == "decode"
    assert second["started_at"] >= first["started_at"]
    assert path.stat().st_mode & 0o077 == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_live_status_never_serializes_private_values(tmp_path):
    path = tmp_path / "proof_live_status.json"
    status = AtomicLiveStatus(
        path,
        supervisor_pid=os.getpid(),
        min_interval_s=0,
    )
    status.emit(
        phase="/Users/private/prompt",
        role="api_key=secret",
        state="review",
        active_obligation_id="ROOT",
        source="agent_gan_repl",
        force=True,
    )
    serialized = path.read_text()
    assert "/Users/" not in serialized
    assert "private/prompt" not in serialized
    assert "secret" not in serialized


def test_live_status_reports_exact_orchestration_state(tmp_path, monkeypatch):
    state_path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        resume_origin="DECOMPOSER",
        last_transition_reason="decomposer protocol repair failed",
        strategy_reused=True,
        retry_counters={"DECOMPOSER": 1},
        active_gate="HOST_TYPED_IR_GATE",
        adapter_status="",
        typed_ir_hash="b" * 64,
        proposition_hash="c" * 64,
        elaborated_theorem_id="typed_test",
        lean_contract_id="lean-signature-test",
        lean_contract_version=1,
        lean_symbol_table_id="lean-symbols-test",
        lean_symbol_table_version=1,
        formalizer_unit_hashes={"PARENT_SIGNATURE": "a" * 64},
        candidate_count=9,
        exploration_contract_id="DEC-test",
        exploration_selected_candidate_ids=["XC-1", "XC-2", "XC-3"],
        exploration_current_index=1,
        exploration_current_candidate_id="XC-2",
        exploration_formalization_status="PENDING",
        exploration_reduction_status="NOT_STARTED",
        exploration_rejections={"XC-0": ["EXACT_DUPLICATE"]},
        exploration_candidate_refs=[{
            "candidate_id": "XC-1",
            "memo_sha256": "m" * 64,
        }],
    )
    save_orchestration_checkpoint(state_path, checkpoint)
    monkeypatch.setenv(
        "KAKEYA_ORCHESTRATION_STATE_PATH",
        str(state_path),
    )
    live_path = tmp_path / "proof_live_status.json"
    status = AtomicLiveStatus(
        live_path,
        supervisor_pid=os.getpid(),
        min_interval_s=0,
    )
    status.emit(
        phase="decomposer_prefill",
        role="decomposer",
        state="prefill",
        force=True,
    )
    live = json.loads(live_path.read_text())
    assert live["orchestration_state"] == "DECOMPOSER"
    assert live["active_role"] == "decomposer"
    assert live["resume_origin"] == "DECOMPOSER"
    assert live["retry_count"] == 1
    assert live["strategy_reused"] is False
    assert live["architecture_version"] == ARCHITECTURE_VERSION
    assert live["active_gate"] == "HOST_TYPED_IR_GATE"
    assert live["typed_ir_hash"] == "b" * 64
    assert live["proposition_hash"] == "c" * 64
    assert live["elaborated_theorem_id"] == "typed_test"
    assert live["lean_contract_id"] == "lean-signature-test"
    assert live["lean_contract_version"] == 1
    assert live["lean_symbol_table_id"] == "lean-symbols-test"
    assert live["lean_symbol_table_version"] == 1
    assert live["validated_formalizer_unit_hashes"] == {
        "PARENT_SIGNATURE": "a" * 64,
    }
    exploration = live["decomposition_exploration"]
    assert exploration["generated"] == 9
    assert exploration["selected_candidate_ids"] == ["XC-1", "XC-2", "XC-3"]
    assert exploration["current_candidate_id"] == "XC-2"
    assert exploration["rejected_reason_codes"] == ["EXACT_DUPLICATE"]
    assert "memo_sha256" not in json.dumps(exploration)


@pytest.mark.parametrize(
    "state",
    (
        ProofState.MATH_IR_TRANSLATION,
        ProofState.HOST_TYPED_IR_GATE,
        ProofState.LEAN_ELABORATION_GATE,
        ProofState.PROOF_SEARCH,
    ),
)
def test_zero_agent_gate_checkpoints_resume_before_strategy_replanning(state):
    assert state in RESUMABLE_ORCHESTRATION_STATES


def test_blocked_status_exposes_resume_role_without_private_state(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "proof_orchestration.json"
    save_orchestration_checkpoint(
        state_path,
        OrchestrationCheckpoint(
            state=ProofState.DEFINITION_AUDITOR.value,
            current_role="definition_auditor",
            adapter_status="ADAPTER_BLOCKED",
            blocked_reason="legacy definition transport malformed",
        ),
    )
    monkeypatch.setenv("KAKEYA_ORCHESTRATION_STATE_PATH", str(state_path))
    live_path = tmp_path / "proof_live_status.json"
    AtomicLiveStatus(
        live_path,
        supervisor_pid=os.getpid(),
        min_interval_s=0,
    ).emit(
        phase="blocked_idle",
        role="orchestrator",
        state="idle",
        force=True,
    )
    live = json.loads(live_path.read_text())
    assert live["execution_state"] == "idle"
    assert live["execution_phase"] == "blocked_idle"
    assert live["adapter_status"] == "ADAPTER_BLOCKED"
    assert live["blocked_category"] == "ADAPTER_BLOCKED"
    assert live["blocked_reason"] == "legacy definition transport malformed"
    assert live["resume_role"] == "definition_auditor"
    assert BLOCKED_HEARTBEAT_INTERVAL_S <= 30


def test_live_status_distinguishes_decomposition_search_fields(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        decomposition_iteration=12,
        viewpoint="local_global_bridge",
        semantic_rejection={"proposal_sha256": "a" * 64},
        novel_proposals=11,
        strategy_reused=True,
    )
    save_orchestration_checkpoint(state_path, checkpoint)
    monkeypatch.setenv("KAKEYA_ORCHESTRATION_STATE_PATH", str(state_path))
    live_path = tmp_path / "proof_live_status.json"
    AtomicLiveStatus(
        live_path,
        supervisor_pid=os.getpid(),
        min_interval_s=0,
    ).emit(
        phase="decomposer_decode",
        role="decomposer",
        state="decode",
        force=True,
    )
    live = json.loads(live_path.read_text())
    assert live["decomposition_iteration"] == 12
    assert ("protocol_" + "attempt") not in live
    assert live["viewpoint"] == "local_global_bridge"
    assert live["semantic_rejection"] is True
    assert live["novel_proposals"] == 11


def test_semantic_proposal_archive_does_not_consume_adapter_budget(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        decomposition_iteration=1,
        viewpoint="definitions",
        strategy_reused=True,
    )
    for index in range(11):
        archive_decomposition_rejection(
            path,
            checkpoint,
            proposal={"child": {"statement": f"proposal {index}"}},
            rejection_reasons=["child is not strictly simpler"],
            semantic_hash=f"{index:064x}",
            structural_signature=f"{index + 20:064x}",
            source_run_id=f"run-{index}",
        )
        checkpoint.begin_decomposition_iteration(
            f"viewpoint-{index}",
            "semantic proposal rejected",
        )
        save_orchestration_checkpoint(path, checkpoint)
    restarted = load_orchestration_checkpoint(path)
    assert restarted.decomposition_iteration == 12
    assert restarted.retry_counters == {}
    assert ("protocol_" + "attempt") not in restarted.__dataclass_fields__
    assert restarted.novel_proposals == 11
    assert restarted.strategy_reused is False
    assert restarted.proof_state == ProofState.DECOMPOSER
    compact = compact_decomposition_novelty_ledger(restarted)
    assert compact["count"] == 11
    assert compact["novel"] == 11
    assert len(compact["recent"]) == 2
    assert compact["duplicate_gate"] == (
        "semantic_hash+structural_signature"
    )
    assert compact["viewpoints"]["count"] == 11
    assert compact == compact_decomposition_novelty_ledger(
        load_orchestration_checkpoint(path),
    )
    manifest_hash = compact["manifest"].removeprefix("sha256:")
    manifest_path = (
        path.with_suffix(".semantic-proposals")
        / "manifests"
        / f"{manifest_hash}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["records"]) == 11
    assert manifest["records"][0]["rejection_reasons"] == [
        "child is not strictly simpler",
    ]
    assert manifest["records"][0]["rejection_reason_codes"] == [
        "NOT_STRICTLY_SIMPLER",
    ]
    assert len(json.dumps(compact)) < 2052 * 4


@pytest.mark.parametrize("proposal_count", [10, 50, 100])
def test_decomposition_novelty_context_is_bounded(
    tmp_path,
    proposal_count,
):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
    )
    for index in range(proposal_count):
        checkpoint.viewpoint = f"viewpoint-{index}"
        checkpoint.viewpoints_tried.append(checkpoint.viewpoint)
        archive_decomposition_rejection(
            path,
            checkpoint,
            proposal={
                "child": {
                    "statement": (
                        f"Whole semantic statement {index}; never truncated."
                    ),
                },
            },
            rejection_reasons=[
                "structural-signature-duplicate; no genuine delta",
            ],
            semantic_hash=f"{index:064x}",
            structural_signature=f"{index + 1000:064x}",
            source_run_id=f"run-{index}",
        )
    compact = compact_decomposition_novelty_ledger(checkpoint)
    assert len(compact["recent"]) == 2
    assert len(compact["viewpoints"]["base_ids"]) <= 7
    assert len(compact["viewpoints"]["recent_ids"]) <= 2
    assert len(json.dumps(compact, sort_keys=True)) < 1600
    manifest_hash = compact["manifest"].removeprefix("sha256:")
    manifest_path = (
        path.with_suffix(".semantic-proposals")
        / "manifests"
        / f"{manifest_hash}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["records"]) == proposal_count
    final_proposal = (
        path.with_suffix(".semantic-proposals")
        / f"{proposal_count - 1:064x}.json"
    )
    # Artifact filenames are proposal-content hashes, not semantic hashes.
    archived_path = checkpoint.decomposition_proposals[-1]["path"]
    archived = json.loads(open(archived_path, encoding="utf-8").read())
    assert archived["child"]["statement"].endswith("never truncated.")
    assert not final_proposal.exists()


def test_orchestration_transition_table_and_retry_block(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint()
    checkpoint.transition(
        ProofState.RESEARCH_CONTRACT_GATE,
        "tournament-complete",
    )
    checkpoint.transition(ProofState.DECOMPOSER, "missing-definitions")
    assert checkpoint.retry(
        ProofState.DECOMPOSER,
        "malformed JSON",
        1,
    )
    assert not checkpoint.retry(
        ProofState.DECOMPOSER,
        "missing EOS",
        1,
    )
    assert checkpoint.proof_state == ProofState.BLOCKED
    assert "retry budget exhausted" in checkpoint.blocked_reason
    save_orchestration_checkpoint(path, checkpoint)
    assert load_orchestration_checkpoint(path).proof_state == ProofState.BLOCKED
    assert path.stat().st_mode & 0o077 == 0


def test_blocked_is_quiescent_until_typed_event():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        retry_counters={"DECOMPOSER": 2, "PROVER": 1},
    )
    assert not checkpoint.retry(ProofState.DECOMPOSER, "bad artifact", 1)
    before = dict(checkpoint.retry_counters)
    assert not checkpoint.retry(ProofState.DECOMPOSER, "bad artifact", 1)
    assert checkpoint.retry_counters == before
    with pytest.raises(ValueError, match="explicit typed event"):
        checkpoint.transition(ProofState.DECOMPOSER, "automatic retry")
    event = BlockedExitEvent(
        event_id="evt-1",
        event_type=BlockedEventType.OPERATOR_UNBLOCK.value,
        reason="protocol_parser_hardened",
        target_state=ProofState.DECOMPOSER.value,
        reset_role=ProofState.DECOMPOSER.value,
    )
    apply_blocked_exit_event(checkpoint, event)
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.retry_counters == {"DECOMPOSER": 0, "PROVER": 1}
    assert checkpoint.last_blocked_event_id == "evt-1"


def test_seven_host_gate_defects_backjump_and_persist_invalidation(tmp_path):
    errors = [
        "claimed counterexample has no verified evidence",
        "reduction theorem conclusion differs from exact parent proposition",
        "parent signature failed: expected exactly one theorem declaration",
        "child L1 signature failed: forbidden Lean command in generated signature",
        "child L1 rejected: bidirectionally entails ancestor ROOT",
        "reduction theorem signature failed or changed",
        "complete reduction proof failed or targets another theorem",
    ]
    defects = classify_host_gate_defects(errors)
    assert [item.code for item in defects] == [
        "UNVERIFIED_COUNTEREXAMPLE",
        "REDUCTION_CONCLUSION_MISMATCH",
        "PARENT_SIGNATURE_INVALID",
        "CHILD_SIGNATURE_INVALID",
        "CYCLIC_OR_EQUIVALENT_CHILD",
        "REDUCTION_SIGNATURE_INVALID",
        "REDUCTION_PROOF_INVALID",
    ]
    assert defects[0].hard_invalid is False
    assert earliest_invalid_role(defects) == ProofState.DECOMPOSER

    checkpoint = OrchestrationCheckpoint(
        state=ProofState.BLOCKED.value,
        current_role="blocked",
        validated_artifacts={},
    )
    path = tmp_path / "orchestration.json"
    hashes = {}
    dependencies = {
        "definition_auditor": [],
        "counterexample_worker": [],
        "decomposer": [],
        "formalizer": [],
        "prover": [],
        "adversarial_proponent": [],
    }
    for role in dependencies:
        ref = persist_validated_artifact(
            path,
            checkpoint,
            role=role,
            payload={"role": role},
            dependencies=dependencies[role],
            source_run_id=f"run:{role}",
        )
        hashes[role] = ref.sha256
    invalidated = {
        role: hashes[role]
        for role in (
            "decomposer",
            "formalizer",
            "prover",
            "adversarial_proponent",
        )
    }
    reused = {
        role: hashes[role]
        for role in ("definition_auditor", "counterexample_worker")
    }
    event = BlockedExitEvent(
        event_id="host_gate_defects_backjump",
        event_type=BlockedEventType.HOST_GATE_DEFECTS_BACKJUMP.value,
        reason="seven exact host defects classified",
        target_state=ProofState.DECOMPOSER.value,
        reset_role=ProofState.DECOMPOSER.value,
        metadata={
            "defects": [
                {
                    "code": item.code,
                    "source_role": item.source_role,
                    "message": item.message,
                }
                for item in defects
            ],
            "invalidated_artifact_hashes": invalidated,
            "reuse_map": reused,
        },
    )
    apply_blocked_exit_event(checkpoint, event)
    save_orchestration_checkpoint(path, checkpoint)
    restarted = load_orchestration_checkpoint(path)
    assert restarted.proof_state == ProofState.DECOMPOSER
    assert set(restarted.validated_artifacts) == {
        "definition_auditor",
        "counterexample_worker",
    }
    assert set(restarted.invalidated_artifacts) == set(invalidated.values())
    advisory = restarted.advisory_artifacts[
        hashes["counterexample_worker"]
    ]
    assert advisory["verified"] is False
    assert advisory["premise"] is False
    assert advisory["public"] is False
    assert advisory["certificate_gate"] is False
    assert restarted.retry_counters["DECOMPOSER"] == 0


def test_identical_state_artifact_error_cycle_blocks_early():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
    )
    assert checkpoint.retry(ProofState.DECOMPOSER, "same malformed JSON", 9)
    assert not checkpoint.retry(
        ProofState.DECOMPOSER,
        "same   malformed JSON",
        9,
    )
    assert checkpoint.proof_state == ProofState.BLOCKED
    assert "identical state/artifact/error cycle" in checkpoint.blocked_reason


def test_supervisor_restarts_stay_blocked_without_inference(tmp_path, monkeypatch):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.BLOCKED.value,
        current_role="blocked",
        blocked_reason="repair exhausted",
    )
    save_orchestration_checkpoint(path, checkpoint)
    calls = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda *_args: calls.append(True),
    )
    status = SimpleNamespace(emit=lambda **_kwargs: None)
    args = SimpleNamespace(
        iterations=3,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(tmp_path / "no-event.json"),
        blocked_policy="wait",
        blocked_poll_interval_s=0,
        _live_status=status,
    )
    assert run_supervisor_iterations(args) == 0
    assert calls == []
    assert load_orchestration_checkpoint(path).proof_state == ProofState.BLOCKED


def test_many_blocked_ticks_log_once_and_rate_limit_heartbeat(
    tmp_path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "proof_orchestration.json"
    live_path = tmp_path / "proof_live_status.json"
    save_orchestration_checkpoint(
        path,
        OrchestrationCheckpoint(
            state=ProofState.BLOCKED.value,
            current_role="blocked",
            blocked_reason="one long unchanged blocker",
        ),
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda *_args: pytest.fail("BLOCKED must not start inference"),
    )
    status = AtomicLiveStatus(
        live_path,
        supervisor_pid=os.getpid(),
        min_interval_s=0,
    )
    args = SimpleNamespace(
        iterations=20,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(tmp_path / "no-event.json"),
        blocked_policy="wait",
        blocked_poll_interval_s=0,
        _live_status=status,
    )
    assert run_supervisor_iterations(args) == 0
    output = capsys.readouterr().out
    assert output.count("phase=blocked-idle") == 1
    assert output.count("reason=one long unchanged blocker") == 1
    assert "blocked-liveness" not in output
    assert "suppressed_ticks" not in output
    assert json.loads(live_path.read_text())["sequence"] == 1


@pytest.mark.parametrize(
    "adapter_status",
    ("ADAPTER_BLOCKED", "INTEGRATION_BLOCKED"),
)
def test_adapter_blockers_sleep_without_inference(
    tmp_path,
    monkeypatch,
    adapter_status,
):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        adapter_status=adapter_status,
        blocked_reason="quiet adapter blocker",
    )
    save_orchestration_checkpoint(path, checkpoint)
    calls = []
    sleeps = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda *_args: calls.append(True),
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    args = SimpleNamespace(
        iterations=3,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(tmp_path / "no-event.json"),
        blocked_policy="wait",
        blocked_poll_interval_s=30,
        blocked_heartbeat_interval_s=60,
        _live_status=SimpleNamespace(emit=lambda **_kwargs: None),
    )
    assert run_supervisor_iterations(args) == 0
    assert calls == []


def test_legacy_definition_adapter_block_migrates_to_typed_run(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "proof_orchestration.json"
    save_orchestration_checkpoint(
        path,
        OrchestrationCheckpoint(
            state=ProofState.DEFINITION_AUDITOR.value,
            current_role="definition_auditor",
            adapter_status="ADAPTER_BLOCKED",
            blocked_reason=(
                "definition_auditor: malformed DEFINITION_AUDIT "
                "Artifact JSON: transport-incomplete Artifact JSON"
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda _args, iteration: calls.append(iteration) or {
            "research_outcome": "KEPT",
        },
    )
    args = SimpleNamespace(
        iterations=1,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(tmp_path / "no-event.json"),
        blocked_policy="wait",
        blocked_poll_interval_s=0,
        _live_status=SimpleNamespace(emit=lambda **_kwargs: None),
    )
    assert run_supervisor_iterations(args) == 0
    checkpoint = load_orchestration_checkpoint(path)
    assert calls == [0]
    assert checkpoint.adapter_status == ""
    assert checkpoint.recovery_events[-1]["event_type"] == (
        "LEGACY_DEFINITION_OUTPUT_AUDIT_ONLY"
    )


def test_quiescent_infrastructure_blocker_retries_inference(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "proof_orchestration.json"
    save_orchestration_checkpoint(
        path,
        OrchestrationCheckpoint(
            state=ProofState.SYNTHESIS.value,
            current_role="synthesis",
            adapter_status="INFRASTRUCTURE_BLOCKED",
            blocked_reason="transient cache route unavailable",
        ),
    )
    calls = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda _args, iteration: calls.append(iteration) or {},
    )
    args = SimpleNamespace(
        iterations=1,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(tmp_path / "no-event.json"),
        _live_status=SimpleNamespace(emit=lambda **_kwargs: None),
    )

    assert run_supervisor_iterations(args) == 0
    checkpoint = load_orchestration_checkpoint(path)
    assert calls == [0]
    assert checkpoint.adapter_status == ""
    assert checkpoint.blocked_reason == ""
    assert checkpoint.recovery_events[-1]["event_type"] == (
        "QUIESCENT_INFRASTRUCTURE_RETRY"
    )


def test_changed_blocked_reason_emits_one_new_full_log(
    tmp_path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.BLOCKED.value,
        current_role="blocked",
        blocked_reason="first blocker",
    )
    save_orchestration_checkpoint(path, checkpoint)
    changed = False

    def change_reason(_seconds):
        nonlocal changed
        if changed:
            return
        changed = True
        current = load_orchestration_checkpoint(path)
        current.blocked_reason = "second blocker"
        save_orchestration_checkpoint(path, current)

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.time.sleep",
        change_reason,
    )
    args = SimpleNamespace(
        iterations=4,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(tmp_path / "no-event.json"),
        blocked_policy="wait",
        blocked_poll_interval_s=0,
        _live_status=SimpleNamespace(emit=lambda **_kwargs: None),
    )
    assert run_supervisor_iterations(args) == 0
    output = capsys.readouterr().out
    assert output.count("phase=blocked-idle") == 2
    assert output.count("reason=first blocker") == 1
    assert output.count("reason=second blocker") == 1
    assert "transition=" not in output


def test_blocked_logger_has_no_periodic_summary(capsys):
    logger = BlockedIdleLogger()
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.BLOCKED.value,
        blocked_reason="do not repeat this detailed failure",
    )
    for _ in range(100):
        logger.observe(checkpoint)
    output = capsys.readouterr().out
    assert output.count("phase=blocked-idle") == 1
    assert "phase=blocked-liveness" not in output
    assert output.count("reason=do not repeat this detailed failure") == 1


def test_supervisor_consumes_operator_unblock_once(tmp_path, monkeypatch):
    path = tmp_path / "proof_orchestration.json"
    event_path = tmp_path / "operator_event.json"
    save_orchestration_checkpoint(
        path,
        OrchestrationCheckpoint(
            state=ProofState.BLOCKED.value,
            current_role="blocked",
            blocked_reason="repair exhausted",
            retry_counters={"DECOMPOSER": 14, "PROVER": 2},
        ),
    )
    event_path.write_text(json.dumps({
        "event_id": "evt-unblock",
        "event_type": BlockedEventType.OPERATOR_UNBLOCK.value,
        "reason": "protocol_parser_hardened",
        "target_state": ProofState.DECOMPOSER.value,
        "reset_role": ProofState.DECOMPOSER.value,
    }))
    calls = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda _args, iteration: calls.append(iteration) or {},
    )
    args = SimpleNamespace(
        iterations=1,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(event_path),
    )
    assert run_supervisor_iterations(args) == 0
    checkpoint = load_orchestration_checkpoint(path)
    assert calls == [0]
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.retry_counters == {"DECOMPOSER": 0, "PROVER": 2}
    assert not event_path.exists()
    journal = (
        tmp_path / "proof_orchestration.journal.jsonl"
    ).read_text()
    assert journal.count("evt-unblock") == 1


def test_operator_unblock_restores_normal_output_without_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "proof_orchestration.json"
    event_path = tmp_path / "operator_event.json"
    save_orchestration_checkpoint(
        path,
        OrchestrationCheckpoint(
            state=ProofState.BLOCKED.value,
            current_role="blocked",
            blocked_reason="repair exhausted",
        ),
    )
    event_path.write_text(json.dumps({
        "event_id": "evt-unblock-summary",
        "event_type": BlockedEventType.OPERATOR_UNBLOCK.value,
        "reason": "operator repaired parser",
        "target_state": ProofState.DECOMPOSER.value,
        "reset_role": ProofState.DECOMPOSER.value,
    }))
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda _args, _iteration: {},
    )
    args = SimpleNamespace(
        iterations=1,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(path),
        operator_event_file=str(event_path),
    )
    assert run_supervisor_iterations(args) == 0
    output = capsys.readouterr().out
    assert output.count("phase=blocked-idle") == 1
    assert "reason=repair exhausted" in output
    assert "phase=blocked-transition" not in output
    assert "suppressed_ticks" not in output


def test_orchestration_typed_failure_routes():
    assert classify_failure(
        "decomposer",
        "malformed JSON after output budget",
    ) == ProofState.DECOMPOSER
    assert classify_failure(
        "formalizer",
        "Lean elaboration failed",
    ) == ProofState.FORMALIZER
    assert classify_failure(
        "prover",
        "type mismatch in theorem signature",
    ) == ProofState.FORMALIZER
    assert classify_failure(
        "prover",
        "tactic could not close goal",
    ) == ProofState.PROVER
    assert classify_failure(
        "judge",
        "formalization changed parent",
    ) == ProofState.FORMALIZER
    assert classify_failure(
        "judge",
        "confirmed mathematical approach_failed",
    ) == ProofState.APPROACH_FAILED


def test_artifact_hash_or_dependency_mismatch_is_not_reused(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_AUDITOR.value,
    )
    ref = persist_validated_artifact(
        path,
        checkpoint,
        role="definition_auditor",
        payload={"definitions": [{"symbol": "x"}]},
        dependencies=[],
        source_run_id="run:definition",
    )
    Path(ref.path).write_text('{"tampered":true}')
    from autoresearch.prefill.orchestration_state import (
        load_validated_artifacts,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_validated_artifacts(load_orchestration_checkpoint(path))


def _candidate():
    return {
        "candidate_id": "trial",
        "target_obligation_id": "RH-C1",
        "hypothesis": "Construct and attack one explicit operator.",
        "generator_directive": "Define the operator.",
        "critic_directive": "Falsify the operator.",
        "prefill_compute_chunk_tokens": 256,
        "snapshot_mode": "final_only",
        "max_segment_seconds": 300.0,
        "require_full_context": True,
        "allow_fallback": False,
    }


def test_candidate_render_is_executable_and_strict(tmp_path):
    candidate = _candidate()
    validate_candidate(candidate)
    path = tmp_path / "candidate.py"
    path.write_text(render_candidate(candidate))
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert namespace["CANDIDATE_ID"] == "trial"
    assert namespace["PREFILL_COMPUTE_CHUNK_TOKENS"] == 256
    bad = {**candidate, "allow_fallback": True}
    try:
        validate_candidate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("fallback candidate must be rejected")


def test_infrastructure_failure_fingerprint_is_stable_and_specific():
    failed = {
        "research_outcome": "EVALUATION_FAILED",
        "error": "RuntimeError: GAN benchmark is not completed: failed",
        "failure_class": "infrastructure",
    }
    assert infrastructure_failure_fingerprint(failed)
    assert infrastructure_failure_fingerprint(failed) == (
        infrastructure_failure_fingerprint({
            **failed,
            "error": "  RUNTIMEERROR:  GAN benchmark is not completed: failed ",
        })
    )
    assert infrastructure_failure_fingerprint({
        **failed,
        "error": "different failure",
    }) != infrastructure_failure_fingerprint(failed)
    assert infrastructure_failure_fingerprint({
        **failed,
        "research_outcome": "FALSIFIED",
    }) == ""


def test_gan_failure_reason_preserves_semantic_error():
    output = (
        "[inference-failed] time=now run=br_1 "
        "error=SemanticResponseIncomplete: "
        "SEMANTIC_RESPONSE_INCOMPLETE: Generator stopped before EOS\n"
    )
    assert extract_gan_failure_reason(output) == (
        "SemanticResponseIncomplete: SEMANTIC_RESPONSE_INCOMPLETE: "
        "Generator stopped before EOS"
    )
    assert extract_gan_failure_reason("no structured failure") == ""
    assert infrastructure_failure_fingerprint({
        "research_outcome": "EVALUATION_FAILED",
        "error": "",
    }) == ""


def test_strategy_schema_repair_prefers_current_branch_leaf():
    current = _candidate()
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "Operator construction.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C1-child",
            "statement": "Prove the operator is self-adjoint.",
            "status": "UNRESOLVED",
            "parent_id": "RH-C1",
        },
        {
            "obligation_id": "RH-C2",
            "statement": "Zero convergence.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
    ]}
    repaired, fields = repair_candidate_schema(
        {
            "candidate_id": "trial-child",
            "hypothesis": "The proposed domain yields a symmetric operator.",
        },
        current=current,
        ledger=ledger,
    )
    assert repaired["target_obligation_id"] == "RH-C1-child"
    assert repaired["prefill_compute_chunk_tokens"] == 256
    assert "RH-C1-child" in repaired["generator_directive"]
    assert "strictly smaller" in repaired["critic_directive"]
    assert set(fields) == {
        "target_obligation_id",
        "generator_directive",
        "critic_directive",
        "prefill_compute_chunk_tokens",
    }


def test_strategy_schema_repair_accepts_uppercase_and_alias_keys():
    repaired, fields = repair_candidate_schema(
        {
            "CANDIDATE_ID": "alias-trial",
            "TARGET": "RH-C2",
            "HYPOTHESIS": "Test convergence.",
            "GENERATOR_PROMPT": "Construct the approximation.",
            "CRITIC_PROMPT": "Falsify the approximation.",
            "CHUNK_TOKENS": "128",
        },
        current=_candidate(),
        ledger={"obligations": [{
            "obligation_id": "RH-C2",
            "statement": "Zero convergence.",
            "status": "UNRESOLVED",
            "parent_id": "",
        }]},
    )
    validate_candidate({
        **repaired,
        "snapshot_mode": "final_only",
        "require_full_context": True,
        "allow_fallback": False,
    })
    assert repaired["candidate_id"] == "alias-trial"
    assert repaired["prefill_compute_chunk_tokens"] == 256
    assert set(fields) == {
        "candidate_id",
        "target_obligation_id",
        "hypothesis",
        "generator_directive",
        "critic_directive",
        "prefill_compute_chunk_tokens",
    }


def test_strategy_schema_repair_flattens_nested_hypothesis_and_plan():
    repaired, fields = repair_candidate_schema(
        {
            "candidate_id": "candidate-v2-c2-sub-01",
            "hypothesis": {
                "statement": "Construct regularized analytic continuations.",
                "target_obligation": "RH-C2-child",
            },
            "plan": {
                "steps": [
                    "Define a regularization kernel.",
                    "Attempt to prove compact convergence.",
                ],
            },
        },
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger={"obligations": [
            {
                "obligation_id": "RH-C2",
                "statement": "Zero convergence.",
                "status": "UNRESOLVED",
                "parent_id": "",
            },
            {
                "obligation_id": "RH-C2-child",
                "statement": "Prove regularized compact convergence.",
                "status": "UNRESOLVED",
                "parent_id": "RH-C2",
            },
        ]},
    )
    assert repaired["hypothesis"] == (
        "Construct regularized analytic continuations."
    )
    assert repaired["target_obligation_id"] == "RH-C2-child"
    assert "Step 1: Define a regularization kernel." in (
        repaired["generator_directive"]
    )
    assert isinstance(repaired["hypothesis"], str)
    assert {
        "hypothesis",
        "target_obligation_id",
        "generator_directive",
        "critic_directive",
        "prefill_compute_chunk_tokens",
    }.issubset(set(fields))


def test_strategy_parser_converts_prose_plan_and_ignores_latex_braces():
    output = r"""
The next step is to address the pending leaf: **RH-C2-wrong**.

### Plan for Next Run
1. **Objective**: Formulate the mathematical framework for the Zero-Exclusion Lemma.
2. **Mathematical Strategy**:
   * Investigate $\tilde{F}_N(s)$ using Rouché's Theorem.
   * Determine conditions for $Z(\tilde{F}_N,D) \to P(F,D)$.
3. **Constraint**: Do not assume RH.

**Targeting Leaf**: `RH-C2-correct`
"""
    candidate = _extract_json(output)
    assert candidate["strategy_parse_mode"] == "prose"
    assert candidate["target_obligation_id"] == "RH-C2-correct"
    assert "Zero-Exclusion Lemma" in candidate["hypothesis"]
    assert len(candidate["plan"]["steps"]) >= 2


def test_strategy_parser_accepts_python_literal_candidate():
    candidate = _extract_json(
        "```python\n{'candidate_id': 'trial', 'hypothesis': 'test'}\n```",
    )
    assert candidate["candidate_id"] == "trial"
    assert candidate["strategy_parse_mode"] == "python-literal"


def test_strategy_repairs_invalid_json_latex_escapes():
    candidate = _extract_json(
        r'''```json
{"candidate_id":"trial","hypothesis":"sequence \{z_n\} has density \rho"}
```''',
    )
    assert candidate["hypothesis"] == r"sequence \{z_n\} has density \rho"
    assert candidate["strategy_parse_mode"] == "json-escape-repaired"


def test_strategy_host_adapter_accepts_exact_production_fenced_latex_fixture():
    output = r'''```json
{
  "candidate_id": "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-763645cd6b-40ef83e052-b80cd1343b-3be9e8e78f-fbee0281ff-e4fff64467",
  "target_obligation_id": "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-763645cd6b-40ef83e052-b80cd1343b-3be9e8e78f-fbee0281ff",
  "hypothesis": "Construct a specific sequence of poles {z_n} with density \rho > \rho_c and genus p such that the partial sums of the Mittag-Leffler expansion fail to approximate the target rational function in the \delta-neighborhood of s_0, thereby establishing a lower bound for \rho_c.",
  "generator_directive": "Construct a concrete counterexample sequence {z_n} for a fixed genus p and density \rho > \rho_c that violates the convergence to the target rational form within the \delta-neighborhood, or define the functional form of \rho_c in terms of p and \epsilon.",
  "critic_directive": "Verify if the constructed sequence {z_n} satisfies the density requirement \rho > \rho_c and if the resulting growth order of the function f(s) is strictly greater than p, or if the sum fails to converge to the target form as specified.",
  "prefill_compute_chunk_tokens": 256
}
```'''
    candidate, mode = parse_strategy_candidate_transport(output)
    assert candidate["hypothesis"].count(r"\rho") == 3
    assert candidate["generator_directive"].endswith(r"p and \epsilon.")
    assert candidate["prefill_compute_chunk_tokens"] == 256
    assert mode == (
        "host-unwrapped-json-fence+host-repaired-json-escapes"
    )


@pytest.mark.parametrize(
    "output",
    [
        'prose\n```json\n{"candidate_id":"x"}\n```',
        '```python\n{"candidate_id":"x"}\n```',
        '```json\n{"candidate_id":"x"}\n```\ntrailing',
        '{"candidate_id":"x"} {"candidate_id":"y"}',
    ],
)
def test_strategy_host_adapter_does_not_weaken_single_object_gate(output):
    with pytest.raises(ValueError):
        parse_strategy_candidate_transport(output)


def test_keep_requires_novel_mathematical_advancement():
    baseline = {
        "proof_obligations_unresolved": "5",
        "metric_cold_critic_prefill_s": "500",
    }
    assert should_keep({
        "accepted": True,
        "research_outcome": "SUPPORTED",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 4,
        "metric_cold_critic_prefill_s": 900,
    }, baseline)
    assert should_keep({
        "accepted": True,
        "research_outcome": "FALSIFIED",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 900,
    }, baseline)
    assert should_keep({
        "accepted": True,
        "research_outcome": "DECOMPOSED",
        "created_obligation_ids": ["RH-C1-child"],
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 900,
    }, baseline)
    assert not should_keep({
        "accepted": True,
        "research_outcome": "DECOMPOSED",
        "created_obligation_ids": [],
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)
    assert not should_keep({
        "accepted": True,
        "research_outcome": "SUPPORTED",
        "created_obligation_ids": [],
        "hypothesis_novel": True,
        "proof_obligations_total": 1,
        "proof_obligations_covered": 0,
        "proof_obligations_unresolved": 4,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)
    assert not should_keep({
        "accepted": True,
        "research_outcome": "INCONCLUSIVE",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)
    assert not should_keep({
        "accepted": False,
        "research_outcome": "SUPPORTED",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 0,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)


def test_parse_research_verdict_uses_last_complete_critic_block():
    output = """
critic> ### AUTORESEARCH_VERDICT
Candidate ID: candidate-v3
Outcome: FALSIFIED
Evidence: The proposed positivity implication fails for the explicit test function at n=7.
New frontier: Characterize the admissible test functions for which the implication remains valid.
"""
    verdict = parse_research_verdict(output, "candidate-v3")
    assert verdict["outcome"] == "FALSIFIED"
    assert "admissible test functions" in verdict["new_frontier"]


def test_parse_host_generated_research_verdict_event():
    output = (
        '[autoresearch-verdict] {"candidate_id":"candidate-v4",'
        '"outcome":"DECOMPOSED",'
        '"evidence":"The Critic isolated a concrete missing convergence lemma.",'
        '"new_frontier":"Prove locally uniform convergence on compact subsets."}'
    )
    verdict = parse_research_verdict(output, "candidate-v4")
    assert verdict["outcome"] == "DECOMPOSED"


def test_pending_leaf_ids_excludes_unresolved_parents():
    ledger = {"obligations": [
        {"obligation_id": "RH-C1", "status": "UNRESOLVED", "parent_id": ""},
        {
            "obligation_id": "RH-C1-child",
            "status": "UNRESOLVED",
            "parent_id": "RH-C1",
        },
        {"obligation_id": "RH-C2", "status": "UNRESOLVED", "parent_id": ""},
    ]}
    assert _pending_leaf_ids(ledger) == ["RH-C1-child", "RH-C2"]


def test_pending_leaf_ids_excludes_stale_premise_descendants():
    ledger = {"obligations": [
        {
            "obligation_id": "ROOT",
            "status": "DISPROVED",
            "invalidation_kind": "PREMISE",
            "parent_id": "",
        },
        {
            "obligation_id": "ROOT-stale",
            "status": "UNRESOLVED",
            "parent_id": "ROOT",
        },
        {
            "obligation_id": "SOUND",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
    ]}
    assert _pending_leaf_ids(ledger) == ["SOUND"]


def test_host_candidate_targets_deepest_current_branch_leaf():
    current = {**_candidate(), "target_obligation_id": "RH-C2"}
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "Unrelated leaf.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C2",
            "statement": "Root convergence claim.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C2-child",
            "statement": "Construct a compact convergence counterexample.",
            "status": "UNRESOLVED",
            "parent_id": "RH-C2",
            "last_evidence": "The previous approximation failed at a pole.",
        },
    ]}
    candidate = build_host_candidate(current, ledger)
    assert candidate["target_obligation_id"] == "RH-C2-child"
    assert candidate["hypothesis"] == (
        "Construct a compact convergence counterexample."
    )
    assert "do not rename" in candidate["generator_directive"]
    assert candidate["prefill_compute_chunk_tokens"] == 256


def test_host_candidate_rolls_back_to_nearest_valid_ancestor():
    rejected_target = "RH-C2-duplicate-child"
    current = {
        **_candidate(),
        "target_obligation_id": rejected_target,
    }
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "Unrelated leaf.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C2-gap",
            "statement": "Falsify the density-singularity implication.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": rejected_target,
            "statement": "Renamed density-singularity implication.",
            "status": "REJECTED_DUPLICATE",
            "parent_id": "RH-C2-gap",
        },
    ]}
    candidate = build_host_candidate(current, ledger)
    assert candidate["target_obligation_id"] == "RH-C2-gap"


def test_host_resume_candidate_uses_bound_root_not_quarantined_child():
    current = {**_candidate(), "target_obligation_id": "RH-C1"}
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C0-root",
            "statement": "RiemannHypothesis",
            "status": "UNRESOLVED",
            "formal_status": "FORMALIZED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C1",
            "statement": "Stale quarantined child.",
            "status": "QUARANTINED",
            "parent_id": "RH-C0-root",
        },
    ]}
    candidate = build_host_candidate(
        current,
        ledger,
        target_id="RH-C0-root",
    )
    assert candidate["target_obligation_id"] == "RH-C0-root"
    assert candidate["hypothesis"] == "RiemannHypothesis"
    assert "RH-C1" not in candidate["generator_directive"]


def test_host_candidate_uses_recorded_premise_backjump_target():
    current = {**_candidate(), "target_obligation_id": "ROOT-bad"}
    ledger = {
        "backjump_target_id": "ROOT",
        "no_go_lessons": [{
            "refuted_premise": "Every admissible kernel is positive.",
            "source_obligation_id": "ROOT-bad",
        }],
        "obligations": [
            {
                "obligation_id": "ROOT",
                "statement": "Find a sound replacement reduction.",
                "status": "UNRESOLVED",
                "parent_id": "",
            },
            {
                "obligation_id": "ROOT-bad",
                "statement": "Use universal kernel positivity.",
                "status": "DISPROVED",
                "invalidation_kind": "PREMISE",
                "parent_id": "ROOT",
            },
            {
                "obligation_id": "UNRELATED",
                "statement": "Unrelated unresolved branch.",
                "status": "UNRESOLVED",
                "parent_id": "",
            },
        ],
    }
    candidate = build_host_candidate(current, ledger)
    assert candidate["target_obligation_id"] == "ROOT"
    assert "Every admissible kernel is positive" in (
        candidate["generator_directive"]
    )


def test_repeated_gemma_uses_one_deterministic_host_fallback():
    current = _candidate()
    ledger = {"obligations": [{
        "obligation_id": "RH-C1",
        "statement": "Prove the compactness estimate for the explicit kernel.",
        "status": "UNRESOLVED",
        "parent_id": "",
    }]}
    selected, mode, _, _, used_fallback = select_novel_candidate(
        {**current, "candidate_id": "gemma-repeat"},
        strategy_mode="gemma",
        current=current,
        ledger=ledger,
        results=[],
    )
    assert used_fallback
    assert mode == "host_strategy_deferred"
    assert selected["candidate_id"].startswith("host-leaf-")
    assert selected["hypothesis"] == ledger["obligations"][0]["statement"]


def test_duplicate_gemma_and_host_is_nonfatal_and_preserves_candidate(tmp_path):
    current = _candidate()
    ledger = {"obligations": [{
        "obligation_id": "RH-C1",
        "statement": "Prove the compactness estimate for the explicit kernel.",
        "status": "UNRESOLVED",
        "parent_id": "",
    }]}
    host = build_host_candidate(current, ledger)
    _, _, host_hypothesis_sha, host_candidate_sha, _ = select_novel_candidate(
        host,
        strategy_mode="host",
        current=current,
        ledger=ledger,
        results=[],
    )
    candidate_path = tmp_path / "candidate.py"
    accepted_bytes = render_candidate(current).encode()
    candidate_path.write_bytes(accepted_bytes)
    with pytest.raises(CandidateNoveltyStagnation) as caught:
        select_novel_candidate(
            {**current, "candidate_id": "gemma-repeat"},
            strategy_mode="gemma",
            current=current,
            ledger=ledger,
            results=[{
                "hypothesis_sha256": host_hypothesis_sha,
                "candidate_sha256": host_candidate_sha,
            }],
        )
    assert caught.value.strategy_mode == "host_strategy_deferred"
    assert "duplicate" in str(caught.value)
    assert candidate_path.read_bytes() == accepted_bytes


def test_stagnated_iteration_does_not_exit_long_running_loop(monkeypatch):
    rows = iter([
        {
            "research_outcome": "STAGNATED",
            "error": "CandidateNoveltyStagnation: duplicate",
        },
        {
            "research_outcome": "SUPPORTED",
            "kept": True,
        },
    ])
    calls = []

    def fake_iteration(_args, iteration):
        calls.append(iteration)
        return next(rows)

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        fake_iteration,
    )
    args = SimpleNamespace(
        iterations=2,
        max_consecutive_infrastructure_failures=2,
    )
    assert run_supervisor_iterations(args) == 0
    assert calls == [0, 1]


def test_host_missing_definition_backjump_continues_and_restarts_idempotently(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.HOST_TYPED_IR_GATE.value,
        current_role="host_typed_ir_gate",
        target_obligation_id="ROOT",
        candidate_sha256="a" * 64,
        strategy_reused=True,
        ledger_version=92,
    )
    definition_ref = persist_validated_artifact(
        state_path,
        checkpoint,
        role="definition_auditor",
        payload={"missing_definitions": ["density"]},
        dependencies=[],
        source_run_id="definition-run",
    )
    save_orchestration_checkpoint(state_path, checkpoint)
    calls = []
    sleeps = []
    live_phases = []

    def fake_iteration(_args, iteration):
        current = load_orchestration_checkpoint(state_path)
        calls.append((iteration, current.state))
        assert current.validated_artifacts[
            "definition_auditor"
        ].sha256 == definition_ref.sha256
        assert current.strategy_reused is False
        if current.proof_state == ProofState.HOST_TYPED_IR_GATE:
            current.transition(
                ProofState.SYNTHESIS,
                "typed-evidence-backjump:MISSING_DEFINITION_ENVIRONMENT",
                strategy_reused=True,
            )
            save_orchestration_checkpoint(state_path, current)
            return {
                "research_outcome": "INCONCLUSIVE",
                "failure_class": "",
                "supervisor_outcome": "CONTINUE",
            }
        assert current.proof_state == ProofState.SYNTHESIS
        return {
            "research_outcome": "INCONCLUSIVE",
            "failure_class": "",
            "strategy_mode": "resumed",
            "supervisor_outcome": "CONTINUE",
        }

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        fake_iteration,
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    args = SimpleNamespace(
        iterations=2,
        max_consecutive_infrastructure_failures=2,
        orchestration_state_file=str(state_path),
        stop_file=str(tmp_path / "stop"),
        continuation_backoff_s=0,
        _live_status=SimpleNamespace(
            emit=lambda **kwargs: live_phases.append(kwargs.get("phase")),
        ),
    )

    assert run_supervisor_iterations(args) == 0
    assert calls == [
        (0, ProofState.HOST_TYPED_IR_GATE.value),
        (1, ProofState.SYNTHESIS.value),
    ]
    persisted = load_orchestration_checkpoint(state_path)
    assert is_resumable_checkpoint(
        persisted,
        candidate_sha256="a" * 64,
    )
    assert persisted.proof_state == ProofState.SYNTHESIS
    assert persisted.last_transition_reason == (
        "typed-evidence-backjump:MISSING_DEFINITION_ENVIRONMENT"
    )
    assert set(persisted.validated_artifacts) == {"definition_auditor"}
    assert "supervisor_exit" not in live_phases
    assert sleeps == [0]

    calls.clear()
    args.iterations = 1
    assert run_supervisor_iterations(args) == 0
    assert calls == [(0, ProofState.SYNTHESIS.value)]
    assert load_orchestration_checkpoint(
        state_path,
    ).proof_state == ProofState.SYNTHESIS


def test_missing_definition_precontract_route_starts_next_run_without_strategy(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "proof_orchestration.json"
    candidate_sha256 = "a" * 64
    typed_ir_hash = "d" * 64
    theorem_id = "typed_ea7631dcb3e2b55b"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.HOST_TYPED_IR_GATE.value,
        current_role="host_typed_ir_gate",
        target_obligation_id="ROOT",
        candidate_sha256=candidate_sha256,
        strategy_sha256=candidate_sha256,
        strategy_reused=True,
        typed_ir_hash=typed_ir_hash,
        elaborated_theorem_id=theorem_id,
        selected_move_id="REGISTER_DEFINITION_OBLIGATION",
    )
    save_orchestration_checkpoint(state_path, checkpoint)
    calls = []
    sleeps = []

    def fake_iteration(_args, iteration):
        current = load_orchestration_checkpoint(state_path)
        calls.append((iteration, current.state, current.target_obligation_id))
        if iteration == 0:
            current.target_obligation_id = "ROOT:typed-reframe:2b65585aa8fa"
            current.transition(
                ProofState.DECOMPOSER,
                "precontract-semantic-routing:MISSING_DEFINITION",
                strategy_reused=True,
            )
            save_orchestration_checkpoint(state_path, current)
            return {
                "research_outcome": "EVALUATION_FAILED",
                "failure_class": "infrastructure",
                "error": "RuntimeError: GAN benchmark is not completed: running",
                "supervisor_outcome": "CONTINUE",
            }
        assert should_resume_downstream(
            current,
            candidate_sha256=candidate_sha256,
            force_strategy=False,
            strategy_trigger_exists=False,
        )
        assert current.selected_move_id == "REGISTER_DEFINITION_OBLIGATION"
        assert current.typed_ir_hash == typed_ir_hash
        assert current.elaborated_theorem_id == theorem_id
        return {
            "research_outcome": "INCONCLUSIVE",
            "failure_class": "",
            "supervisor_outcome": "ITERATION_COMPLETE",
        }

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        fake_iteration,
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    args = SimpleNamespace(
        iterations=2,
        max_consecutive_infrastructure_failures=1,
        orchestration_state_file=str(state_path),
        stop_file=str(tmp_path / "stop"),
        continuation_backoff_s=0.25,
        continuation_max_backoff_s=1.0,
    )

    assert run_supervisor_iterations(args) == 0
    assert calls == [
        (0, ProofState.HOST_TYPED_IR_GATE.value, "ROOT"),
        (
            1,
            ProofState.DECOMPOSER.value,
            "ROOT:typed-reframe:2b65585aa8fa",
        ),
    ]
    assert sleeps == [0.25]
    persisted = load_orchestration_checkpoint(state_path)
    assert persisted.proof_state == ProofState.DECOMPOSER
    assert persisted.strategy_reused is False
    assert persisted.selected_move_id == "REGISTER_DEFINITION_OBLIGATION"
    assert persisted.typed_ir_hash == typed_ir_hash
    assert persisted.elaborated_theorem_id == theorem_id


def test_repeated_semantic_routes_back_off_without_opening_circuit(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        candidate_sha256="a" * 64,
        last_transition_reason=(
            "precontract-semantic-routing:MISSING_DEFINITION"
        ),
        strategy_reused=True,
    )
    save_orchestration_checkpoint(state_path, checkpoint)
    calls = []
    sleeps = []
    row = {
        "research_outcome": "EVALUATION_FAILED",
        "failure_class": "infrastructure",
        "error": "RuntimeError: GAN benchmark is not completed: running",
        "supervisor_outcome": "CONTINUE",
    }

    def fake_iteration(_args, iteration):
        calls.append(iteration)
        return dict(row)

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        fake_iteration,
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    args = SimpleNamespace(
        iterations=3,
        max_consecutive_infrastructure_failures=1,
        orchestration_state_file=str(state_path),
        stop_file=str(tmp_path / "stop"),
        continuation_backoff_s=1.0,
        continuation_max_backoff_s=8.0,
    )

    assert is_nonfatal_semantic_continuation(row, checkpoint)
    assert run_supervisor_iterations(args) == 0
    assert calls == [0, 1, 2]
    assert sleeps == [1.0, 2.0]


def test_wrapped_resume_validation_failure_is_nonfatal_integration():
    error = RuntimeError(
        "GAN benchmark is not completed: failed; "
        "ResumeValidationError: resumed report has no complete Critic artifact ref"
    )
    assert failure_class_for_exception(error) == "integration"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.STRATEGY_TOURNAMENT.value,
        candidate_sha256="a" * 64,
    )
    assert is_resumable_checkpoint(
        checkpoint,
        candidate_sha256="a" * 64,
    )
    assert not is_resumable_checkpoint(
        checkpoint,
        candidate_sha256="b" * 64,
    )


def test_unbounded_supervisor_exits_only_on_explicit_stop_file(
    tmp_path,
    monkeypatch,
):
    stop_file = tmp_path / "stop"
    calls = []

    def fake_iteration(_args, iteration):
        calls.append(iteration)
        if iteration == 1:
            stop_file.write_text("operator requested stop")
        return {"research_outcome": "INCONCLUSIVE", "failure_class": ""}

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        fake_iteration,
    )
    args = SimpleNamespace(
        iterations=None,
        max_consecutive_infrastructure_failures=2,
        stop_file=str(stop_file),
    )
    assert run_supervisor_iterations(args) == 0
    assert calls == [0, 1]


def test_stagnation_does_not_weaken_infrastructure_circuit_breaker(monkeypatch):
    failed = {
        "research_outcome": "EVALUATION_FAILED",
        "error": "RuntimeError: worker unavailable",
        "failure_class": "infrastructure",
    }
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda _args, _iteration: failed,
    )
    args = SimpleNamespace(
        iterations=3,
        max_consecutive_infrastructure_failures=2,
    )
    assert run_supervisor_iterations(args) == 2


def test_integration_failures_do_not_open_infrastructure_circuit(monkeypatch):
    failed = {
        "research_outcome": "EVALUATION_FAILED",
        "error": "ResumeValidationError: stale Critic artifact",
        "failure_class": "integration",
    }
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.run_iteration",
        lambda _args, _iteration: failed,
    )
    args = SimpleNamespace(
        iterations=2,
        max_consecutive_infrastructure_failures=2,
    )
    assert run_supervisor_iterations(args) == 0


def test_strategy_is_triggered_only_by_events(tmp_path):
    progress = {
        "kept": "True",
        "research_outcome": "DECOMPOSED",
        "created_obligation_ids": '["child"]',
    }
    inconclusive = {
        "kept": "False",
        "research_outcome": "INCONCLUSIVE",
        "created_obligation_ids": "[]",
    }
    assert strategy_trigger_reason(
        [progress, inconclusive, inconclusive],
        stagnation_rounds=3,
    ) == ""
    assert strategy_trigger_reason(
        [progress, inconclusive, inconclusive, inconclusive],
        stagnation_rounds=3,
    ) == "stagnation-3"
    assert strategy_trigger_reason(
        [progress, inconclusive, inconclusive, {
            "research_outcome": "STAGNATED",
            "invalidation_kind": "STRATEGY_STAGNATION",
        }],
        stagnation_rounds=3,
    ) == ""
    assert strategy_trigger_reason(
        [{"kept": "True", "research_outcome": "FALSIFIED"}],
        stagnation_rounds=3,
    ) == "branch-falsified"
    assert strategy_trigger_reason(
        [{
            "kept": "True",
            "research_outcome": "FALSIFIED",
            "invalidation_kind": "PREMISE_INVALIDATED",
        }],
        stagnation_rounds=3,
    ) == "premise-invalidated"
    assert strategy_trigger_reason(
        [{
            "kept": "False",
            "research_outcome": "INCONCLUSIVE",
            "invalidation_kind": "PREMISE_SUSPECTED",
        }],
        stagnation_rounds=3,
    ) == ""
    assert strategy_trigger_reason(
        [{
            "kept": "True",
            "research_outcome": "INCONCLUSIVE",
            "invalidation_kind": "APPROACH_FAILED",
        }],
        stagnation_rounds=3,
    ) == "branch-falsified"
    trigger = tmp_path / "request_strategy"
    trigger.write_text("replan")
    assert strategy_trigger_reason(
        [],
        stagnation_rounds=3,
        trigger_file=trigger,
    ) == "manual-trigger-file"


def test_strategy_budget_error_preserves_exact_admission_counts():
    error = StrategyPrefillBudgetExceeded(11935, 8448)
    assert error.token_count == 11935
    assert error.max_tokens == 8448
    assert "without truncation: 11935 > 8448" in str(error)


def test_semantic_admission_and_dynamic_output_reserve_never_slice():
    token_ids = list(range(2053))
    with pytest.raises(
        SemanticUnitTooLarge,
        match="SEMANTIC_UNIT_TOO_LARGE",
    ):
        admit_token_ids(
            "indivisible target",
            token_ids,
            configured_prefill_tokens=6144,
            max_retained_tokens=2052,
        )
    assert token_ids == list(range(2053))
    assert downstream_output_cap(
        max_retained_tokens=2052,
        fixed_downstream_tokens=1700,
        configured_output_tokens=1000,
        control_reserve_tokens=32,
    ) == 320


def test_strategy_candidate_rejects_multi_step_plan():
    with pytest.raises(ValueError, match="exactly one step"):
        validate_candidate({
            **_candidate(),
            "plan": {"steps": ["first", "second"]},
        })


def test_strategy_state_keeps_exact_one_step_interface_only():
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "unrelated root",
            "status": "UNRESOLVED",
            "parent_id": "",
            "last_evidence": "unrelated evidence",
        },
        {
            "obligation_id": "RH-C2",
            "statement": "exact root statement",
            "status": "UNRESOLVED",
            "parent_id": "",
            "last_evidence": "exact root evidence",
        },
        {
            "obligation_id": "RH-C2-child",
            "statement": "exact child statement",
            "status": "UNRESOLVED",
            "parent_id": "RH-C2",
            "last_evidence": "exact child evidence",
        },
    ]}
    results = (
        "candidate_id\ttarget_obligation_id\tresearch_outcome\t"
        "research_evidence\tnew_frontier\tkept\terror\t"
        "hypothesis_sha256\n"
        "c1\tRH-C1\tINCONCLUSIVE\tunrelated result\tx\tFalse\t\th1\n"
        "c2\tRH-C2-child\tDECOMPOSED\texact result\tfrontier\tTrue\t\th2\n"
    )
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger=ledger,
        results_text=results,
    )
    interface = state["proof_step_interface"]
    assert interface["target_obligation_id"] == "RH-C2-child"
    assert interface["target_statement"] == "exact child statement"
    assert interface["current_target_evidence"] == "exact child evidence"
    assert interface["parent_interface"]["statement_hash"]
    serialized = str(state)
    assert "exact child evidence" in serialized
    assert "exact root statement" not in serialized
    assert "exact root evidence" not in serialized
    assert "exact result" not in serialized
    assert "unrelated root" not in serialized
    assert "unrelated result" not in serialized


def test_strategy_state_carries_lossless_no_go_lessons():
    premise = "Every admissible kernel is positive."
    evidence = (
        "The explicit admissible polynomial kernel changes sign while "
        "satisfying every required boundary condition."
    )
    ledger = {
        "backjump_target_id": "ROOT",
        "no_go_lessons": [{
            "claim_hash": "abc123",
            "refuted_premise": premise,
            "evidence": evidence,
            "source_obligation_id": "ROOT-bad",
            "run_id": "br_refute",
        }],
        "obligations": [{
            "obligation_id": "ROOT",
            "statement": "Find a replacement reduction.",
            "status": "UNRESOLVED",
            "parent_id": "",
        }, {
            "obligation_id": "ROOT-bad",
            "statement": premise,
            "status": "DISPROVED",
            "invalidation_kind": "PREMISE",
            "parent_id": "ROOT",
        }],
    }
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "ROOT-bad"},
        ledger=ledger,
        results_text="",
    )
    lessons = state["proof_step_interface"]["active_no_go_lessons"]
    assert lessons[0]["refuted_premise"] == premise
    assert lessons[0]["evidence"] == evidence


def test_strategy_state_keeps_target_unit_and_hashes_history():
    shared = "Complete exact evidence that must appear once without truncation."
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C2",
            "statement": "Root statement.",
            "status": "UNRESOLVED",
            "parent_id": "",
            "last_evidence": shared,
        },
    ]}
    results = (
        "candidate_id\ttarget_obligation_id\tresearch_outcome\t"
        "research_evidence\tnew_frontier\tkept\terror\t"
        "hypothesis_sha256\n"
        f"c2\tRH-C2\tINCONCLUSIVE\t{shared}\tRoot statement."
        "\tFalse\t\th2\n"
    )
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger=ledger,
        results_text=results,
    )
    serialized = str(state)
    assert serialized.count(shared) == 1
    assert serialized.count("Root statement.") == 1
    interface = state["proof_step_interface"]
    assert interface["current_target_evidence"] == shared
    archive = interface["archive_manifest"]
    assert archive["record_count"] == 1
    assert len(archive["ordered_records_sha256"]) == 64


def test_strategy_state_exposes_latest_semantic_failure_for_smaller_step():
    ledger = {"obligations": [{
        "obligation_id": "RH-C2",
        "statement": "Exact current target.",
        "status": "UNRESOLVED",
        "parent_id": "",
    }]}
    error = (
        "SemanticResponseIncomplete: SEMANTIC_RESPONSE_INCOMPLETE: "
        "Generator stopped before EOS after 320 tokens"
    )
    results = (
        "target_obligation_id\thypothesis_sha256\tresearch_outcome\terror\n"
        f"RH-C2\thash-1\tEVALUATION_FAILED\t{error}\n"
    )
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger=ledger,
        results_text=results,
    )
    latest = state["proof_step_interface"]["archive_manifest"][
        "latest_failure"
    ]
    assert latest["kind"] == "SEMANTIC_RESPONSE_INCOMPLETE"
    assert latest["role"] == "Generator"
    assert latest["response_tokens"] == 320
    assert "error" not in latest


def test_strategy_prompt_bounded_interface_for_11_nodes_and_28_runs():
    obligations = []
    for index in range(11):
        obligations.append({
            "obligation_id": f"RH-N{index}",
            "statement": (
                f"Complete exact ancestry statement {index}: "
                + "mathematical condition " * 12
            ),
            "status": "UNRESOLVED",
            "parent_id": f"RH-N{index - 1}" if index else "",
            "last_evidence": (
                f"ancestry evidence {index} " + "detail " * 80
            ),
        })
    header = (
        "timestamp\texperiment_id\trun_id\tcandidate_id\t"
        "target_obligation_id\thypothesis_sha256\tresearch_outcome\t"
        "invalidation_kind\tresearch_evidence\tnew_frontier\tkept\terror\n"
    )
    rows = []
    for index in range(28):
        critical = index % 7 == 0
        rows.append("\t".join((
            str(index),
            f"experiment-{index}",
            f"run-{index}",
            f"candidate-{index}",
            "RH-N10",
            f"hypothesis-{index % 4}",
            "FALSIFIED" if critical else "INCONCLUSIVE",
            "",
            f"evidence-{index}-" + "exact mathematical evidence " * 45,
            f"frontier-{index}-" + "exact frontier statement " * 30,
            "True" if critical else "False",
            "" if index % 5 else "worker timeout fingerprint",
        )))
    results = header + "\n".join(rows) + "\n"
    ledger = {"obligations": obligations}
    current = {**_candidate(), "target_obligation_id": "RH-N0"}
    state = build_strategy_research_state(
        current=current,
        ledger=ledger,
        results_text=results,
    )
    interface = state["proof_step_interface"]
    assert interface["target_obligation_id"] == "RH-N10"
    assert "Complete exact ancestry statement 10" in (
        interface["target_statement"]
    )
    assert "Complete exact ancestry statement 9" not in str(state)
    archive = interface["archive_manifest"]
    assert archive["record_count"] == 28
    assert len(archive["ordered_records_sha256"]) == 64
    serialized = json.dumps(state, ensure_ascii=False)
    assert "evidence-27-" not in serialized
    assert "evidence-1-" not in serialized
    program = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "program.md"
    ).read_text()
    contract = build_strategy_contract(program)
    assert "target_obligation_id must equal TARGET_LEAF_ID" in contract
    assert "no fallback" in contract
    prompt = build_strategy_prompt(
        program=program,
        current=current,
        results_text=results,
        ledger=ledger,
    )
    assert "\n\nPROGRAM:\n" not in prompt
    proxy_tokens = (len(prompt.encode("utf-8")) + 3) // 4
    assert proxy_tokens <= 2052


def test_results_are_append_only_and_best_is_selected(tmp_path):
    path = tmp_path / "results.tsv"
    common = {
        "timestamp": 1,
        "experiment_id": "e1",
        "run_id": "br_1",
        "candidate_id": "c1",
        "target_obligation_id": "RH-C1",
        "constraints_pass": True,
        "accepted": True,
        "kept": True,
        "metric_cold_critic_prefill_s": 500,
        "baseline_metric_s": "",
        "proof_obligations_total": 5,
        "proof_obligations_covered": 5,
        "proof_obligations_unresolved": 5,
        "compute_chunk_tokens": 256,
        "candidate_sha256": "a",
        "report_path": "/tmp/r1.json",
    }
    append_result(path, common)
    append_result(path, {
        **common,
        "timestamp": 2,
        "experiment_id": "e2",
        "candidate_id": "c2",
        "metric_cold_critic_prefill_s": 450,
    })
    rows = read_results(path)
    assert len(rows) == 2
    assert best_kept(rows)["candidate_id"] == "c2"


def test_append_result_migrates_legacy_results_header(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text("timestamp\tcandidate_id\n1\tlegacy\n")
    append_result(path, {
        "timestamp": 2,
        "candidate_id": "new",
        "hypothesis_sha256": "sha",
        "research_outcome": "FALSIFIED",
    })
    rows = read_results(path)
    assert rows[0]["candidate_id"] == "legacy"
    assert rows[0]["research_outcome"] == ""
    assert rows[1]["hypothesis_sha256"] == "sha"


def test_runtime_health_check_is_read_only(monkeypatch):
    ports = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._wait_port",
        lambda host, port: ports.append((host, port)),
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: {"online_nodes": 1, "kv_hit_rate": 0.75},
    )
    summary = check_runtime_health("169.254.27.104:53051")
    assert summary["kv_hit_rate"] == 0.75
    assert ports == [
        ("169.254.27.104", 53051),
        ("127.0.0.1", 51051),
        ("127.0.0.1", 8090),
    ]


def _benchmark_report(status, *, generation="generation-a", version=1):
    return {
        "id": "br_exact",
        "generation": generation,
        "status": status,
        "report_version": version,
        "finished_at": 2.0 if status != "running" else None,
        "stages": [] if status == "running" else [{"name": "final"}],
    }


def test_benchmark_finalization_accepts_delayed_running_to_completed(
    monkeypatch,
):
    reports = iter([
        _benchmark_report("running", version=0),
        _benchmark_report("completed"),
    ])
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: next(reports),
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor.time.sleep",
        lambda _seconds: None,
    )
    report = wait_for_benchmark_finalization(
        dashboard="http://dashboard",
        run_id="br_exact",
        generation="generation-a",
        process_exit_code=0,
        timeout_s=1,
    )
    assert report["status"] == "completed"
    assert report["report_version"] == 1


def test_benchmark_finalization_running_timeout_is_typed(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: _benchmark_report("running", version=0),
    )
    with pytest.raises(BenchmarkFinalizationTimeout) as raised:
        wait_for_benchmark_finalization(
            dashboard="http://dashboard",
            run_id="br_exact",
            generation="generation-a",
            process_exit_code=0,
            timeout_s=0,
        )
    assert raised.value.code == "BENCHMARK_FINALIZATION_TIMEOUT"
    assert raised.value.last_report["status"] == "running"
    assert raised.value.process_exit_code == 0


def test_benchmark_finalization_preserves_failed_terminal(monkeypatch):
    failed = _benchmark_report("failed")
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: failed,
    )
    report = wait_for_benchmark_finalization(
        dashboard="http://dashboard",
        run_id="br_exact",
        generation="generation-a",
        process_exit_code=0,
    )
    error = BenchmarkTerminalFailure(report, process_exit_code=0)
    assert error.report["status"] == "failed"
    assert "terminal failure" in str(error)


def test_benchmark_finalization_rejects_concurrent_run_generation(
    monkeypatch,
):
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: _benchmark_report(
            "completed",
            generation="generation-other",
        ),
    )
    with pytest.raises(ReportValidationError, match="generation mismatch"):
        wait_for_benchmark_finalization(
            dashboard="http://dashboard",
            run_id="br_exact",
            generation="generation-a",
            process_exit_code=0,
        )


def test_strategy_prefill_heartbeat_reports_delta(monkeypatch, capsys):
    heartbeat = StrategyPrefillHeartbeat(interval_s=0.01)
    heartbeat._baseline = {
        "remote_jobs": 10,
        "remote_job_tokens_total": 300,
        "remote_job_tokens_computed": 300,
        "remote_hits": 2,
        "tokens_reused": 20,
    }
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: {"prefill": {
            "remote_jobs": 11,
            "remote_job_tokens_total": 300,
            "remote_job_tokens_computed": 128,
            "remote_hits": 3,
            "tokens_reused": 84,
        }},
    )
    heartbeat._emit()
    output = capsys.readouterr().out
    assert "128/300 tokens (42.7%)" in output
    assert "remote_hits=1 reused=64" in output


def test_supervisor_preserves_runtime_and_cache_across_iterations():
    source = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "supervisor.py"
    ).read_text()
    body = source[source.index("def run_iteration"):source.index("def main")]
    assert body.index("audit_ledger_semantic_duplicates(") < body.index(
        "previous_ledger = _backup",
    )
    assert body.index("check_runtime_health(") < body.index(
        "strategy_trigger_reason(",
    )
    assert "phase=runtime-health-check" in body
    assert "phase=deterministic-candidate" in body
    assert "mode=gemma trigger=" in body
    assert "except StrategyPrefillBudgetExceeded" in body
    assert "phase=strategy-deferred-budget" in body
    assert "except SemanticResponseIncomplete" in body
    assert "phase=strategy-deferred-semantic" in body
    assert "if not gan_completed:" in body
    assert "phase=completed-run-preserved" in body
    assert "deploy_candidate" not in source
    assert "clear_primary_cache" not in source
    assert "launchctl" not in source
    assert "bootout" not in source
    assert "results_text[-" not in source
    propose_body = source[
        source.index("def propose_candidate"):
        source.index("def check_runtime_health")
    ]
    assert propose_body.index("admit_token_ids(") < propose_body.index(
        "session.append(ids)",
    )
    run_body = source[
        source.index("def run_gan_experiment"):
        source.index("def read_results")
    ]
    assert '"--max-retained-tokens"' in run_body


def test_gan_subprocess_output_is_streamed_not_captured():
    source = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "supervisor.py"
    ).read_text()
    body = source[
        source.index("def run_gan_experiment"):
        source.index("def read_results")
    ]
    assert "subprocess.Popen(" in body
    assert "stderr=subprocess.STDOUT" in body
    assert 'print(line, end="", flush=True)' in body
    assert "capture_output=True" not in body
