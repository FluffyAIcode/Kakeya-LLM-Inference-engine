import json
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
    load_checkpoint,
)
from autoresearch.prefill.root_bootstrap import (
    MATHLIB_REVISION,
    bootstrap_rh_root,
    validate_root_sources,
)
from scripts.agent_gan_repl import ProofObligation


ROOT = Path(__file__).resolve().parents[3]


def _ledger() -> dict:
    quarantined = ProofObligation(
        obligation_id="RH-C1",
        statement="quarantined historical child",
        status="QUARANTINED",
        quarantine_reason="TARGET_INTERFACE_EXHAUSTED:243595",
        quarantine_root_id="RH-C1",
        quarantine_prior_status="UNRESOLVED",
        quarantine_evidence_type="CONTENT_ADDRESSED_EXHAUSTION",
        quarantine_evidence_source="243595",
        quarantine_reversible_status="ACTIVE",
    )
    stale = ProofObligation(
        obligation_id="RH-C2",
        statement="stale historical branch",
    )
    return {
        "ledger_id": "rh-rigorous-obligations-v1",
        "obligations": [asdict(quarantined), asdict(stale)],
        "version": 95,
        "schema_version": 1,
        "no_go_lessons": [],
        "backjump_target_id": "ROOT_UNAVAILABLE",
    }


def test_pinned_root_cards_and_all_candidate_branches_elaborate():
    cards, candidates = validate_root_sources(ROOT)
    assert {item.declaration_name for item in cards} == {
        "riemannZeta",
        "completedRiemannZeta",
        "completedRiemannZeta₀",
        "riemannZeta_neg_two_mul_nat_add_one",
        "RiemannHypothesis",
    }
    assert all(item.source_revision == MATHLIB_REVISION for item in cards)
    by_id = {item.candidate_id: item for item in candidates}
    assert by_id["RH-ROOT-MATHLIB"].accepted
    assert by_id["RH-ROOT-EXPANDED"].accepted
    assert (
        by_id["RH-ROOT-EXPANDED"].equivalence_theorem
        == "kakeya_rh_expanded_iff_canonical"
    )
    assert not by_id["RH-ROOT-CRITICAL-STRIP"].accepted
    assert "UNPROVED_EQUIVALENCE_TO_CANONICAL_ROOT" in (
        by_id["RH-ROOT-CRITICAL-STRIP"].rejection_codes
    )
    assert not by_id["RH-ROOT-COMPLETED-LAMBDA"].accepted
    assert "COMPLETED_LAMBDA_NOT_COMPLETED_XI" in (
        by_id["RH-ROOT-COMPLETED-LAMBDA"].rejection_codes
    )
    assert not by_id["RH-ROOT-TRIVIAL-ZEROS-INCLUDED"].accepted
    assert "TRIVIAL_ZEROS_NOT_EXCLUDED" in (
        by_id["RH-ROOT-TRIVIAL-ZEROS-INCLUDED"].rejection_codes
    )
    assert not by_id["RH-ROOT-VACUOUS"].accepted
    assert "VACUOUS_ANTECEDENT" in (
        by_id["RH-ROOT-VACUOUS"].rejection_codes
    )
    assert not by_id["RH-ROOT-COMPLETED-XI"].elaborated
    assert "LEAN_ELABORATION_FAILED" in (
        by_id["RH-ROOT-COMPLETED-XI"].rejection_codes
    )


def test_atomic_root_creation_is_idempotent_and_isolates_quarantine(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    checkpoint_path = tmp_path / "proof_orchestration.json"
    ledger = _ledger()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.BLOCKED.value,
        current_role="blocked",
        target_obligation_id="RH-C1",
        ledger_id=ledger["ledger_id"],
        ledger_version=95,
        blocked_reason="ROOT_UNAVAILABLE",
        strategy_run_status="FINISHED",
        strategy_run_id="stale-rh-c1-run",
        strategy_intent_hash="stale-rh-c1-intent",
        strategy_selection_provenance={"target": "RH-C1"},
    )
    rh_c1_before = ledger["obligations"][0]

    first = bootstrap_rh_root(
        project_root=ROOT,
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        ledger=ledger,
    )
    committed = json.loads(ledger_path.read_text())
    restored = load_checkpoint(checkpoint_path)
    assert first.changed
    assert first.ledger_version == 96
    assert committed["backjump_target_id"] == ""
    assert committed["version"] == 96
    assert committed["obligations"][0] == rh_c1_before
    root = next(
        item for item in committed["obligations"]
        if item["obligation_id"] == first.root_id
    )
    assert root["parent_id"] == ""
    assert root["status"] == "UNRESOLVED"
    assert root["formal_status"] == "FORMALIZED"
    assert root["proposition_hash"] == first.proposition_hash
    assert restored is not None
    assert restored.proof_state == ProofState.STRATEGY_TOURNAMENT
    assert restored.target_obligation_id == first.root_id
    assert restored.target_context_hash
    assert restored.elaborated_theorem_id == "KakeyaRiemannHypothesisRoot"
    assert restored.strategy_run_status == "CONFIGURATION_REQUIRED"
    assert restored.strategy_run_id == ""
    assert restored.strategy_intent_hash == ""
    assert restored.strategy_selection_provenance == {}
    assert restored.validated_artifacts["definition_auditor"].target_obligation_id == (
        first.root_id
    )
    assert all(
        reference.target_obligation_id != "RH-C1"
        for reference in restored.validated_artifacts.values()
    )

    second = bootstrap_rh_root(
        project_root=ROOT,
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        checkpoint=restored,
        ledger=committed,
    )
    assert not second.changed
    assert second.certificate_hash == first.certificate_hash
    assert json.loads(ledger_path.read_text())["version"] == 96
    records = [
        json.loads(line)
        for line in (
            tmp_path / "proof_root_bootstrap.journal.jsonl"
        ).read_text().splitlines()
    ]
    assert [item["phase"] for item in records] == ["PREPARED", "COMMITTED"]
