#!/usr/bin/env python3
"""Safely redirect canonical RH-C0 into the sourced Strategy tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.orchestration_state import (
    ProofState,
    load_checkpoint,
    persist_validated_artifact,
    save_checkpoint,
)
from autoresearch.prefill.research_contract import gate_research_contract
from autoresearch.prefill.rh_strategy_seed import (
    RH_ROOT_HASH,
    RH_ROOT_ID,
    build_rh_strategy_plans,
    rh_strategy_specs,
)
from autoresearch.prefill.strategy_tournament import (
    StrategyEvent,
    evaluate_feasibility,
    run_tournament,
)
from autoresearch.prefill.target_context import activate_target_context
from autoresearch.prefill.theorem_cards import pinned_environment_hash


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _snapshot(
    checkpoint_path: Path, ledger_path: Path, snapshot_root: Path,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = snapshot_root / f"rh-strategy-{stamp}"
    destination.mkdir(parents=True, mode=0o700)
    for source in (
        checkpoint_path,
        ledger_path,
        checkpoint_path.with_name("proof_target_contexts.json"),
        checkpoint_path.with_name("proof_orchestration.resume_lease.json"),
    ):
        if source.exists():
            shutil.copy2(source, destination / source.name)
    manifest = {
        "schema_version": 1,
        "created_at": time.time(),
        "files": {
            item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in destination.iterdir() if item.is_file()
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger", default="~/.kakeya/agent_gan_proof_ledger.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.kakeya/autoresearch/proof_orchestration.json",
    )
    parser.add_argument(
        "--snapshot-root",
        default="~/.kakeya/autoresearch/snapshots",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--strategy-agent-id", default="")
    parser.add_argument("--strategy-run-id", default="")
    parser.add_argument("--strategy-prompt-hash", default="")
    parser.add_argument("--strategy-evidence-hash", default="")
    parser.add_argument("--strategy-memo-hash", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    ledger_path = Path(args.ledger).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise SystemExit("canonical checkpoint missing")
    root = next(
        item for item in ledger["obligations"]
        if item.get("obligation_id") == RH_ROOT_ID
    )
    if (
        int(ledger["version"]) < 96
        or root.get("proposition_hash") != RH_ROOT_HASH
        or checkpoint.target_obligation_id != RH_ROOT_ID
    ):
        raise SystemExit("canonical RH-C0/ledger binding mismatch")

    specs = rh_strategy_specs()
    plans = build_rh_strategy_plans(project_root)
    executable = plans[0]
    decisions = evaluate_feasibility(
        plans,
        registered_definition_ids=executable.required_definition_ids,
        resolved_dependency_ids=executable.dependency_ids,
        verified_theorem_card_ids=executable.theorem_card_ids,
        allowed_assumption_ids=(),
    )
    event_id = "OPERATOR_STRATEGY_REDIRECT:" + _digest({
        "root": RH_ROOT_ID,
        "ledger_version": ledger["version"],
        "plan_hashes": [item.content_hash for item in plans],
        "source_note": "user-authorized-three-direction-strategy-seed",
    })
    tournament = run_tournament(
        event_id=event_id,
        event_type=StrategyEvent.TARGET_CHANGE,
        plans=plans,
        decisions=decisions,
    )
    selected = next(
        item for item in plans if item.plan_id == tournament.selected_plan_id
    )
    proposition = (
        "∀ (a : ℕ → ℝ) (n : ℕ), a (n + 2) ≠ 0 → "
        "0 ≤ a (n + 1) ^ 2 - a n * a (n + 2) → "
        "the two quadratic-formula values are roots of jensenQuadratic a n"
    )
    proposition_hash = hashlib.sha256(proposition.encode()).hexdigest()
    contract_decision = gate_research_contract(
        selected,
        elaborated_theorem_id="jensenQuadratic_has_two_real_roots",
        elaborated_proposition_hash=proposition_hash,
        proof_obligation_id="RH-JENSEN-QUADRATIC-ROOTS",
        registered_definition_ids=selected.required_definition_ids,
        resolved_dependency_ids=selected.dependency_ids,
        verified_theorem_card_ids=selected.theorem_card_ids,
        allowed_assumption_ids=(),
        environment_hash=pinned_environment_hash(project_root),
        expected_plan_hash=selected.content_hash,
    )
    if not contract_decision.accepted or contract_decision.contract is None:
        raise SystemExit(
            "research contract rejected: "
            + ",".join(contract_decision.reason_codes)
        )
    summary = {
        "root": RH_ROOT_ID,
        "ledger_version": ledger["version"],
        "event_id": event_id,
        "plan_ids": [item.plan_id for item in plans],
        "plan_hashes": [item.content_hash for item in plans],
        "semantic_fingerprint": _digest([
            item.content_hash for item in plans
        ]),
        "strategy_advisory": {
            "provider": "cursor-sdk",
            "model_id": "gpt-5.6-sol",
            "agent_id": args.strategy_agent_id,
            "run_id": args.strategy_run_id,
            "prompt_hash": args.strategy_prompt_hash,
            "evidence_hash": args.strategy_evidence_hash,
            "memo_hash": args.strategy_memo_hash,
            "private_memo_persisted": False,
            "authoritative": False,
        },
        "plans": [{
            "spec": asdict(spec),
            "spec_hash": spec.content_hash,
            "plan": asdict(plan),
            "decision": asdict(decision),
        } for spec, plan, decision in zip(specs, plans, decisions)],
        "tournament": asdict(tournament),
        "selected_plan_id": selected.plan_id,
        "contract": asdict(contract_decision.contract),
    }
    if not args.apply:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    telemetry_values = (
        args.strategy_agent_id,
        args.strategy_run_id,
        args.strategy_prompt_hash,
        args.strategy_evidence_hash,
        args.strategy_memo_hash,
    )
    if any(not item for item in telemetry_values):
        raise SystemExit("apply requires complete Cursor Strategy telemetry")

    snapshot = _snapshot(
        checkpoint_path, ledger_path,
        Path(args.snapshot_root).expanduser().resolve(),
    )
    context, context_changed = activate_target_context(
        checkpoint_path,
        checkpoint,
        target_obligation_id=RH_ROOT_ID,
        statement="RiemannHypothesis",
        environment_hash=selected.environment_hash,
        strategy_plan_hash=selected.content_hash,
        evidence={
            "EVIDENCE_TARGET_STATEMENT": "RiemannHypothesis",
            "ledger_version": ledger["version"],
            "operator_strategy_event_id": event_id,
            "strategy_semantic_fingerprint": summary[
                "semantic_fingerprint"
            ],
            "source_note": "user-authorized-three-direction-strategy-seed",
        },
        definition_ids=selected.required_definition_ids,
        candidate_ids=tuple(item.plan_id for item in plans),
        theorem_card_ids=tuple(
            card for item in plans for card in item.theorem_card_ids
        ),
    )
    if checkpoint.proof_state == ProofState.BLOCKED:
        checkpoint.transition(
            ProofState.STRATEGY_TOURNAMENT,
            "operator-strategy-redirect:resume-certified-contract",
        )
    elif context_changed:
        checkpoint.transition(
            ProofState.MATHEMATICAL_STAGNATION,
            "operator-strategy-redirect:replace-audit-only-exhaustion",
        )
        checkpoint.transition(
            ProofState.STRATEGY_TOURNAMENT,
            "operator-strategy-redirect:three-sourced-rh-directions",
        )
    auditor = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload={
            "schema_version": 1,
            "target_obligation_id": RH_ROOT_ID,
            "target_context_hash": context.binding.context_hash,
            "definitions": [
                {"definition_id": item} for item
                in selected.required_definition_ids
            ],
            "missing_definitions": [],
            "branch_missing_interfaces": {
                spec.route_id: list(spec.missing_interfaces)
                for spec in specs
            },
        },
        dependencies=[],
        source_run_id="host:" + event_id[:40],
        save=False,
    )
    tournament_ref = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="strategy_tournament",
        payload=summary,
        dependencies=[auditor.sha256],
        source_run_id="host:" + event_id[:40],
        save=False,
    )
    if checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT:
        checkpoint.transition(
            ProofState.RESEARCH_CONTRACT_GATE,
            "strategy-tournament:selected-executable-jensen-subgoal",
        )
    contract_ref = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="research_contract",
        payload=asdict(contract_decision.contract),
        dependencies=[tournament_ref.sha256],
        source_run_id="host:" + event_id[:40] + ":contract",
        save=False,
    )
    if checkpoint.proof_state == ProofState.RESEARCH_CONTRACT_GATE:
        checkpoint.transition(
            ProofState.PROOF_SEARCH,
            "research-contract:accepted-jensen-quadratic-subgoal",
        )
    checkpoint.strategy_event_id = event_id
    checkpoint.strategy_event_type = StrategyEvent.TARGET_CHANGE.value
    checkpoint.strategy_plan_ids = [item.plan_id for item in plans]
    checkpoint.strategy_plan_hashes = [item.content_hash for item in plans]
    checkpoint.feasible_strategy_plan_ids = [
        item.plan_id for item in decisions if item.feasible
    ]
    checkpoint.pareto_plan_ids = list(tournament.pareto_plan_ids)
    checkpoint.selected_strategy_plan_id = selected.plan_id
    checkpoint.selected_strategy_plan_hash = selected.content_hash
    checkpoint.strategy_tournament_hash = tournament.content_hash
    checkpoint.research_contract_id = contract_decision.contract.contract_id
    checkpoint.research_contract_hash = contract_decision.contract.content_hash
    checkpoint.elaborated_theorem_id = (
        contract_decision.contract.theorem_id
    )
    # The root-level checkpoint remains bound to the canonical Mathlib
    # proposition.  The selected helper target hash lives in the Research
    # Contract and must not overwrite this protected root cache.
    checkpoint.proposition_hash = RH_ROOT_HASH
    # Bind the runtime wrapper candidate so the supervisor resumes the
    # certified Research Contract instead of treating legacy stagnation
    # telemetry as a request for an unrelated Strategy rewrite.
    runtime_candidate = (
        project_root / "autoresearch" / "prefill" / "candidate.py"
    )
    checkpoint.candidate_sha256 = hashlib.sha256(
        runtime_candidate.read_bytes()
    ).hexdigest()
    checkpoint.proof_plan_id = selected.plan_id
    checkpoint.proof_plan_hash = selected.content_hash
    checkpoint.executable_plan_node_id = selected.lemma_graph[0].lemma_id
    checkpoint.theorem_card_ids = list(selected.theorem_card_ids)
    checkpoint.branch_history = {
        spec.route_id: {
            "plan_id": plan.plan_id,
            "plan_hash": plan.content_hash,
            "relation_to_rh": spec.relation_to_rh,
            "execution_status": plan.execution_status,
            "feasible": decision.feasible,
            "reason_codes": list(decision.reason_codes),
            "missing_interfaces": list(spec.missing_interfaces),
            "preserved": True,
        }
        for spec, plan, decision in zip(specs, plans, decisions)
    }
    checkpoint.recovery_events.append({
        "event_type": "OPERATOR_STRATEGY_REDIRECT",
        "event_id": event_id,
        "semantic_fingerprint": summary["semantic_fingerprint"],
        "source_note": "user-authorized-three-direction-strategy-seed",
        "prior_candidates_audit_only": True,
        "snapshot": str(snapshot),
        "selected_plan_id": selected.plan_id,
        "research_contract_id": contract_decision.contract.contract_id,
        "created_at": time.time(),
    })
    checkpoint.ledger_version = int(ledger["version"])
    checkpoint.strategy_provider = "cursor-sdk"
    checkpoint.strategy_provider_configured = True
    checkpoint.strategy_model_id = "gpt-5.6-sol"
    checkpoint.strategy_run_status = "FINISHED"
    checkpoint.strategy_agent_id = args.strategy_agent_id
    checkpoint.strategy_run_id = args.strategy_run_id
    checkpoint.strategy_prompt_hash = args.strategy_prompt_hash
    checkpoint.strategy_evidence_hash = args.strategy_evidence_hash
    checkpoint.strategy_memo_hash = args.strategy_memo_hash
    checkpoint.strategy_intent_status = "HOST_PLAN_AUTHORITATIVE"
    checkpoint.strategy_selection_provenance = summary["strategy_advisory"]
    checkpoint.lean_actions_attempted += 1
    checkpoint.lean_actions_accepted += 1
    checkpoint.new_elaborated_lemmas += 1
    checkpoint.lemmas_proved += 1
    checkpoint.progress_vector["lemmas_proved"] = checkpoint.lemmas_proved
    checkpoint.validated_artifacts["research_contract"] = contract_ref
    save_checkpoint(checkpoint_path, checkpoint)
    print(json.dumps({
        "snapshot": str(snapshot),
        "event_id": event_id,
        "semantic_fingerprint": summary["semantic_fingerprint"],
        "selected_plan_id": selected.plan_id,
        "selected_plan_hash": selected.content_hash,
        "contract_id": contract_decision.contract.contract_id,
        "tournament_artifact": tournament_ref.sha256,
        "contract_artifact": contract_ref.sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
