#!/usr/bin/env python3
"""Mandatory static architecture gate for model/host ownership."""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.prefill.orchestration_state import (
    ALLOWED_TRANSITIONS,
    ARCHITECTURE_VERSION,
    OrchestrationCheckpoint,
    ProofState,
)
from autoresearch.prefill.typed_transport import (
    ROLE_TRANSPORT_REGISTRY,
    transport_prompt,
)
from scripts import agent_gan_repl


EXPECTED_ROLES = {
    "strategy_tournament_selector",
    "strategy_critic_selector",
    "synthesis",
    "premise_suspicion",
    "premise_auditor",
    "premise_proponent",
    "definition_auditor",
    "counterexample_worker",
    "decomposer",
    "math_ir_translator",
    "proof_action_selector",
    "adversarial_proponent",
    "judge",
}
FORBIDDEN_ACTIVE_PROMPT_TEXT = (
    "json",
    "artifact:",
    "schema_version",
    "```",
    ":= by",
    "lean source",
    "latex",
)
FORBIDDEN_MODEL_FIELD_FRAGMENTS = (
    "claim",
    "proposition",
    "statement",
    "assumption",
    "conclusion",
    "source",
    "lean",
    "json",
    "text",
)
ALLOWED_REFERENCE_FIELDS = {"parent_claim_ref", "target_proposition_ref"}


def _fail(message: str) -> None:
    raise SystemExit(f"typed role contract violation: {message}")


def main() -> None:
    if set(ROLE_TRANSPORT_REGISTRY) != EXPECTED_ROLES:
        _fail("typed transport role registry is incomplete")
    for role in sorted(EXPECTED_ROLES):
        contract = ROLE_TRANSPORT_REGISTRY[role]
        for field in contract.fields:
            if (
                any(
                    fragment in field.name
                    for fragment in FORBIDDEN_MODEL_FIELD_FRAGMENTS
                )
                and field.name not in ALLOWED_REFERENCE_FIELDS
            ):
                _fail(f"{role} has model-owned escape field {field.name}")
            if field.value_kind not in {
                "enum", "identifier", "reference", "proof_step",
            }:
                _fail(f"{role}.{field.name} is not field-typed")
        prompt = transport_prompt(role).casefold()
        for forbidden in FORBIDDEN_ACTIVE_PROMPT_TEXT:
            if forbidden in prompt:
                _fail(f"{role} prompt requests forbidden syntax: {forbidden}")
    active_source = inspect.getsource(agent_gan_repl._typed_role_messages)
    active_tree = ast.parse(active_source)
    forbidden_calls = []
    for node in ast.walk(active_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = getattr(node.func.value, "id", "")
            if owner == "json" and node.func.attr in {"loads", "dumps"}:
                forbidden_calls.append(f"json.{node.func.attr}")
    if forbidden_calls:
        _fail("active role prompt serializes JSON")
    for role in ("decomposer", "math_ir_translator"):
        fields = {field.name for field in ROLE_TRANSPORT_REGISTRY[role].fields}
        if "move_id" not in fields or "typed_ir_step" in fields:
            _fail(f"{role} must use only the Host constrained move selector")
    synthesis_fields = {
        field.name for field in ROLE_TRANSPORT_REGISTRY["synthesis"].fields
    }
    if synthesis_fields != {"choice_code", "reason_code"}:
        _fail("synthesis must expose only one short choice and one reason code")
    definition_fields = {
        field.name
        for field in ROLE_TRANSPORT_REGISTRY["definition_auditor"].fields
    }
    if definition_fields != {
        "target_ref",
        "symbol_id",
        "domain_id",
        "topology_id",
        "definition_id",
        "missing_definition_id",
        "audit_outcome",
    }:
        _fail("Definition Auditor must expose only constrained registry IDs")
    synthesis_prompt = transport_prompt(
        "synthesis",
        registered_choices={"choice_code": tuple("ABCDEFGHI")},
    )
    if "candidate_id" in synthesis_prompt or "C1_" in synthesis_prompt:
        _fail("synthesis transport exposes a long candidate ID")
    state_fields = set(OrchestrationCheckpoint.__dataclass_fields__)
    removed_field = "protocol_" + "attempt"
    if removed_field in state_fields:
        _fail("adapter attempts remain in mathematical checkpoint")
    source = (
        Path(agent_gan_repl.__file__).read_text(encoding="utf-8")
        + Path(inspect.getfile(OrchestrationCheckpoint)).read_text(encoding="utf-8")
    )
    migration_marker = "typed_ir_host_constrained_selector_v2"
    if migration_marker not in source:
        _fail("production v2 migration path is absent")
    if "creative_decomposition_synthesis_moves_v3" not in source:
        _fail("production creative decomposition v3 migration path is absent")
    dispatch_source = inspect.getsource(
        agent_gan_repl.run_certified_decomposition,
    )
    if ".migration_event" in dispatch_source:
        _fail("migration_event is used as an executable feature flag")
    required_dispatch = (
        "require_typed_dispatch(orchestration_checkpoint)",
        "return _run_typed_ir_v2(",
    )
    if any(marker not in dispatch_source for marker in required_dispatch):
        _fail("typed dispatch is not fail-closed")
    if ARCHITECTURE_VERSION < 7:
        _fail("strategy tournament replacement requires architecture 7")
    for legacy in (
        ProofState.NEEDS_STRATEGY,
        ProofState.GENERATOR,
        ProofState.CRITIC,
    ):
        if ALLOWED_TRANSITIONS[legacy]:
            _fail(f"legacy state has executable edges: {legacy.value}")
    proof_fields = {
        field.name
        for field in ROLE_TRANSPORT_REGISTRY["proof_action_selector"].fields
    }
    if proof_fields != {
        "goal_id", "action_id", "operand_id", "substitution_id",
    }:
        _fail("proof selector must expose short registered IDs only")
    typed_source = inspect.getsource(agent_gan_repl._run_typed_ir_v2)
    for legacy_call in (
        "_decomposer_repair_messages(",
        "_decomposer_payload_result(",
        "parse_certified_artifact(",
        "repair_json_backslashes(",
        "parse_proof_plan(",
        "render_proof_plan(",
        'run_role(\n            "proof_search"',
    ):
        if legacy_call in typed_source:
            _fail(f"typed path reaches legacy call {legacy_call}")
    print(
        f"typed role contracts passed: {len(EXPECTED_ROLES)} roles, "
        f"{len(state_fields)} checkpoint fields",
    )


if __name__ == "__main__":
    main()
