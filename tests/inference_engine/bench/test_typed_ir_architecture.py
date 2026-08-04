import json
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.prefill.host_compiler import run_host_gates
from autoresearch.prefill.creative_decomposition import MOVE_REGISTRY
from autoresearch.prefill.definition_registry import (
    build_definition_choice_registry,
    serialize_definition_audit,
)
from autoresearch.prefill.lean_gate import validate_lean_proof, validate_lean_signature
from autoresearch.prefill.math_ir import (
    GateStatus,
    TypedIRError,
    build_decomposer_candidate_registry,
    build_lean_declaration,
    parse_math_ir,
    parse_proof_plan,
    render_proof_plan,
    validate_dependency_graph,
    validate_math_ir,
    verify_proposition_equivalence,
)
from autoresearch.prefill.orchestration_state import (
    ARCHITECTURE_VERSION,
    SCHEMA_VERSION,
    CheckpointCompatibilityError,
    OrchestrationCheckpoint,
    ProofState,
    current_capability_manifest,
    integration_block_incompatible_checkpoint,
    load_checkpoint,
    require_typed_dispatch,
    save_checkpoint,
)
from autoresearch.prefill.typed_transport import (
    AdapterError,
    ROLE_TRANSPORT_REGISTRY,
    decode_role_fields,
    host_artifact,
    transport_prompt,
    typed_transport_complete,
)
from scripts import agent_gan_repl


ROOT = Path(__file__).resolve().parents[3]
VALID_IR = (
    "symbol truth True",
    "conclusion truth",
)


def _transport_fixture(role):
    blocks = []
    for field in ROLE_TRANSPORT_REGISTRY[role].fields:
        if not field.required:
            continue
        if field.choices:
            value = field.choices[0]
        elif field.value_kind == "reference":
            value = f"ref:{field.name}"
        else:
            value = f"value_{field.name}"
        if field.name == "proof_step":
            value = "trivial"
        blocks.extend((field.name, value, f"/{field.name}"))
    return "\n".join((*blocks, "END"))


@pytest.mark.parametrize("role", sorted(ROLE_TRANSPORT_REGISTRY))
def test_every_model_role_uses_bounded_typed_transport(role):
    fixture = _transport_fixture(role)
    decoded = decode_role_fields(fixture, role)
    assert decoded.role == role
    assert len(decoded.transport_hash) == 64
    assert typed_transport_complete(fixture, role)
    prompt = transport_prompt(role)
    assert "{" not in prompt
    assert "Artifact" not in prompt
    assert "schema_version" not in prompt
    assert ":= by" not in prompt


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("target_ref\nref:x\n/target_ref", "INCOMPLETE"),
        ("unknown\nx\n/unknown\nEND", "UNKNOWN_FIELD"),
        ("target_ref\n```lean\n/target_ref\nEND", "MODEL_OWNED_SYNTAX"),
        ("target_ref\ntheorem x : True\n/target_ref\nEND", "MODEL_OWNED_SYNTAX"),
        ("target_ref\n$P$\n/target_ref\nEND", "MODEL_OWNED_SYNTAX"),
    ],
)
def test_adapter_failures_are_precise_and_outside_proof_state(text, code):
    checkpoint = OrchestrationCheckpoint(state=ProofState.DECOMPOSER.value)
    before = (checkpoint.mathematical_retries, dict(checkpoint.retry_counters))
    with pytest.raises(AdapterError) as raised:
        decode_role_fields(text, "definition_auditor")
    assert raised.value.code == code
    assert (checkpoint.mathematical_retries, checkpoint.retry_counters) == before


def test_host_alone_adds_version_hash_bindings_and_json_envelope():
    decoded = decode_role_fields(
        _transport_fixture("proof_action_selector"),
        "proof_action_selector",
    )
    envelope = host_artifact(
        decoded,
        host_bindings={"target": "ROOT"},
        dependencies=["a" * 64],
    )
    assert envelope["schema_version"] == 4
    assert envelope["bindings"] == {"target": "ROOT"}
    assert len(envelope["content_hash"]) == 64
    assert "schema_version" not in decoded.values


def test_definition_auditor_uses_only_registered_ids_and_host_serialization():
    target_ref = "claim:" + "a" * 64
    registry = build_definition_choice_registry(
        target_ref,
        "A fixed genus sequence density statement.",
    )
    text = "\n".join((
        f"target_ref {target_ref};",
        "symbol_id SYM_SEQUENCE;",
        "symbol_id SYM_SEQUENCE_DENSITY;",
        "domain_id DOM_COMPLEX_SEQUENCE;",
        "domain_id DOM_POSITIVE_REAL;",
        "definition_id DEF_GENUS;",
        "missing_definition_id DEF_SEQUENCE_DENSITY;",
        "audit_outcome MISSING_DEFINITION;",
        "END;",
    ))
    decoded = decode_role_fields(
        text,
        "definition_auditor",
        registered_choices=registry.registered_output_choices,
    )
    payload, envelope = serialize_definition_audit(
        decoded,
        registry,
        target_obligation_id="ROOT",
        parent_statement_hash="b" * 64,
        root_goal_hash="c" * 64,
        producer_run_id="run:definition:typed",
    )
    assert "{" not in text and "$" not in text and "\\" not in text
    assert not {"symbol", "domain", "topology"} & set(decoded.values)
    assert payload["producer_role"] == "definition_auditor"
    assert payload["missing_definitions"] == [{
        "definition_id": "DEF_SEQUENCE_DENSITY",
        "content_ref": "content:definition:sequence_density",
        "label": "Sequence density",
        "symbol_ids": ["SYM_SEQUENCE", "SYM_SEQUENCE_DENSITY"],
        "domain_ids": ["DOM_COMPLEX_SEQUENCE", "DOM_POSITIVE_REAL"],
        "topology_ids": [],
        "required_type_id": "TYPE_SEQUENCE_DENSITY",
        "obligation_label": "D1",
    }]
    assert envelope["bindings"]["artifact_kind"] == "DEFINITION_AUDIT"
    assert len(envelope["content_hash"]) == 64


def test_definition_unknown_choice_semantically_supports_reframe():
    registry = build_definition_choice_registry("claim:" + "a" * 64)
    decoded = decode_role_fields(
        "\n".join((
            f"target_ref {registry.target_ref};",
            "audit_outcome REFRAME_REQUIRED;",
            "END;",
        )),
        "definition_auditor",
        registered_choices=registry.registered_output_choices,
    )
    payload, _ = serialize_definition_audit(
        decoded,
        registry,
        target_obligation_id="ROOT",
        parent_statement_hash="b" * 64,
        root_goal_hash="c" * 64,
        producer_run_id="run:reframe",
    )
    assert len(payload["missing_definitions"]) == len(registry.definitions)


def test_compact_field_transport_matches_streaming_model_capability():
    text = "\n".join((
        "parent_claim_ref claim:abc123;",
        "child_kind LEMMA;",
        "outline_step_id derive_bound;",
        "move_id A;",
        "END;",
    ))
    decoded = decode_role_fields(
        text,
        "decomposer",
        registered_choices={
            "parent_claim_ref": ("claim:abc123",),
            "move_id": ("A",),
        },
    )
    assert decoded.values["move_id"] == "A"
    assert typed_transport_complete(text, "decomposer")


def test_valid_math_ir_structural_typing_and_hash_stability():
    first = validate_math_ir(parse_math_ir(VALID_IR))
    second = validate_math_ir(parse_math_ir(VALID_IR))
    assert first.content_hash == second.content_hash
    assert first.node_types == {"truth": "Prop"}


@pytest.mark.parametrize(
    ("lines", "code"),
    [
        (("binder x Missing", "var n x", "conclusion n"), "UNKNOWN_TYPE"),
        (("symbol n Missing", "conclusion n"), "UNKNOWN_SYMBOL"),
        (
            ("symbol t True", "op n missing t", "conclusion n"),
            "UNKNOWN_OPERATOR",
        ),
        (("var n missing", "conclusion n"), "BINDER_SCOPE"),
        (
            ("symbol t True", "op n not n", "conclusion n"),
            "EXPRESSION_CYCLE",
        ),
        (("symbol n density", "conclusion n"), "SYMBOL_TYPE_MISMATCH"),
    ],
)
def test_math_ir_fails_closed_with_provenance(lines, code):
    with pytest.raises(TypedIRError) as raised:
        validate_math_ir(parse_math_ir(lines))
    assert raised.value.code == code
    assert raised.value.owner


@pytest.mark.parametrize(
    "raw", ["theorem T := by", r"op x \forall y", "$P$", "```lean"],
)
def test_math_ir_rejects_raw_lean_latex_and_fences(raw):
    with pytest.raises(TypedIRError, match="RAW_SOURCE_FORBIDDEN"):
        parse_math_ir((raw,))


def test_dependency_cycle_and_unknown_dependency_are_semantic_backjumps():
    with pytest.raises(TypedIRError) as cycle:
        validate_dependency_graph({"A": ("B",), "B": ("A",)})
    assert cycle.value.code == "DEPENDENCY_CYCLE"
    assert cycle.value.status == GateStatus.SEMANTIC_BACKJUMP
    with pytest.raises(TypedIRError) as missing:
        validate_dependency_graph({"A": ("B",)})
    assert missing.value.code == "UNKNOWN_DEPENDENCY"


def test_deterministic_lean_builder_golden_and_equivalence():
    validated = validate_math_ir(parse_math_ir(VALID_IR))
    built = build_lean_declaration(validated)
    assert built.declaration_source == "theorem hostGeneratedTheorem : (True) := by"
    assert built == build_lean_declaration(validated)
    verify_proposition_equivalence(validated, built)
    with pytest.raises(TypedIRError, match="PROPOSITION_HASH_MISMATCH"):
        verify_proposition_equivalence(
            validated,
            replace(built, proposition_hash="0" * 64),
        )


def test_host_gates_elaborate_and_cache_idempotently(tmp_path):
    first = run_host_gates(
        VALID_IR,
        project_root=ROOT,
        cache_dir=tmp_path,
        lean_validator=validate_lean_signature,
    )
    second = run_host_gates(
        VALID_IR,
        project_root=ROOT,
        cache_dir=tmp_path,
        lean_validator=validate_lean_signature,
    )
    assert first.ok and second.ok
    assert first.compilation == second.compilation
    assert [item.stage for item in first.evidence] == [
        "TYPED_TRANSPORT_VALIDATION",
        "TYPED_MATH_IR_STRUCTURAL_VALIDATION",
        "SYMBOL_TYPE_OPERATOR_RESOLUTION",
        "LEAN_AST_SOURCE_GENERATION",
        "LEAN_ELABORATION",
        "PROPOSITION_HASH_EQUIVALENCE",
    ]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_missing_definition_elaboration_backjumps_to_decomposer(tmp_path):
    lines = (
        "binder z NatToComplex",
        "var zNode z",
        "symbol densityNode density zNode",
        "op positive gt_real densityNode densityNode",
        "conclusion positive",
    )
    result = run_host_gates(
        lines,
        project_root=ROOT,
        cache_dir=tmp_path,
        lean_validator=validate_lean_signature,
    )
    assert not result.ok
    assert result.failure_status == GateStatus.SEMANTIC_BACKJUMP.value
    assert result.backjump_owner == "decomposer"
    assert result.evidence[-1].code == "MISSING_DEFINITION_ENVIRONMENT"


def test_production_host_missing_density_routes_to_synthesis_without_budget():
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.HOST_TYPED_IR_GATE.value,
        current_role="host_typed_ir_gate",
        ledger_version=92,
        orchestration_id=(
            "br_1f3a474a5ab89814:decomposition:4463cbac63f3"
        ),
        candidate_sha256=(
            "61eea711797017589523762db5ba58e3b1df69e727ace8410b8f1a16ad4c558d"
        ),
        candidate_set_hash=(
            "da8bab10ea4cc43fb46ede77ae158d2dcbd9cac6b681ce460f04c7fddf2311d5"
        ),
        ranking_hash=(
            "9731659d2c4fca38c3c187d3eb9e61f00d23b1a67b3f314102d12a6d340c0c2d"
        ),
        selected_move_id="DENSITY_LOWER_BOUND",
    )
    budget_before = (
        checkpoint.mathematical_retries,
        dict(checkpoint.retry_counters),
        checkpoint.protocol_error_count,
    )

    checkpoint.transition(
        ProofState.SYNTHESIS,
        "typed-evidence-backjump:MISSING_DEFINITION_ENVIRONMENT",
        strategy_reused=True,
    )

    assert checkpoint.proof_state == ProofState.SYNTHESIS
    assert checkpoint.current_role == "synthesis"
    assert checkpoint.strategy_reused is False
    assert (
        checkpoint.mathematical_retries,
        checkpoint.retry_counters,
        checkpoint.protocol_error_count,
    ) == budget_before


@pytest.mark.parametrize(
    ("state", "target"),
    (
        (ProofState.HOST_TYPED_IR_GATE, ProofState.SYNTHESIS),
        (ProofState.HOST_TYPED_IR_GATE, ProofState.DECOMPOSER),
        (ProofState.HOST_TYPED_IR_GATE, ProofState.MATH_IR_TRANSLATION),
        (ProofState.LEAN_ELABORATION_GATE, ProofState.DECOMPOSER),
        (ProofState.LEAN_ELABORATION_GATE, ProofState.MATH_IR_TRANSLATION),
        (ProofState.PROOF_SEARCH, ProofState.DECOMPOSER),
        (ProofState.PROOF_SEARCH, ProofState.MATH_IR_TRANSLATION),
    ),
)
def test_typed_gate_semantic_backjumps_are_budget_neutral(state, target):
    checkpoint = OrchestrationCheckpoint(
        state=state.value,
        current_role=state.value.lower(),
    )
    before = (
        checkpoint.mathematical_retries,
        dict(checkpoint.retry_counters),
        checkpoint.protocol_error_count,
    )
    checkpoint.transition(
        target,
        "typed-evidence-backjump:production-fixture",
        strategy_reused=True,
    )
    assert checkpoint.proof_state == target
    assert (
        checkpoint.mathematical_retries,
        checkpoint.retry_counters,
        checkpoint.protocol_error_count,
    ) == before


def test_registered_proof_plan_renders_and_elaborates():
    compilation = build_lean_declaration(validate_math_ir(parse_math_ir(VALID_IR)))
    plan = parse_proof_plan(
        ("trivial",),
        theorem_id=compilation.theorem_id,
        proposition_hash=compilation.proposition_hash,
    )
    source = render_proof_plan(plan, compilation)
    assert source.endswith("\n  trivial")
    assert validate_lean_proof(source, project_root=ROOT).ok
    with pytest.raises(TypedIRError, match="UNSUPPORTED_TACTIC"):
        parse_proof_plan(
            ("simp_all",),
            theorem_id=compilation.theorem_id,
            proposition_hash=compilation.proposition_hash,
        )


def test_v4_checkpoint_migrates_free_text_transport_to_audit_only(tmp_path):
    path = tmp_path / "checkpoint.json"
    legacy_field = "protocol_" + "attempt"
    path.write_text(json.dumps({
        "schema_version": 4,
        "state": "BLOCKED",
        "current_role": "blocked",
        legacy_field: 2,
        "validated_artifacts": {
            "decomposer": {
                "role": "decomposer",
                "sha256": "d" * 64,
                "schema_version": 1,
                "dependencies": [],
                "path": "/audit/decomposer.json",
                "source_run_id": "old",
                "validated_at": 1.0,
            },
        },
    }))
    migrated = load_checkpoint(path)
    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.architecture_version == ARCHITECTURE_VERSION
    assert migrated.proof_state == ProofState.STRATEGY_TOURNAMENT
    assert "d" * 64 in migrated.invalidated_artifacts
    assert migrated.invalidated_artifacts["d" * 64]["audit_only"]
    assert legacy_field not in migrated.__dataclass_fields__
    save_checkpoint(path, migrated)
    serialized = json.loads(path.read_text())
    assert legacy_field not in serialized


def test_registry_has_no_model_owned_mathematical_text_escape_fields():
    forbidden = {
        "claim", "proposition", "statement", "assumption", "conclusion",
        "source", "lean", "json", "text",
    }
    allowed = {"parent_claim_ref", "target_proposition_ref"}
    for contract in ROLE_TRANSPORT_REGISTRY.values():
        for field in contract.fields:
            assert field.name in allowed or not any(
                fragment in field.name for fragment in forbidden
            )
            assert field.value_kind != "free_text"


def test_decomposer_prompt_snapshot_has_only_refs_and_choice_selector():
    prompt = transport_prompt(
        "decomposer",
        registered_choices={
            "parent_claim_ref": ("claim:" + "a" * 64,),
            "child_kind": ("DEFINITION",),
            "definition_id": ("DEF_DENSITY",),
            "move_id": ("A", "B"),
        },
    )
    assert "claim (required)" not in prompt
    assert "assumption" not in prompt
    assert "conclusion (required)" not in prompt
    assert "proof_outline" not in prompt
    assert "math_ir" not in prompt
    assert "parent_claim_ref" in prompt
    assert "typed_ir_step" not in prompt
    assert "move_id" in prompt
    assert "choose exactly A or B" in prompt
    assert "choose exactly claim:" + "a" * 64 in prompt


def test_decomposer_transport_to_host_gate_has_zero_model_json_or_lean(tmp_path):
    parent_ref = "claim:" + "a" * 64
    registry = build_decomposer_candidate_registry(
        target_ref=parent_ref,
        viewpoint="definitions",
        dependency_ids=("d" * 64,),
    )
    text = "\n".join((
        f"parent_claim_ref {parent_ref};",
        "child_kind LEMMA;",
        "move_id A;",
        "END;",
    ))
    decoded = decode_role_fields(
        text,
        "decomposer",
        registered_choices={
            "parent_claim_ref": (parent_ref,),
            "child_kind": ("LEMMA",),
            "move_id": registry.choices,
        },
    )
    selected = registry.resolve(decoded.values["move_id"])
    envelope = host_artifact(
        decoded,
        host_bindings={
            "target_hash": "b" * 64,
            "candidate_hash": selected.candidate_hash,
        },
        dependencies=(),
    )
    assert "{" not in text and "theorem" not in text and ":= by" not in text
    assert envelope["bindings"]["target_hash"] == "b" * 64
    result = run_host_gates(
        selected.typed_payload,
        project_root=ROOT,
        cache_dir=tmp_path,
        lean_validator=validate_lean_signature,
    )
    assert result.compilation.declaration_source.startswith("theorem ")
    assert "LEAN_AST_SOURCE_GENERATION" in {
        evidence.stage for evidence in result.evidence
    }
    assert result.failure_status in {"", GateStatus.SEMANTIC_BACKJUMP.value}


def test_candidate_registry_is_complete_scoped_typed_and_hash_bound():
    registry = build_decomposer_candidate_registry(
        target_ref="claim:" + "a" * 64,
        viewpoint="constructive_witness",
        dependency_ids=("d" * 64,),
    )
    assert registry.choices == (
        "A", "B", "C", "D", "CASE_SPLIT", "RESTRICT_DOMAIN",
        "REMOVE_IRRELEVANT_ASSUMPTION", "HOLOMORPHIC_EXTENSION",
        "SINGULARITY_CONTRADICTION",
    )
    assert len(registry.registry_hash) == 64
    for candidate in registry.candidates:
        validated = validate_math_ir(parse_math_ir(candidate.typed_payload))
        assert validated.content_hash == candidate.typed_ir_hash
        assert len(candidate.candidate_hash) == 64


def test_selector_rejects_out_of_scope_choice_without_registry_mutation():
    registry = build_decomposer_candidate_registry(
        target_ref="claim:" + "a" * 64,
        viewpoint="definitions",
    )
    before = registry
    with pytest.raises(TypedIRError) as raised:
        registry.resolve("Z")
    assert raised.value.code == "INVALID_DECOMPOSITION_CHOICE"
    assert raised.value.status == GateStatus.ADAPTER_BLOCKED
    assert registry == before


def test_empty_allowed_moves_produces_semantic_no_move_registry():
    registry = build_decomposer_candidate_registry(
        target_ref="claim:" + "a" * 64,
        viewpoint="definitions",
        allowed_transformations=(),
    )
    assert registry.candidates == ()
    assert registry.choices == ()


def test_latest_model_owned_invalid_dsl_line_is_forbidden_transport():
    text = "\n".join((
        "parent_claim_ref claim:abc123;",
        "child_kind DEFINITION;",
        "definition_id L1;",
        "typed_ir_step binder d density;",
        "END;",
    ))
    with pytest.raises(AdapterError) as raised:
        decode_role_fields(text, "decomposer")
    assert raised.value.code == "UNKNOWN_FIELD"
    assert raised.value.field == "typed_ir_step"


@pytest.mark.parametrize(
    "legacy_field",
    (
        "claim", "assumption", "conclusion", "proof_outline", "math_ir",
        "typed_ir_step",
    ),
)
def test_legacy_free_text_decomposer_fields_fail_before_proof_state(legacy_field):
    text = f"{legacy_field}\nplain text\n/{legacy_field}\nEND"
    with pytest.raises(AdapterError) as raised:
        decode_role_fields(text, "decomposer")
    assert raised.value.code == "UNKNOWN_FIELD"


def test_new_state_machine_has_explicit_zero_agent_gates():
    values = {state.value for state in ProofState}
    assert {
        "DECOMPOSER",
        "MATH_IR_TRANSLATION",
        "HOST_TYPED_IR_GATE",
        "LEAN_ELABORATION_GATE",
        "PROOF_SEARCH",
        "ADVERSARIAL_REVIEW",
        "JUDGE",
    } <= values
    assert "FORMALIZER" not in values
    assert "PROVER" not in values


def test_production_schema_lost_migration_event_still_typed_dispatches(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        architecture_version=5,
        schema_version=7,
        migration_event="",
        ledger_version=89,
        candidate_sha256="61eea711797017589523762db5ba58e3b1df69e727ace8410b8f1a16ad4c558d",
    )
    save_checkpoint(path, checkpoint)
    recovered = load_checkpoint(path)
    assert recovered is not None
    require_typed_dispatch(recovered)
    dispatch_source = inspect.getsource(agent_gan_repl.run_certified_decomposition)
    assert ".migration_event" not in dispatch_source
    assert "return _run_typed_ir_v2(" in dispatch_source


def test_architecture_or_registry_mismatch_integration_blocks():
    architecture = OrchestrationCheckpoint(architecture_version=4)
    assert integration_block_incompatible_checkpoint(architecture)
    assert architecture.adapter_status == "INTEGRATION_BLOCKED"
    with pytest.raises(CheckpointCompatibilityError):
        require_typed_dispatch(architecture)
    registry = OrchestrationCheckpoint(typed_transport_registry_hash="0" * 64)
    assert integration_block_incompatible_checkpoint(registry)
    assert "TYPED_TRANSPORT_REGISTRY_HASH_MISMATCH" in registry.blocked_reason


def test_checkpoint_recreation_initializes_atomic_capability_manifest(tmp_path):
    checkpoint = OrchestrationCheckpoint()
    manifest = current_capability_manifest()
    for name, expected in manifest.items():
        assert getattr(checkpoint, name) == expected
    path = tmp_path / "proof_orchestration.json"
    save_checkpoint(path, checkpoint)
    persisted = json.loads(path.read_text())
    assert {
        name: persisted[name] for name in manifest
    } == manifest
    require_typed_dispatch(load_checkpoint(path))


def test_legacy_capability_document_migrates_to_tournament(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    path.write_text(json.dumps({
        "state": "DECOMPOSER",
        "architecture_version": 5,
        "schema_version": 7,
        "migration_event": "",
    }))
    checkpoint = load_checkpoint(path)
    assert checkpoint.adapter_status == ""
    assert checkpoint.proof_state == ProofState.STRATEGY_TOURNAMENT
    assert checkpoint.migration_event == "cursor_strategy_oprover_advisor_v1"


def test_host_dual_injection_preserves_malformed_assumption_bytes_and_hash():
    malformed_fixture = [r"$p \in \mathbb{N$", r"$\epsilon > 0$"]
    assumptions, digest, restriction = (
        agent_gan_repl._host_owned_assumption_contract(
            malformed_fixture,
            MOVE_REGISTRY["DENSITY_LOWER_BOUND"],
        )
    )
    assert assumptions == malformed_fixture
    assert assumptions[0].endswith(r"\mathbb{N$")
    assert digest == agent_gan_repl._canonical_json_hash(malformed_fixture)
    assert restriction is None


def test_restrict_domain_is_explicit_typed_move_not_hidden_assumption():
    assumptions = [r"$p \in \mathbb{N}$"]
    injected, digest, restriction = (
        agent_gan_repl._host_owned_assumption_contract(
            assumptions,
            MOVE_REGISTRY["RESTRICT_DOMAIN"],
        )
    )
    assert injected == assumptions
    assert digest == agent_gan_repl._canonical_json_hash(assumptions)
    assert MOVE_REGISTRY["RESTRICT_DOMAIN"].assumptions_added == ()
    assert restriction["move_id"] == "RESTRICT_DOMAIN"
    assert restriction["provenance"] == "host_move_registry"
    assert restriction["antecedent_ids"] == [
        "positive_radius", "subdomain_of_parent",
    ]
    assert len(restriction["content_hash"]) == 64


def test_legacy_decomposer_repair_is_unreachable_from_architecture5():
    typed_source = inspect.getsource(agent_gan_repl._run_typed_ir_v2)
    for legacy_name in (
        "_decomposer_repair_messages",
        "_decomposer_payload_result",
        "parse_certified_artifact",
        "repair_json_backslashes",
    ):
        assert legacy_name not in typed_source


def test_blocked_checkpoint_is_journaled_and_restart_deterministic(tmp_path):
    path = tmp_path / "proof_orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        migration_event="",
    )
    checkpoint.adapter_blocked(
        "fixture capability block",
        status="INTEGRATION_BLOCKED",
    )
    save_checkpoint(path, checkpoint)
    journal = path.with_name("proof_orchestration.journal.jsonl")
    record = json.loads(journal.read_text().splitlines()[-1])
    assert record["kind"] == "blocked_transition"
    assert record["adapter_status"] == "INTEGRATION_BLOCKED"
    first = load_checkpoint(path)
    second = load_checkpoint(path)
    assert first.state == second.state
    assert first.adapter_status == second.adapter_status
    assert first.typed_transport_registry_hash == second.typed_transport_registry_hash
    assert first.capability_flags == second.capability_flags
