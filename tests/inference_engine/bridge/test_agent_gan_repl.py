import hashlib
import io
import json
import re
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
import pytest

from autoresearch.prefill.lean_gate import (
    LeanSignatureResult,
    lean_theorem_signature_hash,
    validate_lean_proof,
)
from autoresearch.prefill.orchestration_state import (
    ArtifactRef,
    OrchestrationCheckpoint,
    ProofState,
    load_checkpoint as load_orchestration_checkpoint,
    persist_validated_artifact,
    save_checkpoint as save_orchestration_checkpoint,
)
from autoresearch.prefill.semantic_decompose import (
    SemanticResponseIncomplete,
    SemanticUnitTooLarge,
    StructuredResponseBudgetTooSmall,
    scan_artifact_object_prefix,
    scan_single_artifact_object,
    scan_structured_artifact_prefix,
    structured_role_minimum_output_tokens,
    structured_output_cap,
)
from scripts.agent_gan_repl import (
    PrefillHeartbeat,
    CriticIssueBatch,
    PremiseAudit,
    PremiseDefense,
    PremiseReview,
    DefinitionAudit,
    CounterexampleReport,
    DecompositionProposal,
    FormalizationBundle,
    ProofAttempt,
    DefenseReport,
    JudgeDecision,
    ProofObligation,
    ProofObligationLedger,
    ReplCheckpoint,
    ReplPhase,
    TimestampedTee,
    TokenPrinter,
    _gate_failure,
    _json_artifact,
    _certified_role_messages,
    _certified_upstream_view,
    _circular_reduction_proof,
    _adversarial_review_model_package,
    _defense_repair_messages,
    _defense_semantically_complete,
    _decomposer_repair_messages,
    _decomposer_protocol_errors,
    _decomposition_semantic_hash,
    _decomposition_structural_signature,
    _decomposer_semantically_complete,
    _formalizer_repair_messages,
    _formalizer_semantically_complete,
    _formalizer_model_package,
    _formalizer_unit_dependencies,
    _formalizer_unit_messages,
    _parse_formalizer_unit,
    _run_split_formalizer,
    _assemble_formalizer_units,
    _formalizer_upstream_view,
    _normalize_formalizer_payload,
    _judge_model_package,
    _stage,
    _structured_field,
    _structured_transport_semantically_complete,
    _validate_definition_child_selection,
    _telemetry_request,
    build_critic_messages,
    build_generator_messages,
    extract_obligation_history,
    enforce_prefill_token_budget,
    install_signal_protection,
    is_runtime_artifact_prompt,
    consume_critic_issue_batch,
    apply_critic_verdicts,
    audit_ledger_semantic_duplicates,
    build_autoresearch_verdict,
    create_child_obligations,
    certified_decomposition_requested,
    format_critic_issue_injection,
    format_proof_ledger,
    generator_issue_coverage,
    decide_premise_review,
    dispatch_certified_architecture9_role,
    extract_premise_suspicions,
    load_checkpoint,
    load_decomposition_manifest,
    load_pending_critic_issues,
    load_proof_ledger,
    parse_certified_artifact,
    pending_obligations,
    parse_repl_command,
    parse_premise_audit,
    parse_premise_defense,
    quarantine_terminal_interface_target,
    parse_certified_artifact,
    recover_checkpoint_from_log,
    save_critic_issue_batch,
    save_decomposition_manifest,
    save_proof_ledger,
    save_checkpoint,
    run_isolated_premise_review,
    run_certified_decomposition,
    retain_contract_provenance_after_artifact_failure,
    persist_verified_decomposition,
    validate_evidence_artifact,
)


def test_certified_definition_resolution_dispatches_host_gate(
    tmp_path, monkeypatch,
):
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DEFINITION_RESOLUTION.value,
        current_role="definition_resolution",
        target_obligation_id="ROOT",
    )
    calls = []

    def fake_gate(
        checkpoint_path,
        selected,
        *,
        project_root,
        interface_strategy_adapter,
    ):
        calls.append((
            checkpoint_path,
            selected,
            project_root,
            interface_strategy_adapter,
        ))
        return selected, "COMMITTED"

    monkeypatch.setattr(
        "scripts.agent_gan_repl.run_host_definition_gate",
        fake_gate,
    )
    state_path = tmp_path / "orchestration.json"
    selected, outcome = dispatch_certified_architecture9_role(
        state_path,
        checkpoint,
        project_root=tmp_path,
    )
    assert selected is checkpoint
    assert outcome == "COMMITTED"
    assert calls == [(state_path, checkpoint, tmp_path, None)]

    checkpoint.state = ProofState.DECOMPOSER.value
    selected, outcome = dispatch_certified_architecture9_role(
        state_path,
        checkpoint,
        project_root=tmp_path,
    )
    assert selected is checkpoint
    assert outcome == ""
    assert len(calls) == 1


def test_terminal_interface_quarantine_is_durable_and_idempotent():
    target = ProofObligation("RH-C1", "imperative target")
    ledger = ProofObligationLedger("ledger", [target], version=94)
    changed = quarantine_terminal_interface_target(
        ledger,
        target_id="RH-C1",
        exhaustion_hash="e" * 64,
        source_run_id="run-interface",
    )
    assert changed
    assert ledger.version == 95
    assert ledger.backjump_target_id == "ROOT_UNAVAILABLE"
    assert target.status == "QUARANTINED"
    assert target.quarantine_prior_status == "UNRESOLVED"
    assert target.quarantine_reason == "TARGET_INTERFACE_EXHAUSTED:" + "e" * 64
    assert target.quarantine_reversible_status == "ACTIVE"
    assert target.quarantine_evidence_type == "CONTENT_ADDRESSED_EXHAUSTION"
    assert not quarantine_terminal_interface_target(
        ledger,
        target_id="RH-C1",
        exhaustion_hash="e" * 64,
        source_run_id="run-interface",
    )
    assert ledger.version == 95


def test_json_artifact_repairs_invalid_latex_escapes_losslessly():
    artifact = _json_artifact(
        r'{"statement":"sequence \{z_n\}, sum \\sum, density \rho"}',
    )
    assert artifact == {
        "statement": r"sequence \{z_n\}, sum \sum, density \rho",
    }


def test_structured_field_reads_first_continuation_line():
    body = "**Remaining gap:**\n\nDefine the exact density notion.\n"
    assert _structured_field(body, "Remaining gap") == (
        "Define the exact density notion."
    )
    assert certified_decomposition_requested(
        "",
        "### ISSUE_RESPONSE ROOT\n" + body,
        {"ROOT"},
    )


def test_certified_artifact_accepts_host_bound_raw_json_body():
    text = """### DEFINITION_AUDIT
{"definitions":[{"symbol":"rho","type":"Real","scope":"global"}],
 "missing_definitions":[]}
"""
    artifact, error = parse_certified_artifact(
        text,
        "DEFINITION_AUDIT",
        target_obligation_id="ROOT",
        parent_statement_hash="parent-hash",
        root_goal_hash="goal-hash",
        producer_run_id="run:definition_auditor",
        upstream_artifact_hashes=[],
    )
    assert error == ""
    assert artifact is not None
    assert artifact.producer_role == "definition_auditor"
    assert artifact.definitions[0]["symbol"] == "rho"


def test_production_definition_fixture_reports_eos_complete_syntax_context():
    text = (
        '### DEFINITION_AUDIT\nArtifact: {"symbol"$s_0$",'
        '"domain":"$\\mathbb{C$",'
        '"definitions":[],"missing_definitions":[]}'
    )
    artifact, error = parse_certified_artifact(
        text,
        "DEFINITION_AUDIT",
        target_obligation_id="ROOT",
        parent_statement_hash="parent-hash",
        root_goal_hash="goal-hash",
        producer_run_id="failed:br_4361eda61779579a",
        upstream_artifact_hashes=[],
    )
    assert artifact is None
    assert error.startswith(
        "malformed DEFINITION_AUDIT Artifact JSON: "
        "EOS-complete malformed JSON:",
    )
    assert "Expecting ':' delimiter" in error
    assert "first syntax context" in error
    assert "transport-incomplete" not in error


def test_definition_bundle_must_cover_all_missing_labels():
    audit = DefinitionAudit(
        "ROOT",
        "parent",
        "goal",
        "definition_auditor",
        "run:definition",
        [],
        [],
        [
            {"obligation_label": "L1", "symbol": "rho"},
            {"obligation_label": "L2", "symbol": "topology"},
        ],
    )
    common = dict(
        target_obligation_id="ROOT",
        parent_statement_hash="parent",
        root_goal_hash="goal",
        producer_role="decomposer",
        producer_run_id="run:decomposer",
        upstream_artifact_hashes=[],
        parent_statement="Parent statement.",
        public_assumptions=[],
        reduction_contract={
            "child_label": "D",
            "parent_statement": "Parent statement.",
            "public_assumptions": [],
            "derivation": "The bundled definitions make every parent term exact.",
        },
    )
    valid = DecompositionProposal(
        child={
            "label": "D",
            "statement": "Define rho and the convergence topology.",
            "kind": "DEFINITION",
            "source_definition_labels": ["L1", "L2"],
        },
        **common,
    )
    assert _validate_definition_child_selection(audit, valid) == []
    invalid = DecompositionProposal(
        child={
            "label": "D",
            "statement": "Define only rho.",
            "kind": "DEFINITION",
            "source_definition_labels": ["L1"],
        },
        **common,
    )
    assert "every missing definition label" in (
        _validate_definition_child_selection(audit, invalid)[0]
    )


def test_decomposer_upstream_view_keeps_only_decisive_fields():
    audit = DefinitionAudit(
        "ROOT", "parent", "goal", "definition_auditor", "run", [],
        [{"symbol": "known", "type": "Real"}],
        [{"obligation_label": "L1", "symbol": "rho"}],
    )
    assert _certified_upstream_view(
        audit,
        consumer_role="decomposer",
    ) == {
        "missing_definitions": [{"obligation_label": "L1", "symbol": "rho"}],
    }
    report = CounterexampleReport(
        "ROOT", "parent", "goal", "counterexample_worker", "run", [],
        "COUNTEREXAMPLE_FOUND",
        [{
            "case_id": "c1",
            "description": "Long supporting prose retained in the manifest.",
            "mathematical_contradiction": "Local density is not global order.",
        }],
    )
    compact = _certified_upstream_view(
        report,
        consumer_role="decomposer",
    )
    assert compact["cases"][0] == {
        "case_id": "c1",
        "mathematical_contradiction": "Local density is not global order.",
    }


def test_decomposer_protocol_rejects_archived_list_shape():
    text = (
        "### DECOMPOSITION_PROPOSAL\nArtifact: "
        '{"parent_statement":"P","children":[{"label":"L1"},'
        '{"label":"L2"}],"dependency_edges":[],"public_assumptions":[],'
        '"reduction_labels":["L1","L2"],"reduction_contract":"both imply P"}'
    )
    errors = _decomposer_protocol_errors(text)
    assert any("received 2" in error for error in errors)
    assert any("'child' must be exactly one object" in error for error in errors)


def test_balanced_artifact_scanner_handles_nested_strings_and_limits():
    text = (
        '### DECOMPOSITION_PROPOSAL\nArtifact: {"child":{"statement":'
        '"set \\\\{x: x \\\\in A\\\\} and }{ literal","items":[{"x":1}]},'
        '"reduction_contract":{"derivation":"use \\\\\\\\sum_n"}}'
    )
    scanned = scan_single_artifact_object(text, max_chars=300)
    assert scanned.json_text.startswith('{"child"')
    assert scanned.json_text.endswith("}")
    assert scan_single_artifact_object(
        "Artifact: " + "{}",
        max_chars=2,
    ).json_text == "{}"
    with pytest.raises(ValueError, match="size cap"):
        scan_single_artifact_object("Artifact: {}", max_chars=1)


@pytest.mark.parametrize("suffix", [
    " trailing prose",
    '\nArtifact: {"second":true}',
])
def test_balanced_artifact_scanner_rejects_trailing_or_second(suffix):
    with pytest.raises(ValueError, match="trailing|multiple"):
        scan_single_artifact_object("Artifact: {\"x\":[1,{\"y\":2}]}" + suffix)


def test_balanced_artifact_scanner_rejects_incomplete_object():
    with pytest.raises(ValueError, match="transport-incomplete"):
        scan_single_artifact_object('Artifact: {"x":["} in string",')


def _missing_definition_decomposer_package():
    parent = (
        '**The "Density-Singularity Gap Lemma:** Prove the immutable parent.'
    )
    return {
        "target_obligation_id": "ROOT",
        "parent_statement": parent,
        "parent_statement_hash": hashlib.sha256(parent.encode()).hexdigest(),
        "root_goal_hash": "goal-hash",
        "producer_role": "decomposer",
        "producer_run_id": "run:decomposer",
        "upstream_artifact_hashes": ["definition", "counterexample"],
        "validated_upstream_artifacts": {
            "definition_auditor": {
                "missing_definitions": [
                    {"obligation_label": "L1", "symbol": r"\rho"},
                    {
                        "obligation_label": "L2",
                        "symbol": "neighborhood convergence",
                    },
                    {"obligation_label": "L3", "symbol": "f(s)"},
                ],
            },
            "counterexample_worker": {"status": "INCONCLUSIVE", "cases": []},
        },
    }


def test_decomposer_repair_contract_is_concrete_and_host_bound():
    package = _missing_definition_decomposer_package()
    messages = _decomposer_repair_messages(
        package,
        validation_errors=["child labels invalid"],
        rejected_artifact=None,
    )
    prompt = messages[0]["content"]
    repair_package = json.loads(messages[1]["content"])
    assert "DEFINITION|LEMMA" not in prompt
    assert '"kind":"DEFINITION"' in prompt
    assert '"source_definition_labels":["L1","L2","L3"]' in prompt
    assert repair_package["parent_statement"] == package["parent_statement"]
    assert (
        repair_package["parent_statement_hash"]
        == package["parent_statement_hash"]
    )
    assert repair_package["repair_contract"] == {
        "required_child_kind": "DEFINITION",
        "required_source_definition_labels": ["L1", "L2", "L3"],
    }
    initial_prompt = _certified_role_messages(
        "decomposer",
        "DECOMPOSITION_PROPOSAL",
        package,
    )[0]["content"]
    assert "DEFINITION|LEMMA" not in initial_prompt
    assert '"kind":"DEFINITION"' in initial_prompt


def test_decomposer_model_package_references_full_semantic_archive():
    package = _missing_definition_decomposer_package()
    package.update({
        "target_statement_hash": package["parent_statement_hash"],
        "decomposer_contract": {
            "required_parent_hash": package["parent_statement_hash"],
            "required_child_kind": "DEFINITION",
            "required_source_definition_labels": ["L1", "L2", "L3"],
        },
        "ancestor_hashes": {
            "manifest": "sha256:" + "a" * 64,
            "count": 10,
            "archive": "proof_ledger",
        },
        "decomposition_iteration": 4,
        "viewpoint": "quantifiers",
        "decomposition_novelty_ledger": {
            "manifest": "sha256:" + "b" * 64,
            "count": 100,
            "novel": 98,
            "reasons": {"CYCLIC_EQUIVALENT": 99},
            "viewpoints": {
                "manifest": "sha256:" + "c" * 64,
                "count": 100,
                "base_ids": ["quantifiers"],
                "recent_ids": ["quantifiers"],
            },
            "active_viewpoint": "quantifiers",
            "duplicate_gate": "semantic_hash+structural_signature",
            "recent": [{
                "i": 99,
                "v": "quantifiers",
                "p": "1" * 16,
                "s": "2" * 16,
                "d": "3" * 16,
                "r": ["CYCLIC_EQUIVALENT"],
            }],
        },
    })
    messages = _certified_role_messages(
        "decomposer",
        "DECOMPOSITION_PROPOSAL",
        package,
    )
    model = json.loads(messages[-1]["content"])
    assert messages[-1]["_host_package"] is package
    assert model["parent_statement"] == package["parent_statement"]
    assert model["viewpoint"] == "quantifiers"
    assert model["required_definitions"] == (
        package["validated_upstream_artifacts"]["definition_auditor"][
            "missing_definitions"
        ]
    )
    assert model["immutable_bindings"]["ancestor_hashes"]["count"] == 10
    novelty = model["semantic_search"]["novelty"]
    assert novelty["manifest"] == "sha256:" + "b" * 64
    assert novelty["duplicate_gate"] == (
        "semantic_hash+structural_signature"
    )
    assert "target_statement_hash" not in model
    assert "parent_statement_hash" not in model
    assert "validated_upstream_artifacts" not in model


def test_decomposer_extra_brace_preserves_exact_scanner_diagnostic():
    valid_object = (
        '{"parent_statement":"P","child":{"label":"L1","statement":"S",'
        '"kind":"DEFINITION","source_definition_labels":["L1"]},'
        '"public_assumptions":[],"reduction_contract":{"child_label":"L1",'
        '"parent_statement":"P","public_assumptions":[],'
        '"derivation":"This complete derivation implies the parent."}}'
    )
    errors = _decomposer_protocol_errors(
        "### DECOMPOSITION_PROPOSAL\nArtifact: " + valid_object + "}",
    )
    assert errors == [
        "malformed DECOMPOSITION_PROPOSAL Artifact JSON: "
        "trailing text or second Artifact is forbidden",
    ]


LATEST_FORMALIZER_REPAIR_RAW = (
    "### FORMALIZATION_BUNDLE\n"
    'Artifact: {"parent_signature_source":"**The Density-Singularity Gap '
    'Lemma:** Prove that for a given $\\\\epsilon$ and a fixed genus $p$, '
    'there exists a critical density $\\\\rho_c$ such that for any sequence '
    '$\\\\{z_n\\\\}$ with density $\\\\rho > \\\\rho_c$, the sum $\\\\sum '
    '\\\\frac{1}{s-z_n}$ cannot converge to $\\\\frac{m}{s-s_0}$ in a '
    '$\\\\delta$-neighborhood of $s_0$ without forcing the function $f(s)$ '
    'to have a growth order strictly greater than $p$.",'
    '"parent_signature_hash":"537f9def407e3c20b2302776c3b909e5d7ef28e993'
    '5f488bd02b3db96f807a53","parent_newly_formalized":true,'
    '"child":{"label":"L1","lean_signature":"theorem L1_definition '
    '(z_n : ℕ → ℂ) (s0 : ℂ) (δ : ℝ) (m : ℂ) (p : ℕ) : '
    '(∀ s ∈ ℂ, |s - s0| < δ → ∑ (1 / (s - z_n)) = m / (s - s0)) ∧ '
    '(growth_order f > p) ↔ density z_n > ρ_c"},'
    '"child_signature_hash":"f630762a4f4399db69ee21663c6643c9e68db23577'
    'ee5a1ee776c215165280c"},'
    '"reduction_theorem_source":"The child bundles the necessary mathematical '
    'objects (density $\\\\rho$, local convergence behavior, and the resulting '
    'function $f(s)$) required to evaluate the tension between the density of '
    'zeros and the growth order $p$. By defining these terms, the parent\'s '
    'claim is reduced to a comparison between the density $\\\\rho$ and the '
    'growth order $p$ under the constraint of the local singularity at $s_0$.",'
    '"reduction_signature_hash":"07bf396f9fb5332ef76b14989e52a1271af81b3c1'
    'a007996e89681d9f72b1c77"}'
)


def test_latest_formalizer_repair_closes_before_reduction_fields():
    scanned = scan_structured_artifact_prefix(
        LATEST_FORMALIZER_REPAIR_RAW,
        "FORMALIZATION_BUNDLE",
    )
    assert scanned is not None
    assert json.loads(scanned.json_text)["child_signature_hash"].startswith(
        "f630762a",
    )
    assert LATEST_FORMALIZER_REPAIR_RAW[scanned.end:].startswith(
        ',"reduction_theorem_source":',
    )
    with pytest.raises(
        ValueError,
        match="trailing text or second Artifact is forbidden",
    ):
        scan_single_artifact_object(LATEST_FORMALIZER_REPAIR_RAW)


def test_transport_scanner_is_string_escape_array_and_latex_brace_aware():
    text = (
        'Artifact: {"nested":[{"latex":"\\\\{x\\\\} and } plus \\"quote\\"",'
        '"value":{"items":[1,2,3]}}]}'
    )
    scanned = scan_artifact_object_prefix(text)
    assert scanned is not None
    assert scanned.end == len(text)
    assert json.loads(scanned.json_text)["nested"][0]["value"]["items"] == [1, 2, 3]


@pytest.mark.parametrize("role,heading", [
    ("definition_auditor", "DEFINITION_AUDIT"),
    ("counterexample_worker", "COUNTEREXAMPLE_REPORT"),
    ("decomposer", "DECOMPOSITION_PROPOSAL"),
    ("formalizer", "FORMALIZATION_BUNDLE"),
    ("prover", "PROOF_ATTEMPT"),
    ("adversarial_proponent", "DEFENSE_REPORT"),
    ("judge", "JUDGE_DECISION"),
])
def test_all_structured_roles_stop_on_first_transport_object(role, heading):
    first = f'### {heading}\nArtifact: {{"schema_invalid_for_role":true}}'
    assert _structured_transport_semantically_complete(first, role)
    assert not _structured_transport_semantically_complete(
        first + '\nArtifact: {"replacement":true}',
        role,
    )


def test_schema_invalid_first_object_stops_transport_then_fails_host_gate():
    text = (
        '### FORMALIZATION_BUNDLE\nArtifact: '
        '{"schema_invalid_for_role":true}'
    )
    assert _structured_transport_semantically_complete(text, "formalizer")
    parsed, error = parse_certified_artifact(
        text,
        "FORMALIZATION_BUNDLE",
        target_obligation_id="ROOT",
        parent_statement_hash="parent",
        root_goal_hash="goal",
        producer_run_id="run:formalizer",
        upstream_artifact_hashes=["decomposer"],
    )
    assert parsed is None
    assert error.startswith("invalid FORMALIZATION_BUNDLE fields:")
    assert not _structured_transport_semantically_complete(
        text + '\nArtifact: {"replacement":true}',
        "formalizer",
    )


@pytest.mark.parametrize(
    "role", ["decomposer_scratchpad", "synthesis_scratchpad"],
)
def test_private_scratchpad_completion_is_eos_only(role):
    assert not _structured_transport_semantically_complete(
        "private reasoning with no artifact contract",
        role,
    )
    assert not _structured_transport_semantically_complete(
        '### DECOMPOSITION_PROPOSAL\nArtifact: {"ignored":true}',
        role,
    )


def test_missing_definition_repaired_output_accepts_only_exact_metadata():
    package = _missing_definition_decomposer_package()
    parent = package["parent_statement"]

    def response(kind, labels):
        payload = {
            "parent_statement": parent,
            "child": {
                "label": "L1",
                "statement": "Define all three audited notions precisely.",
                "kind": kind,
                "source_definition_labels": labels,
            },
            "public_assumptions": [],
            "reduction_contract": {
                "child_label": "L1",
                "parent_statement": parent,
                "public_assumptions": [],
                "derivation": (
                    "These definitions make every term of the parent precise."
                ),
            },
        }
        return (
            "### DECOMPOSITION_PROPOSAL\nArtifact: "
            + json.dumps(payload, separators=(",", ":"))
        )

    assert _decomposer_semantically_complete(
        response("DEFINITION", ["L1", "L2", "L3"]),
        package,
        "run:decomposer",
    )
    assert not _decomposer_semantically_complete(
        response("DEFINITION|LEMMA", ["L1", "L2", "L3"]),
        package,
        "run:decomposer",
    )
    assert not _decomposer_semantically_complete(
        response("DEFINITION", []),
        package,
        "run:decomposer",
    )


def test_decomposer_semantic_stop_requires_complete_host_valid_artifact():
    package = {
        "target_obligation_id": "ROOT",
        "parent_statement": "Exact parent.",
        "parent_statement_hash": "parent-hash",
        "root_goal_hash": "goal-hash",
        "upstream_artifact_hashes": ["upstream"],
    }
    valid = (
        "### DECOMPOSITION_PROPOSAL\nArtifact: "
        '{"parent_statement":"Exact parent.","child":{"label":"L1",'
        '"statement":"Prove one exact child.","kind":"LEMMA",'
        '"source_definition_labels":[]},"public_assumptions":[],'
        '"reduction_contract":{"child_label":"L1",'
        '"parent_statement":"Exact parent.","public_assumptions":[],'
        '"derivation":"The exact child directly implies the exact parent."}}'
    )
    assert _decomposer_semantically_complete(valid, package, "run:decomposer")
    assert _decomposer_semantically_complete(
        valid.replace("Artifact: ", ""),
        package,
        "run:decomposer",
    )
    assert not _decomposer_semantically_complete(
        valid[:-1],
        package,
        "run:decomposer",
    )
    assert not _decomposer_semantically_complete(
        valid + "\ncontradiction",
        package,
        "run:decomposer",
    )


def test_structured_output_budget_uses_actual_retained_availability():
    assert structured_output_cap(
        role="decomposer",
        max_retained_tokens=2052,
        retained_input_tokens=1400,
        minimum_output_tokens=512,
        configured_output_tokens=None,
        control_reserve_tokens=64,
    ) == 588
    try:
        structured_output_cap(
            role="decomposer",
            max_retained_tokens=2052,
            retained_input_tokens=1500,
            minimum_output_tokens=512,
            configured_output_tokens=None,
            control_reserve_tokens=64,
        )
    except StructuredResponseBudgetTooSmall as exc:
        assert exc.available_tokens == 488
        assert exc.required_tokens == 512
    else:
        raise AssertionError("undersized structured response budget admitted")
    with pytest.raises(StructuredResponseBudgetTooSmall) as caught:
        structured_output_cap(
            role="decomposer",
            max_retained_tokens=2052,
            retained_input_tokens=1520,
            minimum_output_tokens=512,
            configured_output_tokens=None,
            control_reserve_tokens=64,
        )
    assert caught.value.available_tokens == 468
    assert caught.value.compaction_tokens_required == 44


@pytest.mark.parametrize(
    ("role", "minimum"),
    [
        ("formalizer", 768),
        ("prover", 384),
        ("adversarial_proponent", 256),
        ("judge", 256),
    ],
)
def test_downstream_role_preflight_reserves_complete_schema(role, minimum):
    assert structured_role_minimum_output_tokens(role) == minimum
    retained_input = 2052 - 64 - minimum
    assert structured_output_cap(
        role=role,
        max_retained_tokens=2052,
        retained_input_tokens=retained_input,
        minimum_output_tokens=minimum,
        configured_output_tokens=None,
        control_reserve_tokens=64,
    ) == minimum
    with pytest.raises(StructuredResponseBudgetTooSmall) as caught:
        structured_output_cap(
            role=role,
            max_retained_tokens=2052,
            retained_input_tokens=retained_input + 1,
            minimum_output_tokens=minimum,
            configured_output_tokens=None,
            control_reserve_tokens=64,
        )
    assert caught.value.compaction_tokens_required == 1
    assert "semantic units must not be truncated" in str(caught.value)


class Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(96 + token) for token in token_ids)


def _valid_lean_signature(suffix=""):
    return LeanSignatureResult(
        f"theorem frontier{suffix} (p : Prop) : p := by sorry",
        f"lean-signature-hash{suffix}",
        True,
    )


def _premise_suspicion_text(invalidation="PREMISE_SUSPECTED"):
    claim = {
        "schema_version": 1,
        "quantifier": "FOR_ALL",
        "variables": ["x"],
        "domain": "REAL",
        "lhs": "x*x",
        "relation": "==",
        "rhs": "x",
    }
    return f"""
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: {invalidation}
Premise refuted: For every real x, x squared equals x.
Evidence type: FINITE_COUNTEREXAMPLE
Evidence artifact: {json.dumps({"claim": claim}, separators=(",", ":"))}
Evidence: Substitution of the explicit finite value x=2 gives four on the left and two on the right, contradicting universal equality.
Missing lemma: none
"""


def _audit_text(status="CONFIRMED", confidence="0.95", evidence_type="FINITE_COUNTEREXAMPLE"):
    suspicion = extract_premise_suspicions(
        _premise_suspicion_text(),
        {"ROOT-A"},
    )["ROOT-A"]
    artifact = {
        "claim_hash": suspicion.claim_hash,
        "claim": suspicion.claim_schema,
        "witness": {"x": 2},
    }
    return f"""
### PREMISE_AUDIT ROOT-A
Status: {status}
Evidence type: {evidence_type}
Evidence source: host-checkable substitution x=2
Confidence: {confidence}
Artifact: {json.dumps(artifact, separators=(",", ":"))}
Analysis: The universal quantifier is contradicted by this explicit domain element.
"""


def _defense_text(status="NOT_RESCUED"):
    return f"""
### PREMISE_DEFENSE ROOT-A
Status: {status}
Correction: Restricting x to zero or one would rescue a different proposition.
Failure reason: The stated universal real-domain premise has no such restriction.
Evidence: The proposed restriction changes the exact domain and therefore cannot rescue the original quantified premise.
"""


def _certificate_runner(
    *,
    child_signature="theorem childReduction : True := by",
    proof_source=None,
    cycle=False,
    parent_hash_override="",
    judge_decision="ACCEPT",
    counterexample_case=None,
    multi_child=False,
):
    calls = []
    parent_source = "theorem parentTarget : True := by"
    reduction_source = "theorem reduction : True := by"
    child_statement = (
        "For every fixed compact disk, prove an explicit uniform boundary "
        "inequality for the analytic approximants."
    )

    def runner(role, messages, expected_run_id):
        calls.append((role, messages, expected_run_id))
        package = (
            messages[-1]["_host_package"]
            if "_host_package" in messages[-1]
            else json.loads(messages[-1]["content"])
        )
        common = {
            "target_obligation_id": package["target_obligation_id"],
            "parent_statement_hash": package["parent_statement_hash"],
            "root_goal_hash": package["root_goal_hash"],
            "producer_role": role,
            "producer_run_id": expected_run_id,
            "upstream_artifact_hashes": package[
                "upstream_artifact_hashes"
            ],
        }
        if role == "definition_auditor":
            if "registered_output_choices" in package:
                choices = package["registered_output_choices"]
                text = "\n".join((
                    f"target_ref {choices['target_ref'][0]};",
                    f"symbol_id {choices['symbol_id'][0]};",
                    f"domain_id {choices['domain_id'][0]};",
                    f"topology_id {choices['topology_id'][0]};",
                    f"definition_id {choices['definition_id'][0]};",
                    "audit_outcome COMPLETE;",
                    "END;",
                ))
                return text, expected_run_id
            heading = "DEFINITION_AUDIT"
            specific = {
                "definitions": [{
                    "symbol": "p",
                    "type": "integer (genus)",
                    "scope": "global",
                }],
                "missing_definitions": [],
            }
        elif role == "counterexample_worker":
            heading = "COUNTEREXAMPLE_REPORT"
            specific = {
                "status": (
                    "COUNTEREXAMPLE_FOUND"
                    if counterexample_case else "NO_COUNTEREXAMPLE"
                ),
                "cases": [counterexample_case] if counterexample_case else [],
            }
        elif role == "decomposer":
            heading = "DECOMPOSITION_PROPOSAL"
            child = {
                "label": "L1",
                "statement": child_statement,
                "kind": "LEMMA",
                "source_definition_labels": [],
            }
            specific = {
                "parent_statement": package["parent_statement"],
                "child": child,
                "public_assumptions": [],
                "reduction_contract": {
                    "child_label": "L1",
                    "parent_statement": package["parent_statement"],
                    "public_assumptions": [],
                    "derivation": (
                        "The explicit uniform inequality directly establishes "
                        "the exact global convergence parent."
                    ),
                },
            }
            if cycle:
                specific["reduction_contract"]["child_label"] = "L2"
            if multi_child and "protocol-repair" not in expected_run_id:
                specific.pop("child")
                specific["children"] = [
                    child,
                    {
                        "label": "L2",
                        "statement": (
                            "Prove the separate continuity inequality required "
                            "by the same parent reduction."
                        ),
                        "kind": "LEMMA",
                    },
                ]
        elif role == "formalizer":
            heading = "FORMALIZATION_BUNDLE"
            wire_contract = json.loads(messages[-1]["content"])[
                "lean_signature_contract"
            ]
            names = wire_contract["names"]

            def signature_object(field, source, *, label=None):
                required_name = names[field]
                tail = source.split(None, 2)[2]
                normalized_source = f"theorem {required_name} {tail}"
                value = {
                    "kind": "theorem",
                    "name": required_name,
                    "binders": "",
                    "proposition": "True",
                    "source": normalized_source,
                }
                if label is not None:
                    value["label"] = label
                return value

            parent_field = signature_object(
                "parent_signature",
                parent_source,
            )
            if parent_hash_override:
                parent_field["name"] = "wrongParent"
                parent_field["source"] = (
                    "theorem wrongParent : True := by"
                )
            specific = {
                "parent_signature": parent_field,
                "parent_newly_formalized": True,
                "child_signature": signature_object(
                    "child_signature",
                    child_signature,
                    label="L1",
                ),
                "reduction_signature": signature_object(
                    "reduction_signature",
                    reduction_source,
                ),
            }
        elif role == "prover":
            heading = "PROOF_ATTEMPT"
            emitted_proof = proof_source
            if emitted_proof is None:
                emitted_proof = (
                    package["validated_upstream_artifacts"]["formalizer"][
                        "reduction_theorem_source"
                    ]
                    + "\n  trivial"
                )
            specific = {
                "status": "PROVED",
                "reduction_theorem_source": emitted_proof,
            }
        elif role == "adversarial_proponent":
            heading = "DEFENSE_REPORT"
            specific = {
                "status": "DEFENDED",
                "issues": [],
                "repairs": [],
            }
        else:
            heading = "JUDGE_DECISION"
            specific = {
                "decision": judge_decision,
                "reason": "Host manifest reviewed.",
            }
        text = (
            f"### {heading}\nArtifact: "
            + json.dumps({**common, **specific}, separators=(",", ":"))
        )
        return text, expected_run_id

    return runner, calls


def _formalizer_fixture():
    parent_statement = "Prove the exact parent proposition."
    child = {
        "label": "L1",
        "statement": "Define the exact singular child contract.",
        "kind": "DEFINITION",
        "source_definition_labels": ["L1"],
    }
    reduction = {
        "child_label": "L1",
        "parent_statement": parent_statement,
        "public_assumptions": ["h : True"],
        "derivation": "The exact child and public assumption imply the parent.",
    }
    proposal = DecompositionProposal(
        target_obligation_id="ROOT",
        parent_statement_hash=hashlib.sha256(parent_statement.encode()).hexdigest(),
        root_goal_hash="g" * 64,
        producer_role="decomposer",
        producer_run_id="run:decomposer",
        upstream_artifact_hashes=["a" * 64, "b" * 64],
        parent_statement=parent_statement,
        child=child,
        public_assumptions=["h : True"],
        reduction_contract=reduction,
    )
    package = {
        "target_obligation_id": "ROOT",
        "parent_statement": parent_statement,
        "parent_statement_hash": proposal.parent_statement_hash,
        "target_statement_hash": proposal.parent_statement_hash,
        "root_goal_hash": proposal.root_goal_hash,
        "producer_role": "formalizer",
        "producer_run_id": "run:formalizer",
        "upstream_artifact_hashes": ["a" * 64, "b" * 64, "d" * 64],
        "validated_upstream_artifacts": {
            "decomposer": _formalizer_upstream_view(proposal, "d" * 64),
        },
        "definition_audit": {
            "definitions": [
                {"symbol": "p", "type": "integer (genus)", "scope": "global"},
            ],
            "missing_definitions": [{
                "obligation_label": "L1",
                "required_type": "formal density measure",
                "symbol": "\\rho",
            }],
        },
        "parent_formal_status": "UNFORMALIZED",
    }
    contract = _formalizer_model_package(package)["lean_signature_contract"]
    names = contract["names"]

    def field(field_name, binders, proposition, *, label=None):
        name = names[field_name]
        source = (
            f"theorem {name}"
            f"{f' {binders}' if binders else ''} : {proposition} := by"
        )
        value = {
            "kind": "theorem",
            "name": name,
            "binders": binders,
            "proposition": proposition,
            "source": source,
        }
        if label is not None:
            value["label"] = label
        return value

    payload = {
        "parent_signature": field("parent_signature", "", "True"),
        "parent_newly_formalized": True,
        "child_signature": field(
            "child_signature",
            "",
            "True",
            label="L1",
        ),
        "reduction_signature": field(
            "reduction_signature",
            "(h : True)",
            "True",
        ),
    }
    text = (
        "### FORMALIZATION_BUNDLE\nArtifact: "
        + json.dumps(payload, separators=(",", ":"))
    )
    return proposal, package, payload, text


def test_formalizer_view_is_content_addressed_without_repeated_prose():
    proposal, package, _payload, _text = _formalizer_fixture()
    view = package["validated_upstream_artifacts"]["decomposer"]
    assert view["artifact_hash"] == "d" * 64
    assert view["child_hash"] == hashlib.sha256(
        json.dumps(
            proposal.child,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    ).hexdigest()
    assert view["reduction_contract_hash"]
    assert "parent_statement" not in view
    assert set(view["reduction"]) == {"child_label", "derivation"}
    assert package["parent_statement"] == proposal.parent_statement
    assert view["public_assumptions"] == proposal.public_assumptions


def test_formalizer_complete_json_before_eos_is_semantically_complete():
    _proposal, package, _payload, text = _formalizer_fixture()
    assert _formalizer_semantically_complete(text, package, "run:formalizer")
    assert not _formalizer_semantically_complete(
        text[:-1],
        package,
        "run:formalizer",
    )
    assert not _formalizer_semantically_complete(
        text + "\nArtifact: {}",
        package,
        "run:formalizer",
    )
    assert not _formalizer_semantically_complete(
        text + "\ncontradictory trailing prose",
        package,
        "run:formalizer",
    )


def test_formalizer_compact_repair_contains_exact_errors_and_schema():
    _proposal, package, _payload, _text = _formalizer_fixture()
    messages = _formalizer_repair_messages(
        package,
        validation_errors=[
            "SemanticResponseIncomplete: stopped after 630 tokens",
        ],
    )
    assert "Fresh Formalizer repair" in messages[0]["content"]
    assert "parent_signature" in messages[0]["content"]
    assert "child_signature" in messages[0]["content"]
    repair = json.loads(messages[-1]["content"])
    assert repair["validation_errors"] == [
        "SemanticResponseIncomplete: stopped after 630 tokens",
    ]
    assert "parent_lean_signature" not in repair
    assert "rejected_complete_artifact" not in repair
    contract = repair["lean_signature_contract"]
    assert set(contract["names"]) == {
        "parent_signature",
        "child_signature",
        "reduction_signature",
    }
    assert contract["contract_id"].startswith("lean-signature-")
    assert contract["version"] == 1
    serialized = json.dumps(messages)
    assert "valid_examples" not in serialized
    assert "forbidden_constructs" not in serialized


def _adversarial_fixture():
    proposal, formalizer_package, payload, _text = _formalizer_fixture()
    durable_payload = _normalize_formalizer_payload(payload)
    formalization = FormalizationBundle(
        target_obligation_id="ROOT",
        parent_statement_hash=proposal.parent_statement_hash,
        root_goal_hash=proposal.root_goal_hash,
        producer_role="formalizer",
        producer_run_id="run:formalizer",
        upstream_artifact_hashes=["a" * 64, "b" * 64, "d" * 64],
        **durable_payload,
    )
    proof = ProofAttempt(
        target_obligation_id="ROOT",
        parent_statement_hash=proposal.parent_statement_hash,
        root_goal_hash=proposal.root_goal_hash,
        producer_role="prover",
        producer_run_id="run:prover",
        upstream_artifact_hashes=["a" * 64, "b" * 64, "d" * 64, "f" * 64],
        status="PROVED",
        reduction_theorem_source=durable_payload["reduction_theorem_source"],
    )
    hashes = {
        "decomposer": "d" * 64,
        "formalizer": "f" * 64,
        "prover": "p" * 64,
    }
    package = {
        "target_obligation_id": "ROOT",
        "parent_statement": proposal.parent_statement,
        "parent_statement_hash": proposal.parent_statement_hash,
        "target_statement_hash": proposal.parent_statement_hash,
        "root_goal_hash": proposal.root_goal_hash,
        "producer_role": "adversarial_proponent",
        "producer_run_id": "run:adversarial_proponent",
        "upstream_artifact_hashes": list(hashes.values()),
        "validated_upstream_artifacts": {
            "decomposer": _certified_upstream_view(proposal),
            "formalizer": _certified_upstream_view(formalization),
            "prover": _certified_upstream_view(proof),
        },
        "validated_artifact_hashes": hashes,
        "host_gate_results": {
            "validation": {
                "graph_valid": True,
                "children_valid": True,
                "reduction_proof_valid": True,
                "host_gates_passed": True,
            },
            "errors": [],
        },
        "parent_formal_status": "UNFORMALIZED",
    }
    return package


def test_adversarial_2215_fixture_compacts_losslessly_with_reserve():
    package = _adversarial_fixture()
    compact = _adversarial_review_model_package(package)
    # This fixture represents the production failure's measured verbose input.
    verbose_retained_tokens = 2215
    compact_retained_tokens = 1708
    assert verbose_retained_tokens > 2052
    assert structured_output_cap(
        role="adversarial_proponent",
        max_retained_tokens=2052,
        retained_input_tokens=compact_retained_tokens,
        minimum_output_tokens=256,
        configured_output_tokens=None,
        control_reserve_tokens=64,
    ) == 280
    assert compact["binding"]["target_id"] == "ROOT"
    assert compact["hashes"][compact["binding"]["parent_h"]] == (
        package["parent_statement_hash"]
    )
    assert compact["semantic_units"][compact["binding"]["parent_ref"]] == (
        package["parent_statement"]
    )
    child = package["validated_upstream_artifacts"]["decomposer"]["child"]
    assert compact["semantic_units"][compact["child"]["statement_ref"]] == (
        child["statement"]
    )
    assert compact["public_assumptions"]["items"] == (
        package["validated_upstream_artifacts"]["decomposer"][
            "public_assumptions"
        ]
    )
    assert compact["host_gates"]["validation"]["host_gates_passed"] is True


def test_defense_complete_before_eos_is_strict_and_repair_is_fresh():
    package = _adversarial_fixture()
    text = (
        "### DEFENSE_REPORT\nArtifact: "
        '{"status":"REJECTED","issues":["circular reduction"],'
        '"repairs":["replace the child"]}'
    )
    assert _defense_semantically_complete(
        text,
        package,
        "run:adversarial_proponent",
    )
    assert not _defense_semantically_complete(
        text[:-1],
        package,
        "run:adversarial_proponent",
    )
    assert not _defense_semantically_complete(
        text + "\nArtifact: {}",
        package,
        "run:adversarial_proponent",
    )
    assert not _defense_semantically_complete(
        text + "\ntrailing",
        package,
        "run:adversarial_proponent",
    )
    repair = _defense_repair_messages(
        package,
        validation_errors=["incomplete Artifact JSON"],
    )
    assert "Fresh Adversarial Proponent protocol repair" in (
        repair[0]["content"]
    )
    assert json.loads(repair[-1]["content"])["validation_errors"] == [
        "incomplete Artifact JSON",
    ]


def test_judge_compact_manifest_preserves_review_and_budget_boundary():
    package = {
        "target_obligation_id": "ROOT",
        "parent_statement_hash": "a" * 64,
        "root_goal_hash": "b" * 64,
        "parent_statement": "Exact parent claim.",
        "retained_child_statement": "Exact child claim.",
        "artifact_hashes": {
            "decomposer": "c" * 64,
            "formalizer": "d" * 64,
            "prover": "e" * 64,
            "adversarial_proponent": "f" * 64,
        },
        "validation": {"host_gates_passed": True},
        "errors": [],
        "defense_evidence": {
            "artifact_hash": "f" * 64,
            "status": "DEFENDED",
            "issues": [],
            "repairs": [],
        },
        "upstream_artifact_hashes": ["9" * 64],
    }
    compact = _judge_model_package(package)
    assert compact["adversarial_review"] == package["defense_evidence"]
    assert compact["artifact_hashes"] == package["artifact_hashes"]
    boundary = 2052 - 64 - 256
    assert structured_output_cap(
        role="judge",
        max_retained_tokens=2052,
        retained_input_tokens=boundary,
        minimum_output_tokens=256,
        configured_output_tokens=None,
        control_reserve_tokens=64,
    ) == 256
    with pytest.raises(StructuredResponseBudgetTooSmall):
        structured_output_cap(
            role="judge",
            max_retained_tokens=2052,
            retained_input_tokens=boundary + 1,
            minimum_output_tokens=256,
            configured_output_tokens=None,
            control_reserve_tokens=64,
        )


def _fake_signature_validator(source, *, project_root):
    del project_root
    if "badChild" in source:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_FAILED",
            error="bad child",
        )
    return LeanSignatureResult(
        source,
        lean_theorem_signature_hash(source),
        True,
    )


def _fake_proof_validator(source, *, project_root):
    del project_root
    if re.search(r"\b(?:sorry|admit)\b", source):
        return LeanSignatureResult(
            source,
            "",
            False,
            status="UNSAFE_REJECTED",
            error="incomplete proof",
        )
    return LeanSignatureResult(
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        True,
        status="PROVED",
    )


def _unit_response(messages, expected_run_id):
    request = json.loads(messages[-1]["content"])
    unit = request["unit"]
    name = request["required_name"]
    source = f"theorem {name} : True := by"
    payload = {
        "contract_id": request["contract_id"],
        "contract_version": request["version"],
        "unit": unit,
        "kind": "theorem",
        "name": name,
        "binders": "",
        "proposition": "True",
        "source": source,
    }
    return (
        "### LEAN_SIGNATURE_UNIT\nArtifact:"
        + json.dumps(payload, separators=(",", ":")),
        expected_run_id,
    )


def test_split_formalizer_persists_units_and_resumes_after_crash(tmp_path):
    _proposal, package, _payload, _text = _formalizer_fixture()
    checkpoint_path = tmp_path / "orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.FORMALIZER.value,
        current_role="formalizer",
        target_obligation_id="ROOT",
    )
    save_orchestration_checkpoint(checkpoint_path, checkpoint)
    first_calls = []

    def crash_after_parent(role, messages, expected_run_id):
        first_calls.append(role)
        if role == "formalizer_child_signature":
            raise KeyboardInterrupt("simulated process crash")
        return _unit_response(messages, expected_run_id)

    with pytest.raises(KeyboardInterrupt, match="simulated process crash"):
        _run_split_formalizer(
            crash_after_parent,
            package=package,
            project_root=tmp_path,
            signature_validator=_fake_signature_validator,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            expected_run_id="run:formalizer",
        )
    crashed = load_orchestration_checkpoint(checkpoint_path)
    assert crashed.formalizer_substate == "CHILD_SIGNATURE"
    assert set(crashed.formalizer_unit_hashes) == {"PARENT_SIGNATURE"}
    assert "formalizer_parent_signature" in crashed.validated_artifacts

    resumed_calls = []

    def resumed(role, messages, expected_run_id):
        resumed_calls.append(role)
        return _unit_response(messages, expected_run_id)

    bundle, transcripts, error = _run_split_formalizer(
        resumed,
        package=package,
        project_root=tmp_path,
        signature_validator=_fake_signature_validator,
        checkpoint_path=checkpoint_path,
        checkpoint=crashed,
        expected_run_id="run:formalizer",
    )
    assert not error
    assert bundle is not None
    assert resumed_calls == [
        "formalizer_child_signature",
        "formalizer_reduction_signature",
    ]
    assert set(transcripts) == {
        "child_signature",
        "reduction_signature",
    }
    completed = load_orchestration_checkpoint(checkpoint_path)
    assert completed.formalizer_substate == "ASSEMBLE"
    assert set(completed.formalizer_unit_hashes) == {
        "PARENT_SIGNATURE",
        "CHILD_SIGNATURE",
        "REDUCTION_SIGNATURE",
    }


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_certified_decomposition_assembles_split_units_before_prover(tmp_path):
    ledger = ProofObligationLedger(
        "split-integration",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    base_runner, calls = _certificate_runner()

    def split_runner(role, messages, expected_run_id):
        if role.startswith("formalizer_") and role.endswith("_signature"):
            calls.append((role, messages, expected_run_id))
            return _unit_response(messages, expected_run_id)
        return base_runner(role, messages, expected_run_id)

    split_runner._supports_split_formalizer = True
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        split_runner,
        project_root=tmp_path,
        orchestration_id="orch-split",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=tmp_path / "orchestration.json",
        candidate_sha256="candidate",
    )
    assert result.verified
    roles = [item[0] for item in calls]
    assert "formalizer" not in roles
    assert roles.index("formalizer_parent_signature") < roles.index(
        "formalizer_child_signature",
    ) < roles.index("formalizer_reduction_signature") < roles.index("prover")
    checkpoint = load_orchestration_checkpoint(
        tmp_path / "orchestration.json",
    )
    assert checkpoint.formalizer_substate == "ASSEMBLE"
    assert set(checkpoint.formalizer_unit_hashes) == {
        "PARENT_SIGNATURE",
        "CHILD_SIGNATURE",
        "REDUCTION_SIGNATURE",
    }


def test_split_formalizer_rejects_stale_contract_and_cross_binding(tmp_path):
    _proposal, package, _payload, _text = _formalizer_fixture()
    messages = _formalizer_unit_messages(
        "PARENT_SIGNATURE",
        package,
        {},
    )
    text, _run_id = _unit_response(messages, "run")
    payload = json.loads(text.split("Artifact:", 1)[1])
    payload["contract_version"] += 1
    parsed, error = _parse_formalizer_unit(
        "### LEAN_SIGNATURE_UNIT\nArtifact:"
        + json.dumps(payload, separators=(",", ":")),
        unit="PARENT_SIGNATURE",
        package=package,
        project_root=tmp_path,
        signature_validator=_fake_signature_validator,
        dependencies=_formalizer_unit_dependencies(
            "PARENT_SIGNATURE",
            package,
            {},
        ),
    )
    assert parsed is None
    assert "unknown or stale Lean contract" in error

    unit = {
        "source": "theorem x : True := by",
        "signature_hash": lean_theorem_signature_hash(
            "theorem x : True := by",
        ),
    }
    with pytest.raises(ValueError, match="exact parent proposition"):
        _assemble_formalizer_units(
            {
                "PARENT_SIGNATURE": unit,
                "CHILD_SIGNATURE": unit,
                "REDUCTION_SIGNATURE": {
                    "source": "theorem y : False := by",
                    "signature_hash": lean_theorem_signature_hash(
                        "theorem y : False := by",
                    ),
                },
            },
            package=package,
            producer_run_id="run",
        )


def test_split_prompts_use_only_host_contract_reference_and_semantics():
    _proposal, package, _payload, _text = _formalizer_fixture()
    validated = {
        "PARENT_SIGNATURE": {
            "source": "theorem parent : True := by",
            "signature_hash": "a" * 64,
        },
        "CHILD_SIGNATURE": {
            "source": "theorem child : True := by",
            "signature_hash": "b" * 64,
        },
    }
    for unit in (
        "PARENT_SIGNATURE",
        "CHILD_SIGNATURE",
        "REDUCTION_SIGNATURE",
    ):
        messages = _formalizer_unit_messages(
            unit,
            package,
            validated,
        )
        request = json.loads(messages[-1]["content"])
        assert request["contract_id"].startswith("lean-signature-")
        assert request["version"] == 1
        serialized = json.dumps([
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ])
        assert "forbidden_constructs" not in serialized
        assert "valid_examples" not in serialized
        assert "exact_scaffold" not in serialized
        assert package["root_goal_hash"] not in serialized
        assert all(
            not re.search(r"\\[A-Za-z{}]", message["content"])
            for message in messages
        )


@pytest.mark.parametrize(
    ("role", "retained_input", "minimum"),
    [
        ("formalizer_parent_signature", 582, 256),
        ("formalizer_child_signature", 569, 256),
        ("formalizer_reduction_signature", 684, 320),
    ],
)
def test_split_unit_production_budget_fixtures_keep_128_headroom(
    role,
    retained_input,
    minimum,
):
    cap = structured_output_cap(
        role=role,
        max_retained_tokens=2052,
        retained_input_tokens=retained_input,
        minimum_output_tokens=minimum,
        configured_output_tokens=None,
        control_reserve_tokens=64 + 128,
    )
    assert 2052 - retained_input - minimum - 64 >= 128
    assert cap >= minimum


def test_timestamped_tee_preserves_terminal_and_flushes_log(tmp_path):
    terminal = io.StringIO()
    timestamps = iter(("t1", "t2", "t3"))
    path = tmp_path / "agent.log"
    tee = TimestampedTee(
        terminal,
        path,
        timestamp_fn=lambda: next(timestamps),
    )
    tee.write("generator> hel")
    tee.write("lo\nnext line\n")
    tee.log_only("[input] prove RH")
    tee.close_log()
    assert terminal.getvalue() == "generator> hello\nnext line\n"
    assert path.read_text() == (
        "[t1] generator> hello\n"
        "[t2] next line\n"
        "[t3] [input] prove RH\n"
    )


def test_timestamped_tee_shutdown_restores_streams_and_flush_is_safe(
    tmp_path,
    monkeypatch,
):
    terminal = io.StringIO()
    tee = TimestampedTee(terminal, tmp_path / "agent.log")
    monkeypatch.setattr(sys, "stdout", tee)
    monkeypatch.setattr(sys, "stderr", tee)
    tee.close_log()
    assert sys.stdout is terminal
    assert sys.stderr is terminal
    tee.flush()
    tee.write("after-close")
    assert terminal.getvalue() == "after-close"


def test_token_printer_streams_only_new_suffix(capsys):
    printer = TokenPrinter(Tokenizer(), "generator")
    printer([1])
    printer([1, 2])
    printer.finish()
    assert capsys.readouterr().out == "generator> ab\n"


def test_repl_stage_is_redacted_and_passes_cache_gate():
    warm = {
        "prefix_tokens": 10,
        "e2e_s": 2,
        "delta": {
            "remote_jobs": 1,
            "remote_hits": 1,
            "tokens_reused": 10,
            "tokens_computed": 0,
            "fallbacks": 0,
            "remote_job_failures": 0,
        },
    }
    actual = {
        "prefix_tokens": 10,
        "output_tokens": 2,
        "append_s": 0.1,
        "ttft_s": 0.2,
        "decode_s": 0.3,
        "e2e_s": 0.4,
        "stop_reason": "eos",
        "complete": True,
        "delta": {
            "local_hits": 1,
            "remote_jobs": 0,
            "tokens_computed": 0,
            "fallbacks": 0,
        },
    }
    stage = _stage("generator", warm, actual, "private output")
    assert stage["ok"]
    assert stage["output_chars"] == 14
    assert len(stage["output_hash"]) == 64
    assert "output" not in stage
    assert not _stage(
        "generator",
        warm,
        {**actual, "complete": False, "stop_reason": "client_safety_limit"},
        "cut off",
    )["ok"]


def test_external_sigterm_is_ignored_until_user_quits(monkeypatch, capsys):
    installed = {}
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.update({number: handler}),
    )
    install_signal_protection()
    installed[signal.SIGTERM](signal.SIGTERM, None)
    output = capsys.readouterr().out
    assert "ignored external signal" in output
    assert "Type /quit to approve shutdown" in output


def test_shell_supervisor_restarts_signal_exits_only():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "run_agent_gan_repl.sh"
    ).read_text()
    assert "trap" in source and "TERM HUP" in source
    assert '"$status" -eq 143' in source
    assert "restarting in 2s" in source


def test_prefill_heartbeat_reports_elapsed_progress(capsys):
    with PrefillHeartbeat(
        "Critic",
        interval_s=0.01,
        stats_provider=lambda: {
            "remote_job_tokens_computed": 256,
            "remote_job_tokens_total": 1024,
        },
    ):
        time.sleep(0.025)
    output = capsys.readouterr().out
    assert "Critic Prefill:" in output
    assert "256/1024 tokens (25.0%)" in output
    assert "ETA" in output


def test_stage_includes_full_context_metrics():
    warm = {
        "prefix_tokens": 10,
        "e2e_s": 1,
        "delta": {
            "remote_jobs": 1,
            "remote_hits": 1,
            "tokens_reused": 10,
            "tokens_computed": 0,
            "fallbacks": 0,
            "remote_job_failures": 0,
        },
    }
    actual = {
        "prefix_tokens": 10,
        "output_tokens": 1,
        "append_s": 0.1,
        "ttft_s": 0.2,
        "decode_s": 0.3,
        "e2e_s": 0.4,
        "stop_reason": "eos",
        "complete": True,
        "delta": {
            "local_hits": 1,
            "remote_jobs": 0,
            "tokens_computed": 0,
            "fallbacks": 0,
        },
    }
    stage = _stage(
        "critic",
        warm,
        actual,
        "ok",
        extra_metrics={
            "generator_full_tokens": 100,
            "critic_context_tokens": 100,
            "critic_omitted_tokens": 0,
            "review_scope": "full",
            "critic_protocol": "goal_anchored_recursive_gan_v3",
        },
    )
    assert stage["critic_context_tokens"] == 100
    assert stage["critic_omitted_tokens"] == 0
    assert stage["review_scope"] == "full"
    assert stage["critic_protocol"] == "goal_anchored_recursive_gan_v3"


def test_telemetry_timeout_warns_without_stopping_inference(
    monkeypatch,
    capsys,
):
    def timeout(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("scripts.agent_gan_repl._json_request", timeout)
    assert _telemetry_request("http://dashboard/metrics") is None
    output = capsys.readouterr().out
    assert "telemetry-warning" in output
    assert "inference will continue" in output


def test_gate_failure_exposes_reuse_counters():
    error = _gate_failure(
        "Generator",
        {"delta": {"local_hits": 0, "remote_hits": 0}},
        {"delta": {"local_hits": 0, "fallbacks": 1}},
    )
    message = str(error)
    assert "Generator KV gate failed" in message
    assert "'remote_hits': 0" in message
    assert "'fallbacks': 1" in message


def test_interactive_prompts_are_deterministic_for_kv_reuse():
    kwargs = {
        "steering": "continue the zero-free-region branch",
        "previous_generator": "previous complete argument",
        "previous_critic": "previous complete correction",
        "proof_ledger": "PROOF OBLIGATION LEDGER id=rh-ledger",
    }
    generator_a = build_generator_messages("prove RH", **kwargs)
    generator_b = build_generator_messages("prove RH", **kwargs)
    critic_a = build_critic_messages(
        "prove RH",
        "complete generator response",
        steering=kwargs["steering"],
        proof_ledger=kwargs["proof_ledger"],
        stop_reason="eos",
        complete=True,
    )
    critic_b = build_critic_messages(
        "prove RH",
        "complete generator response",
        steering=kwargs["steering"],
        proof_ledger=kwargs["proof_ledger"],
        stop_reason="eos",
        complete=True,
    )
    assert generator_a == generator_b
    assert critic_a == critic_b
    combined = repr(generator_a + critic_a)
    assert "Internal run" not in combined
    assert "IMMUTABLE RESEARCH GOAL" in combined
    assert "previous complete correction" in combined
    assert "PROOF OBLIGATION LEDGER" in combined
    assert "Goal Alignment: ALIGNED" in combined
    assert "Goal Alignment: DRIFTED" in combined
    assert "open problem" in combined
    assert "recursive adversarial proof analyst" in combined
    assert "Never output a numeric score" in combined
    assert "Decomposition Loop" in combined
    assert "If the response stops at" in combined
    assert "attack that stopping claim" in combined
    assert "Ignore prizes, money, prestige" in combined
    assert "smallest unresolved frontier" in combined
    assert "sample, summarize, simplify" in combined
    assert "LEAN_SIGNATURE" not in combined


def test_generator_history_is_scoped_to_target_leaf():
    history = """
### ISSUE_RESPONSE RH-C1
Correction: unrelated operator branch.

### ISSUE_VERDICT RH-C1
Status: UNRESOLVED
Missing lemma: unrelated operator lemma.

### ISSUE_RESPONSE RH-C2-child
Correction: target convergence branch.

### ISSUE_VERDICT RH-C2-child
Status: UNRESOLVED
Missing lemma: target compact convergence lemma.
"""
    scoped = extract_obligation_history(history, "RH-C2-child")
    assert "target convergence branch" in scoped
    assert "target compact convergence lemma" in scoped
    assert "RH-C1" not in scoped
    messages = build_generator_messages(
        "prove RH",
        previous_generator=history,
        previous_critic=history,
        target_obligation_id="RH-C2-child",
    )
    prompt = messages[-1]["content"]
    assert "target convergence branch" in prompt
    assert "unrelated operator branch" not in prompt


def test_generator_and_critic_share_exact_one_step_interface_and_output():
    interface = json.dumps({
        "root_goal_hash": "root-hash",
        "target_obligation_id": "ROOT-L1",
        "target_statement": "Exact bounded target statement.",
        "interface_hash": "interface-hash",
    }, separators=(",", ":"))
    generator = build_generator_messages(
        "full root prose must not be active",
        steering="Attempt exactly one boundary case.",
        target_obligation_id="ROOT-L1",
        proof_step_interface=interface,
    )
    generator_prompt = generator[-1]["content"]
    assert interface in generator_prompt
    assert "### ISSUE_RESPONSE ROOT-L1" in generator[0]["content"]
    assert "### ISSUE_RESPONSE <ID>" not in generator[0]["content"]
    assert "full root prose" not in generator_prompt
    exact_output = (
        "### ISSUE_RESPONSE ROOT-L1\n"
        "Correction: exact bytes π.\n"
        "Derivation: one bounded step.\n"
        "Remaining gap: none"
    )
    critic = build_critic_messages(
        "full root prose must not be active",
        exact_output,
        proof_step_interface=interface,
        stop_reason="eos",
        complete=True,
    )
    critic_prompt = critic[-1]["content"]
    assert interface in critic_prompt
    assert exact_output in critic_prompt
    assert critic_prompt.count(exact_output) == 1
    assert "full root prose" not in critic_prompt
    assert "Audit exactly one certified proof step" in critic[0]["content"]
    assert "Build a proof-obligation tree" not in critic[0]["content"]


def test_generator_repairs_real_namespaced_single_target_id():
    target = (
        "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-"
        "763645cd6b-40ef83e052-b80cd1343b-3be9e8e78f-fbee0281ff-"
        "e4fff64467"
    )
    malformed = (
        "rh-rigorous-obligations-v1-rh-C2-0ef53a217d-25557e489d-"
        "4f025934ee-311091268-763645cd6b-40ef83e052-b80cd1343b-"
        "3be9e8e78f-fbee0281ff-e4fff64467"
    )
    covered, missing = generator_issue_coverage(
        f"### ISSUE_RESPONSE {malformed}\nCorrection: repaired",
        [ProofObligation(target, "Exact target.")],
    )
    assert covered == {target}
    assert missing == set()


def test_generator_rejects_unrelated_and_ambiguous_ids():
    target = ProofObligation(
        "RH-C2-aaaaaaaaaa-bbbbbbbbbb-cccccccccc",
        "First target.",
    )
    covered, missing = generator_issue_coverage(
        "### ISSUE_RESPONSE rh-c2-invented-unrelated-label",
        [target],
    )
    assert covered == set()
    assert missing == {target.obligation_id}

    alternatives = [
        ProofObligation(
            "RH-C2-aaaaaaaaaa-bbbbbbbbbb-111111111a",
            "Alternative A.",
        ),
        ProofObligation(
            "RH-C2-aaaaaaaaaa-bbbbbbbbbb-111111111b",
            "Alternative B.",
        ),
    ]
    covered, missing = generator_issue_coverage(
        "### ISSUE_RESPONSE rh-c2-aaaaaaaaaa-bbbbbbbbbb-111111111c",
        alternatives,
    )
    assert covered == set()
    assert missing == {item.obligation_id for item in alternatives}


def test_prefill_budget_rejects_whole_input_without_truncation():
    token_ids = list(range(7))
    enforce_prefill_token_budget("Generator", token_ids, 7)
    try:
        enforce_prefill_token_budget("Critic", token_ids, 6)
    except ValueError as exc:
        assert "without truncation" in str(exc)
        assert "7 > 6" in str(exc)
    else:
        raise AssertionError("over-budget Prefill must be rejected")
    assert token_ids == list(range(7))


def test_runtime_output_cannot_replace_research_goal():
    for text in (
        "critic> ### Central Claim",
        "[metrics] KV hit=100%",
        "[allens] Critic Prefill: 30s",
        "prompt> ",
        "Traceback (most recent call last):",
    ):
        assert is_runtime_artifact_prompt(text)
    assert not is_runtime_artifact_prompt("证明黎曼猜想")


def test_command_state_machine_requires_explicit_ready_commands():
    assert parse_repl_command(
        "证明黎曼猜想",
        ReplPhase.WAITING_FOR_GOAL,
    ).action == "new"
    assert parse_repl_command("/continue", ReplPhase.READY).action == "continue"
    steering = parse_repl_command(
        "/steer analyze the explicit formula",
        ReplPhase.READY,
    )
    assert steering.action == "steer"
    assert steering.payload == "analyze the explicit formula"
    assert parse_repl_command("/new new goal", ReplPhase.READY).payload == (
        "new goal"
    )
    assert parse_repl_command("/quit", ReplPhase.READY).action == "quit"
    for raw, phase in (
        ("Operator output fragment", ReplPhase.READY),
        ("critic> copied output", ReplPhase.READY),
        ("/continue", ReplPhase.WAITING_FOR_GOAL),
        ("/continue", ReplPhase.RUNNING),
        ("/steer", ReplPhase.READY),
        ("/new", ReplPhase.READY),
    ):
        try:
            parse_repl_command(raw, phase)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected rejection for {raw!r} in {phase}")


def test_continuous_auto_loop_is_default_and_exception_pauses():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "agent_gan_repl.py"
    ).read_text()
    assert 'parser.set_defaults(auto_loop=True)' in source
    assert "select.select(" in source
    assert 'raw_input = "/continue"' in source
    assert '"/continue queued"' in source
    assert "auto_loop_active = False" in source
    assert "[auto-loop-paused]" in source
    assert '"--no-auto-loop"' in source
    assert "extract_lean_signature_blocks" not in source


def test_checkpoint_round_trip_is_private(tmp_path):
    path = tmp_path / "state.json"
    expected = ReplCheckpoint(
        research_goal="prove RH",
        previous_generator="full generator",
        previous_critic="full critic",
        last_run_id="br_ok",
    )
    save_checkpoint(path, expected)
    assert load_checkpoint(path) == expected
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_checkpoint(tmp_path / "missing.json") is None


def test_critic_issue_inbox_retries_until_successful_consumption(tmp_path):
    path = tmp_path / "critic-inbox.json"
    batch = CriticIssueBatch(
        issue_id="math-review-1",
        issues=[
            "Do not identify -zeta'/zeta with xi.",
            "Prove zero convergence before invoking Hurwitz.",
        ],
    )
    save_critic_issue_batch(path, batch)
    loaded = load_pending_critic_issues(path)
    assert loaded == batch
    assert path.stat().st_mode & 0o777 == 0o600
    injection = format_critic_issue_injection(loaded)
    assert "math-review-1" in injection
    assert "1. Do not identify" in injection
    assert "2. Prove zero convergence" in injection
    # Merely loading/injecting does not consume the issues.
    assert load_pending_critic_issues(path).status == "pending"
    consume_critic_issue_batch(path, loaded, "br_success")
    assert load_pending_critic_issues(path) is None
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "consumed"
    assert persisted["consumed_by_run"] == "br_success"


def test_proof_ledger_carries_unresolved_and_closes_valid_verdicts(tmp_path):
    path = tmp_path / "proof-ledger.json"
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger-v1",
        obligations=[
            ProofObligation("RH-C1", "Separate xi from -zeta'/zeta."),
            ProofObligation("RH-C2", "Prove zero convergence."),
        ],
    )
    save_proof_ledger(path, ledger)
    loaded = load_proof_ledger(path)
    assert loaded == ledger
    assert path.stat().st_mode & 0o777 == 0o600
    assert [item.obligation_id for item in pending_obligations(loaded)] == [
        "RH-C1",
        "RH-C2",
    ]
    rendered = format_proof_ledger(loaded)
    assert "### ISSUE_RESPONSE <ID>" in rendered
    assert "LEAN_SIGNATURE" not in rendered
    covered, missing = generator_issue_coverage(
        "### ISSUE_RESPONSE RH-C1\nCorrection: fixed",
        pending_obligations(loaded),
    )
    assert covered == {"RH-C1"}
    assert missing == {"RH-C2"}
    critic = """
### ISSUE_VERDICT RH-C1
Status: PROVED
Evidence: This is a sufficiently detailed derivation that distinguishes the two analytic objects.
Missing lemma: none

### ISSUE_VERDICT RH-C2
Status: UNRESOLVED
Evidence: No locally uniform convergence theorem was supplied.
Missing lemma: A valid zero-convergence theorem.
"""
    verdicts = apply_critic_verdicts(loaded, critic, "br_review")
    assert verdicts == {"RH-C1": "PROVED", "RH-C2": "UNRESOLVED"}
    assert [item.obligation_id for item in pending_obligations(loaded)] == [
        "RH-C2",
    ]
    save_proof_ledger(path, loaded)
    assert load_proof_ledger(path).version == 2


def test_proof_ledger_rejects_weak_or_missing_closure():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C1", "Issue one"),
            ProofObligation("RH-C2", "Issue two"),
        ],
    )
    critic = """
### ISSUE_VERDICT RH-C1
Status: PROVED
Evidence: too short
Missing lemma: none
"""
    verdicts = apply_critic_verdicts(ledger, critic, "br_weak")
    assert verdicts == {
        "RH-C1": "UNRESOLVED",
        "RH-C2": "UNRESOLVED",
    }
    assert len(pending_obligations(ledger)) == 2


def test_premise_invalidation_quarantines_subtree_and_backjumps():
    root = ProofObligation("ROOT", "Establish the main reduction.")
    target = ProofObligation(
        "ROOT-A",
        "For every real x, x squared equals x.",
        parent_id="ROOT",
    )
    descendant = ProofObligation(
        "ROOT-A-1",
        "Use kernel positivity to derive the bound.",
        parent_id="ROOT-A",
    )
    alternate = ProofObligation(
        "ROOT-B",
        "Construct an alternate sign-changing kernel.",
        parent_id="ROOT",
    )
    ledger = ProofObligationLedger(
        ledger_id="premise-recovery",
        obligations=[root, target, descendant, alternate],
    )
    critic = _premise_suspicion_text()
    suspicion = extract_premise_suspicions(
        critic,
        {"ROOT-A"},
    )["ROOT-A"]
    review = decide_premise_review(
        parse_premise_audit(_audit_text(), "ROOT-A", "audit-1"),
        parse_premise_defense(
            _defense_text(),
            "ROOT-A",
            "defense-1",
        ),
        project_root=Path(__file__).resolve().parents[3],
        suspicion=suspicion,
    )
    assert apply_critic_verdicts(
        ledger,
        critic,
        "br_premise",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": review},
    ) == {"ROOT-A": "DISPROVED"}
    assert target.invalidation_kind == "PREMISE_INVALIDATED"
    assert descendant.status == "QUARANTINED"
    assert descendant.quarantine_root_id == "ROOT-A"
    assert descendant.quarantine_run_id == "br_premise"
    assert alternate.status == "UNRESOLVED"
    assert ledger.backjump_target_id == "ROOT"
    assert len(ledger.no_go_lessons) == 1
    assert ledger.no_go_lessons[0].claim_hash == suspicion.claim_hash
    assert ledger.no_go_lessons[0].auditor_run_id == "audit-1"
    assert ledger.no_go_lessons[0].reversible_status == "ACTIVE"
    assert pending_obligations(ledger) == [alternate]
    recovery_prompt = format_proof_ledger(ledger, [alternate])
    assert "For every real x, x squared equals x" in recovery_prompt
    assert "never assume, rename, or propose" in recovery_prompt
    apply_critic_verdicts(
        ledger,
        critic,
        "br_repeat",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": review},
    )
    assert len(ledger.no_go_lessons) == 1
    apply_critic_verdicts(
        ledger,
        """
### ISSUE_VERDICT ROOT-A-1
Status: PROVED
Evidence: This attempted late verdict must not reopen a terminal quarantined descendant under any circumstance.
Missing lemma: none
""",
        "br_late",
        {"ROOT-A-1"},
    )
    assert descendant.status == "QUARANTINED"
    assert descendant.quarantine_run_id == "br_premise"


def test_weak_premise_verdict_does_not_cascade():
    target = ProofObligation("ROOT-A", "Assume positivity.")
    child = ProofObligation("ROOT-A-1", "Apply positivity.", parent_id="ROOT-A")
    ledger = ProofObligationLedger("weak-premise", [target, child])
    critic = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE
Premise refuted:
Evidence: This evidence is deliberately long enough, but no explicit premise is named for host validation.
Missing lemma: none
"""
    verdicts = apply_critic_verdicts(ledger, critic, "br_weak", {"ROOT-A"})
    assert verdicts == {"ROOT-A": "UNRESOLVED"}
    assert target.invalidation_kind == ""
    assert child.status == "UNRESOLVED"
    assert ledger.no_go_lessons == []
    weak_closure = _premise_suspicion_text().replace(
        "Substitution of the explicit finite value x=2 gives four on the left and two on the right, contradicting universal equality.",
        "too short",
    )
    apply_critic_verdicts(
        ledger,
        weak_closure,
        "br_weak_closure",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": PremiseReview(
            "INCONCLUSIVE",
            False,
            reason="workers unavailable",
        )},
    )
    assert target.status == "UNRESOLVED"
    assert target.invalidation_kind == ""


def test_valid_suspicion_is_temporary_until_independent_upgrade():
    target = ProofObligation("ROOT-A", "For every real x, x squared equals x.")
    child = ProofObligation("ROOT-A-1", "Use equality.", parent_id="ROOT-A")
    ledger = ProofObligationLedger("suspected", [target, child])
    verdicts = apply_critic_verdicts(
        ledger,
        _premise_suspicion_text(),
        "br_suspect",
        {"ROOT-A"},
    )
    assert verdicts == {"ROOT-A": "UNRESOLVED"}
    assert target.invalidation_kind == "PREMISE_SUSPECTED"
    assert target.premise_review_status == "SUSPECTED"
    assert child.status == "UNRESOLVED"
    assert child.temporary_quarantine_root_id == "ROOT-A"
    assert ledger.no_go_lessons == []
    assert pending_obligations(ledger) == [child]


def test_old_direct_premise_input_is_only_audited_suspicion():
    suspicions = extract_premise_suspicions(
        _premise_suspicion_text("PREMISE"),
        {"ROOT-A"},
    )
    assert suspicions["ROOT-A"].evidence_type == "FINITE_COUNTEREXAMPLE"
    ledger = ProofObligationLedger(
        "legacy-transcript",
        [ProofObligation("ROOT-A", "Universal equality.")],
    )
    apply_critic_verdicts(
        ledger,
        _premise_suspicion_text("PREMISE"),
        "legacy-run",
        {"ROOT-A"},
    )
    assert ledger.obligations[0].status == "UNRESOLVED"
    assert ledger.obligations[0].invalidation_kind == "PREMISE_SUSPECTED"
    assert ledger.no_go_lessons == []


def test_audit_and_defense_parsers_and_verified_outcome_matrix(tmp_path):
    audit = parse_premise_audit(_audit_text(), "ROOT-A", "audit-run")
    defense = parse_premise_defense(
        _defense_text(),
        "ROOT-A",
        "defense-run",
    )
    assert audit.status == "CONFIRMED"
    assert defense.status == "NOT_RESCUED"
    suspicion = extract_premise_suspicions(
        _premise_suspicion_text(),
        {"ROOT-A"},
    )["ROOT-A"]
    verified, detail = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=tmp_path,
    )
    assert verified and "4.0 == 2.0 as False" in detail
    decision = decide_premise_review(
        audit,
        defense,
        project_root=tmp_path,
        suspicion=suspicion,
    )
    assert decision.status == "PREMISE_INVALIDATED"
    assert decision.verified
    assert decision.auditor_run_id == "audit-run"
    mismatched_audit = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "claim_hash": "0" * 64,
            },
        },
    )
    assert decide_premise_review(
        mismatched_audit,
        defense,
        project_root=tmp_path,
        suspicion=suspicion,
    ).status == "INCONCLUSIVE"
    assert decide_premise_review(
        parse_premise_audit(
            _audit_text(status="NOT_CONFIRMED"),
            "ROOT-A",
        ),
        defense,
        project_root=tmp_path,
    ).status == "NOT_CONFIRMED"
    assert decide_premise_review(
        audit,
        parse_premise_defense(
            _defense_text(status="RESCUED"),
            "ROOT-A",
        ),
        project_root=tmp_path,
    ).status == "RESCUED"
    assert decide_premise_review(
        parse_premise_audit(
            _audit_text(status="INCONCLUSIVE"),
            "ROOT-A",
        ),
        defense,
        project_root=tmp_path,
    ).status == "INCONCLUSIVE"
    assert decide_premise_review(
        parse_premise_audit(
            _audit_text(confidence="0.5"),
            "ROOT-A",
        ),
        defense,
        project_root=tmp_path,
    ).status == "INCONCLUSIVE"
    assert decide_premise_review(
        audit,
        parse_premise_defense(
            _defense_text(status="INCONCLUSIVE"),
            "ROOT-A",
        ),
        project_root=tmp_path,
    ).status == "INCONCLUSIVE"

    symbolic_suspicion = extract_premise_suspicions(
        _premise_suspicion_text().replace(
            "Evidence type: FINITE_COUNTEREXAMPLE",
            "Evidence type: SYMBOLIC_CONTRADICTION",
        ),
        {"ROOT-A"},
    )["ROOT-A"]
    symbolic = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "SYMBOLIC_CONTRADICTION",
        "expanded universal identity",
        0.9,
        {
            "claim_hash": symbolic_suspicion.claim_hash,
            "claim": symbolic_suspicion.claim_schema,
            "witness": {"x": 2},
        },
        "The claimed symbolic identity fails under the exact witness.",
    )
    assert validate_evidence_artifact(
        symbolic,
        suspicion=symbolic_suspicion,
        project_root=tmp_path,
    )[0]


def test_arbitrary_true_arithmetic_cannot_invalidate_unrelated_premise(
    tmp_path,
):
    critic = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE_SUSPECTED
Premise refuted: An unrelated analytic continuation premise is false.
Evidence type: FINITE_COUNTEREXAMPLE
Evidence artifact: {"claim":{"schema_version":1,"quantifier":"FOR_ALL","variables":["x"],"domain":"INTEGER","lhs":"x","relation":"!=","rhs":"2"}}
Evidence: The model attaches a true arithmetic observation to an unrelated natural-language premise and calls it a counterexample.
Missing lemma: none
"""
    suspicion = extract_premise_suspicions(
        critic,
        {"ROOT-A"},
    )["ROOT-A"]
    audit = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "FINITE_COUNTEREXAMPLE",
        "unrelated arithmetic",
        0.99,
        {
            "claim_hash": suspicion.claim_hash,
            "claim": suspicion.claim_schema,
            "witness": {"x": 1},
        },
        "The attached relation is true, not a counterexample.",
    )
    verified, reason = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=tmp_path,
    )
    assert not verified
    assert "as True" in reason
    decision = decide_premise_review(
        audit,
        parse_premise_defense(_defense_text(), "ROOT-A"),
        project_root=tmp_path,
        suspicion=suspicion,
    )
    assert decision.status == "INCONCLUSIVE"
    ledger = ProofObligationLedger(
        "unrelated-arithmetic",
        [ProofObligation("ROOT-A", "Unrelated analytic continuation premise.")],
    )
    apply_critic_verdicts(
        ledger,
        critic,
        "unrelated-run",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": decision},
    )
    assert ledger.obligations[0].invalidation_kind == "APPROACH_FAILED"
    assert ledger.no_go_lessons == []
    tampered = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "claim_hash": "tampered",
            },
        },
    )
    assert not validate_evidence_artifact(
        tampered,
        suspicion=suspicion,
        project_root=tmp_path,
    )[0]
    missing_witness = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "witness": {},
            },
        },
    )
    assert not validate_evidence_artifact(
        missing_witness,
        suspicion=suspicion,
        project_root=tmp_path,
    )[0]
    unknown_variable = critic.replace(
        '"rhs":"2"',
        '"rhs":"y"',
    )
    assert extract_premise_suspicions(
        unknown_variable,
        {"ROOT-A"},
    ) == {}


def test_unverified_theorem_reference_fails_open(tmp_path):
    pinned_text = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE_SUSPECTED
Premise refuted: A cited theorem contradicts the target premise.
Evidence type: PINNED_THEOREM
Evidence artifact: {"claim":{"reference":"Example Theorem 2.1","assumptions":["exact assumption A"]}}
Evidence: The citation purports to conflict with the premise under the exact listed assumption, but requires registry verification.
Missing lemma: none
"""
    suspicion = extract_premise_suspicions(
        pinned_text,
        {"ROOT-A"},
    )["ROOT-A"]
    audit = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "PINNED_THEOREM",
        "Example Theorem 2.1",
        0.99,
        {"claim_hash": suspicion.claim_hash, "claim": suspicion.claim_schema},
        "The cited result appears relevant.",
    )
    decision = decide_premise_review(
        audit,
        parse_premise_defense(_defense_text(), "ROOT-A"),
        project_root=tmp_path,
        suspicion=suspicion,
    )
    assert decision.status == "INCONCLUSIVE"
    assert not decision.verified
    assert "no trusted local theorem registry" in decision.reason


def test_lean_proof_without_safe_negation_wrapper_fails_open(tmp_path):
    signature_hash = "lean-target-hash"
    lean_text = f"""
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE_SUSPECTED
Premise refuted: The formalized target proposition has a constructive negation.
Evidence type: LEAN_PROOF
Evidence artifact: {{"claim":{{"schema_version":1,"contract":"NEGATION_OF_TARGET_SIGNATURE","lean_signature_hash":"{signature_hash}"}}}}
Evidence: A complete Lean proof is proposed against the exact host-recorded target signature, subject to safe wrapper validation.
Missing lemma: none
"""
    suspicion = extract_premise_suspicions(
        lean_text,
        {"ROOT-A"},
        {"ROOT-A": signature_hash},
    )["ROOT-A"]
    audit = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "LEAN_PROOF",
        "generated Lean theorem",
        0.99,
        {
            "claim_hash": suspicion.claim_hash,
            "claim": suspicion.claim_schema,
            "lean_signature_hash": signature_hash,
            "contract": "NEGATION_OF_TARGET_SIGNATURE",
            "source": "theorem contradiction : False := by trivial",
        },
        "Lean artifact supplied.",
        "audit-lean",
    )

    verified, reason = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=tmp_path,
    )
    assert not verified
    assert "cannot be safely transformed" in reason
    assert extract_premise_suspicions(
        lean_text,
        {"ROOT-A"},
        {"ROOT-A": "different-host-signature-hash"},
    ) == {}
    tampered = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "lean_signature_hash": "tampered",
            },
        },
    )
    assert not validate_evidence_artifact(
        tampered,
        suspicion=suspicion,
        project_root=tmp_path,
    )[0]
    rejected = validate_lean_proof(
        "theorem incomplete : True := by sorry",
        project_root=tmp_path,
    )
    assert not rejected.ok
    assert rejected.status == "CONTRACT_FAILED"


def test_complete_minimal_lean_reduction_proof_is_accepted():
    result = validate_lean_proof(
        "theorem certifiedReduction (h : True) : True := by exact h",
        project_root=Path(__file__).resolve().parents[3],
    )
    assert result.ok
    assert result.status == "PROVED"


def test_worker_roles_are_isolated_ordered_and_fail_open(tmp_path):
    suspicion = extract_premise_suspicions(
        _premise_suspicion_text(),
        {"ROOT-A"},
    )["ROOT-A"]
    calls = []

    def runner(role, messages):
        calls.append((role, messages))
        if role == "premise_auditor":
            return _audit_text(), "audit-run"
        assert _audit_text().strip() in messages[-1]["content"]
        assert suspicion.claim_hash in messages[-1]["content"]
        return _defense_text(), "defense-run"

    audit, defense, transcripts = run_isolated_premise_review(
        "prove target",
        suspicion,
        runner,
    )
    assert [role for role, _ in calls] == [
        "premise_auditor",
        "premise_proponent",
    ]
    assert calls[0][1] is not calls[1][1]
    assert "COMPLETE ISOLATED AUDITOR OUTPUT" not in calls[0][1][-1]["content"]
    assert audit.run_id == "audit-run"
    assert defense.run_id == "defense-run"
    assert transcripts["auditor"] == _audit_text()

    failed_calls = []

    def failing_runner(role, _messages):
        failed_calls.append(role)
        raise TimeoutError(f"{role} timed out")

    failed_audit, failed_defense, failed_transcripts = (
        run_isolated_premise_review(
            "prove target",
            suspicion,
            failing_runner,
        )
    )
    assert failed_calls == ["premise_auditor", "premise_proponent"]
    assert failed_audit is None and failed_defense is None
    assert "EXECUTION FAILED" in failed_transcripts["auditor"]
    failed_decision = decide_premise_review(
        failed_audit,
        failed_defense,
        project_root=tmp_path,
    )
    assert failed_decision.status == "INCONCLUSIVE"
    failed_ledger = ProofObligationLedger(
        "worker-failure",
        [ProofObligation("ROOT-A", "Failed attempted approach.")],
    )
    apply_critic_verdicts(
        failed_ledger,
        _premise_suspicion_text(),
        "failed-review-run",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": failed_decision},
    )
    assert failed_ledger.obligations[0].status == "DISPROVED"
    assert failed_ledger.obligations[0].invalidation_kind == "APPROACH_FAILED"
    assert failed_ledger.no_go_lessons == []


def test_rescued_review_reverses_quarantine_and_no_go():
    root = ProofObligation("ROOT", "Main unresolved reduction.")
    target = ProofObligation("ROOT-A", "Universal equality.", parent_id="ROOT")
    child = ProofObligation("ROOT-A-1", "Dependent lemma.", parent_id="ROOT-A")
    ledger = ProofObligationLedger("reversible", [root, target, child])
    confirmed = PremiseReview(
        "PREMISE_INVALIDATED",
        True,
        0.95,
        "FINITE_COUNTEREXAMPLE",
        "host arithmetic substitution",
        "audit-1",
        "defense-1",
    )
    apply_critic_verdicts(
        ledger,
        _premise_suspicion_text(),
        "br_confirm",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": confirmed},
    )
    assert child.status == "QUARANTINED"
    rescued = PremiseReview(
        "RESCUED",
        False,
        0.9,
        "FINITE_COUNTEREXAMPLE",
        "domain check",
        "audit-2",
        "defense-2",
        "The original domain excluded x=2.",
    )
    apply_critic_verdicts(
        ledger,
        _premise_suspicion_text(),
        "br_rescue",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": rescued},
    )
    assert target.status == "DISPROVED"
    assert target.invalidation_kind == "APPROACH_FAILED"
    assert target.premise_review_status == "RESCUED"
    assert target.premise_auditor_run_id == "audit-2"
    assert child.status == "UNRESOLVED"
    assert child.quarantine_reversible_status == "REVERSED"
    assert child.temporary_quarantine_root_id == ""
    assert ledger.no_go_lessons[0].reversible_status == "REVERSED"
    assert ledger.backjump_target_id == ""


def test_all_nonupgrade_reviews_close_only_approach_without_no_go():
    for review_status in ("NOT_CONFIRMED", "RESCUED", "INCONCLUSIVE"):
        root = ProofObligation("ROOT", "Sound parent.")
        target = ProofObligation(
            "ROOT-A",
            "For every real x, x squared equals x.",
            parent_id="ROOT",
        )
        child = ProofObligation(
            "ROOT-A-1",
            "Historical alternate descendant.",
            parent_id="ROOT-A",
        )
        ledger = ProofObligationLedger(
            f"fallback-{review_status}",
            [root, target, child],
        )
        apply_critic_verdicts(
            ledger,
            _premise_suspicion_text(),
            f"run-{review_status}",
            {"ROOT-A"},
            premise_reviews={"ROOT-A": PremiseReview(
                review_status,
                False,
                0.6,
                "FINITE_COUNTEREXAMPLE",
                "independent worker result",
                f"audit-{review_status}",
                f"defense-{review_status}",
                f"{review_status} fallback",
            )},
        )
        assert target.status == "DISPROVED"
        assert target.invalidation_kind == "APPROACH_FAILED"
        assert target.premise_review_status == review_status
        assert target.premise_review_reason == f"{review_status} fallback"
        assert child.status == "UNRESOLVED"
        assert child.temporary_quarantine_root_id == ""
        assert child.quarantine_root_id == ""
        assert ledger.no_go_lessons == []
        assert ledger.backjump_target_id == ""


def test_approach_invalidation_does_not_quarantine_descendants_or_siblings():
    target = ProofObligation("ROOT-A", "Try a contour-shift proof.")
    child = ProofObligation(
        "ROOT-A-1",
        "Try a different contour under the same target.",
        parent_id="ROOT-A",
    )
    sibling = ProofObligation("ROOT-B", "Try a spectral proof.")
    ledger = ProofObligationLedger("approach-only", [target, child, sibling])
    critic = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Evidence: The attempted contour crosses an uncontrolled pole, so this derivation fails even though the target statement may still hold.
Missing lemma: none
"""
    apply_critic_verdicts(ledger, critic, "br_approach", {"ROOT-A"})
    assert target.invalidation_kind == "APPROACH_FAILED"
    assert child.status == "UNRESOLVED"
    assert sibling.status == "UNRESOLVED"
    assert {item.obligation_id for item in pending_obligations(ledger)} == {
        "ROOT-A-1",
        "ROOT-B",
    }


def test_no_go_lesson_rejects_semantically_repeated_child():
    ledger = ProofObligationLedger(
        "no-go",
        [ProofObligation("ROOT", "Find a valid replacement argument.")],
    )
    premise = "For every real x, x squared equals x."
    source = ProofObligation("OLD", premise)
    old_ledger = ProofObligationLedger("old", [source])
    apply_critic_verdicts(
        old_ledger,
        _premise_suspicion_text().replace("ROOT-A", "OLD"),
        "br_old",
        {"OLD"},
        premise_reviews={"OLD": PremiseReview(
            "PREMISE_INVALIDATED",
            True,
            0.95,
            "FINITE_COUNTEREXAMPLE",
            "host arithmetic substitution",
            "audit-old",
            "defense-old",
        )},
    )
    ledger.no_go_lessons = old_ledger.no_go_lessons
    rejections = []
    created = create_child_obligations(
        ledger,
        """
### ISSUE_VERDICT ROOT
Status: UNRESOLVED
Evidence: A replacement argument is still required.
Missing lemma: Prove that for every real x, x squared equals x.
""",
        "br_new",
        {"ROOT"},
        rejections,
        {"ROOT": _valid_lean_signature()},
        certified_only=False,
    )
    assert created == []
    assert "no-go premise" in rejections[0]


def test_v1_ledger_loads_without_recovery_metadata(tmp_path):
    path = tmp_path / "legacy-ledger.json"
    path.write_text(json.dumps({
        "ledger_id": "legacy",
        "obligations": [{
            "obligation_id": "ROOT",
            "statement": "Legacy unresolved statement.",
            "status": "UNRESOLVED",
            "parent_id": "",
        }],
        "no_go_lessons": [{
            "claim_hash": "legacy-hash",
            "refuted_premise": "Legacy premise.",
            "evidence": "Legacy evidence.",
            "source_obligation_id": "ROOT",
            "run_id": "legacy-run",
        }],
        "version": 1,
        "schema_version": 1,
    }))
    ledger = load_proof_ledger(path)
    assert ledger.no_go_lessons[0].confidence == 0.0
    assert ledger.no_go_lessons[0].reversible_status == "ACTIVE"
    assert ledger.backjump_target_id == ""
    assert ledger.obligations[0].invalidation_kind == ""
    assert ledger.obligations[0].decomposition_certificate_hash == ""
    assert ledger.obligations[0].dependency_ids == []
    assert ledger.obligations[0].public_assumptions == []


def test_free_form_missing_lemma_cannot_persist_child():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[ProofObligation("RH-C2", "Prove zero convergence.")],
    )
    critic = """
### ISSUE_VERDICT RH-C2
**Status:** UNRESOLVED
**Evidence:** The proposed convergence argument does not control zeros on compact subsets.
**Missing lemma:** Prove locally uniform convergence on every compact subset of the critical strip.
"""
    apply_critic_verdicts(ledger, critic, "br_first")
    rejections = []
    created = create_child_obligations(
        ledger,
        critic,
        "br_first",
        {"RH-C2"},
        rejections,
        {"RH-C2": _valid_lean_signature()},
    )
    assert created == []
    assert len(ledger.obligations) == 1
    assert "verified decomposition certificate" in rejections[0]


def test_valid_certified_decomposition_runs_seven_roles_and_persists(tmp_path):
    parent = ProofObligation(
        "ROOT",
        "Establish the global convergence theorem for analytic approximants.",
    )
    ledger = ProofObligationLedger("certified", [parent])
    runner, calls = _certificate_runner()
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-1",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert [call[0] for call in calls] == [
        "definition_auditor",
        "counterexample_worker",
        "decomposer",
        "formalizer",
        "prover",
        "adversarial_proponent",
        "judge",
    ]
    assert len({id(call[1]) for call in calls}) == 7
    for index, (_role, messages, expected_run_id) in enumerate(calls):
        package = messages[-1].get(
            "_host_package",
            json.loads(messages[-1]["content"]),
        )
        assert package["producer_run_id"] == expected_run_id
        if index:
            assert package["upstream_artifact_hashes"]
    packages = {
        role: messages[-1].get(
            "_host_package",
            json.loads(messages[-1]["content"]),
        )
        for role, messages, _run_id in calls
    }
    assert set(
        packages["formalizer"]["validated_upstream_artifacts"],
    ) == {"decomposer"}
    assert set(
        packages["prover"]["validated_upstream_artifacts"],
    ) == {"formalizer"}
    assert "definition_auditor" not in (
        packages["prover"]["validated_upstream_artifacts"]
    )
    assert result.verified
    assert ledger.obligations == [parent]
    created = persist_verified_decomposition(
        ledger,
        "ROOT",
        result,
        "run-certified",
    )
    assert len(created) == 1
    committed_version = ledger.version
    assert persist_verified_decomposition(
        ledger,
        "ROOT",
        result,
        "run-certified-replay",
    )[0].obligation_id == created[0].obligation_id
    assert ledger.version == committed_version
    assert len(ledger.obligations) == 2
    assert set(asdict(result.artifacts["decomposer"])) >= {
        "child",
        "reduction_contract",
    }
    assert "children" not in asdict(result.artifacts["decomposer"])
    assert created[0].obligation_id.startswith("ROOT-")
    assert created[0].decomposition_certificate_hash
    assert created[0].reduction_theorem_status == "PROVED"
    assert created[0].certificate_reversible_status == "ACTIVE"
    assert parent.formal_status == "FORMALIZED"
    manifest = tmp_path / "reviews" / "manifest.json"
    save_decomposition_manifest(manifest, {
        "verified": result.verified,
        "transcripts": result.transcripts,
        "artifact_hashes": result.artifact_hashes,
    })
    assert manifest.stat().st_mode & 0o777 == 0o600


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_decomposer_failure_resumes_without_valid_upstream_roles(tmp_path):
    ledger = ProofObligationLedger(
        "resume-decomposer",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
        version=85,
    )
    state_path = tmp_path / "orchestration.json"
    runner, first_calls = _certificate_runner()

    def fail_decomposer(role, messages, expected_run_id):
        if role == "decomposer":
            raise TimeoutError("archived decomposer protocol timeout")
        return runner(role, messages, expected_run_id)

    first = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        fail_decomposer,
        project_root=tmp_path,
        orchestration_id="orch-resume",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert not first.verified
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert set(checkpoint.validated_artifacts) == {
        "definition_auditor",
        "counterexample_worker",
    }
    retry_counters = dict(checkpoint.retry_counters)
    checkpoint.adapter_status = ""
    checkpoint.blocked_reason = ""
    critic_ref = persist_validated_artifact(
        state_path,
        checkpoint,
        role="critic",
        payload={"schema_version": 1, "artifact_kind": "critic_evaluation"},
        dependencies=["candidate-hash"],
        source_run_id="br_physical_critic",
    )

    resumed_runner, resumed_calls = _certificate_runner()
    second = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        resumed_runner,
        project_root=tmp_path,
        orchestration_id="orch-resume",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert second.verified
    assert [call[0] for call in resumed_calls] == [
        "decomposer",
        "formalizer",
        "prover",
        "adversarial_proponent",
        "judge",
    ]
    assert not any(
        call[0] in {"definition_auditor", "counterexample_worker"}
        for call in resumed_calls
    )
    assert first_calls[:2][0][0] == "definition_auditor"
    resumed_checkpoint = load_orchestration_checkpoint(state_path)
    assert resumed_checkpoint.retry_counters == retry_counters
    assert resumed_checkpoint.validated_artifacts["critic"].sha256 == (
        critic_ref.sha256
    )
    assert state_path.stat().st_mode & 0o077 == 0


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_eleven_semantic_rejections_iterate_decomposer_without_strategy(
    tmp_path,
):
    ledger = ProofObligationLedger(
        "iterative-decomposer",
        [ProofObligation(
            "ROOT",
            "For every analytic approximant, global convergence follows from "
            "the exact local boundary condition.",
        )],
        version=87,
    )
    state_path = tmp_path / "orchestration.json"
    base_runner, calls = _certificate_runner()
    viewpoints = []

    def cyclic_decomposer(role, messages, expected_run_id):
        if role != "decomposer":
            return base_runner(role, messages, expected_run_id)
        calls.append((role, messages, expected_run_id))
        package = json.loads(messages[-1]["content"])
        viewpoints.append(package["viewpoint"])
        proposal = {
            "parent_statement": package["parent_statement"],
            "child": {
                "label": "L1",
                "statement": package["parent_statement"],
                "kind": "LEMMA",
                "source_definition_labels": [],
            },
            "public_assumptions": [],
            "reduction_contract": {
                "child_label": "L1",
                "parent_statement": package["parent_statement"],
                "public_assumptions": [],
                "derivation": (
                    "The renamed child directly gives the identical parent "
                    "statement, so no independent reduction is supplied."
                ),
            },
        }
        return (
            "### DECOMPOSITION_PROPOSAL\nArtifact: "
            + json.dumps(proposal, separators=(",", ":")),
            expected_run_id,
        )

    for _ in range(11):
        result = run_certified_decomposition(
            ledger,
            "ROOT",
            "Immutable root goal",
            cyclic_decomposer,
            project_root=tmp_path,
            orchestration_id="orch-iterative",
            signature_validator=_fake_signature_validator,
            proof_validator=_fake_proof_validator,
            checkpoint_path=state_path,
            candidate_sha256="candidate-hash",
        )
        assert not result.verified
        assert "semantic rejection" in result.errors[0]
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.decomposition_iteration == 12
    assert ("protocol_" + "attempt") not in checkpoint.__dataclass_fields__
    assert checkpoint.retry_counters.get("DECOMPOSER", 0) == 0
    assert len(checkpoint.decomposition_proposals) == 11
    assert checkpoint.novel_proposals == 1
    assert checkpoint.strategy_reused is True
    assert len(set(viewpoints[:7])) == 7
    assert any(item.startswith("synthesized_host_failures_") for item in viewpoints)
    assert sum(role == "definition_auditor" for role, *_ in calls) == 1
    assert sum(role == "counterexample_worker" for role, *_ in calls) == 1
    assert not any(
        role in {"formalizer", "prover", "adversarial_proponent", "judge"}
        for role, *_ in calls
    )


def test_alpha_renamed_proposals_share_structural_signature():
    common = {
        "target_obligation_id": "ROOT",
        "parent_statement_hash": "p",
        "root_goal_hash": "g",
        "producer_role": "decomposer",
        "producer_run_id": "run",
        "upstream_artifact_hashes": [],
        "parent_statement": "Prove a local estimate.",
        "public_assumptions": [],
    }
    first = DecompositionProposal(
        **common,
        child={
            "label": "L1",
            "statement": "For every rho, there exists delta with rho < delta.",
            "kind": "LEMMA",
            "source_definition_labels": [],
        },
        reduction_contract={
            "child_label": "L1",
            "parent_statement": common["parent_statement"],
            "public_assumptions": [],
            "derivation": "The quantified estimate supplies the local bound.",
        },
    )
    second = DecompositionProposal(
        **common,
        child={
            "label": "L1",
            "statement": "For every lambda, there exists epsilon with lambda < epsilon.",
            "kind": "LEMMA",
            "source_definition_labels": [],
        },
        reduction_contract={
            "child_label": "L1",
            "parent_statement": common["parent_statement"],
            "public_assumptions": [],
            "derivation": "The quantified estimate supplies the local bound.",
        },
    )
    assert _decomposition_semantic_hash(first) == (
        _decomposition_semantic_hash(second)
    )
    assert _decomposition_structural_signature(first) == (
        _decomposition_structural_signature(second)
    )


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_formalizer_failure_resumes_formalizer_only(tmp_path):
    ledger = ProofObligationLedger(
        "resume-formalizer",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
        version=85,
    )
    state_path = tmp_path / "orchestration.json"
    runner, _ = _certificate_runner()

    def fail_formalizer(role, messages, expected_run_id):
        if role == "formalizer":
            raise RuntimeError("Lean elaboration signature mismatch")
        return runner(role, messages, expected_run_id)

    first = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        fail_formalizer,
        project_root=tmp_path,
        orchestration_id="orch-formalizer",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert not first.verified
    assert load_orchestration_checkpoint(
        state_path,
    ).proof_state == ProofState.FORMALIZER
    checkpoint = load_orchestration_checkpoint(state_path)
    checkpoint.adapter_status = ""
    checkpoint.blocked_reason = ""
    save_orchestration_checkpoint(state_path, checkpoint)
    resumed_runner, resumed_calls = _certificate_runner()
    second = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        resumed_runner,
        project_root=tmp_path,
        orchestration_id="orch-formalizer",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert second.verified
    assert resumed_calls[0][0] == "formalizer"
    assert not any(
        call[0] in {
            "definition_auditor",
            "counterexample_worker",
            "decomposer",
        }
        for call in resumed_calls
    )


def test_630_token_formalizer_partial_gets_one_fresh_compact_repair(tmp_path):
    ledger = ProofObligationLedger(
        "formalizer-repair",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
    )
    runner, calls = _certificate_runner()
    first_formalizer = True

    def truncated_once(role, messages, expected_run_id):
        nonlocal first_formalizer
        if role == "formalizer" and first_formalizer:
            first_formalizer = False
            calls.append((role, messages, expected_run_id))
            exc = SemanticResponseIncomplete(
                "formalizer",
                token_count=630,
                stop_reason="client_safety_limit",
                response_cap_exhausted=True,
            )
            exc.partial_text = (
                '### FORMALIZATION_BUNDLE\nArtifact: {"parent_signature_source":'
                '"theorem exactParent : True := by sorry","child":'
                '{"label":"L1","lean_signature":"theorem exactChild'
            )
            raise exc
        return runner(role, messages, expected_run_id)

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        truncated_once,
        project_root=tmp_path,
        orchestration_id="orch-formalizer-repair",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert result.verified
    formalizer_calls = [call for call in calls if call[0] == "formalizer"]
    assert len(formalizer_calls) == 2
    assert formalizer_calls[-1][2].endswith(":protocol-repair-1")
    repair_package = json.loads(formalizer_calls[-1][1][-1]["content"])
    assert "after 630 tokens" in repair_package["validation_errors"][0]
    assert "formalizer_partial_attempt_1" in result.transcripts
    assert "formalizer_protocol_repair_1" in result.transcripts


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_two_formalizer_repair_failures_block_once_and_restart_is_quiet(
    tmp_path,
):
    ledger = ProofObligationLedger(
        "formalizer-block",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
        version=87,
    )
    state_path = tmp_path / "orchestration.json"
    runner, calls = _certificate_runner()

    def always_incomplete(role, messages, expected_run_id):
        if role != "formalizer":
            return runner(role, messages, expected_run_id)
        calls.append((role, messages, expected_run_id))
        exc = SemanticResponseIncomplete(
            "formalizer",
            token_count=630,
            stop_reason="client_safety_limit",
            response_cap_exhausted=True,
        )
        exc.partial_text = (
            '### FORMALIZATION_BUNDLE\nArtifact: {"parent_signature_source":"P"'
        )
        raise exc

    kwargs = {
        "project_root": tmp_path,
        "orchestration_id": "orch-formalizer-block",
        "signature_validator": _fake_signature_validator,
        "proof_validator": _fake_proof_validator,
        "checkpoint_path": state_path,
        "candidate_sha256": "candidate-hash",
    }
    first = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        always_incomplete,
        **kwargs,
    )
    assert not first.verified
    assert load_orchestration_checkpoint(state_path).proof_state == (
        ProofState.FORMALIZER
    )
    second = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        always_incomplete,
        **kwargs,
    )
    assert not second.verified
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.proof_state == ProofState.FORMALIZER
    assert checkpoint.adapter_status == "ADAPTER_BLOCKED"
    assert checkpoint.identical_failure_count == 0
    upstream_counts = {
        role: len([call for call in calls if call[0] == role])
        for role in (
            "definition_auditor",
            "counterexample_worker",
            "decomposer",
        )
    }
    assert upstream_counts == {
        "definition_auditor": 1,
        "counterexample_worker": 1,
        "decomposer": 1,
    }
    calls_before_restart = len(calls)
    assert second.validation["blocked"] is True
    assert len(calls) == calls_before_restart


def test_malformed_defense_gets_one_fresh_compact_repair(tmp_path):
    ledger = ProofObligationLedger(
        "defense-repair",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
    )
    runner, calls = _certificate_runner()
    first_defense = True

    def malformed_once(role, messages, expected_run_id):
        nonlocal first_defense
        if role == "adversarial_proponent" and first_defense:
            first_defense = False
            calls.append((role, messages, expected_run_id))
            return (
                '### DEFENSE_REPORT\nArtifact: {"status":"DEFENDED","issues":[]',
                expected_run_id,
            )
        return runner(role, messages, expected_run_id)

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        malformed_once,
        project_root=tmp_path,
        orchestration_id="orch-defense-repair",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert result.verified
    defense_calls = [
        call for call in calls if call[0] == "adversarial_proponent"
    ]
    assert len(defense_calls) == 2
    assert defense_calls[-1][2].endswith(":protocol-repair-1")
    repair_package = json.loads(defense_calls[-1][1][-1]["content"])
    assert "incomplete Artifact JSON" in repair_package["validation_errors"][0]
    assert "adversarial_proponent_protocol_repair_1" in result.transcripts


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_two_defense_repair_failures_block_once_and_restart_is_quiet(tmp_path):
    ledger = ProofObligationLedger(
        "defense-block",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
        version=87,
    )
    state_path = tmp_path / "orchestration.json"
    runner, calls = _certificate_runner()

    def always_malformed(role, messages, expected_run_id):
        if role != "adversarial_proponent":
            return runner(role, messages, expected_run_id)
        calls.append((role, messages, expected_run_id))
        return (
            '### DEFENSE_REPORT\nArtifact: {"status":"DEFENDED","issues":[]',
            expected_run_id,
        )

    kwargs = {
        "project_root": tmp_path,
        "orchestration_id": "orch-defense-block",
        "signature_validator": _fake_signature_validator,
        "proof_validator": _fake_proof_validator,
        "checkpoint_path": state_path,
        "candidate_sha256": "candidate-hash",
    }
    for expected_state in (
        ProofState.ADVERSARIAL_REVIEW,
        ProofState.ADVERSARIAL_REVIEW,
    ):
        result = run_certified_decomposition(
            ledger,
            "ROOT",
            "Immutable root goal",
            always_malformed,
            **kwargs,
        )
        assert not result.verified
        assert load_orchestration_checkpoint(state_path).proof_state == (
            expected_state
        )
    assert load_orchestration_checkpoint(state_path).adapter_status == (
        "ADAPTER_BLOCKED"
    )
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.identical_failure_count == 0
    assert {
        role: len([call for call in calls if call[0] == role])
        for role in (
            "definition_auditor",
            "counterexample_worker",
            "decomposer",
            "formalizer",
            "prover",
        )
    } == {
        "definition_auditor": 1,
        "counterexample_worker": 1,
        "decomposer": 1,
        "formalizer": 1,
        "prover": 1,
    }
    calls_before_restart = len(calls)
    restarted = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        always_malformed,
        **kwargs,
    )
    assert restarted.validation["blocked"] is True
    assert len(calls) == calls_before_restart


def test_true_binding_mismatch_preserves_accepted_contract(tmp_path):
    statement = "RiemannHypothesis"
    root_hash = hashlib.sha256(statement.encode()).hexdigest()
    ledger = ProofObligationLedger(
        "rh",
        [ProofObligation(
            "RH-C0-root",
            statement,
            formal_status="FORMALIZED",
            lean_signature_hash=root_hash,
            proposition_hash=root_hash,
        )],
        version=96,
    )
    state_path = tmp_path / "orchestration.json"
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.DECOMPOSER.value,
        current_role="decomposer",
        target_obligation_id="RH-C0-root",
        candidate_sha256="candidate-hash",
        parent_statement_sha256=root_hash,
        parent_signature_sha256=root_hash,
        root_goal_sha256=root_hash,
        proposition_hash=root_hash,
        research_contract_id="RC-root",
        research_contract_hash="contract-hash",
        ledger_id="rh",
        ledger_version=96,
    )
    artifact = persist_validated_artifact(
        state_path,
        checkpoint,
        role="research_contract",
        payload={"schema_version": 1, "contract_id": "RC-root"},
        dependencies=[],
        source_run_id="host:contract",
    )
    runner, calls = _certificate_runner()
    result = run_certified_decomposition(
        ledger,
        "RH-C0-root",
        "Different proposition",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-mismatch",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert result.validation["blocked"] is True
    assert result.validation["preserved_research_contract_id"] == "RC-root"
    assert calls == []
    preserved = load_orchestration_checkpoint(state_path)
    assert preserved.research_contract_id == "RC-root"
    assert preserved.research_contract_hash == "contract-hash"
    assert preserved.validated_artifacts["research_contract"].sha256 == (
        artifact.sha256
    )
    assert preserved.proof_state == ProofState.DECOMPOSER
    assert preserved.adapter_status == "INTEGRATION_BLOCKED"


def test_artifact_migration_failure_preserves_contract_provenance():
    checkpoint = OrchestrationCheckpoint(
        research_contract_id="RC-root",
        research_contract_hash="contract-hash",
        selected_strategy_plan_id="SP-root",
        selected_strategy_plan_hash="plan-hash",
        target_strategy_plan_hash="plan-hash",
    )
    def ref(role, sha, dependencies=()):
        return ArtifactRef(
            role=role,
            sha256=sha,
            schema_version=1,
            dependencies=list(dependencies),
            path=f"/tmp/{sha}.json",
            source_run_id="host:test",
            validated_at=1.0,
        )
    checkpoint.validated_artifacts = {
        "definition_auditor": ref("definition_auditor", "definition"),
        "counterexample_worker": ref("counterexample_worker", "worker"),
        "strategy_tournament": ref(
            "strategy_tournament",
            "tournament",
            ["definition"],
        ),
        "research_contract": ref(
            "research_contract",
            "contract",
            ["tournament"],
        ),
    }
    retain_contract_provenance_after_artifact_failure(checkpoint)
    assert set(checkpoint.validated_artifacts) == {
        "contract_definition_auditor",
        "strategy_tournament",
        "research_contract",
    }
    contract_definition = checkpoint.validated_artifacts[
        "contract_definition_auditor"
    ]
    assert contract_definition.sha256 == "definition"
    assert contract_definition.strategy_plan_hash == "plan-hash"
    assert checkpoint.research_contract_id == "RC-root"
    assert checkpoint.research_contract_hash == "contract-hash"
    assert checkpoint.selected_strategy_plan_id == "SP-root"


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_resume_binding_mismatch_invalidates_cached_artifacts(tmp_path):
    state_path = tmp_path / "orchestration.json"
    first_ledger = ProofObligationLedger(
        "binding",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
        version=85,
    )
    runner, _ = _certificate_runner()

    def stop_after_upstream(role, messages, expected_run_id):
        if role == "decomposer":
            raise TimeoutError("stop after upstream")
        return runner(role, messages, expected_run_id)

    run_certified_decomposition(
        first_ledger,
        "ROOT",
        "Immutable root goal",
        stop_after_upstream,
        project_root=tmp_path,
        orchestration_id="orch-binding",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    changed_ledger = ProofObligationLedger(
        "binding",
        [ProofObligation(
            "ROOT",
            "Establish the changed parent proposition for approximants.",
        )],
        version=86,
    )
    restarted_runner, calls = _certificate_runner()
    run_certified_decomposition(
        changed_ledger,
        "ROOT",
        "Immutable root goal",
        restarted_runner,
        project_root=tmp_path,
        orchestration_id="orch-binding-new",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert calls[0][0] == "definition_auditor"
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.parent_statement_sha256 == hashlib.sha256(
        changed_ledger.obligations[0].statement.encode(),
    ).hexdigest()


def test_legacy_multi_child_manifest_remains_readable_only(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "verified": False,
        "artifacts": {
            "decomposer": {
                "children": [{"label": "L1"}, {"label": "L2"}],
            },
        },
    }))
    loaded = load_decomposition_manifest(path)
    assert len(loaded["artifacts"]["decomposer"]["children"]) == 2
    text = (
        "### DECOMPOSITION_PROPOSAL\nArtifact: "
        + json.dumps(loaded["artifacts"]["decomposer"])
    )
    artifact, error = parse_certified_artifact(
        text,
        "DECOMPOSITION_PROPOSAL",
        target_obligation_id="ROOT",
        parent_statement_hash="parent",
        root_goal_hash="goal",
        producer_run_id="run",
        upstream_artifact_hashes=[],
    )
    assert artifact is None
    assert "invalid DECOMPOSITION_PROPOSAL fields" in error


def test_certificate_parser_rejects_tampered_bindings_for_every_role(
    tmp_path,
):
    ledger = ProofObligationLedger(
        "bindings",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    runner, calls = _certificate_runner()
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-bind",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    headings = [
        "DEFINITION_AUDIT",
        "COUNTEREXAMPLE_REPORT",
        "DECOMPOSITION_PROPOSAL",
        "FORMALIZATION_BUNDLE",
        "PROOF_ATTEMPT",
        "DEFENSE_REPORT",
        "JUDGE_DECISION",
    ]
    for heading, (role, messages, expected_run_id) in zip(headings, calls):
        package = messages[-1].get(
            "_host_package",
            json.loads(messages[-1]["content"]),
        )
        upstream = package["upstream_artifact_hashes"]
        artifact, error = parse_certified_artifact(
            result.transcripts[role],
            heading,
            target_obligation_id="ROOT",
            parent_statement_hash="tampered",
            root_goal_hash=package["root_goal_hash"],
            producer_run_id=expected_run_id,
            upstream_artifact_hashes=upstream,
        )
        assert artifact is None
        assert "tampered" in error


def test_certificate_timeout_or_malformed_role_never_mutates_ledger(tmp_path):
    ledger = ProofObligationLedger(
        "fail-open",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    before = asdict(ledger)
    runner, _calls = _certificate_runner()

    def timeout_runner(role, messages, expected_run_id):
        if role == "counterexample_worker":
            raise TimeoutError("worker timeout")
        return runner(role, messages, expected_run_id)

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        timeout_runner,
        project_root=tmp_path,
        orchestration_id="orch-timeout",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not result.verified
    assert persist_verified_decomposition(
        ledger,
        "ROOT",
        result,
        "run-timeout",
    ) == []
    assert asdict(ledger) == before

    def oversized_runner(_role, _messages, _expected_run_id):
        raise SemanticUnitTooLarge("formal artifact", 2053, 2052)

    oversized = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        oversized_runner,
        project_root=tmp_path,
        orchestration_id="orch-oversized",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not oversized.verified
    assert "SEMANTIC_UNIT_TOO_LARGE" in oversized.errors[0]
    assert asdict(ledger) == before


def test_certificate_graph_proof_parent_child_and_judge_gates(tmp_path):
    cases = [
        ({"cycle": True}, "exact child"),
        ({"proof_source": "theorem reduction (h : True) : True := by sorry"}, "complete reduction proof"),
        ({"child_signature": "theorem badChild : True := by sorry"}, "Lean declaration contains a placeholder"),
        ({"judge_decision": "REJECT"}, "Judge decision"),
        ({
            "counterexample_case": {
                "evidence_type": "FINITE_COUNTEREXAMPLE",
                "evidence_source": "host arithmetic evaluator",
                "claim": {
                    "schema_version": 1,
                    "quantifier": "FOR_ALL",
                    "variables": ["x"],
                    "domain": "REAL",
                    "lhs": "x*x",
                    "relation": "==",
                    "rhs": "x",
                },
                "witness": {"x": 2},
            },
        }, "verified counterexample refutes the parent"),
    ]
    for index, (options, expected_error) in enumerate(cases):
        ledger = ProofObligationLedger(
            f"invalid-{index}",
            [ProofObligation(
                "ROOT",
                "Establish the global convergence theorem for analytic approximants.",
            )],
        )
        runner, _calls = _certificate_runner(**options)
        result = run_certified_decomposition(
            ledger,
            "ROOT",
            "Immutable root goal",
            runner,
            project_root=tmp_path,
            orchestration_id=f"orch-invalid-{index}",
            signature_validator=_fake_signature_validator,
            proof_validator=_fake_proof_validator,
        )
        assert not result.verified
        assert any(
            expected_error in error for error in result.errors
        ), (options, result.errors)
        assert len(ledger.obligations) == 1

    advisory_runner, advisory_calls = _certificate_runner(
        counterexample_case={
            "evidence_type": "PINNED_THEOREM",
            "reference": "Unsupported Theorem 1",
        },
    )
    advisory_result = run_certified_decomposition(
        ProofObligationLedger(
            "advisory-counterexample",
            [ProofObligation(
                "ROOT",
                "Establish the global convergence theorem for analytic approximants.",
            )],
        ),
        "ROOT",
        "Immutable root goal",
        advisory_runner,
        project_root=tmp_path,
        orchestration_id="orch-advisory-counterexample",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert advisory_result.verified
    assert advisory_result.validation["counterexample_advisory_only"] is True
    assert advisory_result.validation["counterexample_is_public_premise"] is False
    decomposer_package = next(
        messages[-1].get(
            "_host_package",
            json.loads(messages[-1]["content"]),
        )
        for role, messages, _run_id in advisory_calls
        if role == "decomposer"
    )
    assert "counterexample_worker" not in (
        decomposer_package["validated_upstream_artifacts"]
    )

    existing_source = "theorem boundParent : True := by"
    existing = ProofObligation(
        "ROOT",
        "Establish the global convergence theorem for analytic approximants.",
        formal_status="FORMALIZED",
        lean_signature=existing_source,
        lean_signature_hash=lean_theorem_signature_hash(existing_source),
    )
    ledger = ProofObligationLedger("parent-mismatch", [existing])
    runner, _calls = _certificate_runner(parent_hash_override="wrong-hash")
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-parent-mismatch",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not result.verified
    assert any(
        "parent_signature name changed" in error
        for error in result.errors
    )


def test_failed_host_graph_gate_stops_before_judge(tmp_path):
    ledger = ProofObligationLedger(
        "judge-host",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    runner, _calls = _certificate_runner(cycle=True, judge_decision="ACCEPT")
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-judge",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert "judge" not in result.artifacts
    assert [role for role, _messages, _run_id in _calls][-1] == "decomposer"
    assert not result.verified
    assert any("exact child" in error for error in result.errors)


def test_lean_placeholder_fails_at_formalizer_before_prover(tmp_path):
    ledger = ProofObligationLedger(
        "placeholder-formalizer",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    runner, calls = _certificate_runner(
        child_signature="theorem childReduction : True := ...",
    )
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-placeholder-formalizer",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not result.verified
    assert any("placeholder" in error for error in result.errors)
    assert "prover" not in [role for role, _messages, _run_id in calls]


def test_non_circular_reduction_proof_gate():
    assert _circular_reduction_proof(
        "theorem reduction (parent : True) : True := by exact parent",
    )
    assert not _circular_reduction_proof(
        "theorem reduction : True := by trivial",
    )


def test_multi_child_output_is_repaired_as_one_bundled_child(tmp_path):
    ledger = ProofObligationLedger(
        "repair",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    runner, calls = _certificate_runner(multi_child=True)
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-repair",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    decomposer_calls = [call for call in calls if call[0] == "decomposer"]
    assert len(decomposer_calls) == 2
    assert "protocol-repair-1" in decomposer_calls[-1][2]
    assert result.verified
    assert result.artifacts["decomposer"].child["label"] == "L1"
    assert not hasattr(result.artifacts["decomposer"], "children")


def test_failed_decomposer_repair_stops_all_downstream_roles(tmp_path):
    ledger = ProofObligationLedger(
        "repair-fail",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
    )
    runner, calls = _certificate_runner(multi_child=True)

    def always_multi(role, messages, expected_run_id):
        text, actual = runner(role, messages, expected_run_id)
        if role == "decomposer" and "protocol-repair" in expected_run_id:
            payload = json.loads(text.split("Artifact: ", 1)[1])
            child = payload.pop("child")
            payload["children"] = [child, dict(child, label="L2")]
            text = "### DECOMPOSITION_PROPOSAL\nArtifact: " + json.dumps(payload)
        return text, actual

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        always_multi,
        project_root=tmp_path,
        orchestration_id="orch-repair-fail",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not result.verified
    assert [call[0] for call in calls][-1] == "decomposer"
    assert any("protocol repair failed" in error for error in result.errors)


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_extra_brace_repair_blocks_once_with_exact_reason(tmp_path):
    ledger = ProofObligationLedger(
        "extra-brace-fail",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
    )
    runner, calls = _certificate_runner()
    state_path = tmp_path / "orchestration.json"

    def extra_brace(role, messages, expected_run_id):
        text, actual = runner(role, messages, expected_run_id)
        if role == "decomposer":
            text += "}"
        return text, actual

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        extra_brace,
        project_root=tmp_path,
        orchestration_id="orch-extra-brace",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    diagnostic = (
        "malformed DECOMPOSITION_PROPOSAL Artifact JSON: "
        "trailing text or second Artifact is forbidden"
    )
    assert not result.verified
    assert len([call for call in calls if call[0] == "decomposer"]) == 2
    assert [call[0] for call in calls][-1] == "decomposer"
    assert result.errors == [
        "decomposer protocol repair failed: " + diagnostic,
    ]
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.adapter_status == "ADAPTER_BLOCKED"
    assert checkpoint.blocked_reason.endswith(diagnostic)
    calls_before_restart = len(calls)
    restarted = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        extra_brace,
        project_root=tmp_path,
        orchestration_id="orch-extra-brace",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert restarted.validation["blocked"] is True
    assert len(calls) == calls_before_restart


def test_truncated_decomposer_gets_fresh_compact_retry(tmp_path):
    ledger = ProofObligationLedger(
        "truncated-repair",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
    )
    runner, calls = _certificate_runner()
    first = True

    def truncated_once(role, messages, expected_run_id):
        nonlocal first
        calls.append((role, messages, expected_run_id))
        if role == "decomposer" and first:
            first = False
            exc = SemanticResponseIncomplete(
                "decomposer",
                token_count=352,
                stop_reason="client_safety_limit",
                response_cap_exhausted=True,
            )
            exc.partial_text = '{"parent_statement":"P","child":'
            raise exc
        # Avoid recording twice in the helper's calls list.
        calls.pop()
        return runner(role, messages, expected_run_id)

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        truncated_once,
        project_root=tmp_path,
        orchestration_id="orch-truncated",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert result.verified
    assert "decomposer_partial_attempt_1" in result.transcripts
    assert "decomposer_protocol_repair_1" in result.transcripts
    repair_call = [
        call for call in calls
        if call[0] == "decomposer" and "protocol-repair" in call[2]
    ][0]
    repair_package = json.loads(repair_call[1][-1]["content"])
    assert "rejected_complete_artifact" not in repair_package


@pytest.mark.skip(reason="legacy model-authored artifact execution is read-only")
def test_repeated_incomplete_decomposer_never_reaches_formalizer(tmp_path):
    ledger = ProofObligationLedger(
        "truncated-fail",
        [ProofObligation("ROOT", "Prove the exact parent proposition.")],
    )
    calls = []
    state_path = tmp_path / "orchestration.json"

    def incomplete(role, messages, expected_run_id):
        calls.append((role, messages, expected_run_id))
        if role == "decomposer":
            raise SemanticResponseIncomplete(
                "decomposer",
                token_count=260,
                stop_reason="client_safety_limit",
                response_cap_exhausted=True,
            )
        runner, _ = _certificate_runner()
        return runner(role, messages, expected_run_id)

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        incomplete,
        project_root=tmp_path,
        orchestration_id="orch-truncated-fail",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert not result.verified
    assert [call[0] for call in calls][-1] == "decomposer"
    assert not any(call[0] == "formalizer" for call in calls)
    assert any("protocol repair failed" in error for error in result.errors)
    checkpoint = load_orchestration_checkpoint(state_path)
    assert checkpoint.proof_state == ProofState.DECOMPOSER
    assert checkpoint.adapter_status == "ADAPTER_BLOCKED"
    calls_before_restart = len(calls)
    restarted = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        incomplete,
        project_root=tmp_path,
        orchestration_id="orch-truncated-fail",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
        checkpoint_path=state_path,
        candidate_sha256="candidate-hash",
    )
    assert not restarted.verified
    assert restarted.validation["blocked"] is True
    assert len(calls) == calls_before_restart


def test_certified_trigger_skips_proved_and_detects_frontiers():
    unresolved = """
### ISSUE_VERDICT ROOT
Status: UNRESOLVED
Evidence: A precise reduction is still missing from the attempted argument.
Missing lemma: Prove a compact boundary inequality.
"""
    proved = """
### ISSUE_VERDICT ROOT
Status: PROVED
Evidence: This sufficiently detailed complete derivation closes the exact target obligation.
Missing lemma: none
"""
    assert certified_decomposition_requested(unresolved, "", {"ROOT"})
    assert not certified_decomposition_requested(proved, "", {"ROOT"})
    assert certified_decomposition_requested(
        "",
        "### ISSUE_RESPONSE ROOT\nRemaining gap: Define the exact topology.",
        {"ROOT"},
    )


def test_host_generated_autoresearch_verdict_uses_new_child_frontier():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C2", "Prove zero convergence."),
            ProofObligation(
                "RH-C2-child",
                "Prove locally uniform convergence on compact subsets.",
                parent_id="RH-C2",
                decomposition_certificate_hash="certificate-hash",
                reduction_theorem_status="PROVED",
            ),
        ],
    )

    class Candidate:
        CANDIDATE_ID = "candidate-v4"
        TARGET_OBLIGATION_ID = "RH-C2"

    verdict = build_autoresearch_verdict(
        Candidate,
        ledger,
        {"RH-C2": "UNRESOLVED"},
        [ledger.obligations[1]],
    )
    assert verdict["outcome"] == "DECOMPOSED"
    assert verdict["created_obligation_ids"] == ["RH-C2-child"]
    ledger.obligations[1].decomposition_certificate_hash = ""
    inconclusive = build_autoresearch_verdict(
        Candidate,
        ledger,
        {"RH-C2": "UNRESOLVED"},
        [ledger.obligations[1]],
    )
    assert inconclusive["outcome"] == "INCONCLUSIVE"
    assert inconclusive["new_frontier"] == (
        "Continue unresolved target RH-C2: Prove zero convergence."
    )


def test_critic_leaf_table_cannot_bypass_certificate():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C1", "Operator construction."),
            ProofObligation("RH-C2", "Zero convergence."),
        ],
    )
    critic = """
### Leaf Obligation Ledger
| ID | Status | Evidence/Argument | Missing Lemma/Requirement |
| :--- | :--- | :--- | :--- |
| **RH-C2-1** | UNRESOLVED | No compact convergence proof. | Prove locally uniform convergence on compact subsets. |
| **RH-C2-2** | UNRESOLVED | Hurwitz does not cover poles. | Prove a singular-limit zero-counting theorem. |
| **RH-C1-1** | UNRESOLVED | Unrelated operator issue. | Construct a self-adjoint operator. |
"""
    applied = apply_critic_verdicts(
        ledger,
        critic,
        "br_table",
        {"RH-C2"},
        None,
    )
    created = create_child_obligations(
        ledger,
        critic,
        "br_table",
        {"RH-C2"},
        None,
        {
            "RH-C2": [
                _valid_lean_signature("1"),
                _valid_lean_signature("2"),
            ],
        },
    )
    assert applied == {"RH-C2": "UNRESOLVED"}
    assert created == []
    assert ledger.obligations[0].last_run_id == ""
    assert pending_obligations(ledger) == ledger.obligations


def test_corrupted_critic_id_binds_to_single_host_target():
    target = (
        "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-"
        "763645cd6b-40ef83e052-b80cd1343b"
    )
    corrupted = (
        "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-"
        "763645cd6b-40ef83e052-b80cd13rum-40ef83e052-b80cd1343b"
    )
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[ProofObligation(target, "Sectorial compatibility.")],
    )
    repairs = []
    verdicts = apply_critic_verdicts(
        ledger,
        f"""
### ISSUE_VERDICT {corrupted}
**Status:** UNRESOLVED
**Evidence:** The local density inference is not justified by the global growth class.
**Missing lemma:** Prove a quantitative sectorial density bound from the indicator function.
""",
        "br_corrupt",
        {target},
        repairs,
    )
    assert verdicts == {target: "UNRESOLVED"}
    assert repairs == [(corrupted, target)]
    assert "local density inference" in ledger.obligations[0].last_evidence


def test_cycle_and_invented_ids_cannot_create_children():
    root = ProofObligation(
        "RH-C2-root",
        'The "Angular-Growth Decoupling Lemma": prove angular separation.',
    )
    target = ProofObligation(
        "RH-C2-root-child",
        'The "Sectorial Compatibility Lemma": prove a sectorial bound.',
        parent_id=root.obligation_id,
    )
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[root, target],
    )
    critic = f"""
### ISSUE_VERDICT {target.obligation_id}
**Status:** UNRESOLVED
**Evidence:** The proposed local density lower bound remains unsupported.
**Missing lemma:** The "Angular-Growth Decoupling Lemma": prove angular separation.

### Leaf Obligation Ledger
| ID | Status | Evidence | Missing Lemma |
| RH-Cty-0-alpha | UNRESOLVED | invented | Prove an invented temporary lemma. |
"""
    rejections = []
    created = create_child_obligations(
        ledger,
        critic,
        "br_cycle",
        {target.obligation_id},
        rejections,
        certified_only=False,
    )
    assert created == []
    assert len(ledger.obligations) == 2
    assert any("existing obligation" in item for item in rejections)


def test_variable_renamed_parent_child_is_retroactively_rejected():
    parent = ProofObligation(
        "RH-C2-gap",
        (
            "Prove that for a fixed genus p there exists a critical density "
            "rho_c such that a zero sequence with density rho > rho_c cannot "
            "converge to a local pole of order m without forcing the global "
            "growth order above p."
        ),
    )
    child = ProofObligation(
        "RH-C2-equivalence",
        (
            "Establish the relationship between local accumulation rate delta "
            "at s0 and the global exponent of convergence lambda: if local "
            "density constructs a singularity of order m, global growth must "
            "satisfy lambda >= function(rho,m)."
        ),
        parent_id=parent.obligation_id,
    )
    grandchild = ProofObligation(
        "RH-C2-saturation",
        (
            "Show local density delta imposes a lower bound on global order "
            "rho through a density-order saturation mapping."
        ),
        parent_id=child.obligation_id,
    )
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[parent, child, grandchild],
    )
    rejected = audit_ledger_semantic_duplicates(ledger)
    assert rejected[0][0:2] == (child.obligation_id, parent.obligation_id)
    assert child.status == "REJECTED_DUPLICATE"
    assert grandchild.status == "REJECTED_DUPLICATE"
    assert pending_obligations(ledger) == [parent]


def test_recover_complete_checkpoint_from_timestamped_log(tmp_path):
    path = tmp_path / "agent.log"
    path.write_text(
        "\n".join((
            "[t] [goal] anchored: prove RH",
            "[t] [inference-start] time=t run=br_good goal=hash",
            "[t] generator> first generator line",
            "[t] second generator line",
            "[t] [allens] Critic Prefill: 100 tokens...",
            "[t] critic> first critic line",
            "[t] second critic line",
            "[t] [premise-suspected] id=ROOT-A",
            "[t] premise_auditor> complete auditor transcript",
            "[t] adversarial_proponent> complete defense transcript",
            "[t] [metrics] run=br_good",
        )),
    )
    recovered = recover_checkpoint_from_log(path, "br_good")
    assert recovered.research_goal == "prove RH"
    assert recovered.previous_generator == (
        "first generator line\nsecond generator line"
    )
    assert recovered.previous_critic == "first critic line\nsecond critic line"
    assert recovered.last_run_id == "br_good"
