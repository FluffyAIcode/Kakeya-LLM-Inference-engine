from __future__ import annotations

import json

from autoresearch.prefill.candidate_representation import (
    MAPPER_CAPABILITY_VERSION,
    PRIMITIVE_CATALOG,
    RepresentationOutcome,
    analyze_private_candidate_representation,
    authorize_representation_retry,
    mapper_capability_hash,
    persist_representation_gap_report,
)
from autoresearch.prefill.orchestration_state import (
    ALLOWED_TRANSITIONS,
    CANDIDATE_REPRESENTATION_MIGRATION_EVENT,
    SCHEMA_VERSION,
    OrchestrationCheckpoint,
    ProofState,
    load_checkpoint,
    save_checkpoint,
)


NINE_CANDIDATE_FIXTURE = (
    ("XC-60bb98d69c92ace9cbcf", "DEFINITION",
     "An operator spectral measure and essential spectrum proposal.",
     {"DISCONNECTED_FROM_TARGET"}, {"SPECTRAL_MEASURE", "ESSENTIAL_SPECTRUM"}),
    ("XC-7dad81f1b910b31201d3", "LOCAL_LEMMA",
     "A zeta zero sequence proposal where the Montgomery Pair Correlation "
     "Conjecture holds and GUE statistics are used.",
     set(), {"ZETA_ZERO_SEQUENCE", "PAIR_CORRELATION_PREDICATE", "GUE_SPACING_LIMIT"}),
    ("XC-391aa29d047b771c349b", "CASE_SPLIT",
     "A stochastic PDE operator case split and spectral gap proposal.",
     {"DISCONNECTED_FROM_TARGET"}, {"SPECTRAL_GAP"}),
    ("XC-261076a02069094a733b", "SUFFICIENT_CONDITION",
     "For any L-function satisfying the Riemann Hypothesis, choose an analytic map.",
     set(), {"DIRICHLET_L_FUNCTION"}),
    ("XC-d3c0a173f7ff728d29a2", "EQUIVALENT_CRITERION",
     "A zeta zero sequence GUE condition equivalent to the assertion that no "
     "zeros lie outside the critical line.",
     {"UNSUPPORTED_EQUIVALENCE_TO_PARENT"}, {"ZETA_ZERO_SEQUENCE", "GUE_SPACING_LIMIT"}),
    ("XC-cd47dff5163aead9c7c2", "OBSTRUCTION_OR_COUNTEREXAMPLE",
     "Zeta eigenvalues correspond to zeros in a Selberg class, despite a "
     "functional equation.",
     set(), {"ZETA_ZERO_SEQUENCE", "SELBERG_CLASS"}),
    ("XC-2db4f96358712a89737b", "SPECIAL_CASE",
     "An operator spectral gap near a meromorphic pole.",
     {"DISCONNECTED_FROM_TARGET"}, {"SPECTRAL_GAP"}),
    ("XC-cb2458b81fd4fb8a35b0", "BRIDGE_THEOREM",
     r"Let zeta zeros be denoted by \rho_j = \frac{1}{2} + i\gamma_j.",
     {"PARENT_ASSUMED_IN_CANDIDATE"}, {"ZETA_ZERO_SEQUENCE"}),
    ("XC-57b7b90f4f843ec80d21", "TOY_MODEL_ANALOGUE",
     "A finite matrix rank-1 operator perturbation and spectral gap toy model.",
     {"DISCONNECTED_FROM_TARGET", "NO_CHILD_TO_PARENT_RELATION"},
     {"RANK_ONE_PERTURBATION", "SPECTRAL_GAP"}),
)


def _analyze(candidate_id: str, category: str, text: str):
    return analyze_private_candidate_representation(
        candidate_id=candidate_id,
        candidate_hash=("a" * 64),
        category=category,
        memo_text=text,
        target_obligation_id="RH-C0-7024d428ede1",
        target_context_hash="context",
        environment_hash="environment",
    )


def test_exact_nine_candidate_fixture_has_safe_gaps_not_falsehoods():
    reports = []
    for candidate_id, category, text, semantic, missing in NINE_CANDIDATE_FIXTURE:
        report = _analyze(candidate_id, category, text)
        reports.append(report)
        assert report.category == category
        assert semantic.issubset(report.semantic_issue_ids)
        assert missing.issubset(report.missing_primitive_ids)
        assert report.outcome == RepresentationOutcome.SEMANTIC_REJECTION.value
        assert not any(
            token in json.dumps(report.__dict__).casefold()
            for token in ("mathematically_false", "candidate_is_false")
        )
    assert len({report.report_id for report in reports}) == 9
    assert all(not report.private_text_exposed for report in reports)
    assert all(not report.ledger_mutation_allowed for report in reports)


def test_representation_routes_are_typed_and_prioritized():
    mapper = _analyze(
        "mapper", "LOCAL_LEMMA",
        "A critical line claim using a continuous complex map.",
    )
    assert mapper.outcome == RepresentationOutcome.MAPPER_EXTENSION_REQUIRED.value

    definition = _analyze(
        "definition", "LOCAL_LEMMA",
        "A zeta zero sequence local density claim.",
    )
    assert definition.outcome == RepresentationOutcome.DEFINITION_RESOLUTION.value

    hidden = _analyze(
        "hidden", "LOCAL_LEMMA",
        "A zeta claim where the Montgomery Pair Correlation Conjecture holds.",
    )
    assert hidden.outcome == RepresentationOutcome.SEMANTIC_REJECTION.value

    exhausted = _analyze(
        "exhausted", "LOCAL_LEMMA",
        "A critical line claim with no supported interpretation.",
    )
    assert exhausted.outcome == RepresentationOutcome.REPRESENTATION_EXHAUSTED.value


def test_retry_requires_changed_binding_and_is_consumed_once():
    report = _analyze(
        "mapper", "LOCAL_LEMMA",
        "A critical line claim using a continuous complex map.",
    )
    allowed, fingerprint, reason = authorize_representation_retry(
        report,
        mapper_registry_hash=report.mapper_registry_hash,
        environment_hash=report.environment_hash,
        consumed_fingerprints=(),
    )
    assert not allowed
    assert not fingerprint
    assert reason == "REPRESENTATION_RETRY_HASH_UNCHANGED"

    allowed, fingerprint, reason = authorize_representation_retry(
        report,
        mapper_registry_hash="changed",
        environment_hash=report.environment_hash,
        consumed_fingerprints=(),
    )
    assert allowed and fingerprint
    assert reason == "REPRESENTATION_RETRY_AUTHORIZED"
    replay = authorize_representation_retry(
        report,
        mapper_registry_hash="changed",
        environment_hash=report.environment_hash,
        consumed_fingerprints=(fingerprint,),
    )
    assert replay == (
        False, fingerprint, "REPRESENTATION_RETRY_ALREADY_CONSUMED",
    )


def test_report_persistence_is_private_text_free_and_idempotent(tmp_path):
    private_marker = "NEVER-PERSIST-PRIVATE-MEMO"
    report = _analyze(
        "private", "LOCAL_LEMMA",
        f"A critical line continuous complex map. {private_marker}",
    )
    first = persist_representation_gap_report(report, tmp_path)
    second = persist_representation_gap_report(report, tmp_path)
    assert first == second
    encoded = first.read_text()
    assert private_marker not in encoded
    assert json.loads(encoded)["private_text_exposed"] is False
    assert first.stat().st_mode & 0o077 == 0


def test_state_machine_inserts_analysis_before_rejection():
    assert ProofState.CANDIDATE_REPRESENTATION_ANALYSIS in (
        ALLOWED_TRANSITIONS[ProofState.CANDIDATE_FORMALIZATION]
    )
    assert ProofState.CANDIDATE_FORMALIZATION in (
        ALLOWED_TRANSITIONS[ProofState.CANDIDATE_REPRESENTATION_ANALYSIS]
    )
    assert ProofState.DEFINITION_RESOLUTION in (
        ALLOWED_TRANSITIONS[ProofState.CANDIDATE_REPRESENTATION_ANALYSIS]
    )


def test_mapper_catalog_is_closed_content_addressed_and_sourced():
    assert MAPPER_CAPABILITY_VERSION == 1
    assert len(mapper_capability_hash()) == 64
    assert all(item.primitive_id == key for key, item in PRIMITIVE_CATALOG.items())
    assert all(
        item.source_kind in {
            "LEAN_CORE", "MATHLIB", "HOST_REGISTRY", "NEW_DEFINITION",
        }
        for item in PRIMITIVE_CATALOG.values()
    )
    assert all(
        not item.trusted or bool(item.source_ref)
        for item in PRIMITIVE_CATALOG.values()
    )
    assert PRIMITIVE_CATALOG["CONTINUOUS_MAP"].source_ref == (
        "continuousComplexMap"
    )
    assert PRIMITIVE_CATALOG["HOLOMORPHIC_MAP"].source_ref == (
        "holomorphicComplexMapOn"
    )
    assert PRIMITIVE_CATALOG["FILTER_LIMIT"].source_ref == (
        "convergesComplexSequence"
    )


def test_mapper_capability_migration_keeps_old_reports_audit_only(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = OrchestrationCheckpoint(
        representation_report_refs={
            "XC-old": {
                "sha256": "a" * 64,
                "mapper_registry_hash": "old",
                "audit_only": True,
            },
        },
    )
    save_checkpoint(path, checkpoint)
    raw = json.loads(path.read_text())
    raw["schema_version"] = 12
    raw.pop("candidate_mapper_version")
    raw.pop("candidate_mapper_hash")
    path.write_text(json.dumps(raw))

    migrated = load_checkpoint(path)
    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.migration_event == CANDIDATE_REPRESENTATION_MIGRATION_EVENT
    assert migrated.candidate_mapper_hash == mapper_capability_hash()
    assert (
        migrated.representation_report_refs["XC-old"]["mapper_registry_hash"]
        == "old"
    )
    assert migrated.representation_report_refs["XC-old"]["audit_only"] is True
