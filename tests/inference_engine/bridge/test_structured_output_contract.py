import json

import pytest

from autoresearch.prefill.semantic_decompose import (
    STRUCTURED_ARTIFACT_CONTRACTS,
    lint_structured_prompt,
    scan_single_artifact_object,
    structured_transport_complete,
)


EXPECTED_STRUCTURED_ROLES = {
    "strategy",
    "premise_suspicion",
    "premise_auditor",
    "premise_proponent",
    "definition_auditor",
    "counterexample_worker",
    "decomposer",
    "formalizer",
    "formalizer_parent_signature",
    "formalizer_child_signature",
    "formalizer_reduction_signature",
    "prover",
    "adversarial_proponent",
    "judge",
}


def _value(field):
    if field in {
        "definitions",
        "missing_definitions",
        "cases",
        "public_assumptions",
        "issues",
        "repairs",
    }:
        return []
    if field in {"child", "reduction_contract", "artifact", "claim"}:
        return {}
    if field in {"parent_newly_formalized"}:
        return True
    if field in {"confidence"}:
        return 0.5
    if field in {"prefill_compute_chunk_tokens"}:
        return 256
    return f"value-for-{field}"


def _closure_fixture(role):
    contract = STRUCTURED_ARTIFACT_CONTRACTS[role]
    payload = {
        field: _value(field)
        for field in contract.required_fields
    }
    artifact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if contract.heading is None:
        if contract.marker:
            return f"{contract.marker} {artifact}"
        return artifact
    return f"### {contract.heading}\n{contract.marker} {artifact}"


CLOSURE_FIXTURES = {
    role: _closure_fixture(role)
    for role in EXPECTED_STRUCTURED_ROLES
}


def test_every_structured_role_has_registered_closure_fixture():
    assert set(STRUCTURED_ARTIFACT_CONTRACTS) == EXPECTED_STRUCTURED_ROLES
    assert set(CLOSURE_FIXTURES) == EXPECTED_STRUCTURED_ROLES
    for role, fixture in CLOSURE_FIXTURES.items():
        contract = STRUCTURED_ARTIFACT_CONTRACTS[role]
        if contract.transport_stop:
            assert structured_transport_complete(fixture, role)


def test_valid_nested_json_handles_latex_braces_quotes_and_backslashes():
    text = (
        '### FORMALIZATION_BUNDLE\nArtifact: {"nested":[{"latex":'
        '"\\\\{x\\\\} and } with \\"quote\\" and \\\\\\\\","array":[1,{"x":2}]}]}'
    )
    assert structured_transport_complete(text, "formalizer")
    scanned = scan_single_artifact_object(text)
    assert json.loads(scanned.json_text)["nested"][0]["array"][1] == {"x": 2}


def test_markdown_fence_never_becomes_transport_complete():
    fenced = "```json\n" + CLOSURE_FIXTURES["formalizer"] + "\n```"
    assert not structured_transport_complete(fenced, "formalizer")
    with pytest.raises(ValueError):
        scan_single_artifact_object(fenced)


def test_prompt_lint_fails_invalid_union_and_post_json_prose():
    with pytest.raises(ValueError, match="union placeholder"):
        lint_structured_prompt('Artifact: {"status":"GOOD|BAD"}')
    with pytest.raises(ValueError, match="prose after JSON"):
        lint_structured_prompt("Follow the JSON with prose after the JSON.")
    lint_structured_prompt(
        "Return compact JSON. Status must be GOOD or BAD. "
        "End immediately after the matching final }."
    )
