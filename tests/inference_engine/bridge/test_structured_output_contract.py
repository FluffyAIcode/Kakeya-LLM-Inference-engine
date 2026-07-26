import json

import pytest

from autoresearch.prefill.semantic_decompose import (
    STRUCTURED_ARTIFACT_CONTRACTS,
    lint_structured_prompt,
    scan_single_artifact_object,
    structured_transport_complete,
)
from scripts.agent_gan_inference_demo import _infer, decode_complete_response
from scripts.agent_gan_repl import parse_certified_artifact


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


class CharTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token) for token in token_ids)


class Session:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = 0
        self.last_stop_reason = None
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.exited = True

    def append(self, token_ids):
        self.appended = list(token_ids)

    def generate(self, *, max_tokens):
        tokens, reason = self.chunks[self.calls]
        self.calls += 1
        assert len(tokens) <= max_tokens
        yield from tokens
        self.last_stop_reason = reason


class Client:
    def __init__(self, session):
        self.session = session

    def create_session(self, **_kwargs):
        return self.session


def _stream(text, role, *, max_response_tokens=0, chunks=None):
    tokenizer = CharTokenizer()
    session = Session(chunks or [([ord(char) for char in text], 2)])
    tokens, metrics = _infer(
        Client(session),
        [],
        [9],
        max(1, len(text)),
        lambda: {},
        max_response_tokens=max_response_tokens,
        semantic_complete=lambda generated: structured_transport_complete(
            tokenizer.decode(generated),
            role,
        ),
    )
    return tokenizer.decode(tokens), metrics, session


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


def test_final_brace_split_across_chunks_stops_and_cleans_session():
    complete = CLOSURE_FIXTURES["prover"]
    partial, final = complete[:-1], complete[-1]
    text, metrics, session = _stream(
        complete,
        "prover",
        chunks=[
            ([ord(char) for char in partial], 1),
            ([ord(final), ord("x")], 2),
        ],
    )
    assert text == complete
    assert metrics["stop_reason"] == "semantic_complete"
    assert metrics["eos_reached"] is False
    assert metrics["response_cap_exhausted"] is False
    assert metrics["output_tokens"] == len(complete)
    assert session.exited


@pytest.mark.parametrize("suffix", [
    "}",
    "\ntrailing prose",
    '\nArtifact: {"replacement":true}',
    '\n{"second":"object"}',
])
def test_stream_cuts_suffix_but_strict_historical_parser_rejects(suffix):
    complete = CLOSURE_FIXTURES["decomposer"]
    streamed, metrics, session = _stream(
        complete + suffix,
        "decomposer",
    )
    assert streamed == complete
    assert metrics["stop_reason"] == "semantic_complete"
    assert session.exited
    with pytest.raises(
        ValueError,
        match="trailing text or second Artifact is forbidden|multiple Artifact markers",
    ):
        scan_single_artifact_object(complete + suffix)


def test_markdown_fence_never_becomes_transport_complete():
    fenced = "```json\n" + CLOSURE_FIXTURES["formalizer"] + "\n```"
    assert not structured_transport_complete(fenced, "formalizer")
    with pytest.raises(ValueError):
        scan_single_artifact_object(fenced)


def test_incomplete_json_at_cap_is_not_complete_or_no_eos_success():
    incomplete = CLOSURE_FIXTURES["judge"][:-1]
    streamed, metrics, session = _stream(
        incomplete,
        "judge",
        max_response_tokens=len(incomplete),
        chunks=[([ord(char) for char in incomplete], 1)],
    )
    assert streamed == incomplete
    assert metrics["stop_reason"] == "client_safety_limit"
    assert metrics["complete"] is False
    assert metrics["response_cap_exhausted"] is True
    assert session.exited
    with pytest.raises(ValueError, match="incomplete Artifact JSON"):
        scan_single_artifact_object(incomplete)
    with pytest.raises(Exception, match="SEMANTIC_RESPONSE_INCOMPLETE"):
        decode_complete_response(
            CharTokenizer(),
            "judge",
            [ord(char) for char in incomplete],
            metrics,
        )


def test_complete_json_before_eos_is_semantic_complete_not_no_eos():
    complete = CLOSURE_FIXTURES["judge"]
    streamed, metrics, session = _stream(
        complete + "tokens-model-would-have-appended",
        "judge",
    )
    assert streamed == complete
    assert metrics["stop_reason"] == "semantic_complete"
    assert metrics["complete"] is True
    assert metrics["eos_reached"] is False
    assert session.exited


def test_schema_invalid_first_object_is_not_replaced_by_valid_second():
    invalid = (
        '### FORMALIZATION_BUNDLE\nArtifact: {"schema_invalid":true}'
    )
    valid_second = CLOSURE_FIXTURES["formalizer"]
    streamed, metrics, _session = _stream(
        invalid + "\n" + valid_second,
        "formalizer",
    )
    assert streamed == invalid
    assert metrics["stop_reason"] == "semantic_complete"
    parsed, error = parse_certified_artifact(
        streamed,
        "FORMALIZATION_BUNDLE",
        target_obligation_id="ROOT",
        parent_statement_hash="parent",
        root_goal_hash="goal",
        producer_run_id="run:formalizer",
        upstream_artifact_hashes=["decomposer"],
    )
    assert parsed is None
    assert error.startswith("invalid FORMALIZATION_BUNDLE fields:")


def test_prompt_lint_fails_invalid_union_and_post_json_prose():
    with pytest.raises(ValueError, match="union placeholder"):
        lint_structured_prompt('Artifact: {"status":"GOOD|BAD"}')
    with pytest.raises(ValueError, match="prose after JSON"):
        lint_structured_prompt("Follow the JSON with prose after the JSON.")
    lint_structured_prompt(
        "Return compact JSON. Status must be GOOD or BAD. "
        "End immediately after the matching final }."
    )
