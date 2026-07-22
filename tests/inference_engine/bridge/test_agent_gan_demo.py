from scripts.agent_gan_inference_demo import (
    _agent_cache_gate,
    _infer,
    _output_metadata,
    build_critic_context,
    decode_complete_response,
)
from autoresearch.prefill.semantic_decompose import (
    SemanticResponseIncomplete,
    SemanticUnitTooLarge,
)
import pytest


def test_agent_gate_accepts_remote_compute_or_exact_remote_cache_hit():
    warm = {
        "remote_jobs": 1,
        "remote_hits": 1,
        "tokens_reused": 10,
        "tokens_computed": 0,
        "fallbacks": 0,
        "remote_job_failures": 0,
    }
    actual = {
        "local_hits": 1,
        "remote_jobs": 0,
        "tokens_computed": 0,
        "fallbacks": 0,
    }
    assert _agent_cache_gate(warm, actual)
    assert _agent_cache_gate({**warm, "remote_jobs": 0}, actual)
    assert _agent_cache_gate(
        {**warm, "remote_hits": 0, "local_hits": 1, "remote_jobs": 0},
        actual,
    )
    assert not _agent_cache_gate({**warm, "remote_hits": 0}, actual)
    assert not _agent_cache_gate({**warm, "tokens_reused": 0}, actual)
    assert not _agent_cache_gate({**warm, "fallbacks": 1}, actual)
    assert not _agent_cache_gate({**warm, "remote_job_failures": 1}, actual)
    assert not _agent_cache_gate(warm, {**actual, "local_hits": 0})
    assert not _agent_cache_gate(warm, {**actual, "fallbacks": 1})


def test_agent_output_report_is_redacted():
    result = _output_metadata("private model output")
    assert result["output_chars"] == 20
    assert len(result["output_hash"]) == 64
    assert "output" not in result


class Session:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.last_stop_reason = None
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

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
        self.session_kwargs = []

    def create_session(self, **kwargs):
        self.session_kwargs.append(kwargs)
        return self.session


def test_infer_continues_chunks_until_eos():
    session = Session([([1, 2], 1), ([3], 2)])
    streamed = []
    tokens, metrics = _infer(
        Client(session),
        [],
        [9],
        2,
        lambda: {},
        on_token=lambda values: streamed.append(list(values)),
        max_response_tokens=0,
    )
    assert tokens == [1, 2, 3]
    assert streamed == [[1], [1, 2], [1, 2, 3]]
    assert metrics["stop_reason"] == "eos"
    assert metrics["complete"] is True


def test_infer_uses_explicit_isolated_role_session_label():
    client = Client(Session([([1], 2)]))
    _infer(
        client,
        [2],
        [9],
        1,
        lambda: {},
        client_label="agent-gan-premise-auditor",
    )
    assert client.session_kwargs == [{
        "eos_token_ids": [2],
        "client_label": "agent-gan-premise-auditor",
    }]


def test_infer_rejects_oversized_unit_before_session_append():
    client = Client(Session([([1], 2)]))
    with pytest.raises(SemanticUnitTooLarge):
        _infer(
            client,
            [],
            list(range(2053)),
            1,
            lambda: {},
            max_retained_tokens=2052,
        )
    assert client.session_kwargs == []


def test_seven_decomposition_roles_each_get_fresh_session():
    class FreshClient:
        def __init__(self):
            self.sessions = []

        def create_session(self, **kwargs):
            session = Session([([1], 2)])
            self.sessions.append((kwargs, session))
            return session

    roles = [
        "definition_auditor",
        "counterexample_worker",
        "decomposer",
        "formalizer",
        "prover",
        "adversarial_proponent",
        "judge",
    ]
    client = FreshClient()
    for role in roles:
        _infer(
            client,
            [],
            [9],
            1,
            lambda: {},
            client_label=f"agent-gan-{role}",
        )
    assert [item[0]["client_label"] for item in client.sessions] == [
        f"agent-gan-{role}" for role in roles
    ]
    assert len({id(item[1]) for item in client.sessions}) == 7


def test_infer_reports_explicit_client_safety_limit():
    tokens, metrics = _infer(
        Client(Session([([1, 2], 1)])),
        [],
        [9],
        2,
        lambda: {},
        max_response_tokens=2,
    )
    assert tokens == [1, 2]
    assert metrics["stop_reason"] == "client_safety_limit"
    assert metrics["complete"] is False
    assert metrics["eos_reached"] is False
    assert metrics["response_cap_exhausted"] is True


def test_infer_accepts_exact_eos_at_response_cap_without_slicing():
    tokens, metrics = _infer(
        Client(Session([([1, 2], 2)])),
        [],
        [9],
        2,
        lambda: {},
        max_response_tokens=2,
    )
    assert tokens == [1, 2]
    assert metrics["stop_reason"] == "eos"
    assert metrics["complete"] is True
    assert metrics["eos_reached"] is True
    assert metrics["response_cap_exhausted"] is False


def test_capped_generator_is_rejected_before_decode_or_critic_construction():
    class CountingTokenizer(CharTokenizer):
        def __init__(self):
            self.decode_calls = 0

        def decode(self, token_ids, **kwargs):
            self.decode_calls += 1
            return super().decode(token_ids, **kwargs)

    tokenizer = CountingTokenizer()
    critic_constructed = False
    tokens, metrics = _infer(
        Client(Session([([ord("I"), ord("f"), ord(" ")], 1)])),
        [],
        [9],
        3,
        lambda: {},
        max_response_tokens=3,
    )
    # Deliberately failing KV metrics prove semantic classification wins.
    metrics["delta"] = {"local_hits": 0, "fallbacks": 9}
    with pytest.raises(
        SemanticResponseIncomplete,
        match="SEMANTIC_RESPONSE_INCOMPLETE.*response_cap_exhausted=True",
    ) as exc:
        generator_text = decode_complete_response(
            tokenizer,
            "Generator",
            tokens,
            metrics,
        )
        critic_constructed = True
        build_critic_context(tokenizer, generator_text)

    assert exc.value.response_cap_exhausted is True
    assert exc.value.stop_reason == "client_safety_limit"
    assert tokenizer.decode_calls == 0
    assert critic_constructed is False


def test_complete_response_decodes_exact_tokens():
    tokenizer = CharTokenizer()
    text = decode_complete_response(
        tokenizer,
        "Generator",
        [97, 98, 99],
        {
            "stop_reason": "eos",
            "complete": True,
            "response_cap_exhausted": False,
        },
    )
    assert text == "abc"


def test_infer_stops_repeated_nonsemantic_chunks():
    session = Session([
        ([32, 10], 1),
        ([10, 32], 1),
        ([32, 32], 1),
        ([65], 2),
    ])
    tokens, metrics = _infer(
        Client(session),
        [],
        [9],
        2,
        lambda: {},
        max_response_tokens=0,
        semantic_progress=lambda chunk: bool(
            "".join(chr(token) for token in chunk).strip()
        ),
        max_semantic_stall_chunks=3,
    )
    assert tokens == [32, 10, 10, 32, 32, 32]
    assert metrics["stop_reason"] == "semantic_stall"
    assert metrics["complete"] is False
    assert session.calls == 3


class CharTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token) for token in token_ids)


def test_critic_context_preserves_complete_generator_response():
    context, metrics = build_critic_context(CharTokenizer(), "abcdefghij")
    assert context == "abcdefghij"
    assert metrics["generator_full_tokens"] == 10
    assert metrics["critic_context_tokens"] == 10
    assert metrics["critic_omitted_tokens"] == 0
    assert metrics["review_scope"] == "full"
    assert metrics["critic_protocol"] == "recursive_proof_decomposition_v2"
