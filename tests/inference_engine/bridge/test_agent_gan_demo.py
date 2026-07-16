from scripts.agent_gan_inference_demo import (
    _agent_cache_gate,
    _infer,
    _output_metadata,
    build_critic_evidence,
)


def test_agent_gate_requires_remote_warmup_and_primary_hot_inference():
    warm = {"remote_jobs": 1, "remote_hits": 1}
    actual = {
        "local_hits": 1,
        "remote_jobs": 0,
        "tokens_computed": 0,
        "fallbacks": 0,
    }
    assert _agent_cache_gate(warm, actual)
    assert not _agent_cache_gate({**warm, "remote_jobs": 0}, actual)
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

    def create_session(self, **_kwargs):
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


class CharTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token) for token in token_ids)


def test_critic_evidence_is_bounded_and_explicit_about_omission():
    evidence, metrics = build_critic_evidence(CharTokenizer(), "abcdefghij", 4)
    assert "ab" in evidence and "ij" in evidence
    assert "6 generator tokens omitted" in evidence
    assert metrics["generator_full_tokens"] == 10
    assert metrics["critic_omitted_tokens"] == 6
    full, full_metrics = build_critic_evidence(CharTokenizer(), "abc", 4)
    assert full == "abc"
    assert full_metrics["critic_omitted_tokens"] == 0
    try:
        build_critic_evidence(CharTokenizer(), "abc", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected evidence budget validation")
