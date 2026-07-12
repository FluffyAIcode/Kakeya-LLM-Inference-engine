from __future__ import annotations

from scripts.chat_grpc import _generate_and_print


class Tokenizer:
    def decode(self, token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens
        return " ".join(str(token) for token in token_ids)


class Session:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = 0
        self.last_stop_reason = None
        self.last_total_duration_seconds = 0.0
        self.appended = None

    def append(self, token_ids):
        self.appended = list(token_ids)

    def generate(self, *, max_tokens):
        tokens, reason, seconds = self.chunks[self.calls]
        self.calls += 1
        assert len(tokens) <= max_tokens
        yield from tokens
        self.last_stop_reason = reason
        self.last_total_duration_seconds = seconds


def test_continues_max_token_chunks_until_eos(capsys):
    session = Session([
        ([11, 12], 1, 1.0),
        ([21, 22], 2, 2.0),
    ])
    count = _generate_and_print(session, Tokenizer(), [9], max_tokens=2)
    output = capsys.readouterr()
    assert count == 4
    assert session.calls == 2
    assert session.appended == [9]
    assert "11 12 21 22" in output.out
    assert "4 tokens" in output.err
    assert "1.33 tok/s" in output.err
    assert "stop=eos" in output.err


def test_optional_response_cap_is_explicit(capsys):
    session = Session([
        ([1, 2], 1, 1.0),
        ([3, 4], 1, 1.0),
    ])
    count = _generate_and_print(
        session,
        Tokenizer(),
        [9],
        max_tokens=2,
        max_response_tokens=4,
    )
    output = capsys.readouterr()
    assert count == 4
    assert "stop=client_safety_limit" in output.err
    assert "--max-response-tokens 4" in output.err


def test_no_progress_breaks_continuation_loop(capsys):
    session = Session([
        ([], 1, 0.1),
    ])
    assert _generate_and_print(session, Tokenizer(), [9], max_tokens=2) == 0
    assert "stop=no_progress" in capsys.readouterr().err
