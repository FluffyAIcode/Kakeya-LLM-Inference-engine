"""Integration tests for :class:`_StreamingDetokenizer`.

PR-N3 migration of the former Linux-side ``test_streaming.py`` (which
used the ``DeterministicTokenizer._intern`` private method).
``_StreamingDetokenizer`` is a thin wrapper around any tokenizer
that exposes ``decode()``; tests here drive it with the real Qwen3
tokenizer.
"""

from __future__ import annotations

from typing import List

import pytest

from inference_engine.server.streaming import _StreamingDetokenizer


@pytest.fixture
def real_tokenizer(real_speculative_engine):
    return real_speculative_engine.tokenizer


def test_detokenizer_emits_full_text_via_per_token_deltas(real_tokenizer):
    """Sum of per-token deltas equals ``tokenizer.decode`` of the
    full id list. Uses real Qwen3 tokenizer's ``encode`` to get a
    known-good prompt → ids round-trip."""
    ids = real_tokenizer.encode("hello world", add_special_tokens=False)
    detok = _StreamingDetokenizer(real_tokenizer)
    pieces: List[str] = [detok.feed(t) for t in ids]
    full = real_tokenizer.decode(ids, skip_special_tokens=True)
    assert "".join(pieces) == full


def test_detokenizer_starts_empty(real_tokenizer):
    """Fresh detokenizer has no internal state; first feed returns
    the decoded text of that one token."""
    ids = real_tokenizer.encode("hi", add_special_tokens=False)
    detok = _StreamingDetokenizer(real_tokenizer)
    first_delta = detok.feed(ids[0])
    assert first_delta == real_tokenizer.decode(
        [ids[0]], skip_special_tokens=True,
    )


def test_detokenizer_is_per_instance(real_tokenizer):
    """Two detokenizers fed the same id sequence each produce the
    same outputs — they don't share mutable state."""
    ids = real_tokenizer.encode("foo", add_special_tokens=False)
    a = _StreamingDetokenizer(real_tokenizer)
    b = _StreamingDetokenizer(real_tokenizer)
    out_a = "".join(a.feed(t) for t in ids)
    out_b = "".join(b.feed(t) for t in ids)
    assert out_a == out_b


def test_detokenizer_handles_special_tokens(real_tokenizer):
    """Special tokens (e.g., the canonical EOS) decode to empty under
    ``skip_special_tokens=True``; the delta on that token is empty."""
    if real_tokenizer.eos_token_id is None:
        pytest.skip("tokenizer has no canonical EOS")
    detok = _StreamingDetokenizer(real_tokenizer)
    # Feed any normal token first so the detokenizer has running state.
    normal_ids = real_tokenizer.encode("ok", add_special_tokens=False)
    for t in normal_ids:
        detok.feed(t)
    eos_delta = detok.feed(int(real_tokenizer.eos_token_id))
    assert eos_delta == ""
