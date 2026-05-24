"""Unit tests for :class:`_StreamingDetokenizer`.

The other public symbols that previously lived in
``inference_engine.server.streaming`` (``iter_token_deltas`` and
``run_blocking``) were removed when the route layer started routing
every request through :class:`Scheduler`. The detokenizer is the
last surviving piece.
"""

from __future__ import annotations

from typing import List

import pytest

from inference_engine.server.streaming import _StreamingDetokenizer

pytestmark = pytest.mark.asyncio


async def test_detokenizer_emits_full_text_via_per_token_deltas(tokenizer):
    """Sum of per-token deltas equals tokenizer.decode of the full id list."""
    ids = [
        tokenizer._intern("hello"),
        tokenizer._intern("world"),
        tokenizer._intern("!"),
    ]
    detok = _StreamingDetokenizer(tokenizer)
    pieces: List[str] = [detok.feed(t) for t in ids]
    full = tokenizer.decode(ids, skip_special_tokens=True)
    assert "".join(pieces) == full


async def test_detokenizer_handles_special_tokens(tokenizer):
    """Special tokens (id 0 = <|im_end|>) decode to empty under
    skip_special_tokens=True; the delta on that token is empty."""
    eos = tokenizer.eos_token_id
    hello = tokenizer._intern("hello")
    detok = _StreamingDetokenizer(tokenizer)
    d1 = detok.feed(hello)
    d2 = detok.feed(eos)
    assert d1 == "hello"
    assert d2 == ""


async def test_detokenizer_starts_empty(tokenizer):
    """Fresh detokenizer has no internal state; first feed returns
    the full decoded text of that one token."""
    h = tokenizer._intern("hi")
    detok = _StreamingDetokenizer(tokenizer)
    assert detok.feed(h) == "hi"


async def test_detokenizer_is_per_instance(tokenizer):
    """Two detokenizers fed the same id should each return the same
    delta — i.e. they don't share mutable state."""
    h = tokenizer._intern("foo")
    a = _StreamingDetokenizer(tokenizer)
    b = _StreamingDetokenizer(tokenizer)
    assert a.feed(h) == "foo"
    assert b.feed(h) == "foo"
