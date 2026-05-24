"""Unit tests for :mod:`inference_engine.server.tokenizer`.

Tests use the :class:`DeterministicTokenizer` from ``conftest.py`` —
a real concrete class that satisfies the ``Tokenizer`` protocol.
"""

from __future__ import annotations

from inference_engine.server.tokenizer import Tokenizer, resolve_eos_ids


def test_deterministic_tokenizer_satisfies_protocol_runtime_check(tokenizer):
    assert isinstance(tokenizer, Tokenizer)


def test_resolve_eos_includes_canonical_eos(tokenizer):
    ids = resolve_eos_ids(tokenizer)
    assert tokenizer.eos_token_id in ids


def test_resolve_eos_dedupes(tokenizer):
    """If <|im_end|> resolves to the same id as eos_token_id (which is
    the default in our DeterministicTokenizer where both are id 0),
    the result must contain it exactly once."""
    ids = resolve_eos_ids(tokenizer)
    assert len(ids) == len(set(ids))


def test_resolve_eos_includes_im_end_when_distinct():
    """Construct a tokenizer where <|im_end|> id differs from eos_id."""

    class _Distinct:
        eos_token_id = 7
        unk_token_id = 99

        def apply_chat_template(self, *a, **kw):  # pragma: no cover - unused
            raise NotImplementedError

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, token):
            if token == "<|im_end|>":
                return 13
            return None

    ids = resolve_eos_ids(_Distinct())
    assert ids == [7, 13]


def test_resolve_eos_returns_empty_when_no_eos():
    """A tokenizer that reports neither eos_token_id nor <|im_end|>
    yields an empty list — the engine constructor will reject it
    elsewhere, but resolve_eos_ids itself must not paper over it."""

    class _NoEos:
        eos_token_id = None
        unk_token_id = None

        def apply_chat_template(self, *a, **kw):  # pragma: no cover - unused
            raise NotImplementedError

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, token):
            return None

    assert resolve_eos_ids(_NoEos()) == []


def test_resolve_eos_drops_im_end_equal_to_unk():
    """If the tokenizer maps <|im_end|> to its unk_token_id (which
    means the special token is *not* in vocab), we must not treat
    that as a real EOS id."""

    class _ImEndIsUnk:
        eos_token_id = 5
        unk_token_id = 99

        def apply_chat_template(self, *a, **kw):  # pragma: no cover - unused
            raise NotImplementedError

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, token):
            if token == "<|im_end|>":
                return 99  # same as unk
            return None

    assert resolve_eos_ids(_ImEndIsUnk()) == [5]
