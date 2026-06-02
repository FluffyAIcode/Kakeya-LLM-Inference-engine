"""Integration tests for :mod:`inference_engine.server.tokenizer`.

PR-N3 migration of the former Linux-side ``test_tokenizer.py`` (which
used ``_BrokenTokenizer``, ``_EmptyTemplateTokenizer``, ``_NoEosTokenizer``
test mirrors). Validates the ``Tokenizer`` protocol's
``resolve_eos_ids`` helper against the real HF Qwen3 tokenizer, since
that's the only tokenizer the production code path actually consumes.
"""

from __future__ import annotations

import pytest

from inference_engine.server.tokenizer import Tokenizer, resolve_eos_ids


@pytest.fixture
def real_tokenizer(real_speculative_engine):
    return real_speculative_engine.tokenizer


def test_real_tokenizer_satisfies_tokenizer_protocol(real_tokenizer):
    """Structural typing check: the HF Qwen3 tokenizer satisfies the
    :class:`Tokenizer` protocol. Catches accidental protocol drift if
    a new method gets added without a real-tokenizer impl."""
    assert callable(real_tokenizer.apply_chat_template)
    assert callable(real_tokenizer.decode)
    assert callable(real_tokenizer.convert_tokens_to_ids)
    assert hasattr(real_tokenizer, "eos_token_id")
    assert hasattr(real_tokenizer, "unk_token_id")
    _: Tokenizer = real_tokenizer  # type: ignore[assignment]


def test_resolve_eos_ids_includes_canonical_eos(real_tokenizer):
    """``resolve_eos_ids`` must include the tokenizer's canonical
    ``eos_token_id`` when set."""
    eos_ids = resolve_eos_ids(real_tokenizer)
    if real_tokenizer.eos_token_id is not None:
        assert int(real_tokenizer.eos_token_id) in eos_ids


def test_resolve_eos_ids_includes_qwen3_im_end(real_tokenizer):
    """For Qwen3-family tokenizers, ``resolve_eos_ids`` adds
    ``<|im_end|>`` (the actual chat-template end-of-turn marker)
    in addition to the model's canonical EOS."""
    im_end = real_tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = resolve_eos_ids(real_tokenizer)
    if im_end is not None and im_end != real_tokenizer.unk_token_id:
        assert int(im_end) in eos_ids


def test_resolve_eos_ids_is_deduplicated(real_tokenizer):
    """No duplicates in the returned list — ordering is preserved
    but each id appears at most once."""
    eos_ids = resolve_eos_ids(real_tokenizer)
    assert len(eos_ids) == len(set(eos_ids))


def test_apply_chat_template_returns_list_of_ints(real_tokenizer):
    """The contract the server route handler relies on:
    ``apply_chat_template(..., tokenize=True, return_dict=False)``
    returns a flat ``list[int]``. Catches transformers 5.x defaulting
    to a different return type without our explicit override.
    """
    ids = real_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi."},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_decode_round_trips_through_apply_chat_template(real_tokenizer):
    """Sanity: tokens encoded via ``apply_chat_template`` decode to a
    string that contains some recognizable prompt content."""
    ids = real_tokenizer.apply_chat_template(
        [{"role": "user", "content": "kakeya"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    decoded = real_tokenizer.decode(ids, skip_special_tokens=True)
    assert "kakeya" in decoded.lower()
