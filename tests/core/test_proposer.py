"""Unit tests for `kv_cache_proposer.proposer.DLMProposer`."""

from __future__ import annotations

import pytest
import torch

from kv_cache_proposer.proposer import (
    DLMProposer,
    ProposerConfig,
    BlockProposal,
    ProposerStats,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_proposer_loads_with_defaults(proposer_session: DLMProposer) -> None:
    p = proposer_session
    assert p.mask_id is not None
    assert p.pad_id is not None
    assert p.stats.weight_bytes > 0
    assert p.stats.total_blocks == 0


# ---------------------------------------------------------------------------
# encode_chat
# ---------------------------------------------------------------------------

def test_encode_chat_returns_int_list(proposer_session: DLMProposer, short_chat_messages) -> None:
    ids = proposer_session.encode_chat(short_chat_messages)
    assert isinstance(ids, list)
    assert all(isinstance(t, int) for t in ids)
    assert len(ids) > 5


# ---------------------------------------------------------------------------
# propose_block — happy path
# ---------------------------------------------------------------------------

def test_propose_block_returns_unmasked_tokens(
    proposer_session: DLMProposer, short_chat_messages
) -> None:
    prefix = proposer_session.encode_chat(short_chat_messages)
    proposal = proposer_session.propose_block(prefix, block_size=4, num_steps=4)
    assert isinstance(proposal, BlockProposal)
    assert proposal.diffusion_steps == 4
    assert proposal.forward_passes == 4
    assert proposal.peak_activation_bytes > 0
    assert len(proposal.tokens) == 4
    assert all(t != proposer_session.mask_id for t in proposal.tokens)
    # Use the model config's vocab_size, which includes special tokens
    # (the tokenizer's `vocab_size` excludes added special tokens).
    upper = proposer_session.model.config.vocab_size
    assert all(0 <= t < upper for t in proposal.tokens)


def test_propose_block_clamps_steps_to_block_size(
    proposer_session: DLMProposer, short_chat_messages
) -> None:
    prefix = proposer_session.encode_chat(short_chat_messages)
    # Asking for more steps than block_size collapses to block_size.
    proposal = proposer_session.propose_block(prefix, block_size=2, num_steps=10)
    assert proposal.diffusion_steps == 2  # clamped


def test_propose_block_updates_running_stats(
    proposer_session: DLMProposer, short_chat_messages
) -> None:
    pre_blocks = proposer_session.stats.total_blocks
    pre_steps = proposer_session.stats.total_diffusion_steps
    pre_passes = proposer_session.stats.total_forward_passes
    prefix = proposer_session.encode_chat(short_chat_messages)
    proposer_session.propose_block(prefix, block_size=4, num_steps=4)
    assert proposer_session.stats.total_blocks == pre_blocks + 1
    assert proposer_session.stats.total_diffusion_steps == pre_steps + 4
    assert proposer_session.stats.total_forward_passes == pre_passes + 4
    assert proposer_session.stats.peak_activation_bytes > 0


# ---------------------------------------------------------------------------
# propose_block — argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_block", [0, -1, -10])
def test_propose_block_rejects_nonpositive_block_size(
    proposer_session: DLMProposer, bad_block: int
) -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        proposer_session.propose_block([1, 2, 3], block_size=bad_block, num_steps=2)


@pytest.mark.parametrize("bad_steps", [0, -1, -5])
def test_propose_block_rejects_nonpositive_num_steps(
    proposer_session: DLMProposer, bad_steps: int
) -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        proposer_session.propose_block([1, 2, 3], block_size=4, num_steps=bad_steps)


# ---------------------------------------------------------------------------
# Tokenizer / pad invariants
# ---------------------------------------------------------------------------

def test_proposer_construct_requires_mask_id(monkeypatch) -> None:
    """When the tokenizer reports no mask_token_id we must fail loudly,
    not silently emit pad."""
    from transformers import AutoTokenizer
    real_from_pretrained = AutoTokenizer.from_pretrained

    class _NoMaskTok:
        def __init__(self, inner):
            self._inner = inner
        def __getattr__(self, name):
            return getattr(self._inner, name)
        @property
        def mask_token_id(self):
            return None

    def _patched(*args, **kwargs):
        return _NoMaskTok(real_from_pretrained(*args, **kwargs))

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _patched)
    with pytest.raises(RuntimeError, match="mask_token_id"):
        DLMProposer(ProposerConfig(dtype=torch.bfloat16, device="cpu"))


def test_proposer_construct_requires_pad_or_eos(monkeypatch) -> None:
    from transformers import AutoTokenizer
    real_from_pretrained = AutoTokenizer.from_pretrained

    class _NoPadEosTok:
        def __init__(self, inner):
            self._inner = inner
        def __getattr__(self, name):
            return getattr(self._inner, name)
        @property
        def pad_token_id(self):
            return None
        @property
        def eos_token_id(self):
            return None

    def _patched(*args, **kwargs):
        return _NoPadEosTok(real_from_pretrained(*args, **kwargs))

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _patched)
    with pytest.raises(RuntimeError, match="neither pad nor eos"):
        DLMProposer(ProposerConfig(dtype=torch.bfloat16, device="cpu"))


def test_proposer_stats_dataclass_defaults() -> None:
    s = ProposerStats()
    assert s.total_blocks == 0
    assert s.total_diffusion_steps == 0
    assert s.total_forward_passes == 0
    assert s.peak_activation_bytes == 0
    assert s.weight_bytes == 0


# ---------------------------------------------------------------------------
# Defensive / invariant paths
# ---------------------------------------------------------------------------

def test_propose_block_with_smaller_steps_than_block(
    proposer_session: DLMProposer, short_chat_messages
) -> None:
    """When num_steps < block_size, the schedule contains zero-transfer
    steps in front-loaded order. We just need to exercise the path and
    confirm no mask leaks out."""
    prefix = proposer_session.encode_chat(short_chat_messages)
    proposal = proposer_session.propose_block(prefix, block_size=8, num_steps=3)
    assert len(proposal.tokens) == 8
    assert all(t != proposer_session.mask_id for t in proposal.tokens)


def test_propose_block_underfill_raises(
    monkeypatch, proposer_session: DLMProposer, short_chat_messages
) -> None:
    """If the diffusion schedule somehow leaves <mask> tokens behind
    (e.g. due to a numerical regression in a future torch version), the
    proposer must fail loudly rather than emit mask ids."""
    import torch as _torch
    real_topk = _torch.topk

    # Force topk to return an empty selection so unmasking never happens.
    def _empty_topk(*args, **kwargs):
        v, _ = real_topk(*args, **kwargs)
        return v[:0], v[:0].long()

    monkeypatch.setattr(_torch, "topk", _empty_topk)
    prefix = proposer_session.encode_chat(short_chat_messages)
    with pytest.raises(RuntimeError, match="masked positions"):
        proposer_session.propose_block(prefix, block_size=4, num_steps=4)


def test_encode_chat_rejects_non_list_return(
    monkeypatch, proposer_session: DLMProposer, short_chat_messages
) -> None:
    """If the tokenizer's apply_chat_template returns something other than
    a list (e.g. a future API change), encode_chat must raise."""
    real = proposer_session.tokenizer.apply_chat_template

    def _bad_return(*args, **kwargs):
        return {"input_ids": real(*args, **kwargs)}  # dict, not list

    monkeypatch.setattr(proposer_session.tokenizer, "apply_chat_template", _bad_return)
    with pytest.raises(RuntimeError, match="expected list"):
        proposer_session.encode_chat(short_chat_messages)


def test_encode_chat_rejects_non_int_elements(
    monkeypatch, proposer_session: DLMProposer, short_chat_messages
) -> None:
    real = proposer_session.tokenizer.apply_chat_template

    def _bad_elements(*args, **kwargs):
        ids = real(*args, **kwargs)
        return [str(t) for t in ids]  # list of str instead of int

    monkeypatch.setattr(proposer_session.tokenizer, "apply_chat_template", _bad_elements)
    with pytest.raises(RuntimeError, match="not.*all int"):
        proposer_session.encode_chat(short_chat_messages)
