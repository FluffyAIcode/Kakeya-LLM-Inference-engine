"""Unit tests for DistributedFusedDecoder: the remote-DFlash+f_θ fused loop.

A fake verifier models the true greedy continuation; fake remotes return either
perfect or wrong drafts. The output must be byte-identical to local greedy in
BOTH cases (correctness containment), with acceptance differing."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pytest

from inference_engine.distributed import tensor_codec as tc
from inference_engine.distributed.dflash_service import DraftResult, RestoreResult
from inference_engine.distributed.fused_decode import (
    CommitResult,
    DistributedFusedDecoder,
)


def _w() -> tc.WireTensor:
    return tc.encode_array(np.zeros((1, 1, 2), dtype=np.float32))


class _FakeVerifier:
    """Greedy verifier over a fixed true continuation. Accepts a leading draft
    prefix iff it matches the true greedy tokens; commits bonus + (correction on
    a partial accept)."""

    def __init__(self, true_seq: Sequence[int]) -> None:
        self.true_seq = list(true_seq)
        self._pos = 0
        self._ctx = 0
        self._candidate: List[int] = []
        self.prefilled = None
        self.seed_positions: List[int] = []
        self.extend_calls: List[List[int]] = []

    @property
    def context_len(self) -> int:
        return self._ctx

    def prefill(self, prompt_ids, restored, evicted_positions) -> None:
        self.prefilled = (list(prompt_ids), list(restored), list(evicted_positions))
        self._ctx = len(prompt_ids)
        self._pos = 0

    def aux_over_prompt(self):
        return [_w(), _w()]  # num_aux = 2

    def next_greedy(self) -> int:
        return self.true_seq[self._pos]

    def verify_block(self, candidate: Sequence[int]) -> int:
        accepted = 0
        for i, tok in enumerate(candidate):
            if self._pos + i < len(self.true_seq) and tok == self.true_seq[self._pos + i]:
                accepted += 1
            else:
                break
        self._candidate = list(candidate)
        return accepted

    def commit(self, accepted: int) -> CommitResult:
        cand = self._candidate
        if accepted == len(cand):
            committed = list(cand)
        else:
            correction = self.true_seq[self._pos + accepted]
            committed = list(cand[:accepted]) + [correction]
        positions = list(range(self._ctx, self._ctx + len(committed)))
        self._ctx += len(committed)
        self._pos += len(committed)
        self.extend_calls.append(positions)
        return CommitResult(tokens=committed, aux=[_w(), _w()],
                            positions=positions, stop=False)


class _FakeRemote:
    """Records calls; drafts are perfect (match true_seq), wrong (zeros), or a
    fixed list. close() not called by the decoder."""

    def __init__(self, *, true_seq=None, prompt_len=3, wrong=False) -> None:
        self.true_seq = list(true_seq or [])
        self.prompt_len = prompt_len
        self.wrong = wrong
        self.calls: List[str] = []
        self.seed_positions: List[int] = []
        self.extend_positions: List[List[int]] = []
        self._draft_pos = 0

    def restore(self, prompt_ids, *, sink, window, s5_exact_full_attn):
        self.calls.append("restore")
        return RestoreResult(restored=[], evicted_positions=[], prompt_len=len(prompt_ids))

    def seed_context(self, aux, positions):
        self.calls.append("seed_context")
        self.seed_positions = list(positions)
        return len(positions)

    def draft_block(self, *, bonus_token_id, context_len, block_size):
        self.calls.append("draft_block")
        if self.wrong:
            drafts = [999_999] * block_size  # never matches
        else:
            # perfect: the true tokens that FOLLOW the bonus at this context
            start = context_len - self.prompt_len + 1  # position after bonus
            drafts = [
                self.true_seq[start + i] if start + i < len(self.true_seq) else 0
                for i in range(block_size)
            ]
        return DraftResult(draft_token_ids=drafts, forward_passes=1, peak_activation_bytes=0)

    def extend_context(self, aux, positions):
        self.calls.append("extend_context")
        self.extend_positions.append(list(positions))
        return len(positions)


TRUE = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
PROMPT = [1, 2, 3]


def _decode(*, wrong: bool, block_size: int = 4, max_new: int = 8, eos=()):
    verifier = _FakeVerifier(TRUE)
    remote = _FakeRemote(true_seq=TRUE, prompt_len=len(PROMPT), wrong=wrong)
    dec = DistributedFusedDecoder(remote, verifier, block_size=block_size,
                                  sink=4, window=64, eos_ids=eos)
    res = dec.generate(PROMPT, max_new)
    return res, verifier, remote


def test_perfect_drafts_byte_identical_and_high_acceptance():
    res, _, remote = _decode(wrong=False, max_new=8)
    assert res.output_token_ids == TRUE[:8]
    assert remote.calls[:2] == ["restore", "seed_context"]
    # perfect drafts -> every drafted token accepted
    assert res.total_proposed > 0
    assert res.total_accepted == res.total_proposed
    assert res.acceptance_rate == 1.0


def test_wrong_drafts_byte_identical_but_zero_acceptance():
    res, _, _ = _decode(wrong=True, max_new=8)
    assert res.output_token_ids == TRUE[:8]   # SAME output as perfect drafts
    assert res.total_proposed > 0
    assert res.total_accepted == 0
    assert res.acceptance_rate == 0.0


def test_block_size_one_proposes_nothing():
    res, _, remote = _decode(wrong=False, block_size=1, max_new=5)
    assert res.output_token_ids == TRUE[:5]
    assert res.total_proposed == 0
    assert res.acceptance_rate == 0.0
    assert res.blocks == 5  # one token per block


def test_seed_and_extend_positions_are_contiguous():
    res, verifier, remote = _decode(wrong=False, max_new=8)
    assert remote.seed_positions == [0, 1, 2]  # prompt positions
    # extend positions continue from prompt_len with no gaps/overlaps
    flat = [p for chunk in remote.extend_positions for p in chunk]
    assert flat == list(range(len(PROMPT), len(PROMPT) + len(flat)))


def test_eos_stops_generation():
    verifier = _FakeVerifier(TRUE)
    remote = _FakeRemote(true_seq=TRUE, prompt_len=len(PROMPT), wrong=False)
    dec = DistributedFusedDecoder(remote, verifier, block_size=4, eos_ids=[13])
    res = dec.generate(PROMPT, 12)
    assert res.stopped_on_eos
    assert res.output_token_ids[-1] == 13
    assert 14 not in res.output_token_ids


def test_max_new_tokens_is_respected_exactly():
    res, _, _ = _decode(wrong=False, block_size=4, max_new=6)
    assert len(res.output_token_ids) == 6
    assert res.output_token_ids == TRUE[:6]


def test_prefill_receives_restore_payload():
    res, verifier, _ = _decode(wrong=False, max_new=4)
    prompt, restored, evicted = verifier.prefilled
    assert prompt == PROMPT
    assert restored == [] and evicted == []


@pytest.mark.parametrize("kwargs,msg", [
    ({"block_size": 0}, "block_size"),
])
def test_constructor_validation(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        DistributedFusedDecoder(_FakeRemote(), _FakeVerifier(TRUE), **kwargs)


def test_generate_validation():
    dec = DistributedFusedDecoder(_FakeRemote(true_seq=TRUE), _FakeVerifier(TRUE))
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        dec.generate([], 4)
    with pytest.raises(ValueError, match="max_new_tokens must be"):
        dec.generate(PROMPT, 0)


def test_acceptance_rate_zero_when_nothing_proposed():
    from inference_engine.distributed.fused_decode import DistributedFusedResult
    assert DistributedFusedResult(output_token_ids=[]).acceptance_rate == 0.0
