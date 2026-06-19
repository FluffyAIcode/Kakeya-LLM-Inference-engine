"""Unit tests for inference_engine.distributed.ngram (ADR 0009).

The prompt-lookup proposer is the always-available proposer capability
(design doc §4); these tests pin its draft semantics: longest suffix
match wins, most recent occurrence wins, exact-block-size output, and
the DLMProposer contract's argument validation.

Coverage target: 100% on ``inference_engine/distributed/ngram.py``.
"""

from __future__ import annotations

import pytest

from inference_engine.distributed.ngram import NGramProposer


def test_constructor_validation():
    with pytest.raises(ValueError, match="min_ngram_size"):
        NGramProposer(min_ngram_size=0)
    with pytest.raises(ValueError, match="max_ngram_size"):
        NGramProposer(max_ngram_size=1, min_ngram_size=2)
    with pytest.raises(ValueError, match="fallback_token_id"):
        NGramProposer(fallback_token_id=-1)


def test_propose_block_argument_validation():
    p = NGramProposer()
    with pytest.raises(ValueError, match="block_size"):
        p.propose_block([1, 2, 3], block_size=0, num_steps=1)
    with pytest.raises(ValueError, match="num_steps"):
        p.propose_block([1, 2, 3], block_size=4, num_steps=0)
    with pytest.raises(ValueError, match="committed_token_ids"):
        p.propose_block([], block_size=4, num_steps=1)


def test_repeated_sequence_is_continued():
    # Prefix "10 20 30 40 10 20" — the suffix bigram (10, 20) occurred
    # at position 0, followed by 30, 40. The draft must copy that
    # continuation.
    p = NGramProposer()
    proposal = p.propose_block(
        [10, 20, 30, 40, 10, 20], block_size=2, num_steps=1,
    )
    assert proposal.tokens == [30, 40]
    assert proposal.forward_passes == 1
    assert proposal.diffusion_steps == 0
    assert proposal.peak_activation_bytes == 0


def test_block_is_always_exactly_block_size():
    p = NGramProposer(fallback_token_id=99)
    # The suffix bigram (10, 20) matches at position 0; the historical
    # continuation [30, 10, 20] is one short of the block, so the
    # final slot is padded with the fallback id.
    proposal = p.propose_block([10, 20, 30, 10, 20], block_size=4, num_steps=1)
    assert proposal.tokens == [30, 10, 20, 99]


def test_no_match_pads_with_fallback():
    p = NGramProposer(fallback_token_id=7)
    proposal = p.propose_block([1, 2, 3, 4, 5], block_size=3, num_steps=1)
    assert proposal.tokens == [7, 7, 7]


def test_most_recent_occurrence_wins():
    # The suffix (1, 2) occurs twice: at position 0 followed by 5, and
    # at position 3 followed by 6. The MORE RECENT occurrence (pos 3)
    # must drive the draft so it tracks the current repetition pattern.
    p = NGramProposer()
    proposal = p.propose_block(
        [1, 2, 5, 1, 2, 6, 1, 2], block_size=1, num_steps=1,
    )
    assert proposal.tokens == [6]


def test_longest_suffix_match_beats_shorter_one():
    # Suffix trigram (8, 1, 2) matches at position 0 (continued by 3);
    # the shorter bigram (1, 2) also matches at position 4 (continued
    # by 9). The longer, more specific match must win.
    p = NGramProposer(max_ngram_size=3)
    proposal = p.propose_block(
        [8, 1, 2, 3, 1, 2, 9, 8, 1, 2], block_size=1, num_steps=1,
    )
    assert proposal.tokens == [3]


def test_single_token_prefix_has_no_searchable_history():
    p = NGramProposer(fallback_token_id=0)
    proposal = p.propose_block([5], block_size=2, num_steps=1)
    assert proposal.tokens == [0, 0]


def test_num_steps_is_ignored_like_dflash():
    p = NGramProposer()
    a = p.propose_block([1, 2, 1, 2], block_size=2, num_steps=1)
    b = p.propose_block([1, 2, 1, 2], block_size=2, num_steps=64)
    assert a.tokens == b.tokens


def test_stats_accounting():
    p = NGramProposer()
    assert p.stats.weight_bytes == 0
    p.propose_block([1, 2, 1, 2], block_size=2, num_steps=1)
    p.propose_block([1, 2, 1, 2], block_size=2, num_steps=1)
    assert p.stats.total_blocks == 2
    assert p.stats.total_forward_passes == 2
    assert p.stats.total_diffusion_steps == 0
