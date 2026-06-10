"""Model-free prompt-lookup proposer (design doc §4).

Implements prompt-lookup decoding (PLD): find the longest suffix
n-gram of the committed prefix that also occurs *earlier* in the
prefix, and draft the tokens that historically followed that earlier
occurrence. Agentic and long-session text is highly self-repetitive
(tool schemas, JSON keys, quoted context), which is exactly where this
earns nonzero acceptance for zero weight bytes and microsecond
latency.

This is a real proposer, not a test double: it satisfies the
``DLMProposer.propose_block`` contract (ADR 0001) byte-for-byte —
exactly ``block_size`` tokens out, ``ProposerStats`` accounting — and
ships as the always-available proposer capability every fleet node can
advertise (reserved ``model_id="ngram"``). Acceptance containment
applies as with any proposer: a useless draft costs throughput, never
correctness, because the verifier-side accept rule is unchanged.
"""

from __future__ import annotations

from typing import List, Optional

from kv_cache_proposer.proposer import BlockProposal, ProposerStats

DEFAULT_MAX_NGRAM = 4
DEFAULT_MIN_NGRAM = 1


class NGramProposer:
    """Prompt-lookup block proposer.

    Parameters
    ----------
    max_ngram_size / min_ngram_size
        Suffix n-gram lengths to try, longest first. Longer matches
        are more specific and produce better continuations.
    fallback_token_id
        Used to pad the draft up to ``block_size`` when the prefix has
        no usable match or the matched continuation is shorter than
        the block. The contract requires exactly ``block_size`` tokens;
        padding with a fixed id keeps the block well-formed while the
        verifier rejects from the first wrong position onward.
    """

    def __init__(
        self,
        *,
        max_ngram_size: int = DEFAULT_MAX_NGRAM,
        min_ngram_size: int = DEFAULT_MIN_NGRAM,
        fallback_token_id: int = 0,
    ) -> None:
        if min_ngram_size <= 0:
            raise ValueError("min_ngram_size must be > 0")
        if max_ngram_size < min_ngram_size:
            raise ValueError("max_ngram_size must be >= min_ngram_size")
        if fallback_token_id < 0:
            raise ValueError("fallback_token_id must be >= 0")
        self.max_ngram_size = max_ngram_size
        self.min_ngram_size = min_ngram_size
        self.fallback_token_id = fallback_token_id
        self.stats = ProposerStats(weight_bytes=0)

    def _lookup_continuation(
        self, committed: List[int], block_size: int,
    ) -> Optional[List[int]]:
        """Longest-suffix-match continuation, or None when no match.

        Searches the most recent earlier occurrence first so drafts
        track the *current* repetition pattern (e.g. the latest tool-
        call schema) rather than a stale one from the distant prefix.
        """
        n = len(committed)
        for size in range(min(self.max_ngram_size, n - 1), self.min_ngram_size - 1, -1):
            suffix = committed[n - size:]
            # Earlier occurrences only: the window [i, i+size) must end
            # strictly before the suffix starts, so the continuation
            # token at i+size exists and is real history.
            for i in range(n - size - 1, -1, -1):
                if committed[i:i + size] == suffix:
                    continuation = committed[i + size:i + size + block_size]
                    if continuation:
                        return continuation
        return None

    def propose_block(
        self,
        committed_token_ids: List[int],
        block_size: int,
        num_steps: int,
    ) -> BlockProposal:
        """Draft exactly ``block_size`` tokens. ``num_steps`` is
        accepted for contract compatibility and ignored (lookup is a
        single pass, like DFlash)."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not committed_token_ids:
            raise ValueError("committed_token_ids must be non-empty")

        continuation = self._lookup_continuation(
            list(committed_token_ids), block_size,
        ) or []
        tokens = list(continuation[:block_size])
        if len(tokens) < block_size:
            tokens.extend(
                [self.fallback_token_id] * (block_size - len(tokens))
            )

        self.stats.total_blocks += 1
        self.stats.total_forward_passes += 1
        return BlockProposal(
            tokens=tokens,
            diffusion_steps=0,
            forward_passes=1,
            peak_activation_bytes=0,
        )
