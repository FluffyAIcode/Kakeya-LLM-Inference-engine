"""Speculative decoding loop with rejection sampling.

Greedy variant. The output of this decoder is **bit-equivalent** to greedy AR
decoding from the verifier alone, *modulo* the lossy KV cache (sink+window).
With sink+window large enough to cover the full sequence, equivalence is
exact.

Algorithm (per outer step):

  1. Proposer drafts a block of L tokens ``d[0..L-1]`` conditioned on the
     committed prefix.
  2. Verifier runs one parallel forward pass over ``d``, producing ``logits``
     of shape ``[L, V]``.
  3. Walk ``i = 0..L-1``: if
         argmax(prev_logits) == d[i]
     accept ``d[i]`` and advance ``prev_logits = logits[i]``; else break.
  4. Let ``accepted`` be the count from step 3. The verifier's preferred next
     token is ``argmax(prev_logits)``. That token is *guaranteed* to differ
     from ``d[accepted]`` (if ``accepted < L``); otherwise it is the
     "bonus" token.
  5. Truncate cache to ``accepted`` of the L provisional slots, then forward
     the correction/bonus through the cache so its K/V is committed too.

The optional ``on_token`` callback fires once per **committed** token, in
emission order, with the integer token id. Streaming front-ends use this to
display partial output as it's produced rather than waiting for the whole
generation to finish — critical for any prompt that needs more than a few
tokens of response.

This is the textbook greedy speculative decoding scheme used by DiffuSpec for
diffusion-LM drafters; rejection sampling at temperature 0 collapses to
argmax-equality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Set

import torch

from .proposer import DLMProposer, BlockProposal
from .verifier import SinkWindowVerifier


# Public type alias for streaming callbacks. The callback receives one
# token id at a time, in output order. Returning truthy from the callback
# requests the loop to stop (e.g. user pressed Ctrl-C in a chat client).
TokenCallback = Callable[[int], Optional[bool]]


@dataclass
class SpeculativeRunResult:
    output_token_ids: List[int]
    """The list of generated token ids (excluding the prompt)."""

    accepted_per_block: List[int]
    """How many proposer tokens were accepted in each block."""

    proposed_per_block: List[int]
    """The block size used for each block (== L unless near the end)."""

    proposer_forward_calls: int
    proposer_diffusion_steps: int
    verifier_forward_calls: int
    verifier_tokens_consumed: int
    proposer_peak_activation_bytes: int
    proposer_weight_bytes: int
    verifier_peak_kv_bytes: int
    verifier_final_kv_bytes: int
    verifier_peak_activation_bytes: int
    verifier_weight_bytes: int
    verifier_final_kv_token_count: int
    wall_time_seconds: float

    @property
    def total_proposed(self) -> int:
        return sum(self.proposed_per_block)

    @property
    def total_accepted(self) -> int:
        return sum(self.accepted_per_block)

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed


class SpeculativeDecoder:
    def __init__(
        self,
        proposer: DLMProposer,
        verifier: SinkWindowVerifier,
        block_size: int = 16,
        num_diffusion_steps: int = 16,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be > 0")
        self.proposer = proposer
        self.verifier = verifier
        self.block_size = block_size
        self.num_diffusion_steps = num_diffusion_steps

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: Optional[Iterable[int]] = None,
        on_token: Optional[TokenCallback] = None,
    ) -> SpeculativeRunResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        eos_set: Set[int] = set(eos_token_ids or [])
        # Local helper: invoke the streaming callback (if any) and return
        # True if the caller asked us to stop early.
        def _emit(tokens: List[int]) -> bool:
            if on_token is None:
                return False
            for tok in tokens:
                if bool(on_token(tok)):
                    return True
            return False

        t0 = time.perf_counter()
        # Reset stats so successive runs report cleanly.
        self.proposer.stats.total_blocks = 0
        self.proposer.stats.total_diffusion_steps = 0
        self.proposer.stats.total_forward_passes = 0
        self.proposer.stats.peak_activation_bytes = 0
        self.verifier.stats.forward_calls = 0
        self.verifier.stats.tokens_consumed = 0
        self.verifier.stats.peak_kv_bytes = 0
        self.verifier.stats.peak_activation_bytes = 0

        # ADR 0007 §2.4: dispatch on path-selection. ContinuationPlan
        # reuses cached prefix; NewSession runs full prefill (the
        # v0.3.0-rc1 behavior). Output is bit-identical between the
        # two paths for the same input (§2.7); the only difference
        # is the prefill cost.
        from .path_plan import ContinuationPlan, NewSession
        plan = self.verifier.path_select(prompt_ids)
        if isinstance(plan, ContinuationPlan):
            self.verifier.prefill_incremental(plan.new_tokens)
        else:
            assert isinstance(plan, NewSession), (
                f"path_select must return ContinuationPlan or NewSession, "
                f"got {type(plan).__name__}"
            )
            self.verifier.prefill(plan.prompt)
        committed: List[int] = list(prompt_ids)
        generated: List[int] = []
        accepted_per_block: List[int] = []
        proposed_per_block: List[int] = []
        stop = False

        while len(generated) < max_new_tokens and not stop:
            remaining = max_new_tokens - len(generated)
            L = min(self.block_size, remaining)
            proposal: BlockProposal = self.proposer.propose_block(
                committed_token_ids=committed,
                block_size=L,
                num_steps=self.num_diffusion_steps,
            )
            d = proposal.tokens
            if len(d) != L:  # pragma: no cover - proposer.propose_block guarantees len == block_size
                raise RuntimeError(
                    f"Proposer returned {len(d)} tokens; expected exactly {L}. "
                    "Refusing to continue with a malformed block."
                )

            block_logits = self.verifier.forward_block(d)  # [L, V]
            prev_logits = self.verifier.next_token_logits
            accepted = 0
            for i in range(L):
                pred = int(torch.argmax(prev_logits).item())
                if pred == d[i]:
                    accepted += 1
                    prev_logits = block_logits[i]
                else:
                    break

            # `prev_logits` now predicts the token at position
            # (committed_len + accepted). It is also exactly the verifier's
            # preferred next token (correction if accepted<L, bonus if all).
            correction_or_bonus = int(torch.argmax(prev_logits).item())

            self.verifier.commit_or_truncate(forwarded=L, accepted=accepted)
            committed.extend(d[:accepted])
            generated.extend(d[:accepted])
            accepted_per_block.append(accepted)
            proposed_per_block.append(L)

            if any(t in eos_set for t in d[:accepted]):
                # find first EOS in accepted prefix and trim
                for i, t in enumerate(d[:accepted]):
                    if t in eos_set:
                        # discard everything after EOS
                        excess = accepted - (i + 1)
                        if excess > 0:
                            generated = generated[:-excess]
                            committed = committed[:-excess]
                        stop = True
                        break
                if stop:
                    # Emit the accepted prefix up to and including EOS,
                    # then exit. Streaming clients see the partial answer
                    # in real time.
                    _emit(d[: i + 1])
                    break

            # Stream the accepted-prefix tokens (no EOS in them on this
            # branch; callback fires before the early-exit checks below).
            if accepted > 0:
                if _emit(d[:accepted]):
                    stop = True
                    break

            if len(generated) >= max_new_tokens:
                break

            # Commit the correction/bonus token: forward it so its K/V is in
            # the cache. `append_token` itself updates `verifier.next_token_logits`
            # to predict the *following* position, so we don't reassign here.
            self.verifier.append_token(correction_or_bonus)
            committed.append(correction_or_bonus)
            generated.append(correction_or_bonus)
            if _emit([correction_or_bonus]):
                stop = True
                break
            if correction_or_bonus in eos_set:
                stop = True
                break

        elapsed = time.perf_counter() - t0
        # final KV bytes
        final_kv_bytes = self._kv_bytes(self.verifier)
        return SpeculativeRunResult(
            output_token_ids=generated,
            accepted_per_block=accepted_per_block,
            proposed_per_block=proposed_per_block,
            proposer_forward_calls=self.proposer.stats.total_forward_passes,
            proposer_diffusion_steps=self.proposer.stats.total_diffusion_steps,
            verifier_forward_calls=self.verifier.stats.forward_calls,
            verifier_tokens_consumed=self.verifier.stats.tokens_consumed,
            proposer_peak_activation_bytes=self.proposer.stats.peak_activation_bytes,
            proposer_weight_bytes=self.proposer.stats.weight_bytes,
            verifier_peak_kv_bytes=self.verifier.stats.peak_kv_bytes,
            verifier_final_kv_bytes=final_kv_bytes,
            verifier_peak_activation_bytes=self.verifier.stats.peak_activation_bytes,
            verifier_weight_bytes=self.verifier.stats.weight_bytes,
            verifier_final_kv_token_count=self.verifier.cache_logical_size,
            wall_time_seconds=elapsed,
        )

    @staticmethod
    def _kv_bytes(verifier: SinkWindowVerifier) -> int:
        if verifier.cache is None:
            return 0
        total = 0
        layers = getattr(verifier.cache, "layers", verifier.cache)
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None:
                if hasattr(keys, "numel"):
                    total += keys.numel() * keys.element_size()
                else:  # pragma: no cover - MLX cache path
                    total += int(keys.size) * int(keys.dtype.size)
            if values is not None:
                if hasattr(values, "numel"):
                    total += values.numel() * values.element_size()
                else:  # pragma: no cover - MLX cache path
                    total += int(values.size) * int(values.dtype.size)
        return total
