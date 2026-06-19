"""Distributed fused speculative decode: a local restoring verifier (gemma-4)
driving a REMOTE DFlash+f_θ proposer (ADR 0009 §4 F3).

This mirrors the single-host MLX fused loop
(``inference_engine.backends.mlx.fused_specdecode.fused_specdecode_generate``)
but the drafter context K/V + f_θ restoration live on another host, reached via
:class:`~inference_engine.distributed.dflash_service.RemoteDFlashProposer`.

Per turn: ``restore`` (prompt → f_θ-projected verifier K/V on host B → verifier
prefill on host A) then ``seed_context`` (verifier aux hidden → drafter context
on host B). Per block: ``draft_block`` (bonus → drafts) → local verify → commit →
``extend_context`` (committed aux → grow drafter context).

The decoder is framework-agnostic: the verifier hides all mlx/torch math behind
:class:`RestoringVerifier`, and aux/K-V cross the verifier↔decoder boundary as
:class:`~inference_engine.distributed.tensor_codec.WireTensor`. Correctness
containment is structural — the verifier's greedy verify decides every token, so
the output is byte-identical to local greedy regardless of remote drafts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Tuple

from inference_engine.distributed.dflash_service import RemoteDFlashProposer, RestoreResult
from inference_engine.distributed.tensor_codec import WireTensor


@dataclass
class CommitResult:
    """The tokens committed by one block + their verifier aux (to grow the
    remote drafter context) + their absolute positions."""

    tokens: List[int]
    aux: List[WireTensor]
    positions: List[int]
    stop: bool  # an EOS token is among `tokens` -> generation should halt


class RestoringVerifier(Protocol):
    """Local verifier contract the distributed loop drives. An MLX adapter over
    ``MLXRestoredIncrementalVerifier`` implements this on the Mac; tests inject a
    fake. All tensors are :class:`WireTensor`.

    Contract:
      * ``context_len`` — committed token count (prompt + accepted).
      * ``prefill`` — prefill with the remote f_θ-projected K/V banks; set the
        next-token state.
      * ``aux_over_prompt`` — aux-layer hidden over all prompt positions (seeds
        the remote drafter context).
      * ``next_greedy`` — argmax of the current next-token logits (the bonus).
      * ``verify_block`` — verify forward over the candidate; return how many
        leading tokens greedy-match (>=1; index-0 bonus is always accepted).
      * ``commit`` — drop rejected K/V, append the correction on a partial
        accept, advance next-token state, return committed tokens + aux +
        positions.
    """

    @property
    def context_len(self) -> int: ...

    def prefill(
        self, prompt_ids: Sequence[int],
        restored: Sequence[Tuple[int, WireTensor, WireTensor]],
        evicted_positions: Sequence[int],
    ) -> None: ...

    def aux_over_prompt(self) -> List[WireTensor]: ...

    def next_greedy(self) -> int: ...

    def verify_block(self, candidate: Sequence[int]) -> int: ...

    def commit(self, accepted: int) -> CommitResult: ...


@dataclass
class DistributedFusedResult:
    output_token_ids: List[int]
    blocks: int = 0
    total_proposed: int = 0
    total_accepted: int = 0
    stopped_on_eos: bool = False
    restore: RestoreResult | None = field(default=None, repr=False)

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0


class DistributedFusedDecoder:
    """Greedy fused spec-decode with a remote DFlash+f_θ proposer."""

    def __init__(
        self,
        remote: RemoteDFlashProposer,
        verifier: RestoringVerifier,
        *,
        block_size: int = 4,
        sink: int = 4,
        window: int = 64,
        s5_exact_full_attn: bool = True,
        eos_ids: Sequence[int] = (),
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.remote = remote
        self.verifier = verifier
        self.block_size = block_size
        self.sink = sink
        self.window = window
        self.s5_exact_full_attn = s5_exact_full_attn
        self.eos_ids = set(int(t) for t in eos_ids)

    def generate(
        self, prompt_ids: Sequence[int], max_new_tokens: int,
    ) -> DistributedFusedResult:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        prompt_ids = list(prompt_ids)

        # --- prefill / restoration (once) ---------------------------------
        restore = self.remote.restore(
            prompt_ids, sink=self.sink, window=self.window,
            s5_exact_full_attn=self.s5_exact_full_attn,
        )
        self.verifier.prefill(prompt_ids, restore.restored, restore.evicted_positions)
        self.remote.seed_context(
            self.verifier.aux_over_prompt(), list(range(len(prompt_ids))))

        result = DistributedFusedResult(output_token_ids=[], restore=restore)

        # --- decode blocks -------------------------------------------------
        while len(result.output_token_ids) < max_new_tokens:
            remaining = max_new_tokens - len(result.output_token_ids)
            L = min(self.block_size, remaining)
            bonus = self.verifier.next_greedy()
            # Always request >=1 draft (the wire contract); USE only L-1 of them.
            n_drafts = max(L - 1, 1)
            drafts = self.remote.draft_block(
                bonus_token_id=bonus, context_len=self.verifier.context_len,
                block_size=n_drafts,
            ).draft_token_ids
            candidate = [bonus] + list(drafts[: L - 1])  # length L
            accepted = self.verifier.verify_block(candidate)
            commit = self.verifier.commit(accepted)

            result.blocks += 1
            proposed = len(candidate) - 1  # drafts actually used (bonus excluded)
            result.total_proposed += proposed
            result.total_accepted += max(accepted - 1, 0)

            self.remote.extend_context(commit.aux, commit.positions)

            # Respect max_new_tokens even if a block committed extra (correction).
            for tok in commit.tokens:
                if len(result.output_token_ids) >= max_new_tokens:
                    break
                result.output_token_ids.append(tok)
                if tok in self.eos_ids:
                    result.stopped_on_eos = True
                    break
            if commit.stop or result.stopped_on_eos:
                result.stopped_on_eos = True
                break
        return result
