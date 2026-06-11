"""Distributed speculative decoding glue (ADR 0009 §4.3).

Two layers:

1. :func:`accept_block` — the greedy accept rule as a pure function
   over tensors. This is the *same* rule ``SpeculativeDecoder`` applies
   in-process; having it as a standalone, weight-free function lets the
   Linux CI gate pin the rule's semantics (the correctness-containment
   argument for accepting drafts from gossip-discovered peers rests
   entirely on this function never changing with draft provenance).

2. :class:`DistributedSpeculativeDecoder` — the v0.2 greedy
   spec-decode loop driven by a :class:`RemoteProposer`. It subclasses
   ``SpeculativeDecoder`` and changes nothing about the loop: the
   draft source is the only difference, which is the whole point —
   output remains bit-equivalent to local greedy AR decoding (modulo
   the lossy sink+window cache, exactly as documented for the local
   decoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from inference_engine.distributed.placement import SpecDecodePlacement
from inference_engine.distributed.proposer_service import RemoteProposer
from kv_cache_proposer.speculative import SpeculativeDecoder


@dataclass(frozen=True)
class BlockAcceptance:
    """Outcome of greedy verification of one draft block."""

    accepted: int
    """Length of the accepted draft prefix (0..L)."""

    correction_or_bonus: int
    """The verifier's preferred token at position (prefix + accepted):
    a correction when ``accepted < L`` (guaranteed to differ from the
    first rejected draft token), the bonus token when the whole block
    was accepted."""


def accept_block(
    prev_logits: torch.Tensor,
    draft: List[int],
    block_logits: torch.Tensor,
) -> BlockAcceptance:
    """Greedy (temperature-0) accept rule for one draft block.

    Parameters
    ----------
    prev_logits
        Verifier logits ``[V]`` predicting the first draft position
        (i.e. ``verifier.next_token_logits`` before the block forward).
    draft
        The L drafted token ids.
    block_logits
        Verifier logits ``[L, V]`` from the parallel forward over the
        draft; row ``i`` predicts position ``i + 1``.

    Accept ``draft[i]`` while ``argmax`` of the running logits equals
    it; stop at the first mismatch. Identical to the inline loop in
    ``SpeculativeDecoder.generate`` (kv_cache_proposer/speculative.py).
    """
    if block_logits.dim() != 2:
        raise ValueError(
            f"block_logits must be [L, V]; got shape {tuple(block_logits.shape)}"
        )
    if block_logits.shape[0] != len(draft):
        raise ValueError(
            f"block_logits has {block_logits.shape[0]} rows for a draft of "
            f"{len(draft)} tokens"
        )
    if not draft:
        raise ValueError("draft must be non-empty")

    accepted = 0
    running = prev_logits
    for i, draft_token in enumerate(draft):
        pred = int(torch.argmax(running).item())
        if pred != draft_token:
            break
        accepted += 1
        running = block_logits[i]
    return BlockAcceptance(
        accepted=accepted,
        correction_or_bonus=int(torch.argmax(running).item()),
    )


class DistributedSpeculativeDecoder(SpeculativeDecoder):
    """Greedy spec decode with the proposer on another node.

    The loop, accept rule, EOS handling, and streaming callback are
    inherited unchanged from :class:`SpeculativeDecoder`; only the
    draft source differs (a :class:`RemoteProposer` gRPC client). A
    remote failure surfaces as ``RemoteProposerError`` from
    ``generate`` — the verifier's session state is intact, so the
    caller may re-plan placement and resume.
    """

    def __init__(
        self,
        proposer: RemoteProposer,
        verifier: object,
        block_size: int = 16,
        num_diffusion_steps: int = 16,
    ) -> None:
        super().__init__(
            proposer=proposer,  # type: ignore[arg-type] - structural DLMProposer contract
            verifier=verifier,  # type: ignore[arg-type] - SinkWindowVerifier or MLX drop-in
            block_size=block_size,
            num_diffusion_steps=num_diffusion_steps,
        )

    @classmethod
    def from_placement(
        cls,
        placement: SpecDecodePlacement,
        verifier: object,
        *,
        block_size: int = 16,
        num_diffusion_steps: int = 16,
        timeout_s: float = 60.0,
    ) -> "DistributedSpeculativeDecoder":
        """Build a decoder from a planned placement + a loaded verifier.

        The caller is responsible for having loaded ``verifier`` per
        ``placement.verifier_model`` on this node (this node should be
        ``placement.verifier_node``); the proposer side needs no local
        state — just the placed node's address and model id.
        """
        proposer = RemoteProposer(
            placement.proposer_node.grpc_address,
            model_id=placement.proposer_model.model_id,
            timeout_s=timeout_s,
        )
        return cls(proposer=proposer, verifier=verifier,
                   block_size=block_size, num_diffusion_steps=num_diffusion_steps)
