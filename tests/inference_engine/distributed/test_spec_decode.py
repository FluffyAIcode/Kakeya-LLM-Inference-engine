"""Unit tests for inference_engine.distributed.spec_decode (ADR 0009).

``accept_block`` is the load-bearing function of the correctness-
containment argument (a remote draft can change throughput, never
tokens) — these tests pin it as a pure function over small tensors.
The full decoder loop against a real verifier is covered by the Mac
integration test (tests/integration/test_distributed_spec_decode_real.py)
per the verifier-dependent split convention.

Coverage target: 100% on
``inference_engine/distributed/spec_decode.py``.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.distributed.capability import (
    NGRAM_MODEL_ID,
    CapabilityRole,
    ModelCapability,
    NodeCapability,
)
from inference_engine.distributed.placement import SpecDecodePlacement
from inference_engine.distributed.proposer_service import RemoteProposer
from inference_engine.distributed.spec_decode import (
    BlockAcceptance,
    DistributedSpeculativeDecoder,
    accept_block,
)


def _logits_for(token: int, vocab: int = 8) -> torch.Tensor:
    """Logits whose argmax is ``token``."""
    row = torch.zeros(vocab)
    row[token] = 1.0
    return row


def _block_logits(tokens, vocab: int = 8) -> torch.Tensor:
    return torch.stack([_logits_for(t, vocab) for t in tokens])


# ---------------------------------------------------------------------------
# accept_block
# ---------------------------------------------------------------------------


def test_full_acceptance_yields_bonus_token():
    # Verifier agrees with every draft token; row i predicts position
    # i+1, so the last row's argmax (7) is the bonus token.
    draft = [3, 5, 2]
    result = accept_block(
        _logits_for(3), draft, _block_logits([5, 2, 7]),
    )
    assert result == BlockAcceptance(accepted=3, correction_or_bonus=7)


def test_first_token_rejection_yields_correction():
    draft = [3, 5, 2]
    result = accept_block(
        _logits_for(4), draft, _block_logits([5, 2, 7]),
    )
    assert result.accepted == 0
    assert result.correction_or_bonus == 4  # the verifier's own pick


def test_partial_acceptance_stops_at_first_mismatch():
    # Verifier: pos0→3 (match), pos1→6 (draft says 5 — reject).
    draft = [3, 5, 2]
    result = accept_block(
        _logits_for(3), draft, _block_logits([6, 2, 7]),
    )
    assert result.accepted == 1
    assert result.correction_or_bonus == 6


def test_correction_always_differs_from_first_rejected_draft_token():
    # By construction: acceptance stopped because argmax != draft[i],
    # and the correction IS that argmax.
    draft = [1, 2, 3, 4]
    result = accept_block(
        _logits_for(1), draft, _block_logits([2, 0, 0, 0]),
    )
    assert result.accepted == 2
    assert result.correction_or_bonus == 0
    assert result.correction_or_bonus != draft[result.accepted]


def test_single_token_block():
    result = accept_block(_logits_for(2), [2], _block_logits([5]))
    assert result == BlockAcceptance(accepted=1, correction_or_bonus=5)


def test_rejects_empty_draft():
    with pytest.raises(ValueError, match="non-empty"):
        accept_block(_logits_for(0), [], torch.zeros((0, 8)))


def test_rejects_non_2d_block_logits():
    with pytest.raises(ValueError, match=r"\[L, V\]"):
        accept_block(_logits_for(0), [1], torch.zeros(8))


def test_rejects_row_count_mismatch():
    with pytest.raises(ValueError, match="2 rows for a draft of 1"):
        accept_block(_logits_for(0), [1], torch.zeros((2, 8)))


# ---------------------------------------------------------------------------
# DistributedSpeculativeDecoder construction
# ---------------------------------------------------------------------------


def _placement() -> SpecDecodePlacement:
    verifier_model = ModelCapability("Qwen/Qwen3-0.6B", CapabilityRole.VERIFIER)
    proposer_model = ModelCapability(NGRAM_MODEL_ID, CapabilityRole.PROPOSER)
    return SpecDecodePlacement(
        verifier_node=NodeCapability(
            node_id="a", grpc_address="a:50051", models=(verifier_model,),
        ),
        verifier_model=verifier_model,
        proposer_node=NodeCapability(
            node_id="b", grpc_address="127.0.0.1:59999", models=(proposer_model,),
        ),
        proposer_model=proposer_model,
    )


def test_from_placement_targets_the_placed_proposer_node():
    # Channel construction is lazy in gRPC: no listener is needed to
    # build the decoder, only to call it.
    decoder = DistributedSpeculativeDecoder.from_placement(
        _placement(), verifier=object(), block_size=8,
        num_diffusion_steps=4, timeout_s=12.0,
    )
    try:
        assert isinstance(decoder.proposer, RemoteProposer)
        assert decoder.proposer.address == "127.0.0.1:59999"
        assert decoder.proposer.model_id == NGRAM_MODEL_ID
        assert decoder.proposer.timeout_s == 12.0
        assert decoder.block_size == 8
        assert decoder.num_diffusion_steps == 4
    finally:
        decoder.proposer.close()


def test_decoder_inherits_speculative_decoder_validation():
    proposer = RemoteProposer("127.0.0.1:59999")
    try:
        with pytest.raises(ValueError, match="block_size"):
            DistributedSpeculativeDecoder(
                proposer, verifier=object(), block_size=0,
            )
        with pytest.raises(ValueError, match="num_diffusion_steps"):
            DistributedSpeculativeDecoder(
                proposer, verifier=object(), num_diffusion_steps=0,
            )
    finally:
        proposer.close()
