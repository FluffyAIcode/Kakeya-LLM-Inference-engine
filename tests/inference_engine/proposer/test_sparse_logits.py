"""Unit tests for `inference_engine.proposer.sparse_logits.SparseLogitsProposer`.

Real Qwen3 weights, no mocks. The headline test is
`test_sparse_path_emits_identical_tokens_to_dense`: under greedy
temperature-0 decoding the sparse path must produce the exact same
token sequence as the dense `kv_cache_proposer.proposer.DLMProposer`
parent for the same inputs.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.proposer import SparseLogitsProposer
from kv_cache_proposer.proposer import (
    BlockProposal,
    DLMProposer,
    ProposerConfig,
)


# Reuse the platform-neutral fixtures from tests/core/conftest.py for the
# dense-side oracle; pytest collects conftest.py up the directory tree
# automatically, so `proposer_session` and `short_chat_messages` are
# already in scope.


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_sparse_proposer_loads(sparse_proposer: SparseLogitsProposer) -> None:
    assert sparse_proposer.mask_id is not None
    assert sparse_proposer.pad_id is not None
    assert sparse_proposer.stats.weight_bytes > 0
    # The override resolves backbone / lm_head at construction
    assert sparse_proposer._backbone is not None
    assert sparse_proposer._lm_head is not None


def test_sparse_proposer_rejects_model_without_backbone(monkeypatch) -> None:
    """If the upstream model factors backbone differently we must fail
    loudly rather than fall back to the dense path."""
    from kv_cache_proposer.proposer import DLMProposer as _DLMProposer

    real_init = _DLMProposer.__init__

    def _patched_init(self, config=None):
        real_init(self, config)
        # Strip the .model attribute on the underlying model
        delattr(self.model, "model")

    monkeypatch.setattr(_DLMProposer, "__init__", _patched_init)
    with pytest.raises(RuntimeError, match=r"backbone as `\.model`"):
        SparseLogitsProposer(ProposerConfig(dtype=torch.bfloat16, device="cpu"))


def test_sparse_proposer_rejects_model_without_lm_head(monkeypatch) -> None:
    from kv_cache_proposer.proposer import DLMProposer as _DLMProposer

    real_init = _DLMProposer.__init__

    def _patched_init(self, config=None):
        real_init(self, config)
        delattr(self.model, "lm_head")

    monkeypatch.setattr(_DLMProposer, "__init__", _patched_init)
    with pytest.raises(RuntimeError, match=r"`\.lm_head`"):
        SparseLogitsProposer(ProposerConfig(dtype=torch.bfloat16, device="cpu"))


# ---------------------------------------------------------------------------
# Argument validation (mirrors DLMProposer's contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_block", [0, -1])
def test_propose_block_rejects_nonpositive_block_size(
    sparse_proposer: SparseLogitsProposer, bad_block: int
) -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        sparse_proposer.propose_block([1, 2, 3], block_size=bad_block, num_steps=2)


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_propose_block_rejects_nonpositive_num_steps(
    sparse_proposer: SparseLogitsProposer, bad_steps: int
) -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        sparse_proposer.propose_block([1, 2, 3], block_size=4, num_steps=bad_steps)


def test_propose_block_clamps_steps_to_block_size(
    sparse_proposer: SparseLogitsProposer, short_chat_messages
) -> None:
    prefix = sparse_proposer.encode_chat(short_chat_messages)
    proposal = sparse_proposer.propose_block(prefix, block_size=2, num_steps=10)
    assert proposal.diffusion_steps == 2  # clamped


# ---------------------------------------------------------------------------
# Output correctness
# ---------------------------------------------------------------------------

def test_propose_block_returns_unmasked_tokens(
    sparse_proposer: SparseLogitsProposer, short_chat_messages
) -> None:
    prefix = sparse_proposer.encode_chat(short_chat_messages)
    proposal = sparse_proposer.propose_block(prefix, block_size=4, num_steps=4)
    assert isinstance(proposal, BlockProposal)
    assert len(proposal.tokens) == 4
    assert all(t != sparse_proposer.mask_id for t in proposal.tokens)
    upper = sparse_proposer.model.config.vocab_size
    assert all(0 <= t < upper for t in proposal.tokens)


def test_propose_block_records_activation_peak(
    sparse_proposer: SparseLogitsProposer, short_chat_messages
) -> None:
    prefix = sparse_proposer.encode_chat(short_chat_messages)
    sparse_proposer.stats.peak_activation_bytes = 0
    proposal = sparse_proposer.propose_block(prefix, block_size=4, num_steps=4)
    assert proposal.peak_activation_bytes > 0
    assert sparse_proposer.stats.peak_activation_bytes >= proposal.peak_activation_bytes


@pytest.mark.parametrize(
    "block_size,num_steps",
    [
        (4, 4),       # k=1 every step
        (8, 4),       # k=2 every step
        (8, 3),       # mixed: front-loaded remainder
        (1, 1),       # smallest possible
        (16, 8),      # mid-size, k=2
    ],
)
def test_sparse_path_emits_identical_tokens_to_dense(
    proposer_session: DLMProposer,
    sparse_proposer: SparseLogitsProposer,
    short_chat_messages,
    block_size: int,
    num_steps: int,
) -> None:
    """**Headline correctness test.**

    Greedy + temperature 0 → both paths must be deterministic functions
    of the input. The dense path's logits shape is ``[1, T, V]`` and the
    sparse path's is ``[1, n_masked, V]``; their TOKEN OUTPUTS must
    nevertheless be identical because:

    * argmax over the V dim is invariant to whether other positions are
      computed (the operations on each masked row are independent).
    * top-k by confidence selects the same set of positions to commit.
    * The unmask-by-confidence schedule is deterministic (`num_transfer`
      computed identically from `num_masked`, `num_steps`).

    bf16 numerical noise in the lm_head matmul could in principle flip
    an argmax, but (a) Qwen3-0.6B's lm_head outputs typically have
    margins much larger than the bf16 reduction error, and (b) the
    test asserts equality, not approximate equality — if a flip happens
    on real-world inputs we want to know.
    """
    prefix = proposer_session.encode_chat(short_chat_messages)
    dense = proposer_session.propose_block(prefix, block_size=block_size, num_steps=num_steps)
    sparse = sparse_proposer.propose_block(prefix, block_size=block_size, num_steps=num_steps)
    assert sparse.tokens == dense.tokens, (
        f"sparse path diverged from dense path at L={block_size}, K={num_steps}\n"
        f"  dense:  {dense.tokens}\n"
        f"  sparse: {sparse.tokens}"
    )
    assert sparse.diffusion_steps == dense.diffusion_steps
    assert sparse.forward_passes == dense.forward_passes


def test_sparse_path_activation_peak_smaller_than_dense(
    proposer_session: DLMProposer,
    sparse_proposer: SparseLogitsProposer,
    short_chat_messages,
) -> None:
    """The whole point: at the same operating point, the sparse path
    must consume strictly less activation memory than the dense path.

    Dense logits are ``[1, T, V]``; sparse logits are ``[1, n_masked, V]``
    where ``n_masked`` peaks at ``block_size`` (first denoising step,
    everything still masked). With T=prompt_len + block_size,
    ``n_masked / T = block_size / (prompt_len + block_size)``, so the
    activation ratio shrinks as the prefix grows.
    """
    prefix = sparse_proposer.encode_chat(short_chat_messages)
    L, K = 8, 4
    proposer_session.stats.peak_activation_bytes = 0
    sparse_proposer.stats.peak_activation_bytes = 0
    dense = proposer_session.propose_block(prefix, block_size=L, num_steps=K)
    sparse = sparse_proposer.propose_block(prefix, block_size=L, num_steps=K)
    # Sparse logits are at most L positions; dense are (prefix_len + L)
    # positions. Strict-less is asserted.
    assert sparse.peak_activation_bytes < dense.peak_activation_bytes
    # Quantitative bound: sparse / dense ≈ L / (prefix_len + L). Allow
    # 25% slack for byte-padding / alignment differences.
    expected_ratio = L / (len(prefix) + L)
    actual_ratio = sparse.peak_activation_bytes / dense.peak_activation_bytes
    assert actual_ratio <= expected_ratio * 1.25, (
        f"sparse activation reduction smaller than expected: "
        f"actual ratio {actual_ratio:.3f}, expected ≤ {expected_ratio * 1.25:.3f}"
    )
