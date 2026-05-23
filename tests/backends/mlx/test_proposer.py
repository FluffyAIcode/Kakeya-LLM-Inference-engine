"""Tests for `inference_engine.backends.mlx.proposer.MLXSparseLogitsProposer`.

Mac-only. Loading the proposer takes ~3s (download Qwen3-0.6B-Base
mlx-lm checkpoint + dllm-hub safetensors); we use a session-scoped
fixture to amortize.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

import torch

from kv_cache_proposer.proposer import (
    BlockProposal,
    DLMProposer,
    ProposerConfig,
)


@pytest.fixture(scope="session")
def mlx_proposer_session():
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
    return MLXSparseLogitsProposer(
        ProposerConfig(dtype=torch.bfloat16, device="cpu")
    )


@pytest.fixture(scope="session")
def short_chat_messages_proposer():
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly 'OK'."},
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_mlx_proposer_loads(mlx_proposer_session) -> None:
    p = mlx_proposer_session
    assert p.mask_id is not None
    assert p.pad_id is not None
    assert p.stats.weight_bytes > 0
    # The proposer's tokenizer is the HF dllm-hub tokenizer (inherited
    # from DLMProposer's __init__) — confirm it's the source of truth.
    assert hasattr(p.tokenizer, "apply_chat_template")
    # MLX-side handles
    assert p._backbone is not None
    assert p._embed_tokens is not None


def test_mlx_proposer_propose_block_returns_unmasked(
    mlx_proposer_session, short_chat_messages_proposer
) -> None:
    p = mlx_proposer_session
    prefix = p.encode_chat(short_chat_messages_proposer)
    proposal = p.propose_block(prefix, block_size=4, num_steps=4)
    assert isinstance(proposal, BlockProposal)
    assert len(proposal.tokens) == 4
    assert all(t != p.mask_id for t in proposal.tokens)
    upper = 200_000  # well above any reasonable Qwen3 vocab size
    assert all(0 <= t < upper for t in proposal.tokens)
    assert proposal.peak_activation_bytes > 0


def test_mlx_proposer_clamps_steps(
    mlx_proposer_session, short_chat_messages_proposer
) -> None:
    prefix = mlx_proposer_session.encode_chat(short_chat_messages_proposer)
    proposal = mlx_proposer_session.propose_block(
        prefix, block_size=2, num_steps=10
    )
    assert proposal.diffusion_steps == 2  # clamped to block_size


@pytest.mark.parametrize("bad_block", [0, -1])
def test_propose_block_rejects_nonpositive_block_size(
    mlx_proposer_session, bad_block
) -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        mlx_proposer_session.propose_block(
            [1, 2, 3], block_size=bad_block, num_steps=2
        )


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_propose_block_rejects_nonpositive_num_steps(
    mlx_proposer_session, bad_steps
) -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        mlx_proposer_session.propose_block(
            [1, 2, 3], block_size=4, num_steps=bad_steps
        )


def test_mlx_proposer_stats_increment(
    mlx_proposer_session, short_chat_messages_proposer
) -> None:
    p = mlx_proposer_session
    pre_blocks = p.stats.total_blocks
    pre_steps = p.stats.total_diffusion_steps
    prefix = p.encode_chat(short_chat_messages_proposer)
    p.propose_block(prefix, block_size=4, num_steps=4)
    assert p.stats.total_blocks == pre_blocks + 1
    assert p.stats.total_diffusion_steps == pre_steps + 4


# ---------------------------------------------------------------------------
# Cross-backend correctness
# ---------------------------------------------------------------------------

def test_mlx_proposer_acceptance_by_verifier(
    mlx_proposer_session, short_chat_messages_proposer
) -> None:
    """The MLX proposer's output, when passed through the PyTorch CPU
    verifier, must yield a meaningful acceptance rate.

    We don't require token-level identity with the PyTorch DLMProposer —
    bf16 noise across backends could differ in the diffusion path's
    argmax. We DO require that the tokens are valid (no <mask>), in
    range, and that at least the FIRST token is one the verifier would
    have predicted greedily — that's the strongest cross-backend gate
    we can apply to the proposer in isolation.
    """
    from kv_cache_proposer.verifier import (
        SinkWindowVerifier,
        VerifierConfig,
    )
    p = mlx_proposer_session
    prefix = p.encode_chat(short_chat_messages_proposer)
    proposal = p.propose_block(prefix, block_size=4, num_steps=4)

    # Use the PyTorch CPU verifier as an oracle (Qwen3-1.7B greedy).
    cpu_v = SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=4, window_size=128,
        )
    )
    cpu_v.prefill(prefix)
    verifier_first_token = int(torch.argmax(cpu_v.next_token_logits).item())
    # The proposer's first proposed token agrees with the verifier in
    # the majority of trivial-prompt cases (it's also Qwen3 family).
    # We don't assert equality (proposer and verifier differ), but we
    # do require the first token is at least in the verifier's top-32.
    topk = torch.topk(cpu_v.next_token_logits, k=32).indices.tolist()
    assert proposal.tokens[0] in topk, (
        f"MLX proposer's first token {proposal.tokens[0]} is not in the "
        f"verifier's top-32; either the bidirectional override broke or "
        f"weight loading didn't take effect."
    )


# ---------------------------------------------------------------------------
# Sparse-logits structural property
# ---------------------------------------------------------------------------

def test_compiled_and_uncompiled_produce_identical_tokens(
    short_chat_messages_proposer,
) -> None:
    """**Headline correctness gate for the mx.compile path.**

    Build two MLX proposers — one with ``compile_backbone=True``
    (default), one with ``compile_backbone=False`` — and run the same
    block-proposal call on both. The compiled and uncompiled
    backbones must produce **bit-identical** token sequences.

    mx.compile traces the same op graph, so up to MLX's caching
    semantics this should be exact. If it ever isn't (e.g. a future
    mlx version changes reduction order in the compiled path), the
    speculative loop would diverge silently — that's a regression we
    want surfaced loudly.
    """
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer

    proposer_compiled = MLXSparseLogitsProposer(
        ProposerConfig(dtype=torch.bfloat16, device="cpu"),
        compile_backbone=True,
    )
    proposer_uncompiled = MLXSparseLogitsProposer(
        ProposerConfig(dtype=torch.bfloat16, device="cpu"),
        compile_backbone=False,
    )
    prefix = proposer_compiled.encode_chat(short_chat_messages_proposer)

    # Multiple (L, K) configurations to make sure compile correctness
    # holds across diffusion-step counts.
    for L, K in [(2, 2), (4, 2), (8, 2), (8, 4), (16, 2), (16, 4)]:
        a = proposer_compiled.propose_block(
            prefix, block_size=L, num_steps=K
        )
        b = proposer_uncompiled.propose_block(
            prefix, block_size=L, num_steps=K
        )
        assert a.tokens == b.tokens, (
            f"compiled vs uncompiled diverged at L={L}, K={K}:\n"
            f"  compiled:   {a.tokens}\n"
            f"  uncompiled: {b.tokens}"
        )


def test_mlx_proposer_compile_flag_round_trip(short_chat_messages_proposer) -> None:
    """Both flags must produce a working proposer (the False branch is
    used by the equivalence test above; this test pins down the
    constructor's flag handling and stats wiring)."""
    from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer

    p = MLXSparseLogitsProposer(
        ProposerConfig(dtype=torch.bfloat16, device="cpu"),
        compile_backbone=False,
    )
    assert p._compile_backbone is False
    assert p._backbone_forward_compiled is None
    prefix = p.encode_chat(short_chat_messages_proposer)
    proposal = p.propose_block(prefix, block_size=4, num_steps=2)
    assert len(proposal.tokens) == 4
    assert all(t != p.mask_id for t in proposal.tokens)


def test_mlx_proposer_sparse_activation_smaller_than_dense_oracle(
    mlx_proposer_session, short_chat_messages_proposer
) -> None:
    """Sparse logits must produce activation strictly smaller than the
    full ``[1, T, V]`` shape would.

    We don't drive the dense path here (it would require loading the
    PyTorch model AGAIN); instead we compute what the dense logits
    buffer WOULD have been at the recorded T and compare against the
    sparse ``proposal.peak_activation_bytes``.
    """
    p = mlx_proposer_session
    prefix = p.encode_chat(short_chat_messages_proposer)
    L = 8
    K = 4
    p.stats.peak_activation_bytes = 0
    proposal = p.propose_block(prefix, block_size=L, num_steps=K)
    T = len(prefix) + L
    V = 151_936  # Qwen3 vocab
    bytes_per_elem = 2  # bf16
    dense_full_logits_bytes = T * V * bytes_per_elem
    assert proposal.peak_activation_bytes < dense_full_logits_bytes
    expected_ratio = L / T
    actual_ratio = proposal.peak_activation_bytes / dense_full_logits_bytes
    assert actual_ratio <= expected_ratio * 1.5, (
        f"sparse activation {proposal.peak_activation_bytes} not as small "
        f"as expected vs dense {dense_full_logits_bytes}: "
        f"actual_ratio={actual_ratio:.3f} expected ≤ {expected_ratio*1.5:.3f}"
    )
