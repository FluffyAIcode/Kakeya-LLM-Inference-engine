"""Unit tests for :mod:`training.repr_align.proposer_surgery`.

Tests are pure-CPU and use tiny random tensors (no real Qwen3
weights). The full real-weight surgery path is exercised by the
Stage-2 entry script and validated end-to-end in Stage 4 evaluation;
these tests verify the surgery's structural and gradient
invariants on a small fixture so CI can enforce them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from training.repr_align.proposer_surgery import (
    ReprAlignedSurgery,
    SurgeryConfig,
    _extract_weight,
    _resolve_dotted_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


VOCAB = 32
D_V = 8
D_Q = 4


@pytest.fixture
def config() -> SurgeryConfig:
    return SurgeryConfig(
        verifier_hidden_dim=D_V,
        proposer_hidden_dim=D_Q,
        vocab_size=VOCAB,
    )


@pytest.fixture
def embed_weight() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(VOCAB, D_V)


@pytest.fixture
def lm_head_weight() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(VOCAB, D_V)


@pytest.fixture
def surgery(
    config: SurgeryConfig,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> ReprAlignedSurgery:
    return ReprAlignedSurgery(
        config=config,
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
    )


# ---------------------------------------------------------------------------
# SurgeryConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"verifier_hidden_dim": 0},
        {"verifier_hidden_dim": -1},
        {"proposer_hidden_dim": 0},
        {"proposer_hidden_dim": -3},
        {"vocab_size": 0},
        {"vocab_size": -7},
        {"bridge_init_std": 0.0},
        {"bridge_init_std": -0.5},
    ],
)
def test_surgery_config_rejects_non_positive_values(kwargs):
    base = dict(verifier_hidden_dim=8, proposer_hidden_dim=4, vocab_size=16)
    base.update(kwargs)
    with pytest.raises(ValueError):
        SurgeryConfig(**base)


def test_surgery_config_default_init_std():
    cfg = SurgeryConfig(verifier_hidden_dim=8, proposer_hidden_dim=4, vocab_size=16)
    assert cfg.bridge_init_std == 0.02


# ---------------------------------------------------------------------------
# Construction: shapes, weight copying, freeze status
# ---------------------------------------------------------------------------


def test_construction_creates_correctly_shaped_modules(surgery):
    assert isinstance(surgery.frozen_embed, nn.Embedding)
    assert surgery.frozen_embed.weight.shape == (VOCAB, D_V)

    assert isinstance(surgery.frozen_lm_head, nn.Linear)
    assert surgery.frozen_lm_head.weight.shape == (VOCAB, D_V)
    assert surgery.frozen_lm_head.bias is None

    assert surgery.W_in.weight.shape == (D_Q, D_V)
    assert surgery.W_in.bias is None
    assert surgery.W_out.weight.shape == (D_V, D_Q)
    assert surgery.W_out.bias is None


def test_frozen_components_have_requires_grad_false(surgery):
    assert surgery.frozen_embed.weight.requires_grad is False
    assert surgery.frozen_lm_head.weight.requires_grad is False


def test_bridge_components_have_requires_grad_true(surgery):
    assert surgery.W_in.weight.requires_grad is True
    assert surgery.W_out.weight.requires_grad is True


def test_embed_weight_is_copied_not_aliased(
    config: SurgeryConfig, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor
):
    s = ReprAlignedSurgery(
        config=config, embed_weight=embed_weight, lm_head_weight=lm_head_weight
    )
    # Mutating the source must not affect the surgery's internal copy.
    original_first_row = s.frozen_embed.weight[0].clone()
    embed_weight.zero_()
    assert torch.equal(s.frozen_embed.weight[0], original_first_row)


def test_lm_head_weight_is_copied_not_aliased(
    config: SurgeryConfig, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor
):
    s = ReprAlignedSurgery(
        config=config, embed_weight=embed_weight, lm_head_weight=lm_head_weight
    )
    original_first_row = s.frozen_lm_head.weight[0].clone()
    lm_head_weight.zero_()
    assert torch.equal(s.frozen_lm_head.weight[0], original_first_row)


def test_embed_weight_values_match_source(
    config: SurgeryConfig, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor
):
    s = ReprAlignedSurgery(
        config=config, embed_weight=embed_weight, lm_head_weight=lm_head_weight
    )
    assert torch.equal(s.frozen_embed.weight, embed_weight)


def test_lm_head_weight_values_match_source(
    config: SurgeryConfig, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor
):
    s = ReprAlignedSurgery(
        config=config, embed_weight=embed_weight, lm_head_weight=lm_head_weight
    )
    assert torch.equal(s.frozen_lm_head.weight, lm_head_weight)


# ---------------------------------------------------------------------------
# Weight validation: shape, dim count, mismatch
# ---------------------------------------------------------------------------


def test_construction_rejects_1d_embed_weight(config: SurgeryConfig):
    bad = torch.randn(VOCAB * D_V)
    good = torch.randn(VOCAB, D_V)
    with pytest.raises(ValueError, match="embed_weight must be 2-D"):
        ReprAlignedSurgery(config=config, embed_weight=bad, lm_head_weight=good)


def test_construction_rejects_3d_embed_weight(config: SurgeryConfig):
    bad = torch.randn(2, VOCAB, D_V)
    good = torch.randn(VOCAB, D_V)
    with pytest.raises(ValueError, match="embed_weight must be 2-D"):
        ReprAlignedSurgery(config=config, embed_weight=bad, lm_head_weight=good)


def test_construction_rejects_wrong_vocab_size(config: SurgeryConfig):
    bad = torch.randn(VOCAB + 1, D_V)
    good = torch.randn(VOCAB, D_V)
    with pytest.raises(ValueError, match=r"shape\[0\] = 33 does not match"):
        ReprAlignedSurgery(config=config, embed_weight=bad, lm_head_weight=good)


def test_construction_rejects_wrong_hidden_dim(config: SurgeryConfig):
    bad = torch.randn(VOCAB, D_V + 1)
    good = torch.randn(VOCAB, D_V)
    with pytest.raises(ValueError, match=r"shape\[1\] = 9 does not match"):
        ReprAlignedSurgery(config=config, embed_weight=bad, lm_head_weight=good)


def test_construction_rejects_wrong_lm_head_shape(config: SurgeryConfig):
    good = torch.randn(VOCAB, D_V)
    bad = torch.randn(VOCAB - 1, D_V)
    with pytest.raises(ValueError, match=r"lm_head_weight.shape\[0\] = 31"):
        ReprAlignedSurgery(config=config, embed_weight=good, lm_head_weight=bad)


def test_construction_rejects_1d_lm_head_weight(config: SurgeryConfig):
    good = torch.randn(VOCAB, D_V)
    bad = torch.randn(VOCAB * D_V)
    with pytest.raises(ValueError, match="lm_head_weight must be 2-D"):
        ReprAlignedSurgery(config=config, embed_weight=good, lm_head_weight=bad)


# ---------------------------------------------------------------------------
# Forward / endpoint shape contracts
# ---------------------------------------------------------------------------


def test_embed_returns_proposer_space_shape(surgery):
    input_ids = torch.randint(0, VOCAB, (2, 5))
    out = surgery.embed(input_ids)
    assert out.shape == (2, 5, D_Q)


def test_project_to_verifier_space_returns_verifier_shape(surgery):
    h_q = torch.randn(2, 5, D_Q)
    out = surgery.project_to_verifier_space(h_q)
    assert out.shape == (2, 5, D_V)


def test_lm_logits_returns_vocab_shape(surgery):
    h_q = torch.randn(2, 5, D_Q)
    out = surgery.lm_logits(h_q)
    assert out.shape == (2, 5, VOCAB)


def test_lm_logits_equals_lm_head_of_projected_hidden(surgery):
    h_q = torch.randn(2, 5, D_Q)
    via_endpoint = surgery.lm_logits(h_q)
    via_explicit = surgery.frozen_lm_head(surgery.project_to_verifier_space(h_q))
    assert torch.allclose(via_endpoint, via_explicit)


def test_forward_returns_three_tensors_with_correct_shapes(surgery):
    input_ids = torch.randint(0, VOCAB, (2, 5))
    h_q = torch.randn(2, 5, D_Q)
    embeds, projected, logits = surgery(input_ids, h_q)
    assert embeds.shape == (2, 5, D_Q)
    assert projected.shape == (2, 5, D_V)
    assert logits.shape == (2, 5, VOCAB)


def test_project_preserves_arbitrary_leading_shape(surgery):
    h_q = torch.randn(3, 4, 5, D_Q)
    out = surgery.project_to_verifier_space(h_q)
    assert out.shape == (3, 4, 5, D_V)


# ---------------------------------------------------------------------------
# Gradient flow: only bridges, never the frozen modules
# ---------------------------------------------------------------------------


def test_backward_populates_bridge_grads_only(surgery):
    input_ids = torch.randint(0, VOCAB, (1, 3))
    embeds = surgery.embed(input_ids)
    h_q = embeds + torch.randn_like(embeds) * 0.01
    logits = surgery.lm_logits(h_q)
    loss = logits.sum()
    loss.backward()

    assert surgery.W_in.weight.grad is not None
    assert surgery.W_out.weight.grad is not None
    assert surgery.frozen_embed.weight.grad is None
    assert surgery.frozen_lm_head.weight.grad is None


def test_backward_does_not_touch_frozen_when_path_excludes_them(surgery):
    h_q = torch.randn(1, 3, D_Q)
    projected = surgery.project_to_verifier_space(h_q)
    projected.sum().backward()

    assert surgery.W_out.weight.grad is not None
    assert surgery.W_in.weight.grad is None
    assert surgery.frozen_embed.weight.grad is None
    assert surgery.frozen_lm_head.weight.grad is None


# ---------------------------------------------------------------------------
# Parameter-count introspection
# ---------------------------------------------------------------------------


def test_trainable_parameters_counts_only_bridges(surgery):
    expected = D_V * D_Q + D_Q * D_V
    assert surgery.trainable_parameters() == expected


def test_frozen_parameters_counts_embed_and_lm_head(surgery):
    expected = VOCAB * D_V + VOCAB * D_V
    assert surgery.frozen_parameters() == expected


def test_total_parameters_split_correctly(surgery):
    total = sum(p.numel() for p in surgery.parameters())
    assert surgery.trainable_parameters() + surgery.frozen_parameters() == total


# ---------------------------------------------------------------------------
# Bridge initialization
# ---------------------------------------------------------------------------


def test_bridges_initialized_to_finite_values(surgery):
    assert torch.isfinite(surgery.W_in.weight).all()
    assert torch.isfinite(surgery.W_out.weight).all()


def test_bridges_init_std_within_truncated_normal_bounds(
    config: SurgeryConfig, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor
):
    big_dim_config = SurgeryConfig(
        verifier_hidden_dim=128,
        proposer_hidden_dim=64,
        vocab_size=VOCAB,
        bridge_init_std=0.1,
    )
    big_embed = torch.randn(VOCAB, 128)
    big_head = torch.randn(VOCAB, 128)
    s = ReprAlignedSurgery(
        config=big_dim_config, embed_weight=big_embed, lm_head_weight=big_head
    )
    # truncation at +/- 2*std means abs values <= 0.2
    assert s.W_in.weight.abs().max().item() <= 0.2 + 1e-6
    assert s.W_out.weight.abs().max().item() <= 0.2 + 1e-6
    # std should be in the same ballpark as the requested 0.1
    assert 0.03 < s.W_in.weight.std().item() < 0.15
    assert 0.03 < s.W_out.weight.std().item() < 0.15


# ---------------------------------------------------------------------------
# from_weights classmethod
# ---------------------------------------------------------------------------


def test_from_weights_infers_dims_from_embed(embed_weight, lm_head_weight):
    s = ReprAlignedSurgery.from_weights(
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        proposer_hidden_dim=D_Q,
    )
    assert s.config.vocab_size == VOCAB
    assert s.config.verifier_hidden_dim == D_V
    assert s.config.proposer_hidden_dim == D_Q
    assert s.config.bridge_init_std == 0.02


def test_from_weights_propagates_init_std(embed_weight, lm_head_weight):
    s = ReprAlignedSurgery.from_weights(
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        proposer_hidden_dim=D_Q,
        bridge_init_std=0.05,
    )
    assert s.config.bridge_init_std == 0.05


def test_from_weights_rejects_non_2d_embed(lm_head_weight):
    bad = torch.randn(VOCAB * D_V)
    with pytest.raises(ValueError, match="embed_weight must be 2-D"):
        ReprAlignedSurgery.from_weights(
            embed_weight=bad,
            lm_head_weight=lm_head_weight,
            proposer_hidden_dim=D_Q,
        )


# ---------------------------------------------------------------------------
# from_verifier_module classmethod and dotted-path resolution
# ---------------------------------------------------------------------------


class _FakeQwen3Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, D_V)


class _FakeQwen3Verifier(nn.Module):
    """Minimal stand-in for a HF Qwen3 verifier's module layout.

    We only care about the two attributes the surgery extracts:
    ``model.embed_tokens`` (input embedding) and ``lm_head`` (output
    projection). Everything else a real verifier would have is
    irrelevant to Stage 1.
    """

    def __init__(self):
        super().__init__()
        self.model = _FakeQwen3Inner()
        self.lm_head = nn.Linear(D_V, VOCAB, bias=False)


def test_from_verifier_module_default_paths_work():
    verifier = _FakeQwen3Verifier()
    s = ReprAlignedSurgery.from_verifier_module(
        verifier=verifier, proposer_hidden_dim=D_Q
    )
    assert s.config.vocab_size == VOCAB
    assert s.config.verifier_hidden_dim == D_V
    assert torch.equal(s.frozen_embed.weight, verifier.model.embed_tokens.weight)
    assert torch.equal(s.frozen_lm_head.weight, verifier.lm_head.weight)


def test_from_verifier_module_custom_paths_work():
    class _AltLayout(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _FakeQwen3Inner()
            self.head = nn.Linear(D_V, VOCAB, bias=False)
            self.transformer.embed_tokens = self.transformer.embed_tokens

    verifier = _AltLayout()
    s = ReprAlignedSurgery.from_verifier_module(
        verifier=verifier,
        proposer_hidden_dim=D_Q,
        embed_module_path="transformer.embed_tokens",
        lm_head_module_path="head",
    )
    assert torch.equal(s.frozen_lm_head.weight, verifier.head.weight)


def test_from_verifier_module_propagates_bridge_init_std():
    verifier = _FakeQwen3Verifier()
    s = ReprAlignedSurgery.from_verifier_module(
        verifier=verifier, proposer_hidden_dim=D_Q, bridge_init_std=0.07
    )
    assert s.config.bridge_init_std == 0.07


def test_resolve_dotted_path_raises_on_missing_attribute():
    verifier = _FakeQwen3Verifier()
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _resolve_dotted_path(verifier, "model.nonexistent")


def test_resolve_dotted_path_raises_on_root_missing_attribute():
    verifier = _FakeQwen3Verifier()
    with pytest.raises(AttributeError, match=r"at '<root>'"):
        _resolve_dotted_path(verifier, "missing_top")


def test_resolve_dotted_path_raises_on_non_module_resolution():
    class _NotAModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.scalar = 42

    obj = _NotAModule()
    with pytest.raises(TypeError, match="resolved to int"):
        _resolve_dotted_path(obj, "scalar")


def test_extract_weight_raises_on_missing_weight_attr():
    class _NoWeight(nn.Module):
        pass

    with pytest.raises(AttributeError, match="has no 'weight' attribute"):
        _extract_weight(_NoWeight(), name="some.path")


def test_extract_weight_raises_on_non_tensor_weight():
    class _BadWeight(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = "not a tensor"

    with pytest.raises(TypeError, match="is str, expected torch.Tensor"):
        _extract_weight(_BadWeight(), name="bad")


# ---------------------------------------------------------------------------
# Tied-embedding behaviour: when verifier ties embed and lm_head, the
# surgery still produces independent frozen copies.
# ---------------------------------------------------------------------------


def test_from_verifier_module_handles_tied_embeddings():
    """If verifier ties embed and lm_head, surgery still copies independently.

    A real Qwen3 model that ties weights has
    ``model.embed_tokens.weight is lm_head.weight``. Both should
    flow into independent frozen modules in the surgery, neither
    requiring grad.
    """

    class _TiedVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _FakeQwen3Inner()
            self.lm_head = nn.Linear(D_V, VOCAB, bias=False)
            # Tie weights.
            self.lm_head.weight = self.model.embed_tokens.weight

    verifier = _TiedVerifier()
    s = ReprAlignedSurgery.from_verifier_module(
        verifier=verifier, proposer_hidden_dim=D_Q
    )
    assert s.frozen_embed.weight.data_ptr() != s.frozen_lm_head.weight.data_ptr()
    assert torch.equal(s.frozen_embed.weight, s.frozen_lm_head.weight)
    assert s.frozen_embed.weight.requires_grad is False
    assert s.frozen_lm_head.weight.requires_grad is False


# ---------------------------------------------------------------------------
# Integration sanity: a manual two-stage forward (embed -> identity backbone
# -> lm_logits) produces logits of correct shape and finite values.
# ---------------------------------------------------------------------------


def test_manual_pipeline_with_identity_backbone_produces_finite_logits(surgery):
    input_ids = torch.randint(0, VOCAB, (2, 4))

    proposer_embeds = surgery.embed(input_ids)
    # Stand-in for the proposer backbone: identity transformation.
    h_q = proposer_embeds
    logits = surgery.lm_logits(h_q)

    assert logits.shape == (2, 4, VOCAB)
    assert torch.isfinite(logits).all()
