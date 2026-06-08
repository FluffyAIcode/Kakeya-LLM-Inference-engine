"""Linux CI unit tests for inference_engine/v04/restored_attention.py.

These tests exercise the per-attention-layer K/V preparation primitive
that turns captured proposer K/V (pre-norm pre-RoPE) into merged K/V
in the post-norm post-RoPE layout the verifier's attention consumes.

Tests are pure PyTorch — no HF transformers model dependency. The
RoPE math is implemented locally with the same interleaved-half
formulation HF transformers uses; integration cross-check against
HF's ``apply_rotary_pos_emb`` lives on the K1.D Mac M4 reviewer.

Test classes:

* TestRotateHalf — the half-rotation primitive on small fixtures.
* TestApplyRopeToKAtPositions — RoPE math on K-only with synthetic
  cos/sin tables; bit-exact reference; broadcasting; rank/shape
  validation raises.
* TestSlicePositionEmbeddings — index_select wrapper with the same
  position-list contract as kv_merge.
* TestPrepareRestoredAttentionKV — end-to-end on a synthetic
  k_norm + cos/sin + captured K/V, including (a) bit-exact identity
  when k_norm is identity and RoPE rotates by zero (b) shape
  preservation (c) evicted positions take captured branch with
  correct norm + RoPE applied (d) non-evicted positions preserve
  K_local exactly (e) empty evicted list is identity (f) gradient
  flow through captured K matches K1.B's contract.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from inference_engine.v04.restored_attention import (
    apply_rope_to_k_at_positions,
    prepare_restored_attention_kv,
    slice_position_embeddings,
    _rotate_half,
)


# ---------------------------------------------------------------------------
# _rotate_half
# ---------------------------------------------------------------------------


class TestRotateHalf:
    def test_simple_4d_input(self):
        # head_dim=4, x = [1, 2, 3, 4]
        # rotate_half: [-x_2, x_1] = [-3, -4, 1, 2]
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        rotated = _rotate_half(x)
        expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])
        assert torch.equal(rotated, expected)

    def test_preserves_shape(self):
        x = torch.randn(2, 3, 4, 8)  # head_dim=8
        rotated = _rotate_half(x)
        assert rotated.shape == x.shape

    def test_preserves_dtype_and_device(self):
        x = torch.randn(1, 2, 4, dtype=torch.float64)
        rotated = _rotate_half(x)
        assert rotated.dtype == torch.float64

    def test_double_rotate_negates(self):
        """Rotating twice gives -x for the standard interleaved-half RoPE."""
        x = torch.randn(2, 4, 6, 8)
        twice = _rotate_half(_rotate_half(x))
        assert torch.allclose(twice, -x)


# ---------------------------------------------------------------------------
# apply_rope_to_k_at_positions
# ---------------------------------------------------------------------------


class TestApplyRopeToKAtPositions:
    def test_identity_when_cos_one_sin_zero(self):
        """If cos = 1 everywhere and sin = 0, RoPE is identity."""
        B, H, T, D = 1, 2, 4, 8
        k = torch.randn(B, H, T, D)
        cos = torch.ones(B, T, D)
        sin = torch.zeros(B, T, D)
        rotated = apply_rope_to_k_at_positions(k, cos, sin)
        assert torch.allclose(rotated, k)

    def test_pure_rotation_when_cos_zero_sin_one(self):
        """If cos = 0 and sin = 1, the result is rotate_half(k)."""
        B, H, T, D = 1, 2, 3, 4
        k = torch.randn(B, H, T, D)
        cos = torch.zeros(B, T, D)
        sin = torch.ones(B, T, D)
        rotated = apply_rope_to_k_at_positions(k, cos, sin)
        expected = _rotate_half(k)
        assert torch.allclose(rotated, expected)

    def test_manual_reference_on_2d(self):
        """Hand-rolled reference for T=1 head_dim=4 to pin the math
        exactly. With k = [1, 2, 3, 4], cos = [c, c, c, c], sin = [s, s, s, s]:
            rotate_half(k) = [-3, -4, 1, 2]
            result = [c*1 + s*(-3), c*2 + s*(-4), c*3 + s*1, c*4 + s*2]
        """
        c, s = 0.5, 0.7
        k = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # B=1, H=1, T=1, D=4
        cos = torch.full((1, 1, 4), c)
        sin = torch.full((1, 1, 4), s)
        rotated = apply_rope_to_k_at_positions(k, cos, sin)
        expected = torch.tensor([[[[
            c * 1 - s * 3,
            c * 2 - s * 4,
            c * 3 + s * 1,
            c * 4 + s * 2,
        ]]]])
        assert torch.allclose(rotated, expected, atol=1e-6)

    def test_broadcasting_over_heads(self):
        """cos/sin are [B, T, head_dim] and broadcast across the head
        dim of k. Two heads with different content should both rotate
        identically."""
        B, H, T, D = 1, 3, 2, 4
        k = torch.randn(B, H, T, D)
        cos = torch.randn(B, T, D)
        sin = torch.randn(B, T, D)
        rotated = apply_rope_to_k_at_positions(k, cos, sin)
        # Manual reference: for each head independently
        for h in range(H):
            k_h = k[:, h:h+1]  # [B, 1, T, D]
            ref_h = apply_rope_to_k_at_positions(k_h, cos, sin)
            assert torch.allclose(rotated[:, h:h+1], ref_h)

    def test_rank_3_k_raises(self):
        with pytest.raises(ValueError, match="must be 4-D"):
            apply_rope_to_k_at_positions(
                torch.randn(2, 4, 8),
                torch.randn(1, 4, 8),
                torch.randn(1, 4, 8),
            )

    def test_rank_4_cos_raises(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            apply_rope_to_k_at_positions(
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),  # rank 4, wrong
                torch.randn(1, 4, 8),
            )

    def test_cos_sin_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="cos shape .* != sin shape"):
            apply_rope_to_k_at_positions(
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 4, 8),
                torch.randn(1, 5, 8),
            )

    def test_cos_dimension_mismatch_with_k_raises(self):
        with pytest.raises(ValueError, match="incompatible with k shape"):
            apply_rope_to_k_at_positions(
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 4, 16),  # head_dim mismatch
                torch.randn(1, 4, 16),
            )


# ---------------------------------------------------------------------------
# slice_position_embeddings
# ---------------------------------------------------------------------------


class TestSlicePositionEmbeddings:
    def test_basic_slice(self):
        cos = torch.arange(20.0).view(1, 10, 2)
        sin = torch.arange(20.0).view(1, 10, 2) * -1
        cos_s, sin_s = slice_position_embeddings(cos, sin, [2, 5, 7])
        assert cos_s.shape == (1, 3, 2)
        assert torch.equal(cos_s[0, 0], cos[0, 2])
        assert torch.equal(cos_s[0, 1], cos[0, 5])
        assert torch.equal(cos_s[0, 2], cos[0, 7])
        assert torch.equal(sin_s[0, 0], sin[0, 2])

    def test_unsorted_positions_raises(self):
        cos = torch.randn(1, 8, 4)
        sin = torch.randn(1, 8, 4)
        with pytest.raises(ValueError, match="sorted ascending"):
            slice_position_embeddings(cos, sin, [5, 2, 7])

    def test_duplicate_positions_raises(self):
        cos = torch.randn(1, 8, 4)
        sin = torch.randn(1, 8, 4)
        with pytest.raises(ValueError, match="sorted ascending"):
            slice_position_embeddings(cos, sin, [2, 2, 5])

    def test_empty_positions_raises(self):
        cos = torch.randn(1, 8, 4)
        sin = torch.randn(1, 8, 4)
        with pytest.raises(ValueError, match="non-empty"):
            slice_position_embeddings(cos, sin, [])

    def test_negative_position_raises(self):
        cos = torch.randn(1, 8, 4)
        sin = torch.randn(1, 8, 4)
        with pytest.raises(ValueError, match="must lie in"):
            slice_position_embeddings(cos, sin, [-1, 5])

    def test_position_at_seqlen_raises(self):
        cos = torch.randn(1, 8, 4)
        sin = torch.randn(1, 8, 4)
        with pytest.raises(ValueError, match="must lie in"):
            slice_position_embeddings(cos, sin, [3, 8])

    def test_cos_sin_rank_mismatch_raises(self):
        cos = torch.randn(1, 8, 4, 4)  # rank 4
        sin = torch.randn(1, 8, 4)
        with pytest.raises(ValueError, match="must be 3-D"):
            slice_position_embeddings(cos, sin, [3])

    def test_cos_sin_shape_mismatch_raises(self):
        cos = torch.randn(1, 8, 4)
        sin = torch.randn(1, 8, 8)
        with pytest.raises(ValueError, match="cos shape .* != sin shape"):
            slice_position_embeddings(cos, sin, [3])


# ---------------------------------------------------------------------------
# prepare_restored_attention_kv — synthetic-norm + manual-cos/sin
# ---------------------------------------------------------------------------


class _IdentityNorm(nn.Module):
    """A k_norm stand-in that is the identity. Lets us check that the
    pre-norm step does not corrupt anything when the norm is a no-op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ScaleNorm(nn.Module):
    """A k_norm stand-in that multiplies by a fixed per-head-dim scale.
    Used to verify the norm is actually applied to captured K."""

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale  # broadcast over leading dims


class TestPrepareRestoredAttentionKV:
    def _make_inputs(
        self,
        B: int = 1, H: int = 2, T: int = 6, D: int = 4,
        n_evicted: int = 2,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        torch.manual_seed(seed)
        K_local = torch.randn(B, H, T, D, dtype=dtype)
        V_local = torch.randn(B, H, T, D, dtype=dtype)
        captured_K = torch.randn(B, n_evicted, H, D, dtype=dtype)
        captured_V = torch.randn(B, n_evicted, H, D, dtype=dtype)
        cos = torch.randn(B, T, D, dtype=dtype)
        sin = torch.randn(B, T, D, dtype=dtype)
        return K_local, V_local, captured_K, captured_V, cos, sin

    def test_empty_evicted_returns_clone_of_local(self):
        K_local, V_local, _, _, cos, sin = self._make_inputs(T=6, n_evicted=2)
        K_merged, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=torch.empty(1, 0, 2, 4),
            captured_V=torch.empty(1, 0, 2, 4),
            evicted_positions=[],
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        assert torch.equal(K_merged, K_local)
        assert torch.equal(V_merged, V_local)
        # Also a clone, not a view
        K_merged.fill_(999.0)
        assert (K_local != 999.0).any()

    def test_output_shape_matches_local(self):
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            B=1, H=2, T=8, D=4, n_evicted=3,
        )
        K_merged, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=[2, 4, 6],
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        assert K_merged.shape == K_local.shape
        assert V_merged.shape == V_local.shape

    def test_non_evicted_positions_preserve_K_local_exactly(self):
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            T=8, n_evicted=2,
        )
        positions = [2, 5]
        K_merged, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=positions,
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        kept = [p for p in range(8) if p not in positions]
        for p in kept:
            assert torch.equal(K_merged[:, :, p, :], K_local[:, :, p, :])
            assert torch.equal(V_merged[:, :, p, :], V_local[:, :, p, :])

    def test_evicted_K_equals_normed_then_roped_captured(self):
        """At evicted positions, K_merged should equal
        apply_rope( k_norm(captured_K), cos[evicted], sin[evicted] ).
        """
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            T=8, n_evicted=2,
        )
        positions = [3, 6]
        # Apply identity norm + manual RoPE for the reference
        K_merged, _ = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=positions,
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )

        # Manual reference: transpose captured K → [B, H, n_evicted, D]
        # → apply RoPE with cos/sin[evicted] → at-evicted-position match
        cos_e, sin_e = slice_position_embeddings(cos, sin, positions)
        captured_K_attn_layout = captured_K.transpose(1, 2)
        K_ref_at_evicted_attn = apply_rope_to_k_at_positions(
            captured_K_attn_layout, cos_e, sin_e,
        )  # [B, H, n_evicted, D]

        for slot, p in enumerate(positions):
            assert torch.allclose(
                K_merged[:, :, p, :],
                K_ref_at_evicted_attn[:, :, slot, :],
                atol=1e-6,
            )

    def test_evicted_V_equals_captured_V_no_norm_no_rope(self):
        """V at evicted positions is captured_V directly (no norm, no
        RoPE), simply transposed to attention layout."""
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            T=8, n_evicted=2,
        )
        positions = [3, 6]
        _, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=positions,
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        for slot, p in enumerate(positions):
            # captured_V is [B, n_evicted, H, D]; merged is [B, H, T, D]
            assert torch.equal(V_merged[:, :, p, :], captured_V[:, slot, :, :])

    def test_k_norm_is_actually_applied_to_captured(self):
        """If k_norm scales by 2.0, captured K at evicted should get
        the scale before RoPE."""
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            T=8, n_evicted=2, D=4,
        )
        positions = [3, 6]

        scale_factor = torch.full((4,), 2.0)
        scale_norm = _ScaleNorm(scale_factor)

        K_merged_with_scale, _ = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=positions,
            k_norm=scale_norm,
            position_embeddings=(cos, sin),
        )

        K_merged_identity, _ = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=positions,
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )

        # At evicted positions, scaled-norm result should be
        # 2x the identity-norm result (RoPE is linear in K)
        for p in positions:
            assert torch.allclose(
                K_merged_with_scale[:, :, p, :],
                2.0 * K_merged_identity[:, :, p, :],
                atol=1e-6,
            )
        # At kept positions, both are unchanged from K_local
        for p in [pp for pp in range(8) if pp not in positions]:
            assert torch.equal(K_merged_with_scale[:, :, p, :], K_local[:, :, p, :])

    def test_shape_validation_K_local_rank_3(self):
        with pytest.raises(ValueError, match="K_local must be 4-D"):
            prepare_restored_attention_kv(
                K_local=torch.randn(2, 4, 8),
                V_local=torch.randn(2, 4, 8),
                captured_K_pre_norm=torch.randn(1, 2, 2, 4),
                captured_V=torch.randn(1, 2, 2, 4),
                evicted_positions=[3, 5],
                k_norm=_IdentityNorm(),
                position_embeddings=(torch.randn(1, 8, 4), torch.randn(1, 8, 4)),
            )

    def test_shape_validation_V_local_mismatch(self):
        with pytest.raises(ValueError, match="K_local shape .* != V_local shape"):
            prepare_restored_attention_kv(
                K_local=torch.randn(1, 2, 8, 4),
                V_local=torch.randn(1, 3, 8, 4),  # different num_heads
                captured_K_pre_norm=torch.randn(1, 2, 2, 4),
                captured_V=torch.randn(1, 2, 2, 4),
                evicted_positions=[3, 5],
                k_norm=_IdentityNorm(),
                position_embeddings=(torch.randn(1, 8, 4), torch.randn(1, 8, 4)),
            )

    def test_shape_validation_captured_K_rank_3(self):
        with pytest.raises(ValueError, match="captured_K_pre_norm must be 4-D"):
            prepare_restored_attention_kv(
                K_local=torch.randn(1, 2, 8, 4),
                V_local=torch.randn(1, 2, 8, 4),
                captured_K_pre_norm=torch.randn(2, 2, 4),  # rank 3
                captured_V=torch.randn(1, 2, 2, 4),
                evicted_positions=[3, 5],
                k_norm=_IdentityNorm(),
                position_embeddings=(torch.randn(1, 8, 4), torch.randn(1, 8, 4)),
            )

    def test_gradient_flows_through_captured_K(self):
        """Cross-model f_θ training in K2/K3 needs gradient through
        the captured K branch (post-norm post-RoPE)."""
        torch.manual_seed(0)
        B, H, T, D = 1, 2, 6, 4
        K_local = torch.randn(B, H, T, D)
        V_local = torch.randn(B, H, T, D)
        captured_K = torch.randn(B, 2, H, D, requires_grad=True)
        captured_V = torch.randn(B, 2, H, D, requires_grad=True)
        cos = torch.randn(B, T, D)
        sin = torch.randn(B, T, D)
        K_merged, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=[2, 4],
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        loss = K_merged.sum() + V_merged.sum()
        loss.backward()
        assert captured_K.grad is not None
        assert captured_V.grad is not None
        # K1.B's contract: gradient on captured at all its slots
        assert (captured_K.grad != 0).any()
        assert (captured_V.grad != 0).any()

    def test_dtype_preservation_bf16(self):
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            T=6, n_evicted=2, dtype=torch.bfloat16,
        )
        K_merged, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=[1, 4],
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        assert K_merged.dtype == torch.bfloat16
        assert V_merged.dtype == torch.bfloat16

    def test_consecutive_evicted_block(self):
        """The common case: a contiguous block of evicted positions
        coming from compute_evicted_positions output for sink+window
        configuration."""
        K_local, V_local, captured_K, captured_V, cos, sin = self._make_inputs(
            T=20, n_evicted=8,
        )
        # sink=4, window=8, T=20 → evicted = [4..11]
        positions = list(range(4, 12))
        K_merged, V_merged = prepare_restored_attention_kv(
            K_local=K_local, V_local=V_local,
            captured_K_pre_norm=captured_K,
            captured_V=captured_V,
            evicted_positions=positions,
            k_norm=_IdentityNorm(),
            position_embeddings=(cos, sin),
        )
        # All sink+window positions preserved
        for p in list(range(4)) + list(range(12, 20)):
            assert torch.equal(K_merged[:, :, p, :], K_local[:, :, p, :])
            assert torch.equal(V_merged[:, :, p, :], V_local[:, :, p, :])
        # All evicted positions overridden by captured (V no norm/RoPE)
        for slot, p in enumerate(positions):
            assert torch.equal(V_merged[:, :, p, :], captured_V[:, slot, :, :])
