"""Linux CI tests for inference_engine.v04.f_theta.

Covers the shape contract, parameter counts, save/load round-trip,
and dtype/device dispatch. No actual training — that's exercised by
scripts/research/k3_f_theta_train.py + the integration evidence run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from inference_engine.v04.f_theta import FThetaConfig, FThetaProjection


def _gemma4_dflash_config(rank: int = 256) -> FThetaConfig:
    """Production K3 config: Gemma 4 26B-A4B verifier + DFlash 0.4B drafter."""
    return FThetaConfig(
        drafter_num_layers=5,
        drafter_num_kv_heads=2,
        drafter_head_dim=128,
        verifier_num_layers=30,
        verifier_num_kv_heads=8,
        verifier_head_dim=256,
        rank=rank,
    )


def _tiny_config() -> FThetaConfig:
    """Tiny config for fast tests."""
    return FThetaConfig(
        drafter_num_layers=2,
        drafter_num_kv_heads=2,
        drafter_head_dim=4,
        verifier_num_layers=3,
        verifier_num_kv_heads=4,
        verifier_head_dim=8,
        rank=16,
    )


class TestFThetaConfig:

    def test_drafter_kv_dim(self):
        c = _tiny_config()
        assert c.drafter_kv_dim == 2 * 4

    def test_verifier_kv_dim(self):
        c = _tiny_config()
        assert c.verifier_kv_dim == 4 * 8

    def test_encoder_in_features(self):
        c = _tiny_config()
        assert c.encoder_in_features == 2 * (2 * 4)

    def test_production_dimensions(self):
        c = _gemma4_dflash_config()
        assert c.drafter_kv_dim == 256
        assert c.verifier_kv_dim == 2048
        assert c.encoder_in_features == 5 * 256

    def test_to_from_json_round_trip(self):
        c1 = _gemma4_dflash_config(rank=128)
        d = c1.to_json_dict()
        c2 = FThetaConfig.from_json_dict(d)
        assert c1 == c2


class TestForwardShapes:

    def test_forward_k_shape(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        B, T = 2, 7
        x = torch.randn(B, T, c.encoder_in_features)
        y = m.forward_k(x)
        assert tuple(y.shape) == (
            B, T, c.verifier_num_layers, c.verifier_num_kv_heads, c.verifier_head_dim,
        )

    def test_forward_v_shape(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        B, T = 1, 3
        x = torch.randn(B, T, c.encoder_in_features)
        y = m.forward_v(x)
        assert tuple(y.shape) == (
            B, T, c.verifier_num_layers, c.verifier_num_kv_heads, c.verifier_head_dim,
        )

    def test_forward_k_rejects_wrong_rank(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        with pytest.raises(ValueError, match="expected"):
            m.forward_k(torch.randn(c.encoder_in_features))  # 1-D input

    def test_forward_k_rejects_wrong_feature_dim(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        with pytest.raises(ValueError, match="encoder_in_features"):
            m.forward_k(torch.randn(2, 7, c.encoder_in_features + 1))


class TestForwardKVPack:
    """forward_kv_pack accepts the natural KVCapture layout
    [B, T, num_kv_heads, head_dim] per layer (list of tensors).
    """

    def test_returns_paired_k_v(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        B, T = 2, 5
        k_per_layer = [
            torch.randn(B, T, c.drafter_num_kv_heads, c.drafter_head_dim)
            for _ in range(c.drafter_num_layers)
        ]
        v_per_layer = [
            torch.randn(B, T, c.drafter_num_kv_heads, c.drafter_head_dim)
            for _ in range(c.drafter_num_layers)
        ]
        k_out, v_out = m.forward_kv_pack(k_per_layer, v_per_layer)
        expected = (B, T, c.verifier_num_layers, c.verifier_num_kv_heads, c.verifier_head_dim)
        assert tuple(k_out.shape) == expected
        assert tuple(v_out.shape) == expected

    def test_rejects_wrong_layer_count(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        B, T = 1, 3
        k_per_layer = [
            torch.randn(B, T, c.drafter_num_kv_heads, c.drafter_head_dim)
            for _ in range(c.drafter_num_layers - 1)  # one short
        ]
        v_per_layer = [
            torch.randn(B, T, c.drafter_num_kv_heads, c.drafter_head_dim)
            for _ in range(c.drafter_num_layers)
        ]
        with pytest.raises(ValueError, match="drafter layers"):
            m.forward_kv_pack(k_per_layer, v_per_layer)

    def test_consistency_with_explicit_concat(self):
        """forward_kv_pack must equal forward_k(flatten + concat) explicitly."""
        c = _tiny_config()
        torch.manual_seed(0)
        m = FThetaProjection(c)
        m.eval()
        B, T = 2, 4
        k_per_layer = [
            torch.randn(B, T, c.drafter_num_kv_heads, c.drafter_head_dim)
            for _ in range(c.drafter_num_layers)
        ]
        v_per_layer = [
            torch.randn(B, T, c.drafter_num_kv_heads, c.drafter_head_dim)
            for _ in range(c.drafter_num_layers)
        ]
        k_out_pack, v_out_pack = m.forward_kv_pack(k_per_layer, v_per_layer)
        k_concat = torch.cat([k.flatten(-2, -1) for k in k_per_layer], dim=-1)
        v_concat = torch.cat([v.flatten(-2, -1) for v in v_per_layer], dim=-1)
        with torch.no_grad():
            k_out_direct = m.forward_k(k_concat)
            v_out_direct = m.forward_v(v_concat)
        assert torch.allclose(k_out_pack, k_out_direct, atol=1e-6)
        assert torch.allclose(v_out_pack, v_out_direct, atol=1e-6)


class TestParameterCount:
    """Lock the parameter-count contract so future architecture changes
    are explicit (not silent regressions in training cost)."""

    def test_tiny_param_count(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        # encoder_k, encoder_v: 2 × (encoder_in × rank) = 2 × 16×16 = 512
        # decoders_k: 3 × (rank × verifier_kv_dim) = 3 × 16×32 = 1536
        # decoders_v: same = 1536
        # Total: 512 + 1536 + 1536 = 3584
        n = sum(p.numel() for p in m.parameters())
        assert n == 512 + 1536 + 1536

    def test_production_param_count_in_expected_range(self):
        """Production f_θ should be ~31.8M params (rank=256)."""
        c = _gemma4_dflash_config(rank=256)
        m = FThetaProjection(c)
        n = sum(p.numel() for p in m.parameters())
        # encoder_k + encoder_v: 2 * 5 * 256 * 256 = 655,360
        # decoders_k: 30 * 256 * 2048 = 15,728,640
        # decoders_v: same = 15,728,640
        # total ≈ 32,112,640
        assert 30_000_000 < n < 35_000_000


class TestSaveLoadRoundTrip:

    def test_save_and_load_preserves_outputs(self, tmp_path):
        c = _tiny_config()
        torch.manual_seed(42)
        m1 = FThetaProjection(c).eval()
        # Run a forward, snapshot output
        B, T = 1, 3
        x_k = torch.randn(B, T, c.encoder_in_features)
        x_v = torch.randn(B, T, c.encoder_in_features)
        with torch.no_grad():
            y_k_1 = m1.forward_k(x_k)
            y_v_1 = m1.forward_v(x_v)

        # Save
        m1.save_pretrained(tmp_path)
        assert (tmp_path / "f_theta_config.json").is_file()
        assert (tmp_path / "f_theta_weights.pt").is_file()

        # Load
        m2 = FThetaProjection.from_pretrained(tmp_path)
        assert m2.config == m1.config
        with torch.no_grad():
            y_k_2 = m2.forward_k(x_k)
            y_v_2 = m2.forward_v(x_v)

        assert torch.allclose(y_k_1, y_k_2)
        assert torch.allclose(y_v_1, y_v_2)

    def test_load_rejects_missing_config(self, tmp_path):
        # Write only weights, no config
        m = FThetaProjection(_tiny_config())
        torch.save(m.state_dict(), tmp_path / "f_theta_weights.pt")
        with pytest.raises(FileNotFoundError, match="f_theta_config.json"):
            FThetaProjection.from_pretrained(tmp_path)

    def test_load_rejects_missing_weights(self, tmp_path):
        c = _tiny_config()
        import json
        (tmp_path / "f_theta_config.json").write_text(
            json.dumps(c.to_json_dict()),
        )
        with pytest.raises(FileNotFoundError, match="f_theta_weights.pt"):
            FThetaProjection.from_pretrained(tmp_path)

    def test_load_rejects_non_directory(self):
        with pytest.raises(FileNotFoundError, match="must be a directory"):
            FThetaProjection.from_pretrained("/tmp/not_a_real_directory")


class TestDeviceDtypeDispatch:

    def test_to_dtype(self):
        m = FThetaProjection(_tiny_config())
        m_bf16 = m.to(torch.bfloat16)
        for p in m_bf16.parameters():
            assert p.dtype == torch.bfloat16

    def test_load_with_dtype_override(self, tmp_path):
        m1 = FThetaProjection(_tiny_config())
        m1.save_pretrained(tmp_path)
        m2 = FThetaProjection.from_pretrained(tmp_path, dtype=torch.bfloat16)
        for p in m2.parameters():
            assert p.dtype == torch.bfloat16


class TestGradientFlow:
    """f_θ must be trainable end-to-end. Verify gradients flow through
    encoder + decoders during a backward pass."""

    def test_gradients_flow_for_k_path(self):
        c = _tiny_config()
        m = FThetaProjection(c)
        B, T = 1, 3
        x = torch.randn(B, T, c.encoder_in_features, requires_grad=False)
        target = torch.randn(
            B, T, c.verifier_num_layers,
            c.verifier_num_kv_heads, c.verifier_head_dim,
        )
        out = m.forward_k(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        # encoder_k should have a grad
        assert m.encoder_k.weight.grad is not None
        assert m.encoder_k.weight.grad.abs().sum() > 0
        # All K decoders should have grads
        for dec in m.decoders_k:
            assert dec.weight.grad is not None
            assert dec.weight.grad.abs().sum() > 0
        # encoder_v / decoders_v should NOT (separate path)
        assert m.encoder_v.weight.grad is None
        for dec in m.decoders_v:
            assert dec.weight.grad is None
