"""Linux CI tests for inference_engine.v04.cross_model_dlm_verifier_mlx.

Covers ONLY what's testable without mlx (Apple Silicon only):

* Module import + public API surface
* `_MLXLayerWiring` dataclass shape
* Dimension validation contract on construction (uses synthetic
  drafter + a stub mlx_verifier with the right `.config` /
  `.model.layers` shape)

Mac M4-only validation is via end-to-end run of
``scripts/research/k3_integrated_niah_eval_mac.py`` producing
acceptance + recall + memory evidence (gate booleans in the JSON).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from inference_engine.v04 import DFlashDrafter, FThetaProjection
from inference_engine.v04.cross_model_dlm_verifier_mlx import (
    MLXCrossModelDLMRestoredVerifier,
    _MLXLayerWiring,
)
from inference_engine.v04.dflash_drafter import DFlashConfig
from inference_engine.v04.f_theta import FThetaConfig


# ---------------------------------------------------------------------------
# Synthetic stubs that mimic the mlx_lm Gemma 4 Model surface enough
# for dimension validation to run.
# ---------------------------------------------------------------------------


class _StubAttention:
    def __init__(self, n_kv_heads: int, head_dim: int, has_kv: bool, is_sliding: bool, use_k_eq_v: bool):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.has_kv = has_kv
        self.is_sliding = is_sliding
        self.use_k_eq_v = use_k_eq_v


class _StubLayer:
    def __init__(self, n_kv_heads: int, head_dim: int, has_kv: bool = True,
                 is_sliding: bool = False, use_k_eq_v: bool = False):
        self.self_attn = _StubAttention(
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            has_kv=has_kv, is_sliding=is_sliding, use_k_eq_v=use_k_eq_v,
        )


class _StubInner:
    def __init__(self, layers):
        self.layers = layers


class _StubMLXVerifier:
    """Minimal stub of mlx_lm.load() return value for dimension-validation
    tests. Real mlx verifier has nested structure: model.language_model.model.layers
    OR model.model.layers depending on multimodal vs text-only wrapper."""

    def __init__(self, num_layers: int, n_kv_heads: int, head_dim: int):
        # Use the .model.layers shape (text-only wrapper)
        self.model = _StubInner([
            _StubLayer(n_kv_heads=n_kv_heads, head_dim=head_dim)
            for _ in range(num_layers)
        ])


def _tiny_drafter_config() -> DFlashConfig:
    return DFlashConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, intermediate_size=32,
        vocab_size=64, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=64, block_size=4, mask_token_id=3,
        target_layer_ids=(1, 3), final_logit_softcapping=30.0,
    )


def _aligned_f_theta_config() -> FThetaConfig:
    return FThetaConfig(
        drafter_num_layers=2, drafter_num_kv_heads=2, drafter_head_dim=4,
        verifier_num_layers=3, verifier_num_kv_heads=4, verifier_head_dim=8,
        rank=16,
    )


class TestModuleSurface:

    def test_imports(self):
        from inference_engine.v04 import cross_model_dlm_verifier_mlx as m
        assert hasattr(m, "MLXCrossModelDLMRestoredVerifier")
        assert hasattr(m, "_MLXLayerWiring")

    def test_public_class_signature_stable(self):
        import inspect
        sig = inspect.signature(MLXCrossModelDLMRestoredVerifier.__init__)
        params = sig.parameters
        for needed in ("mlx_verifier", "drafter", "f_theta",
                       "sink_size", "window_size"):
            assert needed in params, f"missing {needed}"


class TestLayerWiringDataclass:

    def test_fields_present(self):
        w = _MLXLayerWiring(
            layer_idx=0, has_kv=True, is_sliding=False,
            use_k_eq_v=False, n_kv_heads=4, head_dim=8,
        )
        assert w.layer_idx == 0
        assert w.has_kv is True
        assert w.n_kv_heads == 4


class TestConstruction:

    def test_aligned_dimensions_construct(self):
        f_cfg = _aligned_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        # Stub mlx verifier matching f_θ verifier dims
        mlx_verifier = _StubMLXVerifier(
            num_layers=f_cfg.verifier_num_layers,
            n_kv_heads=f_cfg.verifier_num_kv_heads,
            head_dim=f_cfg.verifier_head_dim,
        )
        v = MLXCrossModelDLMRestoredVerifier(
            mlx_verifier=mlx_verifier,
            drafter=drafter,
            f_theta=f_theta,
            sink_size=2, window_size=4,
        )
        assert v.sink_size == 2
        assert len(v._wirings) == f_cfg.verifier_num_layers
        assert v._wirings[0].n_kv_heads == f_cfg.verifier_num_kv_heads

    def test_layer_count_mismatch_rejected(self):
        f_cfg = _aligned_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        mlx_verifier = _StubMLXVerifier(
            num_layers=f_cfg.verifier_num_layers + 1,  # mismatch
            n_kv_heads=f_cfg.verifier_num_kv_heads,
            head_dim=f_cfg.verifier_head_dim,
        )
        with pytest.raises(ValueError, match="verifier_num_layers"):
            MLXCrossModelDLMRestoredVerifier(
                mlx_verifier=mlx_verifier, drafter=drafter, f_theta=f_theta,
            )

    def test_drafter_layer_count_mismatch_rejected(self):
        f_cfg = _aligned_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter_cfg = DFlashConfig(
            hidden_size=16, num_hidden_layers=3,  # mismatch (f_cfg expects 2)
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            intermediate_size=32, vocab_size=64, rms_norm_eps=1e-6,
            rope_theta=10000.0, max_position_embeddings=64,
            block_size=4, mask_token_id=3, target_layer_ids=(1, 3),
            final_logit_softcapping=30.0,
        )
        drafter = DFlashDrafter(drafter_cfg)
        mlx_verifier = _StubMLXVerifier(
            num_layers=f_cfg.verifier_num_layers,
            n_kv_heads=f_cfg.verifier_num_kv_heads,
            head_dim=f_cfg.verifier_head_dim,
        )
        with pytest.raises(ValueError, match="drafter_num_layers"):
            MLXCrossModelDLMRestoredVerifier(
                mlx_verifier=mlx_verifier, drafter=drafter, f_theta=f_theta,
            )

    def test_negative_sink_or_window_rejected(self):
        f_cfg = _aligned_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        mlx_verifier = _StubMLXVerifier(
            num_layers=f_cfg.verifier_num_layers,
            n_kv_heads=f_cfg.verifier_num_kv_heads,
            head_dim=f_cfg.verifier_head_dim,
        )
        with pytest.raises(ValueError, match="non-negative"):
            MLXCrossModelDLMRestoredVerifier(
                mlx_verifier=mlx_verifier, drafter=drafter, f_theta=f_theta,
                sink_size=-1,
            )

    def test_missing_layers_attr_rejected(self):
        f_cfg = _aligned_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        # Verifier with no .model.layers
        bad = type("Bad", (), {})()
        with pytest.raises(AttributeError, match="layers"):
            MLXCrossModelDLMRestoredVerifier(
                mlx_verifier=bad, drafter=drafter, f_theta=f_theta,
            )


class TestLayerWiringDerivation:

    def test_default_attention_attrs(self):
        f_cfg = _aligned_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        mlx_verifier = _StubMLXVerifier(
            num_layers=3, n_kv_heads=4, head_dim=8,
        )
        v = MLXCrossModelDLMRestoredVerifier(
            mlx_verifier=mlx_verifier, drafter=drafter, f_theta=f_theta,
        )
        # Default stub: has_kv=True, is_sliding=False, use_k_eq_v=False
        for w in v._wirings:
            assert w.has_kv is True
            assert w.is_sliding is False
            assert w.use_k_eq_v is False
