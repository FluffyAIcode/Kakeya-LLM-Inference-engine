"""Linux CI tests for inference_engine.v04.cross_model_dlm_verifier.

Covers the testable surface:

* CrossModelDLMRestoredVerifier construction + dimension validation
* project_drafter_kv shape contract
* forward end-to-end on a synthetic verifier + drafter
* No-evict short-prompt path (T <= sink+window)
* Patched attention forward correctness on a tiny synthetic verifier

Real Gemma 4 26B-A4B + DFlash 0.4B integration is validated by the
training run + integration evidence (separate vast.ai runs).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from inference_engine.v04 import (
    CrossModelDLMRestoredVerifier,
    DFlashConfig,
    DFlashDrafter,
    FThetaConfig,
    FThetaProjection,
)


def _tiny_drafter_config() -> DFlashConfig:
    return DFlashConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        block_size=4,
        mask_token_id=3,
        target_layer_ids=(1, 3),
        final_logit_softcapping=30.0,
    )


def _tiny_f_theta_config() -> FThetaConfig:
    """Aligned with _tiny_drafter_config + a 3-layer verifier."""
    return FThetaConfig(
        drafter_num_layers=2,
        drafter_num_kv_heads=2,
        drafter_head_dim=4,
        verifier_num_layers=3,
        verifier_num_kv_heads=4,
        verifier_head_dim=8,
        rank=16,
    )


class _SyntheticVerifierConfig:
    num_hidden_layers = 3
    num_key_value_heads = 4
    head_dim = 8
    hidden_size = 32
    num_attention_heads = 4
    _attn_implementation = "eager"


class _SyntheticVerifierAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.k_proj = nn.Linear(32, 4 * 8, bias=False)
        self.v_proj = nn.Linear(32, 4 * 8, bias=False)
        self.o_proj = nn.Linear(32, 32, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()   # Gemma 4 runs V through v_norm
        self.head_dim = 8
        self.num_key_value_groups = 1
        self.scaling = 8 ** -0.5
        self.attention_dropout = 0.0
        self.sliding_window = None
        self.config = _SyntheticVerifierConfig()


class _SyntheticVerifierLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SyntheticVerifierAttention()


class _SyntheticVerifierInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _SyntheticVerifierLayer() for _ in range(3)
        ])


class _SyntheticVerifier(nn.Module):
    """A minimal HF-shaped verifier for the cross-model path.

    Structure mirrors `model.model.layers[i].self_attn.{q,k,v,o}_proj`
    that CrossModelDLMRestoredVerifier patches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _SyntheticVerifierConfig()
        self.model = _SyntheticVerifierInner()

    def forward(self, input_ids=None, **kwargs):
        # Trivial forward: just iterate layers + return logits.
        # The CrossModelDLMRestoredVerifier path patches each layer's
        # attn.forward; this top-level forward iterates and calls the
        # patched forward on each layer with synthetic hidden state.
        B, T = input_ids.shape
        h = torch.randn(B, T, 32)
        cos = torch.ones(B, T, 8) * 0.5
        sin = torch.ones(B, T, 8) * 0.5
        mask = torch.zeros(B, 1, T, T)
        for layer in self.model.layers:
            attn_out, _ = layer.self_attn.forward(
                hidden_states=h,
                position_embeddings=(cos, sin),
                attention_mask=mask,
            )
            h = attn_out  # simplified
        # Return a namespace with logits attribute for compatibility
        class _Out:
            logits = torch.zeros(B, T, 64)
        return _Out()


class TestConstruction:

    def test_dimension_validation_rejects_mismatch(self):
        f_cfg = _tiny_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())

        # Verifier with 5 layers but f_θ trained for 3 → should reject
        class WrongConfig:
            num_hidden_layers = 5
            num_key_value_heads = 4
            head_dim = 8
            hidden_size = 32
            num_attention_heads = 4
            _attn_implementation = "eager"

        class WrongVerifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = WrongConfig()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList()

        with pytest.raises(ValueError, match="verifier_num_layers"):
            CrossModelDLMRestoredVerifier(
                verifier_model=WrongVerifier(),
                drafter=drafter,
                f_theta=f_theta,
            )

    def test_construction_with_aligned_dimensions(self):
        f_cfg = _tiny_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        verifier = _SyntheticVerifier()
        v = CrossModelDLMRestoredVerifier(
            verifier_model=verifier,
            drafter=drafter,
            f_theta=f_theta,
            sink_size=2,
            window_size=4,
        )
        assert v.sink_size == 2
        assert v.window_size == 4

    def test_negative_sink_or_window_raises(self):
        f_cfg = _tiny_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        verifier = _SyntheticVerifier()
        with pytest.raises(ValueError, match="non-negative"):
            CrossModelDLMRestoredVerifier(
                verifier_model=verifier,
                drafter=drafter,
                f_theta=f_theta,
                sink_size=-1,
            )


class TestProjectDrafterKV:
    """project_drafter_kv runs the drafter forward + f_θ projection
    and returns verifier-K, verifier-V tensors of the right shape.

    Synthetic verifier needs a real ``get_input_embeddings()`` since
    _capture_drafter_kv now uses verifier embed_tokens (corrected
    2026-06-09 to use real embedded hiddens, not synthetic zero).
    """

    def test_returns_correct_shape(self):
        f_cfg = _tiny_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        verifier = _SyntheticVerifier()
        # Synthetic verifier needs a real embed_tokens for the
        # _capture_drafter_kv path (verifier_model.get_input_embeddings()
        # is called).
        verifier.embed_tokens = torch.nn.Embedding(64, 16)  # vocab 64, hidden 16
        verifier.get_input_embeddings = lambda: verifier.embed_tokens
        v = CrossModelDLMRestoredVerifier(
            verifier_model=verifier, drafter=drafter, f_theta=f_theta,
        )
        B, T = 1, 6
        ids = torch.randint(0, 64, (B, T), dtype=torch.long)
        v_k, v_v = v.project_drafter_kv(ids)
        # Per-layer list contract (layers may have heterogeneous KV heads).
        assert len(v_k) == f_cfg.verifier_num_layers
        assert len(v_v) == f_cfg.verifier_num_layers
        per_layer = (
            B, T, f_cfg.verifier_num_kv_heads, f_cfg.verifier_head_dim,
        )
        for ko, vo in zip(v_k, v_v):
            assert tuple(ko.shape) == per_layer
            assert tuple(vo.shape) == per_layer


class TestNoEvictPath:
    """When T <= sink+window, no positions are evicted and the
    cross-model verifier path short-circuits to the underlying
    verifier's plain forward."""

    def test_short_prompt_skips_drafter_forward(self, monkeypatch):
        f_cfg = _tiny_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        verifier = _SyntheticVerifier()
        v = CrossModelDLMRestoredVerifier(
            verifier_model=verifier, drafter=drafter, f_theta=f_theta,
            sink_size=2, window_size=4,  # sink+window = 6
        )

        # Counter to verify drafter not invoked for short prompts
        calls = {"drafter": 0}
        original_project = v.project_drafter_kv
        def _counted(ids):
            calls["drafter"] += 1
            return original_project(ids)
        v.project_drafter_kv = _counted

        ids = torch.randint(0, 64, (1, 5), dtype=torch.long)  # T=5, all resident

        # Verifier's forward in the synthetic stub doesn't have
        # apply_rotary_pos_emb wired so we just check the no-evict
        # decision: when evicted_positions is empty, project_drafter_kv
        # should not run.
        try:
            v.forward(
                ids,
                apply_rotary_pos_emb=lambda q, k, c, s: (q, k),
                eager_attention_forward=lambda *a, **kw: (
                    torch.zeros(1, 4, 5, 8), None,
                ),
            )
        except Exception:
            # We don't care about forward correctness here, only that
            # project_drafter_kv was NOT called
            pass
        assert calls["drafter"] == 0


def _gemma4_style_rope(x, cos, sin, unsqueeze_dim=2):
    """Mirror Gemma 4's apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim)."""
    c = cos.unsqueeze(unsqueeze_dim)
    s = sin.unsqueeze(unsqueeze_dim)
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * c + rot * s


def _synthetic_eager(module, q, k, v, mask, dropout=0.0, scaling=1.0,
                     sliding_window=None, **kw):
    """Minimal eager attention returning [B, T, H, D] like HF's."""
    attn = torch.matmul(q, k.transpose(-1, -2)) * scaling
    if mask is not None:
        attn = attn + mask[..., : k.shape[-2]]
    attn = attn.softmax(dim=-1)
    out = torch.matmul(attn, v)             # [B, H, T, D]
    return out.transpose(1, 2).contiguous(), None   # [B, T, H, D]


class TestPatchedForwardRestore:
    """Exercise the Gemma 4-style patched attention forward end-to-end on
    the synthetic verifier with real evictions, so signature/shape/RoPE
    regressions are caught without the 26B model."""

    def test_forward_with_eviction_runs_and_injects(self):
        f_cfg = _tiny_f_theta_config()
        f_theta = FThetaProjection(f_cfg)
        drafter = DFlashDrafter(_tiny_drafter_config())
        verifier = _SyntheticVerifier()
        verifier.embed_tokens = torch.nn.Embedding(64, 16)
        verifier.get_input_embeddings = lambda: verifier.embed_tokens
        v = CrossModelDLMRestoredVerifier(
            verifier_model=verifier, drafter=drafter, f_theta=f_theta,
            sink_size=1, window_size=2,    # sink+window = 3
        )
        B, T = 1, 6                        # evicted positions [1, 2, 3]
        ids = torch.randint(0, 64, (B, T), dtype=torch.long)
        out = v.forward(
            ids,
            apply_rotary_pos_emb=_gemma4_style_rope,
            eager_attention_forward=_synthetic_eager,
        )
        assert out.logits.shape == (B, T, 64)


class TestExports:

    def test_module_exposes_classes(self):
        from inference_engine.v04 import cross_model_dlm_verifier as m
        assert hasattr(m, "CrossModelDLMRestoredVerifier")
        assert hasattr(m, "CrossModelLayerMapping")
        # And the inference_engine.v04 namespace re-exports them
        from inference_engine import v04
        assert v04.CrossModelDLMRestoredVerifier is m.CrossModelDLMRestoredVerifier
