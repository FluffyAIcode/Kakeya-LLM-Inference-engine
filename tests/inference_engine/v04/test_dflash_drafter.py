"""Linux CI unit tests for the K3 native DFlash drafter (Stage 1).

These cover the deterministic, model-free parts of
``inference_engine/v04/dflash_drafter.py``:

* ``DFlashConfig`` parsing incl. the +1 aux-layer shift (PR #41703).
* ``DFlashDrafter`` shapes: aux projection, backbone forward, weight
  layout (HF state-dict load), block-diffusion ``draft_block``.
* ``DFlashProposer`` conforming to the engine ``propose_block`` contract.

No HF model downloads — a tiny synthetic config + synthetic verifier
embed/lm_head are used so the suite runs in well under a second. The
trained-weight acceptance profile is the Stage-2 H200 validation task.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from inference_engine.v04.dflash_drafter import (
    AuxHiddenProvider,
    DFlashConfig,
    DFlashDrafter,
    DFlashProposer,
)
from kv_cache_proposer.proposer import BlockProposal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_hf_config() -> dict:
    return {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "final_logit_softcapping": 30.0,
        "dflash_config": {
            "block_size": 4,
            "mask_token_id": 3,
            "target_layer_ids": [1, 3],
        },
    }


def _tiny_cfg() -> DFlashConfig:
    return DFlashConfig.from_hf_config(_tiny_hf_config())


def _synthetic_verifier_heads(cfg: DFlashConfig):
    """Return (embed_fn, lm_head_fn) backed by a tiny shared embedding."""
    torch.manual_seed(0)
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        return embed(ids)

    def lm_head_fn(h: torch.Tensor) -> torch.Tensor:
        logits = head(h)
        if cfg.final_logit_softcapping is not None:
            cap = cfg.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits

    return embed_fn, lm_head_fn


class _SyntheticAuxProvider(AuxHiddenProvider):
    def __init__(self, cfg: DFlashConfig):
        self.cfg = cfg

    def aux_hidden_context(self, committed_token_ids):
        torch.manual_seed(len(committed_token_ids) + 1)
        C = len(committed_token_ids)
        return [
            torch.randn(1, C, self.cfg.hidden_size)
            for _ in range(self.cfg.num_aux_layers)
        ]


# ---------------------------------------------------------------------------
# DFlashConfig
# ---------------------------------------------------------------------------


class TestDFlashConfig:
    def test_parses_core_fields(self):
        cfg = _tiny_cfg()
        assert cfg.hidden_size == 16
        assert cfg.num_hidden_layers == 2
        assert cfg.block_size == 4
        assert cfg.mask_token_id == 3
        assert cfg.final_logit_softcapping == 30.0

    def test_aux_layer_ids_are_shifted_plus_one(self):
        """PR #41703: HF DFlash semantics shift target_layer_ids by +1."""
        cfg = _tiny_cfg()
        assert cfg.target_layer_ids == (1, 3)
        assert cfg.aux_layer_ids == (2, 4)

    def test_num_aux_and_fc_in_features(self):
        cfg = _tiny_cfg()
        assert cfg.num_aux_layers == 2
        assert cfg.fc_in_features == 2 * 16

    def test_real_gemma4_dflash_layer_ids(self):
        """The real checkpoint's [1,6,11,17,22,27] -> [2,7,12,18,23,28]."""
        hf = _tiny_hf_config()
        hf["dflash_config"]["target_layer_ids"] = [1, 6, 11, 17, 22, 27]
        cfg = DFlashConfig.from_hf_config(hf)
        assert cfg.aux_layer_ids == (2, 7, 12, 18, 23, 28)
        assert cfg.num_aux_layers == 6

    def test_missing_target_layer_ids_raises(self):
        hf = _tiny_hf_config()
        hf["dflash_config"].pop("target_layer_ids")
        with pytest.raises(ValueError, match="target_layer_ids"):
            DFlashConfig.from_hf_config(hf)

    def test_missing_mask_token_raises(self):
        hf = _tiny_hf_config()
        hf["dflash_config"].pop("mask_token_id")
        with pytest.raises(ValueError, match="mask_token_id"):
            DFlashConfig.from_hf_config(hf)


# ---------------------------------------------------------------------------
# DFlashDrafter — shapes / projection / backbone
# ---------------------------------------------------------------------------


class TestDFlashDrafterProjection:
    def test_combine_aux_shape(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        aux = [torch.randn(1, 5, cfg.hidden_size) for _ in range(cfg.num_aux_layers)]
        out = m.combine_aux(aux)  # fc only (no hidden_norm)
        assert out.shape == (1, 5, cfg.hidden_size)
        assert torch.isfinite(out).all()

    def test_combine_aux_wrong_count_raises(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        with pytest.raises(ValueError, match="aux hidden states"):
            m.combine_aux([torch.randn(1, 5, cfg.hidden_size)])  # only 1, need 2

    def test_precompute_context_kv_shapes(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        C = 7
        ctx = torch.randn(1, C, cfg.hidden_size)
        kv = m.precompute_context_kv(ctx, torch.arange(C))
        assert len(kv) == cfg.num_hidden_layers
        for ck, cv in kv:
            assert ck.shape == (1, cfg.num_key_value_heads, C, cfg.head_dim)
            assert cv.shape == (1, cfg.num_key_value_heads, C, cfg.head_dim)
            assert torch.isfinite(ck).all() and torch.isfinite(cv).all()


# ---------------------------------------------------------------------------
# Weight layout — HF state-dict load
# ---------------------------------------------------------------------------


class TestWeightLoading:
    def test_load_matching_state_dict(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        # A state dict with exactly the model's param names + shapes.
        state = {k: torch.randn_like(p) for k, p in m.named_parameters()}
        m.load_state_dict_from_hf(state, strict=True)
        for k, p in m.named_parameters():
            assert torch.allclose(p, state[k])

    def test_expected_hf_weight_names_present(self):
        """Guards the HF layout contract: fc/hidden_norm/norm + per-layer
        qwen3 names must exist (so from_pretrained over the real
        safetensors maps 1:1)."""
        cfg = _tiny_cfg()
        names = set(dict(DFlashDrafter(cfg).named_parameters()))
        assert "fc.weight" in names
        assert "hidden_norm.weight" in names
        assert "norm.weight" in names
        for i in range(cfg.num_hidden_layers):
            for sub in [
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            ]:
                assert f"layers.{i}.{sub}" in names

    def test_strict_load_rejects_mismatch(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        state = {k: torch.randn_like(p) for k, p in m.named_parameters()}
        state["bogus.weight"] = torch.randn(3)
        with pytest.raises(ValueError, match="mismatch"):
            m.load_state_dict_from_hf(state, strict=True)

    def test_fc_shape_matches_real_checkpoint_convention(self):
        """fc is Linear(num_aux*hidden -> hidden); real ckpt fc.weight is
        [hidden, num_aux*hidden]."""
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        assert m.fc.weight.shape == (cfg.hidden_size, cfg.fc_in_features)


# ---------------------------------------------------------------------------
# Block-diffusion draft_block
# ---------------------------------------------------------------------------


class TestDraftBlock:
    def _ctx(self, cfg, C=6):
        return [torch.randn(1, C, cfg.hidden_size) for _ in range(cfg.num_aux_layers)]

    def test_returns_block_size_non_mask_tokens(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        embed_fn, lm_head_fn = _synthetic_verifier_heads(cfg)
        toks = m.draft_block(self._ctx(cfg), 7, embed_fn, lm_head_fn, block_size=4)
        assert len(toks) == 4
        assert all(isinstance(t, int) for t in toks)
        assert all(t != cfg.mask_token_id for t in toks), (
            "draft must not emit the mask token"
        )
        assert all(0 <= t < cfg.vocab_size for t in toks)

    def test_single_forward_various_block_sizes(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        embed_fn, lm_head_fn = _synthetic_verifier_heads(cfg)
        for L in (1, 2, 3, 4):
            toks = m.draft_block(self._ctx(cfg), 5, embed_fn, lm_head_fn, block_size=L)
            assert len(toks) == L

    def test_invalid_args_raise(self):
        cfg = _tiny_cfg()
        m = DFlashDrafter(cfg)
        embed_fn, lm_head_fn = _synthetic_verifier_heads(cfg)
        with pytest.raises(ValueError, match="block_size"):
            m.draft_block(self._ctx(cfg), 5, embed_fn, lm_head_fn, block_size=0)

    def test_deterministic(self):
        """Greedy + fixed weights ⇒ identical drafts across calls."""
        cfg = _tiny_cfg()
        torch.manual_seed(7)
        m = DFlashDrafter(cfg)
        embed_fn, lm_head_fn = _synthetic_verifier_heads(cfg)
        ctx = self._ctx(cfg)
        a = m.draft_block(ctx, 7, embed_fn, lm_head_fn, block_size=4)
        b = m.draft_block(ctx, 7, embed_fn, lm_head_fn, block_size=4)
        assert a == b


# ---------------------------------------------------------------------------
# DFlashProposer — engine propose_block contract
# ---------------------------------------------------------------------------


class TestDFlashProposer:
    def _proposer(self):
        cfg = _tiny_cfg()
        torch.manual_seed(3)
        drafter = DFlashDrafter(cfg)
        embed_fn, lm_head_fn = _synthetic_verifier_heads(cfg)
        return DFlashProposer(
            drafter, _SyntheticAuxProvider(cfg), embed_fn, lm_head_fn,
        ), cfg

    def test_propose_block_returns_blockproposal(self):
        prop, cfg = self._proposer()
        out = prop.propose_block([10, 11, 12], block_size=4, num_steps=4)
        assert isinstance(out, BlockProposal)
        assert len(out.tokens) == 4
        # DFlash drafts the whole block in a single non-causal forward.
        assert out.diffusion_steps == 1
        assert out.forward_passes == 1
        assert all(t != cfg.mask_token_id for t in out.tokens)

    def test_propose_block_length_matches_request(self):
        prop, _ = self._proposer()
        for L in (1, 2, 3, 4):
            out = prop.propose_block([5, 6], block_size=L, num_steps=4)
            assert len(out.tokens) == L

    def test_propose_block_validates_args(self):
        prop, _ = self._proposer()
        with pytest.raises(ValueError):
            prop.propose_block([1, 2], block_size=0, num_steps=1)
        with pytest.raises(ValueError):
            prop.propose_block([1, 2], block_size=4, num_steps=0)
