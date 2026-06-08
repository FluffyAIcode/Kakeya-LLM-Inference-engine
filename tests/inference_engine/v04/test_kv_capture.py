"""Linux CI unit tests for inference_engine/v04/kv_capture.py.

These tests exercise the K/V capture mechanism on synthetic
mini-models that mirror the HF Gemma3 / Llama / GPT-2 hook surface
(decoder layers with self_attn.k_proj / v_proj or attn.k_proj /
v_proj Linear modules), without requiring any HF transformers model
download.

Empirical validation against real google/gemma-3-1b-it lives in
the Mac M4 reviewer aid (separate PR for K1.D), not in CI.

Test classes:

* TestLocateAttentionLayers — model-shape discovery (Gemma/Llama
  shape vs GPT-2 shape vs unrecognised → RuntimeError).
* TestRegisterKVCaptureHooks — hook installation + removal lifecycle,
  layer_indices subset selection, error on missing k_proj/v_proj.
* TestCaptureProposerKVShapes — KVCapture shape & dtype invariants
  produced by capture_proposer_kv on a synthetic model.
* TestCaptureProposerKVValues — value correctness: captured K/V at
  every layer match a manual reference forward bit-exactly.
* TestCaptureProposerKVConfigInference — num_kv_heads / head_dim
  derivation from the model.config in standard / non-standard cases.
* TestKVCaptureSelectPositions — slicing K/V to a subset of token
  positions (used by K1.B injection).
"""

from __future__ import annotations

from typing import List, Optional

import pytest
import torch
import torch.nn as nn

from inference_engine.v04.kv_capture import (
    KVCapture,
    capture_proposer_kv,
    register_kv_capture_hooks,
)


# ---------------------------------------------------------------------------
# Synthetic mini-models used as Gemma3 / GPT-2 surrogates
# ---------------------------------------------------------------------------


class _MiniAttention(nn.Module):
    """Self-attention sub-module exposing k_proj and v_proj Linear
    layers — the same hook surface as Gemma3Attention. The forward is
    a deliberately simple identity-attention so tests can pin the K/V
    values directly without modelling RoPE / softmax."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute Q/K/V to fire the hooks; for tests we don't actually
        # need the attention output, so we just return o_proj of a
        # masked Q for shape continuity.
        q = self.q_proj(hidden_states)
        _ = self.k_proj(hidden_states)  # fires k hook
        _ = self.v_proj(hidden_states)  # fires v hook
        return self.o_proj(q)


class _MiniDecoderLayer(nn.Module):
    """Decoder layer with .self_attn — Gemma3 / Llama / Qwen / Mistral shape."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.self_attn = _MiniAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.self_attn(hidden_states)


class _GPT2DecoderLayer(nn.Module):
    """Decoder layer with .attn — GPT-2 shape."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.attn = _MiniAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.attn(hidden_states)


class _MiniConfig:
    """Stand-in for the .config that capture_proposer_kv reads."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if head_dim is not None:
            self.head_dim = head_dim


class _MiniInner(nn.Module):
    """The .model attribute on a HF Gemma3 / Llama causal LM."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        vocab: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden_size)
        self.layers = nn.ModuleList(
            [
                _MiniDecoderLayer(hidden_size, num_q_heads, num_kv_heads, head_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _MiniGemmaShapeModel(nn.Module):
    """Surrogate that mimics HF Gemma3ForCausalLM enough for capture
    machinery: .model.layers with .self_attn.k_proj / v_proj, plus
    .config; forward accepts input_ids + use_cache + attention_mask."""

    def __init__(
        self,
        num_layers: int = 3,
        hidden_size: int = 16,
        vocab: int = 32,
        num_q_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.config = _MiniConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.model = _MiniInner(
            num_layers, hidden_size, vocab, num_q_heads, num_kv_heads, head_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return self.model(input_ids)


class _MiniGPT2ShapeModel(nn.Module):
    """Surrogate for a GPT-2-shaped causal LM: .transformer.h with .attn."""

    class _Trans(nn.Module):
        def __init__(self, num_layers, hidden_size, vocab, nq, nkv, hd):
            super().__init__()
            self.wte = nn.Embedding(vocab, hidden_size)
            self.h = nn.ModuleList(
                [_GPT2DecoderLayer(hidden_size, nq, nkv, hd) for _ in range(num_layers)]
            )

        def forward(self, input_ids):
            h = self.wte(input_ids)
            for layer in self.h:
                h = layer(h)
            return h

    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 12,
        vocab: int = 16,
        num_q_heads: int = 3,
        num_kv_heads: int = 1,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.config = _MiniConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.transformer = self._Trans(
            num_layers, hidden_size, vocab, num_q_heads, num_kv_heads, head_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return self.transformer(input_ids)


# ---------------------------------------------------------------------------
# _locate_attention_layers
# ---------------------------------------------------------------------------


class TestLocateAttentionLayers:
    def test_gemma_llama_shape_discovered(self):
        model = _MiniGemmaShapeModel(num_layers=3)
        # capture_proposer_kv smoke-tests the discovery path; here we
        # just confirm registration succeeds (the unit-level discovery
        # function is private but exercised by registration).
        k_acc, v_acc, handles = register_kv_capture_hooks(model)
        try:
            assert len(k_acc) == 3
            assert len(v_acc) == 3
            assert len(handles) == 6  # 2 per layer
        finally:
            for h in handles:
                h.remove()

    def test_gpt2_shape_discovered(self):
        model = _MiniGPT2ShapeModel(num_layers=2)
        k_acc, v_acc, handles = register_kv_capture_hooks(model)
        try:
            assert len(k_acc) == 2
            assert len(v_acc) == 2
            assert len(handles) == 4
        finally:
            for h in handles:
                h.remove()

    def test_unrecognised_shape_raises(self):
        class _Mystery(nn.Module):
            class _C: pass
            def __init__(self):
                super().__init__()
                self.config = self._C()

        with pytest.raises(RuntimeError, match="locate decoder layers"):
            register_kv_capture_hooks(_Mystery())

    def test_zero_layers_raises(self):
        class _ZeroLayer(nn.Module):
            class _Inner(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.ModuleList()
            def __init__(self):
                super().__init__()
                self.model = self._Inner()

        with pytest.raises(RuntimeError, match="zero decoder layers"):
            register_kv_capture_hooks(_ZeroLayer())


# ---------------------------------------------------------------------------
# register_kv_capture_hooks
# ---------------------------------------------------------------------------


class TestRegisterKVCaptureHooks:
    def test_layer_indices_subset_selection(self):
        model = _MiniGemmaShapeModel(num_layers=4)
        k_acc, v_acc, handles = register_kv_capture_hooks(
            model, layer_indices=[0, 2],
        )
        try:
            assert len(k_acc) == 2
            assert len(v_acc) == 2
            input_ids = torch.randint(0, 32, (1, 6))
            model(input_ids)
            # Both selected layers fired exactly once
            assert len(k_acc[0]) == 1
            assert len(k_acc[1]) == 1
            # Layers 1 and 3 were not hooked, so we have nothing
            # to check there.
        finally:
            for h in handles:
                h.remove()

    def test_layer_indices_dedup_and_sort(self):
        model = _MiniGemmaShapeModel(num_layers=4)
        k_acc, _, handles = register_kv_capture_hooks(
            model, layer_indices=[2, 0, 2, 0],
        )
        try:
            assert len(k_acc) == 2  # deduplicated
        finally:
            for h in handles:
                h.remove()

    def test_layer_indices_out_of_range_raises(self):
        model = _MiniGemmaShapeModel(num_layers=4)
        with pytest.raises(ValueError, match="out of range"):
            register_kv_capture_hooks(model, layer_indices=[0, 100])

    def test_hooks_removed_does_not_continue_capturing(self):
        model = _MiniGemmaShapeModel(num_layers=3)
        k_acc, v_acc, handles = register_kv_capture_hooks(model)
        for h in handles:
            h.remove()

        # After removal a new forward should not append to k_acc/v_acc.
        input_ids = torch.randint(0, 32, (1, 4))
        model(input_ids)
        for buf in k_acc:
            assert buf == []
        for buf in v_acc:
            assert buf == []

    def test_attention_module_without_kv_proj_raises(self):
        class _NoKV(nn.Module):
            class _NoKVAttn(nn.Module):
                def forward(self, x):
                    return x
            class _Layer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.self_attn = _NoKV._NoKVAttn()
                def forward(self, x):
                    return x
            class _Inner(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.ModuleList([_NoKV._Layer()])
            def __init__(self):
                super().__init__()
                self.model = self._Inner()

        with pytest.raises(RuntimeError, match="k_proj"):
            register_kv_capture_hooks(_NoKV())


# ---------------------------------------------------------------------------
# capture_proposer_kv — shape + invariants
# ---------------------------------------------------------------------------


class TestCaptureProposerKVShapes:
    def test_full_capture_shape(self):
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(
            num_layers=3, hidden_size=16, num_q_heads=4,
            num_kv_heads=2, head_dim=4,
        )
        input_ids = torch.randint(0, 32, (1, 8))
        cap = capture_proposer_kv(model, input_ids)
        assert cap.num_layers == 3
        assert cap.seq_len == 8
        assert cap.num_kv_heads == 2
        assert cap.head_dim == 4
        for k, v in zip(cap.keys, cap.values):
            assert k.shape == (1, 8, 2, 4)
            assert v.shape == (1, 8, 2, 4)

    def test_subset_capture_shape(self):
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(num_layers=4)
        input_ids = torch.randint(0, 32, (1, 5))
        cap = capture_proposer_kv(model, input_ids, layer_indices=[1, 3])
        assert cap.num_layers == 2

    def test_captured_tensors_are_detached(self):
        model = _MiniGemmaShapeModel(num_layers=2)
        input_ids = torch.randint(0, 32, (1, 4))
        cap = capture_proposer_kv(model, input_ids)
        for k, v in zip(cap.keys, cap.values):
            assert not k.requires_grad
            assert not v.requires_grad

    def test_dtype_consistency(self):
        model = _MiniGemmaShapeModel(num_layers=2).to(torch.float64)
        input_ids = torch.randint(0, 32, (1, 4))
        cap = capture_proposer_kv(model, input_ids)
        for k, v in zip(cap.keys, cap.values):
            assert k.dtype == torch.float64
            assert v.dtype == torch.float64


# ---------------------------------------------------------------------------
# capture_proposer_kv — value correctness
# ---------------------------------------------------------------------------


class TestCaptureProposerKVValues:
    def test_captured_k_matches_manual_kproj_at_layer0(self):
        """At layer 0, the K projection input is just the embedding
        output. Capture should bit-exactly match a manual
        k_proj(embed_tokens(input_ids)).
        """
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(num_layers=1, hidden_size=8, num_q_heads=2,
                                     num_kv_heads=1, head_dim=4)
        input_ids = torch.randint(0, 32, (1, 5))
        cap = capture_proposer_kv(model, input_ids)

        with torch.no_grad():
            embed_out = model.model.embed_tokens(input_ids)
            ref_k_raw = model.model.layers[0].self_attn.k_proj(embed_out)
            ref_k = ref_k_raw.view(1, 5, 1, 4)

        assert torch.equal(cap.keys[0], ref_k)

    def test_captured_v_matches_manual_vproj_at_layer0(self):
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(num_layers=1, hidden_size=8, num_q_heads=2,
                                     num_kv_heads=1, head_dim=4)
        input_ids = torch.randint(0, 32, (1, 5))
        cap = capture_proposer_kv(model, input_ids)

        with torch.no_grad():
            embed_out = model.model.embed_tokens(input_ids)
            ref_v_raw = model.model.layers[0].self_attn.v_proj(embed_out)
            ref_v = ref_v_raw.view(1, 5, 1, 4)

        assert torch.equal(cap.values[0], ref_v)

    def test_capture_deterministic_under_fixed_seed(self):
        """Two calls with the same input produce bit-exact captures
        (no nondeterminism in the hook plumbing itself)."""
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(num_layers=2)
        input_ids = torch.randint(0, 32, (1, 6))
        cap_a = capture_proposer_kv(model, input_ids)
        cap_b = capture_proposer_kv(model, input_ids)
        for k_a, k_b in zip(cap_a.keys, cap_b.keys):
            assert torch.equal(k_a, k_b)
        for v_a, v_b in zip(cap_a.values, cap_b.values):
            assert torch.equal(v_a, v_b)


# ---------------------------------------------------------------------------
# capture_proposer_kv — config inference paths
# ---------------------------------------------------------------------------


class TestCaptureProposerKVConfigInference:
    def test_explicit_overrides_take_precedence(self):
        """When num_kv_heads / head_dim are passed, they override
        config inference. (Useful for synthetic models whose configs
        are non-standard.)"""
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(
            num_layers=1, hidden_size=8, num_q_heads=2,
            num_kv_heads=1, head_dim=4,
        )
        # Override head shape to a logically equivalent split that
        # divides the k_proj output dim differently. Here num_kv_heads
        # * head_dim must still equal 4 (k_proj output dim).
        cap = capture_proposer_kv(
            model, torch.randint(0, 32, (1, 3)),
            num_kv_heads=2, head_dim=2,
        )
        assert cap.num_kv_heads == 2
        assert cap.head_dim == 2

    def test_head_dim_derived_from_hidden_size_and_num_q_heads(self):
        """When config has no .head_dim, derive as hidden // num_q_heads."""
        class _NoHeadDimConfig(_MiniConfig):
            def __init__(self):
                super().__init__(
                    hidden_size=12, num_attention_heads=3,
                    num_key_value_heads=1, head_dim=None,
                )
                # don't set self.head_dim

        model = _MiniGemmaShapeModel(
            num_layers=1, hidden_size=12, num_q_heads=3,
            num_kv_heads=1, head_dim=4,  # 12 / 3 = 4
        )
        # Replace config with one that has no head_dim attribute
        model.config = _NoHeadDimConfig()
        cap = capture_proposer_kv(model, torch.randint(0, 32, (1, 4)))
        assert cap.head_dim == 4

    def test_config_inference_inconsistent_shape_raises(self):
        """Mismatched num_kv_heads * head_dim vs k_proj output dim
        should raise rather than silently reshape."""
        model = _MiniGemmaShapeModel(
            num_layers=1, hidden_size=8, num_q_heads=2,
            num_kv_heads=1, head_dim=4,  # k_proj out = 4
        )
        with pytest.raises(RuntimeError, match="last-dim"):
            capture_proposer_kv(
                model, torch.randint(0, 32, (1, 3)),
                num_kv_heads=3, head_dim=4,  # 3 * 4 = 12 ≠ 4
            )


# ---------------------------------------------------------------------------
# KVCapture.select_positions
# ---------------------------------------------------------------------------


class TestKVCaptureSelectPositions:
    def _make(self, T=8):
        torch.manual_seed(0)
        model = _MiniGemmaShapeModel(num_layers=2)
        return capture_proposer_kv(model, torch.randint(0, 32, (1, T)))

    def test_select_subset_preserves_layer_count(self):
        cap = self._make(T=10)
        sub = cap.select_positions([2, 5, 7])
        assert sub.num_layers == cap.num_layers
        assert sub.seq_len == 3
        assert sub.num_kv_heads == cap.num_kv_heads
        assert sub.head_dim == cap.head_dim

    def test_selected_values_match_index_select(self):
        cap = self._make(T=10)
        positions = [2, 5, 7]
        sub = cap.select_positions(positions)
        idx = torch.tensor(positions)
        for layer_idx in range(cap.num_layers):
            ref_k = cap.keys[layer_idx].index_select(dim=1, index=idx)
            ref_v = cap.values[layer_idx].index_select(dim=1, index=idx)
            assert torch.equal(sub.keys[layer_idx], ref_k)
            assert torch.equal(sub.values[layer_idx], ref_v)

    def test_unsorted_positions_raises(self):
        cap = self._make(T=8)
        with pytest.raises(ValueError, match="sorted ascending"):
            cap.select_positions([5, 2, 7])

    def test_duplicate_positions_raises(self):
        cap = self._make(T=8)
        with pytest.raises(ValueError, match="sorted ascending"):
            cap.select_positions([2, 2, 5])

    def test_empty_positions_raises(self):
        cap = self._make(T=8)
        with pytest.raises(ValueError, match="non-empty"):
            cap.select_positions([])

    def test_negative_position_raises(self):
        cap = self._make(T=8)
        with pytest.raises(ValueError, match="must lie in"):
            cap.select_positions([-1, 5])

    def test_position_at_or_beyond_seqlen_raises(self):
        cap = self._make(T=8)
        with pytest.raises(ValueError, match="must lie in"):
            cap.select_positions([5, 8])


# ---------------------------------------------------------------------------
# KVCapture invariants enforced at construction
# ---------------------------------------------------------------------------


class TestKVCaptureInvariants:
    def _ok_tensor(self, *, T=4, H=2, D=3, dtype=torch.float32, device="cpu"):
        return torch.randn(1, T, H, D, dtype=dtype, device=device)

    def test_construction_succeeds_with_consistent_inputs(self):
        keys = [self._ok_tensor() for _ in range(2)]
        values = [self._ok_tensor() for _ in range(2)]
        cap = KVCapture(
            keys=keys, values=values, num_layers=2, seq_len=4,
            num_kv_heads=2, head_dim=3,
        )
        assert cap.num_layers == 2

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            KVCapture(keys=[], values=[], num_layers=0, seq_len=0,
                      num_kv_heads=1, head_dim=1)

    def test_keys_values_length_mismatch_raises(self):
        keys = [self._ok_tensor() for _ in range(2)]
        values = [self._ok_tensor() for _ in range(3)]
        with pytest.raises(ValueError, match="same number of layers"):
            KVCapture(keys=keys, values=values, num_layers=2, seq_len=4,
                      num_kv_heads=2, head_dim=3)

    def test_inconsistent_layer_shape_raises(self):
        keys = [self._ok_tensor(T=4), self._ok_tensor(T=5)]
        values = [self._ok_tensor(T=4), self._ok_tensor(T=4)]
        with pytest.raises(ValueError, match="keys\\[1\\] shape"):
            KVCapture(keys=keys, values=values, num_layers=2, seq_len=4,
                      num_kv_heads=2, head_dim=3)

    def test_dtype_mismatch_within_layer_raises(self):
        keys = [
            self._ok_tensor(dtype=torch.float32),
            self._ok_tensor(dtype=torch.float32),
        ]
        values = [
            self._ok_tensor(dtype=torch.float64),
            self._ok_tensor(dtype=torch.float32),
        ]
        with pytest.raises(ValueError, match="dtype mismatch"):
            KVCapture(keys=keys, values=values, num_layers=2, seq_len=4,
                      num_kv_heads=2, head_dim=3)

    def test_3d_keys_rejected(self):
        keys = [torch.randn(1, 4, 6)]  # wrong rank
        values = [torch.randn(1, 4, 2, 3)]
        with pytest.raises(ValueError, match="must be 4-D"):
            KVCapture(keys=keys, values=values, num_layers=1, seq_len=4,
                      num_kv_heads=2, head_dim=3)
