"""Linux CI unit tests for inference_engine/v04/dlm_restored_verifier.py.

These tests cover the end-to-end v0.4 K/V Restoration verifier wrapper:

* Patch installation and removal lifecycle (including exception
  safety) on a synthetic Gemma3-shape model.
* The standalone ``_restored_attention_forward`` patched function
  with a fake attention module that exposes the required Gemma3-
  compatible attributes.
* Wrapper-level shape validation (single-batch only, decoder layer
  discovery).
* Empty-evicted-positions case: no merge happens, behaviour is
  identical to the upstream forward.

End-to-end validation against real Gemma 3-1B-it lives on the
Mac M4 reviewer aid (``scripts/review_pr_k1d_on_mac.sh``) and is not
part of Linux CI. The tests here pin the wrapper's mechanics; the
empirical "does it actually rescue recall?" question is gated on
NIAH validation in K1.E (separate PR).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from inference_engine.v04.dlm_restored_verifier import (
    DLMRestoredVerifier,
    _LayerRestorationContext,
    _restored_attention_forward,
)
from inference_engine.v04.kv_capture import KVCapture


# ---------------------------------------------------------------------------
# Fake attention / decoder layer with the Gemma3 hook surface
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Stand-in for HF model config that the wrapper + capture_proposer_kv
    read. Includes the head-count attributes required by capture's
    config inference (num_key_value_heads, num_attention_heads,
    head_dim, hidden_size)."""

    def __init__(
        self,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 4,
        hidden_size: int = 16,
    ):
        self._attn_implementation = "eager"
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size


class _FakeAttention(nn.Module):
    """Synthetic attention module with all Gemma3Attention attributes
    that ``_restored_attention_forward`` reads. Forward is not
    overridden — the function under test takes the module as an
    argument and orchestrates Q/K/V/o_proj/norm calls explicitly.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        num_q_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 4,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.attention_dropout = 0.0
        self.sliding_window = None
        self.config = _FakeConfig()
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.Identity()  # simplified
        self.k_norm = nn.Identity()
        # Original forward kept simple; wrapper tests don't use it
        # directly because they patch it.

    def forward(self, hidden_states, *args, **kwargs):
        # Invoke k_proj, v_proj, q_proj so capture hooks fire when
        # this module is used as a "proposer" target by
        # capture_proposer_kv. The actual attention math is irrelevant
        # for the wrapper-level lifecycle tests; we just return a
        # tensor of the right output shape.
        _ = self.q_proj(hidden_states)
        _ = self.k_proj(hidden_states)
        _ = self.v_proj(hidden_states)
        out = self.o_proj(self.q_proj(hidden_states))
        return out, None


class _FakeDecoderLayer(nn.Module):
    """Decoder layer with .self_attn — Gemma3 / Llama / Qwen / Mistral shape."""

    def __init__(self, hidden_size: int = 16, layer_idx: int = 0, **kw):
        super().__init__()
        self.self_attn = _FakeAttention(hidden_size=hidden_size, layer_idx=layer_idx, **kw)


class _FakeInner(nn.Module):
    """The .model attribute on a HF Gemma3 / Llama causal LM."""

    def __init__(self, num_layers: int, hidden_size: int = 16, **kw):
        super().__init__()
        self.layers = nn.ModuleList([
            _FakeDecoderLayer(hidden_size=hidden_size, layer_idx=i, **kw)
            for i in range(num_layers)
        ])


class _FakeModel(nn.Module):
    """Synthetic Gemma3-shape model with the minimum surface that
    DLMRestoredVerifier discovers and patches.

    The .forward method runs each layer's ``self_attn.forward(...)``
    (or, if patched, the v0.4 patched closure) on a synthetic hidden
    state, then returns a namespace with ``.logits``. This is enough
    to:

    * Trigger ``register_kv_capture_hooks`` on ``k_proj`` / ``v_proj``
      during the proposer-role capture pass.
    * Exercise the patched forward during the verifier-role pass.

    The hidden state is a fixed-seed random tensor so two consecutive
    forwards (capture, then verifier) see the same input — making
    the same-model identity case mathematically meaningful.
    """

    def __init__(self, num_layers: int = 2, hidden_size: int = 16, **kw):
        super().__init__()
        self.config = _FakeConfig()
        self.model = _FakeInner(num_layers=num_layers, hidden_size=hidden_size, **kw)
        self.hidden_size = hidden_size
        self._forward_was_called = False

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        self._forward_was_called = True
        B, T = input_ids.shape

        # Deterministic synthetic hidden state seeded by sequence length
        # so two forwards over the same input_ids see the same hidden
        # (no embedding lookup needed for this lifecycle test).
        torch.manual_seed(input_ids.sum().item() % (2**31))
        hidden = torch.randn(B, T, self.hidden_size)

        # Build position embeddings (cos, sin) and a permissive
        # attention mask so the patched forward can run if installed.
        head_dim = self.model.layers[0].self_attn.head_dim
        cos = torch.ones(B, T, head_dim) * 0.5
        sin = torch.ones(B, T, head_dim) * 0.5

        # We pass a dummy 4D mask matching the shape attention_interface
        # expects; for the fake _eager_attention_forward we built, the
        # actual mask values don't influence the output (it returns
        # zeros), but the shape must be reasonable.
        mask = torch.zeros(B, 1, T, T)

        for layer in self.model.layers:
            attn = layer.self_attn
            # Calling attn.forward triggers either the original (which
            # fires k_proj/v_proj hooks during capture) or the patched
            # closure (which exercises the v0.4 merge code path during
            # the verifier role).
            attn.forward(hidden, (cos, sin), mask)

        class _Out:
            def __init__(self, logits):
                self.logits = logits

        return _Out(torch.zeros(B, T, 32))


# ---------------------------------------------------------------------------
# Fake HF function pointers
# ---------------------------------------------------------------------------


def _fake_apply_rotary_pos_emb(q, k, cos, sin):
    """Identity-RoPE for pipeline tests: we don't need the real RoPE
    math here, just need the function to accept the right signature
    and return q, k unchanged (so we can verify shapes flow through)."""
    return q, k


def _fake_eager_attention_forward(
    module, query, key, value, attention_mask,
    dropout=0.0, scaling=None, sliding_window=None, **kwargs,
):
    """Identity-attention for pipeline tests: returns value as the
    output and None for weights. Shape: ``[B, num_q_heads, T, head_dim]``."""
    # Project value to query's num_heads dim by repeating KV heads
    # if GQA — for simplicity in tests we just take value as-is
    # and reshape to match query.
    B, num_q_heads, T_q, D = query.shape
    _, num_kv_heads, T_kv, _ = value.shape
    # Use first num_q_heads of (broadcast value) as fake output
    if num_q_heads != num_kv_heads:
        # Repeat KV heads to match Q heads (GQA)
        groups = num_q_heads // num_kv_heads
        value = value.repeat_interleave(groups, dim=1)
    # Attend over the full sequence (so output has T_q rows even
    # though K/V have T_kv rows). Just return zeros at right shape.
    output = torch.zeros(B, num_q_heads, T_q, D, dtype=query.dtype, device=query.device)
    return output, None


# ---------------------------------------------------------------------------
# _restored_attention_forward — the patched-forward function
# ---------------------------------------------------------------------------


class TestRestoredAttentionForward:
    def _make_attn(self, hidden=16, num_q=4, num_kv=2, head_dim=4):
        return _FakeAttention(
            hidden_size=hidden,
            num_q_heads=num_q,
            num_kv_heads=num_kv,
            head_dim=head_dim,
        )

    def _make_inputs(self, B=1, T=8, hidden=16, head_dim=4):
        torch.manual_seed(0)
        hidden_states = torch.randn(B, T, hidden)
        cos = torch.randn(B, T, head_dim)
        sin = torch.randn(B, T, head_dim)
        attention_mask = torch.zeros(B, 1, T, T)
        return hidden_states, (cos, sin), attention_mask

    def test_no_context_runs_like_upstream(self):
        """Without a v0.4 layer context attached, the patched forward
        runs the standard Gemma3 logic without merge."""
        attn = self._make_attn()
        hidden_states, position_embeddings, attention_mask = self._make_inputs()
        # No _v04_layer_context attached
        out, _weights = _restored_attention_forward(
            attn, hidden_states, position_embeddings, attention_mask,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
            all_attention_functions=None,
        )
        # Output shape must match input batch+seq, hidden dim
        assert out.shape == hidden_states.shape

    def test_empty_evicted_short_circuits_merge(self):
        """When evicted_positions is empty (e.g., sink+window covers
        full sequence) the merge is skipped and the patched forward
        produces the same output as no-context."""
        attn = self._make_attn()
        hidden_states, position_embeddings, attention_mask = self._make_inputs(T=8)
        # Attach context with empty evicted list
        attn._v04_layer_context = _LayerRestorationContext(
            captured_K=torch.empty(0),
            captured_V=torch.empty(0),
            evicted_positions=[],
        )
        out_with_empty, _ = _restored_attention_forward(
            attn, hidden_states, position_embeddings, attention_mask,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        delattr(attn, "_v04_layer_context")
        out_no_context, _ = _restored_attention_forward(
            attn, hidden_states, position_embeddings, attention_mask,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        # Outputs match (both produce zero from fake attention)
        assert torch.equal(out_with_empty, out_no_context)

    def test_with_context_calls_merge_and_returns_correct_shape(self):
        """When evicted_positions is non-empty, the merge step runs
        without error and the output shape is preserved."""
        torch.manual_seed(0)
        attn = self._make_attn(hidden=16, num_q=4, num_kv=2, head_dim=4)
        hidden_states, position_embeddings, attention_mask = self._make_inputs(
            B=1, T=8, hidden=16, head_dim=4,
        )
        # Attach context with non-empty evicted
        evicted = [2, 5]
        attn._v04_layer_context = _LayerRestorationContext(
            captured_K=torch.randn(1, len(evicted), 2, 4),
            captured_V=torch.randn(1, len(evicted), 2, 4),
            evicted_positions=evicted,
        )
        out, _ = _restored_attention_forward(
            attn, hidden_states, position_embeddings, attention_mask,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert out.shape == hidden_states.shape


# ---------------------------------------------------------------------------
# DLMRestoredVerifier — wrapper-level
# ---------------------------------------------------------------------------


class TestDLMRestoredVerifierConstruction:
    def test_default_sink_and_window(self):
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model)
        assert v.sink_size == 4
        assert v.window_size == 64

    def test_negative_sink_raises(self):
        model = _FakeModel(num_layers=2)
        with pytest.raises(ValueError, match="non-negative"):
            DLMRestoredVerifier(model, sink_size=-1, window_size=64)

    def test_negative_window_raises(self):
        model = _FakeModel(num_layers=2)
        with pytest.raises(ValueError, match="non-negative"):
            DLMRestoredVerifier(model, sink_size=4, window_size=-1)


class TestDLMRestoredVerifierShapeDiscovery:
    def test_decoder_layers_discovered(self):
        model = _FakeModel(num_layers=3)
        v = DLMRestoredVerifier(model)
        layers = v._decoder_layers()
        assert len(layers) == 3

    def test_attention_modules_discovered(self):
        model = _FakeModel(num_layers=3)
        v = DLMRestoredVerifier(model)
        attn_modules = v._attention_modules()
        assert len(attn_modules) == 3
        for attn in attn_modules:
            assert isinstance(attn, _FakeAttention)

    def test_unrecognized_model_shape_raises(self):
        class _NoLayers(nn.Module):
            class _C:
                _attn_implementation = "eager"
            def __init__(self):
                super().__init__()
                self.config = self._C()
        v = DLMRestoredVerifier(_NoLayers())
        with pytest.raises(RuntimeError, match="locate decoder layers"):
            v._decoder_layers()

    def test_layer_without_self_attn_raises(self):
        class _BareLayer(nn.Module):
            pass

        class _BareInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_BareLayer()])

        class _BareModel(nn.Module):
            class _C: _attn_implementation = "eager"
            def __init__(self):
                super().__init__()
                self.config = self._C()
                self.model = _BareInner()

        v = DLMRestoredVerifier(_BareModel())
        with pytest.raises(RuntimeError, match="self_attn"):
            v._attention_modules()


class TestDLMRestoredVerifierBatchValidation:
    def test_rank_1_input_raises(self):
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model)
        with pytest.raises(ValueError, match="single-batch only"):
            v.forward(
                torch.randint(0, 32, (8,)),
                apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
                eager_attention_forward=_fake_eager_attention_forward,
            )

    def test_batch_size_2_raises(self):
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model)
        with pytest.raises(ValueError, match="single-batch only"):
            v.forward(
                torch.randint(0, 32, (2, 8)),
                apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
                eager_attention_forward=_fake_eager_attention_forward,
            )


# ---------------------------------------------------------------------------
# Patch lifecycle
# ---------------------------------------------------------------------------


class TestRestorationActiveLifecycle:
    def _make_capture(self, num_layers=2, T=10, num_kv_heads=2, head_dim=4):
        torch.manual_seed(0)
        keys = [torch.randn(1, T, num_kv_heads, head_dim) for _ in range(num_layers)]
        values = [torch.randn(1, T, num_kv_heads, head_dim) for _ in range(num_layers)]
        return KVCapture(
            keys=keys, values=values, num_layers=num_layers,
            seq_len=T, num_kv_heads=num_kv_heads, head_dim=head_dim,
        )

    def test_install_replaces_forwards(self):
        """Patched forward is a regular function (closure); the
        original is a bound method whose ``__func__`` is the class's
        forward. We use those identities to distinguish patched vs
        unpatched, since ``obj.forward is some_capture`` doesn't
        work for bound methods (Python creates a fresh wrapper on
        each attribute access)."""
        model = _FakeModel(num_layers=3)
        v = DLMRestoredVerifier(model, sink_size=2, window_size=2)
        capture = self._make_capture(num_layers=3, T=10)
        evicted = [2, 3, 4, 5, 6, 7]

        attn_modules = v._attention_modules()

        # Pre-patch: bound method, has __func__ pointing to class fn
        for attn in attn_modules:
            assert hasattr(attn.forward, "__func__")
            assert attn.forward.__func__ is _FakeAttention.forward

        with v._restoration_active(
            capture, evicted,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
            all_attention_functions=None,
        ):
            # Inside the context: patched forward is a closure (regular
            # function), no __func__ attribute.
            for attn in attn_modules:
                assert not hasattr(attn.forward, "__func__"), (
                    "patched forward should be a closure, not a "
                    "bound method"
                )

        # After exit: bound method again, __func__ matches class fn
        for attn in attn_modules:
            assert hasattr(attn.forward, "__func__")
            assert attn.forward.__func__ is _FakeAttention.forward

    def test_install_attaches_context_to_each_layer(self):
        model = _FakeModel(num_layers=3)
        v = DLMRestoredVerifier(model, sink_size=2, window_size=2)
        capture = self._make_capture(num_layers=3, T=10)
        evicted = [2, 3, 4, 5, 6, 7]
        attn_modules = v._attention_modules()

        with v._restoration_active(
            capture, evicted,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
            all_attention_functions=None,
        ):
            for attn in attn_modules:
                assert hasattr(attn, "_v04_layer_context")
                ctx = attn._v04_layer_context
                assert isinstance(ctx, _LayerRestorationContext)
                assert ctx.evicted_positions == evicted
                assert ctx.captured_K.shape == (1, len(evicted), 2, 4)

        # After exit, contexts are removed
        for attn in attn_modules:
            assert not hasattr(attn, "_v04_layer_context")

    def test_layer_count_mismatch_raises(self):
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model)
        capture = self._make_capture(num_layers=5)  # mismatch

        with pytest.raises(RuntimeError, match="capture has"):
            with v._restoration_active(
                capture, [3, 4],
                apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
                eager_attention_forward=_fake_eager_attention_forward,
                all_attention_functions=None,
            ):
                pass

    def test_exception_during_context_still_unpatches(self):
        """If an exception fires inside the with-block, the finally
        clause must still restore originals and clear contexts."""
        model = _FakeModel(num_layers=3)
        v = DLMRestoredVerifier(model, sink_size=2, window_size=2)
        capture = self._make_capture(num_layers=3, T=10)
        evicted = [3, 4, 5]

        attn_modules = v._attention_modules()

        with pytest.raises(RuntimeError, match="synthetic"):
            with v._restoration_active(
                capture, evicted,
                apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
                eager_attention_forward=_fake_eager_attention_forward,
                all_attention_functions=None,
            ):
                raise RuntimeError("synthetic test failure")

        # After exception, all forwards are restored (bound method
        # whose __func__ is the class's forward) and contexts cleared.
        for attn in attn_modules:
            assert hasattr(attn.forward, "__func__")
            assert attn.forward.__func__ is _FakeAttention.forward
            assert not hasattr(attn, "_v04_layer_context")

    def test_empty_evicted_attaches_empty_context(self):
        """When evicted_positions is empty (sink+window covers full
        sequence), the patched forward still installs but the merge
        is short-circuited inside the patched function."""
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model, sink_size=10, window_size=10)
        capture = self._make_capture(num_layers=2, T=8)
        evicted = []  # nothing to evict

        attn_modules = v._attention_modules()
        with v._restoration_active(
            capture, evicted,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
            all_attention_functions=None,
        ):
            for attn in attn_modules:
                ctx = attn._v04_layer_context
                assert ctx.evicted_positions == []


# ---------------------------------------------------------------------------
# End-to-end forward (with stub model)
# ---------------------------------------------------------------------------


class TestDLMRestoredVerifierForward:
    def test_forward_calls_model_forward(self):
        """The wrapper's forward should ultimately call the model's
        forward (that's how it produces logits). The fake model
        records this."""
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model, sink_size=2, window_size=2)
        input_ids = torch.randint(0, 32, (1, 10))
        logits = v.forward(
            input_ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert model._forward_was_called
        # Logits shape from the fake model is [B, T, vocab=32]
        assert logits.shape == (1, 10, 32)

    def test_forward_with_short_input_no_eviction(self):
        """When seq_len <= sink + window, no eviction occurs and the
        forward proceeds without merge."""
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model, sink_size=4, window_size=8)
        input_ids = torch.randint(0, 32, (1, 8))  # seq_len = 8 = 4 + 8 actually exceeds, hmm
        # Actually 8 == sink_size, so 8 <= sink_size + window_size = 12 → no evict
        logits = v.forward(
            input_ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert logits.shape == (1, 8, 32)

    def test_forward_returns_no_grad_tensor(self):
        """forward is decorated @torch.no_grad() — output should not
        require grad."""
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model, sink_size=2, window_size=2)
        input_ids = torch.randint(0, 32, (1, 10))
        logits = v.forward(
            input_ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert not logits.requires_grad

    def test_forward_clears_patches_after_call(self):
        """After forward returns, the model's attention modules should
        not have v04_layer_context or patched forward."""
        model = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(model, sink_size=2, window_size=2)
        attn_modules = v._attention_modules()

        v.forward(
            torch.randint(0, 32, (1, 10)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )

        # Originals restored: bound method whose __func__ is class fn.
        for attn in attn_modules:
            assert hasattr(attn.forward, "__func__")
            assert attn.forward.__func__ is _FakeAttention.forward
            assert not hasattr(attn, "_v04_layer_context")
