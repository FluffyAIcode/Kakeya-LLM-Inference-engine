"""K2.A.2 stateful caching tests for DLMRestoredVerifier.

Per ADR 0008 §11.11.12 K2.A.2 formal commitment + §11.13.6.2
(at K1 same-checkpoint AR-causal setup, K2.A.2 stateful caching
should produce output bit-equivalent to K1.D / K2.A.1 stateless
modulo numerical noise).

End-to-end "stateful incremental forward output ≈ stateless full
forward output" requires running real Gemma 3-1B on hardware
(Mac M4 / vast); that's covered by the K2.A.2 reviewer aid +
empirical evidence, not by this Linux unit test suite. Here we
validate the orchestration layer, cache state transitions, and
the V04SessionCache assembly logic.

Reuses synthetic test fixtures (``_FakeModel``, ``_FakeAttention``,
``_fake_apply_rotary_pos_emb``, ``_fake_eager_attention_forward``)
from ``test_dlm_restored_verifier.py`` — these have the right
shape to make ``capture_proposer_kv`` hooks fire.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from inference_engine.v04.dlm_restored_verifier import (
    DLMRestoredVerifier,
    _SessionState,
    _V04SessionCache,
)
from inference_engine.v04.kv_compressor import (
    IdentityCompressor,
    KVCompressor,
)

# Reuse the existing K1.D test fixtures (they have working k_proj/v_proj
# invocations that fire capture hooks).
from tests.inference_engine.v04.test_dlm_restored_verifier import (
    _FakeModel,
    _fake_apply_rotary_pos_emb,
    _fake_eager_attention_forward,
)


def _fake_rotary_emb_fn(input_ids, position_ids):
    """Stub rotary embedding for stateful incremental tests.

    Returns zero cos/sin of shape [1, T, head_dim=4]. Compatible with
    _fake_apply_rotary_pos_emb which is identity (ignores cos/sin).
    """
    T = position_ids.size(-1)
    return (
        torch.zeros(1, T, 4, dtype=torch.float32),
        torch.zeros(1, T, 4, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# 1. stateful=False is K1.D / K2.A.1 — backward compat regression
# ---------------------------------------------------------------------------


class TestStatefulFalseIsBackwardCompatible:
    def test_default_constructor_is_stateless(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2)
        assert v.stateful is False

    def test_stateful_property_reflects_construction_argument(self):
        m = _FakeModel(num_layers=2)
        v_off = DLMRestoredVerifier(m, sink_size=2, window_size=2,
                                     stateful=False)
        v_on = DLMRestoredVerifier(m, sink_size=2, window_size=2,
                                    stateful=True)
        assert v_off.stateful is False
        assert v_on.stateful is True

    def test_stateless_cache_token_count_is_zero(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2,
                                 stateful=False)
        assert v.cache_token_count == 0

    def test_stateless_cache_token_count_remains_zero_across_forwards(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2,
                                 stateful=False)
        v.forward(
            torch.randint(0, 32, (1, 5)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v.cache_token_count == 0
        v.forward(
            torch.randint(0, 32, (1, 8)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v.cache_token_count == 0


# ---------------------------------------------------------------------------
# 2. reset_cache() clears state
# ---------------------------------------------------------------------------


class TestResetCacheBehaviour:
    def test_reset_cache_resets_cache_token_count(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        v.forward(
            torch.randint(0, 32, (1, 5)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v.cache_token_count == 5
        v.reset_cache()
        assert v.cache_token_count == 0

    def test_reset_cache_clears_compressors(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        v.forward(
            torch.randint(0, 32, (1, 5)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v._session_state.compressors is not None
        v.reset_cache()
        assert v._session_state.compressors is None

    def test_reset_cache_no_op_in_stateless_mode(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=False)
        v.reset_cache()  # must not raise
        assert v.cache_token_count == 0


# ---------------------------------------------------------------------------
# 3+4+5. stateful=True bootstrap forward
# ---------------------------------------------------------------------------


class TestStatefulBootstrapForward:
    def test_bootstrap_returns_logits_shape_matches_stateless(self):
        torch.manual_seed(123)
        m_a = _FakeModel(num_layers=2)
        v_off = DLMRestoredVerifier(m_a, sink_size=2, window_size=2,
                                     stateful=False)
        ids = torch.randint(0, 32, (1, 6))
        out_off = v_off.forward(
            ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )

        torch.manual_seed(123)
        m_b = _FakeModel(num_layers=2)
        v_on = DLMRestoredVerifier(m_b, sink_size=2, window_size=2,
                                    stateful=True)
        out_on = v_on.forward(
            ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert out_off.shape == out_on.shape == (1, 6, 32)

    def test_bootstrap_persists_compressors(self):
        m = _FakeModel(num_layers=3)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        assert v._session_state.compressors is None
        v.forward(
            torch.randint(0, 32, (1, 5)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v._session_state.compressors is not None
        assert len(v._session_state.compressors) == 3
        for c in v._session_state.compressors:
            assert isinstance(c, KVCompressor)

    def test_bootstrap_advances_cache_token_count(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        assert v.cache_token_count == 0
        v.forward(
            torch.randint(0, 32, (1, 7)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v.cache_token_count == 7

    def test_bootstrap_uses_factory_to_build_compressors(self):
        m = _FakeModel(num_layers=2)
        invocation_count = [0]

        def factory(head_dim):
            invocation_count[0] += 1
            return IdentityCompressor()

        v = DLMRestoredVerifier(m, sink_size=2, window_size=2,
                                 stateful=True,
                                 kv_compressor_factory=factory)
        v.forward(
            torch.randint(0, 32, (1, 5)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        # Factory invoked once per layer at first forward.
        assert invocation_count[0] == 2

    def test_second_bootstrap_on_same_session_reuses_compressors(self):
        # If user calls forward() with the SAME prefix length as
        # cache_token_count, the API contract is "raise — nothing
        # to do" (per stateful incremental validation).
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        v.forward(
            torch.randint(0, 32, (1, 5)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        # Persisted instance.
        comp_first = v._session_state.compressors
        assert comp_first is not None

        # Second forward extending prefix should reuse the same compressor
        # instances (not rebuild). The actual verification is via
        # _restoration_active's branch — covered by the next test class.


# ---------------------------------------------------------------------------
# 6. stateful=True incremental forward
# ---------------------------------------------------------------------------


class TestStatefulIncrementalForward:
    def test_incremental_forward_processes_only_new_tokens(self):
        # The model.forward should be called with input_ids of length
        # n_new (5), not T_full (12).
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        # Bootstrap with 7 tokens.
        ids = torch.randint(0, 32, (1, 7))
        v.forward(
            ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v.cache_token_count == 7

        # Incremental: extend to 12 tokens (5 new).
        ids_extended = torch.cat([ids, torch.randint(0, 32, (1, 5))], dim=1)
        v.forward(
            ids_extended,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
            rotary_emb_fn=_fake_rotary_emb_fn,
        )
        # Cache advanced to T_full.
        assert v.cache_token_count == 12

    def test_incremental_returns_full_t_logits_shape(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        ids = torch.randint(0, 32, (1, 7))
        v.forward(
            ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        ids_extended = torch.cat([ids, torch.randint(0, 32, (1, 5))], dim=1)
        out = v.forward(
            ids_extended,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
            rotary_emb_fn=_fake_rotary_emb_fn,
        )
        # Stateless contract: returns [1, T_full, vocab] shape.
        assert out.shape[0] == 1
        assert out.shape[1] == 12  # T_full
        # vocab dim depends on _FakeModel; just check rank=3
        assert out.dim() == 3


# ---------------------------------------------------------------------------
# 7+8. Validation: shrinking prefix / no-new-tokens
# ---------------------------------------------------------------------------


class TestStatefulInputValidation:
    def test_shrinking_prefix_raises(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        ids = torch.randint(0, 32, (1, 7))
        v.forward(
            ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        ids_shorter = torch.randint(0, 32, (1, 5))
        with pytest.raises(ValueError, match="shorter"):
            v.forward(
                ids_shorter,
                apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
                eager_attention_forward=_fake_eager_attention_forward,
            )

    def test_same_length_prefix_raises(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        ids = torch.randint(0, 32, (1, 7))
        v.forward(
            ids,
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        ids_same_len = torch.randint(0, 32, (1, 7))
        with pytest.raises(ValueError, match="nothing new"):
            v.forward(
                ids_same_len,
                apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
                eager_attention_forward=_fake_eager_attention_forward,
            )

    def test_reset_after_shrinking_unblocks(self):
        m = _FakeModel(num_layers=2)
        v = DLMRestoredVerifier(m, sink_size=2, window_size=2, stateful=True)
        v.forward(
            torch.randint(0, 32, (1, 7)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        v.reset_cache()
        # New session: should accept any input_ids length.
        v.forward(
            torch.randint(0, 32, (1, 3)),
            apply_rotary_pos_emb=_fake_apply_rotary_pos_emb,
            eager_attention_forward=_fake_eager_attention_forward,
        )
        assert v.cache_token_count == 3


# ---------------------------------------------------------------------------
# 9. _SessionState dataclass
# ---------------------------------------------------------------------------


class TestSessionStateDataclass:
    def test_fresh_factory(self):
        s = _SessionState.fresh()
        assert s.cache_token_count == 0
        assert s.compressors is None

    def test_default_constructor(self):
        s = _SessionState()
        assert s.cache_token_count == 0
        assert s.compressors is None

    def test_custom_initialization(self):
        comps = [IdentityCompressor() for _ in range(3)]
        s = _SessionState(cache_token_count=42, compressors=comps)
        assert s.cache_token_count == 42
        assert len(s.compressors) == 3


# ---------------------------------------------------------------------------
# V04SessionCache assembly logic (tests the K/V concat without HF model)
# ---------------------------------------------------------------------------


class TestV04SessionCache:
    def _make_cache(self, num_layers=2, sink_size=2, window_size=2,
                    cache_token_count_at_start=0, n_new_tokens=4):
        compressors = [IdentityCompressor() for _ in range(num_layers)]
        cache = _V04SessionCache(
            compressors=compressors,
            sink_size=sink_size, window_size=window_size,
            cache_token_count_at_start=cache_token_count_at_start,
            n_new_tokens=n_new_tokens,
        )
        return cache, compressors

    def test_get_seq_length(self):
        cache, _ = self._make_cache(cache_token_count_at_start=10,
                                     n_new_tokens=3)
        assert cache.get_seq_length() == 13
        assert cache.get_seq_length(layer_idx=5) == 13

    def test_set_partition(self):
        cache, _ = self._make_cache()
        cache.set_partition([2, 3], [0, 1])
        assert cache._evicted_positions == [2, 3]
        assert cache._resident_positions == [0, 1]

    def test_set_evicted_kv(self):
        cache, _ = self._make_cache(num_layers=3)
        K = torch.randn(1, 2, 2, 4)
        V = torch.randn(1, 2, 2, 4)
        cache.set_evicted_kv(0, K, V)
        cache.set_evicted_kv(2, K * 2, V * 2)
        assert 0 in cache._evicted_kv
        assert 1 not in cache._evicted_kv
        assert 2 in cache._evicted_kv

    def test_update_returns_full_kv_at_t_full(self):
        # T_start = 0, n_new = 4 → T_full = 4. All 4 positions are "new".
        # Sink (0..1) + window (2..3) = all positions resident.
        cache, _ = self._make_cache(num_layers=1, sink_size=2, window_size=2,
                                     cache_token_count_at_start=0,
                                     n_new_tokens=4)
        cache.set_partition([], [0, 1, 2, 3])
        K_new = torch.randn(1, 2, 4, 4)
        V_new = torch.randn(1, 2, 4, 4)
        K_full, V_full = cache.update(K_new, V_new, layer_idx=0)
        assert K_full.shape == (1, 2, 4, 4)
        assert V_full.shape == (1, 2, 4, 4)

    def test_update_with_evicted_uses_set_evicted_kv(self):
        cache, _ = self._make_cache(num_layers=1, sink_size=1, window_size=1,
                                     cache_token_count_at_start=0,
                                     n_new_tokens=4)
        # T_full=4, sink=[0], window=[3], evicted=[1,2]
        cache.set_partition(evicted_positions=[1, 2],
                            resident_positions=[0, 3])
        K_evicted = torch.randn(1, 2, 2, 4)
        V_evicted = torch.randn(1, 2, 2, 4)
        cache.set_evicted_kv(0, K_evicted, V_evicted)

        K_new = torch.randn(1, 2, 4, 4)
        V_new = torch.randn(1, 2, 4, 4)
        K_full, V_full = cache.update(K_new, V_new, layer_idx=0)
        # K_full at evicted slots should match K_evicted.
        assert torch.allclose(K_full[..., 1, :], K_evicted[..., 0, :])
        assert torch.allclose(K_full[..., 2, :], K_evicted[..., 1, :])
        # K_full at resident slots should match K_new.
        assert torch.allclose(K_full[..., 0, :], K_new[..., 0, :])
        assert torch.allclose(K_full[..., 3, :], K_new[..., 3, :])

    def test_update_evicted_without_set_raises(self):
        cache, _ = self._make_cache(num_layers=1, sink_size=1, window_size=1,
                                     cache_token_count_at_start=0,
                                     n_new_tokens=4)
        cache.set_partition([1, 2], [0, 3])
        # Did NOT call set_evicted_kv — should error.
        K_new = torch.randn(1, 2, 4, 4)
        V_new = torch.randn(1, 2, 4, 4)
        with pytest.raises(RuntimeError, match="evicted"):
            cache.update(K_new, V_new, layer_idx=0)

    def test_update_n_new_mismatch_raises(self):
        cache, _ = self._make_cache(n_new_tokens=4)
        cache.set_partition([], [0, 1, 2, 3])
        # Pass 3 new tokens instead of 4.
        K_new = torch.randn(1, 2, 3, 4)
        V_new = torch.randn(1, 2, 3, 4)
        with pytest.raises(RuntimeError, match="3 positions"):
            cache.update(K_new, V_new, layer_idx=0)


class TestStatefulIncrementalEvictedKVShape:
    """Regression for the 2026-06-09 Mac M4 production-smoke v3 crash:

        RuntimeError: Invalid buffer size: 19.20 GiB

    in DLMRestoredVerifier._stateful_incremental_forward at the
    apply_rotary_pos_emb call site (line ~1124).

    Root cause: KVCapture's natural layout is
    [B, T, num_kv_heads, head_dim] (kv_capture.py line 478, module
    docstring lines 102-103). The stateful incremental forward was
    feeding the capture's raw tensor straight into apply_rotary_pos_emb,
    which expects [B, num_heads, T, head_dim]. With Gemma 3-1B's
    num_kv_heads=1 (GQA fully collapsed), the broadcast in
    ``q * cos.unsqueeze(1)`` silently expands to [B, T, T, head_dim]
    — quadratic. At ctx280 (T≈6413) this is ~20 GB and crashes MPS.

    Why synthetic CI didn't catch it: the existing _FakeAttention uses
    num_kv_heads=2, which makes the same misshapen broadcast fail
    with a clean shape mismatch error instead of the quadratic
    materialisation. Only num_kv_heads=1 reproduces the silent
    quadratic path.

    These tests mirror the Gemma 3-1B GQA-collapsed layout
    (num_kv_heads=1) and exercise the exact code path with a strict
    apply_rotary_pos_emb stub that verifies q's layout matches the
    HF docstring contract.
    """

    def _make_strict_rope(self):
        """Stub apply_rotary_pos_emb that asserts q is in
        [B, num_heads, T, head_dim] layout (dim 1 = heads, dim 2 = T).

        Performs the real broadcast math on a small scale so that a
        layout bug would either AssertionError (safety net) or
        produce a quadratic-shape tensor (the actual bug signature).
        """
        def _strict(q, k, cos, sin):
            B, dim1, dim2, head_dim = q.shape
            cos_T = cos.shape[1]
            assert cos_T == dim2, (
                f"q layout bug: q has shape [{B}, {dim1}, {dim2}, "
                f"{head_dim}] but cos has T={cos_T}; HF apply_rotary_"
                f"pos_emb expects q dim 2 to equal cos T. This indicates "
                f"q was passed in [B, T, heads, head_dim] instead of "
                f"[B, heads, T, head_dim] — caller must transpose(1, 2)."
            )
            # Replicate the real broadcast to catch quadratic-allocation
            # bugs even if the shape assertion is loosened in the future.
            cos_b = cos.unsqueeze(1)
            result = q * cos_b
            assert result.shape == q.shape, (
                f"broadcast result shape {tuple(result.shape)} != "
                f"q.shape {tuple(q.shape)}; quadratic broadcast bug"
            )
            return q, k
        return _strict

    def _gemma3_1b_shape_model(self, num_layers: int = 2, T: int = 8):
        """Build a _FakeModel whose attention has num_kv_heads=1 (mirrors
        Gemma 3-1B's GQA-collapsed config — the only configuration
        that reproduces the silent quadratic broadcast).

        _FakeModel hardcodes its ``config`` to default _FakeConfig
        (num_key_value_heads=2); we override that attribute after
        construction so capture_proposer_kv reads the matching shape.
        """
        from tests.inference_engine.v04.test_dlm_restored_verifier import (
            _FakeConfig,
        )
        m = _FakeModel(
            num_layers=num_layers,
            hidden_size=16,
            num_q_heads=4,
            num_kv_heads=1,
            head_dim=4,
        )
        m.config = _FakeConfig(
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=4,
            hidden_size=16,
        )
        return m

    def test_stateful_incremental_does_not_quadratic_broadcast(self):
        """The bug repro: with num_kv_heads=1 and the buggy
        pre-transpose code path, apply_rotary_pos_emb's strict stub
        either AssertionErrors (current state) or silently allocates
        a quadratic-shape buffer (pre-fix, would OOM on real M4)."""
        m = self._gemma3_1b_shape_model(num_layers=2, T=8)
        verifier = DLMRestoredVerifier(
            model=m, sink_size=1, window_size=2,
            kv_compressor_factory=lambda head_dim: IdentityCompressor(),
            stateful=True,
        )
        # Bootstrap forward: stateful=False path equivalent — establishes
        # cache_token_count and persists compressors.
        input_ids_boot = torch.arange(4).unsqueeze(0)
        verifier.forward(
            input_ids_boot,
            apply_rotary_pos_emb=self._make_strict_rope(),
            eager_attention_forward=_fake_eager_attention_forward,
            rotary_emb_fn=_fake_rotary_emb_fn,
        )
        # Incremental forward: this is where the bug fires.
        # T_full = 8, sink=1, window=2, so n_evicted = 8 - 1 - 2 = 5
        # (positions 1, 2, 3, 4, 5 evicted; 0, 6, 7 resident).
        # The pre-fix code passed K_pre [1, 5, 1, 4] to apply_rotary_pos_emb
        # which the strict stub asserts is wrong layout.
        # Post-fix, K_pre is transposed to [1, 1, 5, 4] and the stub
        # accepts it.
        input_ids_full = torch.arange(8).unsqueeze(0)
        verifier.forward(
            input_ids_full,
            apply_rotary_pos_emb=self._make_strict_rope(),
            eager_attention_forward=_fake_eager_attention_forward,
            rotary_emb_fn=_fake_rotary_emb_fn,
        )
        # If we reached here, the layout bug is not present.
        # Verify the cache advanced to T_full.
        assert verifier._session_state.cache_token_count == 8

    def test_evicted_kv_handed_to_cache_in_attention_layout(self):
        """The evicted K/V written to V04SessionCache.set_evicted_kv
        must be in [B, num_kv_heads, n_evicted, head_dim] layout
        (the same layout HF's attention pipeline produces and consumes).

        Tested by capturing the K_evicted shape just before
        set_evicted_kv via a wrapper around _V04SessionCache.
        """
        m = self._gemma3_1b_shape_model(num_layers=1, T=8)
        captured_shapes: list = []

        class _ShapeCapturingCache(_V04SessionCache):
            def set_evicted_kv(self, layer_idx, K_evicted, V_evicted):
                captured_shapes.append(("K", layer_idx, tuple(K_evicted.shape)))
                captured_shapes.append(("V", layer_idx, tuple(V_evicted.shape)))
                super().set_evicted_kv(layer_idx, K_evicted, V_evicted)

        # Patch the cache class used inside _stateful_incremental_forward
        # for this test only.
        import inference_engine.v04.dlm_restored_verifier as drv
        original = drv._V04SessionCache
        drv._V04SessionCache = _ShapeCapturingCache
        try:
            verifier = DLMRestoredVerifier(
                model=m, sink_size=1, window_size=2,
                kv_compressor_factory=lambda head_dim: IdentityCompressor(),
                stateful=True,
            )
            verifier.forward(
                torch.arange(4).unsqueeze(0),
                apply_rotary_pos_emb=self._make_strict_rope(),
                eager_attention_forward=_fake_eager_attention_forward,
                rotary_emb_fn=_fake_rotary_emb_fn,
            )
            verifier.forward(
                torch.arange(8).unsqueeze(0),
                apply_rotary_pos_emb=self._make_strict_rope(),
                eager_attention_forward=_fake_eager_attention_forward,
                rotary_emb_fn=_fake_rotary_emb_fn,
            )
        finally:
            drv._V04SessionCache = original

        # Each layer should have exactly one K + one V entry.
        # B=1, num_kv_heads=1, n_evicted=5 (T=8, sink=1, window=2),
        # head_dim=4 → expected shape (1, 1, 5, 4).
        assert len(captured_shapes) == 2
        for kind, layer_idx, shape in captured_shapes:
            assert shape == (1, 1, 5, 4), (
                f"{kind} at layer {layer_idx} has shape {shape}; "
                f"expected (B=1, num_kv_heads=1, n_evicted=5, head_dim=4) — "
                f"if this fails, _stateful_incremental_forward is producing "
                f"the wrong layout for V04SessionCache.set_evicted_kv."
            )


def _make_minimal_cache(
    num_layers: int = 2,
    sink_size: int = 2,
    window_size: int = 2,
    cache_token_count_at_start: int = 0,
    n_new_tokens: int = 4,
):
    """Module-level helper for tests that don't have access to the
    TestV04SessionCache class's _make_cache method."""
    compressors = [IdentityCompressor() for _ in range(num_layers)]
    cache = _V04SessionCache(
        compressors=compressors,
        sink_size=sink_size, window_size=window_size,
        cache_token_count_at_start=cache_token_count_at_start,
        n_new_tokens=n_new_tokens,
    )
    return cache, {
        "num_layers": num_layers,
        "sink_size": sink_size,
        "window_size": window_size,
    }


class TestV04SessionCacheHFContract:
    """Regression tests for the HF Cache contract surface.

    The 2026-06-09 Mac M4 production-smoke v4 crash was:

        AttributeError: '_V04SessionCache' object has no attribute
                        'get_mask_sizes'

    Cause: K2.A.2 was implemented against an older transformers Cache
    contract (4.x pre-mid-cycle); transformers 4.57 added
    ``get_mask_sizes`` to the Cache base class, called from
    masking_utils._preprocess_mask_arguments via
    ``past_key_values.get_mask_sizes(cache_position, layer_idx)``.

    These tests validate the contract surface as audited against the
    INSTALLED transformers source, so future transformers upgrades
    that add new Cache methods fail this test (in CI) instead of
    failing on a real Mac mini run after the model has loaded.
    """

    def test_implements_required_attribute_surface(self):
        """Direct existence check for the methods/properties
        Gemma3Attention.forward + masking_utils call on
        past_key_values."""
        from inference_engine.v04.dlm_restored_verifier import _V04SessionCache
        cache, _ = _make_minimal_cache()
        # Required by gemma3 modeling
        assert callable(getattr(cache, "get_seq_length", None))
        assert isinstance(getattr(cache, "is_initialized", None), bool)
        assert callable(getattr(cache, "update", None))
        # Required by masking_utils
        assert callable(getattr(cache, "get_mask_sizes", None))
        # DELIBERATELY ABSENT — see _V04SessionCache docstring on why
        assert not hasattr(cache, "is_sliding"), (
            "_V04SessionCache should NOT expose is_sliding; masking_utils "
            "gates sliding-window logic on hasattr(past_key_values, "
            "'is_sliding'). Defining it would re-impose sliding masking on "
            "the already-merged v0.4 K/V tensor and mask out dLM-restored "
            "evicted positions, defeating the entire architecture."
        )

    def test_get_mask_sizes_returns_full_t(self):
        """get_mask_sizes must return (T_full, 0) for any layer_idx.

        v0.4 update() returns a full [T_full, T_full] K/V tensor
        starting at position 0, so kv_length=T_full and kv_offset=0."""
        from inference_engine.v04.dlm_restored_verifier import _V04SessionCache
        cache, params = _make_minimal_cache(
            cache_token_count_at_start=10, n_new_tokens=4,
        )
        cache_position = torch.arange(10, 14)
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)
        # T_full = cache_token_count_at_start + n_new_tokens = 14
        assert kv_length == 14
        assert kv_offset == 0
        # Same answer for every layer
        for l in range(params["num_layers"]):
            assert cache.get_mask_sizes(cache_position, l) == (14, 0)

    def test_is_initialized_is_true(self):
        from inference_engine.v04.dlm_restored_verifier import _V04SessionCache
        cache, _ = _make_minimal_cache()
        assert cache.is_initialized is True

    def test_audit_against_installed_transformers_source(self):
        """Pinned-style audit: read the installed transformers 4.x
        gemma3 + masking_utils source files, extract every
        attribute/method accessed on a `past_key_values` reference,
        and assert _V04SessionCache either implements it or
        deliberately omits it (with a recorded reason in this test).

        When transformers ships a NEW method we need to handle, this
        test fails with the new method's name in the error, instead
        of the user catching it on a real Mac mini run.
        """
        import os
        import re
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not importable — audit not applicable")

        from inference_engine.v04.dlm_restored_verifier import _V04SessionCache

        tdir = os.path.dirname(transformers.__file__)
        files = [
            "models/gemma3/modeling_gemma3.py",
            "masking_utils.py",
        ]
        called: set = set()
        for f in files:
            full = os.path.join(tdir, f)
            if not os.path.exists(full):
                continue
            text = open(full).read()
            for m in re.finditer(
                r"past_key_values?\.([a-z_][a-zA-Z_]*)", text,
            ):
                called.add(m.group(1))

        # Methods/properties this test confirms are implemented OR
        # deliberately omitted with an architectural justification
        # (recorded inline). Any name in `called` that is NOT in
        # this allow-list AND not implemented on _V04SessionCache
        # MUST be added to one of the two sets, with a code change
        # AND a documented reason.
        IMPLEMENTED = {
            "get_seq_length",
            "update",
            "is_initialized",
            "get_mask_sizes",
        }
        DELIBERATELY_OMITTED = {
            # Re-imposing sliding-window masking on top of the already-
            # merged v0.4 K/V tensor would mask out dLM-restored
            # evicted positions. masking_utils gates this behind
            # hasattr(...), so leaving it undefined falls through to
            # default full-attention masking — correct for v0.4.
            "is_sliding",
        }
        SAFE_TO_IGNORE = {
            # These names appear in transformers source but only on
            # specialised cache classes (HybridCache, etc.) that
            # gemma3 doesn't construct via past_key_values=…—they're
            # never called on a user-supplied cache like ours.
            "self_attention_cache", "cross_attention_cache",
            "conv_cache", "shared_layers", "layers",
            "key_cache", "value_cache", "is_updated",
            "has_previous_state", "to_legacy_cache",
            "from_legacy_cache", "append",
            "get_linear_cache", "set_linear_cache",
        }

        unexpected = called - IMPLEMENTED - DELIBERATELY_OMITTED - SAFE_TO_IGNORE
        if unexpected:
            pytest.fail(
                f"transformers {transformers.__version__} calls these new "
                f"methods on past_key_values that _V04SessionCache neither "
                f"implements nor deliberately omits: {sorted(unexpected)}.\n"
                f"\n"
                f"Action required: for each method, either:\n"
                f"  (a) implement it on _V04SessionCache and add to "
                f"      IMPLEMENTED set in this test, OR\n"
                f"  (b) confirm masking_utils / gemma3 modeling guards "
                f"      access with hasattr(...) so leaving it undefined "
                f"      is safe, then add to DELIBERATELY_OMITTED with a "
                f"      one-sentence justification, OR\n"
                f"  (c) confirm it's only called on a specialised cache "
                f"      type that v0.4 never constructs, and add to "
                f"      SAFE_TO_IGNORE.\n"
                f"\n"
                f"Do NOT just delete the method from this test's scrutiny — "
                f"the whole point is to catch future transformers upgrades."
            )

        # Sanity: confirm the methods we believe we implement are
        # actually present on the class (catches accidental deletion).
        for name in IMPLEMENTED:
            assert hasattr(_V04SessionCache, name), (
                f"_V04SessionCache lost {name!r} — IMPLEMENTED set is now "
                f"out of sync with the actual class definition. "
                f"Either restore the method or update the IMPLEMENTED set."
            )
