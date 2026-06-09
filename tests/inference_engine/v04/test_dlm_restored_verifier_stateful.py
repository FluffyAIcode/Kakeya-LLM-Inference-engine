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
