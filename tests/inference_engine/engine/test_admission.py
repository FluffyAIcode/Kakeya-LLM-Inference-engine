"""Unit tests for the Kakeya engine peak-window admission + bounded-KV model."""

from __future__ import annotations

import pytest

from inference_engine.engine.admission import (
    BoundedKVModel,
    exact_layer_indices_for_layer_types,
    max_concurrent_sessions,
)

# gemma-4-26B-A4B: 30 layers (5 full-attn), 8 KV-heads, head_dim 256, bf16.
GEMMA4 = dict(num_layers=30, num_kv_heads=8, head_dim=256,
              n_exact_layers=5, sink=4, window=64, dtype_bytes=2)


def test_per_token_per_layer_bytes():
    m = BoundedKVModel(**GEMMA4)
    assert m.per_token_per_layer_bytes == 2 * 8 * 256 * 2  # 8192


def test_resident_vs_full_kv_at_62k():
    m = BoundedKVModel(**GEMMA4)
    C = 62038
    resident = m.resident_bytes(C)
    full = m.full_kv_bytes(C)
    assert resident == (5 * C + 25 * (4 + 64)) * 8192
    assert full == 30 * C * 8192
    # bounded ~2.55 GB, full ~15.2 GB on a full-attention engine
    assert 2.5e9 < resident < 2.6e9
    assert 15.0e9 < full < 15.5e9
    assert m.advantage_ratio(C) == pytest.approx(full / resident, rel=1e-9)
    assert m.advantage_ratio(C) > 5.9  # ~6x on a full-attention model


def test_resident_grows_only_in_exact_term():
    """Doubling context only grows the exact-layer term (others are capped)."""
    m = BoundedKVModel(**GEMMA4)
    other = (30 - 5) * (4 + 64) * m.per_token_per_layer_bytes
    for C in (4096, 8192, 65536):
        assert m.resident_bytes(C) == 5 * C * m.per_token_per_layer_bytes + other


def test_short_context_caps_other_layers_at_context_len():
    m = BoundedKVModel(**GEMMA4)
    C = 10  # below sink+window
    assert m.resident_bytes(C) == 30 * C * m.per_token_per_layer_bytes


def test_max_concurrent_sessions():
    # 140 GB budget, 52 GB weights, 2.55 GB/session -> ~34
    n = max_concurrent_sessions(memory_budget_bytes=140 * 10**9,
                                model_weight_bytes=52 * 10**9,
                                per_session_bytes=int(2.55e9))
    assert n == int((140e9 - 52e9) // 2.55e9)
    assert 30 <= n <= 36


def test_max_concurrent_zero_when_no_room():
    assert max_concurrent_sessions(memory_budget_bytes=50 * 10**9,
                                   model_weight_bytes=52 * 10**9,
                                   per_session_bytes=int(2.55e9)) == 0
    with pytest.raises(ValueError):
        max_concurrent_sessions(memory_budget_bytes=140, model_weight_bytes=0,
                                per_session_bytes=0)


def test_full_attention_model_advantage_is_large():
    """No native sliding (all layers exact in vLLM's view) → bounded keeps only
    5 exact + sink/window on the rest; advantage is the headline ~6x at 62k."""
    m = BoundedKVModel(num_layers=32, num_kv_heads=8, head_dim=128,
                       n_exact_layers=4, sink=4, window=64)
    assert m.advantage_ratio(65536) > 6.0


def test_exact_layer_indices_from_layer_types():
    lt = (["sliding_attention"] * 5 + ["full_attention"]) * 2
    assert exact_layer_indices_for_layer_types(lt) == [5, 11]


def test_validation():
    with pytest.raises(ValueError):
        BoundedKVModel(num_layers=4, num_kv_heads=8, head_dim=256,
                       n_exact_layers=5, sink=4, window=64)
    with pytest.raises(ValueError):
        BoundedKVModel(num_layers=30, num_kv_heads=8, head_dim=256,
                       n_exact_layers=5, sink=-1, window=64)
