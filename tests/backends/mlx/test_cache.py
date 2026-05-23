"""Tests for `inference_engine.backends.mlx.cache.SinkWindowKVCache`.

Mac-only. The class is small enough to test deterministically with
synthetic K/V tensors — every branch (first update, subsequent update,
trim trigger, full trim, partial trim, drop, error paths) is covered
without needing a real model.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from inference_engine.backends.mlx.cache import (
    SinkWindowKVCache,
    cache_seq_length,
    make_sink_window_cache,
    total_kv_bytes,
)


def _kv_pair(L: int, n_kv_heads: int = 2, head_dim: int = 4, dtype=None):
    """Build (K, V) of shape [1, n_kv_heads, L, head_dim]."""
    dtype = dtype or mx.bfloat16
    keys = mx.zeros((1, n_kv_heads, L, head_dim), dtype=dtype)
    values = mx.ones((1, n_kv_heads, L, head_dim), dtype=dtype)
    return keys, values


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_construction_defaults() -> None:
    c = SinkWindowKVCache()
    assert c.sink_size == 4
    assert c.window_size == 64
    assert c.keys is None
    assert c.values is None
    assert c.offset == 0


@pytest.mark.parametrize(
    "sink,window,err",
    [
        (-1, 8, "sink_size must be >= 0"),
        (4, 0, "window_size must be > 0"),
        (4, -3, "window_size must be > 0"),
    ],
)
def test_construction_validates(sink, window, err) -> None:
    with pytest.raises(ValueError, match=err):
        SinkWindowKVCache(sink_size=sink, window_size=window)


# ---------------------------------------------------------------------------
# update_and_fetch
# ---------------------------------------------------------------------------

def test_first_update_no_trim() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    k, v = _kv_pair(5)
    full_k, full_v = c.update_and_fetch(k, v)
    # Returned full tensors equal the input (cache was empty)
    assert tuple(full_k.shape) == (1, 2, 5, 4)
    assert c.offset == 5
    assert int(c.keys.shape[2]) == 5  # below budget=12, so stored as-is


def test_subsequent_update_below_budget() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)  # budget=12
    c.update_and_fetch(*_kv_pair(5))
    full_k, full_v = c.update_and_fetch(*_kv_pair(3))
    # Return is full concat = 5 + 3 = 8 (still below budget 12)
    assert int(full_k.shape[2]) == 8
    # Stored buffer = 8
    assert int(c.keys.shape[2]) == 8
    assert c.offset == 8


def test_update_triggers_trim() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)  # budget=12
    c.update_and_fetch(*_kv_pair(10))
    # Now 10 stored, offset=10
    full_k, _ = c.update_and_fetch(*_kv_pair(5))
    # Returned: full = 10 + 5 = 15 (this step's K)
    assert int(full_k.shape[2]) == 15
    # Stored: trimmed to budget = 12 (sink 4 + window 8). Offset = 15.
    assert int(c.keys.shape[2]) == 12
    assert c.offset == 15


def test_update_validates_kv_dims() -> None:
    c = SinkWindowKVCache()
    bad_k = mx.zeros((1, 2, 4), dtype=mx.bfloat16)  # ndim=3
    bad_v = mx.zeros((1, 2, 4, 4), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="4-D"):
        c.update_and_fetch(bad_k, bad_v)


def test_update_validates_kv_seq_match() -> None:
    c = SinkWindowKVCache()
    k = mx.zeros((1, 2, 5, 4), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, 4, 4), dtype=mx.bfloat16)  # mismatched seq
    with pytest.raises(RuntimeError, match="mismatched seq dims"):
        c.update_and_fetch(k, v)


# ---------------------------------------------------------------------------
# make_mask
# ---------------------------------------------------------------------------

def test_make_mask_empty_cache_single_token() -> None:
    """N=1 returns None (no mask needed)."""
    c = SinkWindowKVCache()
    assert c.make_mask(1) is None


def test_make_mask_empty_cache_multi_token() -> None:
    """N>1 with empty cache returns the 'causal' string sentinel."""
    c = SinkWindowKVCache()
    m = c.make_mask(8)
    # mlx_lm's create_attention_mask returns 'causal' for offset=0 / N>1
    # without explicit window. The sentinel is interpreted by SDPA.
    assert m == "causal" or hasattr(m, "shape")  # allow either, depending on mlx_lm minor version


def test_make_mask_after_updates_uses_pre_update_buffer_size() -> None:
    """Mask offset must equal pre-update buffer size for K-shape match."""
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    c.update_and_fetch(*_kv_pair(5))
    # Now buffer has 5 tokens. Next make_mask(N=3) should see offset=5.
    m = c.make_mask(3, return_array=True)
    assert hasattr(m, "shape")
    assert m.shape[-1] == 5 + 3  # [3, 8]


def test_make_mask_with_window_size_kwarg() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    c.update_and_fetch(*_kv_pair(5))
    m = c.make_mask(3, window_size=16)
    assert hasattr(m, "shape")


# ---------------------------------------------------------------------------
# trim (drop from end)
# ---------------------------------------------------------------------------

def test_trim_empty_cache_is_zero() -> None:
    c = SinkWindowKVCache()
    assert c.trim(5) == 0


def test_trim_drops_from_end() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    c.update_and_fetch(*_kv_pair(10))
    n = c.trim(3)
    assert n == 3
    assert int(c.keys.shape[2]) == 7
    assert c.offset == 7


def test_trim_clamped_at_buffer_size() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    c.update_and_fetch(*_kv_pair(5))
    n = c.trim(50)
    assert n == 5  # clamped
    assert int(c.keys.shape[2]) == 0
    assert c.offset == 0


def test_trim_zero_is_noop() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    c.update_and_fetch(*_kv_pair(5))
    pre_keys = c.keys
    n = c.trim(0)
    assert n == 0
    assert c.keys is pre_keys


def test_trim_negative_clamped_to_zero() -> None:
    c = SinkWindowKVCache()
    c.update_and_fetch(*_kv_pair(3))
    assert c.trim(-1) == 0


# ---------------------------------------------------------------------------
# Properties / introspection
# ---------------------------------------------------------------------------

def test_size_returns_offset() -> None:
    c = SinkWindowKVCache()
    assert c.size() == 0
    c.update_and_fetch(*_kv_pair(7))
    assert c.size() == 7


def test_empty() -> None:
    c = SinkWindowKVCache()
    assert c.empty() is True
    c.update_and_fetch(*_kv_pair(3))
    assert c.empty() is False


def test_nbytes_empty_zero() -> None:
    c = SinkWindowKVCache()
    assert c.nbytes == 0


def test_nbytes_positive_after_update() -> None:
    c = SinkWindowKVCache(sink_size=4, window_size=8)
    c.update_and_fetch(*_kv_pair(5))
    # K + V each: 1*2*5*4 elems = 40 elems * 2 (bf16) = 80 bytes per tensor
    assert c.nbytes == 160


def test_is_trimmable() -> None:
    c = SinkWindowKVCache()
    assert c.is_trimmable() is True


# ---------------------------------------------------------------------------
# state / meta_state round-trip
# ---------------------------------------------------------------------------

def test_state_round_trip() -> None:
    c = SinkWindowKVCache(sink_size=2, window_size=6)
    c.update_and_fetch(*_kv_pair(5))
    saved = c.state
    saved_meta = c.meta_state

    c2 = SinkWindowKVCache(sink_size=2, window_size=6)
    c2.state = saved
    c2.meta_state = saved_meta
    assert c2.sink_size == 2
    assert c2.window_size == 6
    assert c2.offset == 5
    assert int(c2.keys.shape[2]) == 5


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_make_sink_window_cache_size_matches_model_layers() -> None:
    """Use a 3-layer toy mock for the model.layers attribute (this
    helper doesn't otherwise touch the model)."""
    class _ToyModel:
        layers = [object(), object(), object()]
    cache = make_sink_window_cache(_ToyModel(), sink_size=2, window_size=4)
    assert len(cache) == 3
    for layer in cache:
        assert isinstance(layer, SinkWindowKVCache)
        assert layer.sink_size == 2
        assert layer.window_size == 4


def test_total_kv_bytes_aggregates_layers() -> None:
    cs = [SinkWindowKVCache(sink_size=2, window_size=4) for _ in range(3)]
    for c in cs:
        c.update_and_fetch(*_kv_pair(3))
    total = total_kv_bytes(cs)
    # Each layer: 1*2*3*4*2 = 48 bytes per K, same for V => 96 bytes
    # 3 layers * 96 = 288
    assert total == 288


def test_total_kv_bytes_empty_layers() -> None:
    cs = [SinkWindowKVCache() for _ in range(2)]
    assert total_kv_bytes(cs) == 0


def test_cache_seq_length_picks_first_nonempty() -> None:
    cs = [SinkWindowKVCache() for _ in range(3)]
    cs[2].update_and_fetch(*_kv_pair(7))
    assert cache_seq_length(cs) == 7


def test_cache_seq_length_all_empty() -> None:
    cs = [SinkWindowKVCache() for _ in range(2)]
    assert cache_seq_length(cs) == 0
