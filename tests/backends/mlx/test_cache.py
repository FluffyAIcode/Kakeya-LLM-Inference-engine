"""Tests for `inference_engine.backends.mlx.cache` trim helpers.

Mac-only. We construct a synthetic "cache" — a list of objects with
the same surface as `mlx_lm.models.cache.KVCache` (mutable `keys`,
`values`, `offset`) — so we can hit every branch (null layer, layout
violation, sink/window edge, drop > size) deterministically without
having to drive a real model forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

mx = pytest.importorskip("mlx.core")

from inference_engine.backends.mlx.cache import (
    TrimReport,
    cache_seq_length,
    total_kv_bytes,
    trim_caches_to_sink_window,
    truncate_caches_tail,
)


@dataclass
class _FakeLayer:
    """Duck-types `mlx_lm.models.cache.KVCache` for the surface we use."""
    keys: Optional["mx.array"] = None
    values: Optional["mx.array"] = None
    offset: int = 0


def _filled(seq_len: int, n_kv_heads: int = 2, head_dim: int = 4,
            dtype=None) -> _FakeLayer:
    dtype = dtype or mx.bfloat16
    layer = _FakeLayer()
    layer.keys = mx.zeros((1, n_kv_heads, seq_len, head_dim), dtype=dtype)
    layer.values = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=dtype)
    layer.offset = seq_len
    return layer


# ---------------------------------------------------------------------------
# trim_caches_to_sink_window
# ---------------------------------------------------------------------------

def test_trim_no_op_when_under_budget() -> None:
    layers = [_filled(10), _filled(10)]
    report = trim_caches_to_sink_window(
        layers, sink_size=4, window_size=64, keep_offset=10
    )
    assert isinstance(report, TrimReport)
    assert report.layers_trimmed == 0
    assert report.physical_size_before == 20
    assert report.physical_size_after == 20


def test_trim_shrinks_to_budget() -> None:
    layers = [_filled(100), _filled(100)]
    report = trim_caches_to_sink_window(
        layers, sink_size=4, window_size=8, keep_offset=100
    )
    assert report.layers_trimmed == 2
    for layer in layers:
        assert int(layer.keys.shape[2]) == 12  # 4 + 8
        assert int(layer.values.shape[2]) == 12
        assert layer.offset == 100  # global position preserved for RoPE


def test_trim_skips_null_layers() -> None:
    layers = [_FakeLayer(), _filled(50), _FakeLayer()]
    report = trim_caches_to_sink_window(
        layers, sink_size=4, window_size=8, keep_offset=50
    )
    assert report.layers_skipped_null == 2
    assert report.layers_trimmed == 1


def test_trim_validates_args() -> None:
    layers = [_filled(20)]
    with pytest.raises(ValueError, match="sink_size must be"):
        trim_caches_to_sink_window(
            layers, sink_size=-1, window_size=4, keep_offset=20
        )
    with pytest.raises(ValueError, match="window_size must be"):
        trim_caches_to_sink_window(
            layers, sink_size=4, window_size=0, keep_offset=20
        )


def test_trim_detects_kv_shape_disagreement() -> None:
    bad = _filled(10)
    bad.values = mx.zeros((1, 2, 12, 4), dtype=mx.bfloat16)  # mismatched seq
    with pytest.raises(RuntimeError, match="shape inconsistency"):
        trim_caches_to_sink_window(
            [bad], sink_size=4, window_size=4, keep_offset=10
        )


def test_trim_detects_non_4d_keys() -> None:
    bad = _FakeLayer(
        keys=mx.zeros((4, 8), dtype=mx.bfloat16),
        values=mx.zeros((4, 8), dtype=mx.bfloat16),
    )
    with pytest.raises(RuntimeError, match="expected to be 4-D"):
        trim_caches_to_sink_window(
            [bad], sink_size=4, window_size=4, keep_offset=8
        )


def test_trim_with_zero_sink() -> None:
    layers = [_filled(100)]
    trim_caches_to_sink_window(
        layers, sink_size=0, window_size=10, keep_offset=100
    )
    assert int(layers[0].keys.shape[2]) == 10
    assert layers[0].offset == 100


# ---------------------------------------------------------------------------
# truncate_caches_tail
# ---------------------------------------------------------------------------

def test_truncate_drops_zero_is_no_op() -> None:
    layers = [_filled(10)]
    n = truncate_caches_tail(layers, drop=0, new_offset=10)
    assert n == 0
    assert int(layers[0].keys.shape[2]) == 10


def test_truncate_drops_n() -> None:
    layers = [_filled(10), _filled(10)]
    n = truncate_caches_tail(layers, drop=3, new_offset=7)
    assert n == 2
    for layer in layers:
        assert int(layer.keys.shape[2]) == 7
        assert int(layer.values.shape[2]) == 7
        assert layer.offset == 7


def test_truncate_skips_null_layers() -> None:
    layers = [_FakeLayer(), _filled(8)]
    n = truncate_caches_tail(layers, drop=2, new_offset=6)
    assert n == 1
    assert int(layers[1].keys.shape[2]) == 6


def test_truncate_validates_negative_drop() -> None:
    with pytest.raises(ValueError, match="drop must be"):
        truncate_caches_tail([_filled(10)], drop=-1, new_offset=10)


def test_truncate_overflow_raises() -> None:
    with pytest.raises(RuntimeError, match="drop=20 but layer only has"):
        truncate_caches_tail([_filled(10)], drop=20, new_offset=0)


# ---------------------------------------------------------------------------
# total_kv_bytes / cache_seq_length
# ---------------------------------------------------------------------------

def test_total_kv_bytes_empty_cache() -> None:
    assert total_kv_bytes([_FakeLayer(), _FakeLayer()]) == 0


def test_total_kv_bytes_filled_cache() -> None:
    layer = _filled(10, n_kv_heads=2, head_dim=4, dtype=mx.bfloat16)
    # K: 1*2*10*4 = 80 elements, bf16 = 2 bytes => 160
    # V: same => 160
    assert total_kv_bytes([layer]) == 320


def test_cache_seq_length_empty() -> None:
    assert cache_seq_length([_FakeLayer(), _FakeLayer()]) == 0


def test_cache_seq_length_filled() -> None:
    assert cache_seq_length([_FakeLayer(), _filled(7), _filled(7)]) == 7
