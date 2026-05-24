"""Unit tests for :class:`KVSlab` and :class:`SlabConfig`."""

from __future__ import annotations

import pytest
import torch

from inference_engine.memory.slab import KVSlab, SlabConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> SlabConfig:
    return SlabConfig(
        num_layers=2, num_heads=4, sink_size=2,
        window_size=4, head_dim=8, dtype=torch.float32,
    )


@pytest.fixture
def slab(cfg: SlabConfig) -> KVSlab:
    return KVSlab(cfg)


def _step(cfg: SlabConfig, T: int, *, fill: float = 1.0):
    """Build a (key_steps, value_steps) pair filled with a constant."""
    shape = (cfg.num_layers, cfg.num_heads, T, cfg.head_dim)
    k = torch.full(shape, float(fill), dtype=cfg.dtype)
    v = torch.full(shape, float(fill) * 2, dtype=cfg.dtype)
    return k, v


# ---------------------------------------------------------------------------
# SlabConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"num_layers": 0}, {"num_layers": -1},
    {"num_heads": 0}, {"num_heads": -2},
    {"sink_size": -1},
    {"window_size": 0}, {"window_size": -3},
    {"head_dim": 0}, {"head_dim": -5},
])
def test_slab_config_rejects_invalid_dims(kwargs):
    base = dict(num_layers=2, num_heads=2, sink_size=1, window_size=2, head_dim=4)
    base.update(kwargs)
    with pytest.raises(ValueError):
        SlabConfig(**base)


def test_slab_config_capacity_is_sink_plus_window(cfg):
    assert cfg.capacity == cfg.sink_size + cfg.window_size


def test_slab_config_allows_zero_sink():
    cfg = SlabConfig(num_layers=1, num_heads=1, sink_size=0,
                     window_size=4, head_dim=8)
    assert cfg.capacity == 4


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_allocates_zero_buffers(cfg):
    s = KVSlab(cfg)
    assert s.keys.shape == (cfg.num_layers, cfg.num_heads, cfg.capacity, cfg.head_dim)
    assert s.values.shape == s.keys.shape
    assert s.keys.dtype == cfg.dtype
    assert torch.equal(s.keys, torch.zeros_like(s.keys))
    assert s.logical_size == 0


def test_kv_bytes_equals_keys_plus_values(cfg):
    s = KVSlab(cfg)
    elem = s.keys.element_size()
    expected = 2 * cfg.num_layers * cfg.num_heads * cfg.capacity * cfg.head_dim * elem
    assert s.kv_bytes == expected


# ---------------------------------------------------------------------------
# Append: simple cases under capacity
# ---------------------------------------------------------------------------


def test_append_single_step_grows_logical_size(slab, cfg):
    k, v = _step(cfg, 1, fill=1.0)
    slab.append(k, v)
    assert slab.logical_size == 1


def test_append_multiple_steps_grow_logical_size(slab, cfg):
    k, v = _step(cfg, 3, fill=2.0)
    slab.append(k, v)
    assert slab.logical_size == 3


def test_append_writes_into_correct_slots(slab, cfg):
    k1, v1 = _step(cfg, 2, fill=5.0)
    slab.append(k1, v1)
    # The first 2 slots match what we wrote; remaining slots are zero.
    assert torch.equal(slab.keys[:, :, :2, :], k1)
    assert torch.equal(slab.values[:, :, :2, :], v1)
    assert torch.equal(slab.keys[:, :, 2:, :], torch.zeros_like(slab.keys[:, :, 2:, :]))


def test_append_to_capacity_exactly_works(slab, cfg):
    """Appends in window-sized chunks until exactly at capacity."""
    # Capacity = sink + window, but each append's T must be <= window.
    # Fill in steps of 1 to avoid the window-size cap.
    for _ in range(cfg.capacity):
        slab.append(*_step(cfg, 1, fill=1.0))
    assert slab.logical_size == cfg.capacity
    assert slab.is_full


# ---------------------------------------------------------------------------
# Append: overflow → sink+window slide
# ---------------------------------------------------------------------------


def test_append_overflow_preserves_sink_region(cfg):
    """Sink slots must be byte-identical before and after a slide."""
    s = KVSlab(cfg)
    # Fill with capacity worth of distinct values (each step has its
    # own fill value so we can inspect what survives).
    for i in range(cfg.capacity):
        k, v = _step(cfg, 1, fill=float(100 + i))
        s.append(k, v)
    sink_keys_before = s.keys[:, :, :cfg.sink_size, :].clone()
    sink_values_before = s.values[:, :, :cfg.sink_size, :].clone()

    # Now append one more step → forces a slide of one window position.
    k_new, v_new = _step(cfg, 1, fill=999.0)
    s.append(k_new, v_new)

    assert torch.equal(s.keys[:, :, :cfg.sink_size, :], sink_keys_before)
    assert torch.equal(s.values[:, :, :cfg.sink_size, :], sink_values_before)


def test_append_overflow_drops_oldest_window_position(cfg):
    """After a 1-position slide, the value formerly at slot sink+0 is gone;
    the value formerly at slot sink+1 is now at slot sink."""
    s = KVSlab(cfg)
    for i in range(cfg.capacity):
        k, v = _step(cfg, 1, fill=float(100 + i))
        s.append(k, v)
    # Position sink+1 in the original → fill = 100 + sink + 1
    expected_fill_after_slide = float(100 + cfg.sink_size + 1)

    k_new, v_new = _step(cfg, 1, fill=999.0)
    s.append(k_new, v_new)

    # New "first window slot" should have the previous "second window slot" content.
    assert torch.equal(
        s.keys[:, :, cfg.sink_size, :],
        torch.full(
            (cfg.num_layers, cfg.num_heads, cfg.head_dim),
            expected_fill_after_slide, dtype=cfg.dtype,
        ),
    )


def test_append_overflow_logical_size_caps_at_capacity(cfg):
    s = KVSlab(cfg)
    for _ in range(cfg.capacity + 3):
        k, v = _step(cfg, 1, fill=1.0)
        s.append(k, v)
    assert s.logical_size == cfg.capacity


def test_append_block_overflow_works(cfg):
    """Append a block of length T > 1 that overflows."""
    s = KVSlab(cfg)
    # Pre-fill to (capacity - 1) one step at a time (T-bound is window_size).
    for _ in range(cfg.capacity - 1):
        s.append(*_step(cfg, 1, fill=1.0))
    # Now append 3 → overflow by 2; slide should keep us at capacity.
    # Note: 3 <= window_size (4), so this is a legal append.
    s.append(*_step(cfg, 3, fill=2.0))
    assert s.logical_size == cfg.capacity


# ---------------------------------------------------------------------------
# Append: validation
# ---------------------------------------------------------------------------


def test_append_rejects_3d_tensor(slab, cfg):
    bad = torch.zeros(cfg.num_layers, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype)
    good_v = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    with pytest.raises(ValueError, match=r"key_steps must be 4-D"):
        slab.append(bad, good_v)


def test_append_rejects_wrong_num_layers(slab, cfg):
    bad = torch.zeros((cfg.num_layers + 1, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    good = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    with pytest.raises(ValueError, match=r"shape\[0\]"):
        slab.append(bad, good)


def test_append_rejects_wrong_num_heads(slab, cfg):
    bad = torch.zeros((cfg.num_layers, cfg.num_heads + 1, 1, cfg.head_dim), dtype=cfg.dtype)
    good = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    with pytest.raises(ValueError, match=r"shape\[1\]"):
        slab.append(bad, good)


def test_append_rejects_wrong_head_dim(slab, cfg):
    bad = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim + 1), dtype=cfg.dtype)
    good = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    with pytest.raises(ValueError, match=r"shape\[3\]"):
        slab.append(bad, good)


def test_append_rejects_wrong_dtype(slab, cfg):
    bad = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=torch.float64)
    good = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    with pytest.raises(ValueError, match="dtype="):
        slab.append(bad, good)


def test_append_rejects_zero_T(slab, cfg):
    k = torch.zeros((cfg.num_layers, cfg.num_heads, 0, cfg.head_dim), dtype=cfg.dtype)
    v = torch.zeros_like(k)
    with pytest.raises(ValueError, match="T must be positive"):
        slab.append(k, v)


def test_append_rejects_T_exceeding_window(slab, cfg):
    k, v = _step(cfg, cfg.window_size + 1, fill=1.0)
    with pytest.raises(ValueError, match="exceeds window_size"):
        slab.append(k, v)


def test_append_rejects_value_steps_shape_mismatch(slab, cfg):
    k, _ = _step(cfg, 2, fill=1.0)
    _, v = _step(cfg, 3, fill=1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        slab.append(k, v)


def test_append_rejects_value_steps_wrong_shape(slab, cfg):
    k, _ = _step(cfg, 2, fill=1.0)
    bad_v = torch.zeros((cfg.num_layers, cfg.num_heads, 2, cfg.head_dim + 1), dtype=cfg.dtype)
    with pytest.raises(ValueError, match=r"value_steps"):
        slab.append(k, bad_v)


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


def test_truncate_drops_n_positions(cfg):
    s = KVSlab(cfg)
    k, v = _step(cfg, 4, fill=1.0)
    s.append(k, v)
    dropped = s.truncate(2)
    assert dropped == 2
    assert s.logical_size == 2


def test_truncate_zero_is_noop(cfg):
    s = KVSlab(cfg)
    k, v = _step(cfg, 3, fill=1.0)
    s.append(k, v)
    dropped = s.truncate(0)
    assert dropped == 0
    assert s.logical_size == 3


def test_truncate_rejects_negative(cfg):
    s = KVSlab(cfg)
    with pytest.raises(ValueError, match="drop must be >= 0"):
        s.truncate(-1)


def test_truncate_rejects_more_than_logical(cfg):
    s = KVSlab(cfg)
    k, v = _step(cfg, 2, fill=1.0)
    s.append(k, v)
    with pytest.raises(ValueError, match="exceeds logical_size"):
        s.truncate(5)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_zeros_logical_size(cfg):
    s = KVSlab(cfg)
    k, v = _step(cfg, 2, fill=1.0)
    s.append(k, v)
    s.reset()
    assert s.logical_size == 0


def test_reset_preserves_buffer_identity(cfg):
    """Pool reuse depends on the underlying tensor *object* surviving
    a reset — so we can keep the same allocation across sessions."""
    s = KVSlab(cfg)
    keys_before = s.keys
    s.append(*_step(cfg, 1, fill=1.0))
    s.reset()
    assert s.keys is keys_before


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------


def test_view_shape_equals_logical_size(cfg):
    s = KVSlab(cfg)
    s.append(*_step(cfg, 3, fill=1.0))
    keys, values = s.view(layer_idx=0)
    assert keys.shape == (cfg.num_heads, 3, cfg.head_dim)
    assert values.shape == (cfg.num_heads, 3, cfg.head_dim)


def test_view_empty_slab_returns_zero_length_view(cfg):
    s = KVSlab(cfg)
    keys, values = s.view(layer_idx=0)
    assert keys.shape == (cfg.num_heads, 0, cfg.head_dim)
    assert values.shape == (cfg.num_heads, 0, cfg.head_dim)


def test_view_shares_storage_with_slab(cfg):
    s = KVSlab(cfg)
    k, v = _step(cfg, 2, fill=1.0)
    s.append(k, v)
    keys_view, _values_view = s.view(layer_idx=0)
    # Mutate the view; slab buffer should reflect it.
    keys_view[:, 0, :] = 42.0
    assert torch.all(s.keys[0, :, 0, :] == 42.0)


def test_view_invalid_layer_idx_raises(cfg):
    s = KVSlab(cfg)
    with pytest.raises(IndexError, match="out of range"):
        s.view(layer_idx=cfg.num_layers)


def test_view_negative_layer_idx_raises(cfg):
    s = KVSlab(cfg)
    with pytest.raises(IndexError, match="out of range"):
        s.view(layer_idx=-1)


# ---------------------------------------------------------------------------
# Live KV bytes
# ---------------------------------------------------------------------------


def test_live_kv_bytes_zero_when_empty(cfg):
    s = KVSlab(cfg)
    assert s.live_kv_bytes == 0


def test_live_kv_bytes_grows_with_logical_size(cfg):
    s = KVSlab(cfg)
    s.append(*_step(cfg, 2, fill=1.0))
    elem = s.keys.element_size()
    expected = 2 * cfg.num_layers * cfg.num_heads * 2 * cfg.head_dim * elem
    assert s.live_kv_bytes == expected


def test_is_full_false_when_below_capacity(cfg):
    s = KVSlab(cfg)
    for _ in range(cfg.capacity - 1):
        s.append(*_step(cfg, 1, fill=1.0))
    assert s.is_full is False


def test_is_full_true_at_capacity(cfg):
    s = KVSlab(cfg)
    for _ in range(cfg.capacity):
        s.append(*_step(cfg, 1, fill=1.0))
    assert s.is_full is True
