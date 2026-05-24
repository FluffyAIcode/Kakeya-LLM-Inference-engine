"""Unit tests for :class:`SlabPool`."""

from __future__ import annotations

import threading

import pytest
import torch

from inference_engine.memory.pool import PoolExhausted, SlabPool
from inference_engine.memory.slab import KVSlab, SlabConfig


@pytest.fixture
def cfg() -> SlabConfig:
    return SlabConfig(
        num_layers=2, num_heads=2, sink_size=1,
        window_size=2, head_dim=4, dtype=torch.float32,
    )


@pytest.fixture
def pool(cfg: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=3, slab_config=cfg)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_allocates_n_slabs(pool):
    assert pool.total_count == 3
    assert pool.available_count == 3
    assert pool.in_use_count == 0


def test_construction_rejects_zero_slabs(cfg):
    with pytest.raises(ValueError, match="num_slabs must be positive"):
        SlabPool(num_slabs=0, slab_config=cfg)


def test_construction_rejects_negative_slabs(cfg):
    with pytest.raises(ValueError, match="num_slabs must be positive"):
        SlabPool(num_slabs=-2, slab_config=cfg)


def test_slab_config_property_round_trips(pool, cfg):
    assert pool.slab_config is cfg


# ---------------------------------------------------------------------------
# Acquire / release lifecycle
# ---------------------------------------------------------------------------


def test_acquire_returns_kv_slab(pool):
    s = pool.acquire()
    assert isinstance(s, KVSlab)


def test_acquire_decrements_available_count(pool):
    pool.acquire()
    assert pool.available_count == 2
    assert pool.in_use_count == 1


def test_acquire_three_then_pool_exhausted(pool):
    for _ in range(3):
        pool.acquire()
    with pytest.raises(PoolExhausted):
        pool.acquire()


def test_release_restores_available_count(pool):
    s = pool.acquire()
    assert pool.available_count == 2
    pool.release(s)
    assert pool.available_count == 3
    assert pool.in_use_count == 0


def test_release_after_acquire_release_can_acquire_again(pool):
    s1 = pool.acquire()
    pool.release(s1)
    s2 = pool.acquire()
    assert isinstance(s2, KVSlab)


def test_release_resets_logical_size(pool, cfg):
    s = pool.acquire()
    k = torch.zeros((cfg.num_layers, cfg.num_heads, 1, cfg.head_dim), dtype=cfg.dtype)
    v = torch.zeros_like(k)
    s.append(k, v)
    assert s.logical_size == 1
    pool.release(s)
    assert s.logical_size == 0


def test_release_alien_slab_raises(pool, cfg):
    rogue = KVSlab(cfg)
    with pytest.raises(ValueError, match="does not belong"):
        pool.release(rogue)


def test_double_release_raises(pool):
    s = pool.acquire()
    pool.release(s)
    with pytest.raises(ValueError, match="not currently in use"):
        pool.release(s)


def test_release_a_never_acquired_pool_slab_raises(pool):
    """If somehow we hand a slab to release that we never acquired,
    we error rather than corrupt the free list."""
    # Reach into the pool to grab a never-acquired slab; this is what
    # a buggy caller might accidentally do via an old reference.
    never_acquired = pool._all_slabs[0]  # not formally acquired
    with pytest.raises(ValueError, match="not currently in use"):
        pool.release(never_acquired)


# ---------------------------------------------------------------------------
# acquire_optional
# ---------------------------------------------------------------------------


def test_acquire_optional_returns_slab_when_free(pool):
    s = pool.acquire_optional()
    assert s is not None
    assert isinstance(s, KVSlab)


def test_acquire_optional_returns_none_when_exhausted(pool):
    for _ in range(3):
        pool.acquire()
    assert pool.acquire_optional() is None


# ---------------------------------------------------------------------------
# Concurrency: lock prevents the same slab being handed out twice
# ---------------------------------------------------------------------------


def test_concurrent_acquire_hands_out_distinct_slabs(cfg):
    """N threads each acquire one slab; no two threads should see the
    same slab object."""
    pool = SlabPool(num_slabs=8, slab_config=cfg)
    acquired: list[KVSlab] = []
    acquired_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()  # synchronize start
        s = pool.acquire()
        with acquired_lock:
            acquired.append(s)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 8 slabs are distinct objects.
    assert len({id(s) for s in acquired}) == 8


# ---------------------------------------------------------------------------
# total_kv_bytes
# ---------------------------------------------------------------------------


def test_total_kv_bytes_is_sum_across_slabs(pool, cfg):
    one = KVSlab(cfg)
    expected = 3 * one.kv_bytes
    assert pool.total_kv_bytes == expected
