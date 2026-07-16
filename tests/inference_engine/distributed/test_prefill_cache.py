from __future__ import annotations

import pytest

from inference_engine.distributed.capability import CacheCompatibility
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    chained_block_hashes,
    compatibility_fingerprint,
    total_payload_bytes,
)


def _compat(block_size: int = 2) -> CacheCompatibility:
    return CacheCompatibility(
        model_id="gemma",
        model_revision="weights-1",
        tokenizer_revision="tok-1",
        cache_format_version="kv-v1",
        quantization="4bit",
        rope_hash="rope",
        layer_geometry_hash="geometry",
        kv_dtype="bfloat16",
        block_size_tokens=block_size,
    )


def test_compatibility_fingerprint_is_stable_and_sensitive():
    a = _compat()
    assert compatibility_fingerprint(a) == compatibility_fingerprint(a)
    assert compatibility_fingerprint(a) != compatibility_fingerprint(_compat(4))


def test_chained_hashes_require_longest_contiguous_prefix():
    hashes = chained_block_hashes([1, 2, 3, 4, 5], _compat())
    changed = chained_block_hashes([1, 9, 3, 4, 5], _compat())
    assert len(hashes) == 3
    assert hashes[0] != changed[0]
    assert hashes[1] != changed[1]
    with pytest.raises(ValueError, match="block_size"):
        chained_block_hashes([1], _compat(0))


def test_store_returns_longest_snapshot_only():
    store = PrefixCacheStore(_compat(), max_bytes=100, node_id="peer")
    hashes = store.put_prefix([1, 2, 3, 4, 5], [b"a", b"bb", b"ccc"])
    lease = store.lookup(hashes + [bytes(32)], now=10.0)
    assert lease.hit_block_count == 3
    assert lease.hit_token_count == 5
    assert lease.transfer_bytes == 3
    assert store.fetch(lease.lease_id, now=10.0)[0].payload == b"ccc"


def test_store_miss_expiry_collision_and_lru():
    store = PrefixCacheStore(_compat(), max_bytes=4, node_id="peer")
    hashes = chained_block_hashes([1, 2, 3, 4], _compat())
    store.put(CacheBlock.create(hashes[0], 2, b"aa"))
    assert not store.put(CacheBlock.create(hashes[0], 2, b"aa"))
    with pytest.raises(ValueError, match="collision"):
        store.put(CacheBlock.create(hashes[0], 2, b"zz"))
    store.put(CacheBlock.create(hashes[1], 4, b"bbb"))
    assert hashes[0] not in store.block_hashes()
    stats = store.stats()
    assert stats.evictions == 1
    assert stats.bytes_evicted == 2
    assert stats.put_failures == 1
    miss = store.lookup([hashes[0]], now=20.0)
    assert not miss.lease_id
    lease = store.lookup([hashes[1]], lease_seconds=1, now=20.0)
    with pytest.raises(KeyError, match="expired"):
        store.fetch(lease.lease_id, now=22.0)


def test_validation_and_stats():
    with pytest.raises(ValueError, match="max_bytes"):
        PrefixCacheStore(_compat(), max_bytes=0, node_id="x")
    with pytest.raises(ValueError, match="node_id"):
        PrefixCacheStore(_compat(), max_bytes=1, node_id="")
    with pytest.raises(ValueError, match="SHA"):
        CacheBlock.create(b"x", 1, b"")
    with pytest.raises(ValueError, match="token_count"):
        CacheBlock.create(bytes(32), 0, b"")
    store = PrefixCacheStore(_compat(), max_bytes=10, node_id="x")
    with pytest.raises(ValueError, match="capacity"):
        store.put(CacheBlock.create(bytes(32), 1, b"x" * 11))
    stats = store.stats()
    assert stats.entry_count == 0
    assert stats.max_bytes == 10
    assert stats.put_failures == 1
    with pytest.raises(ValueError, match="one payload"):
        store.put_prefix([1, 2, 3], [b"only-one"])
    with pytest.raises(ValueError, match="lease_seconds"):
        store.lookup([], lease_seconds=0)
    assert total_payload_bytes([
        CacheBlock.create(bytes(32), 1, b"12"),
        CacheBlock.create(bytes.fromhex("01" * 32), 1, b"345"),
    ]) == 5


def test_pinned_eviction_and_missing_leased_block_guards():
    store = PrefixCacheStore(_compat(), max_bytes=10, node_id="x")
    block = CacheBlock.create(bytes(32), 1, b"12345")
    store.put(block)
    lease = store.lookup([block.block_hash], now=1)
    store.max_bytes = 1
    store._evict_to_budget()
    assert store.block_hashes() == (block.block_hash,)
    store._blocks.pop(block.block_hash)
    with pytest.raises(KeyError, match="evicted"):
        store.fetch(lease.lease_id, now=1)


def test_put_rejects_when_active_lease_pins_capacity():
    import time

    store = PrefixCacheStore(_compat(), max_bytes=5, node_id="x")
    first = CacheBlock.create(bytes(32), 1, b"12345")
    store.put(first)
    store.lookup([first.block_hash], now=time.time())
    second = CacheBlock.create(bytes.fromhex("01" * 32), 1, b"abc")
    with pytest.raises(ValueError, match="pinned"):
        store.put(second)
    assert store.block_hashes() == (first.block_hash,)
    assert store.stats().put_failures == 1


def test_resize_evicts_cold_blocks_and_preserves_pinned_budget():
    store = PrefixCacheStore(_compat(), max_bytes=10, node_id="x")
    first = CacheBlock.create(bytes(32), 1, b"12345")
    second = CacheBlock.create(bytes.fromhex("01" * 32), 1, b"abc")
    store.put(first)
    store.put(second)
    assert store.resize(4)
    assert store.block_hashes() == (second.block_hash,)
    assert store.stats().max_bytes == 4
    store.lookup([second.block_hash])
    assert not store.resize(1)
    assert store.stats().max_bytes == 4
    try:
        store.resize(0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected resize validation")
