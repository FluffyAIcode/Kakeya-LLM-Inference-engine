from __future__ import annotations

import asyncio

import grpc
import pytest

from inference_engine.distributed.capability import (
    CacheCapability,
    CacheCompatibility,
    NodeCapability,
    NodeEndpoint,
)
from inference_engine.distributed.prefill_cache import PrefixCacheStore
from inference_engine.distributed.prefill_cache import CacheBlock
from inference_engine.distributed.prefill_cache_service import (
    PrefillCacheServiceServicer,
    add_prefill_cache_service,
    compatible_cache_peers,
    fetch_remote_blocks,
    lookup_best_peer,
    publish_block_sync,
)
from inference_engine.server.proto_gen.kakeya.v1 import (  # noqa: E402
    distributed_pb2,
    distributed_pb2_grpc,
)


def _compat() -> CacheCompatibility:
    return CacheCompatibility(model_id="m", block_size_tokens=2)


@pytest.mark.asyncio
async def test_lookup_and_fetch_over_real_grpc():
    store = PrefixCacheStore(_compat(), max_bytes=1024, node_id="peer")
    hashes = store.put_prefix([1, 2, 3], [b"first", b"latest-snapshot"])
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    address = f"127.0.0.1:{port}"
    add_prefill_cache_service(server, store, cache_address=address, chunk_bytes=4)
    await server.start()
    try:
        hit = await lookup_best_peer([address], _compat(), hashes)
        assert hit is not None
        assert hit.node_id == "peer"
        assert hit.hit_block_count == 2
        assert hit.hit_token_count == 3
        chunks = await fetch_remote_blocks(hit)
        assert b"".join(chunk.data for chunk in chunks) == b"latest-snapshot"
        assert len(chunks) > 1
        async with grpc.aio.insecure_channel(address) as channel:
            stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
            summary = await stub.GetCacheSummary(
                distributed_pb2.GetCacheSummaryRequest(
                    compatibility=_compat().to_proto(),
                ),
            )
            assert summary.node_id == "peer"
            with pytest.raises(grpc.aio.AioRpcError) as error:
                async for _ in stub.FetchBlocks(
                    distributed_pb2.FetchBlocksRequest(lease_id="missing"),
                ):
                    pass
            assert error.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_incompatible_and_dead_peers_are_misses():
    store = PrefixCacheStore(_compat(), max_bytes=1024, node_id="peer")
    hashes = store.put_prefix([1, 2], [b"payload"])
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    address = f"127.0.0.1:{port}"
    add_prefill_cache_service(server, store, cache_address=address)
    await server.start()
    try:
        wrong = CacheCompatibility(model_id="other", block_size_tokens=2)
        assert await lookup_best_peer([], wrong, hashes) is None
        assert await lookup_best_peer([address], wrong, hashes) is None
        assert await lookup_best_peer(["127.0.0.1:1"], _compat(), hashes) is None
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_publish_block_replication():
    store = PrefixCacheStore(_compat(), max_bytes=1024, node_id="peer")
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    address = f"127.0.0.1:{port}"
    add_prefill_cache_service(server, store, cache_address=address)
    await server.start()
    try:
        block = CacheBlock.create(bytes(32), 2, b"snapshot")
        stored = await asyncio.to_thread(
            publish_block_sync,
            address,
            _compat(),
            block,
        )
        assert stored
        assert store.stats().entry_count == 1
        assert not await asyncio.to_thread(
            publish_block_sync,
            address,
            _compat(),
            block,
        )
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_publish_rejects_malformed_streams():
    store = PrefixCacheStore(_compat(), max_bytes=1024, node_id="peer")
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    address = f"127.0.0.1:{port}"
    add_prefill_cache_service(server, store, cache_address=address)
    await server.start()

    async def call(chunks):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
            return await stub.PublishBlock(iter(chunks))

    base = dict(
        block_hash=bytes(32),
        token_count=2,
        total_chunks=1,
        block_sha256=__import__("hashlib").sha256(b"x").digest(),
        compatibility=_compat().to_proto(),
    )
    try:
        with pytest.raises(grpc.aio.AioRpcError):
            await call([])
        with pytest.raises(grpc.aio.AioRpcError):
            await call([
                distributed_pb2.PublishBlockRequest(**base, chunk_index=0, data=b"x"),
                distributed_pb2.PublishBlockRequest(
                    **{**base, "block_hash": bytes.fromhex("01" * 32)},
                    chunk_index=1,
                    data=b"x",
                ),
            ])
        with pytest.raises(grpc.aio.AioRpcError):
            await call([
                distributed_pb2.PublishBlockRequest(
                    **{**base, "block_hash": b"short"},
                    chunk_index=0,
                    data=b"x",
                ),
            ])
        with pytest.raises(grpc.aio.AioRpcError):
            await call([
                distributed_pb2.PublishBlockRequest(
                    **{**base, "compatibility": CacheCompatibility(
                        model_id="wrong",
                    ).to_proto()},
                    chunk_index=0,
                    data=b"x",
                ),
            ])
        with pytest.raises(grpc.aio.AioRpcError):
            await call([
                distributed_pb2.PublishBlockRequest(
                    **{**base, "total_chunks": 2},
                    chunk_index=0,
                    data=b"x",
                ),
            ])
        with pytest.raises(grpc.aio.AioRpcError):
            await call([
                distributed_pb2.PublishBlockRequest(
                    **{**base, "block_sha256": bytes(32)},
                    chunk_index=0,
                    data=b"x",
                ),
            ])
    finally:
        await server.stop(0)


def test_compatible_peer_selection_prefers_cache_address_and_endpoint():
    compatibility = _compat()
    with_cache_address = NodeCapability(
        node_id="a",
        grpc_address="a:1",
        caches=(CacheCapability(compatibility, cache_address="a:2"),),
    )
    with_endpoint = NodeCapability(
        node_id="b",
        grpc_address="b:1",
        caches=(CacheCapability(compatibility),),
        endpoints=(
            NodeEndpoint("b-lan:2", "lan", 50, 2.0),
            NodeEndpoint("b-tb:2", "thunderbolt", 100, 0.4),
        ),
    )
    incompatible = NodeCapability(
        node_id="c",
        grpc_address="c:1",
        caches=(CacheCapability(CacheCompatibility(model_id="x")),),
    )
    fallback = NodeCapability(
        node_id="d",
        grpc_address="d:1",
        caches=(CacheCapability(compatibility),),
    )
    assert compatible_cache_peers(
        [with_cache_address, with_endpoint, incompatible, fallback],
        compatibility,
    ) == ["a:2", "b-tb:2", "d:1"]


def test_service_validation_and_dead_publish():
    store = PrefixCacheStore(_compat(), max_bytes=1024, node_id="peer")
    with pytest.raises(ValueError, match="chunk_bytes"):
        PrefillCacheServiceServicer(
            store,
            cache_address="peer:1",
            chunk_bytes=0,
        )
    block = CacheBlock.create(bytes(32), 1, b"x")
    assert not publish_block_sync("127.0.0.1:1", _compat(), block, timeout_s=0.1)
