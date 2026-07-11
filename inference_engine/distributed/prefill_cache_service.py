"""gRPC service/client for distributed immutable prefill K/V blocks."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Sequence

import grpc

from inference_engine.distributed.capability import (
    CacheCapability,
    CacheCompatibility,
    NodeCapability,
)
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    PrefixLease,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)

DEFAULT_CHUNK_BYTES = 2 * 1024 * 1024


def cache_capability(
    store: PrefixCacheStore,
    *,
    cache_address: str,
    load: float = 0.0,
) -> CacheCapability:
    stats = store.stats()
    return CacheCapability(
        compatibility=store.compatibility,
        cache_address=cache_address,
        cache_bytes_used=stats.bytes_used,
        cache_bytes_free=max(0, stats.max_bytes - stats.bytes_used),
        entry_count=stats.entry_count,
        cache_epoch=stats.cache_epoch,
        load=load,
        tokens_served=stats.tokens_served,
    )


class PrefillCacheServiceServicer(
    distributed_pb2_grpc.PrefillCacheServiceServicer,
):
    def __init__(
        self,
        store: PrefixCacheStore,
        *,
        cache_address: str,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")
        self.store = store
        self.cache_address = cache_address
        self.chunk_bytes = int(chunk_bytes)

    async def GetCacheSummary(  # noqa: N802
        self,
        request: distributed_pb2.GetCacheSummaryRequest,
        context: grpc.aio.ServicerContext,
    ) -> distributed_pb2.GetCacheSummaryResponse:
        requested = CacheCompatibility.from_proto(request.compatibility)
        caches = []
        if requested == self.store.compatibility:
            caches.append(
                cache_capability(
                    self.store,
                    cache_address=self.cache_address,
                ).to_proto(),
            )
        return distributed_pb2.GetCacheSummaryResponse(
            node_id=self.store.node_id,
            caches=caches,
        )

    async def LookupPrefix(  # noqa: N802
        self,
        request: distributed_pb2.LookupPrefixRequest,
        context: grpc.aio.ServicerContext,
    ) -> distributed_pb2.LookupPrefixResponse:
        requested = CacheCompatibility.from_proto(request.compatibility)
        if requested != self.store.compatibility:
            return distributed_pb2.LookupPrefixResponse(
                node_id=self.store.node_id,
                cache_epoch=self.store.stats().cache_epoch,
            )
        lease = self.store.lookup(request.block_hashes)
        return _lease_to_proto(self.store.node_id, lease)

    async def FetchBlocks(  # noqa: N802
        self,
        request: distributed_pb2.FetchBlocksRequest,
        context: grpc.aio.ServicerContext,
    ):
        try:
            blocks = self.store.fetch(request.lease_id)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            return  # pragma: no cover - grpc abort raises
        epoch = self.store.stats().cache_epoch
        for block_index, block in enumerate(blocks):
            total_chunks = max(
                1, (len(block.payload) + self.chunk_bytes - 1) // self.chunk_bytes,
            )
            for chunk_index in range(total_chunks):
                start = chunk_index * self.chunk_bytes
                yield distributed_pb2.FetchBlocksResponse(
                    block_hash=block.block_hash,
                    block_index=block_index,
                    token_count=block.token_count,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    data=block.payload[start:start + self.chunk_bytes],
                    block_sha256=block.payload_sha256,
                    cache_epoch=epoch,
                )

    async def PublishBlock(  # noqa: N802
        self,
        request_iterator,
        context: grpc.aio.ServicerContext,
    ) -> distributed_pb2.PublishBlockResponse:
        parts: dict[int, bytes] = {}
        first = None
        async for chunk in request_iterator:
            if first is None:
                first = chunk
            elif (
                chunk.block_hash != first.block_hash
                or chunk.total_chunks != first.total_chunks
                or chunk.block_sha256 != first.block_sha256
            ):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "inconsistent publish chunk metadata",
                )
            parts[chunk.chunk_index] = bytes(chunk.data)
        if first is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "empty publish stream",
            )
        requested = CacheCompatibility.from_proto(first.compatibility)
        if requested != self.store.compatibility:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "prefill cache compatibility mismatch",
            )
        if len(parts) != first.total_chunks:
            await context.abort(
                grpc.StatusCode.DATA_LOSS,
                "incomplete publish stream",
            )
        payload = b"".join(parts[index] for index in range(first.total_chunks))
        if hashlib.sha256(payload).digest() != bytes(first.block_sha256):
            await context.abort(
                grpc.StatusCode.DATA_LOSS,
                "prefill cache payload checksum mismatch",
            )
        try:
            stored = self.store.put(CacheBlock.create(
                bytes(first.block_hash),
                first.token_count,
                payload,
            ))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return distributed_pb2.PublishBlockResponse(
            stored=stored,
            cache_epoch=self.store.stats().cache_epoch,
        )


def add_prefill_cache_service(
    server: grpc.aio.Server,
    store: PrefixCacheStore,
    *,
    cache_address: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> PrefillCacheServiceServicer:
    servicer = PrefillCacheServiceServicer(
        store,
        cache_address=cache_address,
        chunk_bytes=chunk_bytes,
    )
    distributed_pb2_grpc.add_PrefillCacheServiceServicer_to_server(
        servicer, server,
    )
    return servicer


@dataclass(frozen=True)
class RemotePrefixHit:
    address: str
    node_id: str
    lease_id: str
    hit_block_count: int
    hit_token_count: int
    transfer_bytes: int
    cache_epoch: int
    expires_at_unix: float
    payload_sha256: bytes

    @property
    def found(self) -> bool:
        return self.hit_block_count > 0 and bool(self.lease_id)


async def lookup_peer(
    address: str,
    compatibility: CacheCompatibility,
    block_hashes: Sequence[bytes],
    *,
    timeout_s: float = 3.0,
) -> RemotePrefixHit:
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
            response = await stub.LookupPrefix(
                distributed_pb2.LookupPrefixRequest(
                    compatibility=compatibility.to_proto(),
                    block_hashes=block_hashes,
                ),
                timeout=timeout_s,
            )
    except grpc.aio.AioRpcError:
        return RemotePrefixHit(address, "", "", 0, 0, 0, 0, 0.0, b"")
    return RemotePrefixHit(
        address=address,
        node_id=response.node_id,
        lease_id=response.lease_id,
        hit_block_count=response.hit_block_count,
        hit_token_count=response.hit_token_count,
        transfer_bytes=response.transfer_bytes,
        cache_epoch=response.cache_epoch,
        expires_at_unix=response.lease_expires_at_unix,
        payload_sha256=response.payload_sha256,
    )


async def lookup_best_peer(
    peers: Sequence[str],
    compatibility: CacheCompatibility,
    block_hashes: Sequence[bytes],
    *,
    timeout_s: float = 3.0,
) -> RemotePrefixHit | None:
    """Fan out concurrently and choose longest hit, then smallest transfer."""
    if not peers:
        return None
    hits = await asyncio.gather(*(
        lookup_peer(
            peer,
            compatibility,
            block_hashes,
            timeout_s=timeout_s,
        )
        for peer in peers
    ))
    candidates = [hit for hit in hits if hit.found]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda hit: (
            hit.hit_block_count,
            -hit.transfer_bytes,
            hit.expires_at_unix,
        ),
    )


async def fetch_remote_blocks(
    hit: RemotePrefixHit,
    *,
    timeout_s: float = 30.0,
) -> list[distributed_pb2.FetchBlocksResponse]:
    async with grpc.aio.insecure_channel(hit.address) as channel:
        stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
        stream = stub.FetchBlocks(
            distributed_pb2.FetchBlocksRequest(lease_id=hit.lease_id),
            timeout=timeout_s,
        )
        return [chunk async for chunk in stream]


def publish_block_sync(
    address: str,
    compatibility: CacheCompatibility,
    block: CacheBlock,
    *,
    timeout_s: float = 30.0,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> bool:
    """Publish one immutable snapshot to a peer (used by background workers)."""
    try:
        total_chunks = max(
            1, (block.nbytes + chunk_bytes - 1) // chunk_bytes,
        )

        def chunks():
            for chunk_index in range(total_chunks):
                start = chunk_index * chunk_bytes
                yield distributed_pb2.PublishBlockRequest(
                    block_hash=block.block_hash,
                    token_count=block.token_count,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    data=block.payload[start:start + chunk_bytes],
                    block_sha256=block.payload_sha256,
                    compatibility=compatibility.to_proto(),
                )

        with grpc.insecure_channel(address) as channel:
            stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
            response = stub.PublishBlock(
                chunks(),
                timeout=timeout_s,
            )
    except grpc.RpcError:
        return False
    return bool(response.stored)


def compatible_cache_peers(
    cards: Sequence[NodeCapability],
    compatibility: CacheCompatibility,
) -> list[str]:
    """Choose each compatible card's highest-priority reachable endpoint."""
    peers: list[str] = []
    for card in cards:
        matching = [
            cache
            for cache in card.caches
            if cache.compatibility == compatibility
        ]
        if not matching:
            continue
        cache_address = matching[0].cache_address
        if cache_address:
            peers.append(cache_address)
            continue
        endpoints = sorted(
            card.endpoints,
            key=lambda endpoint: (
                endpoint.priority,
                -endpoint.measured_rtt_ms,
            ),
            reverse=True,
        )
        peers.append(endpoints[0].address if endpoints else card.grpc_address)
    return peers


def _lease_to_proto(
    node_id: str,
    lease: PrefixLease,
) -> distributed_pb2.LookupPrefixResponse:
    return distributed_pb2.LookupPrefixResponse(
        node_id=node_id,
        hit_block_count=lease.hit_block_count,
        hit_token_count=lease.hit_token_count,
        transfer_bytes=lease.transfer_bytes,
        cache_epoch=lease.cache_epoch,
        lease_id=lease.lease_id,
        lease_expires_at_unix=lease.expires_at_unix,
        payload_sha256=lease.payload_sha256,
    )
