"""Synchronous runtime hook that applies distributed prefill-cache hits.

The gRPC RuntimeService uses a synchronous verifier underneath its asyncio
handlers. This hook keeps that contract: peer lookups run concurrently in a
small thread pool, a winning snapshot is imported once, and missing suffix
blocks are prefetched locally before decode begins.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Sequence

import grpc

from inference_engine.backends.mlx.prefill_snapshot import (
    export_mlx_prefill_snapshot,
    import_mlx_prefill_snapshot,
)
from inference_engine.distributed.capability import CacheCompatibility
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    chained_block_hashes,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)


@dataclass
class PrefillReuseStats:
    local_hits: int = 0
    remote_hits: int = 0
    misses: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0
    bytes_received: int = 0


@dataclass(frozen=True)
class _Hit:
    source: str
    lease_id: str
    hit_blocks: int
    hit_tokens: int
    transfer_bytes: int
    payload: bytes | None = None


class DistributedPrefillCacheHook:
    """Prepare a cold verifier using local/remote longest-prefix snapshots."""

    def __init__(
        self,
        local_store: PrefixCacheStore,
        *,
        peers: Sequence[str] = (),
        lookup_timeout_s: float = 2.0,
        fetch_timeout_s: float = 30.0,
        on_reuse=None,
    ) -> None:
        self.local_store = local_store
        self.compatibility = local_store.compatibility
        self.peers = tuple(dict.fromkeys(peer for peer in peers if peer))
        self.lookup_timeout_s = float(lookup_timeout_s)
        self.fetch_timeout_s = float(fetch_timeout_s)
        self.stats = PrefillReuseStats()
        self._on_reuse = on_reuse
        self._publisher = ThreadPoolExecutor(
            max_workers=max(1, min(4, len(self.peers))),
            thread_name_prefix="prefill-kv-publish",
        )

    def prepare(self, verifier: Any, token_ids: Sequence[int]) -> int:
        """Restore the longest prefix, compute suffix, publish all new boundaries.

        Returns the number of tokens reused from cache.
        """
        tokens = [int(token) for token in token_ids]
        if not tokens:
            return 0
        hashes = chained_block_hashes(tokens, self.compatibility)
        hit = self._best_hit(hashes)
        reused = 0
        if hit is not None:
            payload = hit.payload if hit.payload is not None else self._fetch_remote(hit)
            verifier.reset()
            imported = import_mlx_prefill_snapshot(
                payload,
                verifier.cache,
                compatibility=self.compatibility,
            )
            reused = min(imported.token_count, len(tokens))
            verifier.cached_token_sequence = list(imported.cached_token_ids)
            verifier.next_global_position = reused
            if imported.next_token_logits is not None:
                verifier.next_token_logits = imported.next_token_logits
            self.stats.tokens_reused += reused
            if self._on_reuse is not None:
                self._on_reuse(reused)
            if hit.source == "local":
                self.stats.local_hits += 1
            else:
                self.stats.remote_hits += 1
        else:
            self.stats.misses += 1

        self._compute_and_publish(verifier, tokens, hashes, reused)
        return reused

    def _compute_and_publish(
        self,
        verifier: Any,
        tokens: list[int],
        hashes: list[bytes],
        reused: int,
    ) -> None:
        size = self.compatibility.block_size_tokens
        start_block = reused // size
        if reused == 0:
            first_end = min(size, len(tokens))
            verifier.prefill(tokens[:first_end])
            self.stats.tokens_computed += first_end
            self._publish_boundary(verifier, tokens, hashes, 0, first_end)
            start_block = 1
        for block_index in range(start_block, len(hashes)):
            start = block_index * size
            if start < reused:
                continue
            end = min(start + size, len(tokens))
            block_tokens = tokens[start:end]
            if not block_tokens:
                continue
            logits = verifier.forward_block(block_tokens)
            verifier.commit_or_truncate(
                forwarded=len(block_tokens),
                accepted=len(block_tokens),
            )
            verifier.next_token_logits = logits[-1].clone()
            self.stats.tokens_computed += len(block_tokens)
            self._publish_boundary(verifier, tokens, hashes, block_index, end)

    def _publish_boundary(
        self,
        verifier: Any,
        tokens: list[int],
        hashes: list[bytes],
        block_index: int,
        prefix_end: int,
    ) -> None:
        payload = export_mlx_prefill_snapshot(
            verifier.cache,
            token_count=prefix_end,
            cached_token_ids=verifier.cached_token_sequence,
            compatibility=self.compatibility,
            next_token_logits=verifier.next_token_logits,
        )
        block = CacheBlock.create(hashes[block_index], prefix_end, payload)
        self.local_store.put(block)
        if self.peers:
            from inference_engine.distributed.prefill_cache_service import (
                publish_block_sync,
            )
            for peer in self.peers:
                self._publisher.submit(
                    publish_block_sync,
                    peer,
                    self.compatibility,
                    block,
                    timeout_s=self.fetch_timeout_s,
                )

    def close(self) -> None:
        self._publisher.shutdown(wait=False, cancel_futures=True)

    def _best_hit(self, hashes: Sequence[bytes]) -> _Hit | None:
        candidates: list[_Hit] = []
        local = self.local_store.lookup(hashes)
        if local.lease_id:
            blocks = self.local_store.fetch(local.lease_id)
            candidates.append(_Hit(
                source="local",
                lease_id=local.lease_id,
                hit_blocks=local.hit_block_count,
                hit_tokens=local.hit_token_count,
                transfer_bytes=local.transfer_bytes,
                payload=blocks[-1].payload,
            ))
        if self.peers:
            with ThreadPoolExecutor(max_workers=min(8, len(self.peers))) as pool:
                futures = {
                    pool.submit(self._lookup_peer, peer, hashes): peer
                    for peer in self.peers
                }
                for future in as_completed(futures):
                    hit = future.result()
                    if hit is not None:
                        candidates.append(hit)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda hit: (
                hit.hit_tokens,
                hit.source == "local",
                -hit.transfer_bytes,
            ),
        )

    def _lookup_peer(self, peer: str, hashes: Sequence[bytes]) -> _Hit | None:
        try:
            with grpc.insecure_channel(peer) as channel:
                stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
                response = stub.LookupPrefix(
                    distributed_pb2.LookupPrefixRequest(
                        compatibility=self.compatibility.to_proto(),
                        block_hashes=hashes,
                    ),
                    timeout=self.lookup_timeout_s,
                )
        except grpc.RpcError:
            return None
        if not response.lease_id or response.hit_block_count == 0:
            return None
        return _Hit(
            source=peer,
            lease_id=response.lease_id,
            hit_blocks=response.hit_block_count,
            hit_tokens=response.hit_token_count,
            transfer_bytes=response.transfer_bytes,
        )

    def _fetch_remote(self, hit: _Hit) -> bytes:
        parts: dict[int, bytes] = {}
        expected_chunks = 0
        expected_sha = b""
        try:
            with grpc.insecure_channel(hit.source) as channel:
                stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
                for chunk in stub.FetchBlocks(
                    distributed_pb2.FetchBlocksRequest(lease_id=hit.lease_id),
                    timeout=self.fetch_timeout_s,
                ):
                    parts[chunk.chunk_index] = bytes(chunk.data)
                    expected_chunks = chunk.total_chunks
                    expected_sha = bytes(chunk.block_sha256)
        except grpc.RpcError as exc:
            raise RuntimeError(f"remote prefill cache fetch failed: {exc}") from exc
        if expected_chunks <= 0 or len(parts) != expected_chunks:
            raise RuntimeError("remote prefill cache stream was incomplete")
        payload = b"".join(parts[index] for index in range(expected_chunks))
        if hashlib.sha256(payload).digest() != expected_sha:
            raise RuntimeError("remote prefill cache checksum mismatch")
        self.stats.bytes_received += len(payload)
        return payload
