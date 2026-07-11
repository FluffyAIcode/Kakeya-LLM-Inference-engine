"""Immutable, content-addressed distributed prefill K/V cache.

The cache stores an opaque restorable snapshot at selected token-block
boundaries. Model-specific adapters own serialization/import; this module owns
deterministic prefix hashing, exact compatibility matching,
longest-contiguous-prefix lookup, leases, accounting, and memory-pressure
eviction. A hit transfers only the snapshot at the longest matched boundary.

Decode never reads this store. A requester imports a hit once, computes the
missing suffix locally, and keeps the autoregressive loop local.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

from inference_engine.distributed.capability import CacheCompatibility

DEFAULT_LEASE_SECONDS = 30.0


def compatibility_fingerprint(compatibility: CacheCompatibility) -> bytes:
    """Stable SHA-256 of every field that affects K/V interpretation."""
    payload = {
        "block_size_tokens": compatibility.block_size_tokens,
        "cache_format_version": compatibility.cache_format_version,
        "kv_dtype": compatibility.kv_dtype,
        "layer_geometry_hash": compatibility.layer_geometry_hash,
        "model_id": compatibility.model_id,
        "model_revision": compatibility.model_revision,
        "quantization": compatibility.quantization,
        "rope_hash": compatibility.rope_hash,
        "tokenizer_revision": compatibility.tokenizer_revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
    ).digest()


def chained_block_hashes(
    token_ids: Sequence[int],
    compatibility: CacheCompatibility,
) -> list[bytes]:
    """Hash fixed-size token blocks, chaining each hash to its predecessor.

    Chaining means block N is only reusable after blocks 0..N-1 matched, which
    enforces the causal longest-prefix rule and prevents arbitrary-hole reuse.
    """
    size = int(compatibility.block_size_tokens)
    if size <= 0:
        raise ValueError("block_size_tokens must be > 0")
    namespace = compatibility_fingerprint(compatibility)
    previous = bytes(32)
    hashes: list[bytes] = []
    for start in range(0, len(token_ids), size):
        block = token_ids[start:start + size]
        encoded = b"".join(int(t).to_bytes(4, "little", signed=False) for t in block)
        previous = hashlib.sha256(namespace + previous + encoded).digest()
        hashes.append(previous)
    return hashes


@dataclass(frozen=True)
class CacheBlock:
    block_hash: bytes
    token_count: int
    payload: bytes
    payload_sha256: bytes

    @classmethod
    def create(cls, block_hash: bytes, token_count: int, payload: bytes) -> "CacheBlock":
        if len(block_hash) != 32:
            raise ValueError("block_hash must be SHA-256 (32 bytes)")
        if token_count <= 0:
            raise ValueError("token_count must be > 0")
        data = bytes(payload)
        return cls(
            block_hash=bytes(block_hash),
            token_count=int(token_count),
            payload=data,
            payload_sha256=hashlib.sha256(data).digest(),
        )

    @property
    def nbytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class PrefixLease:
    lease_id: str
    block_hashes: tuple[bytes, ...]
    hit_block_count: int
    hit_token_count: int
    transfer_bytes: int
    cache_epoch: int
    expires_at_unix: float
    payload_sha256: bytes


@dataclass(frozen=True)
class CacheStats:
    bytes_used: int
    max_bytes: int
    entry_count: int
    cache_epoch: int
    lookup_hits: int
    lookup_misses: int
    tokens_served: int
    bytes_served: int


class PrefixCacheStore:
    """Thread-safe in-memory LRU of immutable K/V block payloads."""

    def __init__(
        self,
        compatibility: CacheCompatibility,
        *,
        max_bytes: int,
        node_id: str,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if not node_id:
            raise ValueError("node_id must be non-empty")
        self.compatibility = compatibility
        self.max_bytes = int(max_bytes)
        self.node_id = node_id
        self._blocks: OrderedDict[bytes, CacheBlock] = OrderedDict()
        self._leases: dict[str, PrefixLease] = {}
        self._bytes_used = 0
        self._epoch = 1
        self._lookup_hits = 0
        self._lookup_misses = 0
        self._tokens_served = 0
        self._bytes_served = 0
        self._lock = threading.RLock()

    def put(self, block: CacheBlock) -> bool:
        """Publish one immutable block. Returns False for an identical hit."""
        if block.nbytes > self.max_bytes:
            raise ValueError("block payload exceeds cache capacity")
        with self._lock:
            existing = self._blocks.get(block.block_hash)
            if existing is not None:
                if existing.payload_sha256 != block.payload_sha256:
                    raise ValueError("content-address collision with different payload")
                self._blocks.move_to_end(block.block_hash)
                return False
            self._blocks[block.block_hash] = block
            self._bytes_used += block.nbytes
            self._epoch += 1
            self._evict_to_budget()
            return True

    def put_prefix(
        self,
        token_ids: Sequence[int],
        payloads: Sequence[bytes],
    ) -> list[bytes]:
        hashes = chained_block_hashes(token_ids, self.compatibility)
        if len(hashes) != len(payloads):
            raise ValueError("one payload is required for every token block")
        size = self.compatibility.block_size_tokens
        for index, (block_hash, payload) in enumerate(zip(hashes, payloads)):
            prefix_count = min((index + 1) * size, len(token_ids))
            self.put(CacheBlock.create(block_hash, prefix_count, payload))
        return hashes

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        now: float | None = None,
    ) -> PrefixLease:
        """Lease the longest contiguous prefix held by this store."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        now = time.time() if now is None else now
        with self._lock:
            self._expire_leases(now)
            matched: list[CacheBlock] = []
            for raw_hash in block_hashes:
                block_hash = bytes(raw_hash)
                block = self._blocks.get(block_hash)
                if block is None:
                    break
                matched.append(block)
                self._blocks.move_to_end(block_hash)
            if not matched:
                self._lookup_misses += 1
                return PrefixLease("", (), 0, 0, 0, self._epoch, now, bytes(32))
            self._lookup_hits += 1
            lease_id = secrets.token_urlsafe(18)
            snapshot = matched[-1]
            lease = PrefixLease(
                lease_id=lease_id,
                block_hashes=(snapshot.block_hash,),
                hit_block_count=len(matched),
                hit_token_count=snapshot.token_count,
                transfer_bytes=snapshot.nbytes,
                cache_epoch=self._epoch,
                expires_at_unix=now + lease_seconds,
                payload_sha256=snapshot.payload_sha256,
            )
            self._leases[lease_id] = lease
            return lease

    def fetch(self, lease_id: str, *, now: float | None = None) -> tuple[CacheBlock, ...]:
        now = time.time() if now is None else now
        with self._lock:
            self._expire_leases(now)
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError("unknown or expired cache lease")
            blocks: list[CacheBlock] = []
            for block_hash in lease.block_hashes:
                block = self._blocks.get(block_hash)
                if block is None:
                    raise KeyError("leased block was evicted")
                blocks.append(block)
            self._tokens_served += lease.hit_token_count
            self._bytes_served += lease.transfer_bytes
            return tuple(blocks)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                bytes_used=self._bytes_used,
                max_bytes=self.max_bytes,
                entry_count=len(self._blocks),
                cache_epoch=self._epoch,
                lookup_hits=self._lookup_hits,
                lookup_misses=self._lookup_misses,
                tokens_served=self._tokens_served,
                bytes_served=self._bytes_served,
            )

    def block_hashes(self) -> tuple[bytes, ...]:
        with self._lock:
            return tuple(self._blocks)

    def _expire_leases(self, now: float) -> None:
        for lease_id, lease in list(self._leases.items()):
            if now > lease.expires_at_unix:
                del self._leases[lease_id]

    def _pinned_hashes(self) -> set[bytes]:
        return {
            block_hash
            for lease in self._leases.values()
            for block_hash in lease.block_hashes
        }

    def _evict_to_budget(self) -> None:
        pinned = self._pinned_hashes()
        while self._bytes_used > self.max_bytes and self._blocks:
            victim = next((h for h in self._blocks if h not in pinned), None)
            if victim is None:
                break
            block = self._blocks.pop(victim)
            self._bytes_used -= block.nbytes
            self._epoch += 1


def total_payload_bytes(blocks: Iterable[CacheBlock]) -> int:
    return sum(block.nbytes for block in blocks)
