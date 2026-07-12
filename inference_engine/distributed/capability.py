"""Capability cards + the converging gossip registry (ADR 0009 §4.1).

A *capability card* (:class:`NodeCapability`) advertises what one node
can do for the fleet and how to reach it. The :class:`CapabilityRegistry`
is each node's local, eventually-consistent view of every card it has
heard about.

Convergence model (design doc §2): the registry is a last-writer-wins
map keyed by ``node_id`` with ``announced_at_unix`` as the total order
per key. ``merge`` is commutative, associative, and idempotent, so the
order and repetition of gossip exchanges cannot diverge replicas. A
node's own card is authoritative locally and is never overwritten by
gossip.

Time handling: all freshness math takes an explicit ``now`` so tests
are deterministic; callers default to ``time.time()`` (wall clock, NOT
``time.monotonic`` — cards cross host boundaries, so the stamp must be
comparable across machines).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

from inference_engine.server.proto_gen.kakeya.v1 import distributed_pb2

# Reserved model_id for the model-free prompt-lookup proposer
# (inference_engine.distributed.ngram). Kept here so capability
# producers and consumers agree on the spelling without importing the
# proposer implementation.
NGRAM_MODEL_ID = "ngram"

DEFAULT_TTL_SECONDS = 120.0


class CapabilityRole(enum.IntEnum):
    """Python mirror of ``kakeya.v1.CapabilityRole``.

    Values are wire-identical to the proto enum so conversion is a
    plain int cast in both directions.
    """

    UNSPECIFIED = 0
    VERIFIER = 1
    PROPOSER = 2
    EMBEDDER = 3
    TOOL = 4
    PREFILL_CACHE = 5
    PREFILL_COMPUTE = 6


class CompressionCodec(enum.IntEnum):
    UNSPECIFIED = 0
    NONE = 1
    ZLIB = 2
    KAKEYA_LATTICE_D4 = 3


@dataclass(frozen=True)
class ModelCapability:
    """One (model, role) a node offers. See distributed.proto."""

    model_id: str
    role: CapabilityRole
    quantization: str = ""
    tokens_per_second: float = 0.0

    def to_proto(self) -> distributed_pb2.ModelCapability:
        return distributed_pb2.ModelCapability(
            model_id=self.model_id,
            role=int(self.role),
            quantization=self.quantization,
            tokens_per_second=self.tokens_per_second,
        )

    @classmethod
    def from_proto(cls, msg: distributed_pb2.ModelCapability) -> "ModelCapability":
        return cls(
            model_id=msg.model_id,
            role=CapabilityRole(msg.role),
            quantization=msg.quantization,
            tokens_per_second=msg.tokens_per_second,
        )


@dataclass(frozen=True)
class NodeEndpoint:
    """One interface-specific address advertised by a node."""

    address: str
    network: str = ""
    priority: int = 0
    measured_rtt_ms: float = 0.0

    def to_proto(self) -> distributed_pb2.NodeEndpoint:
        return distributed_pb2.NodeEndpoint(
            address=self.address,
            network=self.network,
            priority=self.priority,
            measured_rtt_ms=self.measured_rtt_ms,
        )

    @classmethod
    def from_proto(cls, msg: distributed_pb2.NodeEndpoint) -> "NodeEndpoint":
        return cls(
            address=msg.address,
            network=msg.network,
            priority=msg.priority,
            measured_rtt_ms=msg.measured_rtt_ms,
        )


@dataclass(frozen=True)
class CacheCompatibility:
    """Exact compatibility tuple for reusable prefill K/V blocks."""

    model_id: str
    model_revision: str = ""
    tokenizer_revision: str = ""
    cache_format_version: str = "kakeya-prefill-v2-zlib"
    quantization: str = ""
    rope_hash: str = ""
    layer_geometry_hash: str = ""
    kv_dtype: str = ""
    block_size_tokens: int = 64
    tenant_namespace: str = ""
    sink_size: int = 4
    window_size: int = 64

    def to_proto(self) -> distributed_pb2.CacheCompatibility:
        return distributed_pb2.CacheCompatibility(
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            cache_format_version=self.cache_format_version,
            quantization=self.quantization,
            rope_hash=self.rope_hash,
            layer_geometry_hash=self.layer_geometry_hash,
            kv_dtype=self.kv_dtype,
            block_size_tokens=self.block_size_tokens,
            tenant_namespace=self.tenant_namespace,
            sink_size=self.sink_size,
            window_size=self.window_size,
        )

    @classmethod
    def from_proto(
        cls, msg: distributed_pb2.CacheCompatibility,
    ) -> "CacheCompatibility":
        return cls(
            model_id=msg.model_id,
            model_revision=msg.model_revision,
            tokenizer_revision=msg.tokenizer_revision,
            cache_format_version=msg.cache_format_version,
            quantization=msg.quantization,
            rope_hash=msg.rope_hash,
            layer_geometry_hash=msg.layer_geometry_hash,
            kv_dtype=msg.kv_dtype,
            block_size_tokens=msg.block_size_tokens,
            tenant_namespace=msg.tenant_namespace,
            sink_size=msg.sink_size,
            window_size=msg.window_size,
        )


@dataclass(frozen=True)
class CacheCapability:
    """One compatible prefill-cache offering on a node."""

    compatibility: CacheCompatibility
    cache_address: str = ""
    cache_bytes_used: int = 0
    cache_bytes_free: int = 0
    entry_count: int = 0
    cache_epoch: int = 0
    load: float = 0.0
    tokens_served: int = 0
    bloom_filter: bytes = b""
    default_compression: CompressionCodec = CompressionCodec.NONE
    replication_factor: int = 1
    evictions: int = 0
    bytes_evicted: int = 0
    put_failures: int = 0

    def to_proto(self) -> distributed_pb2.CacheCapability:
        return distributed_pb2.CacheCapability(
            compatibility=self.compatibility.to_proto(),
            cache_address=self.cache_address,
            cache_bytes_used=self.cache_bytes_used,
            cache_bytes_free=self.cache_bytes_free,
            entry_count=self.entry_count,
            cache_epoch=self.cache_epoch,
            load=self.load,
            tokens_served=self.tokens_served,
            bloom_filter=self.bloom_filter,
            default_compression=int(self.default_compression),
            replication_factor=self.replication_factor,
            evictions=self.evictions,
            bytes_evicted=self.bytes_evicted,
            put_failures=self.put_failures,
        )

    @classmethod
    def from_proto(cls, msg: distributed_pb2.CacheCapability) -> "CacheCapability":
        return cls(
            compatibility=CacheCompatibility.from_proto(msg.compatibility),
            cache_address=msg.cache_address,
            cache_bytes_used=msg.cache_bytes_used,
            cache_bytes_free=msg.cache_bytes_free,
            entry_count=msg.entry_count,
            cache_epoch=msg.cache_epoch,
            load=msg.load,
            tokens_served=msg.tokens_served,
            bloom_filter=msg.bloom_filter,
            default_compression=CompressionCodec(msg.default_compression),
            replication_factor=msg.replication_factor,
            evictions=msg.evictions,
            bytes_evicted=msg.bytes_evicted,
            put_failures=msg.put_failures,
        )


@dataclass(frozen=True)
class PrefillWorkerCapability:
    """One prefill-only compute offering on a node."""

    compatibility: CacheCompatibility
    worker_address: str = ""
    max_concurrent_jobs: int = 1
    inflight_jobs: int = 0
    queued_jobs: int = 0
    load: float = 0.0
    tokens_per_second_prefill: float = 0.0
    ram_bytes_free: int = 0
    accepts_compute_jobs: bool = True
    queued_tokens: int = 0

    def to_proto(self) -> distributed_pb2.PrefillWorkerCapability:
        return distributed_pb2.PrefillWorkerCapability(
            compatibility=self.compatibility.to_proto(),
            worker_address=self.worker_address,
            max_concurrent_jobs=self.max_concurrent_jobs,
            inflight_jobs=self.inflight_jobs,
            queued_jobs=self.queued_jobs,
            load=self.load,
            tokens_per_second_prefill=self.tokens_per_second_prefill,
            ram_bytes_free=self.ram_bytes_free,
            accepts_compute_jobs=self.accepts_compute_jobs,
            queued_tokens=self.queued_tokens,
        )

    @classmethod
    def from_proto(
        cls, msg: distributed_pb2.PrefillWorkerCapability,
    ) -> "PrefillWorkerCapability":
        return cls(
            compatibility=CacheCompatibility.from_proto(msg.compatibility),
            worker_address=msg.worker_address,
            max_concurrent_jobs=msg.max_concurrent_jobs,
            inflight_jobs=msg.inflight_jobs,
            queued_jobs=msg.queued_jobs,
            load=msg.load,
            tokens_per_second_prefill=msg.tokens_per_second_prefill,
            ram_bytes_free=msg.ram_bytes_free,
            accepts_compute_jobs=msg.accepts_compute_jobs,
            queued_tokens=msg.queued_tokens,
        )


@dataclass(frozen=True)
class NodeCapability:
    """One node's capability card. See distributed.proto for field docs."""

    node_id: str
    grpc_address: str
    platform: str = ""
    unified_memory_bytes: int = 0
    mlx_version: str = ""
    models: Tuple[ModelCapability, ...] = ()
    announced_at_unix: float = 0.0
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    ring_address: str = ""
    caches: Tuple[CacheCapability, ...] = ()
    endpoints: Tuple[NodeEndpoint, ...] = ()
    prefill_workers: Tuple[PrefillWorkerCapability, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

    def is_expired(self, now: float) -> bool:
        return now > self.announced_at_unix + self.ttl_seconds

    def models_with_role(
        self, role: CapabilityRole, *, model_id: Optional[str] = None,
    ) -> List[ModelCapability]:
        """Models on this card matching ``role`` (and ``model_id`` when pinned)."""
        return [
            m
            for m in self.models
            if m.role == role and (model_id is None or m.model_id == model_id)
        ]

    def to_proto(self) -> distributed_pb2.NodeCapability:
        return distributed_pb2.NodeCapability(
            node_id=self.node_id,
            grpc_address=self.grpc_address,
            platform=self.platform,
            unified_memory_bytes=self.unified_memory_bytes,
            mlx_version=self.mlx_version,
            models=[m.to_proto() for m in self.models],
            announced_at_unix=self.announced_at_unix,
            ttl_seconds=self.ttl_seconds,
            ring_address=self.ring_address,
            caches=[c.to_proto() for c in self.caches],
            endpoints=[e.to_proto() for e in self.endpoints],
            prefill_workers=[w.to_proto() for w in self.prefill_workers],
        )

    @classmethod
    def from_proto(cls, msg: distributed_pb2.NodeCapability) -> "NodeCapability":
        return cls(
            node_id=msg.node_id,
            grpc_address=msg.grpc_address,
            platform=msg.platform,
            unified_memory_bytes=msg.unified_memory_bytes,
            mlx_version=msg.mlx_version,
            models=tuple(ModelCapability.from_proto(m) for m in msg.models),
            announced_at_unix=msg.announced_at_unix,
            ttl_seconds=msg.ttl_seconds,
            ring_address=msg.ring_address,
            caches=tuple(CacheCapability.from_proto(c) for c in msg.caches),
            endpoints=tuple(NodeEndpoint.from_proto(e) for e in msg.endpoints),
            prefill_workers=tuple(
                PrefillWorkerCapability.from_proto(w)
                for w in msg.prefill_workers
            ),
        )


@dataclass
class CapabilityRegistry:
    """Local, converging view of the fleet's capability cards.

    ``self_card`` is this node's own advertisement; it is refreshed
    (re-stamped) on every :meth:`snapshot` so peers always receive a
    fresh ``announced_at_unix`` for us, and it can never be replaced
    by a gossiped card claiming our ``node_id``.

    Not thread-safe by design — same single-asyncio-loop serialization
    argument as :class:`~inference_engine.session.store.SessionStore`
    (ADR 0008 §2.5): all mutation happens on the node's one gRPC event
    loop.
    """

    self_card: NodeCapability
    _peers: Dict[str, NodeCapability] = field(default_factory=dict)

    def merge(
        self, cards: Iterable[NodeCapability], *, now: Optional[float] = None,
    ) -> int:
        """Merge gossiped ``cards``; return how many entries changed.

        Per card: drop if expired, drop if it claims our own node_id,
        keep only if strictly fresher than what we already hold.
        """
        now = time.time() if now is None else now
        changed = 0
        for card in cards:
            if card.node_id == self.self_card.node_id:
                continue
            if card.is_expired(now):
                continue
            held = self._peers.get(card.node_id)
            if held is not None and held.announced_at_unix >= card.announced_at_unix:
                continue
            self._peers[card.node_id] = card
            changed += 1
        return changed

    def evict_expired(self, *, now: Optional[float] = None) -> List[NodeCapability]:
        """Drop and return expired peer cards."""
        now = time.time() if now is None else now
        expired = [c for c in self._peers.values() if c.is_expired(now)]
        for card in expired:
            del self._peers[card.node_id]
        return expired

    def snapshot(self, *, now: Optional[float] = None) -> List[NodeCapability]:
        """Live cards, own (freshly re-stamped) card first.

        The returned list is exactly what goes on the wire in an
        ExchangeCapabilities request or response.
        """
        now = time.time() if now is None else now
        self.self_card = replace(self.self_card, announced_at_unix=now)
        self.evict_expired(now=now)
        peers = sorted(self._peers.values(), key=lambda c: c.node_id)
        return [self.self_card, *peers]

    def get(self, node_id: str) -> Optional[NodeCapability]:
        if node_id == self.self_card.node_id:
            return self.self_card
        return self._peers.get(node_id)

    @property
    def peer_count(self) -> int:
        return len(self._peers)
