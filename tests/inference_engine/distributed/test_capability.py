"""Unit tests for inference_engine.distributed.capability (ADR 0009).

Pins the convergence-bearing semantics of the gossip registry:
last-writer-wins per node_id, own-card protection, TTL expiry, and
lossless proto round-trips. Pure-Python, weight-free — Linux CI gate.

Coverage target: 100% on ``inference_engine/distributed/capability.py``.
"""

from __future__ import annotations

import pytest

from inference_engine.distributed.capability import (
    DEFAULT_TTL_SECONDS,
    NGRAM_MODEL_ID,
    CapabilityRegistry,
    CapabilityRole,
    CacheCapability,
    CacheCompatibility,
    CompressionCodec,
    ModelCapability,
    NodeCapability,
    NodeEndpoint,
    PrefillWorkerCapability,
)

T0 = 1_000_000.0


def _card(
    node_id: str,
    *,
    announced_at: float = T0,
    ttl: float = 60.0,
    models: tuple = (),
    ring_address: str = "",
) -> NodeCapability:
    return NodeCapability(
        node_id=node_id,
        grpc_address=f"{node_id}.local:50051",
        platform="test",
        unified_memory_bytes=16 << 30,
        models=models,
        announced_at_unix=announced_at,
        ttl_seconds=ttl,
        ring_address=ring_address,
    )


# ---------------------------------------------------------------------------
# NodeCapability / ModelCapability
# ---------------------------------------------------------------------------


def test_node_capability_rejects_empty_node_id():
    with pytest.raises(ValueError, match="node_id"):
        _card("")


def test_node_capability_rejects_non_positive_ttl():
    with pytest.raises(ValueError, match="ttl_seconds"):
        _card("a", ttl=0.0)


def test_default_ttl_applies():
    card = NodeCapability(node_id="a", grpc_address="a:1")
    assert card.ttl_seconds == DEFAULT_TTL_SECONDS


def test_expiry_boundary_is_exclusive():
    card = _card("a", announced_at=T0, ttl=60.0)
    assert not card.is_expired(T0 + 60.0)
    assert card.is_expired(T0 + 60.0 + 1e-6)


def test_models_with_role_filters_role_and_model_id():
    verifier = ModelCapability("Qwen/Qwen3-0.6B", CapabilityRole.VERIFIER, "bf16")
    ngram = ModelCapability(NGRAM_MODEL_ID, CapabilityRole.PROPOSER, "none")
    card = _card("a", models=(verifier, ngram))
    assert card.models_with_role(CapabilityRole.VERIFIER) == [verifier]
    assert card.models_with_role(CapabilityRole.PROPOSER) == [ngram]
    assert card.models_with_role(
        CapabilityRole.PROPOSER, model_id="other",
    ) == []
    assert card.models_with_role(CapabilityRole.EMBEDDER) == []


def test_proto_round_trip_is_lossless():
    card = _card(
        "mini-attic",
        announced_at=T0 + 1.5,
        ttl=90.0,
        models=(
            ModelCapability("Qwen/Qwen3-1.7B", CapabilityRole.VERIFIER,
                            "4bit-mlx", tokens_per_second=42.5),
            ModelCapability(NGRAM_MODEL_ID, CapabilityRole.PROPOSER),
        ),
        ring_address="mini-attic:0",
    )
    assert NodeCapability.from_proto(card.to_proto()) == card


def test_model_capability_proto_round_trip():
    model = ModelCapability("m", CapabilityRole.TOOL, "none", 7.0)
    assert ModelCapability.from_proto(model.to_proto()) == model


def test_cache_capability_and_endpoints_proto_round_trip():
    compatibility = CacheCompatibility(
        model_id="gemma",
        model_revision="abc",
        tokenizer_revision="tok",
        cache_format_version="kv-v1",
        quantization="4bit",
        rope_hash="rope",
        layer_geometry_hash="geometry",
        kv_dtype="bfloat16",
        block_size_tokens=64,
        tenant_namespace="tenant",
    )
    card = NodeCapability(
        node_id="cache-peer",
        grpc_address="peer:50051",
        caches=(
            CacheCapability(
                compatibility,
                cache_address="169.254.27.104:52051",
                cache_bytes_used=10,
                cache_bytes_free=20,
                entry_count=3,
                cache_epoch=4,
                load=0.5,
                tokens_served=100,
                bloom_filter=b"filter",
                default_compression=CompressionCodec.ZLIB,
                replication_factor=2,
                evictions=5,
                bytes_evicted=6,
                put_failures=7,
            ),
        ),
        endpoints=(
            NodeEndpoint(
                "169.254.27.104:52051",
                "thunderbolt",
                100,
                0.45,
            ),
        ),
        prefill_workers=(
            PrefillWorkerCapability(
                compatibility,
                worker_address="169.254.27.104:53051",
                max_concurrent_jobs=1,
                inflight_jobs=1,
                queued_jobs=2,
                load=0.5,
                tokens_per_second_prefill=33.0,
                ram_bytes_free=1234,
                queued_tokens=456,
            ),
        ),
    )
    assert NodeCapability.from_proto(card.to_proto()) == card
    assert CapabilityRole.PREFILL_CACHE.value == 5


# ---------------------------------------------------------------------------
# CapabilityRegistry merge semantics
# ---------------------------------------------------------------------------


def test_merge_adds_fresh_peer_cards():
    reg = CapabilityRegistry(self_card=_card("self"))
    assert reg.merge([_card("b"), _card("c")], now=T0) == 2
    assert reg.peer_count == 2


def test_merge_is_last_writer_wins():
    reg = CapabilityRegistry(self_card=_card("self"))
    old = _card("b", announced_at=T0)
    new = _card("b", announced_at=T0 + 10)
    reg.merge([old], now=T0)
    assert reg.merge([new], now=T0 + 10) == 1
    assert reg.get("b").announced_at_unix == T0 + 10
    # Stale (and equal-timestamp) re-announcements are ignored.
    assert reg.merge([old], now=T0 + 10) == 0
    assert reg.merge([new], now=T0 + 10) == 0
    assert reg.get("b").announced_at_unix == T0 + 10


def test_merge_is_idempotent_and_commutative():
    a, b = _card("a", announced_at=T0), _card("b", announced_at=T0 + 1)
    reg1 = CapabilityRegistry(self_card=_card("self"))
    reg2 = CapabilityRegistry(self_card=_card("self"))
    reg1.merge([a], now=T0 + 1)
    reg1.merge([b], now=T0 + 1)
    reg2.merge([b], now=T0 + 1)
    reg2.merge([a], now=T0 + 1)
    reg2.merge([a, b], now=T0 + 1)  # repetition is a no-op
    assert reg1.get("a") == reg2.get("a")
    assert reg1.get("b") == reg2.get("b")
    assert reg1.peer_count == reg2.peer_count == 2


def test_merge_never_overwrites_own_card():
    reg = CapabilityRegistry(self_card=_card("self", announced_at=T0))
    impostor = _card("self", announced_at=T0 + 1000)
    assert reg.merge([impostor], now=T0) == 0
    assert reg.self_card.announced_at_unix == T0
    assert reg.peer_count == 0


def test_merge_drops_already_expired_cards():
    reg = CapabilityRegistry(self_card=_card("self"))
    stale = _card("b", announced_at=T0, ttl=10.0)
    assert reg.merge([stale], now=T0 + 11) == 0
    assert reg.peer_count == 0


# ---------------------------------------------------------------------------
# Snapshot / eviction / lookup
# ---------------------------------------------------------------------------


def test_snapshot_restamps_self_card_and_sorts_peers():
    reg = CapabilityRegistry(self_card=_card("self", announced_at=T0))
    reg.merge([_card("z"), _card("a")], now=T0)
    snap = reg.snapshot(now=T0 + 5)
    assert [c.node_id for c in snap] == ["self", "a", "z"]
    assert snap[0].announced_at_unix == T0 + 5


def test_snapshot_drops_expired_peers():
    reg = CapabilityRegistry(self_card=_card("self"))
    reg.merge([_card("b", announced_at=T0, ttl=10.0)], now=T0)
    assert [c.node_id for c in reg.snapshot(now=T0 + 11)] == ["self"]
    assert reg.peer_count == 0


def test_evict_expired_returns_dropped_cards():
    reg = CapabilityRegistry(self_card=_card("self"))
    reg.merge(
        [_card("b", announced_at=T0, ttl=10.0), _card("c", announced_at=T0)],
        now=T0,
    )
    dropped = reg.evict_expired(now=T0 + 11)
    assert [c.node_id for c in dropped] == ["b"]
    assert reg.peer_count == 1


def test_get_resolves_self_and_peers_and_missing():
    reg = CapabilityRegistry(self_card=_card("self"))
    reg.merge([_card("b")], now=T0)
    assert reg.get("self") is reg.self_card
    assert reg.get("b").node_id == "b"
    assert reg.get("nope") is None


def test_wall_clock_defaults_are_used_when_now_omitted():
    # No explicit `now`: the registry uses time.time(). A card stamped
    # "now" with a long TTL must survive; one from the distant past
    # must not.
    import time

    reg = CapabilityRegistry(self_card=_card("self", announced_at=time.time()))
    fresh = _card("b", announced_at=time.time(), ttl=3600.0)
    ancient = _card("c", announced_at=1.0, ttl=1.0)
    assert reg.merge([fresh, ancient]) == 1
    assert reg.evict_expired() == []
    assert [c.node_id for c in reg.snapshot()] == ["self", "b"]
