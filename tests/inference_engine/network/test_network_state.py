from __future__ import annotations

from inference_engine.distributed.capability import (
    CacheCapability,
    CacheCompatibility,
    CapabilityRegistry,
    NodeCapability,
    NodeEndpoint,
    PrefillWorkerCapability,
)
from inference_engine.distributed.prefill_cache import PrefixCacheStore
from inference_engine.distributed.prefill_cache_runtime import PrefillReuseStats
from inference_engine.network.state import NetworkState


def _state(tmp_path):
    compatibility = CacheCompatibility(model_id="gemma")
    store = PrefixCacheStore(compatibility, max_bytes=1000, node_id="head")
    card = NodeCapability(
        node_id="head",
        grpc_address="head:1",
        platform="mac-m4",
        unified_memory_bytes=24 << 30,
        caches=(
            CacheCapability(
                compatibility,
                cache_address="head:2",
                cache_bytes_free=1000,
                evictions=2,
                bytes_evicted=300,
                put_failures=1,
            ),
        ),
        endpoints=(NodeEndpoint("head:2", "thunderbolt", 100, 0.4),),
        prefill_workers=(PrefillWorkerCapability(
            compatibility,
            worker_address="head:3",
            queued_tokens=128,
            tokens_per_second_prefill=32,
        ),),
    )
    return NetworkState(
        CapabilityRegistry(self_card=card),
        store,
        state_path=tmp_path / "network.json",
        prefill_stats_provider=lambda: {
            "remote_jobs": 2,
            "remote_hits": 1,
            "tokens_reused": 128,
        },
    )


def test_registration_groups_tokens_and_persistence(tmp_path):
    state = _state(tmp_path)
    registration = state.register_node(
        alias="peer",
        address="peer:2",
        region="Hong Kong",
    )
    assert registration["pairing_token"].startswith("kn_pair_")
    group = state.create_group(name="Studio", node_ids=["head", "peer"])
    state.record_tokens(node_id="head", completed=100, kv_assisted=70)
    summary = state.summary()
    assert summary["online_nodes"] == 1
    assert summary["registered_nodes"] == 2
    assert summary["completed_tokens"] == 100
    assert summary["kv_hit_rate"] == 0.7
    assert summary["prefill"]["remote_jobs"] == 2
    assert summary["cache_evictions"] == 2
    assert summary["cache_bytes_evicted"] == 300
    assert summary["cache_put_failures"] == 1
    assert state.prefill_stats()["tokens_reused"] == 128
    assert state.groups()[0]["id"] == group["id"]
    assert state.topology()["edges"][0]["target"] == "peer"

    reloaded = _state(tmp_path)
    assert reloaded.summary()["completed_tokens"] == 100
    assert reloaded.groups()[0]["name"] == "Studio"


def test_state_validates_inputs(tmp_path):
    state = _state(tmp_path)
    for kwargs in (
        {"alias": "", "address": "x", "region": "r"},
        {"alias": "x", "address": "", "region": "r"},
    ):
        try:
            state.register_node(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected registration validation")
    try:
        state.create_group(name="", node_ids=[])
    except ValueError:
        pass
    else:
        raise AssertionError("expected group validation")
    try:
        state.record_tokens(node_id="x", completed=1, kv_assisted=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected token validation")


def test_invalid_persisted_state_falls_back_to_empty(tmp_path):
    path = tmp_path / "network.json"
    path.write_text("{not-json")
    compatibility = CacheCompatibility(model_id="m")
    state = NetworkState(
        CapabilityRegistry(NodeCapability(node_id="head", grpc_address="h:1")),
        PrefixCacheStore(compatibility, max_bytes=10, node_id="head"),
        state_path=path,
    )
    assert state.groups() == []
    assert state.summary()["completed_tokens"] == 0
    assert state.prefill_stats() == {}


def test_prefill_stats_serializes_runtime_dataclass(tmp_path):
    state = _state(tmp_path)
    state.prefill_stats_provider = lambda: PrefillReuseStats(
        remote_jobs=4,
        remote_hits=3,
        tokens_reused=256,
    )
    assert state.prefill_stats()["remote_jobs"] == 4
