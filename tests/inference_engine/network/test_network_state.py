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


def test_benchmark_lifecycle_persistence_and_retention(tmp_path):
    state = _state(tmp_path)
    run = state.create_benchmark(
        kind="distributed_prefill_fleet_benchmark",
        config={"model_id": "gemma"},
        started_at=10,
    )
    assert state.live_benchmark()["id"] == run["id"]
    stage = {
        "name": "remote_compute",
        "hit_source": "remote_worker",
        "ok": True,
        "prefix_tokens": 100,
        "output_tokens": 10,
        "append_s": 5,
        "ttft_s": 5.1,
        "decode_s": 2,
        "e2e_s": 7,
        "delta": {},
    }
    completed = state.update_benchmark(
        run["id"],
        stages=[stage],
        status="completed",
        finished_at=20,
    )
    assert completed["summary"]["decode_tok_s_p50"] == 5
    assert state.live_benchmark() is None
    assert state.list_benchmarks(limit=1)[0]["id"] == run["id"]
    assert state.list_benchmarks(status="completed")[0]["status"] == "completed"
    assert _state(tmp_path).get_benchmark(run["id"])["finished_at"] == 20
    with __import__("pytest").raises(ValueError):
        state.list_benchmarks(limit=0)
    with __import__("pytest").raises(ValueError):
        state.update_benchmark(run["id"], status="invalid")
    with __import__("pytest").raises(KeyError):
        state.get_benchmark("missing")
    with __import__("pytest").raises(ValueError):
        state.create_benchmark(kind="x", config={"prompt": "private"})

    for index in range(205):
        state.create_benchmark(kind="retention", config={"index": index})
    assert len(state._data["benchmark_runs"]) == 200
