from __future__ import annotations

import pytest

from inference_engine.distributed.capability import (
    CacheCapability,
    CacheCompatibility,
    NodeCapability,
    NodeEndpoint,
    PrefillWorkerCapability,
)
from inference_engine.distributed.prefill_scheduler import (
    PrefillCostConfig,
    choose_prefill_worker,
    compatible_prefill_workers,
    estimate_import_ms,
    remote_import_wins,
    select_cache_replicas,
)


COMPAT = CacheCompatibility(model_id="m", tenant_namespace="tenant")


def _card(node: str, *, tps=100.0, load=0.0, free=10, rtt=1.0):
    return NodeCapability(
        node_id=node,
        grpc_address=f"{node}:1",
        endpoints=(NodeEndpoint(f"{node}:1", "lan", 1, rtt),),
        prefill_workers=(PrefillWorkerCapability(
            COMPAT,
            f"{node}:2",
            tokens_per_second_prefill=tps,
            load=load,
            ram_bytes_free=free,
        ),),
        caches=(CacheCapability(
            COMPAT,
            f"{node}:3",
            cache_bytes_free=free,
        ),),
    )


def test_worker_discovery_filters_incompatible_and_disabled():
    disabled = PrefillWorkerCapability(
        COMPAT, "off:1", accepts_compute_jobs=False,
    )
    cards = [
        _card("ok"),
        NodeCapability("off", "off:1", prefill_workers=(disabled,)),
        NodeCapability("bad", "bad:1", prefill_workers=(
            PrefillWorkerCapability(
                CacheCompatibility(model_id="other"), "bad:2",
            ),
        )),
    ]
    assert [w.node_id for w in compatible_prefill_workers(cards, COMPAT)] == ["ok"]
    target = compatible_prefill_workers([
        _card("unknown-tps", tps=0),
    ], COMPAT)[0]
    assert target.queue_eta_ms == 0


def test_choose_fast_worker_and_reject_when_local_is_cheaper():
    cards = [_card("slow", tps=10), _card("fast", tps=1000)]
    remote = PrefillCostConfig(
        local_prefill_tps=10,
        default_worker_tps=10,
        link_mbps=10_000,
        default_rtt_ms=1,
        minimum_savings_ratio=0,
    )
    assert choose_prefill_worker(
        cards,
        COMPAT,
        prompt_tokens=1000,
        estimated_snapshot_bytes=1000,
        config=remote,
    ).node_id == "fast"
    local = PrefillCostConfig(
        local_prefill_tps=10_000,
        default_worker_tps=10,
        link_mbps=100,
        default_rtt_ms=10,
        minimum_savings_ratio=0.1,
    )
    assert choose_prefill_worker(
        cards,
        COMPAT,
        prompt_tokens=100,
        estimated_snapshot_bytes=10_000_000,
        config=local,
    ) is None
    assert choose_prefill_worker(
        [],
        COMPAT,
        prompt_tokens=100,
        estimated_snapshot_bytes=100,
        config=local,
    ) is None


def test_remote_import_cost_gate():
    config = PrefillCostConfig(
        local_prefill_tps=10,
        default_worker_tps=10,
        link_mbps=1000,
        default_rtt_ms=2,
        minimum_savings_ratio=0.1,
    )
    assert remote_import_wins(
        hit_tokens=1000,
        transfer_bytes=1000,
        rtt_ms=1,
        config=config,
    )
    assert not remote_import_wins(
        hit_tokens=1,
        transfer_bytes=100_000_000,
        rtt_ms=100,
        config=config,
    )
    assert estimate_import_ms(0, rtt_ms=2, config=config) == 2


def test_rendezvous_replication_is_deterministic_unique_and_bounded():
    cards = [
        _card("a", free=10),
        _card("b", free=20),
        _card("c", free=30),
        NodeCapability(
            "bad", "bad:1",
            caches=(CacheCapability(
                CacheCompatibility(model_id="other"),
                "",
            ),),
        ),
    ]
    first = select_cache_replicas(
        cards, COMPAT, block_hash=b"h" * 32, replication_factor=2,
    )
    second = select_cache_replicas(
        cards, COMPAT, block_hash=b"h" * 32, replication_factor=2,
    )
    assert first == second
    assert len(first) == 2 and len(set(first)) == 2
    assert select_cache_replicas(
        cards, COMPAT, block_hash=b"h" * 32, replication_factor=0,
    ) == []


def test_cost_config_validation():
    with pytest.raises(ValueError):
        PrefillCostConfig(local_prefill_tps=0)
    with pytest.raises(ValueError):
        PrefillCostConfig(minimum_savings_ratio=1)
    with pytest.raises(ValueError):
        PrefillCostConfig(primary_compute_penalty_ms=-1)

