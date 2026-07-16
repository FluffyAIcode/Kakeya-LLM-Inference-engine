from __future__ import annotations

import asyncio
import threading

import grpc
import pytest

from inference_engine.distributed.capability import (
    CacheCapability,
    CacheCompatibility,
    CompressionCodec,
    NodeCapability,
    NodeEndpoint,
    PrefillWorkerCapability,
)
from inference_engine.distributed.prefill_auth import FleetAuthConfig
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    chained_block_hashes,
)
from inference_engine.distributed.prefill_cache_runtime import (
    DistributedPrefillCacheHook,
)
from inference_engine.distributed.prefill_cache_service import (
    add_prefill_cache_service,
)
from inference_engine.distributed.prefill_compression import compress_payload
from inference_engine.distributed.prefill_scheduler import PrefillCostConfig
from inference_engine.distributed.prefill_worker import (
    PrefillJobStore,
    add_prefill_worker_service,
)

pytestmark = pytest.mark.asyncio


class _Engine:
    def compute_prefill(self, token_ids, block_hashes, *, compression, cancelled):
        assert not cancelled.is_set()
        return [
            CacheBlock.create(
                block_hash,
                min((index + 1) * 2, len(token_ids)),
                compress_payload(b"snapshot-" + bytes([index]), compression),
            )
            for index, block_hash in enumerate(block_hashes)
        ]


class _Verifier:
    def __init__(self):
        self.cache = []
        self.cached_token_sequence = []
        self.next_global_position = 0
        self.next_token_logits = None
        self.prefill_calls = 0

    def reset(self):
        self.cache = []
        self.cached_token_sequence = []
        self.next_global_position = 0

    def prefill(self, tokens):
        self.prefill_calls += 1
        self.cached_token_sequence = list(tokens)
        self.next_global_position = len(tokens)


async def test_dynamic_worker_computes_remote_prefill_and_head_imports(
    monkeypatch,
):
    compatibility = CacheCompatibility(
        model_id="m",
        block_size_tokens=2,
        tenant_namespace="tenant",
    )
    auth = FleetAuthConfig(b"k" * 32, "tenant", "head")
    worker_auth = FleetAuthConfig(b"k" * 32, "tenant", "worker")
    worker_cache = PrefixCacheStore(
        compatibility, max_bytes=1 << 20, node_id="worker",
    )
    jobs = PrefillJobStore(_Engine(), worker_cache)
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    address = f"127.0.0.1:{port}"
    add_prefill_cache_service(
        server, worker_cache, cache_address=address, auth=worker_auth,
    )
    add_prefill_worker_service(
        server,
        jobs,
        node_id="worker",
        cache_address=address,
        auth=worker_auth,
    )
    await server.start()

    card = NodeCapability(
        node_id="worker",
        grpc_address=address,
        endpoints=(NodeEndpoint(address, "lan", 1, 0.1),),
        caches=(CacheCapability(compatibility, address),),
        prefill_workers=(PrefillWorkerCapability(
            compatibility,
            address,
            tokens_per_second_prefill=10_000,
            ram_bytes_free=1 << 30,
        ),),
    )
    final_hash = chained_block_hashes(
        [1, 2, 3, 4],
        compatibility,
        hmac_key=auth.tenant_hash_key(),
    )[-1]
    imported = type("Imported", (), {
        "token_count": 4,
        "cached_token_ids": (1, 2, 3, 4),
        "next_token_logits": "logits",
        "block_hash": final_hash,
    })()
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "import_mlx_prefill_snapshot",
        lambda payload, cache, compatibility: imported,
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "export_mlx_prefill_snapshot",
        lambda *args, **kwargs: b"unused",
    )
    head_store = PrefixCacheStore(
        compatibility, max_bytes=1 << 20, node_id="head",
    )
    hook = DistributedPrefillCacheHook(
        head_store,
        registry_provider=lambda: (card,),
        remote_compute_min_tokens=1,
        worker_poll_interval_s=0.001,
        compression=CompressionCodec.ZLIB,
        cost_config=PrefillCostConfig(
            local_prefill_tps=1,
            default_worker_tps=10_000,
            link_mbps=10_000,
            default_rtt_ms=0.1,
            minimum_savings_ratio=0,
            primary_compute_penalty_ms=1000,
        ),
        auth=auth,
    )
    verifier = _Verifier()
    try:
        reused = await asyncio.to_thread(hook.prepare, verifier, [1, 2, 3, 4])
        assert reused == 4
        assert verifier.prefill_calls == 0
        assert verifier.cached_token_sequence == [1, 2, 3, 4]
        assert verifier.next_token_logits == "logits"
        assert hook.stats.remote_jobs == 1
        assert hook.stats.remote_hits == 1
        assert hook.stats.hot_promotions == 1
        assert hook.stats.hot_promotion_bytes > 0
        assert len(head_store.block_hashes()) == 1
    finally:
        hook.close()
        jobs.close()
        await server.stop(0)

