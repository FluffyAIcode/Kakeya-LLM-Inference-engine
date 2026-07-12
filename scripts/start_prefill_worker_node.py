"""Start a prefill-only MLX compute worker with a co-located RAM cache.

The worker loads the same model as the primary, accepts PrefillWorkerService
jobs, stores immutable snapshots, and never serves user decode.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import platform
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grpc
import torch

from inference_engine.backends.mlx.prefill_worker import MLXPrefillComputeEngine
from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
from inference_engine.distributed.capability import (
    CacheCompatibility,
    CapabilityRegistry,
    CapabilityRole,
    CompressionCodec,
    ModelCapability,
    NodeCapability,
    NodeEndpoint,
    PrefillWorkerCapability,
)
from inference_engine.distributed.exchange import (
    add_capability_service,
    exchange_once,
)
from inference_engine.distributed.prefill_auth import FleetAuthConfig
from inference_engine.distributed.prefill_cache import PrefixCacheStore
from inference_engine.distributed.prefill_cache_service import (
    add_prefill_cache_service,
    cache_capability,
)
from inference_engine.distributed.prefill_worker import (
    PrefillJobStore,
    add_prefill_worker_service,
)
from kv_cache_proposer.verifier import VerifierConfig

_LOG = logging.getLogger("kakeya.prefill-worker")


def compression_codec(name: str) -> CompressionCodec:
    return {
        "none": CompressionCodec.NONE,
        "zlib": CompressionCodec.ZLIB,
        "kakeyalattice-d4": CompressionCodec.KAKEYA_LATTICE_D4,
    }[name]


def physical_memory_bytes() -> int:
    try:
        import os
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


async def serve(args) -> None:
    compatibility = CacheCompatibility(
        model_id=args.cache_model_id or args.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        cache_format_version=args.cache_format_version,
        quantization=args.quantization,
        rope_hash=args.rope_hash,
        layer_geometry_hash=args.layer_geometry_hash,
        kv_dtype=args.kv_dtype,
        block_size_tokens=args.block_size_tokens,
        tenant_namespace=args.tenant_id,
        sink_size=args.sink,
        window_size=args.window,
    )
    auth = (
        FleetAuthConfig.from_file(
            args.fleet_psk_file,
            tenant_id=args.tenant_id,
            node_id=args.node_id,
        )
        if args.fleet_psk_file else None
    )
    if args.max_concurrent_jobs != 1:
        raise SystemExit(
            "MLX prefill workers require --max-concurrent-jobs 1 so the "
            "model and its stream remain on one compute thread",
        )
    store = PrefixCacheStore(
        compatibility,
        max_bytes=int(args.cache_gb * (1 << 30)),
        node_id=args.node_id,
    )

    def engine_factory() -> MLXPrefillComputeEngine:
        verifier = MLXSinkWindowVerifier(VerifierConfig(
            model_id=args.model_id,
            sink_size=args.sink,
            window_size=args.window,
            dtype=torch.bfloat16,
            device="cpu",
        ))
        return MLXPrefillComputeEngine(verifier, compatibility)

    jobs = PrefillJobStore(
        None,
        store,
        engine_factory=engine_factory,
        max_concurrent_jobs=args.max_concurrent_jobs,
        max_jobs=args.max_jobs,
        completed_ttl_s=args.job_ttl_s,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    jobs.warmup()

    def card() -> NodeCapability:
        inflight, queued, load, queued_tokens = jobs.stats()
        worker = PrefillWorkerCapability(
            compatibility=compatibility,
            worker_address=args.advertise,
            max_concurrent_jobs=args.max_concurrent_jobs,
            inflight_jobs=inflight,
            queued_jobs=queued,
            load=load,
            tokens_per_second_prefill=args.prefill_tps,
            ram_bytes_free=max(
                0,
                physical_memory_bytes() - store.stats().bytes_used,
            ),
            queued_tokens=queued_tokens,
        )
        return NodeCapability(
            node_id=args.node_id,
            grpc_address=args.advertise,
            platform=f"{platform.system()}-{platform.machine()}",
            unified_memory_bytes=physical_memory_bytes(),
            models=(
                ModelCapability(
                    args.cache_model_id or args.model_id,
                    CapabilityRole.PREFILL_COMPUTE,
                    args.quantization,
                    args.prefill_tps,
                ),
            ),
            announced_at_unix=time.time(),
            ttl_seconds=args.ttl_seconds,
            caches=(cache_capability(
                store,
                cache_address=args.advertise,
                load=load,
                default_compression=compression_codec(args.cache_compression),
                replication_factor=args.replication_factor,
            ),),
            endpoints=(
                NodeEndpoint(
                    args.advertise,
                    args.network,
                    args.priority,
                    args.rtt_ms,
                ),
            ),
            prefill_workers=(worker,),
        )

    registry = CapabilityRegistry(card())
    server = grpc.aio.server(
        maximum_concurrent_rpcs=args.max_concurrent_rpcs,
    )
    add_capability_service(server, registry)
    add_prefill_cache_service(
        server,
        store,
        cache_address=args.advertise,
        auth=auth,
    )
    add_prefill_worker_service(
        server,
        jobs,
        node_id=args.node_id,
        cache_address=args.advertise,
        auth=auth,
        tokens_per_second_prefill=args.prefill_tps,
    )
    server.add_insecure_port(args.bind)
    await server.start()
    _LOG.info("prefill worker ready on %s for %s", args.bind, args.model_id)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def gossip() -> None:
        while not stop.is_set():
            registry.self_card = replace(card(), announced_at_unix=time.time())
            if args.peer:
                await exchange_once(registry, args.peer, timeout_s=args.gossip_timeout_s)
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.gossip_interval_s)
            except asyncio.TimeoutError:
                pass

    gossip_task = asyncio.create_task(gossip())
    await stop.wait()
    gossip_task.cancel()
    jobs.close()
    await server.stop(grace=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--bind", default="127.0.0.1:53051")
    parser.add_argument("--advertise", default="127.0.0.1:53051")
    parser.add_argument("--peer", action="append", default=[])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--cache-format-version", default="kakeya-prefill-v2-zlib")
    parser.add_argument("--cache-model-id", default="",
                        help="Logical model id used for compatibility; defaults "
                             "to --model-id (which may be a host-specific path).")
    parser.add_argument("--quantization", default="4bit-mlx")
    parser.add_argument("--rope-hash", default="")
    parser.add_argument("--layer-geometry-hash", required=True)
    parser.add_argument("--kv-dtype", default="bfloat16")
    parser.add_argument("--block-size-tokens", type=int, default=64)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--fleet-psk-file")
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--cache-compression",
                        choices=["none", "zlib", "kakeyalattice-d4"],
                        default="zlib")
    parser.add_argument("--replication-factor", type=int, default=1)
    parser.add_argument("--max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=131072)
    parser.add_argument("--job-ttl-s", type=float, default=600.0)
    parser.add_argument("--prefill-tps", type=float, default=20.0)
    parser.add_argument("--max-concurrent-rpcs", type=int, default=32)
    parser.add_argument("--ttl-seconds", type=float, default=120.0)
    parser.add_argument("--gossip-interval-s", type=float, default=10.0)
    parser.add_argument("--gossip-timeout-s", type=float, default=3.0)
    parser.add_argument("--network", default="lan")
    parser.add_argument("--priority", type=int, default=50)
    parser.add_argument("--rtt-ms", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()

