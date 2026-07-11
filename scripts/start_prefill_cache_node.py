"""Start a standalone Kakeya P2P prefill-cache node and optional dashboard.

This process is useful for cache-only peers and control-plane deployment. A
full inference node should instead pass --enable-prefill-cache to
start_grpc_runtime_server.py so the same store is wired into cold prefill.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grpc
import uvicorn

from inference_engine.distributed.capability import (
    CacheCompatibility,
    CompressionCodec,
    CapabilityRegistry,
    CapabilityRole,
    ModelCapability,
    NodeCapability,
    NodeEndpoint,
)
from inference_engine.distributed.exchange import (
    add_capability_service,
    exchange_once,
)
from inference_engine.distributed.prefill_cache import PrefixCacheStore
from inference_engine.distributed.prefill_auth import FleetAuthConfig
from inference_engine.distributed.prefill_cache_service import (
    add_prefill_cache_service,
    cache_capability,
)
from inference_engine.network.api import create_network_app
from inference_engine.network.state import NetworkState


def physical_memory_bytes() -> int:
    try:
        return int(__import__("os").sysconf("SC_PAGE_SIZE")
                   * __import__("os").sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


async def serve(args) -> None:
    compatibility = CacheCompatibility(
        model_id=args.model_id,
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
    store = PrefixCacheStore(
        compatibility,
        max_bytes=int(args.cache_gb * (1 << 30)),
        node_id=args.node_id,
    )
    auth = (
        FleetAuthConfig.from_file(
            args.fleet_psk_file,
            tenant_id=args.tenant_id,
            node_id=args.node_id,
        )
        if args.fleet_psk_file else None
    )
    card = NodeCapability(
        node_id=args.node_id,
        grpc_address=args.advertise,
        platform=args.platform or f"{platform.system()}-{platform.machine()}",
        unified_memory_bytes=args.memory_bytes or physical_memory_bytes(),
        models=(
            ModelCapability(
                args.model_id,
                CapabilityRole.PREFILL_CACHE,
                args.quantization,
            ),
        ),
        announced_at_unix=time.time(),
        ttl_seconds=args.ttl_seconds,
        caches=(cache_capability(
            store,
            cache_address=args.advertise,
            default_compression=(
                CompressionCodec.ZLIB
                if args.cache_compression == "zlib"
                else CompressionCodec.NONE
            ),
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
    )
    registry = CapabilityRegistry(card)
    grpc_server = grpc.aio.server()
    add_capability_service(grpc_server, registry)
    add_prefill_cache_service(
        grpc_server,
        store,
        cache_address=args.advertise,
        auth=auth,
    )
    grpc_server.add_insecure_port(args.bind)
    await grpc_server.start()

    network_state = NetworkState(
        registry,
        store,
        state_path=args.state_path,
    )
    http_server = None
    http_task = None
    if args.http_port:
        http_server = uvicorn.Server(uvicorn.Config(
            create_network_app(network_state, api_key=args.api_key),
            host=args.http_host,
            port=args.http_port,
            log_level="info",
        ))
        http_task = asyncio.create_task(http_server.serve())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async def gossip():
        while not stop.is_set():
            registry.self_card = replace(
                registry.self_card,
                caches=(cache_capability(store, cache_address=args.advertise),),
            )
            await exchange_once(
                registry,
                args.peer,
                timeout_s=args.gossip_timeout_s,
            )
            try:
                await asyncio.wait_for(stop.wait(), args.gossip_interval_s)
            except asyncio.TimeoutError:
                pass

    gossip_task = asyncio.create_task(gossip())
    print(
        f"[prefill-cache] {args.node_id} grpc={args.bind} "
        f"advertise={args.advertise} peers={args.peer}",
        flush=True,
    )
    if args.http_port:
        print(
            f"[prefill-cache] dashboard=http://{args.http_host}:{args.http_port}/network",
            flush=True,
        )
    await stop.wait()
    gossip_task.cancel()
    if http_server is not None:
        http_server.should_exit = True
    if http_task is not None:
        await http_task
    await grpc_server.stop(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--bind", default="0.0.0.0:52051")
    ap.add_argument("--advertise", required=True)
    ap.add_argument("--peer", action="append", default=[])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--tokenizer-revision", default="")
    ap.add_argument("--cache-format-version", default="kakeya-prefill-v2-zlib")
    ap.add_argument("--quantization", default="")
    ap.add_argument("--rope-hash", default="")
    ap.add_argument("--layer-geometry-hash", default="")
    ap.add_argument("--kv-dtype", default="bfloat16")
    ap.add_argument("--block-size-tokens", type=int, default=64)
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--fleet-psk-file", default="")
    ap.add_argument("--cache-compression", choices=["none", "zlib"],
                    default="zlib")
    ap.add_argument("--replication-factor", type=int, default=1)
    ap.add_argument("--cache-gb", type=float, default=4)
    ap.add_argument("--platform", default="")
    ap.add_argument("--memory-bytes", type=int, default=0)
    ap.add_argument("--network", default="lan")
    ap.add_argument("--priority", type=int, default=50)
    ap.add_argument("--rtt-ms", type=float, default=0)
    ap.add_argument("--ttl-seconds", type=float, default=120)
    ap.add_argument("--gossip-interval-s", type=float, default=10)
    ap.add_argument("--gossip-timeout-s", type=float, default=3)
    ap.add_argument("--http-host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=0)
    ap.add_argument("--state-path", default="~/.kakeya/inference_network.json")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
