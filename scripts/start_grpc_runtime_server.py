"""gRPC runtime server launcher (PR-E1b of ADR 0008 Phase E).

Boots a real Qwen3 verifier (CPU or MLX), wires it through a
:class:`SessionStore` + :class:`AppendTokensCoordinator` +
:class:`GenerationCoordinator`, and serves the v0.3 gRPC
``RuntimeService`` defined in ``proto/kakeya/v1/runtime.proto``.

Usage::

    PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
        --backend cpu \
        --verifier-id Qwen/Qwen3-0.6B \
        --bind 127.0.0.1:50051 \
        --capacity 4 --sink 4 --window 64

Multi-host capability plane (ADR 0009, v0.5-M1) — opt-in. Join a
fleet of Kakeya nodes that gossip capability cards and serve remote
draft blocks::

    PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
        --backend mlx --verifier-id mlx-community/Qwen3-1.7B-4bit \
        --node-id mini-attic --advertise 192.168.4.21:50051 \
        --peer 192.168.4.22:50051 --serve-ngram-proposer

This script is the symmetric counterpart of ``scripts/serve.py``
(which boots the deprecated HTTP+SSE shim) and is exempt from unit-
test coverage by the same convention used for ``serve.py`` /
``run_demo.py`` / ``chat.py``: CLI plumbing around already-tested
library code, validated by integration runs and the Mac M4 review
aid (``scripts/review_pr_e1b_on_mac.sh``).
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import logging
import signal
import sys
from typing import Tuple

import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.server.grpc_app import (
    DEFAULT_BIND_ADDRESS,
    GrpcServerConfig,
    create_grpc_server,
)
from inference_engine.session.coordinator import AppendTokensCoordinator
from inference_engine.session.generator import GenerationCoordinator
from inference_engine.session.store import SessionStore
from kv_cache_proposer.verifier import VerifierConfig

_LOG = logging.getLogger("kakeya.grpc-runtime")


def _compression_codec(name: str):
    from inference_engine.distributed.capability import CompressionCodec
    return {
        "none": CompressionCodec.NONE,
        "zlib": CompressionCodec.ZLIB,
        "kakeyalattice-d4": CompressionCodec.KAKEYA_LATTICE_D4,
    }[name]


def _resolve_kv_dims(verifier) -> Tuple[int, int, int]:
    """Derive (num_layers, num_kv_heads, head_dim) from a loaded
    HF / MLX verifier.

    Used purely for slab byte accounting; the verifier maintains its
    own KV cache internally — the slab is a fixed-capacity allocation
    handle that backs ``GetSessionInfo.kv_live_bytes`` and the
    runtime's pool-pressure invariants. Reading the dims from the
    verifier's HF config means the per-session byte numbers reported
    over gRPC match what the verifier is actually holding.
    """
    try:
        cfg = (
            getattr(verifier.model, "config", None)
            or getattr(verifier.model, "args", None)
        )
        if cfg is None:
            raise AttributeError("model exposes neither config nor args")
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        cfg = getattr(cfg, "text_config", None) or cfg
        num_layers = int(getattr(cfg, "num_hidden_layers"))
        num_kv_heads = int(
            getattr(cfg, "num_key_value_heads", None)
            or getattr(cfg, "num_attention_heads")
        )
        head_dim = int(
            getattr(cfg, "head_dim", None)
            or (cfg.hidden_size // cfg.num_attention_heads)
        )
        return num_layers, num_kv_heads, head_dim
    except AttributeError:
        from inference_engine.backends.mlx.cross_model_dlm_verifier import (
            per_layer_kv_geometry,
            resolve_mlx_text_model,
        )
        geometry = per_layer_kv_geometry(resolve_mlx_text_model(verifier.model))
        if not geometry:
            raise
        return (
            len(geometry),
            max(item[0] for item in geometry),
            max(item[1] for item in geometry),
        )


def _build_verifier(
    *,
    backend: str,
    verifier_id: str,
    sink: int,
    window: int,
    drafter_id: str = "",
    f_theta_dir: str = "",
    s5_exact_full_attn: bool = True,
    device: str = "cpu",
):
    cfg = VerifierConfig(
        model_id=verifier_id,
        dtype=torch.bfloat16, device="cpu",
        sink_size=sink, window_size=window,
    )
    if backend == "cpu":
        from kv_cache_proposer.verifier import SinkWindowVerifier
        return SinkWindowVerifier(cfg)
    if backend == "mlx":
        from inference_engine.backends.mlx.env import probe_environment
        env = probe_environment()
        if not env.is_available:
            print(
                f"[grpc-server] MLX unavailable: {env.failure_reason}",
                file=sys.stderr,
            )
            sys.exit(2)
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        return MLXSinkWindowVerifier(cfg)
    if backend == "restored":
        # f_θ + S5 K/V-Restoration verifier (the Kakeya inference path).
        # Requires the DFlash drafter + trained f_θ checkpoint.
        if not drafter_id or not f_theta_dir:
            raise SystemExit(
                "backend=restored requires --drafter-id and --f-theta-dir"
            )
        from inference_engine.v04.build_restored import load_restored_verifier
        return load_restored_verifier(
            verifier_id=verifier_id,
            drafter_id=drafter_id,
            f_theta_dir=f_theta_dir,
            sink_size=sink,
            window_size=window,
            s5_exact_full_attn=s5_exact_full_attn,
            device=device,
        )
    raise SystemExit(f"unknown backend: {backend}")


def _total_memory_bytes() -> int:
    """Best-effort physical memory size; 0 when undeterminable."""
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _build_capability_registry(
    args: argparse.Namespace,
    *,
    backend: str,
    cache_store=None,
):
    """Build this node's CapabilityRegistry from CLI flags + probes."""
    import platform as _platform
    import time

    from inference_engine.backends.mlx.env import probe_environment
    from inference_engine.distributed.capability import (
        NGRAM_MODEL_ID,
        CapabilityRegistry,
        CapabilityRole,
        ModelCapability,
        NodeCapability,
        NodeEndpoint,
    )
    from inference_engine.distributed.mlx_ring import probe_ring_environment

    models = [
        ModelCapability(
            model_id=args.verifier_id,
            role=CapabilityRole.VERIFIER,
            quantization="4bit-mlx" if "4bit" in args.verifier_id else "bf16",
        ),
    ]
    if args.serve_ngram_proposer:
        models.append(
            ModelCapability(
                model_id=NGRAM_MODEL_ID,
                role=CapabilityRole.PROPOSER,
                quantization="none",
            ),
        )
    caches = ()
    if cache_store is not None:
        from inference_engine.distributed.prefill_cache_service import (
            cache_capability,
        )
        models.append(
            ModelCapability(
                model_id=args.verifier_id,
                role=CapabilityRole.PREFILL_CACHE,
                quantization=args.cache_quantization,
            ),
        )
        caches = (
            cache_capability(
                cache_store,
                cache_address=args.cache_advertise or args.advertise or args.bind,
                default_compression=_compression_codec(args.cache_compression),
                replication_factor=args.cache_replication_factor,
            ),
        )

    mlx_env = probe_environment()
    ring_env = probe_ring_environment()
    hostname = _platform.node() or "localhost"
    self_card = NodeCapability(
        node_id=args.node_id or hostname,
        grpc_address=args.advertise or args.bind,
        platform=f"{_platform.system()}-{_platform.machine()}-{backend}".lower(),
        unified_memory_bytes=_total_memory_bytes(),
        mlx_version=mlx_env.mlx_version or "",
        models=tuple(models),
        announced_at_unix=time.time(),
        ttl_seconds=args.capability_ttl_s,
        ring_address=ring_env.ring_address(hostname),
        caches=caches,
        endpoints=(
            NodeEndpoint(
                address=args.advertise or args.bind,
                network=args.network_label,
                priority=args.network_priority,
                measured_rtt_ms=args.measured_rtt_ms,
            ),
        ),
    )
    _LOG.info("capability card: %s @ %s ring=%r models=%s",
              self_card.node_id, self_card.grpc_address,
              self_card.ring_address,
              [(m.model_id, m.role.name) for m in models])
    return CapabilityRegistry(self_card=self_card)


async def _exchange_loop(
    registry,
    peers,
    interval_s: float,
    *,
    cache_store=None,
    cache_address: str = "",
    cache_compression: int = 1,
    cache_replication_factor: int = 1,
) -> None:
    """Periodic gossip with seed peers until the task is cancelled."""
    from inference_engine.distributed.exchange import exchange_once

    while True:
        if cache_store is not None:
            from dataclasses import replace
            from inference_engine.distributed.prefill_cache_service import (
                cache_capability,
            )
            registry.self_card = replace(
                registry.self_card,
                caches=(
                    cache_capability(
                        cache_store,
                        cache_address=cache_address,
                        default_compression=cache_compression,
                        replication_factor=cache_replication_factor,
                    ),
                ),
            )
        report = await exchange_once(registry, peers)
        if report.errors:
            _LOG.warning("gossip errors: %s", report.errors)
        if report.merged_cards:
            _LOG.info(
                "gossip merged %d card(s); fleet size now %d",
                report.merged_cards, registry.peer_count + 1,
            )
        await asyncio.sleep(interval_s)


def _build_token_telemetry_callback(args):
    if not args.network_telemetry_url:
        return None

    def report(count: int, kv_assisted: int = 0) -> None:
        import json
        import threading
        import urllib.request

        def send() -> None:
            body = json.dumps({
                "node_id": args.node_id or "runtime",
                "completed": int(count),
                "kv_assisted": int(kv_assisted),
            }).encode()
            headers = {"Content-Type": "application/json"}
            if args.network_telemetry_api_key:
                headers["X-API-Key"] = args.network_telemetry_api_key
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        args.network_telemetry_url,
                        data=body,
                        headers=headers,
                    ),
                    timeout=3,
                ):
                    pass
            except OSError:
                _LOG.warning("failed to publish token telemetry")

        threading.Thread(target=send, daemon=True).start()

    return report


async def _serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Pre-flight: fail fast on a cold HF cache instead of silently
    # blocking server boot on a multi-GB download with no progress
    # feedback. Skip when --skip-cache-check is set (useful in CI
    # workflows that intentionally trigger first-run download in a
    # controlled context).
    if not args.skip_cache_check:
        from inference_engine.setup import assert_cached_or_raise
        try:
            assert_cached_or_raise(args.verifier_id)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    _LOG.info(
        "loading verifier backend=%s id=%s sink=%d window=%d",
        args.backend, args.verifier_id, args.sink, args.window,
    )
    verifier = _build_verifier(
        backend=args.backend, verifier_id=args.verifier_id,
        sink=args.sink, window=args.window,
        drafter_id=args.drafter_id, f_theta_dir=args.f_theta_dir,
        s5_exact_full_attn=not args.no_s5_exact_full_attn,
        device=args.device,
    )

    num_layers, num_kv_heads, head_dim = _resolve_kv_dims(verifier)
    _LOG.info(
        "verifier dims: layers=%d kv_heads=%d head_dim=%d capacity=%d",
        num_layers, num_kv_heads, head_dim, args.sink + args.window,
    )

    slab_cfg = SlabConfig(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        sink_size=args.sink,
        window_size=args.window,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    pool = SlabPool(num_slabs=args.capacity, slab_config=slab_cfg)

    prefill_store = None
    prefill_hook = None
    prefill_auth = None
    capability_registry_holder = [None]
    if args.enable_prefill_cache:
        if args.backend != "mlx":
            raise SystemExit("--enable-prefill-cache currently requires --backend mlx")
        import hashlib
        from inference_engine.distributed.capability import (
            CacheCompatibility,
        )
        from inference_engine.distributed.prefill_auth import FleetAuthConfig
        from inference_engine.distributed.prefill_cache import PrefixCacheStore
        from inference_engine.distributed.prefill_cache_runtime import (
            DistributedPrefillCacheHook,
        )
        from inference_engine.distributed.prefill_scheduler import (
            PrefillCostConfig,
        )

        geometry = f"{num_layers}:{num_kv_heads}:{head_dim}"
        compatibility = CacheCompatibility(
            model_id=args.cache_model_id or args.verifier_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            cache_format_version=args.cache_format_version,
            quantization=args.cache_quantization,
            rope_hash=args.rope_hash,
            layer_geometry_hash=hashlib.sha256(geometry.encode()).hexdigest(),
            kv_dtype=args.cache_kv_dtype,
            block_size_tokens=args.cache_block_tokens,
            tenant_namespace=args.cache_tenant_id,
            sink_size=args.sink,
            window_size=args.window,
        )
        prefill_store = PrefixCacheStore(
            compatibility,
            max_bytes=int(args.prefill_cache_gb * (1 << 30)),
            node_id=args.node_id or (__import__("platform").node() or "localhost"),
        )
        telemetry_callback = _build_token_telemetry_callback(args)
        if args.fleet_psk_file:
            prefill_auth = FleetAuthConfig.from_file(
                args.fleet_psk_file,
                tenant_id=args.cache_tenant_id,
                node_id=args.node_id or "primary",
            )
        prefill_hook = DistributedPrefillCacheHook(
            prefill_store,
            peers=args.cache_peer,
            registry_provider=lambda: (
                capability_registry_holder[0].snapshot()
                if capability_registry_holder[0] is not None else ()
            ),
            lookup_timeout_s=args.cache_lookup_timeout_s,
            fetch_timeout_s=args.cache_fetch_timeout_s,
            worker_timeout_s=args.prefill_worker_timeout_s,
            remote_compute_min_tokens=args.remote_prefill_min_tokens,
            max_import_bytes=int(args.cache_max_import_gb * (1 << 30)),
            estimated_snapshot_bytes_per_token=args.cache_estimated_bytes_per_token,
            compression=_compression_codec(args.cache_compression),
            replication_factor=args.cache_replication_factor,
            cost_config=PrefillCostConfig(
                local_prefill_tps=args.local_prefill_tps,
                default_worker_tps=args.worker_prefill_tps,
                link_mbps=args.cache_link_mbps,
                default_rtt_ms=args.cache_default_rtt_ms,
                minimum_savings_ratio=args.prefill_min_savings_ratio,
                primary_compute_penalty_ms=args.primary_prefill_penalty_ms,
            ),
            auth=prefill_auth,
            require_remote_compute=(args.prefill_policy == "remote-required"),
            on_reuse=(
                (lambda count: telemetry_callback(count, count))
                if telemetry_callback is not None else None
            ),
        )
        _LOG.info(
            "distributed prefill cache enabled: %.2f GiB, block=%d, peers=%s",
            args.prefill_cache_gb,
            args.cache_block_tokens,
            args.cache_peer,
        )

    # PR-A3c: per-session binding for true multi-tenant serving. Each session
    # gets its own verifier adapter (isolated KV) sharing the model weights via
    # the adapter's spawn(); the registry doubles as cache_inspector + resolver.
    registry = None
    on_session_close = None
    if args.multi_tenant:
        if not hasattr(verifier, "spawn"):
            raise SystemExit(
                f"--multi-tenant requires a verifier that supports per-session "
                f"spawn() (backend=restored); backend={args.backend} does not.")
        from inference_engine.session.verifier_registry import (
            PerSessionVerifierRegistry,
        )
        registry = PerSessionVerifierRegistry(factory=verifier.spawn)
        on_session_close = registry.remove
        _LOG.info("multi-tenant per-session binding ENABLED (PR-A3c)")

    store = SessionStore(
        capacity=args.capacity,
        cache_inspector=registry if registry is not None else verifier,
        slab_pool=pool,
    )
    resolver = registry.get if registry is not None else None
    cache_fill_capture = None
    if args.cache_fill_capture_size:
        from inference_engine.distributed.cache_fill import CacheFillCapture
        cache_fill_capture = CacheFillCapture(
            max_items=args.cache_fill_capture_size,
        )
        _LOG.info(
            "maintenance cache-fill capture enabled: max_items=%d",
            args.cache_fill_capture_size,
        )
    append_coord = AppendTokensCoordinator(
        store,
        verifier,
        resolver=resolver,
        prefill_cache=prefill_hook,
        on_first_append=(
            (
                lambda session, tokens: cache_fill_capture.observe(
                    client_label=session.client_label,
                    token_ids=tokens,
                )
            )
            if cache_fill_capture is not None else None
        ),
    )
    gen_coord = GenerationCoordinator(
        store,
        verifier,
        resolver=resolver,
        on_tokens=_build_token_telemetry_callback(args),
    )

    # Multi-host capability plane (ADR 0009): only constructed when
    # the operator opts in via --enable-capability-exchange (implied
    # by --peer / --serve-ngram-proposer).
    distributed_enabled = bool(
        args.enable_capability_exchange
        or args.peer
        or args.serve_ngram_proposer
        or args.enable_prefill_cache
    )
    registry = None
    proposers = None
    if distributed_enabled:
        registry = _build_capability_registry(
            args,
            backend=args.backend,
            cache_store=prefill_store,
        )
        capability_registry_holder[0] = registry
        if args.serve_ngram_proposer:
            from inference_engine.distributed.capability import NGRAM_MODEL_ID
            from inference_engine.distributed.ngram import NGramProposer
            proposers = {NGRAM_MODEL_ID: NGramProposer()}

    config = GrpcServerConfig(
        bind_address=args.bind,
        max_concurrent_rpcs=args.max_concurrent_rpcs,
    )
    server = create_grpc_server(
        session_store=store,
        append_coordinator=append_coord,
        generation_coordinator=gen_coord,
        config=config,
        on_session_close=on_session_close,
        capability_registry=registry,
        proposers=proposers,
        prefill_cache_store=prefill_store,
        prefill_cache_address=args.cache_advertise or args.advertise or args.bind,
        prefill_auth=prefill_auth,
    )

    await server.start()
    _LOG.info("kakeya gRPC RuntimeService listening on %s", args.bind)

    http_server = None
    http_thread = None
    if args.network_http_port:
        if registry is None or prefill_store is None:
            raise SystemExit(
                "--network-http-port requires --enable-prefill-cache",
            )
        import uvicorn
        from inference_engine.network.api import create_network_app
        from inference_engine.network.state import NetworkState

        network_state = NetworkState(
            registry,
            prefill_store,
            state_path=args.network_state,
            prefill_stats_provider=(
                (lambda: prefill_hook.stats)
                if prefill_hook is not None else None
            ),
        )
        http_server = uvicorn.Server(uvicorn.Config(
            create_network_app(
                network_state,
                api_key=args.network_api_key,
                cache_fill_capture=cache_fill_capture,
            ),
            host=args.network_http_host,
            port=args.network_http_port,
            log_level=args.log_level.lower(),
        ))
        http_thread = threading.Thread(
            target=http_server.run,
            name="kakeya-network-http",
            daemon=True,
        )
        http_thread.start()
        _LOG.info(
            "inference-network dashboard listening on http://%s:%d/network",
            args.network_http_host,
            args.network_http_port,
        )

    exchange_task = None
    if registry is not None and args.peer:
        exchange_task = asyncio.create_task(
            _exchange_loop(
                registry,
                list(args.peer),
                args.exchange_interval_s,
                cache_store=prefill_store,
                cache_address=args.cache_advertise or args.advertise or args.bind,
                cache_compression=int(_compression_codec(args.cache_compression)),
                cache_replication_factor=args.cache_replication_factor,
            ),
        )
        _LOG.info(
            "capability gossip every %.1fs with peers: %s",
            args.exchange_interval_s, args.peer,
        )

    stop_event = asyncio.Event()

    def _on_signal(sig: int) -> None:
        _LOG.info("received signal %d; initiating graceful shutdown", sig)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal, int(sig))
        except NotImplementedError:  # pragma: no cover - Windows
            pass

    await stop_event.wait()
    if exchange_task is not None:
        exchange_task.cancel()
        try:
            await exchange_task
        except asyncio.CancelledError:
            pass
    if http_server is not None:
        http_server.should_exit = True
    if http_thread is not None:
        await asyncio.to_thread(http_thread.join, args.shutdown_grace_s)
    if prefill_hook is not None:
        prefill_hook.close()
    await server.stop(grace=args.shutdown_grace_s)
    _LOG.info("kakeya gRPC RuntimeService stopped cleanly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["cpu", "mlx", "restored"], default="cpu")
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu",
                    help="Torch device for the restored backend "
                         "(e.g. 'cuda' on a GPU host). Ignored by cpu/mlx.")
    ap.add_argument("--drafter-id", default="",
                    help="DFlash drafter id/path (backend=restored).")
    ap.add_argument("--f-theta-dir", default="",
                    help="Trained f_θ checkpoint dir (backend=restored).")
    ap.add_argument("--no-s5-exact-full-attn", action="store_true",
                    help="Disable S5 (keep f_θ for full-attention layers too). "
                         "By default backend=restored uses S5 exact full-attn "
                         "layers for recall.")
    ap.add_argument("--bind", default=DEFAULT_BIND_ADDRESS,
                    help=f"host:port to bind. Default: {DEFAULT_BIND_ADDRESS}")
    ap.add_argument("--capacity", type=int, default=4,
                    help="SessionStore + SlabPool capacity. Each unit is "
                         "one concurrent session worth of (sink+window) KV "
                         "cache. v0.3 single-tenant defaults to 4.")
    ap.add_argument("--sink", type=int, default=4,
                    help="Sink-token KV cache size (per-session).")
    ap.add_argument("--window", type=int, default=64,
                    help="Sliding-window KV cache size (per-session). "
                         "Together with --sink, bounds total KV per session "
                         "to (sink+window) tokens.")
    ap.add_argument("--max-concurrent-rpcs", type=int, default=None,
                    help="Cap on simultaneous in-flight gRPC RPCs. "
                         "Defaults to grpc.aio's default; set explicitly on "
                         "CPU-bound hosts.")
    ap.add_argument("--multi-tenant", action="store_true",
                    help="PR-A3c: per-session verifier binding — each session "
                         "gets its own isolated KV cache (sharing model "
                         "weights). Requires backend=restored (spawn()). "
                         "Without it, v0.3 single-tenant (one shared cache).")
    ap.add_argument("--shutdown-grace-s", type=float, default=5.0,
                    help="Seconds to give in-flight RPCs to finish on "
                         "SIGTERM/SIGINT before hard-aborting.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # --- Multi-host capability plane (ADR 0009, v0.5-M1) -------------
    ap.add_argument("--enable-capability-exchange", action="store_true",
                    help="Serve kakeya.v1.CapabilityService on the same "
                         "port and advertise this node's capability card. "
                         "Implied by --peer / --serve-ngram-proposer.")
    ap.add_argument("--node-id", default="",
                    help="Fleet-unique node identity for the capability "
                         "card. Defaults to the hostname.")
    ap.add_argument("--advertise", default="",
                    help="host:port that PEERS should use to reach this "
                         "node (use the LAN address, not 127.0.0.1). "
                         "Defaults to --bind.")
    ap.add_argument("--peer", action="append", default=[],
                    help="Seed peer address (host:port) for capability "
                         "gossip. Repeatable. The fleet view converges "
                         "as long as seed edges form a connected graph.")
    ap.add_argument("--serve-ngram-proposer", action="store_true",
                    help="Serve kakeya.v1.ProposerService with the "
                         "model-free prompt-lookup proposer and advertise "
                         "it (model_id 'ngram') on the capability card.")
    ap.add_argument("--exchange-interval-s", type=float, default=30.0,
                    help="Seconds between gossip rounds with --peer "
                         "addresses.")
    ap.add_argument("--capability-ttl-s", type=float, default=120.0,
                    help="TTL of this node's capability card; peers drop "
                         "it if not refreshed within this window.")
    # --- Distributed prefill K/V cache --------------------------------
    ap.add_argument("--enable-prefill-cache", action="store_true",
                    help="Enable immutable longest-prefix prefill K/V reuse "
                         "and serve PrefillCacheService.")
    ap.add_argument("--prefill-cache-gb", type=float, default=4.0,
                    help="Maximum local in-memory snapshot cache size.")
    ap.add_argument("--cache-peer", action="append", default=[],
                    help="Peer PrefillCacheService host:port. Repeatable.")
    ap.add_argument("--cache-advertise", default="",
                    help="Peer-reachable PrefillCacheService address. "
                         "Defaults to --advertise/--bind.")
    ap.add_argument("--cache-block-tokens", type=int, default=64,
                    help="Token boundary interval for restorable snapshots.")
    ap.add_argument("--cache-format-version", default="kakeya-prefill-v2-zlib")
    ap.add_argument("--cache-model-id", default="",
                    help="Logical cache model id; defaults to --verifier-id. "
                         "Use this when verifier-id is a host-specific path.")
    ap.add_argument("--cache-quantization", default="",
                    help="Exact model quantization label used in compatibility.")
    ap.add_argument("--cache-kv-dtype", default="bfloat16")
    ap.add_argument("--model-revision", default="",
                    help="Exact weights revision/hash for cache compatibility.")
    ap.add_argument("--tokenizer-revision", default="",
                    help="Exact tokenizer/chat-template revision.")
    ap.add_argument("--rope-hash", default="",
                    help="RoPE/position configuration fingerprint.")
    ap.add_argument("--cache-lookup-timeout-s", type=float, default=2.0)
    ap.add_argument("--cache-fetch-timeout-s", type=float, default=30.0)
    ap.add_argument("--cache-tenant-id", default="default",
                    help="Tenant namespace included in cache compatibility.")
    ap.add_argument("--fleet-psk-file", default="",
                    help="Optional fleet PSK file for authenticated prefill RPCs "
                         "and tenant-HMAC prefix hashes.")
    ap.add_argument("--cache-compression",
                    choices=["none", "zlib", "kakeyalattice-d4"],
                    default="zlib")
    ap.add_argument("--cache-replication-factor", type=int, default=1)
    ap.add_argument("--cache-max-import-gb", type=float, default=1.0,
                    help="Reject remote snapshots whose wire or expanded size "
                         "would exceed this import budget.")
    ap.add_argument("--cache-estimated-bytes-per-token", type=int, default=400000)
    ap.add_argument("--cache-link-mbps", type=float, default=1000.0)
    ap.add_argument("--cache-default-rtt-ms", type=float, default=2.0)
    ap.add_argument("--local-prefill-tps", type=float, default=20.0)
    ap.add_argument("--worker-prefill-tps", type=float, default=20.0)
    ap.add_argument("--prefill-min-savings-ratio", type=float, default=0.10)
    ap.add_argument("--primary-prefill-penalty-ms", type=float, default=5000.0,
                    help="Opportunity cost assigned to blocking the primary "
                         "with prefill; drives work to compute peers.")
    ap.add_argument("--remote-prefill-min-tokens", type=int, default=128)
    ap.add_argument("--prefill-worker-timeout-s", type=float, default=120.0)
    ap.add_argument(
        "--prefill-policy",
        choices=["local-fallback", "remote-required"],
        default="local-fallback",
        help="Use remote workers opportunistically, or require complete remote "
             "prefill so the primary remains decode-only.",
    )
    ap.add_argument("--network-label", default="lan",
                    help="Advertised interface: thunderbolt|lan|tailscale|public.")
    ap.add_argument("--network-priority", type=int, default=50)
    ap.add_argument("--measured-rtt-ms", type=float, default=0.0)
    ap.add_argument("--network-http-host", default="127.0.0.1")
    ap.add_argument("--network-http-port", type=int, default=0,
                    help="Serve the inference-network API/dashboard; 0 disables.")
    ap.add_argument("--network-state",
                    default="~/.kakeya/inference_network.json")
    ap.add_argument("--network-api-key", default="",
                    help="X-API-Key required for registration/group/telemetry writes.")
    ap.add_argument("--network-telemetry-url", default="",
                    help="Optional POST endpoint receiving completed token counters.")
    ap.add_argument("--network-telemetry-api-key", default="")
    ap.add_argument(
        "--cache-fill-capture-size",
        type=int,
        default=0,
        help="Maintenance-only in-memory first-append capture queue size; "
             "0 disables capture.",
    )
    ap.add_argument("--skip-cache-check", action="store_true",
                    help="Skip the HF-cache pre-flight assertion. By "
                         "default the server fails fast if the verifier "
                         "model isn't already cached, pointing the user "
                         "at scripts/kakeya_prewarm.py. Use this flag in "
                         "CI workflows that have populated the cache "
                         "out-of-band, or when intentionally accepting "
                         "the first-run download.")
    args = ap.parse_args()

    return asyncio.run(_serve(args))


if __name__ == "__main__":
    sys.exit(main())
