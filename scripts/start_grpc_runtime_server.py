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
    cfg = verifier.model.config
    # Gemma 4 is multimodal: decoder dims live under config.text_config.
    cfg = getattr(cfg, "text_config", None) or cfg
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    # Qwen3 / Gemma / DeepSeek all support GQA — kv-heads is the
    # dimension that matters for KV cache size, not attention-heads.
    num_kv_heads = int(
        getattr(cfg, "num_key_value_heads", None)
        or getattr(cfg, "num_attention_heads")
    )
    head_dim = int(
        getattr(cfg, "head_dim", None)
        or (cfg.hidden_size // cfg.num_attention_heads)
    )
    return num_layers, num_kv_heads, head_dim


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


def _build_capability_registry(args: argparse.Namespace, *, backend: str):
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
    )
    _LOG.info("capability card: %s @ %s ring=%r models=%s",
              self_card.node_id, self_card.grpc_address,
              self_card.ring_address,
              [(m.model_id, m.role.name) for m in models])
    return CapabilityRegistry(self_card=self_card)


async def _exchange_loop(registry, peers, interval_s: float) -> None:
    """Periodic gossip with seed peers until the task is cancelled."""
    from inference_engine.distributed.exchange import exchange_once

    while True:
        report = await exchange_once(registry, peers)
        if report.errors:
            _LOG.warning("gossip errors: %s", report.errors)
        if report.merged_cards:
            _LOG.info(
                "gossip merged %d card(s); fleet size now %d",
                report.merged_cards, registry.peer_count + 1,
            )
        await asyncio.sleep(interval_s)


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
    append_coord = AppendTokensCoordinator(store, verifier, resolver=resolver)
    gen_coord = GenerationCoordinator(store, verifier, resolver=resolver)

    # Multi-host capability plane (ADR 0009): only constructed when
    # the operator opts in via --enable-capability-exchange (implied
    # by --peer / --serve-ngram-proposer).
    distributed_enabled = bool(
        args.enable_capability_exchange or args.peer or args.serve_ngram_proposer
    )
    registry = None
    proposers = None
    if distributed_enabled:
        registry = _build_capability_registry(args, backend=args.backend)
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
    )

    await server.start()
    _LOG.info("kakeya gRPC RuntimeService listening on %s", args.bind)

    exchange_task = None
    if registry is not None and args.peer:
        exchange_task = asyncio.create_task(
            _exchange_loop(registry, list(args.peer), args.exchange_interval_s),
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
