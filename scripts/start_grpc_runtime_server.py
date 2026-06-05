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
    raise SystemExit(f"unknown backend: {backend}")


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
    store = SessionStore(
        capacity=args.capacity,
        cache_inspector=verifier,
        slab_pool=pool,
    )
    append_coord = AppendTokensCoordinator(store, verifier)
    gen_coord = GenerationCoordinator(store, verifier)

    config = GrpcServerConfig(
        bind_address=args.bind,
        max_concurrent_rpcs=args.max_concurrent_rpcs,
    )
    server = create_grpc_server(
        session_store=store,
        append_coordinator=append_coord,
        generation_coordinator=gen_coord,
        config=config,
    )

    await server.start()
    _LOG.info("kakeya gRPC RuntimeService listening on %s", args.bind)

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
    await server.stop(grace=args.shutdown_grace_s)
    _LOG.info("kakeya gRPC RuntimeService stopped cleanly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["cpu", "mlx"], default="cpu")
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-0.6B")
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
    ap.add_argument("--shutdown-grace-s", type=float, default=5.0,
                    help="Seconds to give in-flight RPCs to finish on "
                         "SIGTERM/SIGINT before hard-aborting.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
