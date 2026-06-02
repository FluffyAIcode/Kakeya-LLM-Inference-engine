"""Shared fixtures and marker plumbing for the integration suite.

Tests under ``tests/integration/`` exercise the v0.3 runtime against
**real** model weights — typically Qwen3-0.6B from the HF cache.
They are NOT part of the Linux unit-test gate (model loading is
HF-cache- and hardware-bound) and are NOT auto-discovered by a bare
``pytest``: every test in this directory gets the
``@pytest.mark.integration`` marker auto-applied below, and you opt
in with ``pytest -m integration tests/integration/``.

This conftest is created independently by PR-E1, PR-N1, PR-N2, PR-N3,
and PR-N4 (they all branched off main while none had merged yet);
the file content is the union and de-duplicates cleanly because each
PR appends its own real-engine / real-runtime fixtures.

Per ADR 0008 §9: this suite is the binding GA gate. Mac M4 reviewer
scripts (``scripts/review_pr_n*_on_mac.sh``) drive it manually
until PR-E2 ships the self-hosted runner workflow.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-mark every test under ``tests/integration/`` with
    ``@pytest.mark.integration`` so contributors don't have to
    repeat the decorator on every test in this directory.

    Standard pytest behavior: tests with this marker run only when
    explicitly selected via ``-m integration``; a bare ``pytest``
    invocation skips them.
    """
    for item in items:
        # str(item.fspath) is reliable across pytest versions; "rootpath"
        # comparisons would also work but require a config dependency.
        if "tests/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# Real engine fixture — used by PR-N2's migrated scheduler tests + future
# integration tests that exercise the HTTP shim or the SpeculativeEngine
# end-to-end. Session-scoped so the model load cost (~3-5s on CPU)
# is paid once across the whole suite.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_speculative_engine():
    """Real :class:`SpeculativeEngine` over real Qwen3-0.6B.

    Mirrors the long-standing fixture under ``tests/system/conftest.py``
    but uses Qwen3-0.6B (not 1.7B) to match the rest of the integration
    suite — keeps the HF cache footprint a single model, faster to
    set up on Mac M4 24 GB.
    """
    import torch

    from inference_engine.proposer import SparseLogitsProposer
    from inference_engine.server.engine import SpeculativeEngine
    from kv_cache_proposer.proposer import ProposerConfig
    from kv_cache_proposer.speculative import SpeculativeDecoder
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        model_id="Qwen/Qwen3-0.6B",
        dtype=torch.bfloat16, device="cpu",
        sink_size=4, window_size=64,
    )
    proposer = SparseLogitsProposer(proposer_cfg)
    verifier = SinkWindowVerifier(verifier_cfg)
    decoder = SpeculativeDecoder(
        proposer=proposer, verifier=verifier,
        block_size=8, num_diffusion_steps=2,
    )
    return SpeculativeEngine(
        decoder=decoder,
        tokenizer=verifier.tokenizer,
        model_id_label="kakeya-integration",
    )


# ---------------------------------------------------------------------------
# Real gRPC runtime fixture — used by PR-N4's SDK integration tests.
# An in-process gRPC server backed by a real verifier on a background
# thread, yielding the host:port string the SDK can connect to.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_grpc_runtime_address():
    """Run an in-process gRPC ``RuntimeService`` backed by a real
    Qwen3-0.6B :class:`SinkWindowVerifier` on a background thread.

    Yields the ``host:port`` address string the SDK can connect to.
    Session-scoped: model load (~3-5 s on CPU) is paid once. Each
    integration SDK test creates its own session via the SDK; the
    underlying verifier is shared and reset on each ``prefill`` call.
    """
    import asyncio
    import threading
    import time

    import grpc
    import torch

    from inference_engine.server.grpc_app import RuntimeServiceServicer
    from inference_engine.server.proto_gen.kakeya.v1 import (
        runtime_pb2_grpc,
    )
    from inference_engine.session import (
        AppendTokensCoordinator,
        GenerationCoordinator,
        SessionStore,
    )
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

    verifier_cfg = VerifierConfig(
        model_id="Qwen/Qwen3-0.6B",
        dtype=torch.bfloat16, device="cpu",
        sink_size=4, window_size=64,
    )
    verifier = SinkWindowVerifier(verifier_cfg)
    store = SessionStore(capacity=4, cache_inspector=verifier)
    append_coord = AppendTokensCoordinator(store, verifier)
    gen_coord = GenerationCoordinator(store, verifier)

    loop = asyncio.new_event_loop()
    holder: dict = {
        "server": None,
        "port": None,
        "started": threading.Event(),
    }

    async def _serve():
        # Build the server INSIDE the worker thread's loop so any
        # internal asyncio.Future is bound to this loop, not the
        # main-thread default loop (the "Future attached to a
        # different loop" failure PR-B4 hit).
        server = grpc.aio.server()
        runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
            RuntimeServiceServicer(
                store,
                append_coordinator=append_coord,
                generation_coordinator=gen_coord,
            ),
            server,
        )
        holder["server"] = server
        holder["port"] = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        holder["started"].set()
        await server.wait_for_termination()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    if not holder["started"].wait(timeout=15.0):
        raise RuntimeError(
            "background gRPC runtime failed to start within 15s",
        )

    address = f"127.0.0.1:{holder['port']}"
    try:
        yield address
    finally:
        async def _shutdown():
            await holder["server"].stop(grace=0.1)

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            fut.result(timeout=2.0)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
        thread.join(timeout=2.0)
        time.sleep(0.05)
        try:
            loop.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
