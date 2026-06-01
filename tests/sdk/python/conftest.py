"""Test fixtures for the Kakeya Python SDK.

The SDK is sync (``grpc.insecure_channel`` + sync stubs); the
runtime under test is async (``grpc.aio.server`` + async
servicer). They are wire-compatible (HTTP/2 gRPC), but the async
server needs an event loop running to respond to RPCs.

The :func:`runtime_address` fixture spins up the async server in a
background thread with its own event loop and yields the
``host:port`` string the SDK can connect to. Cleanup stops the
server through a ``call_soon_threadsafe`` round-trip and joins the
thread.

This pattern keeps SDK tests free of pytest-asyncio dependence
while still exercising the production gRPC machinery. No mocks of
the SUT — only a deterministic ``FakeVerifier`` (already shared
with the coordinator and gRPC-app test suites) so the runtime's
behavior is observable.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import grpc
import pytest

from inference_engine.server.grpc_app import RuntimeServiceServicer
from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2_grpc,
)
from inference_engine.session import (
    AppendTokensCoordinator,
    GenerationCoordinator,
    SessionStore,
)

# Shared FakeVerifier from PR-B2's coordinator suite — same fake the
# rest of the Phase-B test suite uses.
from tests.inference_engine.session.test_coordinator import FakeVerifier


@dataclass
class RuntimeFixture:
    """Convenience handle returned by :func:`runtime_address`."""

    address: str
    store: SessionStore
    verifier: FakeVerifier


def _start_runtime(
    *,
    cache_inspector_enabled: bool = True,
    slab_pool: Optional[object] = None,
    capacity: int = 4,
) -> tuple[RuntimeFixture, threading.Thread, asyncio.AbstractEventLoop, "_ServerHolder"]:
    """Spin up an async runtime in a background thread.

    Returns ``(fixture, thread, loop, server_holder)``. ``server_holder``
    boxes the server reference because the actual ``grpc.aio.Server``
    object is constructed *inside* the worker thread's loop —
    constructing it in the main thread and using it in another
    thread's loop produces ``Future attached to a different loop``
    errors from grpcio's internals.

    ``cache_inspector_enabled``: when True the FakeVerifier is also
    wired into ``SessionStore`` as the ``cache_inspector`` so INV-1
    is enforced. Disable to test paths where INV-1 must not fire.

    ``slab_pool`` / ``capacity``: passed through to ``SessionStore``
    for tests that need a constrained pool (e.g.,
    RESOURCE_EXHAUSTED scenarios where ``capacity > num_slabs``).
    """
    fv = FakeVerifier()
    inspector = fv if cache_inspector_enabled else None
    store_kwargs = {"capacity": capacity, "cache_inspector": inspector}
    if slab_pool is not None:
        store_kwargs["slab_pool"] = slab_pool
    store = SessionStore(**store_kwargs)
    append_coord = AppendTokensCoordinator(store, fv)
    gen_coord = GenerationCoordinator(store, fv)

    loop = asyncio.new_event_loop()
    server_holder = _ServerHolder()
    port_holder: dict = {"port": None, "started": threading.Event()}

    async def _serve() -> None:
        # Construct the server INSIDE the worker thread's loop so
        # any internal asyncio.Future the server allocates is bound
        # to this loop (and not the main-thread default loop).
        server = grpc.aio.server()
        runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
            RuntimeServiceServicer(
                store,
                append_coordinator=append_coord,
                generation_coordinator=gen_coord,
            ),
            server,
        )
        server_holder.server = server
        port_holder["port"] = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        port_holder["started"].set()
        # Block until server.stop() is scheduled from another thread.
        # wait_for_termination() returns cleanly once stop() is invoked,
        # which lets run_until_complete(_serve()) return without a
        # cancellation tantrum.
        await server.wait_for_termination()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started = port_holder["started"].wait(timeout=5.0)
    if not started:  # pragma: no cover - environment fallback
        raise RuntimeError("background gRPC server failed to start")

    fixture = RuntimeFixture(
        address=f"127.0.0.1:{port_holder['port']}",
        store=store,
        verifier=fv,
    )
    return fixture, thread, loop, server_holder


class _ServerHolder:
    """Box for the gRPC server, populated inside the worker thread's
    loop (see ``_start_runtime``)."""

    def __init__(self) -> None:
        self.server: Optional[grpc.aio.Server] = None


def _stop_runtime(
    thread: threading.Thread,
    loop: asyncio.AbstractEventLoop,
    server_holder: "_ServerHolder",
) -> None:
    """Gracefully stop the background-thread runtime.

    Sequence:

      1. Schedule ``server.stop(grace)`` on the worker loop. Once
         it returns, ``server.wait_for_termination()`` (the body of
         ``_serve()``) returns, which lets
         ``loop.run_until_complete(_serve())`` return cleanly.
      2. Wait for the thread to finish naturally.
      3. Close the loop. No tasks are left scheduled, so this is
         a clean close.
    """
    server = server_holder.server
    if server is None:  # pragma: no cover - server creation completed before this is called
        thread.join(timeout=2.0)
        loop.close()
        return

    async def _shutdown() -> None:
        await server.stop(grace=0.1)

    fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
    try:
        fut.result(timeout=2.0)
    except Exception:  # pragma: no cover - best-effort shutdown
        pass
    thread.join(timeout=2.0)
    loop.close()


@pytest.fixture
def runtime_address() -> Iterator[RuntimeFixture]:
    """Yield a live runtime; tear down on test teardown.

    The fixture is function-scoped: each test gets a fresh
    SessionStore + FakeVerifier so cross-test state cannot leak.
    Concurrent tests don't collide because each spins up on a free
    port (``127.0.0.1:0``).
    """
    fixture, thread, loop, server = _start_runtime()
    try:
        yield fixture
    finally:
        _stop_runtime(thread, loop, server)


@pytest.fixture
def runtime_address_no_inspector() -> Iterator[RuntimeFixture]:
    """Variant: store has no ``cache_inspector``, so INV-1 cannot
    fire from the SDK side. Useful for tests that exercise other
    error paths without accidentally tripping INV-1."""
    fixture, thread, loop, server = _start_runtime(
        cache_inspector_enabled=False,
    )
    try:
        yield fixture
    finally:
        _stop_runtime(thread, loop, server)
