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

The SDK tests are **wire-layer** tests: their truth is "the gRPC
encode/decode + error-status-mapping behaves correctly", not "the
verifier produces correct numerics". To make AppendTokens and
Generate respond at all, the runtime needs *some* verifier object
satisfying ``VerifierProtocol``. PR-N1 replaced the previous
shared ``FakeVerifier`` import with a minimum-protocol-conformance
``_MinimalVerifierStub`` defined locally below — scoped strictly
to SDK transport testing. The stub does NOT mirror real verifier
state-mutation contracts; it just satisfies the protocol shape.
End-to-end runtime correctness is covered by
``tests/integration/test_coordinator_real.py`` and the binding
GA-gate ``test_inv3_session_determinism_gate.py`` against real
Qwen3-0.6B (PR-E1).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import grpc
import pytest
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


# ---------------------------------------------------------------------------
# Minimum VerifierProtocol stub for SDK wire-layer tests.
#
# This is NOT a verifier mirror. It satisfies just enough of
# ``inference_engine.session.coordinator.VerifierProtocol`` to make
# AppendTokens and Generate succeed end-to-end so the SDK can
# observe the wire response (status code, payload encoding,
# stream order). Real-numerics verifier validation lives in
# ``tests/integration/test_coordinator_real.py`` per PR-N1's
# no-test-doubles split.
# ---------------------------------------------------------------------------


class _MinimalVerifierStub:
    """Bare-bones ``VerifierProtocol`` impl for transport tests.

    Sink+window=2+4=6, vocab=16. Maintains the same state-mutation
    invariants the real verifier does (cached_token_sequence stays
    in sync with K/V tensor seq dim) so the SessionStore's INV-1
    enforcement works against it. No real attention or model.
    """

    SINK = 2
    WINDOW = 4
    VOCAB = 16

    def __init__(self) -> None:
        self.cached_token_sequence: List[int] = []
        self.next_global_position: int = 0
        self.next_token_logits: torch.Tensor = torch.zeros(self.VOCAB)

    def _trim(self, seq: List[int]) -> List[int]:
        budget = self.SINK + self.WINDOW
        if len(seq) <= budget:
            return list(seq)
        return list(seq[: self.SINK]) + list(seq[-self.WINDOW:])

    def _greedy(self, hist: List[int]) -> torch.Tensor:
        out = torch.zeros(self.VOCAB)
        if hist:
            out[sum(hist[-3:]) % self.VOCAB] = 1.0
        return out

    def k_seq_length(self, session: object) -> int:
        del session
        return len(self.cached_token_sequence)

    def kv_live_bytes(self, session: object) -> int:
        del session
        # Synthetic per-token bytes — irrelevant to wire-layer truth.
        return len(self.cached_token_sequence) * 13

    def prefill(self, prompt_ids: List[int]) -> None:
        self.cached_token_sequence = self._trim(prompt_ids)
        self.next_global_position = len(prompt_ids)
        self.next_token_logits = self._greedy(self.cached_token_sequence)

    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        # Mutate cache: add new tokens, then trim.
        self.cached_token_sequence = self._trim(
            self.cached_token_sequence + list(tokens),
        )
        rows = []
        running = list(self.cached_token_sequence)
        for tok in tokens:
            running = self._trim(running + [tok])
            rows.append(self._greedy(running))
        return torch.stack(rows) if rows else torch.zeros(0, self.VOCAB)

    def commit_or_truncate(self, *, forwarded: int, accepted: int) -> None:
        # forwarded == accepted in prompt-mode; full accept. Advance
        # position by accepted and trim to budget (idempotent).
        del forwarded
        self.next_global_position += accepted
        self.cached_token_sequence = self._trim(self.cached_token_sequence)


# Backward-compat name used by existing fixture code below.
FakeVerifier = _MinimalVerifierStub


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
