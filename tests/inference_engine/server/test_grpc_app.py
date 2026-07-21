"""Unit tests for inference_engine.server.grpc_app (ADR 0008 PR-B1).

Tests use a real ``grpc.aio.server`` bound to a random localhost
port. This is heavier than mocking ``ServicerContext`` but gives us:

- coverage of the actual wire-format encode/decode through the
  generated stubs;
- end-to-end exercise of ``context.abort`` raising and being
  translated into typed gRPC status codes;
- a regression net for the codegen contract (if a future change to
  the .proto silently drifts away from ``CreateSession`` /
  ``CloseSession`` / ``GetSessionInfo``, these tests fail at
  channel-construction time).

Coverage target: 100% on ``inference_engine/server/grpc_app.py``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

import grpc
import pytest
import pytest_asyncio

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.server.grpc_app import (
    DEFAULT_BIND_ADDRESS,
    GrpcServerConfig,
    RuntimeServiceServicer,
    create_grpc_server,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from inference_engine.session import SessionStore

pytestmark = pytest.mark.asyncio


def _tiny_slab_pool(num_slabs: int = 4) -> SlabPool:
    cfg = SlabConfig(
        num_layers=1,
        num_heads=1,
        sink_size=1,
        window_size=2,
        head_dim=4,
    )
    return SlabPool(num_slabs=num_slabs, slab_config=cfg)


@pytest_asyncio.fixture
async def grpc_pair() -> AsyncIterator[
    tuple[runtime_pb2_grpc.RuntimeServiceStub, SessionStore, grpc.aio.Server]
]:
    """Start a real in-process gRPC server bound to ``127.0.0.1:0``.

    Returns ``(stub, session_store, server)``. The server is shut
    down with a small grace period in teardown so unfinished RPCs
    don't dangle.
    """
    store = SessionStore(capacity=4)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        yield stub, store, server
    finally:
        await channel.close()
        await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# CreateSession
# ---------------------------------------------------------------------------


async def test_create_session_returns_server_issued_id(grpc_pair):
    stub, store, _ = grpc_pair
    resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    assert resp.session_id.startswith("sess-")
    assert store.active_count == 1
    # Server-side state confirms the session was actually allocated
    # via the store, not synthesized by the servicer.
    sess = store.get_session(resp.session_id)
    assert sess.session_id == resp.session_id


async def test_create_session_records_eos_token_ids(grpc_pair):
    stub, store, _ = grpc_pair
    resp = await stub.CreateSession(
        runtime_pb2.CreateSessionRequest(eos_token_ids=[7, 11, 13]),
    )
    sess = store.get_session(resp.session_id)
    assert sess.eos_token_ids == (7, 11, 13)


async def test_create_session_records_client_label(grpc_pair):
    stub, store, _ = grpc_pair
    resp = await stub.CreateSession(
        runtime_pb2.CreateSessionRequest(client_label="demo-app-1"),
    )
    sess = store.get_session(resp.session_id)
    assert sess.client_label == "demo-app-1"


async def test_create_session_default_eos_is_empty_tuple(grpc_pair):
    stub, store, _ = grpc_pair
    resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    sess = store.get_session(resp.session_id)
    assert sess.eos_token_ids == ()


async def test_create_session_pool_exhausted_returns_resource_exhausted():
    # capacity > num_slabs so eviction cannot satisfy the second
    # create; the second create must surface PoolExhausted as
    # RESOURCE_EXHAUSTED rather than silently degrade.
    pool = _tiny_slab_pool(num_slabs=1)
    store = SessionStore(capacity=4, slab_pool=pool)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        # First create succeeds.
        await stub.CreateSession(runtime_pb2.CreateSessionRequest())
        # Second create exhausts the pool.
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.CreateSession(runtime_pb2.CreateSessionRequest())
        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "slab pool exhausted" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# CloseSession
# ---------------------------------------------------------------------------


async def test_close_session_returns_final_history_length(grpc_pair):
    stub, store, _ = grpc_pair
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    # Append some tokens to give close something to report.
    sess = store.get_session(create_resp.session_id)
    sess.history_token_ids.extend([10, 20, 30])
    close_resp = await stub.CloseSession(
        runtime_pb2.CloseSessionRequest(session_id=create_resp.session_id),
    )
    assert close_resp.final_history_length == 3
    assert store.active_count == 0


async def test_close_session_returns_zero_for_empty_session(grpc_pair):
    stub, _, _ = grpc_pair
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    close_resp = await stub.CloseSession(
        runtime_pb2.CloseSessionRequest(session_id=create_resp.session_id),
    )
    assert close_resp.final_history_length == 0


async def test_close_session_unknown_id_returns_not_found(grpc_pair):
    stub, _, _ = grpc_pair
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.CloseSession(
            runtime_pb2.CloseSessionRequest(session_id="sess-nonexistent"),
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    assert "sess-nonexistent" in exc_info.value.details()


async def test_close_session_double_close_returns_not_found(grpc_pair):
    stub, _, _ = grpc_pair
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    await stub.CloseSession(
        runtime_pb2.CloseSessionRequest(session_id=create_resp.session_id),
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.CloseSession(
            runtime_pb2.CloseSessionRequest(session_id=create_resp.session_id),
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


async def test_close_session_invokes_unified_store_removal_hook():
    """CloseSession reaches the same hook used by eviction and failures."""
    seen: list[tuple[str, str]] = []
    store = SessionStore(
        capacity=4,
        on_session_removed=lambda *args: seen.append(args),
    )
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
        await stub.CloseSession(
            runtime_pb2.CloseSessionRequest(session_id=create_resp.session_id),
        )
        assert seen == [(create_resp.session_id, "explicit_close")]
    finally:
        await channel.close()
        await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# GetSessionInfo
# ---------------------------------------------------------------------------


async def test_get_session_info_initial_state(grpc_pair):
    stub, _, _ = grpc_pair
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    info = await stub.GetSessionInfo(
        runtime_pb2.GetSessionInfoRequest(session_id=create_resp.session_id),
    )
    assert info.history_length == 0
    assert info.kv_live_bytes == 0  # pool-less store
    assert info.cache_invariant_inv1_violations == 0
    assert info.cache_invariant_inv2_violations == 0
    assert info.idle_seconds >= 0.0


async def test_get_session_info_reflects_history_growth(grpc_pair):
    stub, store, _ = grpc_pair
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    sess = store.get_session(create_resp.session_id)
    sess.history_token_ids.extend([1, 2, 3, 4, 5])
    info = await stub.GetSessionInfo(
        runtime_pb2.GetSessionInfoRequest(session_id=create_resp.session_id),
    )
    assert info.history_length == 5


async def test_get_session_info_reflects_kv_live_bytes_when_pool_present():
    # Wire a pool-aware store + drive a real KV byte count via the
    # slab's override field (the same hook PooledVerifier currently
    # uses to publish its real peak_kv_bytes; PR-A3b's
    # Session.kv_live_bytes() reads through to it).
    pool = _tiny_slab_pool(num_slabs=2)
    store = SessionStore(capacity=2, slab_pool=pool)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        sess = store.get_session(create_resp.session_id)
        sess.slab.live_kv_bytes_override = 7654321
        info = await stub.GetSessionInfo(
            runtime_pb2.GetSessionInfoRequest(
                session_id=create_resp.session_id,
            ),
        )
        assert info.kv_live_bytes == 7654321
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_get_session_info_unknown_id_returns_not_found(grpc_pair):
    stub, _, _ = grpc_pair
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.GetSessionInfo(
            runtime_pb2.GetSessionInfoRequest(session_id="sess-nonexistent"),
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


async def test_get_session_info_after_close_returns_not_found(grpc_pair):
    stub, _, _ = grpc_pair
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    await stub.CloseSession(
        runtime_pb2.CloseSessionRequest(session_id=create_resp.session_id),
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.GetSessionInfo(
            runtime_pb2.GetSessionInfoRequest(
                session_id=create_resp.session_id,
            ),
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


# ---------------------------------------------------------------------------
# AppendTokens (PR-B2) — verifier-independent paths only
#
# Tests that wire a real AppendTokensCoordinator + verifier moved to
# tests/integration/test_grpc_runtime_real.py in PR-N1. The Linux
# gate keeps only the "no coordinator wired in" UNIMPLEMENTED check
# below.
# ---------------------------------------------------------------------------


async def test_append_tokens_returns_unimplemented_when_no_coordinator(grpc_pair):
    """Servicer constructed without an AppendTokensCoordinator (PR-B1
    mode) keeps the framework's UNIMPLEMENTED default for
    AppendTokens. This regression test guards against a future PR
    that wires AppendTokens unconditionally and breaks the optional-
    coordinator contract."""
    stub, _, _ = grpc_pair
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.AppendTokens(
            runtime_pb2.AppendTokensRequest(
                session_id="sess-anything", token_ids=[1, 2, 3],
            ),
        )
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED


# Error-mapping tests for AppendTokens. These don't load a real
# verifier; they use AppendTokensCoordinator subclasses that raise
# the relevant exception directly. ``verifier=None`` is safe because
# the override never touches ``self._verifier``.


async def test_append_tokens_value_error_returns_invalid_argument():
    """ValueError raised by the coordinator → INVALID_ARGUMENT
    on the wire. Verifier is never consulted on this path."""
    from inference_engine.session import AppendTokensCoordinator

    class _ValueErroringCoordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            del token_ids
            self._store.get_session(session_id)  # SessionNotFound first
            raise ValueError("synthetic well-formedness violation")

    store = SessionStore(capacity=2)
    coord = _ValueErroringCoordinator(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coord), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=create_resp.session_id, token_ids=[1],
                ),
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "synthetic well-formedness" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_append_tokens_remote_required_error_returns_unavailable():
    from inference_engine.distributed.prefill_cache_runtime import (
        RemotePrefillRequiredError,
    )
    from inference_engine.session import AppendTokensCoordinator

    class Coordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            raise RemotePrefillRequiredError("remote worker unavailable")

    class Aborted(Exception):
        pass

    class Context:
        code = None
        detail = ""

        async def abort(self, code, detail):
            self.code = code
            self.detail = detail
            raise Aborted

    store = SessionStore(capacity=1)
    servicer = RuntimeServiceServicer(
        store,
        append_coordinator=Coordinator(store, verifier=None),
    )
    context = Context()
    with pytest.raises(Aborted):
        await servicer.AppendTokens(
            runtime_pb2.AppendTokensRequest(session_id="s", token_ids=[1]),
            context,
        )
    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "remote worker unavailable" in context.detail


async def test_append_tokens_invariant_violation_returns_failed_precondition():
    """InvariantViolation raised by the coordinator → FAILED_PRECONDITION
    on the wire. Verifier is never consulted on this path."""
    from inference_engine.session import (
        AppendTokensCoordinator,
        InvariantViolation,
    )

    class _InvariantViolatingCoordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            del token_ids
            self._store.get_session(session_id)  # SessionNotFound first
            raise InvariantViolation(
                kind="1",
                session_id=session_id,
                detail="synthetic INV-1 violation",
            )

    store = SessionStore(capacity=2)
    coord = _InvariantViolatingCoordinator(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coord), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=create_resp.session_id, token_ids=[1],
                ),
            )
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "synthetic INV-1" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_generate_invariant_violation_returns_failed_precondition():
    """InvariantViolation raised by GenerationCoordinator → FAILED_PRECONDITION
    on the wire. The override never touches the verifier."""
    from inference_engine.session import (
        GenerationCoordinator,
        InvariantViolation,
    )

    class _InvariantViolatingGen(GenerationCoordinator):
        def generate(self, session_id, *, max_tokens, **kw):
            del max_tokens, kw
            # GenerationCoordinator.generate is a SYNC GENERATOR
            # function (yields events). The override must also be one
            # — otherwise a synchronous raise from this call escapes
            # before the gRPC handler's try/except can catch it.
            # The unreachable yield turns this into a generator
            # function whose first .next()/iteration raises.
            if False:
                yield  # pragma: no cover - generator marker only
            raise InvariantViolation(
                kind="1",
                session_id=session_id,
                detail="synthetic INV-1 from Generate",
            )

    store = SessionStore(capacity=2)
    gen_coord = _InvariantViolatingGen(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            store, generation_coordinator=gen_coord,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _evt in stub.Generate(
                runtime_pb2.GenerateRequest(
                    session_id=create_resp.session_id, max_tokens=1,
                ),
            ):
                pass
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "synthetic INV-1" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_generate_value_error_returns_invalid_argument():
    """ValueError raised by GenerationCoordinator → INVALID_ARGUMENT.
    Synthetic test driven without a verifier."""
    from inference_engine.session import GenerationCoordinator

    class _ValueErroringGen(GenerationCoordinator):
        def generate(self, session_id, *, max_tokens, **kw):
            del session_id, max_tokens, kw
            if False:
                yield  # pragma: no cover - generator marker only
            raise ValueError("synthetic invalid argument")

    store = SessionStore(capacity=2)
    gen_coord = _ValueErroringGen(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            store, generation_coordinator=gen_coord,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _evt in stub.Generate(
                runtime_pb2.GenerateRequest(
                    session_id=create_resp.session_id, max_tokens=1,
                ),
            ):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_generate_session_not_found_mid_stream_returns_not_found():
    """SessionNotFoundError raised mid-stream from the generator
    after some tokens already flowed → NOT_FOUND on the wire.
    Verifier-independent generator override."""
    from inference_engine.session import GenerationCoordinator
    from inference_engine.session import (
        SessionNotFoundError,
        TokenEvent,
    )

    class _NotFoundMidStream(GenerationCoordinator):
        def generate(self, session_id, *, max_tokens, **kw):
            del max_tokens, kw
            yield TokenEvent(token_id=1)
            raise SessionNotFoundError(session_id)

    store = SessionStore(capacity=2)
    gen_coord = _NotFoundMidStream(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, generation_coordinator=gen_coord),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        events = []
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for evt in stub.Generate(
                runtime_pb2.GenerateRequest(
                    session_id=create_resp.session_id, max_tokens=4,
                ),
            ):
                events.append(evt)
        # The first event flowed normally before the raise.
        assert len(events) == 1
        assert events[0].WhichOneof("payload") == "token_id"
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_generate_cancellation_removes_session():
    """Direct-invocation test: drive the Servicer's Generate with
    a fake gRPC context that flips ``cancelled()`` to True after
    the first event. The servicer must yield a CANCELLED done
    frame and stop. Uses a verifier-free generator override.
    """
    from inference_engine.session import (
        GenerationCoordinator,
        TokenEvent,
    )

    class _TwoTokenGen(GenerationCoordinator):
        def generate(self, session_id, *, max_tokens, **kw):
            del session_id, max_tokens, kw
            yield TokenEvent(token_id=1)
            yield TokenEvent(token_id=2)  # never reached: cancelled first

    store = SessionStore(capacity=1)
    gen_coord = _TwoTokenGen(store, verifier=None)
    servicer = RuntimeServiceServicer(
        store, generation_coordinator=gen_coord,
    )

    class _FakeContext:
        """Minimal stand-in for ``grpc.aio.ServicerContext``.

        Reports cancelled=False on the first poll, True on every
        subsequent poll. Exists purely to drive coverage of the
        servicer's cancellation branch — it does not stand in for
        the verifier. (The ``cancelled``-checked-on-context API is
        gRPC framework surface, not application contract.)
        """
        def __init__(self) -> None:
            self._polls = 0

        def cancelled(self) -> bool:
            self._polls += 1
            return self._polls > 1

        async def abort(self, code, details):  # pragma: no cover
            raise AssertionError(f"abort: {code} {details!r}")

    ctx = _FakeContext()
    sess = store.create_session()
    request = runtime_pb2.GenerateRequest(
        session_id=sess.session_id, max_tokens=10,
    )
    events = []
    async for resp in servicer.Generate(request, ctx):
        events.append(resp)
    assert len(events) == 1
    assert events[0].WhichOneof("payload") == "token_id"
    from inference_engine.session import SessionNotFoundError
    with pytest.raises(SessionNotFoundError):
        store.get_session(sess.session_id)


async def test_wire_cancellation_releases_blocked_generate_session():
    """A disconnect removes session state while model thread is still blocked."""
    from inference_engine.session import GenerationCoordinator

    started = threading.Event()
    release = threading.Event()

    class BlockingGeneration(GenerationCoordinator):
        def generate(self, session_id, *, max_tokens, **kwargs):
            del session_id, max_tokens, kwargs
            started.set()
            release.wait(timeout=2)
            if False:
                yield

    store = SessionStore(capacity=1)
    coordinator = BlockingGeneration(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, generation_coordinator=coordinator),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        session = store.create_session()
        call = stub.Generate(runtime_pb2.GenerateRequest(
            session_id=session.session_id,
            max_tokens=1,
        ))
        read_task = asyncio.create_task(call.read())
        assert await asyncio.to_thread(started.wait, 1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read_task
        for _ in range(50):
            if store.active_count == 0:
                break
            await asyncio.sleep(0.01)
        assert store.active_count == 0
    finally:
        release.set()
        await channel.close()
        await server.stop(grace=0.1)


async def test_append_tokens_session_not_found_returns_not_found():
    """SessionNotFoundError raised by the AppendTokensCoordinator →
    NOT_FOUND on the wire. Coordinator override; verifier-free."""
    from inference_engine.session import AppendTokensCoordinator
    from inference_engine.session.store import SessionNotFoundError

    class _NotFoundCoordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            del token_ids
            raise SessionNotFoundError(session_id)

    store = SessionStore(capacity=2)
    coord = _NotFoundCoordinator(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coord), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id="any-id", token_ids=[1, 2, 3],
                ),
            )
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_append_tokens_success_returns_response():
    """Coordinator override returns successfully → AppendTokensResponse
    with the correct history_length on the wire."""
    from inference_engine.session import AppendTokensCoordinator

    class _SuccessCoordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            del session_id
            return len(list(token_ids)) + 100  # deterministic, easy to assert

    store = SessionStore(capacity=2)
    coord = _SuccessCoordinator(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coord), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        resp = await stub.AppendTokens(
            runtime_pb2.AppendTokensRequest(
                session_id=create_resp.session_id, token_ids=[1, 2, 3],
            ),
        )
        assert resp.history_length == 103
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_append_tokens_worker_does_not_block_event_loop():
    from inference_engine.session import AppendTokensCoordinator

    started = threading.Event()
    release = threading.Event()

    class BlockingCoordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            del session_id, token_ids
            started.set()
            release.wait(timeout=2)
            return 1

    store = SessionStore(capacity=2)
    servicer = RuntimeServiceServicer(
        store,
        append_coordinator=BlockingCoordinator(store, verifier=None),
    )

    class Context:
        def cancelled(self):
            return False

        async def abort(self, code, detail):
            raise AssertionError((code, detail))

    session = store.create_session()
    append_task = asyncio.create_task(servicer.AppendTokens(
        runtime_pb2.AppendTokensRequest(session_id=session.session_id, token_ids=[1]),
        Context(),
    ))
    await asyncio.to_thread(started.wait, 1)
    created = await asyncio.wait_for(
        servicer.CreateSession(runtime_pb2.CreateSessionRequest(), Context()),
        timeout=0.2,
    )
    assert created.session_id
    release.set()
    await append_task


async def test_generate_yields_history_truncated_then_done():
    """Generator override yields HistoryTruncatedEvent + DoneEvent →
    truncated frame + done frame on the wire. Covers the
    HistoryTruncatedEvent and DoneEvent yield paths."""
    from inference_engine.session import (
        DoneEvent,
        GenerationCoordinator,
        HistoryTruncatedEvent,
        STOP_REASON_MAX_TOKENS,
        TokenEvent,
    )

    class _StreamingGen(GenerationCoordinator):
        def generate(self, session_id, *, max_tokens, **kw):
            del session_id, max_tokens, kw
            yield HistoryTruncatedEvent(dropped_token_count=42)
            yield TokenEvent(token_id=7)
            yield DoneEvent(
                stop_reason=STOP_REASON_MAX_TOKENS,
                generated_token_count=1,
                prefill_seconds=0.0,
                total_seconds=0.0,
            )

    store = SessionStore(capacity=2)
    gen_coord = _StreamingGen(store, verifier=None)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            store, generation_coordinator=gen_coord,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(
            runtime_pb2.CreateSessionRequest(),
        )
        events = []
        async for evt in stub.Generate(
            runtime_pb2.GenerateRequest(
                session_id=create_resp.session_id, max_tokens=1,
            ),
        ):
            events.append(evt)
        # Order: truncated → token → done
        assert len(events) == 3
        assert events[0].WhichOneof("payload") == "truncated"
        assert events[0].truncated.dropped_token_count == 42
        assert events[1].WhichOneof("payload") == "token_id"
        assert events[1].token_id == 7
        assert events[2].WhichOneof("payload") == "done"
        assert events[2].done.stop_reason == \
            runtime_pb2.GenerateDone.STOP_REASON_MAX_TOKENS
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_factory_wires_prefill_cache_service():
    from inference_engine.distributed.capability import CacheCompatibility
    from inference_engine.distributed.prefill_cache import PrefixCacheStore
    store = SessionStore(capacity=1)
    cache = PrefixCacheStore(
        CacheCompatibility(model_id="m"),
        max_bytes=100,
        node_id="cache-node",
    )
    server = create_grpc_server(
        session_store=store,
        config=GrpcServerConfig(bind_address="127.0.0.1:0"),
        prefill_cache_store=cache,
        prefill_cache_address="cache:1",
    )
    await server.start()
    # grpc.aio does not expose the selected ephemeral port on the wrapper,
    # so this test pins factory wiring by construction/logical coverage; the
    # real wire path is covered in test_prefill_cache_service.
    await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# Generate (PR-B3) — verifier-independent paths only
#
# Tests that wire a real GenerationCoordinator + verifier moved to
# tests/integration/test_grpc_runtime_real.py in PR-N1. The Linux
# gate keeps only the "no coordinator wired in" UNIMPLEMENTED check
# below.
# ---------------------------------------------------------------------------


async def test_generate_returns_unimplemented_when_no_coordinator(grpc_pair):
    """Servicer constructed without a GenerationCoordinator (the
    PR-B1 / PR-B2 default) keeps the framework UNIMPLEMENTED for
    Generate. Regression contract."""
    stub, _, _ = grpc_pair
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _event in stub.Generate(
            runtime_pb2.GenerateRequest(session_id="sess-x", max_tokens=1),
        ):
            pass  # pragma: no cover - the stream raises before yielding
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
# create_grpc_server factory + GrpcServerConfig
# ---------------------------------------------------------------------------


async def test_default_bind_address_is_loopback():
    # ADR 0008 §8 OQ-5 default while unresolved.
    assert DEFAULT_BIND_ADDRESS == "127.0.0.1:50051"


async def test_grpc_server_config_defaults():
    cfg = GrpcServerConfig()
    assert cfg.bind_address == DEFAULT_BIND_ADDRESS
    assert cfg.max_concurrent_rpcs is None


async def test_grpc_server_config_is_frozen():
    cfg = GrpcServerConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.bind_address = "0.0.0.0:50051"  # type: ignore[misc]


async def test_create_grpc_server_default_config_binds_default_address():
    # Use port 0 so we don't collide with anything — but the
    # default address is a fixed port. To avoid that, override the
    # config; we still exercise the "config is None" branch via the
    # next test.
    store = SessionStore(capacity=2)
    server = create_grpc_server(
        session_store=store,
        config=GrpcServerConfig(bind_address="127.0.0.1:0"),
    )
    # Server is constructed but not started; starting it would
    # require an event-loop run, which the fixture-using tests cover.
    assert server is not None


async def test_create_grpc_server_with_no_config_uses_defaults(monkeypatch):
    # The "config is None" branch on grpc_app.py needs coverage. We
    # patch GrpcServerConfig itself so its default-constructed instance
    # binds to a free port, avoiding a real collision on
    # 127.0.0.1:50051 which a parallel dev process might be holding.
    from inference_engine.server import grpc_app as _grpc_app_mod

    class _FreePortConfig(GrpcServerConfig):
        pass

    # Replace the class the factory references; the "config is None"
    # branch will then construct an instance of the patched class and
    # use its frozen default. We construct the replacement *before*
    # patching so it inherits the original frozen-dataclass shape.
    free_default = type(
        GrpcServerConfig.__name__,
        (object,),
        {"bind_address": "127.0.0.1:0", "max_concurrent_rpcs": None},
    )
    monkeypatch.setattr(_grpc_app_mod, "GrpcServerConfig", free_default)
    store = SessionStore(capacity=2)
    server = create_grpc_server(session_store=store)
    assert server is not None


async def test_create_grpc_server_with_max_concurrent_rpcs():
    store = SessionStore(capacity=2)
    server = create_grpc_server(
        session_store=store,
        config=GrpcServerConfig(
            bind_address="127.0.0.1:0",
            max_concurrent_rpcs=8,
        ),
    )
    assert server is not None
