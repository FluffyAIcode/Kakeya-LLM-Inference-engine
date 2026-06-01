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
# AppendTokens (PR-B2) — wired via AppendTokensCoordinator + FakeVerifier
# ---------------------------------------------------------------------------


# Reuse the FakeVerifier from the coordinator test module rather than
# re-defining it here. The fake is itself fully tested in
# tests/inference_engine/session/test_coordinator.py, so importing it
# here just exercises the gRPC ↔ coordinator wiring on top.
from tests.inference_engine.session.test_coordinator import FakeVerifier  # noqa: E402


@pytest_asyncio.fixture
async def grpc_pair_with_appender() -> AsyncIterator[
    tuple[
        runtime_pb2_grpc.RuntimeServiceStub,
        SessionStore,
        FakeVerifier,
        grpc.aio.Server,
    ]
]:
    """gRPC pair where the Servicer has an AppendTokensCoordinator
    wired in. Yields ``(stub, store, verifier, server)``."""
    from inference_engine.session import AppendTokensCoordinator

    fv = FakeVerifier()
    store = SessionStore(capacity=4, cache_inspector=fv)
    coordinator = AppendTokensCoordinator(store, fv)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coordinator),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        yield stub, store, fv, server
    finally:
        await channel.close()
        await server.stop(grace=0.1)


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


async def test_append_tokens_first_call_triggers_prefill(
    grpc_pair_with_appender,
):
    stub, store, fv, _ = grpc_pair_with_appender
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    resp = await stub.AppendTokens(
        runtime_pb2.AppendTokensRequest(
            session_id=create_resp.session_id,
            token_ids=[10, 20, 30],
        ),
    )
    assert resp.history_length == 3
    # Verifier saw a prefill, not a forward_block (cold cache path).
    kinds = [c[0] for c in fv.call_log]
    assert kinds == ["prefill"]
    sess = store.get_session(create_resp.session_id)
    assert sess.history_token_ids == [10, 20, 30]
    assert sess.next_global_position == 3


async def test_append_tokens_subsequent_call_triggers_incremental(
    grpc_pair_with_appender,
):
    stub, store, fv, _ = grpc_pair_with_appender
    create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    await stub.AppendTokens(
        runtime_pb2.AppendTokensRequest(
            session_id=create_resp.session_id, token_ids=[10, 20, 30],
        ),
    )
    resp = await stub.AppendTokens(
        runtime_pb2.AppendTokensRequest(
            session_id=create_resp.session_id, token_ids=[40, 50],
        ),
    )
    assert resp.history_length == 5
    kinds = [c[0] for c in fv.call_log]
    assert kinds == ["prefill", "forward_block", "commit_or_truncate"]
    assert fv.call_log[1] == ("forward_block", (40, 50))
    assert fv.call_log[2] == ("commit_or_truncate", 2, 2)


async def test_append_tokens_unknown_session_returns_not_found(
    grpc_pair_with_appender,
):
    stub, _, _, _ = grpc_pair_with_appender
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.AppendTokens(
            runtime_pb2.AppendTokensRequest(
                session_id="sess-nonexistent", token_ids=[1, 2, 3],
            ),
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    assert "sess-nonexistent" in exc_info.value.details()


async def test_append_tokens_invariant_violation_returns_failed_precondition():
    """Construct an AppendTokensCoordinator whose verifier triggers
    INV-1 on the first append; the gRPC servicer must surface it as
    FAILED_PRECONDITION (not as INTERNAL or NOT_FOUND)."""
    from inference_engine.session import AppendTokensCoordinator

    class _LyingFakeVerifier(FakeVerifier):
        def k_seq_length(self, session):
            del session
            return 999  # never matches anything the session reports

    fv = _LyingFakeVerifier()
    store = SessionStore(capacity=2, cache_inspector=fv)
    coord = AppendTokensCoordinator(store, fv)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coord), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=create_resp.session_id, token_ids=[1, 2, 3],
                ),
            )
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "INV-1" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_append_tokens_value_error_returns_invalid_argument():
    """Construct an AppendTokensCoordinator that surfaces a ValueError
    from the well-formedness check. We can't push a negative through
    the wire (uint32 blocks it), so we exercise this by manually
    invoking the servicer at the AppendTokensCoordinator level via a
    coordinator that re-raises ValueError on contact.

    The gRPC servicer must surface the ValueError as
    INVALID_ARGUMENT, not as INTERNAL.
    """
    from inference_engine.session import AppendTokensCoordinator

    class _ValueErroringCoordinator(AppendTokensCoordinator):
        def append_tokens(self, session_id, token_ids):
            del token_ids
            # Look up the session first so SessionNotFoundError still
            # wins for unknown ids — only raise ValueError for known
            # sessions, mirroring the real coordinator's order.
            self._store.get_session(session_id)
            raise ValueError("synthetic well-formedness violation")

    fv = FakeVerifier()
    store = SessionStore(capacity=2)
    coord = _ValueErroringCoordinator(store, fv)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coord), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        create_resp = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
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


# ---------------------------------------------------------------------------
# Generate not yet implemented in PR-B2
# ---------------------------------------------------------------------------


async def test_generate_returns_unimplemented(grpc_pair):
    """Generate lands in PR-B3. The framework default UNIMPLEMENTED
    must hold until that PR explicitly overrides the method."""
    stub, _, _ = grpc_pair
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _event in stub.Generate(
            runtime_pb2.GenerateRequest(session_id="sess-x", max_tokens=1),
        ):
            pass  # pragma: no cover - the stream raises before yielding
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED


# ---------------------------------------------------------------------------
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


async def test_create_grpc_server_accepts_append_coordinator():
    """PR-B2 wiring: the factory must accept an append_coordinator
    keyword and plumb it into the Servicer."""
    from inference_engine.session import AppendTokensCoordinator

    fv = FakeVerifier()
    store = SessionStore(capacity=2)
    coord = AppendTokensCoordinator(store, fv)
    server = create_grpc_server(
        session_store=store,
        append_coordinator=coord,
        config=GrpcServerConfig(bind_address="127.0.0.1:0"),
    )
    assert server is not None
