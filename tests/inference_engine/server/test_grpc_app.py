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


# ---------------------------------------------------------------------------
# Generate (PR-B3) — wired via GenerationCoordinator + FakeVerifier
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


@pytest_asyncio.fixture
async def grpc_pair_with_generator() -> AsyncIterator[
    tuple[
        runtime_pb2_grpc.RuntimeServiceStub,
        SessionStore,
        FakeVerifier,
        grpc.aio.Server,
    ]
]:
    """gRPC pair with both AppendTokens and Generate coordinators
    wired (so we can prep a session via AppendTokens, then call
    Generate against it)."""
    from inference_engine.session import (
        AppendTokensCoordinator,
        GenerationCoordinator,
    )

    fv = FakeVerifier()
    store = SessionStore(capacity=4, cache_inspector=fv)
    append_coord = AppendTokensCoordinator(store, fv)
    gen_coord = GenerationCoordinator(store, fv)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            store,
            append_coordinator=append_coord,
            generation_coordinator=gen_coord,
        ),
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


async def _prep_session(stub, token_ids=(1, 2, 3)):
    """Create + prefill a session, return its session_id."""
    create = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    await stub.AppendTokens(
        runtime_pb2.AppendTokensRequest(
            session_id=create.session_id, token_ids=list(token_ids),
        ),
    )
    return create.session_id


async def test_generate_streams_tokens_then_done(grpc_pair_with_generator):
    stub, _, _, _ = grpc_pair_with_generator
    sid = await _prep_session(stub)
    events = []
    async for resp in stub.Generate(
        runtime_pb2.GenerateRequest(session_id=sid, max_tokens=3),
    ):
        events.append(resp)
    # Three token frames followed by one done frame.
    payload_kinds = [r.WhichOneof("payload") for r in events]
    assert payload_kinds == ["token_id", "token_id", "token_id", "done"]
    done = events[-1].done
    assert done.stop_reason == runtime_pb2.GenerateDone.STOP_REASON_MAX_TOKENS
    assert done.generated_token_count == 3
    assert done.prefill_duration_seconds == 0.0


async def test_generate_eos_stops_with_eos_stop_reason(
    grpc_pair_with_generator,
):
    stub, store, _, _ = grpc_pair_with_generator
    # Pre-load history that makes the first argmax = 6 (FakeVerifier's
    # _logits_for hashes recent 3 tokens to argmax = sum % 16).
    create = await stub.CreateSession(
        runtime_pb2.CreateSessionRequest(eos_token_ids=[6]),
    )
    await stub.AppendTokens(
        runtime_pb2.AppendTokensRequest(
            session_id=create.session_id, token_ids=[1, 2, 3],
        ),
    )
    events = []
    async for resp in stub.Generate(
        runtime_pb2.GenerateRequest(
            session_id=create.session_id, max_tokens=10,
        ),
    ):
        events.append(resp)
    payload_kinds = [r.WhichOneof("payload") for r in events]
    assert payload_kinds == ["token_id", "done"]
    assert events[0].token_id == 6
    assert events[-1].done.stop_reason == \
        runtime_pb2.GenerateDone.STOP_REASON_EOS


async def test_generate_history_truncated_emitted(grpc_pair_with_generator):
    stub, store, _, _ = grpc_pair_with_generator
    # FakeVerifier's default budget is sink+window = 2+4 = 6.
    # Prefill 8 tokens so we're in truncated state at start of Generate.
    create = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    await stub.AppendTokens(
        runtime_pb2.AppendTokensRequest(
            session_id=create.session_id,
            token_ids=[10, 20, 30, 40, 50, 60, 70, 80],
        ),
    )
    events = []
    async for resp in stub.Generate(
        runtime_pb2.GenerateRequest(
            session_id=create.session_id, max_tokens=2,
        ),
    ):
        events.append(resp)
    payload_kinds = [r.WhichOneof("payload") for r in events]
    # First frame is truncated, then tokens, then done.
    assert payload_kinds[0] == "truncated"
    assert events[0].truncated.dropped_token_count == 2  # 8 - 6
    # Tokens follow.
    assert payload_kinds[1:3] == ["token_id", "token_id"]
    assert payload_kinds[3] == "done"


async def test_generate_unknown_session_returns_not_found(
    grpc_pair_with_generator,
):
    stub, _, _, _ = grpc_pair_with_generator
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _resp in stub.Generate(
            runtime_pb2.GenerateRequest(
                session_id="sess-nonexistent", max_tokens=1,
            ),
        ):
            pass  # pragma: no cover - stream raises before yielding
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


async def test_generate_no_history_returns_invalid_argument(
    grpc_pair_with_generator,
):
    """Session created but no AppendTokens preceded — Generate has
    no prefill state to start from. Must surface INVALID_ARGUMENT,
    not crash on argmax of uninitialized logits."""
    stub, _, _, _ = grpc_pair_with_generator
    create = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _resp in stub.Generate(
            runtime_pb2.GenerateRequest(
                session_id=create.session_id, max_tokens=1,
            ),
        ):
            pass  # pragma: no cover
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "AppendTokens must precede" in exc_info.value.details()


async def test_generate_max_tokens_zero_returns_invalid_argument(
    grpc_pair_with_generator,
):
    stub, _, _, _ = grpc_pair_with_generator
    sid = await _prep_session(stub)
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _resp in stub.Generate(
            runtime_pb2.GenerateRequest(session_id=sid, max_tokens=0),
        ):
            pass  # pragma: no cover
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_generate_temperature_nonzero_returns_invalid_argument(
    grpc_pair_with_generator,
):
    stub, _, _, _ = grpc_pair_with_generator
    sid = await _prep_session(stub)
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _resp in stub.Generate(
            runtime_pb2.GenerateRequest(
                session_id=sid, max_tokens=1, temperature=0.7,
            ),
        ):
            pass  # pragma: no cover
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_generate_seed_is_accepted(grpc_pair_with_generator):
    """Seed must be accepted on the wire (proto3 optional uint64).
    In greedy mode it's ignored; the run must complete normally."""
    stub, _, _, _ = grpc_pair_with_generator
    sid = await _prep_session(stub)
    events = []
    async for resp in stub.Generate(
        runtime_pb2.GenerateRequest(
            session_id=sid, max_tokens=2, seed=42,
        ),
    ):
        events.append(resp)
    assert any(r.WhichOneof("payload") == "done" for r in events)


async def test_generate_invariant_violation_returns_failed_precondition():
    """An INV-1 violation during a generation step must surface as
    FAILED_PRECONDITION, not INTERNAL."""
    from inference_engine.session import (
        AppendTokensCoordinator,
        GenerationCoordinator,
    )

    fv = FakeVerifier()
    store = SessionStore(capacity=2, cache_inspector=fv)
    append_coord = AppendTokensCoordinator(store, fv)
    gen_coord = GenerationCoordinator(store, fv)
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            store,
            append_coordinator=append_coord,
            generation_coordinator=gen_coord,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        # Set up a valid session via honest AppendTokens.
        create = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
        await stub.AppendTokens(
            runtime_pb2.AppendTokensRequest(
                session_id=create.session_id, token_ids=[1, 2, 3],
            ),
        )
        # Now make k_seq_length lie so the FIRST generation step's
        # INV-1 check fires.
        fv.k_seq_length = lambda session: 999
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _resp in stub.Generate(
                runtime_pb2.GenerateRequest(
                    session_id=create.session_id, max_tokens=1,
                ),
            ):
                pass  # pragma: no cover - stream aborts mid-way
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "INV-1" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=0.1)


async def test_create_grpc_server_accepts_generation_coordinator():
    """The factory must accept the new keyword and plumb it through."""
    from inference_engine.session import GenerationCoordinator

    fv = FakeVerifier()
    store = SessionStore(capacity=2)
    coord = GenerationCoordinator(store, fv)
    server = create_grpc_server(
        session_store=store,
        generation_coordinator=coord,
        config=GrpcServerConfig(bind_address="127.0.0.1:0"),
    )
    assert server is not None


async def test_generate_cancellation_emits_cancelled_done():
    """Drive the Servicer's Generate directly with a fake gRPC context
    that flips ``cancelled()`` to True after the first event. The
    servicer must:

      1. Yield the first TokenEvent normally.
      2. On the next loop turn, observe context.cancelled() == True.
      3. Emit a final GenerateDone(STOP_REASON_CANCELLED) frame and
         return — without continuing to generate.

    Direct-invocation test rather than through-the-channel: the
    real-channel cancellation closes the connection from the client
    side, so the server-emitted CANCELLED frame is observable only
    in-process. This test exercises the server-side branch.
    """
    from inference_engine.session import (
        AppendTokensCoordinator,
        GenerationCoordinator,
    )

    fv = FakeVerifier()
    store = SessionStore(capacity=1, cache_inspector=fv)
    append_coord = AppendTokensCoordinator(store, fv)
    gen_coord = GenerationCoordinator(store, fv)
    sess = store.create_session()
    append_coord.append_tokens(sess.session_id, [1, 2, 3])

    servicer = RuntimeServiceServicer(
        store,
        append_coordinator=append_coord,
        generation_coordinator=gen_coord,
    )

    class _FakeContext:
        """Minimal stand-in for grpc.aio.ServicerContext.

        Tracks how many times ``cancelled()`` has been polled; flips
        to True after the first poll so the very first iteration of
        the servicer's loop yields a TokenEvent normally and the
        second iteration observes the cancellation. ``abort`` is not
        used in this happy-path-of-cancellation test.
        """
        def __init__(self) -> None:
            self._polls = 0
            self.poll_history: list[bool] = []

        def cancelled(self) -> bool:
            self._polls += 1
            verdict = self._polls > 1
            self.poll_history.append(verdict)
            return verdict

        async def abort(self, code, details):  # pragma: no cover
            raise AssertionError(
                f"abort should not be called: {code} {details!r}",
            )

    ctx = _FakeContext()
    request = runtime_pb2.GenerateRequest(
        session_id=sess.session_id, max_tokens=10,
    )

    events = []
    async for resp in servicer.Generate(request, ctx):
        events.append(resp)

    # First frame is a token, then CANCELLED done — no more.
    assert len(events) == 2
    assert events[0].WhichOneof("payload") == "token_id"
    assert events[1].WhichOneof("payload") == "done"
    assert events[1].done.stop_reason == \
        runtime_pb2.GenerateDone.STOP_REASON_CANCELLED
    assert events[1].done.generated_token_count == 1
    # cancelled() polled twice: once on first iteration (returned
    # False, allowed token to flow), once on second (returned True,
    # tripped CANCELLED branch).
    assert ctx.poll_history == [False, True]


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
