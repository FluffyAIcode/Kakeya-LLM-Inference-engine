"""gRPC runtime service for Kakeya — ADR 0008 PR-B1 (Phase B).

Implements three RPCs from `proto/kakeya/v1/runtime.proto` against
the :class:`SessionStore` from PR-A2 / PR-A3b:

- ``CreateSession``
- ``CloseSession``
- ``GetSessionInfo``

The remaining RPCs (``AppendTokens``, ``Generate``) are intentionally
*not* implemented in this PR; calling them returns
``UNIMPLEMENTED`` (the gRPC framework's default for un-overridden
servicer methods, which is the correct stop-gap behavior per ADR
0008 §2.10 "no graceful degradation"). They land in PR-B2 / PR-B3.

The server is **asyncio-based** (``grpc.aio``) per ADR 0008 §2.5:
all RPCs run on a single event loop and serialize SessionStore
access at the asyncio layer. The store itself is single-threaded;
multi-worker support is v0.4 scope (ADR 0008 §4.5).

This module deliberately does *not* depend on FastAPI or the
deprecated HTTP shim. The two surfaces share only the
``SessionStore`` (or, in the deprecated shim's case, will share it
once PR-D1 lands). They can be started together by the same CLI
entry point or separately; the factory ``create_grpc_server`` is
self-contained so any of those wirings is possible without code
change here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import grpc

from inference_engine.memory.pool import PoolExhausted
from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from inference_engine.session import (
    AppendTokensCoordinator,
    InvariantViolation,
    SessionNotFoundError,
    SessionStore,
)

_logger = logging.getLogger(__name__)

DEFAULT_BIND_ADDRESS = "127.0.0.1:50051"
"""Default bind address for the gRPC server.

Per ADR 0008 §8 OQ-5 default while unresolved: bind to loopback
only. Multi-tenant deployments that need to reach the runtime from
another host configure a different bind address explicitly and add
the appropriate auth (also OQ-5 scope; v0.4)."""


@dataclass(frozen=True)
class GrpcServerConfig:
    """Configuration for the gRPC runtime server.

    Kept as a frozen dataclass so the configuration is auditable at
    construction time and accidentally re-binding mid-session is a
    structural impossibility.
    """

    bind_address: str = DEFAULT_BIND_ADDRESS
    """`host:port` to bind. Defaults to loopback per ADR 0008 §8 OQ-5."""

    max_concurrent_rpcs: Optional[int] = None
    """Per-server cap on in-flight RPCs.

    ``None`` defers to grpc.aio's default. PR-B1 leaves this
    unconstrained because the per-session concurrency cap (ADR 0008
    §2.5 ``max_concurrent``) is enforced at the SessionStore /
    Generate level, not at the RPC dispatch level. Set explicitly
    when running on a constrained host where the gRPC handler
    threads themselves are the bottleneck."""


class RuntimeServiceServicer(runtime_pb2_grpc.RuntimeServiceServicer):
    """RuntimeService implementation backed by a :class:`SessionStore`.

    Error mapping (per ADR 0008 §2.6 / §2.10 — every typed
    SessionStore error becomes a typed gRPC status, no silent
    fallback):

    +-----------------------------+----------------------------------+
    | SessionStoreError subclass  | gRPC status                      |
    +=============================+==================================+
    | SessionNotFoundError        | NOT_FOUND                        |
    +-----------------------------+----------------------------------+
    | InvariantViolation          | FAILED_PRECONDITION              |
    +-----------------------------+----------------------------------+
    | (PoolExhausted from pool)   | RESOURCE_EXHAUSTED               |
    +-----------------------------+----------------------------------+
    | ValueError (token-id range) | INVALID_ARGUMENT                 |
    +-----------------------------+----------------------------------+

    PR-B2 adds the AppendTokens RPC which can raise the full set of
    errors above (``InvariantViolation`` from INV-1 / INV-2 mismatch
    detected during the prefill-incremental path, ``ValueError`` from
    well-formedness checks on token ids).

    Generate is still UNIMPLEMENTED in PR-B2; PR-B3 wires it.
    """

    def __init__(
        self,
        session_store: SessionStore,
        *,
        append_coordinator: Optional[AppendTokensCoordinator] = None,
    ) -> None:
        """Construct a Servicer.

        ``append_coordinator`` is the wiring point added in PR-B2. When
        ``None`` (the PR-B1 mode, preserved for tests that don't need a
        verifier), ``AppendTokens`` returns ``UNIMPLEMENTED`` — the same
        framework default used in PR-B1. When non-None, ``AppendTokens``
        runs the §2.3 byte-exact prefill-incremental contract through
        the coordinator and surfaces the typed error mapping above.
        """
        self._store = session_store
        self._append = append_coordinator

    async def CreateSession(  # noqa: N802 — gRPC-generated method casing
        self,
        request: runtime_pb2.CreateSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> runtime_pb2.CreateSessionResponse:
        """Allocate a new session; return its server-issued id.

        ADR 0008 §2.2 contract item 1: clients have no input on the
        ``session_id`` value; this RPC is the only producer.
        """
        try:
            session = self._store.create_session(
                eos_token_ids=list(request.eos_token_ids),
                client_label=request.client_label,
            )
        except PoolExhausted as exc:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"slab pool exhausted: {exc}",
            )
        return runtime_pb2.CreateSessionResponse(
            session_id=session.session_id,
        )

    async def AppendTokens(  # noqa: N802 — gRPC-generated method casing
        self,
        request: runtime_pb2.AppendTokensRequest,
        context: grpc.aio.ServicerContext,
    ) -> runtime_pb2.AppendTokensResponse:
        """Append raw tokens; run the §2.3 byte-exact prefill-incremental.

        When this Servicer was constructed without an
        ``append_coordinator`` (the PR-B1 mode), this returns
        ``UNIMPLEMENTED`` — identical to a non-overridden gRPC method.
        With a coordinator attached, this implements the full §2.3
        contract and surfaces the four typed status mappings.
        """
        if self._append is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "AppendTokens not configured on this Servicer "
                "(coordinator not provided)",
            )
        try:
            new_history_length = self._append.append_tokens(
                session_id=request.session_id,
                token_ids=list(request.token_ids),
            )
        except SessionNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except InvariantViolation as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return runtime_pb2.AppendTokensResponse(
            history_length=new_history_length,
        )

    async def CloseSession(  # noqa: N802
        self,
        request: runtime_pb2.CloseSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> runtime_pb2.CloseSessionResponse:
        """Close a session and return its final history length.

        Returns NOT_FOUND if the session is unknown (closed,
        evicted, never existed — caller cannot distinguish, by
        ADR 0008 §2.6 design).
        """
        try:
            final_length = self._store.close_session(request.session_id)
        except SessionNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return runtime_pb2.CloseSessionResponse(
            final_history_length=final_length,
        )

    async def GetSessionInfo(  # noqa: N802
        self,
        request: runtime_pb2.GetSessionInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> runtime_pb2.GetSessionInfoResponse:
        """Return diagnostic counters for a session.

        Surfaces ADR 0008 §2.8's anomaly-invariant counters; healthy
        operation reports zero for both INV-1 and INV-2. Non-zero
        values are paging-grade — the session itself has by then
        been removed from the store, and a follow-up
        ``GetSessionInfo`` on the same id will return NOT_FOUND.
        """
        try:
            session = self._store.get_session(request.session_id)
        except SessionNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return runtime_pb2.GetSessionInfoResponse(
            history_length=session.history_length,
            kv_live_bytes=session.kv_live_bytes(),
            cache_invariant_inv1_violations=session.inv1_violations,
            cache_invariant_inv2_violations=session.inv2_violations,
            idle_seconds=session.idle_seconds,
        )


def create_grpc_server(
    *,
    session_store: SessionStore,
    append_coordinator: Optional[AppendTokensCoordinator] = None,
    config: Optional[GrpcServerConfig] = None,
) -> grpc.aio.Server:
    """Build, but do not start, a configured gRPC asyncio server.

    The caller invokes ``await server.start()`` and ``await
    server.wait_for_termination()`` (or ``await server.stop(grace)``
    for shutdown). This split is intentional: tests construct
    servers without starting them, and the eventual production
    entry point may want to wire signal handlers between
    construction and start.

    ``append_coordinator`` is the PR-B2 wiring point. Pass an
    :class:`AppendTokensCoordinator` to enable AppendTokens; omit
    (or pass ``None``) to leave AppendTokens at its PR-B1
    UNIMPLEMENTED default — useful for tests that don't need a
    verifier instance.

    The bound port is observable via the returned server's
    ``add_insecure_port`` return value; callers that need the port
    should use the lower-level ``grpc.aio.server()`` directly,
    because PR-B1 returns the constructed server with the port
    already bound (so the asyncio event loop sees the listen socket
    immediately).
    """
    if config is None:
        config = GrpcServerConfig()
    server = grpc.aio.server(
        maximum_concurrent_rpcs=config.max_concurrent_rpcs,
    )
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            session_store, append_coordinator=append_coordinator,
        ),
        server,
    )
    server.add_insecure_port(config.bind_address)
    _logger.info("gRPC RuntimeService bound to %s", config.bind_address)
    return server
