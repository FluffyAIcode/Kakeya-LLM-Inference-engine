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

import asyncio
import inspect
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import grpc

from inference_engine.memory.pool import PoolExhausted
from inference_engine.distributed.prefill_cache_runtime import (
    RemotePrefillRequiredError,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from inference_engine.session import (
    AppendTokensCoordinator,
    DoneEvent,
    GenerationCoordinator,
    HistoryTruncatedEvent,
    InvariantViolation,
    OperationCancelledError,
    SessionNotFoundError,
    SessionGenerationBusyError,
    SessionStore,
    STOP_REASON_CANCELLED,
    STOP_REASON_EOS,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_TRUNCATED,
    TokenEvent,
)


# Mapping from GenerationCoordinator's string stop reasons to the
# protobuf enum. Defined at module level so reviewers can audit the
# 1:1 correspondence at a glance.
_STOP_REASON_TO_PROTO = {
    STOP_REASON_MAX_TOKENS:
        runtime_pb2.GenerateDone.STOP_REASON_MAX_TOKENS,
    STOP_REASON_EOS:
        runtime_pb2.GenerateDone.STOP_REASON_EOS,
    STOP_REASON_CANCELLED:
        runtime_pb2.GenerateDone.STOP_REASON_CANCELLED,
    STOP_REASON_TRUNCATED:
        runtime_pb2.GenerateDone.STOP_REASON_TRUNCATED,
}

_logger = logging.getLogger(__name__)
_STREAM_END = object()


class _ThreadedEventStream:
    """Drain a synchronous model iterator without blocking asyncio."""

    def __init__(self, stream, cancel_event: threading.Event) -> None:
        self._stream = stream
        self._cancel_event = cancel_event
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run,
            name="kakeya-generate",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _put(self, value) -> None:
        while True:
            try:
                self._queue.put(value, timeout=0.1)
                return
            except queue.Full:
                if self._cancel_event.is_set():
                    return

    def _run(self) -> None:
        try:
            for event in self._stream:
                self._put((event, None))
                if self._cancel_event.is_set():
                    break
        except BaseException as exc:
            self._put((None, exc))
        finally:
            self._put((_STREAM_END, None))

    async def next(self):
        return await asyncio.to_thread(self._queue.get)


def _accepts_cancel_event(method) -> bool:
    parameters = inspect.signature(method).parameters.values()
    return any(
        parameter.name == "cancel_event"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )

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
        generation_coordinator: Optional[GenerationCoordinator] = None,
        memory_governor=None,
        liveness=None,
    ) -> None:
        """Construct a Servicer.

        ``append_coordinator`` is the PR-B2 wiring point: when None
        (PR-B1 mode, preserved for tests that don't need a verifier),
        ``AppendTokens`` returns ``UNIMPLEMENTED``; when non-None,
        ``AppendTokens`` runs the §2.3 byte-exact prefill-incremental
        contract.

        ``generation_coordinator`` is the PR-B3 wiring point: same
        optional-default contract for ``Generate``. When None, the
        Generate stream returns ``UNIMPLEMENTED``; when non-None,
        Generate streams TokenEvents / HistoryTruncatedEvents /
        DoneEvent through the gRPC server-streaming response.
        """
        self._store = session_store
        self._append = append_coordinator
        self._generate = generation_coordinator
        self._memory_governor = memory_governor
        self._liveness = liveness
        self._operation_lock = threading.Lock()
        self._active_operations: set[str] = set()

    def _acquire_operation(self, session_id: str) -> bool:
        with self._operation_lock:
            if session_id in self._active_operations:
                return False
            self._active_operations.add(session_id)
            return True

    def _release_operation(self, session_id: str) -> None:
        with self._operation_lock:
            self._active_operations.discard(session_id)

    def _watch_context(
        self,
        context,
        cancel_event: threading.Event,
        session_id: str,
        completed_event: threading.Event | None = None,
    ) -> None:
        callback = getattr(context, "add_done_callback", None)
        if callback is not None:
            def done(done_context) -> None:
                logically_completed = (
                    completed_event is not None and completed_event.is_set()
                )
                if done_context.cancelled() and not logically_completed:
                    cancel_event.set()
                    self._store.remove_session_if_present(
                        session_id, reason="client_cancelled",
                    )
            callback(done)

    async def CreateSession(  # noqa: N802 — gRPC-generated method casing
        self,
        request: runtime_pb2.CreateSessionRequest,
        context: grpc.aio.ServicerContext,
    ) -> runtime_pb2.CreateSessionResponse:
        """Allocate a new session; return its server-issued id.

        ADR 0008 §2.2 contract item 1: clients have no input on the
        ``session_id`` value; this RPC is the only producer.
        """
        if self._memory_governor is not None and self._memory_governor.draining:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Primary runtime is draining due to unified-memory pressure",
            )
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
        if not self._acquire_operation(request.session_id):
            await context.abort(
                grpc.StatusCode.ABORTED,
                f"session {request.session_id!r} already has an active operation",
            )
        cancel_event = threading.Event()
        completed_event = threading.Event()
        self._watch_context(
            context,
            cancel_event,
            request.session_id,
            completed_event,
        )
        try:
            kwargs = {
                "session_id": request.session_id,
                "token_ids": list(request.token_ids),
            }
            if _accepts_cancel_event(self._append.append_tokens):
                kwargs["cancel_event"] = cancel_event
            new_history_length = await asyncio.to_thread(
                self._append.append_tokens, **kwargs,
            )
            completed_event.set()
        except asyncio.CancelledError:
            cancel_event.set()
            self._store.remove_session_if_present(
                request.session_id, reason="client_cancelled",
            )
            raise
        except SessionNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except RemotePrefillRequiredError as exc:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except InvariantViolation as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except OperationCancelledError:
            self._store.remove_session_if_present(
                request.session_id, reason="client_cancelled",
            )
            await context.abort(grpc.StatusCode.CANCELLED, "AppendTokens cancelled")
        finally:
            self._release_operation(request.session_id)
            if self._liveness is not None:
                self._liveness.update("idle")
        return runtime_pb2.AppendTokensResponse(
            history_length=new_history_length,
        )

    async def Generate(  # noqa: N802 — gRPC-generated method casing
        self,
        request: runtime_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ):
        """Stream tokens generated against ``request.session_id``.

        Yields ``runtime_pb2.GenerateResponse`` frames carrying one of:

          * ``token_id``: a committed token, in generation order.
          * ``truncated``: ``HistoryTruncated`` event, emitted at most
            once per call before the first ``token_id`` (per the proto
            contract).
          * ``done``: ``GenerateDone`` terminal frame.

        When this Servicer was constructed without a
        ``generation_coordinator``, returns ``UNIMPLEMENTED`` (PR-B2
        regression contract preserved).

        Cancellation: the loop polls ``context.cancelled()`` after
        every event the coordinator yields. On cancellation we emit
        a ``GenerateDone(STOP_REASON_CANCELLED)`` frame and return.
        Cancellation latency is bounded by one generation step on
        the worst case (the in-flight forward pass finishes before
        the next poll).
        """
        if self._generate is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Generate not configured on this Servicer "
                "(coordinator not provided)",
            )

        seed = request.seed if request.HasField("seed") else None
        temperature = (
            request.temperature
            if request.HasField("temperature") else None
        )
        top_p = request.top_p if request.HasField("top_p") else None
        top_k = request.top_k if request.HasField("top_k") else None

        if not self._acquire_operation(request.session_id):
            await context.abort(
                grpc.StatusCode.ABORTED,
                f"session {request.session_id!r} already has an active operation",
            )
        cancel_event = threading.Event()
        completed_event = threading.Event()
        self._watch_context(
            context,
            cancel_event,
            request.session_id,
            completed_event,
        )
        kwargs = {
            "session_id": request.session_id,
            "max_tokens": request.max_tokens,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        if _accepts_cancel_event(self._generate.generate):
            kwargs["cancel_event"] = cancel_event
        event_stream = self._generate.generate(**kwargs)
        threaded_stream = _ThreadedEventStream(event_stream, cancel_event)
        threaded_stream.start()

        token_count_so_far = 0

        try:
            while True:
                event, error = await threaded_stream.next()
                if error is not None:
                    raise error
                if event is _STREAM_END:
                    completed_event.set()
                    break
                if context.cancelled():
                    threaded_stream.cancel()
                    self._store.remove_session_if_present(
                        request.session_id, reason="client_cancelled",
                    )
                    return

                if isinstance(event, TokenEvent):
                    token_count_so_far += 1
                    yield runtime_pb2.GenerateResponse(
                        token_id=event.token_id,
                    )
                elif isinstance(event, HistoryTruncatedEvent):
                    yield runtime_pb2.GenerateResponse(
                        truncated=runtime_pb2.HistoryTruncated(
                            dropped_token_count=event.dropped_token_count,
                        ),
                    )
                else:
                    # DoneEvent — the only remaining event type per
                    # the GenerateEvent union.
                    assert isinstance(event, DoneEvent)
                    completed_event.set()
                    yield runtime_pb2.GenerateResponse(
                        done=runtime_pb2.GenerateDone(
                            stop_reason=_STOP_REASON_TO_PROTO[
                                event.stop_reason
                            ],
                            generated_token_count=event.generated_token_count,
                            prefill_duration_seconds=event.prefill_seconds,
                            total_duration_seconds=event.total_seconds,
                        ),
                    )
        except SessionNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except InvariantViolation as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except SessionGenerationBusyError as exc:
            await context.abort(grpc.StatusCode.ABORTED, str(exc))
        except asyncio.CancelledError:
            threaded_stream.cancel()
            if not completed_event.is_set():
                self._store.remove_session_if_present(
                    request.session_id, reason="client_cancelled",
                )
            raise
        finally:
            threaded_stream.cancel()
            self._release_operation(request.session_id)

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
    generation_coordinator: Optional[GenerationCoordinator] = None,
    config: Optional[GrpcServerConfig] = None,
    memory_governor=None,
    liveness=None,
    capability_registry: Optional[object] = None,
    proposers: Optional[object] = None,
    default_proposer_model_id: str = "",
    prefill_cache_store: Optional[object] = None,
    prefill_cache_address: str = "",
    prefill_auth: Optional[object] = None,
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

    ``capability_registry`` / ``proposers`` are the ADR 0009 v0.5-M1
    multi-host plane wiring points. Pass a
    :class:`~inference_engine.distributed.capability.CapabilityRegistry`
    to additionally serve ``kakeya.v1.CapabilityService`` on the same
    port, and/or a non-empty ``{model_id: proposer}`` mapping to serve
    ``kakeya.v1.ProposerService``. Both default to off so a v0.3-style
    single-host runtime is byte-for-byte unchanged. The parameters are
    typed loosely (``object``) so this module keeps no import-time
    dependency on the distributed subpackage when the plane is off;
    the imports happen lazily at wiring time.

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
            session_store,
            append_coordinator=append_coordinator,
            generation_coordinator=generation_coordinator,
            memory_governor=memory_governor,
            liveness=liveness,
        ),
        server,
    )
    if capability_registry is not None:
        from inference_engine.distributed.exchange import add_capability_service

        add_capability_service(server, capability_registry)
        _logger.info("gRPC CapabilityService enabled (ADR 0009)")
    if proposers:
        from inference_engine.distributed.proposer_service import (
            add_proposer_service,
        )

        add_proposer_service(
            server, proposers, default_model_id=default_proposer_model_id,
        )
        _logger.info(
            "gRPC ProposerService enabled for models: %s", sorted(proposers),
        )
    if prefill_cache_store is not None:
        from inference_engine.distributed.prefill_cache_service import (
            add_prefill_cache_service,
        )

        add_prefill_cache_service(
            server,
            prefill_cache_store,
            cache_address=prefill_cache_address or config.bind_address,
            auth=prefill_auth,
        )
        _logger.info(
            "gRPC PrefillCacheService enabled at %s",
            prefill_cache_address or config.bind_address,
        )
    server.add_insecure_port(config.bind_address)
    _logger.info("gRPC RuntimeService bound to %s", config.bind_address)
    return server
