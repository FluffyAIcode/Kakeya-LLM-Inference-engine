"""FastAPI app factory and route handlers (PR-D2 of ADR 0008 Phase D).

The HTTP shim is **deprecated** per ADR 0008 §2.7 and slated for
retirement once OpenAI-API consumers migrate to the v0.3 gRPC
surface. PR-D2 refactored this module's internals to drive the
session-bound runtime (``SessionStore`` +
:class:`AppendTokensCoordinator` + :class:`GenerationCoordinator`)
directly, retiring the previous ``Scheduler`` + ``PooledVerifier``
+ :class:`SpeculativeEngine` machinery. Each ``/v1/chat/completions``
request is now a single-shot session: ``CreateSession`` →
``AppendTokens(prompt)`` → ``Generate`` → ``CloseSession`` —
identical semantics to the gRPC ``RuntimeService`` surface.

What this means for users
-------------------------

* **Speculative decoding is no longer applied on the HTTP path.**
  The session-bound runtime is pure autoregressive against the
  verifier; the proposer is wired into the v0.4 alignment work
  (ADR 0004). For now the HTTP shim is roughly the same speed as
  ``transformers``-vanilla AR generation. **Migrate to gRPC** for
  the v0.3 architecture's full perf story.
* Every response carries ``Deprecation: true`` and a
  ``Sunset`` header pointing to the v0.3 GA tag. The OpenAI clients
  ignore these by default but the metadata is in the response for
  proxies / observability tools.
* Admission control is now an :class:`asyncio.Semaphore` instead of
  a full ``Scheduler`` — the queueing and timeout semantics are
  preserved (REJECT vs QUEUE policy with ``queue_max_wait_s``) but
  the in-flight slab-pool bookkeeping moved into ``SessionStore``.

Routes
------

    GET  /healthz
    GET  /metrics
    GET  /v1/models
    POST /v1/chat/completions

Error mapping
-------------

* Pydantic validation errors: 422.
* Tokenizer chat-template rejection: 400.
* Tokenizer with no EOS: 500.
* Pool / admission saturation: 429 with OpenAI error envelope.
* Verifier raises mid-generate: 500 (non-streaming) or terminal
  SSE chunk with ``finish_reason="stop"`` (streaming).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.config import AdmissionPolicy
from inference_engine.server.auth import verify_api_key
from inference_engine.server.config import ServerConfig
from inference_engine.server.errors import (
    build_error_envelope,
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from inference_engine.server.metrics import Metrics
from inference_engine.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
    HealthResponse,
    ListModelsResponse,
    ModelInfo,
)
from inference_engine.server.streaming import _StreamingDetokenizer
from inference_engine.server.tokenizer import resolve_eos_ids
from inference_engine.session import (
    AppendTokensCoordinator,
    DoneEvent,
    GenerationCoordinator,
    HistoryTruncatedEvent,
    SessionStore,
    STOP_REASON_EOS,
    TokenEvent,
)
from inference_engine.session.store import SessionNotFoundError


# Per ADR 0008 §2.7: every HTTP-shim response carries these headers.
# v0.3.0 final ships with the deprecation marker live; the Sunset
# date is cosmetic until a real cutover plan exists.
_DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Sunset": "Wed, 31 Dec 2025 00:00:00 GMT",
    "Link": (
        '</docs/adr/0008-session-bound-runtime-and-grpc-protocol.md>; '
        'rel="successor-version"; type="text/markdown"'
    ),
}


def create_app(
    verifier,
    config: ServerConfig,
    *,
    slab_pool: Optional[SlabPool] = None,
    model_id_label: Optional[str] = None,
) -> FastAPI:
    """Build a FastAPI app bound to a verifier + config.

    Parameters
    ----------
    verifier:
        Anything implementing the verifier protocol consumed by
        :class:`AppendTokensCoordinator` (i.e., :meth:`prefill`,
        :meth:`forward_block`, :meth:`commit_or_truncate`,
        :meth:`k_seq_length`, :meth:`kv_live_bytes`) plus a
        :attr:`tokenizer` attribute satisfying
        :class:`~inference_engine.server.tokenizer.Tokenizer`.
        In production this is a :class:`SinkWindowVerifier`.
    config:
        Process-wide :class:`ServerConfig`. ``max_concurrent``,
        ``admission_policy``, and ``queue_max_wait_s`` drive the
        per-app admission semaphore.
    slab_pool:
        Optional pre-built :class:`SlabPool`. If ``None``, a tiny
        placeholder pool is built for ``max_concurrent`` slots.
        The slab is a session-bookkeeping placeholder; the verifier
        owns the real KV tensors and writes byte counts onto the
        slab via PR-E1c's ``_sync_slab_bytes`` helper.
    model_id_label:
        Returned by ``/v1/models`` and embedded in every
        ``chat.completion`` payload's ``model`` field. Defaults to
        ``config.model_id_label``.
    """
    pool = slab_pool if slab_pool is not None else _build_placeholder_pool(
        config.max_concurrent,
    )
    if pool.total_count != config.max_concurrent:
        raise ValueError(
            f"slab_pool.total_count={pool.total_count} does not match "
            f"config.max_concurrent={config.max_concurrent}"
        )

    store = SessionStore(
        capacity=config.max_concurrent,
        cache_inspector=verifier,
        slab_pool=pool,
    )
    append_coord = AppendTokensCoordinator(store, verifier)
    gen_coord = GenerationCoordinator(store, verifier)

    metrics = Metrics.build()
    metrics.snapshot_scheduler(
        active=0, pool_in_use=0, pool_total=pool.total_count, pending=0,
        kv_live_bytes=0,
    )

    label = model_id_label or config.model_id_label

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # No long-lived worker tasks anymore — every request is a
        # single-shot session. Lifespan is a no-op other than the
        # context-manager protocol the framework needs.
        yield

    app = FastAPI(
        title="Kakeya Inference Engine (HTTP shim, deprecated)",
        description=(
            "DEPRECATED OpenAI-compatible HTTP API. The v0.3 architecture "
            "is gRPC-first; see /docs/adr/0008-session-bound-runtime-"
            "and-grpc-protocol.md. This shim is feature-frozen, pure-AR "
            "(no speculative decoding), and slated for removal once "
            "consumers migrate."
        ),
        version="0.3.0",
        lifespan=lifespan,
    )

    app.state.verifier = verifier
    app.state.config = config
    app.state.store = store
    app.state.append_coord = append_coord
    app.state.gen_coord = gen_coord
    app.state.pool = pool
    app.state.metrics = metrics
    app.state.model_id_label = label
    app.state.admission_sem = asyncio.Semaphore(config.max_concurrent)

    # OpenAI-shape error envelopes for HTTPException + 422 + 500.
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, request_validation_exception_handler,
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # API-key auth (no-op when config.api_keys is empty).
    app.add_middleware(_AuthMiddleware, valid_keys=config.api_keys)
    # Per-request timing + counter.
    app.add_middleware(_MetricsMiddleware, metrics=metrics)
    # ADR 0008 §2.7 Deprecation / Sunset headers on every response.
    app.add_middleware(_DeprecationHeadersMiddleware)

    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class _AuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token gate for ``/v1/*`` routes when api_keys is non-empty."""

    def __init__(self, app, *, valid_keys) -> None:
        super().__init__(app)
        self._valid_keys = frozenset(valid_keys)

    async def dispatch(self, request, call_next):
        try:
            verify_api_key(request, valid_keys=self._valid_keys)
        except StarletteHTTPException as exc:
            return await http_exception_handler(request, exc)
        return await call_next(request)


class _MetricsMiddleware(BaseHTTPMiddleware):
    """Records ``http_requests_total`` + duration histogram per request."""

    def __init__(self, app, *, metrics: Metrics) -> None:
        super().__init__(app)
        self._metrics = metrics

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = self._safe_path(request)
        self._metrics.record_http_request(
            method=request.method,
            path=path,
            status=response.status_code,
            duration_s=duration,
        )
        return response

    @staticmethod
    def _safe_path(request) -> str:
        route = request.scope.get("route")
        if route is not None and hasattr(route, "path"):
            return route.path
        return request.url.path


class _DeprecationHeadersMiddleware(BaseHTTPMiddleware):
    """Stamps ADR 0008 §2.7 deprecation headers onto every response."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for k, v in _DEPRECATION_HEADERS.items():
            response.headers[k] = v
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_placeholder_pool(num_slabs: int) -> SlabPool:
    """Construct a tiny ``SlabPool`` for session bookkeeping.

    The slab is a placeholder; PR-E1c's :func:`_sync_slab_bytes`
    writes the verifier's real KV byte count onto each slab's
    ``live_kv_bytes_override`` after every coordinator mutation,
    so :meth:`Session.kv_live_bytes` reports physically meaningful
    values without the slab actually holding the K/V tensors.
    """
    cfg = SlabConfig(
        num_layers=1, num_heads=1, sink_size=0, window_size=1,
        head_dim=1, dtype=torch.bfloat16,
    )
    return SlabPool(num_slabs=num_slabs, slab_config=cfg)


def _encode_prompt(verifier, req: ChatCompletionRequest) -> List[int]:
    """Apply the verifier's tokenizer's chat template to the request."""
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = verifier.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    if not isinstance(prompt_ids, list) or not all(
        isinstance(t, int) for t in prompt_ids
    ):
        raise ValueError(
            f"chat template returned {type(prompt_ids).__name__}, expected list[int]"
        )
    if not prompt_ids:
        raise ValueError("chat template produced an empty token sequence")
    return prompt_ids


async def _admit(
    *,
    sem: asyncio.Semaphore,
    config: ServerConfig,
) -> None:
    """Acquire the admission semaphore per the configured policy.

    REJECT: fail immediately with HTTPException(429) if the
    semaphore is fully saturated. QUEUE: wait up to
    ``queue_max_wait_s`` then fail. The ``queue_max_wait_s=0``
    sentinel means wait forever.
    """
    if config.admission_policy == AdmissionPolicy.REJECT:
        # Non-blocking: try once.
        if not _try_acquire(sem):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="slab pool exhausted (REJECT policy)",
            )
        return
    # QUEUE policy.
    timeout = (
        None
        if config.queue_max_wait_s == 0
        else config.queue_max_wait_s
    )
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"queue wait exceeded ({config.queue_max_wait_s}s)"
            ),
        ) from exc


def _try_acquire(sem: asyncio.Semaphore) -> bool:
    """Non-blocking semaphore acquire.

    ``asyncio.Semaphore`` lacks a public ``locked()``-with-grab API;
    we inspect the internal ``_value`` (CPython implementation
    detail kept stable across versions; documented in cpython
    ``asyncio/locks.py``).
    """
    if sem._value <= 0:  # noqa: SLF001 - intentional, see docstring
        return False
    sem._value -= 1  # noqa: SLF001
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        label: str = app.state.model_id_label
        return HealthResponse(status="ok", model=label)

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        metrics: Metrics = app.state.metrics
        store: SessionStore = app.state.store
        # Refresh the in-flight gauge from SessionStore so /metrics
        # always reports current state.
        kv_live = store.total_kv_live_bytes
        active = store.active_count
        metrics.snapshot_scheduler(
            active=active,
            pool_in_use=active,
            pool_total=app.state.pool.total_count,
            pending=0,
            kv_live_bytes=kv_live,
        )
        return PlainTextResponse(
            content=metrics.render(),
            media_type=metrics.content_type,
        )

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models() -> ListModelsResponse:
        label: str = app.state.model_id_label
        return ListModelsResponse(
            data=[ModelInfo(id=label, created=int(time.time()))],
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        verifier = app.state.verifier
        store: SessionStore = app.state.store
        append_coord: AppendTokensCoordinator = app.state.append_coord
        gen_coord: GenerationCoordinator = app.state.gen_coord
        config: ServerConfig = app.state.config
        metrics: Metrics = app.state.metrics
        admission_sem: asyncio.Semaphore = app.state.admission_sem
        model_label: str = app.state.model_id_label

        try:
            prompt_ids = _encode_prompt(verifier, req)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"prompt encoding failed: {exc}",
            ) from exc

        eos_token_ids = resolve_eos_ids(verifier.tokenizer)
        if not eos_token_ids:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="server tokenizer has no EOS configuration",
            )

        max_new_tokens = req.max_tokens or config.default_max_new_tokens
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_token_count = len(prompt_ids)

        # Admission control. Failures surface as 429 (REJECT) or
        # 429-after-timeout (QUEUE).
        await _admit(sem=admission_sem, config=config)
        metrics.record_admission(admitted=True)

        session = store.create_session(eos_token_ids=tuple(eos_token_ids))

        try:
            try:
                append_coord.append_tokens(session.session_id, prompt_ids)
            except Exception as exc:  # noqa: BLE001 - surface every prefill error
                store.close_session(session.session_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"prefill error: {type(exc).__name__}: {exc}",
                ) from exc

            if req.stream:
                return EventSourceResponse(
                    _stream_session(
                        gen_coord=gen_coord,
                        session_id=session.session_id,
                        request=request,
                        verifier=verifier,
                        completion_id=completion_id,
                        created=created,
                        model_label=model_label,
                        max_tokens=max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        metrics=metrics,
                        store=store,
                        admission_sem=admission_sem,
                    ),
                    media_type="text/event-stream",
                )

            try:
                output_token_ids, stopped_on_eos = (
                    await _collect_session_tokens(
                        gen_coord=gen_coord,
                        session_id=session.session_id,
                        max_tokens=max_new_tokens,
                        request=request,
                    )
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"engine error: {exc}",
                ) from exc

            completion_text = verifier.tokenizer.decode(
                output_token_ids, skip_special_tokens=True,
            )
            finish_reason = "stop" if stopped_on_eos else "length"

            metrics.record_completion(
                finish_reason=finish_reason,
                n_tokens=len(output_token_ids),
                acceptance_rate=None,
            )

            return JSONResponse(
                content=ChatCompletionResponse(
                    id=completion_id,
                    created=created,
                    model=model_label,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatCompletionResponseMessage(
                                role="assistant", content=completion_text,
                            ),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=prompt_token_count,
                        completion_tokens=len(output_token_ids),
                        total_tokens=(
                            prompt_token_count + len(output_token_ids)
                        ),
                    ),
                ).model_dump(),
            )
        finally:
            # Non-streaming path closes the session here. The streaming
            # path's generator owns its own teardown — the EventSource
            # flow is asynchronous so the session lifecycle is handled
            # in _stream_session's finally.
            if not req.stream:
                try:
                    store.close_session(session.session_id)
                except SessionNotFoundError:
                    pass
                admission_sem.release()


# ---------------------------------------------------------------------------
# Generation drivers
# ---------------------------------------------------------------------------


async def _collect_session_tokens(
    *,
    gen_coord: GenerationCoordinator,
    session_id: str,
    max_tokens: int,
    request: Request,
    disconnect_poll_interval_s: float = 0.05,
) -> tuple[List[int], bool]:
    """Drain the generator coordinator while honoring client disconnects.

    Runs the synchronous ``GenerationCoordinator.generate`` iterator
    in a background thread (it's CPU-bound on the verifier, not
    async), polling ``request.is_disconnected()`` between events.
    Returns ``(emitted_token_ids, stopped_on_eos)``.
    """
    output: List[int] = []
    stopped_on_eos = False

    # Run the generator in a thread to keep the event loop responsive
    # to disconnect polling. Use a queue to pipe events back.
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def _drain():
        try:
            for event in gen_coord.generate(
                session_id, max_tokens=max_tokens,
            ):
                queue.put_nowait(event)
        except Exception as exc:  # noqa: BLE001 - propagate as a queue item
            queue.put_nowait(exc)
        finally:
            queue.put_nowait(sentinel)

    drain_task = asyncio.create_task(asyncio.to_thread(_drain))

    last_disconnect_check = time.monotonic()
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=disconnect_poll_interval_s,
                )
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    drain_task.cancel()
                    raise asyncio.CancelledError() from None
                continue

            if event is sentinel:
                break
            if isinstance(event, BaseException):
                raise event
            if isinstance(event, TokenEvent):
                output.append(event.token_id)
            elif isinstance(event, DoneEvent):
                stopped_on_eos = event.stop_reason == STOP_REASON_EOS
            elif isinstance(event, HistoryTruncatedEvent):
                # Non-streaming path doesn't surface this; ignore.
                continue

            now = time.monotonic()
            if (now - last_disconnect_check) >= disconnect_poll_interval_s:
                last_disconnect_check = now
                if await request.is_disconnected():
                    drain_task.cancel()
                    raise asyncio.CancelledError()
    finally:
        if not drain_task.done():
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, BaseException):
                pass

    return output, stopped_on_eos


async def _stream_session(
    *,
    gen_coord: GenerationCoordinator,
    session_id: str,
    request: Request,
    verifier,
    completion_id: str,
    created: int,
    model_label: str,
    max_tokens: int,
    eos_token_ids: List[int],
    metrics: Metrics,
    store: SessionStore,
    admission_sem: asyncio.Semaphore,
    disconnect_poll_interval_s: float = 0.05,
) -> AsyncIterator[dict]:
    """SSE async generator that drains the GenerationCoordinator."""
    detok = _StreamingDetokenizer(verifier.tokenizer)

    def envelope(content_delta, role_delta, finish_reason) -> dict:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_label,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(
                        role=role_delta, content=content_delta,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )
        return {"data": chunk.model_dump_json()}

    yield envelope(
        content_delta=None, role_delta="assistant", finish_reason=None,
    )

    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    n_tokens = 0
    cancelled_by_disconnect = False

    def _drain():
        try:
            for event in gen_coord.generate(session_id, max_tokens=max_tokens):
                queue.put_nowait(event)
        except Exception as exc:  # noqa: BLE001 - swallow, surface terminal chunk
            queue.put_nowait(exc)
        finally:
            queue.put_nowait(sentinel)

    drain_task = asyncio.create_task(asyncio.to_thread(_drain))
    last_disconnect_check = time.monotonic()
    stopped_on_eos = False

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=disconnect_poll_interval_s,
                )
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    cancelled_by_disconnect = True
                    drain_task.cancel()
                    break
                continue
            if event is sentinel:
                break
            if isinstance(event, BaseException):
                # Verifier raised mid-stream. Once SSE has started
                # there's no way to surface a 500; close gracefully
                # with finish_reason="stop".
                break
            if isinstance(event, TokenEvent):
                n_tokens += 1
                delta = detok.feed(event.token_id)
                if delta:
                    yield envelope(
                        content_delta=delta, role_delta=None, finish_reason=None,
                    )
            elif isinstance(event, DoneEvent):
                stopped_on_eos = event.stop_reason == STOP_REASON_EOS
            elif isinstance(event, HistoryTruncatedEvent):
                # Stream contract: this event arrives BEFORE the
                # first TokenEvent. We don't surface it on the
                # OpenAI wire (no analog). Ignore.
                continue

            now = time.monotonic()
            if (now - last_disconnect_check) >= disconnect_poll_interval_s:
                last_disconnect_check = now
                if await request.is_disconnected():
                    cancelled_by_disconnect = True
                    drain_task.cancel()
                    break
    finally:
        if not drain_task.done():
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, BaseException):
                pass
        try:
            store.close_session(session_id)
        except SessionNotFoundError:
            pass
        admission_sem.release()

    finish_reason = (
        "stop" if (stopped_on_eos and not cancelled_by_disconnect)
        else "length"
    )
    yield envelope(
        content_delta=None, role_delta=None, finish_reason=finish_reason,
    )
    metrics.record_completion(
        finish_reason=finish_reason,
        n_tokens=n_tokens,
        acceptance_rate=None,
    )
    yield {"data": "[DONE]"}
