"""FastAPI app factory and route handlers.

The app is constructed by :func:`create_app` from a fully-initialized
:class:`Engine` (and a :class:`ServerConfig`). All inference flows
through a :class:`Scheduler` constructed inside the factory: routes
never call ``engine.generate`` directly. This is the integration that
makes admission control, fair queuing, slab-pool occupancy, and
graceful shutdown observable / consistent regardless of single-user
or multi-user deployment.

Routes implemented in this commit:

    GET  /healthz
    GET  /v1/models
    POST /v1/chat/completions

OpenAI compatibility notes
--------------------------

* ``stream`` is the load-bearing flag: when true the response is
  ``text/event-stream``; when false it is ``application/json``. We
  branch on it inside the route, not at registration time.
* Sampling parameters (``temperature``, ``top_p``, ``stop``) are
  accepted in the request schema but not applied — the underlying
  decoder is greedy temperature-0 by design (see ADR 0001 §2.2 for
  the rationale).
* ``finish_reason`` is ``"stop"`` if EOS terminated generation OR
  if the client cancelled, ``"length"`` if ``max_tokens`` did. We
  do not yet emit ``"content_filter"`` or ``"function_call"``.

Error mapping
-------------

* Pydantic validation errors: 422 (FastAPI default).
* Tokenizer chat-template rejection: 400.
* Tokenizer with no EOS: 500 (defense in depth; engine constructor
  is supposed to catch this earlier).
* Scheduler rejects (pool full under REJECT policy, queue timeout
  under QUEUE policy): 429 with a JSON body following OpenAI's
  error shape.
* Engine raises mid-generate: 500 (non-streaming) or terminal SSE
  chunk with ``finish_reason="stop"`` (streaming — the SSE
  contract has no graceful way to surface a 500 once the response
  has started; the session error is swallowed at the wire after
  any partial output).

Lifespan
--------

The app registers a FastAPI lifespan context that calls
``scheduler.shutdown()`` when the server stops. Active sessions are
cancelled, queued admissions are rejected, slabs are released. The
HTTP layer becomes externally indistinguishable from "no server here"
within one poll interval after the lifespan exits.
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
from inference_engine.scheduler.config import SchedulerConfig
from inference_engine.scheduler.scheduler import (
    RequestRejected,
    Scheduler,
)
from inference_engine.scheduler.session import Session, SessionState

from .auth import verify_api_key
from .config import ServerConfig
from .engine import Engine
from .errors import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from .metrics import Metrics
from .schemas import (
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
from .streaming import _StreamingDetokenizer
from .tokenizer import resolve_eos_ids


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    engine: Engine,
    config: ServerConfig,
    pool: Optional[SlabPool] = None,
) -> FastAPI:
    """Build a FastAPI app bound to a specific engine + config.

    Parameters
    ----------
    engine:
        Anything implementing :class:`Engine`. In production this is
        a :class:`SpeculativeEngine`; in tests it is a deterministic
        test double.
    config:
        Process-wide :class:`ServerConfig`. The scheduler-related
        fields (``max_concurrent``, ``admission_policy``,
        ``queue_max_wait_s``) drive the internal :class:`Scheduler`.
    pool:
        Optional pre-built :class:`SlabPool`. If ``None``, we build a
        minimal placeholder pool sized for ``config.max_concurrent``
        slots — these slots are pure admission-control bookkeeping
        until a future commit wires the verifier itself to consume
        slabs from the pool. The placeholder slab tensors are
        deliberately tiny (a few bytes per slab) since they are not
        currently read by attention kernels.
    """
    pool = pool if pool is not None else _build_placeholder_pool(config.max_concurrent)
    if pool.total_count != config.max_concurrent:
        raise ValueError(
            f"pool.total_count={pool.total_count} does not match "
            f"config.max_concurrent={config.max_concurrent}"
        )
    scheduler = Scheduler(
        engine=engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=config.max_concurrent,
            admission_policy=config.admission_policy,
            queue_max_wait_s=config.queue_max_wait_s,
        ),
    )
    metrics = Metrics.build()
    metrics.snapshot_scheduler(
        active=0, pool_in_use=0, pool_total=pool.total_count, pending=0,
        kv_live_bytes=0,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: nothing to do; scheduler is already constructed and
        # ready to admit. We yield without doing anything here so unit
        # tests that exercise the route via ASGITransport without
        # explicit lifespan handling still work.
        try:
            yield
        finally:
            await scheduler.shutdown()

    app = FastAPI(
        title="Kakeya Inference Engine",
        description=(
            "OpenAI-compatible HTTP API for the DLM-proposer + AR-verifier "
            "speculative decoder. See https://github.com/FluffyAIcode/"
            "Kakeya-LLM-Inference-engine for source and ADRs."
        ),
        version="0.2.0-dev",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.config = config
    app.state.scheduler = scheduler
    app.state.pool = pool
    app.state.metrics = metrics

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

    _register_routes(app)
    return app


class _AuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token gate for ``/v1/*`` routes when api_keys is non-empty."""

    def __init__(self, app, *, valid_keys) -> None:
        super().__init__(app)
        self._valid_keys = frozenset(valid_keys)

    async def dispatch(self, request, call_next):
        try:
            verify_api_key(request, valid_keys=self._valid_keys)
        except StarletteHTTPException as exc:
            # Re-route through the registered handler so the response
            # carries the OpenAI envelope.
            return await http_exception_handler(request, exc)
        return await call_next(request)


class _MetricsMiddleware(BaseHTTPMiddleware):
    """Records ``http_requests_total`` + duration histogram per request.

    The path label is the matched route's path template (e.g.
    ``/v1/chat/completions``) when available, otherwise the raw URL
    path. We deliberately avoid recording dynamic path segments
    (e.g. session ids) to prevent label-cardinality blow-up.
    """

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
        """Return the route template if matched, else the raw path.

        Starlette stores the matched route on ``request.scope["route"]``
        when available; we fall back to the raw URL for unmatched
        requests (404s).
        """
        route = request.scope.get("route")
        if route is not None and hasattr(route, "path"):
            return route.path
        return request.url.path


def _build_placeholder_pool(num_slabs: int) -> SlabPool:
    """Construct a minimal :class:`SlabPool` for admission-control bookkeeping.

    Slab tensors are 1-element bf16 (2 bytes per K + 2 per V × num_slabs).
    Total memory cost for the default ``num_slabs=1`` pool is ~4 bytes,
    plus Python object overhead. When the verifier-side refactor lands
    that actually consumes slabs as KV storage, callers will pass a
    properly-sized pool to ``create_app`` and this placeholder will
    become unnecessary in production paths.
    """
    cfg = SlabConfig(
        num_layers=1, num_heads=1, sink_size=0, window_size=1,
        head_dim=1, dtype=torch.bfloat16,
    )
    return SlabPool(num_slabs=num_slabs, slab_config=cfg)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        engine: Engine = app.state.engine
        return HealthResponse(status="ok", model=engine.model_id_label)

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        metrics: Metrics = app.state.metrics
        scheduler: Scheduler = app.state.scheduler
        pool: SlabPool = app.state.pool
        # Refresh scheduler-state gauges on every scrape so the
        # exposition reflects "now" rather than the last
        # admission/completion event.
        engine_for_kv: Engine = app.state.engine
        # Read KV bytes directly from the engine's verifier rather
        # than from pool.live_kv_bytes. Rationale: in v0.3 the slab
        # is a session ticket (acquired/released per request) — the
        # verifier holds the real KV cache tensors and is the
        # canonical source of truth. Pool-side accounting only
        # populates once PooledVerifier is wired (a post-v0.3.0
        # change) and otherwise reads 0 even while the verifier
        # cache is several MiB.
        #
        # Gauge semantics: "KV bytes attributable to in-flight
        # sessions". Between turns, the verifier's ``self.cache``
        # still holds the previous turn's tensors — the next
        # prefill calls ``reset()`` which replaces them, but until
        # then ``engine.kv_state()`` reports non-zero residual
        # bytes. Reporting that as "live" misleads observers
        # and breaks the §2.3 KV-bounded check (residual carries
        # forward at the previous turn's peak, never trimmed). We
        # therefore gate the gauge on ``active_count > 0``: an
        # idle server reports 0, a server with an active session
        # reports the verifier's true KV size. This is also how
        # the gauge will naturally behave once PooledVerifier is
        # wired post-v0.3 (the pool aggregation is 0 when no slab
        # is in use).
        kv_live = (
            int(engine_for_kv.kv_state())
            if scheduler.active_count > 0
            else 0
        )
        metrics.snapshot_scheduler(
            active=scheduler.active_count,
            pool_in_use=pool.in_use_count,
            pool_total=pool.total_count,
            pending=scheduler.pending_count,
            kv_live_bytes=kv_live,
        )
        return PlainTextResponse(
            content=metrics.render(),
            media_type=metrics.content_type,
        )

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models() -> ListModelsResponse:
        engine: Engine = app.state.engine
        return ListModelsResponse(
            data=[
                ModelInfo(
                    id=engine.model_id_label,
                    created=int(time.time()),
                )
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        engine: Engine = app.state.engine
        scheduler: Scheduler = app.state.scheduler
        config: ServerConfig = app.state.config
        metrics: Metrics = app.state.metrics

        try:
            prompt_ids = _encode_prompt(engine, req)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"prompt encoding failed: {exc}",
            ) from exc

        eos_token_ids = resolve_eos_ids(engine.tokenizer)
        if not eos_token_ids:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="server tokenizer has no EOS configuration",
            )

        max_new_tokens = req.max_tokens or config.default_max_new_tokens
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_token_count = len(prompt_ids)

        # Submit to the scheduler. Admission failures surface as 429
        # — the canonical OpenAI status for capacity exhaustion.
        try:
            session = await scheduler.submit(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
        except RequestRejected as exc:
            metrics.record_admission(admitted=False)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc
        metrics.record_admission(admitted=True)

        if req.stream:
            return EventSourceResponse(
                _stream_via_scheduler(
                    scheduler=scheduler,
                    session=session,
                    request=request,
                    engine=engine,
                    completion_id=completion_id,
                    created=created,
                    metrics=metrics,
                ),
                media_type="text/event-stream",
            )

        try:
            output_token_ids = await _collect_non_streaming_tokens(
                scheduler=scheduler,
                session=session,
                request=request,
            )
        except asyncio.CancelledError:
            # Client timed out/disconnected while the JSON response was
            # draining. Without explicit cancellation the worker can keep
            # occupying the only slab, causing later queued requests to 429.
            await scheduler.cancel_session(session)
            raise
        except BaseException as exc:
            await scheduler.cancel_session(session)
            # Engine raised mid-generate; surface as 500.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"engine error: {exc}",
            ) from exc

        completion_text = engine.tokenizer.decode(
            output_token_ids, skip_special_tokens=True
        )
        # finish_reason: COMPLETED + last token in eos_set => "stop";
        # otherwise (cap, cancellation, or anything else) => "length"
        # for non-streaming. Cancellation in non-streaming should not
        # happen via this path (no cancel hook on JSON responses), but
        # we keep the conservative mapping.
        if (
            session.state is SessionState.COMPLETED
            and output_token_ids
            and output_token_ids[-1] in set(eos_token_ids)
        ):
            finish_reason = "stop"
        else:
            finish_reason = "length"

        metrics.record_completion(
            finish_reason=finish_reason,
            n_tokens=len(output_token_ids),
            acceptance_rate=_session_acceptance_rate(scheduler, session),
        )
        _emit_path_selection_metric(metrics, session)

        return JSONResponse(
            content=ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=engine.model_id_label,
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
                    total_tokens=prompt_token_count + len(output_token_ids),
                ),
            ).model_dump()
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_prompt(engine: Engine, req: ChatCompletionRequest) -> List[int]:
    """Apply the tokenizer's chat template to the request messages."""
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = engine.tokenizer.apply_chat_template(
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


def _session_acceptance_rate(
    scheduler: Scheduler, session: Session,
) -> Optional[float]:
    """Per-session acceptance rate from the stashed EngineResult.

    The scheduler worker stores ``engine.generate()``'s result on
    ``session.engine_result`` after generation completes (PR 7-4).
    Returns ``None`` if the result is unavailable (session was
    cancelled / failed before the engine returned, or the engine
    is a test double that doesn't expose the field).
    """
    _ = scheduler  # kept for signature stability with existing callers
    result = getattr(session, "engine_result", None)
    if result is None:
        return None
    rate = getattr(result, "acceptance_rate", None)
    if rate is None:
        return None
    return float(rate)


def _emit_path_selection_metric(
    metrics: "Metrics", session: Session,
) -> None:
    """Emit ADR 0007 §2.10 path-selection observability for one
    completed session, if the engine reported the relevant fields.

    Called from both the streaming and non-streaming completion
    paths after the session reaches a terminal state. No-op when
    the engine result is unavailable (e.g., test doubles that
    don't populate path_selection).
    """
    result = getattr(session, "engine_result", None)
    if result is None:
        return
    path = getattr(result, "path_selection", None)
    if path not in ("continuation", "new_session"):
        return
    metrics.record_path_selection(
        path=path,
        tokens_skipped=int(getattr(result, "tokens_skipped", 0)),
        prefill_duration_s=float(
            getattr(result, "prefill_duration_seconds", 0.0)
        ),
    )


async def _collect_non_streaming_tokens(
    *,
    scheduler: Scheduler,
    session: Session,
    request: Request,
    disconnect_poll_interval_s: float = 0.05,
) -> List[int]:
    """Drain a non-streaming session while honoring client disconnects.

    Streaming responses already poll ``request.is_disconnected()`` and
    cancel their scheduler session. JSON responses need the same cleanup:
    a timed-out client otherwise leaves the scheduler worker running until
    generation finishes, which can monopolize a single-slot server.
    """
    output_token_ids: List[int] = []
    last_disconnect_check = time.monotonic()
    async for tok in scheduler.iter_tokens(session):
        output_token_ids.append(int(tok))
        now = time.monotonic()
        if (now - last_disconnect_check) >= disconnect_poll_interval_s:
            last_disconnect_check = now
            if await request.is_disconnected():
                await scheduler.cancel_session(session)
    return output_token_ids


async def _stream_via_scheduler(
    *,
    scheduler: Scheduler,
    session: Session,
    request: Request,
    engine: Engine,
    completion_id: str,
    created: int,
    metrics: Metrics,
    disconnect_poll_interval_s: float = 0.05,
) -> AsyncIterator[dict]:
    """SSE async generator that drains :meth:`Scheduler.iter_tokens`.

    Implements the OpenAI streaming chunk protocol on top of a
    scheduler-managed session. Polls ``request.is_disconnected()`` on
    a wall-clock interval; on disconnect, calls
    ``scheduler.cancel_session`` to short-circuit generation.

    The generator yields ``{"data": "<json>"}`` envelopes (the format
    sse-starlette consumes), terminated by ``{"data": "[DONE]"}``.
    """
    import asyncio

    model_label = engine.model_id_label
    detok = _StreamingDetokenizer(engine.tokenizer)

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

    yield envelope(content_delta=None, role_delta="assistant", finish_reason=None)

    last_disconnect_check = time.monotonic()
    cancelled_by_disconnect = False
    try:
        async for tok in scheduler.iter_tokens(session):
            delta = detok.feed(int(tok))
            if delta:
                yield envelope(
                    content_delta=delta, role_delta=None, finish_reason=None,
                )
            now = time.monotonic()
            if (now - last_disconnect_check) >= disconnect_poll_interval_s:
                last_disconnect_check = now
                if await request.is_disconnected():
                    cancelled_by_disconnect = True
                    await scheduler.cancel_session(session)
                    # Drain remaining tokens (will exit shortly because
                    # the on_token callback inside the scheduler now
                    # returns True).
    except BaseException:  # noqa: BLE001 — surface as terminal chunk
        # Engine errors mid-stream end the SSE stream gracefully; the
        # client sees a finish_reason="stop" with no further content.
        # We deliberately do NOT raise here — once SSE has started,
        # there is no way to send a 500 status; the OpenAI clients
        # also expect graceful termination on errors.
        pass

    # Terminal chunk: derive finish_reason from session state.
    if cancelled_by_disconnect or session.state is SessionState.CANCELLED:
        finish_reason = "stop"
    elif session.state is SessionState.COMPLETED:
        # Did we end on EOS or hit max_tokens?
        if (
            session.output_token_ids
            and session.output_token_ids[-1]
            in set(session.eos_token_ids)
        ):
            finish_reason = "stop"
        else:
            finish_reason = "length"
    else:
        # FAILED or some other terminal — be conservative.
        finish_reason = "stop"

    yield envelope(
        content_delta=None, role_delta=None, finish_reason=finish_reason,
    )
    metrics.record_completion(
        finish_reason=finish_reason,
        n_tokens=len(session.output_token_ids),
        acceptance_rate=_session_acceptance_rate(scheduler, session),
    )
    _emit_path_selection_metric(metrics, session)
    yield {"data": "[DONE]"}
