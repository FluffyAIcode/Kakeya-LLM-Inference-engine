"""FastAPI app factory and route handlers.

The app is constructed by :func:`create_app` from a fully-initialized
:class:`Engine` (and a :class:`ServerConfig`). Routes are registered
inside the factory rather than at module level so tests can spin up
multiple isolated apps with different engines / configs in the same
process.

Routes implemented in this commit:

    GET  /healthz
    GET  /v1/models
    POST /v1/chat/completions

We deliberately do not implement ``POST /v1/completions`` (the
legacy text-completion endpoint). It would duplicate
``/v1/chat/completions`` for our generation model and clients in
2026 universally use the chat endpoint; we keep the surface narrow.

OpenAI compatibility notes
--------------------------

* ``stream`` is the load-bearing flag: when true the response is
  ``text/event-stream``; when false it is ``application/json``. We
  branch on it inside the route, not at registration time.
* Sampling parameters (``temperature``, ``top_p``, ``stop``) are
  accepted in the request schema but not applied — the underlying
  decoder is greedy temperature-0 by design (see ADR 0001 §2.2 for
  the rationale: speculative decoding's correctness proof requires
  greedy or aligned-distribution sampling). Acceptance of the
  parameters keeps off-the-shelf clients happy; ignoring them
  preserves the algorithmic correctness contract.
* ``finish_reason`` is ``"stop"`` if EOS terminated generation,
  ``"length"`` if ``max_tokens`` did. We do not yet emit
  ``"content_filter"`` or ``"function_call"``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import ServerConfig
from .engine import Engine
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
from .streaming import iter_token_deltas, run_blocking
from .tokenizer import resolve_eos_ids


def create_app(engine: Engine, config: ServerConfig) -> FastAPI:
    """Build a FastAPI app bound to a specific engine + config.

    Parameters
    ----------
    engine:
        Anything implementing :class:`Engine`. In production this is
        a :class:`SpeculativeEngine`; in tests it is a deterministic
        test double.
    config:
        Process-wide :class:`ServerConfig`. Stored on the app state
        for routes to read; never re-read from env at request time.
    """
    app = FastAPI(
        title="Kakeya Inference Engine",
        description=(
            "OpenAI-compatible HTTP API for the DLM-proposer + AR-verifier "
            "speculative decoder. See https://github.com/FluffyAIcode/"
            "Kakeya-LLM-Inference-engine for source and ADRs."
        ),
        version="0.2.0-dev",
    )
    app.state.engine = engine
    app.state.config = config
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        engine: Engine = app.state.engine
        return HealthResponse(status="ok", model=engine.model_id_label)

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
        config: ServerConfig = app.state.config

        # Encode prompt via the tokenizer. Unrecognized exceptions
        # propagate as 500; bad requests (rejected templates) come
        # back as 400 with an explanatory message.
        try:
            prompt_ids = _encode_prompt(engine, req)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"prompt encoding failed: {exc}",
            ) from exc

        eos_token_ids = resolve_eos_ids(engine.tokenizer)
        if not eos_token_ids:
            # Should be caught at engine construction; defense in depth.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="server tokenizer has no EOS configuration",
            )

        max_new_tokens = req.max_tokens or config.default_max_new_tokens
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_token_count = len(prompt_ids)

        if req.stream:
            return EventSourceResponse(
                _stream_chat_chunks(
                    engine=engine,
                    request=request,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    completion_id=completion_id,
                    created=created,
                    prompt_token_count=prompt_token_count,
                ),
                media_type="text/event-stream",
            )

        result = await run_blocking(
            engine, prompt_ids, max_new_tokens, eos_token_ids
        )
        completion_text = engine.tokenizer.decode(
            result.output_token_ids, skip_special_tokens=True
        )
        finish_reason = "stop" if result.stopped_on_eos else "length"
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
                    completion_tokens=len(result.output_token_ids),
                    total_tokens=prompt_token_count + len(result.output_token_ids),
                ),
            ).model_dump()
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_prompt(engine: Engine, req: ChatCompletionRequest) -> list[int]:
    """Apply the tokenizer's chat template to the request messages.

    Returns a flat list of int token ids. Raises ``ValueError`` if
    the tokenizer rejects the messages — caller wraps this into a
    400 response.
    """
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = engine.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    if not isinstance(prompt_ids, list) or not all(isinstance(t, int) for t in prompt_ids):
        raise ValueError(
            f"chat template returned {type(prompt_ids).__name__}, expected list[int]"
        )
    if not prompt_ids:
        raise ValueError("chat template produced an empty token sequence")
    return prompt_ids


async def _stream_chat_chunks(
    *,
    engine: Engine,
    request: Request,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_ids: list[int],
    completion_id: str,
    created: int,
    prompt_token_count: int,
) -> AsyncIterator[dict]:
    """Async generator yielding OpenAI-compat SSE event payloads.

    The first event sets ``role: assistant`` (no content). Each
    subsequent event carries a ``content`` delta. The final event
    carries ``finish_reason`` and an empty delta. After the iterator
    exits, ``EventSourceResponse`` emits the literal ``[DONE]``
    sentinel that OpenAI clients expect.

    Yields dicts shaped like ``{"data": "<json-encoded chunk>"}`` —
    that's the format ``EventSourceResponse`` consumes. We don't use
    the simpler "yield string" form because we want the JSON encoded
    by the same serializer as non-streaming (pydantic), to keep field
    ordering and types identical.
    """
    model_label = engine.model_id_label

    def envelope(content_delta: str | None, role_delta: str | None,
                 finish_reason: str | None) -> dict:
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

    async def is_disconnected_callable() -> bool:
        return await request.is_disconnected()

    async for delta_text, is_final, session in iter_token_deltas(
        engine=engine,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        is_disconnected=is_disconnected_callable,
    ):
        if not is_final:
            yield envelope(
                content_delta=delta_text, role_delta=None, finish_reason=None
            )
            continue

        # Terminal chunk — populate finish_reason from session.result.
        result = session.result.engine_result
        if session.result.cancelled_by_disconnect:
            finish_reason = "stop"
        elif result is not None and result.stopped_on_eos:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        yield envelope(
            content_delta=None, role_delta=None, finish_reason=finish_reason,
        )

    # OpenAI clients expect a literal "[DONE]" terminator so they know
    # the stream is finished. sse-starlette will frame this as
    # `data: [DONE]\n\n` like any other event.
    yield {"data": "[DONE]"}
