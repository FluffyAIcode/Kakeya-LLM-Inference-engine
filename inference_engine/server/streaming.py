"""Sync-to-async bridge for streaming generation.

The speculative decoder is synchronous: ``decoder.generate(...)`` runs
to completion in the calling thread, invoking ``on_token`` once per
committed token along the way. SSE handlers, on the other hand, are
async: they ``yield`` text deltas from inside an async generator that
``sse_starlette.EventSourceResponse`` iterates.

This module is the bridge. It runs the synchronous decoder in a worker
thread (``asyncio.to_thread``) while exposing an async generator the
SSE handler can iterate. Tokens flow through an ``asyncio.Queue``;
client disconnect is propagated back into the worker via a flag the
``on_token`` callback consults on each commit.

Disconnect detection runs *both* on idle (``asyncio.wait_for`` timeout)
and on a wall-clock poll interval — the former handles "no tokens
arriving" and the latter handles "tokens arriving faster than the
poll interval", which is the common case under MLX/CUDA verifiers
where per-token latency is single-digit milliseconds. Without the
wall-clock branch a busy stream would never check disconnect and a
client closing its connection mid-stream would be ignored.

Two public functions:

    iter_token_deltas(engine, request, prompt_ids, max_new_tokens,
                      eos_token_ids) -> async iterator of (delta_text, is_final)
        Drives a streaming chat completion. Yields per-token text
        deltas using a streaming detokenizer (so multi-byte sequences
        are emitted on the byte that completes them, not on the
        first byte of the run). The final yield is ``("", True)``,
        signalling the end of the stream; the SSE handler then emits
        the OpenAI ``[DONE]`` sentinel.

    run_blocking(engine, prompt_ids, max_new_tokens, eos_token_ids)
        -> EngineResult
        Drives a non-streaming chat completion. Just defers to
        ``asyncio.to_thread`` so the route handler does not block the
        event loop while generation runs. Returned result feeds the
        non-streaming response shape.

Design notes:

  * No mocking, no fallback. If ``run_coroutine_threadsafe`` fails to
    schedule a put, the exception propagates and the request errors;
    we do not silently drop tokens.
  * Disconnect detection runs on a small polling interval (50 ms);
    starlette's ``Request.is_disconnected()`` is itself a coroutine
    that polls the receive channel, so a tighter loop wouldn't
    discover disconnects faster but would burn CPU.
  * The producer task always drains; on early exit (disconnect or
    exception) we still ``await`` it to ensure thread cleanup. The
    sentinel value distinguishes "real token, possibly id 0" from
    "producer is done".
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple

from .engine import Engine, EngineResult
from .tokenizer import Tokenizer

# Sentinel marker pushed to the queue when generation finishes.
# Distinct from any int token id so we don't have to encode "is_final"
# on every queue entry.
_PRODUCER_DONE = object()


@dataclass
class StreamingResult:
    """Aggregate result available after iter_token_deltas exhausts.

    Populated by the producer task as it returns. Streaming routes
    keep a reference to the :class:`_StreamingSession` and read this
    after the async iterator is fully drained, to populate
    ``finish_reason`` and ``usage`` in the SSE finalization chunk.
    """

    engine_result: Optional[EngineResult]
    cancelled_by_disconnect: bool
    error: Optional[BaseException]


class _StreamingDetokenizer:
    """Incremental decoder that emits valid text deltas only.

    HuggingFace tokenizers can decode partial id sequences, but they
    do *not* guarantee that ``decode([id_n])`` is a substring of
    ``decode([id_0..id_n])`` because BPE merges and special-token
    handling reshape the prefix. The robust pattern (used by
    ``scripts/chat.py``'s on_token):

        full = tokenizer.decode(all_ids_so_far, skip_special_tokens=True)
        delta = full[len(decoded_so_far):]
        decoded_so_far = full

    is what we replicate here. ``feed(token_id)`` returns the new
    text since the last call, which may be the empty string if the
    new token contributes only the first byte of a multi-byte UTF-8
    sequence (the next call will then yield both bytes).
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._all_ids: List[int] = []
        self._decoded_so_far: str = ""

    def feed(self, token_id: int) -> str:
        self._all_ids.append(int(token_id))
        full = self._tokenizer.decode(
            self._all_ids, skip_special_tokens=True
        )
        delta = full[len(self._decoded_so_far):]
        self._decoded_so_far = full
        return delta


class _StreamingSession:
    """Holds the producer task, queue, and disconnect flag for a single stream.

    Created by :func:`iter_token_deltas`, exposed publicly so the
    caller can read ``result`` after the iterator exhausts.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.disconnect_flag: asyncio.Event = asyncio.Event()
        self.result: StreamingResult = StreamingResult(
            engine_result=None, cancelled_by_disconnect=False, error=None
        )


async def iter_token_deltas(
    engine: Engine,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_ids: List[int],
    *,
    is_disconnected: Optional["_DisconnectedCallable"] = None,
    disconnect_poll_interval_s: float = 0.05,
) -> AsyncIterator[Tuple[str, bool, "_StreamingSession"]]:
    """Yield ``(delta_text, is_final, session)`` triples.

    ``is_final`` is ``False`` for normal token deltas and ``True`` for
    the single terminal yield that signals end-of-stream. After the
    terminal yield, the caller can read ``session.result.engine_result``
    for usage stats / finish_reason.

    ``is_disconnected``, if provided, is awaited on each poll interval;
    if it returns ``True`` the disconnect_flag is set, the worker stops
    at the next token boundary, and the iterator emits the terminal
    yield. The session's ``cancelled_by_disconnect`` flag captures
    this for the route layer.

    Empty deltas are *not* yielded — clients shouldn't see empty
    ``data:`` events. (The streaming detokenizer can produce empty
    deltas for multi-byte UTF-8 mid-sequence; we filter them.)
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if not eos_token_ids:
        raise ValueError("eos_token_ids must be non-empty")

    session = _StreamingSession()
    detok = _StreamingDetokenizer(engine.tokenizer)
    loop = asyncio.get_running_loop()

    def on_token(tok_id: int) -> bool:
        if session.disconnect_flag.is_set():
            return True
        # Schedule the put on the event loop. We do not block the
        # worker thread on the result — in the rare case the queue is
        # near full, the put will simply queue up and complete shortly.
        # If the loop has shut down (server killed mid-request) the
        # ensure_future call raises, which we let propagate to the
        # decoder's caller — generate() will surface it as an error.
        asyncio.run_coroutine_threadsafe(session.queue.put(tok_id), loop)
        return False

    async def producer() -> None:
        try:
            session.result.engine_result = await asyncio.to_thread(
                engine.generate,
                prompt_ids, max_new_tokens, eos_token_ids, on_token,
            )
        except BaseException as exc:  # noqa: BLE001 — re-raised after sentinel
            session.result.error = exc
        finally:
            await session.queue.put(_PRODUCER_DONE)

    producer_task = asyncio.create_task(producer())
    last_disconnect_check = time.monotonic()

    async def _maybe_cancel_on_disconnect() -> bool:
        """Return True if a disconnect was detected and we should stop.

        On cancellation, sets the flag and drains the queue until the
        producer sentinel arrives. Caller breaks out of the main loop
        once we return True.
        """
        if is_disconnected is None:
            return False
        if not await is_disconnected():
            return False
        session.disconnect_flag.set()
        session.result.cancelled_by_disconnect = True
        while True:
            tail = await session.queue.get()
            if tail is _PRODUCER_DONE:
                break
        return True

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    session.queue.get(), timeout=disconnect_poll_interval_s
                )
            except asyncio.TimeoutError:
                if await _maybe_cancel_on_disconnect():
                    break
                continue
            if item is _PRODUCER_DONE:
                break
            delta = detok.feed(item)
            if delta:
                yield delta, False, session
            # Periodic disconnect poll on the fast-token path: even
            # if the queue never empties, we still want to honor
            # client disconnects within a single poll interval of
            # the actual TCP close.
            now = time.monotonic()
            if (now - last_disconnect_check) >= disconnect_poll_interval_s:
                last_disconnect_check = now
                if await _maybe_cancel_on_disconnect():
                    break
        yield "", True, session
    finally:
        # Make sure the producer task is awaited so its thread closes.
        # If we got here via an exception, also surface that.
        if not producer_task.done():
            session.disconnect_flag.set()
            await producer_task
        else:
            # Already done; ensure exception is observed.
            await producer_task
        if session.result.error is not None:
            raise session.result.error


async def run_blocking(
    engine: Engine,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_ids: List[int],
) -> EngineResult:
    """Run generation to completion off the event-loop thread.

    No streaming, no callback. Returns the full :class:`EngineResult`
    once generation finishes. Used by ``stream=False`` chat completions.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if not eos_token_ids:
        raise ValueError("eos_token_ids must be non-empty")
    return await asyncio.to_thread(
        engine.generate, prompt_ids, max_new_tokens, eos_token_ids, None
    )


# Type alias for is_disconnected callable, kept as a string forward-
# reference so we don't need to import ``starlette.requests.Request``
# in the streaming module — the module is intentionally HTTP-agnostic.
_DisconnectedCallable = "callable returning awaitable[bool]"  # noqa: F821
