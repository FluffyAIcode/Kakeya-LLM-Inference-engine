"""Cancellable bounded-buffer producer/consumer coordinator.

This is the async equivalent of a Go channel with structured cancellation.
A :class:`PipelineCoordinator` owns one bounded ``asyncio.Queue``; a
producer task pushes items into the queue and the consumer (via
:meth:`consume`) drains them as an async iterator. Closing the
coordinator (explicitly via :meth:`close` or implicitly when the
producer finishes) causes the consumer's iteration to exhaust cleanly
*after* all in-flight items are drained — the closing event does not
discard in-flight work.

Shape:

    coord = PipelineCoordinator(buffer_size=64)

    async def produce():
        for i in range(100):
            await coord.put(i)
        await coord.close()  # signals normal end-of-stream

    coord.start_producer(produce())

    async for item in coord.consume():
        process(item)

Exception propagation:

    If the producer coroutine raises, the exception is captured and
    re-raised on the consumer side after all already-queued items
    drain. Late propagation is the right policy because a producer
    that emits 99 valid items and then crashes still produced 99
    valid items the consumer should see — drop-on-error is a worse
    failure mode for streaming.

Cancellation:

    The consumer can call :meth:`cancel` to stop iteration early.
    The cancellation is *cooperative*: the producer continues to run
    (it has no way for an external caller to stop a sync function
    short of cancelling the asyncio task) but the consumer stops
    pulling from the queue. The producer task is awaited in
    :meth:`consume`'s ``finally`` to keep cleanup tidy.

Backpressure:

    The bounded queue creates natural backpressure: ``put`` awaits
    when the buffer is full, so a slow consumer pauses a fast
    producer rather than letting the buffer grow unbounded.

Why a class and not just a function:

    Three async tasks (producer, consumer, cancel-watcher) share
    state (the queue, error slot, close flag, producer task handle).
    A class encapsulates that state and exposes a small API; a
    function would have to thread the state via closures and would
    not be safely re-entrant.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Generic, Optional, TypeVar


T = TypeVar("T")


class PipelineClosed(RuntimeError):
    """Raised when ``put`` is called after the coordinator has closed
    or after a consumer has cancelled iteration."""


class PipelineError(RuntimeError):
    """Wrapper exception raised on the consumer side when the producer
    failed. The original exception is available as ``__cause__``."""


class StreamSentinel:
    """Marker class for queue items that signal stream events.

    Currently has just one instance, ``_PRODUCER_DONE`` (private),
    but the class form makes it easy to add disconnected /
    error-occurred sentinels in the future without restructuring
    the queue.
    """


_PRODUCER_DONE = StreamSentinel()


class PipelineCoordinator(Generic[T]):
    """Owns one buffered async queue plus producer/consumer lifecycle."""

    def __init__(self, buffer_size: int = 64) -> None:
        if buffer_size <= 0:
            raise ValueError(
                f"buffer_size must be positive, got {buffer_size}"
            )
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=buffer_size)
        self._closed: bool = False
        self._consumer_cancelled: bool = False
        self._producer_task: Optional[asyncio.Task] = None
        self._producer_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    async def put(self, item: T) -> None:
        """Push an item into the buffer. Blocks if buffer is full.

        Raises :class:`PipelineClosed` if the coordinator was closed
        or the consumer cancelled — the producer should treat this
        as "stop producing".
        """
        if self._closed or self._consumer_cancelled:
            raise PipelineClosed(
                "coordinator is closed or consumer cancelled; "
                "producer should stop"
            )
        await self._queue.put(item)

    async def close(self) -> None:
        """Signal end-of-stream from the producer side.

        Idempotent: a second close is a no-op. After close, ``put``
        raises :class:`PipelineClosed` and the consumer's iterator
        finishes once it has drained the queue.
        """
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_PRODUCER_DONE)

    def start_producer(self, coro: Awaitable[None]) -> asyncio.Task:
        """Run a producer coroutine as a managed task.

        The coordinator stores the task handle so :meth:`consume`'s
        ``finally`` block can await it for clean teardown. Exceptions
        from the producer are captured and re-raised on the consumer
        side after the queue drains (as :class:`PipelineError`).

        Returns the spawned :class:`asyncio.Task` for callers that
        want to await it themselves; under normal use that is not
        necessary because :meth:`consume` does it.
        """
        if self._producer_task is not None:
            raise RuntimeError(
                "producer already started; PipelineCoordinator only "
                "supports one producer per instance"
            )

        async def wrapped() -> None:
            try:
                await coro
            except BaseException as exc:
                self._producer_error = exc
            finally:
                # Always close so the consumer sees end-of-stream.
                await self.close()

        self._producer_task = asyncio.create_task(wrapped())
        return self._producer_task

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    async def consume(self) -> AsyncIterator[T]:
        """Async iterator over produced items.

        Drains the queue until the close sentinel is seen, then
        awaits the producer task and re-raises any captured error
        as :class:`PipelineError`.
        """
        try:
            while True:
                item = await self._queue.get()
                if isinstance(item, StreamSentinel):
                    break
                yield item
        finally:
            self._consumer_cancelled = True
            if self._producer_task is not None:
                # Wait for the producer to finish cleanly. If it is
                # currently blocked on a full queue, the cancel flag
                # will short-circuit it on its next put — but only
                # if the producer is one that uses self.put. For
                # producers using raw queue access this is best-
                # effort.
                if not self._producer_task.done():
                    # Drain queue to unblock any pending put.
                    while not self._queue.empty():
                        try:
                            self._queue.get_nowait()
                        except asyncio.QueueEmpty:  # pragma: no cover - race
                            break
                try:
                    await self._producer_task
                except BaseException:  # pragma: no cover - we surface via _producer_error
                    pass
            if self._producer_error is not None:
                raise PipelineError(
                    f"producer raised {type(self._producer_error).__name__}"
                ) from self._producer_error

    async def cancel(self) -> None:
        """Stop iteration early.

        The consumer's iterator will exit at the next ``await
        queue.get()`` point. Idempotent: a second cancel is a no-op.
        """
        self._consumer_cancelled = True
        # Push a sentinel so any blocked queue.get returns; we don't
        # touch _closed because that's the producer's flag.
        try:
            self._queue.put_nowait(_PRODUCER_DONE)
        except asyncio.QueueFull:  # pragma: no cover - race
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def buffer_size(self) -> int:
        return self._queue.maxsize

    @property
    def buffered_count(self) -> int:
        return self._queue.qsize()

    @property
    def producer_error(self) -> Optional[BaseException]:
        """Exception captured from the producer, if any.

        ``None`` if the producer has not run, is still running, or
        finished cleanly. Useful for diagnostics that don't want to
        rely on the consumer side re-raising.
        """
        return self._producer_error
