"""Unit tests for :class:`PipelineCoordinator`.

Tests use real asyncio coroutines and tasks — no mocks. Producer
coroutines are tiny inline async functions that exercise the full
queue / cancel / error / close lifecycle without dragging the
speculative decoder or HF tokenizer into the picture.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from inference_engine.pipeline.coordinator import (
    PipelineClosed,
    PipelineCoordinator,
    PipelineError,
    StreamSentinel,
    _PRODUCER_DONE,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


async def test_construction_default_buffer_size():
    c = PipelineCoordinator()
    assert c.buffer_size == 64


async def test_construction_explicit_buffer_size():
    c = PipelineCoordinator(buffer_size=8)
    assert c.buffer_size == 8


@pytest.mark.parametrize("size", [0, -1, -100])
async def test_construction_rejects_non_positive(size):
    with pytest.raises(ValueError, match="buffer_size must be positive"):
        PipelineCoordinator(buffer_size=size)


# ---------------------------------------------------------------------------
# Happy path: produce and consume
# ---------------------------------------------------------------------------


async def test_produce_and_consume_fifo_order():
    c: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=4)

    async def produce():
        for i in range(5):
            await c.put(i)

    c.start_producer(produce())
    out: List[int] = []
    async for item in c.consume():
        out.append(item)
    assert out == [0, 1, 2, 3, 4]


async def test_consume_terminates_after_close():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        await c.put(42)
        # Producer wrapper auto-closes on return; explicit close also OK.

    c.start_producer(produce())
    out = []
    async for item in c.consume():
        out.append(item)
    assert out == [42]


async def test_close_is_idempotent():
    c: PipelineCoordinator[int] = PipelineCoordinator()
    await c.close()
    await c.close()
    assert c.is_closed is True


async def test_put_after_close_raises():
    c: PipelineCoordinator[int] = PipelineCoordinator()
    await c.close()
    with pytest.raises(PipelineClosed):
        await c.put(99)


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


async def test_buffered_count_grows_with_puts():
    c: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=4)
    await c.put(1)
    await c.put(2)
    assert c.buffered_count == 2


async def test_full_buffer_blocks_put():
    """A full buffer pauses the producer until a consumer drains."""
    c: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=2)
    await c.put(1)
    await c.put(2)
    # Schedule a put that should block (buffer is full).
    blocked_task = asyncio.create_task(c.put(3))
    await asyncio.sleep(0.01)
    assert not blocked_task.done()
    # Drain one item — put should now complete.
    item = await c._queue.get()
    assert item == 1
    await asyncio.wait_for(blocked_task, timeout=0.5)


# ---------------------------------------------------------------------------
# Producer error propagation
# ---------------------------------------------------------------------------


async def test_producer_error_propagates_to_consumer():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        await c.put(1)
        raise RuntimeError("synthetic producer crash")

    c.start_producer(produce())
    out = []
    with pytest.raises(PipelineError):
        async for item in c.consume():
            out.append(item)
    # The valid item before the crash was still delivered.
    assert out == [1]


async def test_producer_error_chains_original_via_cause():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        raise ValueError("kaboom")

    c.start_producer(produce())
    with pytest.raises(PipelineError) as excinfo:
        async for _ in c.consume():
            pass
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "kaboom" in str(excinfo.value.__cause__)


async def test_producer_error_observable_via_property():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        raise KeyError("k")

    c.start_producer(produce())
    # Drain (which raises) so we're past the producer.
    with pytest.raises(PipelineError):
        async for _ in c.consume():
            pass
    assert isinstance(c.producer_error, KeyError)


async def test_producer_error_is_none_on_clean_run():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        await c.put(1)

    c.start_producer(produce())
    async for _ in c.consume():
        pass
    assert c.producer_error is None


# ---------------------------------------------------------------------------
# Consumer cancellation
# ---------------------------------------------------------------------------


async def test_cancel_stops_iteration():
    c: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=10)

    async def produce():
        for i in range(100):
            try:
                await c.put(i)
            except PipelineClosed:
                break

    c.start_producer(produce())
    seen: List[int] = []
    gen = c.consume()
    async for item in gen:
        seen.append(item)
        if len(seen) >= 3:
            await c.cancel()
            break
    # Explicit aclose so the finally block runs and drains.
    await gen.aclose()
    assert len(seen) >= 3


async def test_cancel_drains_blocked_producer():
    """Producer is blocked on full buffer; cancel must drain the
    queue so the producer's pending put can complete and the
    producer task can exit cleanly.

    Exercises the queue-drain path in :meth:`consume`'s finally
    block."""
    c: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=2)
    producer_finished = asyncio.Event()

    async def produce():
        try:
            for i in range(100):
                await c.put(i)
        except PipelineClosed:
            pass
        finally:
            producer_finished.set()

    c.start_producer(produce())
    seen: List[int] = []
    gen = c.consume()
    async for item in gen:
        seen.append(item)
        if len(seen) >= 1:
            await c.cancel()
            break
    # Explicitly close the generator to fire the finally block,
    # which drains the queue and unblocks the producer.
    await gen.aclose()
    # The producer must have exited (either normally on PipelineClosed
    # or because the queue drained enough to let it finish).
    assert producer_finished.is_set()


async def test_cancel_idempotent():
    c: PipelineCoordinator[int] = PipelineCoordinator()
    await c.cancel()
    await c.cancel()
    # Should not raise.


async def test_put_after_cancel_raises():
    c: PipelineCoordinator[int] = PipelineCoordinator()
    await c.cancel()
    with pytest.raises(PipelineClosed):
        await c.put(7)


# ---------------------------------------------------------------------------
# Producer task management
# ---------------------------------------------------------------------------


async def test_start_producer_returns_task():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        await c.put(1)

    task = c.start_producer(produce())
    assert isinstance(task, asyncio.Task)
    async for _ in c.consume():
        pass
    # Task is done after consume exhausts.
    assert task.done()


async def test_start_producer_twice_raises():
    c: PipelineCoordinator[int] = PipelineCoordinator()

    async def produce():
        await c.put(1)

    c.start_producer(produce())
    second = produce()
    try:
        with pytest.raises(RuntimeError, match="producer already started"):
            c.start_producer(second)
    finally:
        # close the un-awaited coroutine to suppress RuntimeWarning
        second.close()
    # Drain the first producer for clean teardown.
    async for _ in c.consume():
        pass


async def test_consume_with_no_producer_drains_close_only():
    """If close is called without a producer, consume returns no items
    and exits cleanly."""
    c: PipelineCoordinator[int] = PipelineCoordinator()
    await c.close()
    out = [item async for item in c.consume()]
    assert out == []


# ---------------------------------------------------------------------------
# Sentinel + isolation
# ---------------------------------------------------------------------------


async def test_stream_sentinel_marker_class():
    """``StreamSentinel`` is the marker base class; ``_PRODUCER_DONE``
    is an instance of it. The consume loop's `isinstance(item,
    StreamSentinel)` branch is what makes future sentinel additions
    forward-compatible."""
    assert isinstance(_PRODUCER_DONE, StreamSentinel)


async def test_buffered_items_drain_before_close_sentinel():
    """Items put before close are visible to the consumer; the close
    sentinel is processed only after they drain."""
    c: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=4)
    await c.put(1)
    await c.put(2)
    await c.put(3)
    await c.close()
    out: List[int] = []
    async for i in c.consume():
        out.append(i)
    assert out == [1, 2, 3]
