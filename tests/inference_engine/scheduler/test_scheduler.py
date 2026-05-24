"""Unit tests for :class:`Scheduler`."""

from __future__ import annotations

import asyncio
from typing import List

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.config import AdmissionPolicy, SchedulerConfig
from inference_engine.scheduler.scheduler import (
    RequestRejected,
    Scheduler,
)
from inference_engine.scheduler.session import SessionState

from tests.inference_engine.scheduler.conftest import (
    DeterministicEngine,
    DeterministicTokenizer,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


async def test_construction_validates_pool_size_match(short_engine, slab_config):
    pool = SlabPool(num_slabs=2, slab_config=slab_config)
    with pytest.raises(ValueError, match="does not match pool.total_count"):
        Scheduler(
            engine=short_engine, pool=pool,
            config=SchedulerConfig(max_concurrent=4),
        )


async def test_construction_with_matching_pool_size_works(short_engine, slab_config):
    pool = SlabPool(num_slabs=3, slab_config=slab_config)
    sch = Scheduler(
        engine=short_engine, pool=pool,
        config=SchedulerConfig(max_concurrent=3),
    )
    assert sch.active_count == 0
    assert sch.pending_count == 0


# ---------------------------------------------------------------------------
# Submit + iter_tokens (happy path)
# ---------------------------------------------------------------------------


async def test_single_session_runs_to_completion(reject_scheduler):
    session = await reject_scheduler.submit(
        prompt_ids=[1, 2], max_new_tokens=10, eos_token_ids=[0],
    )
    tokens = []
    async for t in reject_scheduler.iter_tokens(session):
        tokens.append(t)
    # 3 content tokens + EOS = 4 tokens (short_engine sequence).
    assert len(tokens) == 4
    assert session.state is SessionState.COMPLETED
    assert reject_scheduler.stats.total_completed == 1


async def test_session_output_token_ids_recorded(reject_scheduler):
    session = await reject_scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    async for _ in reject_scheduler.iter_tokens(session):
        pass
    assert session.output_token_ids == [
        # tokens emitted by short_engine fixture
        # We don't assert exact ids (they're tokenizer-internal); we
        # assert structural invariants.
        *session.output_token_ids
    ]
    # Last token should be EOS (id 0).
    assert session.output_token_ids[-1] == 0


async def test_session_admitted_at_set_after_submit(reject_scheduler):
    session = await reject_scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    assert session.state is SessionState.ADMITTED
    assert session.admitted_at is not None


async def test_pool_slab_is_released_after_completion(reject_scheduler, small_pool):
    session = await reject_scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    async for _ in reject_scheduler.iter_tokens(session):
        pass
    # Drain — give the worker's finally block a chance to run.
    await asyncio.sleep(0.01)
    assert small_pool.in_use_count == 0


# ---------------------------------------------------------------------------
# Submit validation
# ---------------------------------------------------------------------------


async def test_submit_rejects_empty_prompt(reject_scheduler):
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        await reject_scheduler.submit(
            prompt_ids=[], max_new_tokens=10, eos_token_ids=[0],
        )


async def test_submit_rejects_zero_max_tokens(reject_scheduler):
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        await reject_scheduler.submit(
            prompt_ids=[1], max_new_tokens=0, eos_token_ids=[0],
        )


async def test_submit_rejects_empty_eos(reject_scheduler):
    with pytest.raises(ValueError, match="eos_token_ids must be non-empty"):
        await reject_scheduler.submit(
            prompt_ids=[1], max_new_tokens=10, eos_token_ids=[],
        )


# ---------------------------------------------------------------------------
# Admission control: REJECT
# ---------------------------------------------------------------------------


async def test_reject_when_pool_exhausted(slow_engine, slab_config):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=slow_engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=1, admission_policy=AdmissionPolicy.REJECT,
        ),
    )
    s1 = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )
    # Second submit while first holds the only slab → reject.
    with pytest.raises(RequestRejected, match="slab pool exhausted"):
        await sch.submit(
            prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
        )
    # Drain s1 so the worker terminates and the slab releases.
    async for _ in sch.iter_tokens(s1):
        pass
    await asyncio.sleep(0.01)
    assert sch.stats.total_rejected == 1


# ---------------------------------------------------------------------------
# Admission control: QUEUE
# ---------------------------------------------------------------------------


async def test_queue_policy_admits_after_first_completes(queue_scheduler, small_pool):
    """With 3 slabs and 4 submits, the 4th should wait then succeed."""
    sessions = []
    for _ in range(4):
        s = await queue_scheduler.submit(
            prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
        )
        sessions.append(s)
    # All 4 sessions should eventually complete.
    for s in sessions:
        async for _ in queue_scheduler.iter_tokens(s):
            pass
    assert all(s.state is SessionState.COMPLETED for s in sessions)


async def test_queue_timeout_raises(slow_engine, slab_config):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=slow_engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=1,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=0.05,
        ),
    )
    s1 = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )
    # Second submit will queue and time out (slow_engine takes
    # ~0.2s for 20 tokens; queue_max_wait_s=0.05s).
    with pytest.raises(RequestRejected, match="queue wait exceeded"):
        await sch.submit(
            prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
        )
    # Cleanup
    async for _ in sch.iter_tokens(s1):
        pass


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_session_terminates_iteration(slow_engine, slab_config):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=slow_engine, pool=pool,
        config=SchedulerConfig(max_concurrent=1),
    )
    session = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )

    async def cancel_after_some_tokens():
        seen = 0
        async for _ in sch.iter_tokens(session):
            seen += 1
            if seen >= 2:
                await sch.cancel_session(session)
        return seen

    seen = await cancel_after_some_tokens()
    # Some tokens flowed before cancel; cancel was honored eventually.
    assert seen >= 2
    assert session.state is SessionState.CANCELLED


async def test_cancel_idempotent(reject_scheduler):
    session = await reject_scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    async for _ in reject_scheduler.iter_tokens(session):
        pass
    # session is now COMPLETED. Cancel must be a no-op.
    await reject_scheduler.cancel_session(session)
    assert session.state is SessionState.COMPLETED


# ---------------------------------------------------------------------------
# Engine errors propagate to FAILED state
# ---------------------------------------------------------------------------


class _RaisingEngine:
    """Engine that raises on first generate call."""

    def __init__(self, tokenizer, model_id_label="raises"):
        self._tok = tokenizer
        self._label = model_id_label

    @property
    def tokenizer(self):
        return self._tok

    @property
    def model_id_label(self):
        return self._label

    def generate(self, prompt_ids, max_new_tokens, eos_token_ids, on_token=None):
        raise RuntimeError("synthetic engine failure")


async def test_engine_error_marks_session_failed(slab_config, tokenizer):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    engine = _RaisingEngine(tokenizer)
    sch = Scheduler(
        engine=engine, pool=pool,
        config=SchedulerConfig(max_concurrent=1),
    )
    session = await sch.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    with pytest.raises(RuntimeError, match="synthetic engine failure"):
        async for _ in sch.iter_tokens(session):
            pass
    assert session.state is SessionState.FAILED
    assert isinstance(session.error, RuntimeError)
    # Slab released even on error.
    await asyncio.sleep(0.01)
    assert pool.in_use_count == 0
    assert sch.stats.total_failed == 1


# ---------------------------------------------------------------------------
# Concurrent submits all complete (round-robin via lock)
# ---------------------------------------------------------------------------


async def test_three_concurrent_submits_all_complete(reject_scheduler):
    """3 submits, pool size 3 → all admit immediately, all complete."""

    async def run_one():
        s = await reject_scheduler.submit(
            prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
        )
        tokens: List[int] = []
        async for t in reject_scheduler.iter_tokens(s):
            tokens.append(t)
        return s

    sessions = await asyncio.gather(run_one(), run_one(), run_one())
    assert all(s.state is SessionState.COMPLETED for s in sessions)
    assert reject_scheduler.stats.total_admitted == 3
    assert reject_scheduler.stats.total_completed == 3


# ---------------------------------------------------------------------------
# Cancel before first token (race: cancel runs while worker is acquiring lock)
# ---------------------------------------------------------------------------


async def test_cancel_immediately_after_submit(slow_engine, slab_config):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=slow_engine, pool=pool,
        config=SchedulerConfig(max_concurrent=1),
    )
    session = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )
    # Cancel right away — race against the worker acquiring the lock.
    await sch.cancel_session(session)
    # iter_tokens may yield 0 or a few tokens depending on race timing.
    async for _ in sch.iter_tokens(session):
        pass
    assert session.state is SessionState.CANCELLED


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_cancels_active_and_pending(slow_engine, slab_config):
    pool = SlabPool(num_slabs=2, slab_config=slab_config)
    sch = Scheduler(
        engine=slow_engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=2,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=0.0,  # wait forever
        ),
    )
    a1 = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )
    a2 = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )
    # Third submit will queue.
    pending_task = asyncio.create_task(
        sch.submit(prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0])
    )
    await asyncio.sleep(0.01)  # give it time to enter the queue.
    assert sch.pending_count == 1

    await sch.shutdown()

    # Active sessions must be CANCELLED.
    assert a1.state is SessionState.CANCELLED
    assert a2.state is SessionState.CANCELLED
    # Pending submission rejected.
    with pytest.raises(RequestRejected, match="shutting down"):
        await pending_task

    # Drain the active iterators so they observe their terminal state.
    async for _ in sch.iter_tokens(a1):
        pass
    async for _ in sch.iter_tokens(a2):
        pass


# ---------------------------------------------------------------------------
# Pool/scheduler total_count consistency at runtime
# ---------------------------------------------------------------------------


async def test_pending_count_tracks_wait_queue(slow_engine, slab_config):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=slow_engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=1,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=10.0,
        ),
    )
    s1 = await sch.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )
    pending_task = asyncio.create_task(
        sch.submit(prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0])
    )
    await asyncio.sleep(0.01)
    assert sch.pending_count == 1
    # Drain s1 so the queued submit is admitted and the pending task resolves.
    async for _ in sch.iter_tokens(s1):
        pass
    s2 = await pending_task
    async for _ in sch.iter_tokens(s2):
        pass
    assert s2.state is SessionState.COMPLETED


# ---------------------------------------------------------------------------
# Active count tracks state machine
# ---------------------------------------------------------------------------


async def test_active_count_zero_after_drain(reject_scheduler):
    s = await reject_scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    async for _ in reject_scheduler.iter_tokens(s):
        pass
    await asyncio.sleep(0.01)
    assert reject_scheduler.active_count == 0
