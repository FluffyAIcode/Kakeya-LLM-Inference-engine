"""Integration tests for :class:`Scheduler`.

PR-N2 migration of the former Linux-side ``test_scheduler.py``,
replacing ``DeterministicEngine`` + ``DeterministicTokenizer``
test doubles with the real :class:`SpeculativeEngine` over
Qwen3-0.6B.

Tests of pure validation (empty prompt, zero max_new_tokens, empty
EOS) live on the Linux gate as
``tests/inference_engine/scheduler/test_scheduler_validation.py`` —
those reject before the engine is touched.

Acceptance: structural invariants (state transitions, slab acquire/
release, admission control behavior, concurrency) — NOT specific
token counts, since real-engine output varies with the prompt.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.config import AdmissionPolicy, SchedulerConfig
from inference_engine.scheduler.scheduler import RequestRejected, Scheduler
from inference_engine.scheduler.session import SessionState

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures: real engine + matching slab pool dims.
# ---------------------------------------------------------------------------


@pytest.fixture
def slab_config(real_speculative_engine):
    """Slab dims must match the real verifier so byte accounting
    plumbed through (sink+window) capacity is meaningful."""
    cfg = real_speculative_engine.tokenizer
    del cfg
    return SlabConfig(
        num_layers=2, num_heads=2, sink_size=1,
        window_size=4, head_dim=16, dtype=torch.bfloat16,
    )


@pytest.fixture
def small_pool(slab_config):
    return SlabPool(num_slabs=3, slab_config=slab_config)


@pytest.fixture
def reject_scheduler(real_speculative_engine, small_pool):
    return Scheduler(
        engine=real_speculative_engine, pool=small_pool,
        config=SchedulerConfig(
            max_concurrent=small_pool.total_count,
            admission_policy=AdmissionPolicy.REJECT,
        ),
    )


@pytest.fixture
def queue_scheduler(real_speculative_engine, small_pool):
    return Scheduler(
        engine=real_speculative_engine, pool=small_pool,
        config=SchedulerConfig(
            max_concurrent=small_pool.total_count,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=10.0,
        ),
    )


def _short_prompt_ids(real_speculative_engine) -> list[int]:
    """A short prompt the real engine can prefill. Uses the
    verifier's tokenizer to encode 'Hi.' which round-trips to a
    handful of tokens."""
    return real_speculative_engine.tokenizer.encode(
        "Hi.", add_special_tokens=False,
    )


def _eos_ids(real_speculative_engine) -> list[int]:
    eos = real_speculative_engine.tokenizer.eos_token_id
    return [int(eos)] if eos is not None else []


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


async def test_construction_validates_pool_size_match(
    real_speculative_engine, slab_config,
):
    pool = SlabPool(num_slabs=2, slab_config=slab_config)
    with pytest.raises(ValueError, match="does not match pool.total_count"):
        Scheduler(
            engine=real_speculative_engine, pool=pool,
            config=SchedulerConfig(max_concurrent=4),
        )


async def test_construction_with_matching_pool_size_works(
    real_speculative_engine, slab_config,
):
    pool = SlabPool(num_slabs=3, slab_config=slab_config)
    sch = Scheduler(
        engine=real_speculative_engine, pool=pool,
        config=SchedulerConfig(max_concurrent=3),
    )
    assert sch.active_count == 0
    assert sch.pending_count == 0


# ---------------------------------------------------------------------------
# Submit + iter_tokens (happy path)
# ---------------------------------------------------------------------------


async def test_single_session_runs_to_completion(
    reject_scheduler, real_speculative_engine,
):
    session = await reject_scheduler.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=4,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    tokens = []
    async for t in reject_scheduler.iter_tokens(session):
        tokens.append(t)
    # At least one token was emitted, session reached COMPLETED.
    assert len(tokens) >= 1
    assert session.state is SessionState.COMPLETED
    assert reject_scheduler.stats.total_completed == 1


async def test_session_admitted_at_set_after_submit(
    reject_scheduler, real_speculative_engine,
):
    session = await reject_scheduler.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=4,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    assert session.state is SessionState.ADMITTED
    assert session.admitted_at is not None
    # Drain so the worker's finally runs.
    async for _ in reject_scheduler.iter_tokens(session):
        pass


async def test_pool_slab_released_after_completion(
    reject_scheduler, small_pool, real_speculative_engine,
):
    session = await reject_scheduler.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=4,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    async for _ in reject_scheduler.iter_tokens(session):
        pass
    await asyncio.sleep(0.01)
    assert small_pool.in_use_count == 0


# ---------------------------------------------------------------------------
# Admission control: REJECT
# ---------------------------------------------------------------------------


async def test_reject_when_pool_exhausted(
    real_speculative_engine, slab_config,
):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=real_speculative_engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=1, admission_policy=AdmissionPolicy.REJECT,
        ),
    )
    s1 = await sch.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=8,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    # Second submit while first holds the only slab → reject.
    with pytest.raises(RequestRejected, match="slab pool exhausted"):
        await sch.submit(
            prompt_ids=_short_prompt_ids(real_speculative_engine),
            max_new_tokens=8,
            eos_token_ids=_eos_ids(real_speculative_engine),
        )
    async for _ in sch.iter_tokens(s1):
        pass
    await asyncio.sleep(0.01)
    assert sch.stats.total_rejected == 1


# ---------------------------------------------------------------------------
# Admission control: QUEUE
# ---------------------------------------------------------------------------


async def test_queue_policy_admits_after_first_completes(
    queue_scheduler, real_speculative_engine,
):
    """Pool size 3, four submits → fourth waits and succeeds."""
    sessions = []
    for _ in range(4):
        s = await queue_scheduler.submit(
            prompt_ids=_short_prompt_ids(real_speculative_engine),
            max_new_tokens=4,
            eos_token_ids=_eos_ids(real_speculative_engine),
        )
        sessions.append(s)
    for s in sessions:
        async for _ in queue_scheduler.iter_tokens(s):
            pass
    assert all(s.state is SessionState.COMPLETED for s in sessions)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_session_terminates_iteration(
    real_speculative_engine, slab_config,
):
    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=real_speculative_engine, pool=pool,
        config=SchedulerConfig(max_concurrent=1),
    )
    session = await sch.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=16,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )

    async def cancel_after_first_token():
        seen = 0
        async for _ in sch.iter_tokens(session):
            seen += 1
            if seen >= 1:
                await sch.cancel_session(session)
        return seen

    seen = await cancel_after_first_token()
    assert seen >= 1
    assert session.state is SessionState.CANCELLED


async def test_cancel_idempotent_after_completion(
    reject_scheduler, real_speculative_engine,
):
    session = await reject_scheduler.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=4,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    async for _ in reject_scheduler.iter_tokens(session):
        pass
    await reject_scheduler.cancel_session(session)
    assert session.state is SessionState.COMPLETED


# ---------------------------------------------------------------------------
# Engine errors propagate
# ---------------------------------------------------------------------------


async def test_engine_error_marks_session_failed(
    real_speculative_engine, slab_config,
):
    """Wrap the real engine in a one-shot error injector; the
    scheduler must propagate the failure into FAILED state and
    release the slab. The wrapper is a parametric error injector
    (composition over the real engine), not a state-mirror double."""

    class _RaiseOnGenerate:
        """One-shot error injector wrapping a real engine.

        Per PR-N1 / PR-N2 the principle is "no state-mirror test
        doubles". A composition wrapper that delegates EVERYTHING
        except a single raise is parametric error injection — the
        same pattern PR-N1 used for gRPC error-mapping tests.
        """
        def __init__(self, inner):
            self._inner = inner

        @property
        def tokenizer(self):
            return self._inner.tokenizer

        @property
        def model_id_label(self):
            return self._inner.model_id_label

        def generate(self, *_args, **_kw):
            raise RuntimeError("synthetic engine failure")

    pool = SlabPool(num_slabs=1, slab_config=slab_config)
    sch = Scheduler(
        engine=_RaiseOnGenerate(real_speculative_engine),
        pool=pool,
        config=SchedulerConfig(max_concurrent=1),
    )
    session = await sch.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=4,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    with pytest.raises(RuntimeError, match="synthetic engine failure"):
        async for _ in sch.iter_tokens(session):
            pass
    assert session.state is SessionState.FAILED
    assert isinstance(session.error, RuntimeError)
    await asyncio.sleep(0.01)
    assert pool.in_use_count == 0
    assert sch.stats.total_failed == 1


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_three_concurrent_submits_all_complete(
    reject_scheduler, real_speculative_engine,
):
    async def run_one():
        s = await reject_scheduler.submit(
            prompt_ids=_short_prompt_ids(real_speculative_engine),
            max_new_tokens=4,
            eos_token_ids=_eos_ids(real_speculative_engine),
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
# Shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_cancels_active_and_pending(
    real_speculative_engine, slab_config,
):
    pool = SlabPool(num_slabs=2, slab_config=slab_config)
    sch = Scheduler(
        engine=real_speculative_engine, pool=pool,
        config=SchedulerConfig(
            max_concurrent=2,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=0.0,
        ),
    )
    a1 = await sch.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=16,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    a2 = await sch.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=16,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    pending_task = asyncio.create_task(
        sch.submit(
            prompt_ids=_short_prompt_ids(real_speculative_engine),
            max_new_tokens=16,
            eos_token_ids=_eos_ids(real_speculative_engine),
        )
    )
    await asyncio.sleep(0.01)
    assert sch.pending_count == 1

    await sch.shutdown()

    assert a1.state is SessionState.CANCELLED
    assert a2.state is SessionState.CANCELLED
    with pytest.raises(RequestRejected, match="shutting down"):
        await pending_task

    async for _ in sch.iter_tokens(a1):
        pass
    async for _ in sch.iter_tokens(a2):
        pass


# ---------------------------------------------------------------------------
# Active count tracks state machine
# ---------------------------------------------------------------------------


async def test_active_count_zero_after_drain(
    reject_scheduler, real_speculative_engine,
):
    s = await reject_scheduler.submit(
        prompt_ids=_short_prompt_ids(real_speculative_engine),
        max_new_tokens=4,
        eos_token_ids=_eos_ids(real_speculative_engine),
    )
    async for _ in reject_scheduler.iter_tokens(s):
        pass
    await asyncio.sleep(0.01)
    assert reject_scheduler.active_count == 0
