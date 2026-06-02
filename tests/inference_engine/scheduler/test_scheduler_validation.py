"""Linux-side validation tests for :class:`Scheduler`.

The scheduler's argument-validation paths in ``Scheduler.submit``
(empty prompt, non-positive max_new_tokens, empty EOS) reject
**before** the engine is touched. They need no engine instance —
``engine=None`` is safe because the validation runs at the entry
of ``submit`` and the engine is only consulted later inside the
worker task that ``submit`` enqueues for an admitted session.

Per PR-N2's no-doubles split, this Linux file replaces the
former ``DeterministicEngine``-driven tests in test_scheduler.py;
the engine-dependent paths (admission control, lifecycle,
cancellation, concurrency, shutdown) moved to
``tests/integration/test_scheduler_real.py``.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.config import AdmissionPolicy, SchedulerConfig
from inference_engine.scheduler.scheduler import Scheduler

pytestmark = pytest.mark.asyncio


@pytest.fixture
def slab_config() -> SlabConfig:
    return SlabConfig(
        num_layers=2, num_heads=2, sink_size=1,
        window_size=2, head_dim=4, dtype=torch.float32,
    )


@pytest.fixture
def small_pool(slab_config: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=3, slab_config=slab_config)


# ---------------------------------------------------------------------------
# Construction validation (engine-independent: only checks pool dims).
# ---------------------------------------------------------------------------


async def test_construction_validates_pool_size_match(slab_config):
    pool = SlabPool(num_slabs=2, slab_config=slab_config)
    with pytest.raises(ValueError, match="does not match pool.total_count"):
        Scheduler(
            engine=None, pool=pool,
            config=SchedulerConfig(max_concurrent=4),
        )


async def test_construction_with_matching_pool_size_works(slab_config):
    pool = SlabPool(num_slabs=3, slab_config=slab_config)
    sch = Scheduler(
        engine=None, pool=pool,
        config=SchedulerConfig(max_concurrent=3),
    )
    assert sch.active_count == 0
    assert sch.pending_count == 0


# ---------------------------------------------------------------------------
# submit() argument validation. Reject before the engine is consulted.
# ---------------------------------------------------------------------------


async def test_submit_rejects_empty_prompt(small_pool):
    sch = Scheduler(
        engine=None, pool=small_pool,
        config=SchedulerConfig(max_concurrent=small_pool.total_count),
    )
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        await sch.submit(
            prompt_ids=[], max_new_tokens=10, eos_token_ids=[0],
        )


async def test_submit_rejects_zero_max_tokens(small_pool):
    sch = Scheduler(
        engine=None, pool=small_pool,
        config=SchedulerConfig(max_concurrent=small_pool.total_count),
    )
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        await sch.submit(
            prompt_ids=[1], max_new_tokens=0, eos_token_ids=[0],
        )


async def test_submit_rejects_empty_eos(small_pool):
    sch = Scheduler(
        engine=None, pool=small_pool,
        config=SchedulerConfig(max_concurrent=small_pool.total_count),
    )
    with pytest.raises(ValueError, match="eos_token_ids must be non-empty"):
        await sch.submit(
            prompt_ids=[1], max_new_tokens=10, eos_token_ids=[],
        )
