"""Scheduler + real engine system tests.

Wraps the real :class:`SpeculativeEngine` in a :class:`Scheduler` and
runs concurrent submissions through it. Verifies admission control,
slab acquisition / release, and per-session lifecycle on real
generation traffic.

Slow: each submitted session generates real tokens. Tests cap
``max_new_tokens`` aggressively (8-16) to keep wall time reasonable.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.config import AdmissionPolicy, SchedulerConfig
from inference_engine.scheduler.scheduler import Scheduler
from inference_engine.scheduler.session import SessionState

pytestmark = pytest.mark.asyncio


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _encode_chat(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


@pytest.fixture
def slab_pool():
    """Tiny pool — system tests verify admission control, not real
    KV-bandwidth scaling."""
    cfg = SlabConfig(
        num_layers=2, num_heads=2, sink_size=1,
        window_size=4, head_dim=16, dtype=torch.bfloat16,
    )
    return SlabPool(num_slabs=2, slab_config=cfg)


@pytest.fixture
def scheduler(real_speculative_engine, slab_pool):
    return Scheduler(
        engine=real_speculative_engine, pool=slab_pool,
        config=SchedulerConfig(
            max_concurrent=slab_pool.total_count,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=120.0,
        ),
    )


# ---------------------------------------------------------------------------
# Single-session round trip
# ---------------------------------------------------------------------------


async def test_single_submission_produces_tokens(real_speculative_engine, scheduler):
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(tokenizer, "Hi")
    eos = _eos_ids(tokenizer)
    session = await scheduler.submit(
        prompt_ids=prompt_ids, max_new_tokens=8, eos_token_ids=eos,
    )
    tokens = []
    async for tok in scheduler.iter_tokens(session):
        tokens.append(tok)
    assert len(tokens) > 0
    assert session.state is SessionState.COMPLETED


async def test_session_output_token_ids_match_iter_tokens(
    real_speculative_engine, scheduler,
):
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(tokenizer, "Hi")
    eos = _eos_ids(tokenizer)
    session = await scheduler.submit(
        prompt_ids=prompt_ids, max_new_tokens=8, eos_token_ids=eos,
    )
    streamed = []
    async for tok in scheduler.iter_tokens(session):
        streamed.append(tok)
    assert session.output_token_ids == streamed


# ---------------------------------------------------------------------------
# Concurrent submissions exercise scheduler's admission + serialization
# ---------------------------------------------------------------------------


async def test_two_concurrent_submissions_both_complete(
    real_speculative_engine, scheduler,
):
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(tokenizer, "Hi")
    eos = _eos_ids(tokenizer)

    async def run_one():
        session = await scheduler.submit(
            prompt_ids=prompt_ids, max_new_tokens=4, eos_token_ids=eos,
        )
        async for _ in scheduler.iter_tokens(session):
            pass
        return session

    a, b = await asyncio.gather(run_one(), run_one())
    assert a.state is SessionState.COMPLETED
    assert b.state is SessionState.COMPLETED
    assert scheduler.stats.total_completed == 2


async def test_third_submission_queues_when_pool_full(
    real_speculative_engine, scheduler, slab_pool,
):
    """Pool size 2; submit 3, third must queue and eventually complete."""
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(tokenizer, "Hi")
    eos = _eos_ids(tokenizer)

    async def run_one():
        session = await scheduler.submit(
            prompt_ids=prompt_ids, max_new_tokens=4, eos_token_ids=eos,
        )
        async for _ in scheduler.iter_tokens(session):
            pass
        return session

    sessions = await asyncio.gather(run_one(), run_one(), run_one())
    assert all(s.state is SessionState.COMPLETED for s in sessions)
    # All slabs released after work.
    assert slab_pool.in_use_count == 0


# ---------------------------------------------------------------------------
# Cancellation against real generation
# ---------------------------------------------------------------------------


async def test_cancellation_terminates_real_generation(
    real_speculative_engine, scheduler,
):
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(
        tokenizer, "Tell me a long story with many details.",
    )
    eos = _eos_ids(tokenizer)
    session = await scheduler.submit(
        prompt_ids=prompt_ids, max_new_tokens=128, eos_token_ids=eos,
    )

    seen = 0
    async for _ in scheduler.iter_tokens(session):
        seen += 1
        if seen >= 2:
            await scheduler.cancel_session(session)
    # Cancellation honored before max_new_tokens reached.
    assert session.state is SessionState.CANCELLED
    assert seen < 128
