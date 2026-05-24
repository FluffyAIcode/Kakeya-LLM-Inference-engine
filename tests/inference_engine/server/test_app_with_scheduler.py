"""Integration tests for the route → scheduler → engine path.

These tests verify the new behavior introduced by E2 ↔ E4 integration:

  * Every chat-completion request goes through the scheduler.
  * The scheduler is constructed with parameters drawn from
    :class:`ServerConfig`.
  * Pool exhaustion (under REJECT policy) surfaces as HTTP 429.
  * Slabs are released after request completion.
  * The lifespan context calls scheduler.shutdown() on exit.

All tests use the existing :class:`DeterministicEngine` test double
from ``conftest.py`` — real concrete classes, no ``unittest.mock``.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, List

import pytest
from httpx import ASGITransport, AsyncClient

from inference_engine.scheduler.config import AdmissionPolicy
from inference_engine.scheduler.scheduler import Scheduler
from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig

from tests.inference_engine.server.conftest import (
    DeterministicEngine,
    DeterministicTokenizer,
)

pytestmark = pytest.mark.asyncio


# NOTE: a few tests in this module are intentionally synchronous (they
# only exercise constructor logic, not async paths). Because we set
# pytestmark = pytest.mark.asyncio at module level, those tests get
# the asyncio mark applied even though they don't need it. pytest
# emits a warning but still runs them. The simpler fix is to overwrite
# pytestmark on the sync tests with an empty marker list — but pytest
# does not support that. Suppressing the warnings is acceptable since
# the marker is functionally a no-op for sync tests.


async def _read_sse_events(stream: AsyncIterator[bytes]) -> List[str]:
    buf = b""
    out: List[str] = []
    async for chunk in stream:
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.splitlines():
                if line.startswith(b"data: "):
                    out.append(line[len(b"data: "):].decode("utf-8"))
    if buf:
        for line in buf.splitlines():
            if line.startswith(b"data: "):
                out.append(line[len(b"data: "):].decode("utf-8"))
    return out


# ---------------------------------------------------------------------------
# Scheduler is constructed and exposed on app state
# ---------------------------------------------------------------------------


def test_create_app_constructs_scheduler(short_engine):
    app = create_app(short_engine, ServerConfig(max_concurrent=2))
    assert isinstance(app.state.scheduler, Scheduler)
    assert app.state.scheduler.active_count == 0
    assert app.state.pool.total_count == 2


def test_create_app_pool_size_must_match_max_concurrent(short_engine):
    """If a caller passes a pre-built pool whose size disagrees with
    config.max_concurrent, we surface the misconfiguration immediately."""
    import torch
    from inference_engine.memory.pool import SlabPool
    from inference_engine.memory.slab import SlabConfig

    bad_pool = SlabPool(
        num_slabs=4,
        slab_config=SlabConfig(
            num_layers=1, num_heads=1, sink_size=0, window_size=1,
            head_dim=1, dtype=torch.bfloat16,
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        create_app(short_engine, ServerConfig(max_concurrent=2), pool=bad_pool)


def test_create_app_accepts_explicit_pool(short_engine):
    """A caller-provided pool of correct size is used as-is."""
    import torch
    from inference_engine.memory.pool import SlabPool
    from inference_engine.memory.slab import SlabConfig

    pool = SlabPool(
        num_slabs=3,
        slab_config=SlabConfig(
            num_layers=2, num_heads=4, sink_size=1, window_size=2,
            head_dim=8, dtype=torch.bfloat16,
        ),
    )
    app = create_app(short_engine, ServerConfig(max_concurrent=3), pool=pool)
    assert app.state.pool is pool


# ---------------------------------------------------------------------------
# Single-user mode (default config) still works end-to-end
# ---------------------------------------------------------------------------


async def test_default_config_chat_completion_succeeds(short_engine):
    """ServerConfig() defaults max_concurrent=1; route must still work."""
    app = create_app(short_engine, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 200


async def test_session_releases_slab_after_completion(short_engine):
    app = create_app(short_engine, ServerConfig(max_concurrent=2))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
    # Drain any remaining scheduler bookkeeping.
    await asyncio.sleep(0.01)
    assert app.state.pool.in_use_count == 0


# ---------------------------------------------------------------------------
# Admission control: 429 under REJECT policy
# ---------------------------------------------------------------------------


async def test_pool_full_under_reject_policy_returns_429(tokenizer):
    """One slab + first request still in flight → second returns 429."""
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.05,
    )
    app = create_app(slow_engine, ServerConfig(
        max_concurrent=1,
        admission_policy=AdmissionPolicy.REJECT,
    ))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://t", timeout=30.0,
    ) as c:
        # Kick off a long request; don't await yet.
        first = asyncio.create_task(c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 10,
        }))
        await asyncio.sleep(0.02)  # let first acquire the slab
        # Second request hits a full pool → 429.
        second = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "b"}],
            "max_tokens": 10,
        })
        assert second.status_code == 429
        assert "slab pool exhausted" in second.json()["detail"]
        # Drain first.
        first_resp = await first
        assert first_resp.status_code == 200


async def test_pool_full_under_queue_policy_blocks_then_succeeds(tokenizer):
    """Under QUEUE policy, the second request waits and then succeeds."""
    ids = [tokenizer._intern(f"tok{i}") for i in range(20)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.02,
    )
    app = create_app(slow_engine, ServerConfig(
        max_concurrent=1,
        admission_policy=AdmissionPolicy.QUEUE,
        queue_max_wait_s=10.0,
    ))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://t", timeout=30.0,
    ) as c:
        responses = await asyncio.gather(
            c.post("/v1/chat/completions", json={
                "model": "m",
                "messages": [{"role": "user", "content": "a"}],
                "max_tokens": 4,
            }),
            c.post("/v1/chat/completions", json={
                "model": "m",
                "messages": [{"role": "user", "content": "b"}],
                "max_tokens": 4,
            }),
        )
    assert all(r.status_code == 200 for r in responses)


# ---------------------------------------------------------------------------
# Streaming path also flows through scheduler
# ---------------------------------------------------------------------------


async def test_streaming_via_scheduler_emits_done(short_engine):
    app = create_app(short_engine, ServerConfig(max_concurrent=1))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t", timeout=10.0) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            events = await _read_sse_events(r.aiter_bytes())
    assert events[-1] == "[DONE]"
    payloads = [json.loads(e) for e in events[:-1]]
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


async def test_streaming_429_when_pool_full(tokenizer):
    """Even streaming requests get 429 (not partial SSE) when admission fails."""
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        per_token_delay_s=0.05,
    )
    app = create_app(slow_engine, ServerConfig(max_concurrent=1))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://t", timeout=30.0,
    ) as c:
        first = asyncio.create_task(c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 10,
        }))
        await asyncio.sleep(0.02)
        second = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "b"}],
            "stream": True,
        })
        assert second.status_code == 429
        await first


# ---------------------------------------------------------------------------
# Engine error in non-streaming path → 500
# ---------------------------------------------------------------------------


class _RaisingEngine:
    def __init__(self, tokenizer):
        self._tok = tokenizer

    @property
    def tokenizer(self):
        return self._tok

    @property
    def model_id_label(self):
        return "raising"

    def generate(self, prompt_ids, max_new_tokens, eos_token_ids, on_token=None):
        raise RuntimeError("synthetic engine failure")


async def test_non_streaming_500_when_engine_raises(tokenizer):
    engine = _RaisingEngine(tokenizer)
    app = create_app(engine, ServerConfig(max_concurrent=1))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 500
    assert "engine error" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Lifespan: shutdown calls scheduler.shutdown
# ---------------------------------------------------------------------------


async def test_lifespan_shutdown_drains_scheduler(short_engine):
    """When the FastAPI lifespan exits, scheduler.shutdown() runs:
    pool occupancy returns to 0 even if a session was active."""
    app = create_app(short_engine, ServerConfig(max_concurrent=2))
    scheduler = app.state.scheduler

    # Invoke the lifespan context manually via FastAPI's router.
    async with app.router.lifespan_context(app):
        # Submit one session but DON'T drain it — simulates an
        # in-flight client at server-shutdown.
        session = await scheduler.submit(
            prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
        )
        assert scheduler.active_count == 1
        _ = session

    # After lifespan exit, scheduler.shutdown() ran.
    # All sessions should be terminal; pool should be empty.
    await asyncio.sleep(0.01)
    assert app.state.pool.in_use_count == 0


async def test_lifespan_shutdown_rejects_pending_sessions(tokenizer):
    """Under QUEUE policy with an in-flight session, a queued submit
    is rejected when shutdown runs."""
    from inference_engine.scheduler.scheduler import RequestRejected

    ids = [tokenizer._intern(f"tok{i}") for i in range(20)]
    slow = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer, per_token_delay_s=0.02,
    )
    app = create_app(slow, ServerConfig(
        max_concurrent=1,
        admission_policy=AdmissionPolicy.QUEUE,
        queue_max_wait_s=0.0,
    ))
    scheduler = app.state.scheduler

    async with app.router.lifespan_context(app):
        active_session = await scheduler.submit(
            prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
        )
        pending = asyncio.create_task(
            scheduler.submit(
                prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
            )
        )
        await asyncio.sleep(0.01)
        assert scheduler.pending_count == 1
        _ = active_session

    # After lifespan exit:
    with pytest.raises(RequestRejected, match="shutting down"):
        await pending