"""End-to-end test: HTTP route → Scheduler → real engine.

Verifies the integration introduced by the E2 ↔ E4 wire-up commit on
real Qwen3 + dllm-hub: the FastAPI app constructed with
``max_concurrent=1`` rejects a second concurrent request with HTTP
429, and the lifespan context cleanly drains the scheduler.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from inference_engine.scheduler.config import AdmissionPolicy
from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig

pytestmark = pytest.mark.asyncio


async def test_real_engine_route_rejects_concurrent_with_429(
    real_speculative_engine,
):
    """First request holds the only slab; second returns 429."""
    app = create_app(real_speculative_engine, ServerConfig(
        max_concurrent=1,
        admission_policy=AdmissionPolicy.REJECT,
    ))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://t", timeout=120.0,
    ) as c:
        first = asyncio.create_task(c.post("/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Tell me a longer story."}],
            "max_tokens": 64,
        }))
        # Give the first request a tick to acquire the slab.
        await asyncio.sleep(0.1)
        second = await c.post("/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
        })
        assert second.status_code == 429
        # Drain the first.
        first_resp = await first
        assert first_resp.status_code == 200


async def test_real_engine_route_queue_admits_after_first_finishes(
    real_speculative_engine,
):
    """Under QUEUE policy two concurrent requests both succeed."""
    app = create_app(real_speculative_engine, ServerConfig(
        max_concurrent=1,
        admission_policy=AdmissionPolicy.QUEUE,
        queue_max_wait_s=120.0,
    ))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://t", timeout=120.0,
    ) as c:
        responses = await asyncio.gather(
            c.post("/v1/chat/completions", json={
                "model": "kakeya-system-test",
                "messages": [{"role": "user", "content": "Hi a"}],
                "max_tokens": 4,
            }),
            c.post("/v1/chat/completions", json={
                "model": "kakeya-system-test",
                "messages": [{"role": "user", "content": "Hi b"}],
                "max_tokens": 4,
            }),
        )
    assert all(r.status_code == 200 for r in responses)


async def test_real_engine_lifespan_shutdown_drains(real_speculative_engine):
    """When the FastAPI lifespan exits, scheduler.shutdown drains
    even if a session is in flight against the real engine."""
    app = create_app(real_speculative_engine, ServerConfig(max_concurrent=2))
    scheduler = app.state.scheduler

    async with app.router.lifespan_context(app):
        # Submit but do not drain — simulates client at server-shutdown.
        prompt_ids = real_speculative_engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
            enable_thinking=False,
        )
        eos = []
        if real_speculative_engine.tokenizer.eos_token_id is not None:
            eos.append(int(real_speculative_engine.tokenizer.eos_token_id))
        session = await scheduler.submit(
            prompt_ids=prompt_ids, max_new_tokens=64, eos_token_ids=eos,
        )
        assert scheduler.active_count == 1
        _ = session

    # After lifespan exit:
    await asyncio.sleep(0.1)
    assert app.state.pool.in_use_count == 0
