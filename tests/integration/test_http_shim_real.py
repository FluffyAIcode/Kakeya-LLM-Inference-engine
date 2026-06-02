"""Integration tests for the HTTP shim (``inference_engine.server.app``).

PR-N3 migration: replaces the former ``test_app_routes.py``,
``test_app_streaming.py``, ``test_app_with_scheduler.py``, and
``test_app_metrics_and_auth.py`` Linux-side suites that were driven
by ``DeterministicEngine`` + ``DeterministicTokenizer`` test doubles.

Coverage scope: ``inference_engine.server.app`` end-to-end against
the real :class:`SpeculativeEngine` over Qwen3-0.6B. Asserts on
HTTP layer correctness (OpenAI-compat shape, auth, error envelopes,
metrics emission, streaming), NOT on specific token output —
real-engine output varies; structural invariants are what matters.

The HTTP shim is feature-frozen per ADR 0008 §2.7 and slated for
refactor onto SessionStore in PR-D2; this test set is the minimum
that proves the route layer still wires through to a real engine
correctly until then.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures: a real-engine-backed FastAPI app per test for isolation.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_app(real_speculative_engine):
    return create_app(
        real_speculative_engine,
        ServerConfig(default_max_new_tokens=4),
    )


@pytest.fixture
def real_app_with_auth(real_speculative_engine):
    return create_app(
        real_speculative_engine,
        ServerConfig(
            default_max_new_tokens=4,
            api_keys=frozenset({"sk-test-secret"}),
        ),
    )


# ---------------------------------------------------------------------------
# /v1/chat/completions — happy path (non-streaming)
# ---------------------------------------------------------------------------


async def test_chat_completions_returns_openai_envelope(real_app):
    async with AsyncClient(
        transport=ASGITransport(app=real_app), base_url="http://t",
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [
                {"role": "user", "content": "Reply with one word."},
            ],
            "max_tokens": 4,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "id" in body
    assert "created" in body
    assert body["model"] == real_app.state.engine.model_id_label
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in {"stop", "length"}


async def test_chat_completions_rejects_empty_messages(real_app):
    async with AsyncClient(
        transport=ASGITransport(app=real_app), base_url="http://t",
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any", "messages": [],
        })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


async def test_chat_completions_rejects_unsupported_role(real_app):
    async with AsyncClient(
        transport=ASGITransport(app=real_app), base_url="http://t",
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "system_v9", "content": "x"}],
        })
    assert r.status_code in {400, 422}


# ---------------------------------------------------------------------------
# /v1/chat/completions — streaming (SSE)
# ---------------------------------------------------------------------------


async def test_chat_completions_streaming_yields_chunks_then_done(real_app):
    async with AsyncClient(
        transport=ASGITransport(app=real_app), base_url="http://t",
    ) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "user", "content": "Hi."}],
            "max_tokens": 4,
            "stream": True,
        }) as r:
            assert r.status_code == 200
            text = ""
            async for chunk in r.aiter_text():
                text += chunk
    # Final SSE marker present.
    assert "data: [DONE]" in text
    # At least one delta chunk before the marker.
    parts = [p for p in text.split("\n\n") if p.startswith("data: {")]
    assert len(parts) >= 1
    # The first content delta is a structural OpenAI chunk shape.
    first = json.loads(parts[0][len("data: "):])
    assert first["object"] == "chat.completion.chunk"
    assert "choices" in first


# ---------------------------------------------------------------------------
# Auth (API keys)
# ---------------------------------------------------------------------------


async def test_auth_required_returns_401_without_token(real_app_with_auth):
    async with AsyncClient(
        transport=ASGITransport(app=real_app_with_auth),
        base_url="http://t",
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "user", "content": "x"}],
        })
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


async def test_auth_succeeds_with_correct_token(real_app_with_auth):
    async with AsyncClient(
        transport=ASGITransport(app=real_app_with_auth),
        base_url="http://t",
    ) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "any",
                "messages": [{"role": "user", "content": "Hi."}],
                "max_tokens": 4,
            },
            headers={"Authorization": "Bearer sk-test-secret"},
        )
    assert r.status_code == 200


async def test_auth_rejects_wrong_token(real_app_with_auth):
    async with AsyncClient(
        transport=ASGITransport(app=real_app_with_auth),
        base_url="http://t",
    ) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "any",
                "messages": [{"role": "user", "content": "Hi."}],
            },
            headers={"Authorization": "Bearer sk-wrong"},
        )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# /metrics + /healthz — public, no auth required
# ---------------------------------------------------------------------------


async def test_healthz_does_not_require_auth(real_app_with_auth):
    async with AsyncClient(
        transport=ASGITransport(app=real_app_with_auth),
        base_url="http://t",
    ) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200


async def test_metrics_does_not_require_auth(real_app_with_auth):
    async with AsyncClient(
        transport=ASGITransport(app=real_app_with_auth),
        base_url="http://t",
    ) as c:
        r = await c.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


async def test_metrics_records_completion_after_request(real_app):
    async with AsyncClient(
        transport=ASGITransport(app=real_app), base_url="http://t",
    ) as c:
        await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "user", "content": "Hi."}],
            "max_tokens": 4,
        })
        m = await c.get("/metrics")
    assert "inference_completions_total" in m.text


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


async def test_models_endpoint_lists_engine_id(real_app):
    async with AsyncClient(
        transport=ASGITransport(app=real_app), base_url="http://t",
    ) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert any(
        m["id"] == real_app.state.engine.model_id_label
        for m in body["data"]
    )
