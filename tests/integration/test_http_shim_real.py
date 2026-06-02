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
    # FastAPI / pydantic surface validation errors as 422 by default;
    # the route layer's request_validation_exception_handler also
    # returns 422 for the empty-messages case.
    assert r.status_code in {400, 422}
    body = r.json()
    # Error envelope shape per server.errors.STATUS_TYPE_MAP:
    #   400 -> "invalid_request_error"
    #   422 -> "invalid_request_error"
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
    assert "[DONE]" in text
    # The contract being tested is "the streaming response carries
    # chat.completion.chunk objects + a [DONE] marker". We don't
    # pin the exact SSE frame separator or per-event delimiting —
    # sse-starlette's wire format varies between '\r\n\r\n' and
    # '\n\n' depending on internal config, and a single SSE event
    # may span multiple data: lines that re-assemble client-side.
    # Substring search for the chunk type avoids parsing the SSE
    # framing entirely; the framing itself is sse-starlette's
    # responsibility, not the route handler's.
    assert "chat.completion.chunk" in text
    # And SOMEWHERE in the stream a content delta object lives —
    # the chunk schema includes a "choices" field on every event.
    assert '"choices"' in text


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
    # Per server.errors.STATUS_TYPE_MAP: 401 -> "authentication_error".
    assert body["error"]["type"] == "authentication_error"


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
