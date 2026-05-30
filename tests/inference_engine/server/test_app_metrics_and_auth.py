"""Integration tests: GET /metrics, OpenAI error envelope, API-key auth.

These exercise the full FastAPI app via :class:`httpx.ASGITransport`.
The deterministic engine + tokenizer test doubles from ``conftest.py``
drive the routes; we never load real models.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from inference_engine.scheduler.config import AdmissionPolicy
from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig

from tests.inference_engine.server.conftest import (
    DeterministicEngine,
    DeterministicTokenizer,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


async def test_metrics_endpoint_returns_prometheus_text(short_engine):
    app = create_app(short_engine, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    text = r.text
    assert "# HELP scheduler_pool_total" in text
    assert "# TYPE http_requests_total counter" in text


async def test_metrics_after_completion_records_finish_reason(short_engine):
    app = create_app(short_engine, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
        r = await c.get("/metrics")
    text = r.text
    assert 'inference_completions_total{finish_reason="stop"}' in text


async def test_metrics_records_429_admission(tokenizer):
    """A pool-full 429 increments scheduler_admission_total{result=rejected}."""
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    slow = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer, per_token_delay_s=0.05,
    )
    app = create_app(slow, ServerConfig(max_concurrent=1))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t", timeout=30.0) as c:
        first = asyncio.create_task(c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 10,
        }))
        await asyncio.sleep(0.02)
        second = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "b"}],
            "max_tokens": 10,
        })
        assert second.status_code == 429
        await first
        r = await c.get("/metrics")
    assert 'scheduler_admission_total{result="rejected"} 1.0' in r.text


async def test_metrics_pool_total_gauge_reflects_config(short_engine):
    app = create_app(short_engine, ServerConfig(max_concurrent=4))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.get("/metrics")
    assert "scheduler_pool_total 4.0" in r.text


async def test_metrics_kv_live_bytes_gauge_present_and_zero_at_idle(
    short_engine,
):
    """The KV-live-bytes gauge must be exposed and read 0 on an idle
    pool (every slab has logical_size == 0). This is the gauge that
    bench_long_session.py scrapes to verify the ADR 0006 §2.3
    KV-bounded claim, so its presence is part of the public contract.
    """
    app = create_app(short_engine, ServerConfig(max_concurrent=2))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.get("/metrics")
    text = r.text
    assert "# HELP scheduler_kv_live_bytes" in text
    assert "scheduler_kv_live_bytes 0.0" in text


# ---------------------------------------------------------------------------
# OpenAI error envelope
# ---------------------------------------------------------------------------


async def test_validation_error_returns_openai_envelope(short_engine):
    app = create_app(short_engine, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "m", "messages": [],
        })
    assert r.status_code == 422
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"
    assert isinstance(body["error"]["message"], str)
    assert "messages" in (body["error"]["param"] or "")


async def test_429_error_envelope_has_rate_limit_type(tokenizer):
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    slow = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer, per_token_delay_s=0.05,
    )
    app = create_app(slow, ServerConfig(max_concurrent=1))
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t", timeout=30.0) as c:
        first = asyncio.create_task(c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 10,
        }))
        await asyncio.sleep(0.02)
        second = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "b"}],
            "max_tokens": 10,
        })
        await first
    body = second.json()
    assert body["error"]["type"] == "rate_limit_error"


async def test_400_error_envelope_has_invalid_request_type(short_engine):
    """Empty chat-template output → 400 with invalid_request_error type."""

    class _EmptyTemplateTokenizer:
        eos_token_id = 0
        unk_token_id = 1

        def apply_chat_template(self, *a, **kw):
            return []

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, t):
            return 0 if t == "<|im_end|>" else None

    class _ProxyEngine:
        def __init__(self, inner, tok):
            self._inner = inner
            self._tok = tok

        @property
        def tokenizer(self):
            return self._tok

        @property
        def model_id_label(self):
            return self._inner.model_id_label

        def generate(self, *a, **kw):  # pragma: no cover
            return self._inner.generate(*a, **kw)

    proxy = _ProxyEngine(short_engine, _EmptyTemplateTokenizer())
    app = create_app(proxy, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# API-key auth
# ---------------------------------------------------------------------------


async def test_no_api_keys_means_no_auth_required(short_engine):
    """With api_keys empty (default), requests succeed without any token."""
    app = create_app(short_engine, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 200


async def test_auth_required_returns_401_without_token(short_engine):
    app = create_app(
        short_engine,
        ServerConfig(api_keys=frozenset({"sk-test"})),
    )
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "authentication_error"
    assert "WWW-Authenticate" in r.headers


async def test_auth_required_succeeds_with_correct_token(short_engine):
    app = create_app(
        short_engine,
        ServerConfig(api_keys=frozenset({"sk-test"})),
    )
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"authorization": "Bearer sk-test"},
        )
    assert r.status_code == 200


async def test_auth_rejects_wrong_token(short_engine):
    app = create_app(
        short_engine,
        ServerConfig(api_keys=frozenset({"sk-test"})),
    )
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"authorization": "Bearer wrong"},
        )
    assert r.status_code == 401


async def test_healthz_does_not_require_auth(short_engine):
    app = create_app(
        short_engine,
        ServerConfig(api_keys=frozenset({"sk-test"})),
    )
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200


async def test_metrics_does_not_require_auth(short_engine):
    """Prometheus scrapers don't carry tokens; /metrics must remain public."""
    app = create_app(
        short_engine,
        ServerConfig(api_keys=frozenset({"sk-test"})),
    )
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://t") as c:
        r = await c.get("/metrics")
    assert r.status_code == 200


async def test_unhandled_exception_returns_500_envelope(short_engine):
    """If something unexpected leaks out of a route, the global
    exception handler still returns a clean OpenAI envelope.

    Note: ``ASGITransport(raise_app_exceptions=False)`` is required
    here because httpx's default behaviour is to re-raise exceptions
    from the inner app — that's useful for catching test-time bugs,
    but we want to verify the registered Exception handler runs and
    sends a real 500 response."""
    app = create_app(short_engine, ServerConfig())

    @app.get("/_internal_error")
    async def _kaboom():
        raise RuntimeError("boom")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://t",
    ) as c:
        r = await c.get("/_internal_error")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["type"] == "server_error"
