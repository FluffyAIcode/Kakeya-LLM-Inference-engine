"""Unit tests for non-streaming HTTP routes.

Uses :class:`httpx.AsyncClient` with :class:`httpx.ASGITransport` —
real ASGI invocation in-process, no socket, no mock. The transport
calls the FastAPI app exactly as a real uvicorn worker would, so
status codes, headers, and JSON bodies are real round-trips through
the route handlers.

Streaming routes are tested separately in test_app_streaming.py.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig

pytestmark = pytest.mark.asyncio


@pytest.fixture
def app(short_engine):
    return create_app(short_engine, ServerConfig())


@pytest.fixture
def app_long(long_engine):
    return create_app(long_engine, ServerConfig())


# ---------------------------------------------------------------------------
# Healthz / models
# ---------------------------------------------------------------------------


async def test_healthz_returns_ok(app, short_engine):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == short_engine.model_id_label


async def test_v1_models_lists_engine_label(app, short_engine):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == short_engine.model_id_label
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"] == "kakeya"


# ---------------------------------------------------------------------------
# /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------


async def test_chat_completions_non_streaming_returns_message(app, short_engine):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == short_engine.model_id_label
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]  # non-empty
    assert body["choices"][0]["finish_reason"] == "stop"
    # Usage is structurally correct.
    u = body["usage"]
    assert u["prompt_tokens"] >= 1
    assert u["completion_tokens"] >= 1
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    # Completion id is ours, not the client's.
    assert body["id"].startswith("chatcmpl-")


async def test_chat_completions_finish_reason_length_when_truncated(app_long):
    """Long engine has no EOS in its first 3 tokens → finish_reason=length."""
    async with AsyncClient(transport=ASGITransport(app=app_long), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 3,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


async def test_chat_completions_uses_default_max_tokens_when_unspecified(short_engine):
    """If client omits max_tokens we use ServerConfig.default_max_new_tokens."""
    cfg = ServerConfig(default_max_new_tokens=2)
    app = create_app(short_engine, cfg)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 200
    body = r.json()
    # short_engine emits hello/world/bang/EOS — capped to 2 means we
    # hit "length" before EOS.
    assert body["usage"]["completion_tokens"] <= 2


async def test_chat_completions_rejects_empty_messages(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya", "messages": [], "stream": False,
        })
    # FastAPI returns 422 for pydantic validation failures.
    assert r.status_code == 422


async def test_chat_completions_rejects_invalid_role(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "tool", "content": "hi"}],
        })
    assert r.status_code == 422


async def test_chat_completions_rejects_negative_max_tokens(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        })
    assert r.status_code == 422


async def test_chat_completions_accepts_unknown_fields(app):
    """Forward-compatibility: unknown OpenAI fields are silently ignored."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "presence_penalty": 0.5,
            "logit_bias": {"99": 1.0},
            "user": "abc",
            "future_field_42": ["whatever"],
        })
    assert r.status_code == 200


async def test_chat_completions_accepts_temperature_and_top_p_no_op(app):
    """Sampling parameters are accepted but not applied. Greedy output
    must match what the engine produces with no temperature."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r1 = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        r2 = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "temperature": 1.5,
            "top_p": 0.7,
        })
    assert r1.json()["choices"][0]["message"]["content"] == \
        r2.json()["choices"][0]["message"]["content"]


async def test_chat_completions_returns_400_on_empty_template(short_engine):
    """If the tokenizer returns an empty list, the route surfaces it
    as 400 with a helpful detail. Exercises the prompt-emptiness
    guard in _encode_prompt."""

    class _EmptyTemplateTokenizer:
        eos_token_id = 0
        unk_token_id = 1

        def apply_chat_template(self, *a, **kw):
            return []

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, t):
            if t == "<|im_end|>":
                return 0
            return None

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

        def generate(self, *a, **kw):  # pragma: no cover - never reached
            return self._inner.generate(*a, **kw)

    proxy = _ProxyEngine(short_engine, _EmptyTemplateTokenizer())
    app = create_app(proxy, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 400
    assert "empty token sequence" in r.json()["detail"]


async def test_chat_completions_handles_chat_template_failure(short_engine):
    """If the tokenizer rejects messages (e.g. returns non-list), the
    route returns 400, not 500."""

    class _BrokenTokenizer:
        eos_token_id = 0
        unk_token_id = 1

        def apply_chat_template(self, *a, **kw):
            return "not a list"

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, t):
            if t == "<|im_end|>":
                return 0
            return None

    # Wrap the engine with a broken tokenizer to exercise the 400 path.
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

        def generate(self, *a, **kw):  # pragma: no cover - never reached
            return self._inner.generate(*a, **kw)

    proxy = _ProxyEngine(short_engine, _BrokenTokenizer())
    app = create_app(proxy, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 400
    assert "prompt encoding failed" in r.json()["detail"]


async def test_chat_completions_returns_500_when_tokenizer_loses_eos(short_engine):
    """If somehow the tokenizer's EOS state degrades after engine
    construction (we don't expect this in practice, but the route's
    defense-in-depth check should still catch it), we return 500
    rather than entering an unbounded generation loop."""

    class _NoEosTokenizer:
        eos_token_id = None
        unk_token_id = None

        def apply_chat_template(self, *a, **kw):
            return [1, 2, 3]

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, t):
            return None

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

        def generate(self, *a, **kw):  # pragma: no cover - never reached
            return self._inner.generate(*a, **kw)

    proxy = _ProxyEngine(short_engine, _NoEosTokenizer())
    app = create_app(proxy, ServerConfig())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 500
    assert "EOS configuration" in r.json()["detail"]
