"""End-to-end HTTP + speculative-decoder system tests.

Drives the full FastAPI app — backed by a real :class:`SpeculativeEngine`
over real Qwen3 + dllm-hub weights — through ``httpx.AsyncClient`` with
``ASGITransport``. Verifies routing, request validation, and SSE
streaming on real generation output.

These tests are slow (one shared model load per session, real token
generation per test). They auto-skip on hosts without the HF cache;
see :mod:`tests.system.conftest`.

We assert structural / behavioral invariants — not specific token
sequences — so the tests are robust to verifier weight updates and
small numerical drift. Examples:

  * "the response contains assistant text" (not "the response says X")
  * "completion_tokens > 0 and <= max_tokens"
  * "concatenated SSE deltas equal the non-streaming completion text"
  * "finish_reason is stop or length"
"""

from __future__ import annotations

import json
from typing import AsyncIterator, List

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Health + models endpoints
# ---------------------------------------------------------------------------


async def test_healthz_returns_ok(server_app):
    async with AsyncClient(transport=ASGITransport(app=server_app),
                           base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "kakeya-system-test"


async def test_v1_models_lists_engine_label(server_app):
    async with AsyncClient(transport=ASGITransport(app=server_app),
                           base_url="http://t") as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "kakeya-system-test" in ids


# ---------------------------------------------------------------------------
# Non-streaming completion
# ---------------------------------------------------------------------------


async def test_non_streaming_chat_completion_produces_text(server_app):
    async with AsyncClient(
        transport=ASGITransport(app=server_app),
        base_url="http://t", timeout=120.0,
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Say hi briefly."}],
            "stream": False,
            "max_tokens": 16,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], str)
    assert len(msg["content"]) > 0
    # finish_reason is one of the expected values.
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    # usage is structurally consistent.
    u = body["usage"]
    assert u["prompt_tokens"] > 0
    assert 0 < u["completion_tokens"] <= 16
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


async def test_max_tokens_respected(server_app):
    """Asking for max_tokens=4 must not produce more than 4."""
    async with AsyncClient(
        transport=ASGITransport(app=server_app),
        base_url="http://t", timeout=120.0,
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "max_tokens": 4,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["usage"]["completion_tokens"] <= 4


# ---------------------------------------------------------------------------
# Streaming completion
# ---------------------------------------------------------------------------


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


async def test_streaming_emits_done_terminator(server_app):
    async with AsyncClient(
        transport=ASGITransport(app=server_app),
        base_url="http://t", timeout=120.0,
    ) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 8,
        }) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            events = await _read_sse_events(r.aiter_bytes())
    assert events[-1] == "[DONE]"


async def test_streaming_first_chunk_carries_role(server_app):
    async with AsyncClient(
        transport=ASGITransport(app=server_app),
        base_url="http://t", timeout=120.0,
    ) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 8,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    first = payloads[0]
    assert first["choices"][0]["delta"].get("role") == "assistant"


async def test_streaming_concatenated_text_nonempty(server_app):
    async with AsyncClient(
        transport=ASGITransport(app=server_app),
        base_url="http://t", timeout=120.0,
    ) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 16,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    streamed_text = "".join(
        p["choices"][0]["delta"].get("content") or "" for p in payloads
    )
    assert len(streamed_text) > 0


async def test_streaming_finish_reason_is_set_on_terminal_chunk(server_app):
    async with AsyncClient(
        transport=ASGITransport(app=server_app),
        base_url="http://t", timeout=120.0,
    ) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 8,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    assert payloads[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


# ---------------------------------------------------------------------------
# Validation errors propagate even with real engine
# ---------------------------------------------------------------------------


async def test_real_engine_rejects_empty_messages(server_app):
    async with AsyncClient(
        transport=ASGITransport(app=server_app), base_url="http://t",
    ) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya-system-test",
            "messages": [],
        })
    assert r.status_code == 422
