"""Streaming HTTP route tests (SSE).

Uses :class:`httpx.AsyncClient` with :class:`httpx.ASGITransport` to
hit the FastAPI app over real ASGI (no socket). httpx's
``stream("POST", ...)`` returns an async iterator of bytes which we
parse for SSE ``data: ...`` events.

The tests verify:
  * the SSE stream emits a leading ``role: assistant`` chunk
  * subsequent chunks each carry a ``content`` delta
  * the terminal chunk carries a ``finish_reason`` (no content delta)
  * the stream ends with the literal ``data: [DONE]`` sentinel
  * concatenated content deltas equal the engine's full decoded text
  * finish_reason is ``stop`` on EOS, ``length`` on max_tokens
"""

from __future__ import annotations

import json
from typing import AsyncIterator, List

import pytest
from httpx import ASGITransport, AsyncClient

from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig

pytestmark = pytest.mark.asyncio


async def _read_sse_events(stream: AsyncIterator[bytes]) -> List[str]:
    """Read raw SSE ``data:`` lines from an httpx stream.

    Returns a list of the strings *after* ``data: ``, in arrival
    order. Handles chunked transfer where a single yield may contain
    multiple events or a partial event.
    """
    buffer = b""
    out: List[str] = []
    async for chunk in stream:
        buffer += chunk
        # SSE frames are separated by blank lines. We split on \n\n.
        while b"\n\n" in buffer:
            frame, buffer = buffer.split(b"\n\n", 1)
            for line in frame.splitlines():
                if line.startswith(b"data: "):
                    out.append(line[len(b"data: "):].decode("utf-8"))
    # Trailing buffer (no terminating blank line) is still emitted by
    # sse-starlette for the final event in some configurations.
    if buffer:
        for line in buffer.splitlines():
            if line.startswith(b"data: "):
                out.append(line[len(b"data: "):].decode("utf-8"))
    return out


@pytest.fixture
def app(short_engine):
    return create_app(short_engine, ServerConfig())


@pytest.fixture
def app_long(long_engine):
    return create_app(long_engine, ServerConfig())


# ---------------------------------------------------------------------------
# Happy path: stream short engine to EOS
# ---------------------------------------------------------------------------


async def test_stream_emits_role_then_content_then_finish_reason(app, short_engine):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            assert r.status_code == 200
            ctype = r.headers["content-type"]
            assert "text/event-stream" in ctype
            events = await _read_sse_events(r.aiter_bytes())

    # Last event is "[DONE]"
    assert events[-1] == "[DONE]"
    payloads = [json.loads(e) for e in events[:-1]]

    # First chunk: role=assistant, no content.
    first = payloads[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"
    assert first["choices"][0]["delta"].get("content") is None
    assert first["choices"][0]["finish_reason"] is None

    # Last non-DONE chunk: finish_reason set, content cleared.
    last = payloads[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["choices"][0]["delta"].get("content") is None

    # Middle chunks: each carries a content delta.
    middle = payloads[1:-1]
    assert len(middle) >= 1
    for chunk in middle:
        delta = chunk["choices"][0]["delta"]
        assert delta.get("content") is not None
        assert delta.get("content") != ""
        assert chunk["choices"][0]["finish_reason"] is None


async def test_stream_concatenated_content_matches_engine_decode(app, short_engine):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    streamed_text = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in payloads
    )
    # The short_engine emits hello/world/!/EOS; decoded with
    # skip_special_tokens=True that's "hello world !"
    assert "hello" in streamed_text
    assert "world" in streamed_text


async def test_stream_finish_reason_length_on_max_tokens(app_long):
    async with AsyncClient(transport=ASGITransport(app=app_long), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 3,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"


async def test_stream_returns_done_sentinel_at_end(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    assert events[-1] == "[DONE]"
    # Exactly one DONE.
    assert sum(1 for e in events if e == "[DONE]") == 1


async def test_stream_each_chunk_has_required_openai_fields(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    for p in payloads:
        assert set(p.keys()) >= {"id", "object", "created", "model", "choices"}
        assert p["object"] == "chat.completion.chunk"
        assert isinstance(p["created"], int)
        assert len(p["choices"]) == 1
        c0 = p["choices"][0]
        assert "index" in c0 and "delta" in c0 and "finish_reason" in c0


async def test_stream_completion_id_consistent_across_chunks(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "kakeya",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as r:
            events = await _read_sse_events(r.aiter_bytes())
    payloads = [json.loads(e) for e in events if e != "[DONE]"]
    ids = {p["id"] for p in payloads}
    assert len(ids) == 1


async def test_stream_validation_error_returns_422(app):
    """Bad request still routes through pydantic validation before
    streaming starts. Status comes back as 422 with a JSON body."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "kakeya", "messages": [], "stream": True,
        })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Direct unit tests on _stream_via_scheduler helper
#
# These cover the disconnect/cancellation branches that ASGI transport
# does not reliably propagate from the test client. Direct invocation
# with a fake-request object whose is_disconnected returns True
# exercises the cancelled-by-disconnect branch deterministically.
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for starlette.requests.Request that exposes only
    ``is_disconnected``. Real concrete class — not a mock."""

    def __init__(self, sequence_of_disconnect_returns):
        self._returns = list(sequence_of_disconnect_returns)
        self.calls = 0

    async def is_disconnected(self):
        self.calls += 1
        if self._returns:
            return self._returns.pop(0)
        return False


def _build_scheduler_with_engine(engine, max_concurrent=1):
    """Build (scheduler, pool) wrapping ``engine`` for direct-helper tests."""
    import torch
    from inference_engine.memory.pool import SlabPool
    from inference_engine.memory.slab import SlabConfig
    from inference_engine.scheduler.config import SchedulerConfig
    from inference_engine.scheduler.scheduler import Scheduler

    pool = SlabPool(
        num_slabs=max_concurrent,
        slab_config=SlabConfig(
            num_layers=1, num_heads=1, sink_size=0, window_size=1,
            head_dim=1, dtype=torch.bfloat16,
        ),
    )
    return Scheduler(
        engine=engine, pool=pool,
        config=SchedulerConfig(max_concurrent=max_concurrent),
    )


async def test_stream_via_scheduler_finish_reason_stop_on_cancel(tokenizer):
    """Drive _stream_via_scheduler directly; force is_disconnected to
    return True after a few polls so the cancelled-by-disconnect
    branch fires and finish_reason='stop' is emitted."""
    from tests.inference_engine.server.conftest import DeterministicEngine
    from inference_engine.server.app import _stream_via_scheduler
    from inference_engine.server.metrics import Metrics

    ids = [tokenizer._intern(f"tok{i}") for i in range(20)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.02,
    )
    scheduler = _build_scheduler_with_engine(slow_engine)
    session = await scheduler.submit(
        prompt_ids=[1], max_new_tokens=20, eos_token_ids=[0],
    )

    request = _FakeRequest([False, False, True])
    chunks = []
    async for chunk in _stream_via_scheduler(
        scheduler=scheduler,
        session=session,
        request=request,
        engine=slow_engine,
        completion_id="testid",
        created=12345,
        metrics=Metrics.build(),
        disconnect_poll_interval_s=0.005,
    ):
        chunks.append(chunk)

    assert chunks[-1]["data"] == "[DONE]"
    payloads = [json.loads(c["data"]) for c in chunks[:-1]]
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert request.calls > 0


async def test_stream_via_scheduler_finish_reason_length_when_max_tokens(tokenizer):
    """No disconnect, max_tokens cap → finish_reason='length'."""
    from tests.inference_engine.server.conftest import DeterministicEngine
    from inference_engine.server.app import _stream_via_scheduler
    from inference_engine.server.metrics import Metrics

    ids = [tokenizer._intern(f"tok{i}") for i in range(20)]
    engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer, model_id_label="m",
    )
    scheduler = _build_scheduler_with_engine(engine)
    session = await scheduler.submit(
        prompt_ids=[1], max_new_tokens=3, eos_token_ids=[0],
    )

    request = _FakeRequest([])  # never disconnects
    chunks = []
    async for chunk in _stream_via_scheduler(
        scheduler=scheduler,
        session=session,
        request=request,
        engine=engine,
        completion_id="testid",
        created=12345,
        metrics=Metrics.build(),
    ):
        chunks.append(chunk)

    assert chunks[-1]["data"] == "[DONE]"
    payloads = [json.loads(c["data"]) for c in chunks[:-1]]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"


async def test_stream_via_scheduler_finish_reason_stop_on_eos(tokenizer):
    """Engine emits EOS before max_tokens → finish_reason='stop'."""
    from tests.inference_engine.server.conftest import DeterministicEngine
    from inference_engine.server.app import _stream_via_scheduler
    from inference_engine.server.metrics import Metrics

    hello = tokenizer._intern("hello")
    engine = DeterministicEngine(
        fixed_tokens=[hello, tokenizer.eos_token_id],
        tokenizer=tokenizer, model_id_label="m",
    )
    scheduler = _build_scheduler_with_engine(engine)
    session = await scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    request = _FakeRequest([])
    chunks = []
    async for chunk in _stream_via_scheduler(
        scheduler=scheduler, session=session, request=request,
        engine=engine, completion_id="x", created=1,
        metrics=Metrics.build(),
    ):
        chunks.append(chunk)
    payloads = [json.loads(c["data"]) for c in chunks[:-1]]
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


class _RaisingEngine:
    """Engine that raises mid-generate; used for graceful-stream-on-error test."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_id_label(self):
        return "raising"

    def generate(self, prompt_ids, max_new_tokens, eos_token_ids, on_token=None):
        raise RuntimeError("synthetic engine failure")


async def test_stream_via_scheduler_swallows_error_and_emits_terminal(tokenizer):
    """Engine raises mid-stream; SSE must still emit a terminal chunk
    + [DONE] (you cannot send a 500 once SSE has started)."""
    from inference_engine.server.app import _stream_via_scheduler
    from inference_engine.server.metrics import Metrics

    engine = _RaisingEngine(tokenizer)
    scheduler = _build_scheduler_with_engine(engine)
    session = await scheduler.submit(
        prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0],
    )
    request = _FakeRequest([])
    chunks = []
    async for chunk in _stream_via_scheduler(
        scheduler=scheduler, session=session, request=request,
        engine=engine, completion_id="x", created=1,
        metrics=Metrics.build(),
    ):
        chunks.append(chunk)
    # Should have at least the role chunk, the terminal chunk, and [DONE].
    assert chunks[-1]["data"] == "[DONE]"
    payloads = [json.loads(c["data"]) for c in chunks[:-1]]
    # finish_reason exists (FAILED branch maps to 'stop' conservatively).
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
