"""Unit tests for :mod:`inference_engine.server.streaming`.

Tests the sync-to-async bridge that drives streaming chat completions.
Uses real :class:`DeterministicEngine` instances (no mocks); the only
async behaviour exercised is asyncio's own (queues, thread offload,
await).

The detokenizer tests verify the per-token delta semantics — including
that empty deltas (multi-byte UTF-8 mid-sequence) are skipped, and
that the delta sum equals the full tokenizer.decode of all ids.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from inference_engine.server.streaming import (
    _StreamingDetokenizer,
    iter_token_deltas,
    run_blocking,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Detokenizer
# ---------------------------------------------------------------------------


async def test_detokenizer_emits_full_text_via_per_token_deltas(tokenizer):
    """Sum of per-token deltas == full decoded sequence."""
    ids = [
        tokenizer._intern("hello"),
        tokenizer._intern("world"),
        tokenizer._intern("!"),
    ]
    detok = _StreamingDetokenizer(tokenizer)
    pieces: List[str] = [detok.feed(t) for t in ids]
    full = tokenizer.decode(ids, skip_special_tokens=True)
    assert "".join(pieces) == full


async def test_detokenizer_handles_special_tokens(tokenizer):
    """Special tokens (id 0 = <|im_end|>) decode to empty under
    skip_special_tokens=True; the delta on that token is empty."""
    eos = tokenizer.eos_token_id
    hello = tokenizer._intern("hello")
    detok = _StreamingDetokenizer(tokenizer)
    d1 = detok.feed(hello)
    d2 = detok.feed(eos)
    assert d1 == "hello"
    assert d2 == ""  # special token contributes nothing visible


# ---------------------------------------------------------------------------
# iter_token_deltas: happy path
# ---------------------------------------------------------------------------


async def test_iter_token_deltas_yields_tokens_then_terminal_marker(short_engine):
    deltas = []
    finals = []
    async for delta, is_final, _session in iter_token_deltas(
        engine=short_engine,
        prompt_ids=[1, 2],
        max_new_tokens=10,
        eos_token_ids=[0],
    ):
        deltas.append(delta)
        finals.append(is_final)
    # Last yield is the terminal "" with is_final=True.
    assert finals[-1] is True
    assert deltas[-1] == ""
    # All earlier yields were non-final.
    assert all(f is False for f in finals[:-1])
    # Empty deltas filtered out (only the terminal one is empty).
    assert all(d for d in deltas[:-1])


async def test_iter_token_deltas_terminal_session_has_engine_result(short_engine):
    sessions = []
    async for _delta, is_final, session in iter_token_deltas(
        engine=short_engine,
        prompt_ids=[1, 2],
        max_new_tokens=10,
        eos_token_ids=[0],
    ):
        if is_final:
            sessions.append(session)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.result.engine_result is not None
    assert s.result.engine_result.stopped_on_eos is True
    assert s.result.cancelled_by_disconnect is False


async def test_iter_token_deltas_max_tokens_truncates(long_engine):
    """Engine has 50 tokens; ask for 3 → only 3 tokens come through."""
    deltas: List[str] = []
    async for delta, is_final, _s in iter_token_deltas(
        engine=long_engine,
        prompt_ids=[1, 2],
        max_new_tokens=3,
        eos_token_ids=[0],
    ):
        if not is_final:
            deltas.append(delta)
    assert len(deltas) == 3


async def test_iter_token_deltas_disconnect_short_circuits(tokenizer):
    """If is_disconnected returns True early, generation stops cleanly
    and the session reports cancelled_by_disconnect=True."""
    from tests.inference_engine.server.conftest import DeterministicEngine

    # 50 tokens with per-token delay of 20ms so the consumer's
    # disconnect poll (interval 5ms) reliably fires multiple times
    # before the producer finishes.
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.02,
    )
    poll_count = 0

    async def is_disc():
        nonlocal poll_count
        poll_count += 1
        # Fire after enough polls to allow at least the first token to
        # have flowed through the queue (per_token_delay_s=0.02 means
        # ~20ms per token; we set the disconnect ~50ms in).
        return poll_count >= 10

    seen: List[str] = []
    final_session = None
    async for delta, is_final, session in iter_token_deltas(
        engine=slow_engine,
        prompt_ids=[1, 2],
        max_new_tokens=50,
        eos_token_ids=[0],
        is_disconnected=is_disc,
        disconnect_poll_interval_s=0.005,
    ):
        if is_final:
            final_session = session
        else:
            seen.append(delta)
    assert final_session is not None
    assert final_session.result.cancelled_by_disconnect is True
    # We did NOT get all 50 tokens — disconnect cut us off.
    assert len(seen) < 50


# ---------------------------------------------------------------------------
# iter_token_deltas: defensive validation
# ---------------------------------------------------------------------------


async def test_iter_token_deltas_rejects_empty_prompt_ids(short_engine):
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        async for _ in iter_token_deltas(
            engine=short_engine, prompt_ids=[],
            max_new_tokens=5, eos_token_ids=[0],
        ):
            pass  # pragma: no cover - generator never yields


async def test_iter_token_deltas_rejects_zero_max_tokens(short_engine):
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        async for _ in iter_token_deltas(
            engine=short_engine, prompt_ids=[1],
            max_new_tokens=0, eos_token_ids=[0],
        ):
            pass  # pragma: no cover - generator never yields


async def test_iter_token_deltas_rejects_empty_eos(short_engine):
    with pytest.raises(ValueError, match="eos_token_ids must be non-empty"):
        async for _ in iter_token_deltas(
            engine=short_engine, prompt_ids=[1],
            max_new_tokens=5, eos_token_ids=[],
        ):
            pass  # pragma: no cover - generator never yields


# ---------------------------------------------------------------------------
# iter_token_deltas: exception propagation
# ---------------------------------------------------------------------------


class _RaisingEngine:
    """Engine double that raises mid-generate. Used to verify exception
    propagation from worker thread back to the streaming caller."""

    def __init__(self, tokenizer, model_id_label="raising-engine"):
        self._tokenizer = tokenizer
        self._model_id_label = model_id_label

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_id_label(self):
        return self._model_id_label

    def generate(self, prompt_ids, max_new_tokens, eos_token_ids, on_token=None):
        raise RuntimeError("synthetic decoder failure")


async def test_iter_token_deltas_propagates_engine_exception(tokenizer):
    engine = _RaisingEngine(tokenizer)
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        async for _ in iter_token_deltas(
            engine=engine, prompt_ids=[1],
            max_new_tokens=5, eos_token_ids=[0],
        ):
            pass  # pragma: no cover - generator raises before yielding


async def test_iter_token_deltas_no_disconnect_callback_runs_to_completion(tokenizer):
    """When ``is_disconnected`` is None (default), the disconnect
    polling branches must not raise; they should be no-ops. Drives a
    slow engine so the wall-clock poll branch *does* fire, exercising
    the ``is_disconnected is None`` early-return path."""
    from tests.inference_engine.server.conftest import DeterministicEngine

    ids = [tokenizer._intern(f"tok{i}") for i in range(5)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.02,
    )
    seen: List[str] = []
    final_session = None
    async for delta, is_final, session in iter_token_deltas(
        engine=slow_engine, prompt_ids=[1],
        max_new_tokens=5, eos_token_ids=[0],
        # no is_disconnected callback
        disconnect_poll_interval_s=0.005,
    ):
        if is_final:
            final_session = session
        else:
            seen.append(delta)
    assert final_session is not None
    assert final_session.result.cancelled_by_disconnect is False
    assert len(seen) == 5


async def test_iter_token_deltas_aclose_mid_stream_cancels_producer(tokenizer):
    """If the consumer closes the generator before the producer is
    done, the finally block must signal disconnect and await the
    producer for clean teardown."""
    from tests.inference_engine.server.conftest import DeterministicEngine

    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    slow_engine = DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.02,
    )
    gen = iter_token_deltas(
        engine=slow_engine, prompt_ids=[1],
        max_new_tokens=50, eos_token_ids=[0],
    )
    saw_at_least_one = False
    async for _delta, is_final, _session in gen:
        if not is_final:
            saw_at_least_one = True
            await gen.aclose()
            break
    assert saw_at_least_one


# ---------------------------------------------------------------------------
# run_blocking
# ---------------------------------------------------------------------------


async def test_run_blocking_returns_engine_result(short_engine):
    res = await run_blocking(
        engine=short_engine,
        prompt_ids=[1, 2],
        max_new_tokens=10,
        eos_token_ids=[0],
    )
    assert res.stopped_on_eos is True
    assert len(res.output_token_ids) == 4  # 3 content tokens + EOS


async def test_run_blocking_does_not_block_event_loop(short_engine):
    """run_blocking is async — concurrent ticks proceed during generate.

    We schedule a sentinel task; if generate were blocking the event
    loop, the sentinel would never run before generate completes.
    """
    sentinel_done = asyncio.Event()

    async def sentinel():
        await asyncio.sleep(0.0)
        sentinel_done.set()

    asyncio.create_task(sentinel())
    await run_blocking(
        engine=short_engine, prompt_ids=[1],
        max_new_tokens=10, eos_token_ids=[0],
    )
    assert sentinel_done.is_set()


async def test_run_blocking_rejects_empty_prompt_ids(short_engine):
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        await run_blocking(
            engine=short_engine, prompt_ids=[],
            max_new_tokens=5, eos_token_ids=[0],
        )


async def test_run_blocking_rejects_zero_max_tokens(short_engine):
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        await run_blocking(
            engine=short_engine, prompt_ids=[1],
            max_new_tokens=0, eos_token_ids=[0],
        )


async def test_run_blocking_rejects_empty_eos(short_engine):
    with pytest.raises(ValueError, match="eos_token_ids must be non-empty"):
        await run_blocking(
            engine=short_engine, prompt_ids=[1],
            max_new_tokens=5, eos_token_ids=[],
        )
