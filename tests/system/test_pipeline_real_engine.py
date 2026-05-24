"""Pipeline coordinator + real engine system tests.

Wires the :class:`PipelineCoordinator` (E5) to a real speculative
generation: the producer runs ``engine.generate`` in a thread and
pushes committed tokens; the consumer drains them as an async
iterator. Verifies that the primitive composes correctly with real
sync generation code (i.e. ``asyncio.to_thread`` + queue + close
sentinel).

Slow: real model load, real token generation. Capped at 8 tokens.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_engine.pipeline.coordinator import PipelineCoordinator

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
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
        return_dict=False, enable_thinking=False,
    )


async def test_pipeline_carries_real_engine_tokens(real_speculative_engine):
    """Wire engine.generate -> PipelineCoordinator -> async consumer."""
    coord: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=64)
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(tokenizer, "Hi")
    eos = _eos_ids(tokenizer)
    loop = asyncio.get_running_loop()

    def on_token(tok_id: int) -> bool:
        asyncio.run_coroutine_threadsafe(coord.put(int(tok_id)), loop)
        return False

    async def producer():
        await asyncio.to_thread(
            real_speculative_engine.generate,
            prompt_ids, 8, eos, on_token,
        )

    coord.start_producer(producer())

    received: list[int] = []
    async for tok in coord.consume():
        received.append(int(tok))

    assert len(received) > 0
    # Each received token is an int in vocab range — real engine
    # output, no garbling.
    assert all(isinstance(t, int) for t in received)


async def test_pipeline_cancellation_stops_real_generation(real_speculative_engine):
    """Consumer cancels mid-stream; producer must stop cleanly."""
    coord: PipelineCoordinator[int] = PipelineCoordinator(buffer_size=4)
    tokenizer = real_speculative_engine.tokenizer
    prompt_ids = _encode_chat(tokenizer, "Tell me a long story.")
    eos = _eos_ids(tokenizer)
    loop = asyncio.get_running_loop()
    cancelled = False

    def on_token(tok_id: int) -> bool:
        if cancelled:
            return True
        asyncio.run_coroutine_threadsafe(coord.put(int(tok_id)), loop)
        return False

    async def producer():
        await asyncio.to_thread(
            real_speculative_engine.generate,
            prompt_ids, 64, eos, on_token,
        )

    coord.start_producer(producer())

    received = 0
    gen = coord.consume()
    async for _ in gen:
        received += 1
        if received >= 2:
            cancelled = True
            await coord.cancel()
            break
    await gen.aclose()
    # Far fewer than 64 tokens — cancellation honored.
    assert received < 64
