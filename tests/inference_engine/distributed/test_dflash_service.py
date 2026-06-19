"""Wire-contract tests for DFlashProposerService over a real grpc.aio server
with the synchronous RemoteDFlashProposer client (driven via asyncio.to_thread,
as the spec-decode loop would)."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Sequence, Tuple

import grpc
import numpy as np
import pytest
import pytest_asyncio

from inference_engine.distributed import tensor_codec as tc
from inference_engine.distributed.dflash_service import (
    DFlashProposerError,
    DFlashProposerServicer,
    DraftResult,
    RemoteDFlashProposer,
    RestoreResult,
    add_dflash_proposer_service,
)
from inference_engine.distributed.tensor_codec import WireTensor

pytestmark = pytest.mark.asyncio


class _FakeEngine:
    """Records calls; returns deterministic WireTensors. Raises to exercise the
    servicer's status-code mapping."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.seeded: List[Tuple[str, List[int]]] = []
        self.bad_draft_count = 0  # if >0, draft_block returns this many tokens

    def restore(self, session_id, prompt_ids, *, sink, window, s5_exact_full_attn, model_id):
        self.calls.append("restore")
        if session_id == "boom":
            raise RuntimeError("engine exploded")  # -> UNKNOWN (uncaught)
        k = tc.encode_array(np.arange(6, dtype=np.float32).reshape(1, 3, 2))
        v = tc.encode_array(np.ones((1, 3, 2), dtype=np.float32))
        restored = [] if s5_exact_full_attn and not prompt_ids else [(7, k, v)]
        return RestoreResult(restored=restored,
                             evicted_positions=[2, 3], prompt_len=len(prompt_ids))

    def seed_context(self, session_id, aux, positions):
        self.calls.append("seed_context")
        if session_id == "missing":
            raise KeyError(session_id)
        self.seeded.append((session_id, list(positions)))
        return len(positions)

    def draft_block(self, session_id, *, bonus_token_id, context_len, block_size):
        self.calls.append("draft_block")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        n = self.bad_draft_count or block_size
        return DraftResult(draft_token_ids=[bonus_token_id + i for i in range(n)],
                           forward_passes=1, peak_activation_bytes=123)

    def extend_context(self, session_id, aux, positions):
        self.calls.append("extend_context")
        return context_len_of(aux, positions)

    def close_session(self, session_id):
        self.calls.append("close_session")


def context_len_of(aux: Sequence[WireTensor], positions) -> int:
    return len(list(positions))


async def _start(engine) -> Tuple[str, grpc.aio.Server]:
    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ])
    add_dflash_proposer_service(server, engine)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return f"127.0.0.1:{port}", server


@pytest_asyncio.fixture
async def served() -> AsyncIterator[Tuple[str, _FakeEngine, grpc.aio.Server]]:
    engine = _FakeEngine()
    address, server = await _start(engine)
    try:
        yield address, engine, server
    finally:
        await server.stop(grace=0.1)


def _aux(k: int) -> List[WireTensor]:
    return [tc.encode_array(np.full((1, k, 4), li, dtype=np.float32)) for li in range(2)]


# NOTE: close() issues a synchronous CloseSession RPC; in-test the grpc.aio
# server shares this thread's event loop, so close() MUST go through
# asyncio.to_thread too (in production the server is on another host — no such
# constraint). _client() yields a remote and closes it off-thread.
class _Remote:
    def __init__(self, address, **kw):
        self.remote = RemoteDFlashProposer(address, **kw)

    async def __aenter__(self):
        return self.remote

    async def __aexit__(self, *exc):
        await asyncio.to_thread(self.remote.close)


async def test_restore_roundtrip(served):
    address, engine, _ = served
    async with _Remote(address, session_id="s1") as remote:
        res = await asyncio.to_thread(remote.restore, [1, 2, 3, 4], sink=2, window=2,
                                      s5_exact_full_attn=False)
    assert res.prompt_len == 4
    assert res.evicted_positions == [2, 3]
    assert len(res.restored) == 1
    layer, k, v = res.restored[0]
    assert layer == 7
    np.testing.assert_array_equal(k.data, np.arange(6, dtype=np.float32).reshape(1, 3, 2))
    np.testing.assert_array_equal(v.data, np.ones((1, 3, 2), dtype=np.float32))
    assert "restore" in engine.calls


async def test_seed_and_extend_and_draft_and_close(served):
    address, engine, _ = served
    async with _Remote(address, session_id="s2") as remote:
        cl = await asyncio.to_thread(remote.seed_context, _aux(5), [0, 1, 2, 3, 4])
        assert cl == 5
        dr = await asyncio.to_thread(
            lambda: remote.draft_block(bonus_token_id=100, context_len=5, block_size=3))
        assert dr.draft_token_ids == [100, 101, 102]
        assert dr.forward_passes == 1 and dr.peak_activation_bytes == 123
        cl2 = await asyncio.to_thread(remote.extend_context, _aux(2), [5, 6])
        assert cl2 == 2
    assert engine.calls.count("close_session") == 1
    assert engine.seeded == [("s2", [0, 1, 2, 3, 4])]


async def test_draft_block_count_mismatch_raises(served):
    address, engine, _ = served
    engine.bad_draft_count = 2  # return 2 when 4 requested
    async with _Remote(address, session_id="s3") as remote:
        with pytest.raises(DFlashProposerError, match="expected 4"):
            await asyncio.to_thread(
                lambda: remote.draft_block(bonus_token_id=1, context_len=0, block_size=4))


async def test_unknown_session_maps_to_not_found(served):
    address, _, _ = served
    async with _Remote(address, session_id="missing") as remote:
        with pytest.raises(DFlashProposerError, match="NOT_FOUND"):
            await asyncio.to_thread(remote.seed_context, _aux(1), [0])


async def test_bad_args_maps_to_invalid_argument(served):
    address, _, _ = served
    async with _Remote(address, session_id="s4") as remote:
        with pytest.raises(DFlashProposerError, match="INVALID_ARGUMENT"):
            await asyncio.to_thread(
                lambda: remote.draft_block(bonus_token_id=1, context_len=0, block_size=0))


async def test_uncaught_engine_error_propagates_as_unknown(served):
    address, _, _ = served
    async with _Remote(address, session_id="boom") as remote:
        with pytest.raises(DFlashProposerError):
            await asyncio.to_thread(remote.restore, [1], sink=1, window=1)


async def test_rpc_failure_on_dead_address_wraps():
    remote = RemoteDFlashProposer("127.0.0.1:1", session_id="x", timeout_s=2.0)
    try:
        with pytest.raises(DFlashProposerError, match="UNAVAILABLE"):
            await asyncio.to_thread(remote.seed_context, _aux(1), [0])
    finally:
        await asyncio.to_thread(remote.close)
