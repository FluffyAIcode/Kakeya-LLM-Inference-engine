"""Unit tests for inference_engine.distributed.proposer_service (ADR 0009).

Wire-contract tests over a real ``grpc.aio`` server with the real
:class:`NGramProposer` serving blocks — the model-free proposer is the
production capability every node advertises, so these tests exercise
the same code path a Mac mini fleet runs, minus only model weights.

``RemoteProposer`` is synchronous by design (the spec-decode loop is
synchronous); inside async tests its blocking calls run via
``asyncio.to_thread`` so the in-process server keeps serving.

Coverage target: 100% on
``inference_engine/distributed/proposer_service.py``.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Tuple

import grpc
import pytest
import pytest_asyncio

from inference_engine.distributed.ngram import NGramProposer
from inference_engine.distributed.proposer_service import (
    ProposerServiceServicer,
    RemoteProposer,
    RemoteProposerError,
    add_proposer_service,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)
from kv_cache_proposer.proposer import BlockProposal

pytestmark = pytest.mark.asyncio


class _WireViolatingProposer:
    """A proposer whose *remote node* violates the block-size wire
    contract (e.g. a buggy or mismatched peer build). Exists to pin the
    client-side refusal: ``RemoteProposer`` must reject short blocks
    exactly as ``SpeculativeDecoder`` refuses malformed in-process
    blocks, because the decoder's accept loop indexes ``[0, L)``."""

    def propose_block(
        self, committed_token_ids: List[int], block_size: int, num_steps: int,
    ) -> BlockProposal:
        return BlockProposal(
            tokens=[0] * max(0, block_size - 1),
            diffusion_steps=0,
            forward_passes=1,
            peak_activation_bytes=0,
        )


async def _start_server(servicer: ProposerServiceServicer) -> Tuple[str, grpc.aio.Server]:
    server = grpc.aio.server()
    distributed_pb2_grpc.add_ProposerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return f"127.0.0.1:{port}", server


@pytest_asyncio.fixture
async def ngram_node() -> AsyncIterator[Tuple[str, grpc.aio.Server]]:
    servicer = ProposerServiceServicer(
        {"ngram": NGramProposer(), "ngram-wide": NGramProposer(max_ngram_size=8)},
        default_model_id="ngram",
    )
    address, server = await _start_server(servicer)
    try:
        yield address, server
    finally:
        await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# Servicer construction
# ---------------------------------------------------------------------------


async def test_servicer_rejects_empty_map():
    with pytest.raises(ValueError, match="non-empty"):
        ProposerServiceServicer({})


async def test_servicer_rejects_unknown_default():
    with pytest.raises(ValueError, match="default_model_id"):
        ProposerServiceServicer({"ngram": NGramProposer()}, default_model_id="x")


async def test_servicer_default_falls_back_to_first_entry():
    servicer = ProposerServiceServicer({"only": NGramProposer()})
    assert servicer.model_ids == ["only"]


# ---------------------------------------------------------------------------
# Wire-level servicer behavior
# ---------------------------------------------------------------------------


async def test_propose_block_over_the_wire(ngram_node):
    address, _ = ngram_node
    async with grpc.aio.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.ProposerServiceStub(channel)
        resp = await stub.ProposeBlock(
            distributed_pb2.ProposeBlockRequest(
                committed_token_ids=[10, 20, 30, 40, 10, 20],
                block_size=2,
                num_steps=1,
            ),
        )
    assert list(resp.token_ids) == [30, 40]
    assert resp.forward_passes == 1
    assert resp.diffusion_steps == 0


async def test_explicit_model_id_selects_proposer(ngram_node):
    address, _ = ngram_node
    async with grpc.aio.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.ProposerServiceStub(channel)
        resp = await stub.ProposeBlock(
            distributed_pb2.ProposeBlockRequest(
                committed_token_ids=[1, 2, 1, 2],
                block_size=1,
                num_steps=1,
                model_id="ngram-wide",
            ),
        )
    assert len(resp.token_ids) == 1


async def test_unknown_model_id_is_not_found(ngram_node):
    address, _ = ngram_node
    async with grpc.aio.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.ProposerServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub.ProposeBlock(
                distributed_pb2.ProposeBlockRequest(
                    committed_token_ids=[1],
                    block_size=1,
                    num_steps=1,
                    model_id="nope",
                ),
            )
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND
    assert "ngram" in excinfo.value.details()


async def test_malformed_arguments_are_invalid_argument(ngram_node):
    address, _ = ngram_node
    async with grpc.aio.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.ProposerServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub.ProposeBlock(
                distributed_pb2.ProposeBlockRequest(
                    committed_token_ids=[],
                    block_size=1,
                    num_steps=1,
                ),
            )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# RemoteProposer (client)
# ---------------------------------------------------------------------------


async def test_remote_proposer_constructor_validation():
    with pytest.raises(ValueError, match="address"):
        RemoteProposer("")
    with pytest.raises(ValueError, match="timeout_s"):
        RemoteProposer("h:1", timeout_s=0)


async def test_remote_proposer_round_trip_and_stats(ngram_node):
    address, _ = ngram_node
    with RemoteProposer(address) as remote:
        proposal = await asyncio.to_thread(
            remote.propose_block, [10, 20, 30, 40, 10, 20], 2, 1,
        )
        assert proposal.tokens == [30, 40]
        assert remote.stats.total_blocks == 1
        assert remote.stats.total_forward_passes == 1
        assert remote.stats.weight_bytes == 0

        second = await asyncio.to_thread(
            remote.propose_block, [1, 2, 1, 2], 1, 1,
        )
        assert len(second.tokens) == 1
        assert remote.stats.total_blocks == 2


async def test_remote_proposer_wraps_grpc_failures():
    remote = RemoteProposer("127.0.0.1:1", timeout_s=2.0)
    try:
        with pytest.raises(RemoteProposerError, match="UNAVAILABLE"):
            await asyncio.to_thread(remote.propose_block, [1, 2], 2, 1)
    finally:
        remote.close()


async def test_remote_proposer_refuses_short_blocks():
    servicer = ProposerServiceServicer({"bad": _WireViolatingProposer()})
    address, server = await _start_server(servicer)
    try:
        with RemoteProposer(address, model_id="bad") as remote:
            with pytest.raises(RemoteProposerError, match="expected exactly 4"):
                await asyncio.to_thread(remote.propose_block, [1, 2], 4, 1)
    finally:
        await server.stop(grace=0.1)


async def test_add_proposer_service_registers_on_server():
    server = grpc.aio.server()
    servicer = add_proposer_service(
        server, {"ngram": NGramProposer()}, default_model_id="ngram",
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        assert servicer.model_ids == ["ngram"]
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = distributed_pb2_grpc.ProposerServiceStub(channel)
            resp = await stub.ProposeBlock(
                distributed_pb2.ProposeBlockRequest(
                    committed_token_ids=[5, 6, 5, 6],
                    block_size=1,
                    num_steps=1,
                ),
            )
            assert len(resp.token_ids) == 1
    finally:
        await server.stop(grace=0.1)
