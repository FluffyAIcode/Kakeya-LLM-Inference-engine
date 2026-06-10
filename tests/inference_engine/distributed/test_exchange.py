"""Unit tests for inference_engine.distributed.exchange (ADR 0009).

Same testing posture as tests/inference_engine/server/test_grpc_app.py:
a real ``grpc.aio`` server on a random localhost port, real generated
stubs, no transport mocking. Two registries gossiping over loopback is
the smallest honest model of two Mac minis gossiping over a LAN.

Coverage target: 100% on ``inference_engine/distributed/exchange.py``
plus the capability/proposer wiring branches of
``inference_engine/server/grpc_app.py``.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Tuple

import grpc
import pytest
import pytest_asyncio

from inference_engine.distributed.capability import (
    NGRAM_MODEL_ID,
    CapabilityRegistry,
    CapabilityRole,
    ModelCapability,
    NodeCapability,
)
from inference_engine.distributed.exchange import (
    CapabilityServiceServicer,
    ExchangeReport,
    add_capability_service,
    exchange_once,
    fetch_node_capability,
)
from inference_engine.distributed.ngram import NGramProposer
from inference_engine.server.grpc_app import GrpcServerConfig, create_grpc_server
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)
from inference_engine.session import SessionStore

pytestmark = pytest.mark.asyncio


def _card(node_id: str, *, models: tuple = ()) -> NodeCapability:
    return NodeCapability(
        node_id=node_id,
        grpc_address=f"{node_id}:50051",
        models=models,
        announced_at_unix=time.time(),
        ttl_seconds=3600.0,
    )


async def _start_node(
    registry: CapabilityRegistry,
) -> Tuple[str, grpc.aio.Server]:
    """Start a real CapabilityService for ``registry`` on a random port."""
    server = grpc.aio.server()
    add_capability_service(server, registry)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return f"127.0.0.1:{port}", server


@pytest_asyncio.fixture
async def two_nodes() -> AsyncIterator[
    Tuple[CapabilityRegistry, CapabilityRegistry, str, str]
]:
    reg_a = CapabilityRegistry(self_card=_card("node-a"))
    reg_b = CapabilityRegistry(
        self_card=_card(
            "node-b",
            models=(ModelCapability(NGRAM_MODEL_ID, CapabilityRole.PROPOSER),),
        ),
    )
    addr_a, server_a = await _start_node(reg_a)
    addr_b, server_b = await _start_node(reg_b)
    try:
        yield reg_a, reg_b, addr_a, addr_b
    finally:
        await server_a.stop(grace=0.1)
        await server_b.stop(grace=0.1)


# ---------------------------------------------------------------------------
# Push-pull gossip convergence
# ---------------------------------------------------------------------------


async def test_one_round_converges_both_sides(two_nodes):
    reg_a, reg_b, _, addr_b = two_nodes
    report = await exchange_once(reg_a, [addr_b])
    assert report.ok
    assert report.errors == {}
    assert report.merged_cards == 1
    # Caller learned the callee's card from the pull...
    assert reg_a.get("node-b") is not None
    assert reg_a.get("node-b").models[0].model_id == NGRAM_MODEL_ID
    # ...and the callee learned the caller's card from the push.
    assert reg_b.get("node-a") is not None


async def test_third_node_view_propagates_transitively(two_nodes):
    # a gossips with b; then c (who only knows b) gossips with b and
    # must learn about a — the connected-seed-graph convergence
    # property from design doc §2.
    reg_a, _, _, addr_b = two_nodes
    await exchange_once(reg_a, [addr_b])

    reg_c = CapabilityRegistry(self_card=_card("node-c"))
    report = await exchange_once(reg_c, [addr_b])
    assert report.merged_cards == 2
    assert reg_c.get("node-a") is not None
    assert reg_c.get("node-b") is not None


async def test_repeat_rounds_are_idempotent(two_nodes):
    reg_a, _, _, addr_b = two_nodes
    first = await exchange_once(reg_a, [addr_b])
    second = await exchange_once(reg_a, [addr_b])
    assert first.merged_cards == 1
    # The second round re-receives node-b's card with a fresher
    # self-stamp (snapshot restamps), so it may merge 1 update — but
    # the fleet view must not grow.
    assert reg_a.peer_count == 1
    assert second.errors == {}


async def test_dead_peer_is_reported_not_raised(two_nodes):
    reg_a, _, _, addr_b = two_nodes
    dead = "127.0.0.1:1"  # nothing listens on port 1
    report = await exchange_once(reg_a, [dead, addr_b], timeout_s=2.0)
    assert not report.ok
    assert dead in report.errors
    assert "UNAVAILABLE" in report.errors[dead]
    # The live peer was still gossiped with.
    assert report.merged_cards == 1
    assert reg_a.get("node-b") is not None


async def test_exchange_report_ok_property():
    assert ExchangeReport(merged_cards=0, errors={}).ok
    assert not ExchangeReport(merged_cards=0, errors={"x": "y"}).ok


# ---------------------------------------------------------------------------
# GetNodeCapability probe
# ---------------------------------------------------------------------------


async def test_fetch_node_capability_returns_self_card(two_nodes):
    _, reg_b, _, addr_b = two_nodes
    card = await fetch_node_capability(addr_b)
    assert card is not None
    assert card.node_id == "node-b"
    # Probing must not have mutated the callee's registry.
    assert reg_b.peer_count == 0


async def test_fetch_node_capability_returns_none_on_dead_peer():
    assert await fetch_node_capability("127.0.0.1:1", timeout_s=2.0) is None


async def test_servicer_exposes_registry(two_nodes):
    reg_a, _, _, _ = two_nodes
    servicer = CapabilityServiceServicer(reg_a)
    assert servicer.registry is reg_a


# ---------------------------------------------------------------------------
# create_grpc_server wiring (ADR 0009 branches of grpc_app)
# ---------------------------------------------------------------------------


async def test_create_grpc_server_serves_capability_and_proposer_planes():
    registry = CapabilityRegistry(self_card=_card("runtime-node"))
    server = create_grpc_server(
        session_store=SessionStore(capacity=1),
        config=GrpcServerConfig(bind_address="127.0.0.1:0"),
        capability_registry=registry,
        proposers={NGRAM_MODEL_ID: NGramProposer()},
    )
    # Rebind to a knowable port: the config-bound port is not
    # observable, so bind a second insecure port for the test channel.
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            cap_stub = distributed_pb2_grpc.CapabilityServiceStub(channel)
            resp = await cap_stub.GetNodeCapability(
                distributed_pb2.GetNodeCapabilityRequest(),
            )
            assert resp.node.node_id == "runtime-node"

            prop_stub = distributed_pb2_grpc.ProposerServiceStub(channel)
            block = await prop_stub.ProposeBlock(
                distributed_pb2.ProposeBlockRequest(
                    committed_token_ids=[1, 2, 1, 2],
                    block_size=2,
                    num_steps=1,
                ),
            )
            assert len(block.token_ids) == 2
    finally:
        await server.stop(grace=0.1)


async def test_create_grpc_server_without_distributed_plane_is_unchanged():
    # Default construction must not register the new services: calling
    # them gets UNIMPLEMENTED, proving a v0.3-style runtime is inert.
    server = create_grpc_server(
        session_store=SessionStore(capacity=1),
        config=GrpcServerConfig(bind_address="127.0.0.1:0"),
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            cap_stub = distributed_pb2_grpc.CapabilityServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as excinfo:
                await cap_stub.GetNodeCapability(
                    distributed_pb2.GetNodeCapabilityRequest(),
                )
            assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        await server.stop(grace=0.1)
