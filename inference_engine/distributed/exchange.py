"""CapabilityService servicer + gossip client (design doc §2).

Server side: :class:`CapabilityServiceServicer` is a thin adapter
between the wire messages and :class:`CapabilityRegistry` — merge what
the caller pushed, reply with the merged snapshot.

Client side: :func:`exchange_once` performs one push-pull round with a
list of peers. Failures are collected per peer, never raised through:
a dead peer must not stop gossip with the live ones (failure model,
design doc §6). The periodic loop wrapper lives in the server
launcher, not here, so this module stays trivially testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import grpc
import grpc.aio

from inference_engine.distributed.capability import (
    CapabilityRegistry,
    NodeCapability,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)

_LOG = logging.getLogger("kakeya.distributed.exchange")

DEFAULT_EXCHANGE_TIMEOUT_S = 5.0


class CapabilityServiceServicer(distributed_pb2_grpc.CapabilityServiceServicer):
    """gRPC adapter over a node's :class:`CapabilityRegistry`."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry

    async def ExchangeCapabilities(  # noqa: N802 - gRPC casing
        self,
        request: distributed_pb2.ExchangeCapabilitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> distributed_pb2.ExchangeCapabilitiesResponse:
        pushed = [NodeCapability.from_proto(m) for m in request.known_nodes]
        changed = self._registry.merge(pushed)
        if changed:
            _LOG.debug("gossip merge updated %d card(s)", changed)
        return distributed_pb2.ExchangeCapabilitiesResponse(
            known_nodes=[c.to_proto() for c in self._registry.snapshot()],
        )

    async def GetNodeCapability(  # noqa: N802 - gRPC casing
        self,
        request: distributed_pb2.GetNodeCapabilityRequest,
        context: grpc.aio.ServicerContext,
    ) -> distributed_pb2.GetNodeCapabilityResponse:
        return distributed_pb2.GetNodeCapabilityResponse(
            node=self._registry.self_card.to_proto(),
        )


def add_capability_service(
    server: grpc.aio.Server, registry: CapabilityRegistry,
) -> CapabilityServiceServicer:
    """Register a CapabilityService for ``registry`` on ``server``."""
    servicer = CapabilityServiceServicer(registry)
    distributed_pb2_grpc.add_CapabilityServiceServicer_to_server(servicer, server)
    return servicer


@dataclass(frozen=True)
class ExchangeReport:
    """Outcome of one gossip round across a peer list."""

    merged_cards: int
    errors: Dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.errors


async def exchange_once(
    registry: CapabilityRegistry,
    peers: Sequence[str],
    *,
    timeout_s: float = DEFAULT_EXCHANGE_TIMEOUT_S,
) -> ExchangeReport:
    """One push-pull gossip round with every address in ``peers``.

    Per peer: push our full snapshot, merge the peer's reply. Errors
    (connect refused, deadline, …) are recorded per peer address and
    do not interrupt the round for remaining peers.
    """
    merged = 0
    errors: Dict[str, str] = {}
    for peer in peers:
        request = distributed_pb2.ExchangeCapabilitiesRequest(
            known_nodes=[c.to_proto() for c in registry.snapshot()],
        )
        try:
            async with grpc.aio.insecure_channel(peer) as channel:
                stub = distributed_pb2_grpc.CapabilityServiceStub(channel)
                response = await stub.ExchangeCapabilities(
                    request, timeout=timeout_s,
                )
        except grpc.aio.AioRpcError as exc:
            errors[peer] = f"{exc.code().name}: {exc.details()}"
            _LOG.warning("gossip with %s failed: %s", peer, errors[peer])
            continue
        merged += registry.merge(
            NodeCapability.from_proto(m) for m in response.known_nodes
        )
    return ExchangeReport(merged_cards=merged, errors=errors)


async def fetch_node_capability(
    address: str,
    *,
    timeout_s: float = DEFAULT_EXCHANGE_TIMEOUT_S,
) -> Optional[NodeCapability]:
    """Liveness probe: fetch one node's own card, or None on failure."""
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            stub = distributed_pb2_grpc.CapabilityServiceStub(channel)
            response = await stub.GetNodeCapability(
                distributed_pb2.GetNodeCapabilityRequest(), timeout=timeout_s,
            )
    except grpc.aio.AioRpcError as exc:
        _LOG.warning("probe of %s failed: %s", address, exc.code().name)
        return None
    return NodeCapability.from_proto(response.node)
