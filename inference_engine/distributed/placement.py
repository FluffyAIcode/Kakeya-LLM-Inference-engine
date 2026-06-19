"""Deterministic spec-decode placement over a fleet snapshot (design doc §3).

Given a converged capability snapshot, pick the verifier host and the
proposer host for an AR-verifier / dLM-proposer pair. The scoring is a
pure function of the snapshot, so every node that holds the same fleet
view computes the same plan with no coordination round.

No-fallback convention (ADR 0008): if a requested role has no live
candidate, raise :class:`PlacementError` — the caller decides whether
to decode without speculation, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from inference_engine.distributed.capability import (
    CapabilityRole,
    ModelCapability,
    NodeCapability,
)


class PlacementError(RuntimeError):
    """No live node can satisfy a requested capability role."""


@dataclass(frozen=True)
class SpecDecodePlacement:
    """One planned AR-verifier / dLM-proposer pairing."""

    verifier_node: NodeCapability
    verifier_model: ModelCapability
    proposer_node: NodeCapability
    proposer_model: ModelCapability

    @property
    def colocated(self) -> bool:
        """True when both roles landed on the same node (degraded mode)."""
        return self.verifier_node.node_id == self.proposer_node.node_id

    def render(self) -> str:
        """Stable single-line summary for logs / demo output."""
        return (
            f"verifier={self.verifier_model.model_id}@{self.verifier_node.node_id}"
            f"({self.verifier_node.grpc_address}) "
            f"proposer={self.proposer_model.model_id}@{self.proposer_node.node_id}"
            f"({self.proposer_node.grpc_address}) "
            f"colocated={self.colocated}"
        )


def _candidates(
    snapshot: Iterable[NodeCapability],
    role: CapabilityRole,
    model_id: Optional[str],
) -> List[Tuple[NodeCapability, ModelCapability]]:
    out: List[Tuple[NodeCapability, ModelCapability]] = []
    for node in snapshot:
        for model in node.models_with_role(role, model_id=model_id):
            out.append((node, model))
    return out


def _score(pair: Tuple[NodeCapability, ModelCapability]) -> Tuple[float, int, str]:
    """Sort key: throughput first, memory tiebreak, node_id for determinism.

    ``node_id`` ascends while the numeric signals descend, so it is
    negated by sorting with ``reverse=True`` on a key whose final
    component is inverted lexicographically — instead we sort ascending
    on ``(-tps, -memory, node_id)`` for an unambiguous total order.
    """
    node, model = pair
    return (-model.tokens_per_second, -node.unified_memory_bytes, node.node_id)


def plan_spec_decode_placement(
    snapshot: Iterable[NodeCapability],
    *,
    verifier_model_id: Optional[str] = None,
    proposer_model_id: Optional[str] = None,
    prefer_remote_proposer: bool = True,
) -> SpecDecodePlacement:
    """Plan an AR-verifier / dLM-proposer placement from ``snapshot``.

    ``prefer_remote_proposer=True`` (default) implements the ADR 0009
    §1 motivation: evict the proposer from the verifier host whenever
    any other live node carries the proposer role, freeing the
    proposer's weight + activation bytes on the verifier host.
    Colocation is the explicit last resort, surfaced via
    ``SpecDecodePlacement.colocated``.
    """
    snapshot = list(snapshot)

    verifier_pool = _candidates(snapshot, CapabilityRole.VERIFIER, verifier_model_id)
    if not verifier_pool:
        raise PlacementError(
            f"no live node advertises a verifier"
            f"{f' for model {verifier_model_id!r}' if verifier_model_id else ''}"
        )
    verifier_node, verifier_model = sorted(verifier_pool, key=_score)[0]

    proposer_pool = _candidates(snapshot, CapabilityRole.PROPOSER, proposer_model_id)
    if not proposer_pool:
        raise PlacementError(
            f"no live node advertises a proposer"
            f"{f' for model {proposer_model_id!r}' if proposer_model_id else ''}"
        )
    if prefer_remote_proposer:
        remote_pool = [
            p for p in proposer_pool if p[0].node_id != verifier_node.node_id
        ]
        if remote_pool:
            proposer_pool = remote_pool
    proposer_node, proposer_model = sorted(proposer_pool, key=_score)[0]

    return SpecDecodePlacement(
        verifier_node=verifier_node,
        verifier_model=verifier_model,
        proposer_node=proposer_node,
        proposer_model=proposer_model,
    )
