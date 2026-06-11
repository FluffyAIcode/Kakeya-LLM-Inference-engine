"""Optional ``mlx.distributed`` ring probe (ADR 0009 §4 item 4).

Mirrors the no-fallback / no-mock pattern of
``inference_engine.backends.mlx.env``: pure metadata probing that runs
on any host. Off Apple Silicon (or outside an ``mlx.launch`` job) the
probe reports a structured "unavailable" with the reason; it never
raises and never falls back.

The ring is the *data plane* of the hybrid decision in ADR 0009: nodes
inside a ring advertise ``ring_address`` on their capability card so
bulk-tensor flows (K3 DFlash aux hidden states, intra-verifier tensor
parallelism) can be promoted off gRPC. Control-plane code never
requires it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from inference_engine.distributed.capability import NodeCapability


@dataclass(frozen=True)
class RingEnvironment:
    """Structured snapshot of mlx.distributed availability.

    ``is_available`` is True only when ``mlx.core.distributed`` imports
    AND ``init()`` reports an initialized group of size >= 1 (i.e. the
    process was launched in a distributed context or single-rank
    fallback-free init succeeded).
    """

    is_available: bool
    backend: str
    rank: int
    world_size: int
    failure_reason: str

    def render(self) -> str:
        """Stable single-line summary for logs / capability cards."""
        if self.is_available:
            return (
                f"mlx ring OK: backend={self.backend} "
                f"rank={self.rank}/{self.world_size}"
            )
        return f"mlx ring UNAVAILABLE ({self.failure_reason})"

    def ring_address(self, hostname: str) -> str:
        """The ``NodeCapability.ring_address`` value to advertise."""
        if not self.is_available:
            return ""
        return f"{hostname}:{self.rank}"


def probe_ring_environment() -> RingEnvironment:
    """Detect mlx.distributed availability without raising."""
    try:
        mx_dist = importlib.import_module("mlx.core.distributed")
    except Exception as exc:
        return RingEnvironment(
            is_available=False,
            backend="",
            rank=0,
            world_size=0,
            failure_reason=(
                f"mlx.core.distributed import failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    # Apple Silicon path from here on; exercised by the Mac M4 gate.
    try:  # pragma: no cover - requires mlx runtime
        if not mx_dist.is_available():
            return RingEnvironment(
                is_available=False,
                backend="",
                rank=0,
                world_size=0,
                failure_reason="mx.distributed.is_available() returned False",
            )
        group = mx_dist.init()
        return RingEnvironment(
            is_available=True,
            backend="ring",
            rank=int(group.rank()),
            world_size=int(group.size()),
            failure_reason="",
        )
    except Exception as exc:  # pragma: no cover - requires mlx runtime
        return RingEnvironment(
            is_available=False,
            backend="",
            rank=0,
            world_size=0,
            failure_reason=f"mx.distributed.init() failed: {type(exc).__name__}: {exc}",
        )


def ring_path_available(a: NodeCapability, b: NodeCapability) -> bool:
    """True when a bulk-tensor flow between ``a`` and ``b`` can be
    promoted from gRPC to the mlx.distributed ring (both nodes
    advertise a ring endpoint). Placement-time predicate; the first
    planned consumer is the K3 DFlash hidden-state flow (F3 in ADR
    0009 §2)."""
    return bool(a.ring_address) and bool(b.ring_address)
