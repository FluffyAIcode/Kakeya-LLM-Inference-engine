"""Multi-host plane for Kakeya (ADR 0009, v0.5-M1).

Subpackage layout:

- :mod:`capability` — capability cards + the converging gossip
  registry (``NodeCapability`` / ``ModelCapability`` /
  ``CapabilityRegistry``).
- :mod:`placement` — deterministic spec-decode placement over a
  fleet snapshot.
- :mod:`exchange` — ``CapabilityService`` gRPC servicer + the
  ``exchange_once`` gossip client.
- :mod:`ngram` — model-free prompt-lookup proposer (the always-
  available proposer capability every node can advertise).
- :mod:`proposer_service` — ``ProposerService`` gRPC servicer +
  ``RemoteProposer`` client (drop-in ``DLMProposer`` substitute).
- :mod:`spec_decode` — pure greedy accept rule +
  ``DistributedSpeculativeDecoder``.
- :mod:`mlx_ring` — optional ``mlx.distributed`` ring probe
  (bulk-tensor data plane advertisement, ADR 0009 §4 item 4).

The control plane (everything except :mod:`mlx_ring`) is fully
platform-neutral and runs on Linux CPU fleets; ``mlx_ring`` degrades
to a structured "unavailable" probe off Apple Silicon.
"""

from inference_engine.distributed.capability import (
    CapabilityRegistry,
    CapabilityRole,
    ModelCapability,
    NodeCapability,
)
from inference_engine.distributed.placement import (
    PlacementError,
    SpecDecodePlacement,
    plan_spec_decode_placement,
)

__all__ = [
    "CapabilityRegistry",
    "CapabilityRole",
    "ModelCapability",
    "NodeCapability",
    "PlacementError",
    "SpecDecodePlacement",
    "plan_spec_decode_placement",
]
