"""Multi-host plane for Kakeya (ADR 0009 / 0016 / 0017).

Subpackage layout:

- :mod:`capability` — capability cards + the converging gossip
  registry (``NodeCapability`` / ``ModelCapability`` /
  ``CapabilityRegistry``).
- :mod:`prefill_worker` — queued prefill-only compute workers.
- :mod:`prefill_cache` / :mod:`prefill_cache_service` — immutable RAM cache.
- :mod:`prefill_cache_runtime` — primary-side orchestration and fallback.
- :mod:`prefill_scheduler` — load/cost-aware worker and replica placement.
- :mod:`prefill_auth` — fleet-PSK authentication and tenant hash isolation.
- :mod:`exchange` — ``CapabilityService`` gRPC servicer + the
  ``exchange_once`` gossip client.
- :mod:`ngram`, :mod:`proposer_service`, :mod:`spec_decode` — legacy research
  proposer paths retained for reproducibility; not the product architecture.
- :mod:`mlx_ring` — optional ``mlx.distributed`` ring probe
  (bulk-tensor data plane advertisement, ADR 0009 §4 item 4).

The control plane (everything except :mod:`mlx_ring`) is fully
platform-neutral and runs on Linux CPU fleets; ``mlx_ring`` degrades
to a structured "unavailable" probe off Apple Silicon.
"""

from inference_engine.distributed.capability import (
    CapabilityRegistry,
    CapabilityRole,
    CacheCompatibility,
    CompressionCodec,
    ModelCapability,
    NodeCapability,
    PrefillWorkerCapability,
)
from inference_engine.distributed.placement import (
    PlacementError,
    SpecDecodePlacement,
    plan_spec_decode_placement,
)

__all__ = [
    "CapabilityRegistry",
    "CapabilityRole",
    "CacheCompatibility",
    "CompressionCodec",
    "ModelCapability",
    "NodeCapability",
    "PrefillWorkerCapability",
    "PlacementError",
    "SpecDecodePlacement",
    "plan_spec_decode_placement",
]
