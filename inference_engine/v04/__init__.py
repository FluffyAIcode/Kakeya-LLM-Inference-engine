"""Kakeya Inference Engine v0.4 architecture.

This subpackage implements the v0.4 GA design as specified in
ADR 0008 §11 (the v0.4 amendment dated 2026-06-08): the verifier
maintains a minimal sink+window KV cache, and at every generation
step accepts transient K/V tensors at evicted positions reconstructed
from the dLM proposer's parallel forward pass.

The architecture's load-bearing fact, recorded in ADR 0008 §11.3:
the dLM proposer has no KV cache, so its K/V tensors at every
position are computed transiently each forward and discarded. This
makes the proposer a constant-memory K/V reconstruction source.

Implementation phases per ADR 0008 §11.7:

* **K1**: same-model toy (proposer and verifier share Gemma 3-1B
  weights). Implement K/V routing infrastructure. Validate on
  synthetic NIAH that recall ≈ oracle when the projection is
  identity.
* **K2**: cross-model toy (proposer = Gemma 3-1B, verifier = Gemma
  3-4B). Train per-layer linear projection f_θ.
* **K3**: production scale.
* **K4**: KakeyaLattice composition.
* **K5**: default flip + docs.

This `__init__.py` is intentionally a thin re-export layer. The
production-style API (a `DLMRestoredVerifier` class wrapping the
whole pipeline) lands in K1.C; K1.A / K1.B build the foundation.
"""

from inference_engine.v04.kv_capture import (
    KVCapture,
    capture_proposer_kv,
    register_kv_capture_hooks,
)
from inference_engine.v04.kv_merge import (
    compute_evicted_positions,
    merge_kv_at_evicted_positions,
)

__all__ = [
    # K1.A — capture
    "KVCapture",
    "capture_proposer_kv",
    "register_kv_capture_hooks",
    # K1.B — merge
    "compute_evicted_positions",
    "merge_kv_at_evicted_positions",
]
