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
from inference_engine.v04.restored_attention import (
    apply_rope_to_k_at_positions,
    prepare_restored_attention_kv,
    slice_position_embeddings,
)
from inference_engine.v04.dlm_restored_verifier import DLMRestoredVerifier
from inference_engine.v04.dflash_drafter import (
    AuxHiddenProvider,
    DFlashConfig,
    DFlashDrafter,
    DFlashProposer,
)
from inference_engine.v04.f_theta import FThetaConfig, FThetaProjection
from inference_engine.v04.cross_model_dlm_verifier import (
    CrossModelDLMRestoredVerifier,
    CrossModelLayerMapping,
)
from inference_engine.v04.kv_compressor import (
    IdentityCompressor,
    KakeyaLatticeCompressor,
    KakeyaLatticeUnavailable,
    KVCompressor,
    make_default_compressor,
)
from inference_engine.v04.niah_eval import (
    DEFAULT_NEEDLE_PREFIXES,
    NIAHEvalResult,
    NIAHSample,
    aggregate_attention_window_metrics,
    aggregate_recall,
    compute_effective_attention_window,
    evaluate,
    format_attention_window_summary,
    format_memory_summary,
    greedy_decode_oracle,
    greedy_decode_sink_window,
    greedy_decode_v04,
    make_niah_dataset,
    make_sink_window_4d_mask,
    recall_predicate,
    record_memory,
    reset_memory_peak,
)

__all__ = [
    # K1.A — capture
    "KVCapture",
    "capture_proposer_kv",
    "register_kv_capture_hooks",
    # K1.B — merge
    "compute_evicted_positions",
    "merge_kv_at_evicted_positions",
    # K1.C — restored attention K/V preparation
    "apply_rope_to_k_at_positions",
    "prepare_restored_attention_kv",
    "slice_position_embeddings",
    # K1.D — end-to-end wrapper
    "DLMRestoredVerifier",
    # K1.E — NIAH validation harness
    "DEFAULT_NEEDLE_PREFIXES",
    "NIAHEvalResult",
    "NIAHSample",
    "aggregate_recall",
    "evaluate",
    "greedy_decode_oracle",
    "greedy_decode_sink_window",
    "greedy_decode_v04",
    "make_niah_dataset",
    "make_sink_window_4d_mask",
    "recall_predicate",
    # K1.G — memory tracking
    "format_memory_summary",
    "record_memory",
    "reset_memory_peak",
    # K1.H — effective attention-window metric
    "aggregate_attention_window_metrics",
    "compute_effective_attention_window",
    "format_attention_window_summary",
    # K2.A — KV compressor protocol + reference impls (see ADR 0008 §11.11)
    "IdentityCompressor",
    "KakeyaLatticeCompressor",
    "KakeyaLatticeUnavailable",
    "KVCompressor",
    "make_default_compressor",
    # K3 — native DFlash drafter (Stage 1: module + proposer; see
    # docs/design/k3-cross-model-dlmrestored-verifier-contract.md)
    "AuxHiddenProvider",
    "DFlashConfig",
    "DFlashDrafter",
    "DFlashProposer",
    # K3 Block C — f_θ K/V projection
    "FThetaConfig",
    "FThetaProjection",
    # K3 Block B — cross-model DLMRestoredVerifier with f_θ-mediated
    # K/V Restoration (the integrated Kakeya inference architecture)
    "CrossModelDLMRestoredVerifier",
    "CrossModelLayerMapping",
]
