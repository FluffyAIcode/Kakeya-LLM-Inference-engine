"""First-run UX helpers for the v0.3 runtime.

Today this package is just the ``prewarm`` module — a small set of
pure-Python helpers used by ``scripts/kakeya_prewarm.py`` and by
``scripts/start_grpc_runtime_server.py``'s cache-check pre-flight.
Kept under ``inference_engine.setup`` rather than under
``inference_engine.server`` because it is platform-neutral
(operates on the HF cache filesystem, no torch / mlx dependency).
"""

from .prewarm import (
    HF_CACHE_DEFAULT,
    PrewarmStatus,
    assert_cached_or_raise,
    cache_dir_for_model,
    free_disk_bytes,
    is_model_in_cache,
    prewarm_model_id,
    snapshot_size_bytes,
)

__all__ = [
    "HF_CACHE_DEFAULT",
    "PrewarmStatus",
    "assert_cached_or_raise",
    "cache_dir_for_model",
    "free_disk_bytes",
    "is_model_in_cache",
    "prewarm_model_id",
    "snapshot_size_bytes",
]
