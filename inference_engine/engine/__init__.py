"""Kakeya Inference Engine (product runtime).

Bounded-KV-native LLM inference engine — the product-grade vLLM replacement
defined in ADR 0015 and `docs/design/kakeya-inference-engine-architecture.md`.

Public surface:
  * :mod:`inference_engine.engine.admission` — peak-window admission + the
    bounded-KV memory model (pure stdlib; the concurrency math).
  * :mod:`inference_engine.engine.kakeya_engine` — the engine runtime
    (chunked restoration prefill + bounded-KV decode). Imports torch lazily.
"""

from inference_engine.engine.admission import (
    BoundedKVModel,
    full_kv_bytes_per_session,
    max_concurrent_sessions,
    resident_kv_bytes_per_session,
)

__all__ = [
    "BoundedKVModel",
    "resident_kv_bytes_per_session",
    "full_kv_bytes_per_session",
    "max_concurrent_sessions",
]
