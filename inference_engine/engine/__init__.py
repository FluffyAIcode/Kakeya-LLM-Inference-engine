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
# kakeya_engine imports torch only lazily (inside methods), so this is safe on
# torch-less hosts; the class is the engine runtime entry point.
from inference_engine.engine.kakeya_engine import KakeyaEngine
# kakeya_vllm defers the vllm import to construction, so importing these names is
# safe on torch/vllm-free hosts; KakeyaVLLM is the v0.5-CUDA product entrypoint.
from inference_engine.engine.kakeya_vllm import (
    KakeyaVLLM,
    KakeyaVLLMConfig,
    kakeya_hf_overrides,
    kakeya_window_total,
)

__all__ = [
    "BoundedKVModel",
    "resident_kv_bytes_per_session",
    "full_kv_bytes_per_session",
    "max_concurrent_sessions",
    "KakeyaEngine",
    "KakeyaVLLM",
    "KakeyaVLLMConfig",
    "kakeya_hf_overrides",
    "kakeya_window_total",
]
