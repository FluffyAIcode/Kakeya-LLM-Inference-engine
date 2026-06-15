"""Peak-window admission + bounded-KV memory model (§2, §7).

Pure stdlib — this is the concurrency math the engine admits sessions by, and
the model that quantifies the bounded-KV advantage over a full-KV engine. A
session's cost is its **bounded resident KV** (exact layers full + other layers
sink+window), NOT its token count — so concurrency does not degrade as
conversations lengthen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BoundedKVModel:
    """Per-session bounded-KV cost model for a given model + engine config.

    Attributes
    ----------
    num_layers, num_kv_heads, head_dim
        Verifier layer layout (KV side).
    n_exact_layers
        Number of full-context (recall-critical) layers kept exact.
    sink, window
        Resident sink + sliding window for the non-exact layers.
    dtype_bytes
        Bytes per KV element (bf16 → 2).
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    n_exact_layers: int
    sink: int
    window: int
    dtype_bytes: int = 2

    def __post_init__(self) -> None:
        if self.n_exact_layers > self.num_layers:
            raise ValueError("n_exact_layers cannot exceed num_layers")
        for name in ("num_layers", "num_kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sink < 0 or self.window <= 0:
            raise ValueError("sink must be >=0 and window must be >0")

    @property
    def per_token_per_layer_bytes(self) -> int:
        """K and V for one token at one layer."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def resident_bytes(self, context_len: int) -> int:
        """Bounded resident KV for a session at logical length ``context_len``."""
        if context_len < 0:
            raise ValueError("context_len must be >= 0")
        n_other = self.num_layers - self.n_exact_layers
        exact = self.n_exact_layers * context_len
        other = n_other * min(context_len, self.sink + self.window)
        return (exact + other) * self.per_token_per_layer_bytes

    def full_kv_bytes(self, context_len: int) -> int:
        """Full-KV cost (every layer keeps full context) — the full-attention
        engine's per-session cost, for the advantage ratio."""
        if context_len < 0:
            raise ValueError("context_len must be >= 0")
        return self.num_layers * context_len * self.per_token_per_layer_bytes

    def advantage_ratio(self, context_len: int) -> float:
        """full-KV / bounded-KV per-session bytes at ``context_len``."""
        b = self.resident_bytes(context_len)
        return self.full_kv_bytes(context_len) / b if b else float("inf")


def resident_kv_bytes_per_session(model: BoundedKVModel, context_len: int) -> int:
    return model.resident_bytes(context_len)


def full_kv_bytes_per_session(model: BoundedKVModel, context_len: int) -> int:
    return model.full_kv_bytes(context_len)


def max_concurrent_sessions(
    *, memory_budget_bytes: int, model_weight_bytes: int,
    per_session_bytes: int,
) -> int:
    """Max sessions that fit: (budget − weights) // per-session KV."""
    if per_session_bytes <= 0:
        raise ValueError("per_session_bytes must be positive")
    free = memory_budget_bytes - model_weight_bytes
    if free <= 0:
        return 0
    return int(free // per_session_bytes)


def exact_layer_indices_for_layer_types(layer_types: Sequence[str]) -> list:
    """Indices of full-attention layers given a hybrid model's layer_types."""
    return [i for i, t in enumerate(layer_types)
            if "full" in str(t).lower()]
