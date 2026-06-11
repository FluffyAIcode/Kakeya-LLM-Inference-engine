"""MLX-side sink+window KV cache.

Subclasses mlx_lm's ``_BaseCache`` so a list of
``SinkWindowKVCache`` instances is a drop-in replacement for the
``KVCache`` list returned by ``mlx_lm.models.cache.make_prompt_cache``.

Why a custom cache class (vs. mutating mlx_lm's KVCache after the
fact, which is what MLX-1b tried):

  * ``KVCache`` uses step=256 buffer pre-allocation. After a write,
    ``self.keys.shape[2]`` is rounded up to a step multiple, while the
    *logical* size is ``self.offset``. Replacing ``self.keys`` with
    a smaller tensor leaves the next ``update_and_fetch`` to allocate
    a fresh buffer and copy ``self.keys[..., :prev, :]`` where
    ``prev = self.offset`` — but ``prev`` may now exceed the new
    buffer's seq dim, so that copy is a silent out-of-bounds. This
    was the source of the divergence after token 34 in MLX-1b
    (`bench_mlx_verifier_1779507043.json`, common_prefix_length=34,
    repeated `3554` token salad after).

  * The right contract is exposed by mlx_lm itself: every cache
    has ``update_and_fetch(keys, values) -> (k, v)`` and (optionally)
    ``make_mask(N, return_array, window_size)``. Implementing those
    two atomically, with our trim happening *inside* update_and_fetch,
    leaves the cache invariants self-consistent and matches the
    interface ``mlx_lm.models.qwen3.Attention`` calls into.

Sink+window semantics (matching ``kv_cache_proposer/verifier.py``):

  * The first ``sink_size`` tokens of the cache (the "attention sinks")
    are never evicted.
  * The most recent ``window_size`` tokens slide forward as new tokens
    arrive.
  * ``self.offset`` tracks the **global** token position (so RoPE on
    the next query rotates at the true distance, regardless of which
    middle tokens have been evicted). The internal buffer is bounded
    by ``sink_size + window_size``.
  * ``update_and_fetch(new_k, new_v)`` returns the *full*
    ``(pre_buffer ++ new_k, pre_buffer_v ++ new_v)`` for THIS step's
    attention — i.e. the model temporarily sees a possibly-oversized
    K during the forward — but stores the trimmed (sink + window)
    tensor for the NEXT step. This preserves correctness inside one
    forward while bounding persistent memory.
  * ``make_mask(N, ...)`` returns a causal mask whose offset reflects
    the cache's *actual* buffer size at the start of this step (not
    the global offset), so it lines up with the K shape returned by
    ``update_and_fetch``.

The module imports ``mlx.core`` at top level; on non-arm64 hosts that
import fails, by design.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

# ``_BaseCache`` and the small helper ``create_attention_mask`` are
# private inside mlx_lm but stable in their public submodule path. We
# import them directly so our class behaves identically to the built-in
# caches anywhere mlx_lm dispatches on the cache type.
from mlx_lm.models.cache import _BaseCache, create_attention_mask


class SinkWindowKVCache(_BaseCache):
    """Sink + sliding-window KV cache.

    Parameters
    ----------
    sink_size
        Number of leading tokens to retain unconditionally.
    window_size
        Number of trailing tokens to retain (sliding window). Must be
        positive.

    Attributes
    ----------
    keys, values
        ``mx.array`` of shape ``[B, n_kv_heads, S, head_dim]`` where
        ``S <= sink_size + window_size``, or ``None`` before the first
        update.
    offset
        Global token position counter. Incremented by ``L`` on each
        ``update_and_fetch`` call. Used by ``Attention.__call__`` to
        rotate the new queries / keys at the correct global position
        for RoPE.
    """

    step = 256  # not used internally (we always allocate fresh) but
    # kept as an attribute for compatibility with mlx_lm code paths
    # that read it.

    def __init__(self, sink_size: int = 4, window_size: int = 64) -> None:
        if sink_size < 0:
            raise ValueError("sink_size must be >= 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.sink_size = sink_size
        self.window_size = window_size
        self.keys: Optional["mx.array"] = None
        self.values: Optional["mx.array"] = None
        self.restored_keys: Optional["mx.array"] = None
        self.restored_values: Optional["mx.array"] = None
        self.offset: int = 0

    # ------------------------------------------------------------------ #
    # The two methods mlx_lm's Attention layer actually calls
    # ------------------------------------------------------------------ #

    def update_and_fetch(self, keys: "mx.array", values: "mx.array"):
        """Append `(keys, values)` and return the full K, V for this step.

        The returned tensors include the new keys/values in their tail,
        so the model can attend over them this step. The *stored* state
        (``self.keys``, ``self.values``) is trimmed to sink+window for
        the next step.
        """
        if keys.ndim != 4 or values.ndim != 4:
            raise RuntimeError(
                "SinkWindowKVCache.update_and_fetch expects 4-D K/V "
                f"(got K.ndim={keys.ndim}, V.ndim={values.ndim})"
            )
        L = int(keys.shape[2])
        if int(values.shape[2]) != L:
            raise RuntimeError(
                f"K and V have mismatched seq dims: K={keys.shape}, V={values.shape}"
            )

        if self.keys is None:
            local_k, local_v = keys, values
        else:
            local_k = mx.concatenate([self.keys, keys], axis=2)
            local_v = mx.concatenate([self.values, values], axis=2)

        if self.restored_keys is not None:
            full_k = mx.concatenate([self.restored_keys, local_k], axis=2)
            full_v = mx.concatenate([self.restored_values, local_v], axis=2)
        else:
            full_k, full_v = local_k, local_v

        # Advance the global counter — this is what RoPE on the NEXT
        # forward will use as `cache.offset`.
        self.offset += L

        budget = self.sink_size + self.window_size
        if int(local_k.shape[2]) > budget:
            # Persistent (next-step) state: keep the sink + sliding window.
            sink_k = local_k[:, :, : self.sink_size, :]
            sink_v = local_v[:, :, : self.sink_size, :]
            tail_k = local_k[:, :, -self.window_size :, :]
            tail_v = local_v[:, :, -self.window_size :, :]
            self.keys = mx.concatenate([sink_k, tail_k], axis=2)
            self.values = mx.concatenate([sink_v, tail_v], axis=2)
        else:
            self.keys = local_k
            self.values = local_v

        # Returned tensors are the *full* concatenation for this step's
        # attention. They may exceed budget by L_new during prefill; the
        # persistent state stored above is bounded.
        return full_k, full_v

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
    ):
        """Build the attention mask for an upcoming forward of length N.

        Called by ``mlx_lm.models.base.create_attention_mask(h, cache)``
        BEFORE ``update_and_fetch`` runs. The mask has shape
        ``[N, pre_len + N]`` matching the K that will be returned by
        ``update_and_fetch``.
        """
        pre_len = 0 if self.keys is None else int(self.keys.shape[2])
        if self.restored_keys is not None:
            pre_len += int(self.restored_keys.shape[2])
        # Delegate to mlx_lm's helper, supplying our pre-update buffer
        # length as the ``offset`` argument (the helper builds a
        # standard causal mask of shape [N, offset+N]).
        return create_attention_mask(
            N, offset=pre_len, return_array=return_array, window_size=window_size
        )

    # ------------------------------------------------------------------ #
    # _BaseCache contract pieces
    # ------------------------------------------------------------------ #

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Drop the last ``n`` tokens from the buffer.

        Used by the speculative decoder when a forwarded block was
        partially or fully rejected: we drop the unaccepted tail K/V so
        the cache reflects only the committed prefix.
        """
        if self.keys is None:
            return 0
        physical = int(self.keys.shape[2])
        n = max(0, min(physical, n))
        if n > 0:
            self.keys = self.keys[..., : physical - n, :]
            self.values = self.values[..., : physical - n, :]
            self.offset -= n
        return n

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes + self.values.nbytes
        if self.restored_keys is not None:
            total += self.restored_keys.nbytes + self.restored_values.nbytes
        return total

    def set_restored_bank(self, keys: "mx.array", values: "mx.array") -> None:
        """Attach post-RoPE restored K/V used only during decode.

        ``keys`` and ``values`` must already be in cache layout
        ``[B, n_kv_heads, T_restored, head_dim]``. The bank is prepended
        transiently in ``update_and_fetch`` but is not part of the local
        sink/window buffer that gets trimmed on every step.
        """
        if keys.ndim != 4 or values.ndim != 4:
            raise RuntimeError("restored K/V bank must be 4-D")
        if int(keys.shape[2]) != int(values.shape[2]):
            raise RuntimeError("restored K/V bank seq dimensions differ")
        self.restored_keys = keys
        self.restored_values = values

    # ---- state / meta_state for save_prompt_cache compatibility ------- #

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.sink_size, self.window_size, self.offset))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.sink_size, self.window_size, self.offset = map(int, v)


def make_sink_window_cache(
    model, *, sink_size: int, window_size: int
) -> list:
    """Build a per-layer list of :class:`SinkWindowKVCache`.

    Mirrors ``mlx_lm.models.cache.make_prompt_cache`` but always returns
    sink+window caches sized to ``(sink_size, window_size)``.
    """
    num_layers = len(model.layers)
    return [
        SinkWindowKVCache(sink_size=sink_size, window_size=window_size)
        for _ in range(num_layers)
    ]


def total_kv_bytes(cache: list) -> int:
    """Sum K/V byte sizes across a per-layer cache list."""
    return sum(int(getattr(layer, "nbytes", 0)) for layer in cache)


def cache_seq_length(cache: list) -> int:
    """Return the seq-length of the first non-empty layer, or 0."""
    for layer in cache:
        keys = getattr(layer, "keys", None)
        if keys is not None:
            return int(keys.shape[2])
    return 0
