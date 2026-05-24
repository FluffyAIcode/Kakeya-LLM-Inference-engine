"""Fixed-capacity sink+window KV slab.

A :class:`KVSlab` is one session's complete KV state, allocated once
at session start and never resized. Capacity equals
``sink_size + window_size``; appends beyond capacity automatically
slide the window forward (the sink prefix is preserved verbatim).

Why fixed-size and not growing:

  * The sink+window invariant turns the cache into a constant-size
    object. There is no need for the dynamic resize machinery a
    growing cache requires.
  * Pre-allocation avoids per-step ``torch.cat`` allocations and
    their associated fragmentation. Attention kernels see one
    contiguous tensor for the lifetime of the session.
  * Pool-managed reuse (see :mod:`pool`) makes session start
    O(1)-allocation rather than O(num_layers × capacity × head_dim).

Layout per layer:

    keys[layer, head, slot, dim]    shape [L, H, C, D]
    values[layer, head, slot, dim]  shape [L, H, C, D]
                                    where C = sink_size + window_size

The "live" region is ``[0, logical_size)``; everything outside is
pre-existing memory that attention kernels must not read. Callers
apply masks based on :attr:`logical_size`.

Sink+window trim semantics, on append of ``T`` new positions when
the slab would overflow:

  * Drop the oldest ``(logical_size + T) - capacity`` positions from
    the *window* region (i.e. positions ``[sink_size,
    logical_size)``). The sink region ``[0, sink_size)`` is never
    touched.
  * Slide the surviving window down so the kept positions occupy
    contiguous slots starting at ``sink_size``.
  * Append ``T`` new positions at the new tail.

This is the same policy the existing
``inference_engine.backends.mlx.cache.SinkWindowKVCache`` enforces;
the slab is the platform-neutral analogue.

The slab does not perform attention itself — it only stores K/V and
exposes views of the live region. Composition with attention happens
one level up (in the verifier wiring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class SlabConfig:
    """Static dimensions of a KV slab.

    All slabs in a single :class:`SlabPool` share one config so
    handed-out tensors are shape-compatible.
    """

    num_layers: int
    num_heads: int
    sink_size: int
    window_size: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.sink_size < 0:
            raise ValueError(f"sink_size must be >= 0, got {self.sink_size}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")

    @property
    def capacity(self) -> int:
        return self.sink_size + self.window_size


class KVSlab:
    """Fixed-capacity sink+window KV cache for a single session.

    All operations are O(num_layers × num_heads × head_dim × T) where
    T is the number of new positions; no allocations on the append
    or trim hot paths after construction.

    Thread-safety: a single slab is **not** thread-safe. The
    :class:`SlabPool` ensures only one consumer holds a slab at a
    time, which is sufficient for both single-session servers and
    continuous-batching schedulers (the scheduler operates on per-
    session slabs sequentially during the batched forward).
    """

    def __init__(self, config: SlabConfig) -> None:
        self.config = config
        shape = (
            config.num_layers, config.num_heads,
            config.capacity, config.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self.values = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self.logical_size = 0

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def append(
        self,
        key_steps: torch.Tensor,
        value_steps: torch.Tensor,
    ) -> None:
        """Append ``T`` positions across all layers atomically.

        Parameters
        ----------
        key_steps:
            Tensor of shape ``[num_layers, num_heads, T, head_dim]``
            with the same dtype as the slab. ``T`` may be 1 (single
            decode step) or larger (prefill or block append).
        value_steps:
            Same shape and dtype as ``key_steps``.

        Behavior on overflow: if ``logical_size + T > capacity``, the
        slab first slides the window to make room, then appends.
        ``T`` itself must be at most ``window_size`` (it is illegal
        to append more new positions than the window can hold);
        otherwise we raise ``ValueError`` rather than silently
        dropping positions.
        """
        self._validate_step_shape(key_steps, name="key_steps")
        self._validate_step_shape(value_steps, name="value_steps")
        if key_steps.shape != value_steps.shape:
            raise ValueError(
                f"key_steps and value_steps shape mismatch: "
                f"{tuple(key_steps.shape)} vs {tuple(value_steps.shape)}"
            )
        T = int(key_steps.shape[2])
        if T == 0:
            raise ValueError("append: T must be positive")
        if T > self.config.window_size:
            raise ValueError(
                f"append: T={T} exceeds window_size={self.config.window_size}; "
                "callers must split into multiple appends"
            )

        if self.logical_size + T > self.config.capacity:
            self._trim_window_to_fit(T)
        sink = self.config.sink_size
        # logical_size is now <= capacity - T, so the destination slice
        # is fully in-bounds.
        end = self.logical_size + T
        self.keys[:, :, self.logical_size:end, :] = key_steps
        self.values[:, :, self.logical_size:end, :] = value_steps
        self.logical_size = end
        # Sanity: the sink region must be untouched after a trim.
        del sink  # used only for the trim invariant

    def truncate(self, drop: int) -> int:
        """Drop the last ``drop`` positions from the live region.

        Used by the speculative decoder when the verifier rejects
        proposed tokens. Returns the number actually dropped (== drop
        unless the slab had fewer live positions, in which case we
        raise rather than silently truncating less).
        """
        if drop < 0:
            raise ValueError(f"drop must be >= 0, got {drop}")
        if drop > self.logical_size:
            raise ValueError(
                f"drop={drop} exceeds logical_size={self.logical_size}"
            )
        self.logical_size -= drop
        return drop

    def reset(self) -> None:
        """Empty the slab; underlying buffers are kept allocated.

        Called by the pool on release so the slab is ready to serve
        a fresh session without re-allocating tensors.
        """
        self.logical_size = 0
        # We do not zero the buffers; logical_size is the truth.
        # Callers must respect logical_size when reading.

    # ------------------------------------------------------------------
    # Read views
    # ------------------------------------------------------------------

    def view(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(keys_view, values_view)`` for a layer's live region.

        Each view has shape ``[num_heads, logical_size, head_dim]``
        (the layer dim is stripped). Views share storage with the
        slab's underlying buffers; mutating them mutates the slab.
        Length-zero views are valid and have shape
        ``[num_heads, 0, head_dim]``.
        """
        if not (0 <= layer_idx < self.config.num_layers):
            raise IndexError(
                f"layer_idx={layer_idx} out of range [0, {self.config.num_layers})"
            )
        return (
            self.keys[layer_idx, :, :self.logical_size, :],
            self.values[layer_idx, :, :self.logical_size, :],
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def is_full(self) -> bool:
        return self.logical_size >= self.config.capacity

    @property
    def kv_bytes(self) -> int:
        """Bytes occupied by the slab's K and V buffers (capacity, not
        logical). This is what shows up in unified-memory accounting
        because the buffers are allocated and resident regardless of
        live size."""
        per_tensor = self.keys.numel() * self.keys.element_size()
        return per_tensor * 2  # keys + values

    @property
    def live_kv_bytes(self) -> int:
        """Bytes for the live region only (logical_size, not capacity).

        Useful for reporting "how much KV is currently in use" vs the
        physical footprint reported by :attr:`kv_bytes`.
        """
        if self.logical_size == 0:
            return 0
        elem = self.keys.element_size()
        per_layer_per_head = self.logical_size * self.config.head_dim * elem
        return (
            self.config.num_layers * self.config.num_heads * per_layer_per_head * 2
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_step_shape(self, t: torch.Tensor, *, name: str) -> None:
        if t.dim() != 4:
            raise ValueError(
                f"{name} must be 4-D [num_layers, num_heads, T, head_dim]; "
                f"got shape {tuple(t.shape)}"
            )
        L, H, _T, D = t.shape
        if L != self.config.num_layers:
            raise ValueError(
                f"{name}.shape[0]={L} does not match num_layers={self.config.num_layers}"
            )
        if H != self.config.num_heads:
            raise ValueError(
                f"{name}.shape[1]={H} does not match num_heads={self.config.num_heads}"
            )
        if D != self.config.head_dim:
            raise ValueError(
                f"{name}.shape[3]={D} does not match head_dim={self.config.head_dim}"
            )
        if t.dtype != self.config.dtype:
            raise ValueError(
                f"{name}.dtype={t.dtype} does not match slab dtype={self.config.dtype}"
            )

    def _trim_window_to_fit(self, n_new: int) -> None:
        """Slide the window so ``n_new`` positions can be appended.

        Drops ``excess = (logical_size + n_new) - capacity`` from the
        oldest part of the window region, slides the surviving window
        down to remain contiguous starting at ``sink_size``, and
        decrements ``logical_size`` accordingly.
        """
        excess = (self.logical_size + n_new) - self.config.capacity
        if excess <= 0:  # pragma: no cover - guarded by caller
            return
        sink = self.config.sink_size
        # Window region currently lives at [sink, logical_size). We
        # want to keep [sink + excess, logical_size) and move it to
        # start at sink.
        src_start = sink + excess
        src_end = self.logical_size
        dst_start = sink
        dst_end = dst_start + (src_end - src_start)
        if src_end > src_start:
            self.keys[:, :, dst_start:dst_end, :] = self.keys[:, :, src_start:src_end, :]
            self.values[:, :, dst_start:dst_end, :] = self.values[:, :, src_start:src_end, :]
        self.logical_size -= excess
