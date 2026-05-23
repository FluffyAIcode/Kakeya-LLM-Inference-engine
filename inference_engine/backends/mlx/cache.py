"""Sink+window trim helpers for `mlx_lm.models.cache.KVCache`.

The probe (Phase MLX-1a) confirmed that `make_prompt_cache(model)`
returns `list[KVCache]` with these per-layer attributes we need:

  * ``keys``    : ``mx.array`` of shape ``[batch, n_kv_heads, seq, head_dim]``
                  or ``None`` before any update
  * ``values``  : same
  * ``offset``  : ``int`` — number of tokens the model has seen for
                  RoPE positioning of the NEXT key
  * ``update_and_fetch(k, v)`` : append + return concatenated K, V

We trim the cache by direct attribute mutation on ``keys`` / ``values``
plus an explicit ``offset`` rewrite. mlx_lm's KVCache stores `keys` /
`values` as plain attributes (not read-only properties), so this
works in mlx-lm 0.31.x. If a future version makes them properties
without setters, the `_assign_kv` helper below will raise `AttributeError`
and we'll see that immediately on the Mac test pass — no silent
fallback.

Important RoPE invariant: after a sink+window trim, the cache holds
sink K/V (with RoPE rotated for global positions [0, sink-1]) plus
window K/V (rotated for [global - window, global - 1]). New queries
are RoPE-rotated for ``cache.offset``, which we leave equal to the
*global* sequence position (NOT the cache's physical length). This
matches StreamingLLM-style attention sinks and is the same behavior
our PyTorch ``SinkWindowVerifier`` enforces.

The module imports `mlx.core` at top level; non-arm64 hosts cannot
import this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx


@dataclass(frozen=True)
class TrimReport:
    """Diagnostic snapshot of one trim pass, for tests and logs."""

    layers_trimmed: int
    layers_skipped_null: int
    physical_size_before: int
    physical_size_after: int


def _kv_shape_seq(arr: "mx.array") -> int:
    """Return the seq-length axis of a [B, H, S, D] cache array."""
    if arr.ndim != 4:
        raise RuntimeError(
            f"KVCache.keys / .values is expected to be 4-D "
            f"[batch, n_kv_heads, seq, head_dim], got ndim={arr.ndim}, "
            f"shape={tuple(arr.shape)}"
        )
    return int(arr.shape[2])


def _assign_kv(layer_cache, new_keys: "mx.array", new_values: "mx.array") -> None:
    """Replace a layer's K/V in place.

    Direct attribute write. Raises if the underlying KVCache has made
    these read-only — which would be a real upstream API change we want
    surfaced (no try/except fallback).
    """
    layer_cache.keys = new_keys
    layer_cache.values = new_values


def trim_caches_to_sink_window(
    cache: List,
    *,
    sink_size: int,
    window_size: int,
    keep_offset: int,
) -> TrimReport:
    """Trim every layer's K/V to ``sink + window`` slots.

    Parameters
    ----------
    cache
        The list returned by ``mlx_lm.models.cache.make_prompt_cache``.
    sink_size
        Number of leading slots to retain (StreamingLLM "attention sinks").
    window_size
        Number of trailing slots to retain.
    keep_offset
        Value to write back to ``layer.offset`` after trimming. We pass
        in the *global* token count so RoPE for new queries is correct
        regardless of how many physical slots remain.

    The function mutates the cache list **in place**. After calling,
    every non-null layer's `keys.shape[2] == sink_size + window_size`
    (or smaller, if the layer hadn't accumulated enough yet).
    """
    if sink_size < 0 or window_size <= 0:
        raise ValueError("sink_size must be >= 0 and window_size must be > 0")
    budget = sink_size + window_size

    layers_trimmed = 0
    layers_skipped_null = 0
    pre_total = 0
    post_total = 0

    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            layers_skipped_null += 1
            continue
        seq_k = _kv_shape_seq(keys)
        seq_v = _kv_shape_seq(values)
        if seq_k != seq_v:
            raise RuntimeError(
                f"KVCache shape inconsistency: keys seq={seq_k} "
                f"vs values seq={seq_v}"
            )
        pre_total += seq_k
        if seq_k <= budget:
            post_total += seq_k
            continue

        sink_k = keys[:, :, :sink_size, :]
        sink_v = values[:, :, :sink_size, :]
        tail_k = keys[:, :, -window_size:, :]
        tail_v = values[:, :, -window_size:, :]
        new_keys = mx.concatenate([sink_k, tail_k], axis=2)
        new_values = mx.concatenate([sink_v, tail_v], axis=2)
        # Force evaluation so the slice / concat doesn't keep the
        # full original tensor alive through a graph reference (we want
        # the memory back).
        mx.eval(new_keys, new_values)
        _assign_kv(layer_cache, new_keys, new_values)
        layer_cache.offset = keep_offset
        post_total += budget
        layers_trimmed += 1

    return TrimReport(
        layers_trimmed=layers_trimmed,
        layers_skipped_null=layers_skipped_null,
        physical_size_before=pre_total,
        physical_size_after=post_total,
    )


def truncate_caches_tail(
    cache: List,
    *,
    drop: int,
    new_offset: int,
) -> int:
    """Drop the last ``drop`` slots from every layer (no sink-preservation).

    Used by the speculative loop after a forward whose tail was
    rejected. Returns the number of layers actually trimmed.

    ``new_offset`` is the post-truncation global token count (the value
    we want each layer's ``offset`` attribute to hold so RoPE is correct
    on the next forward).
    """
    if drop < 0:
        raise ValueError("drop must be >= 0")
    if drop == 0:
        return 0
    layers_trimmed = 0
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            continue
        seq_k = _kv_shape_seq(keys)
        if drop > seq_k:
            raise RuntimeError(
                f"truncate_caches_tail: requested drop={drop} but layer "
                f"only has {seq_k} slots"
            )
        keep = seq_k - drop
        new_keys = keys[:, :, :keep, :]
        new_values = values[:, :, :keep, :]
        mx.eval(new_keys, new_values)
        _assign_kv(layer_cache, new_keys, new_values)
        layer_cache.offset = new_offset
        layers_trimmed += 1
    return layers_trimmed


def total_kv_bytes(cache: List) -> int:
    """Sum K/V tensor bytes across all layers; matches the PyTorch
    `SinkWindowVerifier`'s `peak_kv_bytes` accounting."""
    total = 0
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is not None:
            total += keys.size * keys.dtype.size
        if values is not None:
            total += values.size * values.dtype.size
    return total


def cache_seq_length(cache: List) -> int:
    """Return the seq-length of the first non-null layer, or 0 if empty.

    All layers should have the same seq length after construction; if
    they don't (a real upstream bug), we surface it via the layout
    check in `_kv_shape_seq` rather than here.
    """
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        if keys is not None:
            return _kv_shape_seq(keys)
    return 0
