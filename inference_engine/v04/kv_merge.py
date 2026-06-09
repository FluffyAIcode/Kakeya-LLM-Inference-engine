"""Merge verifier's locally-computed K/V with captured proposer K/V at
evicted positions.

ADR 0008 §11.5 — at every attention layer in the v0.4 verifier, the
attention input K/V is the union of:

* **K_local / V_local** — the verifier's own K/V projections of the
  positions still in its sink+window cache (computed normally during
  the verifier's forward pass).
* **K_captured / V_captured** — K/V at positions the verifier has
  evicted, reconstructed from the dLM proposer's parallel forward
  via the K1.A capture machinery.

This module implements that union as a pure tensor operation. It is
**RoPE-agnostic**: it merges raw projection outputs at any consistent
RoPE state (both pre-RoPE, or both post-RoPE — but never one of
each). The caller is responsible for applying RoPE consistently to
both branches before or after the merge.

For K1.B we merge **pre-RoPE** because that's the form K1.A captures
in (see ``inference_engine/v04/kv_capture.py`` module docstring §
"Why pre-RoPE rather than post-RoPE"). K1.C will apply RoPE inside
the verifier's standard attention forward, after the merge, using
HF's own ``apply_rotary_pos_emb`` so we don't duplicate that
machinery.

API contract
------------

The single public entry point :func:`merge_kv_at_evicted_positions`
takes ``[B, T, num_kv_heads, head_dim]`` local tensors and
``[B, len(evicted), num_kv_heads, head_dim]`` captured tensors plus
the list of evicted positions, and returns ``[B, T, num_kv_heads,
head_dim]`` merged tensors with K/V at evicted positions replaced
by the captured values.

Empty evicted list is the no-op identity (returns a clone of the
local tensors). Position lists are validated for sortedness, dedup,
and range; the captured-tensor T-dim must equal ``len(positions)``;
all shape/dtype/device must be consistent. Mismatches raise
``ValueError`` per ADR 0008 §6.2 (no silent fallback).

The returned merged tensors are clones of the inputs — the caller
can mutate them freely without affecting the input tensors. This
costs an extra allocation per layer per step, which is fine in the
v0.4 architecture where merge happens once per attention forward.
The clone is needed because ``index_copy_`` is in-place; if we
mutated K_local directly we would surprise callers who reuse it
elsewhere.

Differentiability
-----------------

Gradient flows through the captured branch (so a learnable cross-
model projection ``f_θ`` in K2/K3 can be trained end-to-end through
the merge). Gradient flows through the local branch only at
non-evicted positions; at evicted positions the local values are
overwritten and contribute no gradient. This is a deliberate
boundary condition matching the v0.4 architecture: at evicted
positions the verifier's local representation is irrelevant by
design.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def _validate_positions(
    positions: Sequence[int],
    seq_len: int,
) -> List[int]:
    """Validate and return the sorted-deduped list of positions.

    Raises ``ValueError`` on:
    * unsorted input
    * duplicates
    * any position < 0
    * any position >= ``seq_len``
    """
    if not positions:
        return []
    positions_list = list(positions)
    sorted_positions = sorted(set(positions_list))
    if sorted_positions != positions_list:
        raise ValueError(
            "evicted_positions must be sorted ascending with no "
            f"duplicates; got {positions_list}"
        )
    if sorted_positions[0] < 0 or sorted_positions[-1] >= seq_len:
        raise ValueError(
            f"evicted_positions must lie in [0, {seq_len}); "
            f"got [{sorted_positions[0]}, {sorted_positions[-1]}]"
        )
    return sorted_positions


def _validate_shapes(
    K_local: torch.Tensor,
    V_local: torch.Tensor,
    K_captured: torch.Tensor,
    V_captured: torch.Tensor,
    n_evicted: int,
) -> None:
    """Validate the four tensors share consistent batch / head / dim
    structure and that captured tensors' T-dim equals ``n_evicted``.

    Raises ``ValueError`` on any mismatch.
    """
    if K_local.shape != V_local.shape:
        raise ValueError(
            f"K_local shape {tuple(K_local.shape)} != V_local shape "
            f"{tuple(V_local.shape)}"
        )
    if K_captured.shape != V_captured.shape:
        raise ValueError(
            f"K_captured shape {tuple(K_captured.shape)} != V_captured "
            f"shape {tuple(V_captured.shape)}"
        )
    if K_local.dim() != 4:
        raise ValueError(
            "K_local must be 4-D [B, T, num_kv_heads, head_dim]; got "
            f"shape {tuple(K_local.shape)}"
        )
    if K_captured.dim() != 4:
        raise ValueError(
            "K_captured must be 4-D [B, n_evicted, num_kv_heads, "
            f"head_dim]; got shape {tuple(K_captured.shape)}"
        )

    B_local, T_local, H_local, D_local = K_local.shape
    B_cap, T_cap, H_cap, D_cap = K_captured.shape
    if B_local != B_cap:
        raise ValueError(
            f"batch mismatch: K_local B={B_local} K_captured B={B_cap}"
        )
    if H_local != H_cap:
        raise ValueError(
            f"num_kv_heads mismatch: K_local H={H_local} K_captured H={H_cap}"
        )
    if D_local != D_cap:
        raise ValueError(
            f"head_dim mismatch: K_local D={D_local} K_captured D={D_cap}"
        )
    if T_cap != n_evicted:
        raise ValueError(
            f"K_captured T-dim {T_cap} != len(evicted_positions) {n_evicted}"
        )

    if K_local.dtype != K_captured.dtype:
        raise ValueError(
            f"dtype mismatch: K_local {K_local.dtype} K_captured "
            f"{K_captured.dtype}"
        )
    if K_local.device != K_captured.device:
        raise ValueError(
            f"device mismatch: K_local {K_local.device} K_captured "
            f"{K_captured.device}"
        )


def merge_kv_at_evicted_positions(
    K_local: torch.Tensor,
    V_local: torch.Tensor,
    K_captured: torch.Tensor,
    V_captured: torch.Tensor,
    evicted_positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(K_merged, V_merged)`` where evicted positions are
    overridden by the captured proposer K/V and all other positions
    keep the verifier's local K/V.

    Parameters
    ----------
    K_local
        Verifier's K projection at every position. Shape
        ``[B, T, num_kv_heads, head_dim]``.
    V_local
        Verifier's V projection at every position. Same shape as
        ``K_local``.
    K_captured
        Proposer's K projection at the evicted positions only,
        already sliced via :meth:`KVCapture.select_positions`. Shape
        ``[B, len(evicted_positions), num_kv_heads, head_dim]``.
    V_captured
        Same as ``K_captured`` for V.
    evicted_positions
        Sorted-ascending list of positions in ``[0, T)`` whose K/V
        come from the captured branch. Empty list is the no-op
        identity case (returns clones of the local tensors). Per
        ADR 0008 §6.2, unsorted / duplicated / out-of-range inputs
        raise rather than silently coerce.

    Returns
    -------
    A tuple ``(K_merged, V_merged)`` of shape ``[B, T, num_kv_heads,
    head_dim]``. The returned tensors are clones — mutating them does
    not affect the inputs.

    Notes
    -----
    Both inputs and outputs are RoPE-agnostic; the caller must apply
    RoPE consistently to both branches before or after the merge.
    K1.B uses the merge in pre-RoPE; K1.C will apply RoPE inside the
    verifier's standard attention forward (after the merge), reusing
    HF's ``apply_rotary_pos_emb``.

    Gradient flows through ``K_captured`` / ``V_captured`` for the
    evicted positions; through ``K_local`` / ``V_local`` for the
    other positions. The local branch's gradient at evicted positions
    is severed by the override (those tensors are discarded by the
    merge). This is the intentional v0.4 boundary: at evicted
    positions, the verifier's local representation is irrelevant by
    design.
    """
    # Rank check first so K_local.size(1) is meaningfully the T dim.
    # We can't run the full shape validation yet because empty
    # evicted_positions is the no-op identity case where captured
    # tensors are allowed to be empty.
    if K_local.dim() != 4:
        raise ValueError(
            "K_local must be 4-D [B, T, num_kv_heads, head_dim]; got "
            f"shape {tuple(K_local.shape)}"
        )

    sorted_positions = _validate_positions(evicted_positions, K_local.size(1))

    if not sorted_positions:
        # No evictions: identity. Clone so callers can mutate freely.
        return K_local.clone(), V_local.clone()

    _validate_shapes(
        K_local, V_local, K_captured, V_captured, len(sorted_positions),
    )

    idx = torch.tensor(sorted_positions, device=K_local.device, dtype=torch.long)

    K_merged = K_local.clone()
    V_merged = V_local.clone()
    K_merged.index_copy_(dim=1, index=idx, source=K_captured)
    V_merged.index_copy_(dim=1, index=idx, source=V_captured)
    return K_merged, V_merged


def compute_evicted_positions(
    seq_len: int,
    sink_size: int,
    window_size: int,
) -> List[int]:
    """Return the list of token positions that fall **outside** the
    sink+window range over a sequence of length ``seq_len``.

    The v0.4 verifier's permanent KV cache holds K/V at the union
    ``{0, 1, ..., sink-1} ∪ {seq_len-window, ..., seq_len-1}``. All
    other positions are "evicted" — their K/V are reconstructed from
    the dLM proposer's transient forward each step. This helper
    materialises the evicted list once per generation step so callers
    can pass it to :meth:`KVCapture.select_positions` and to
    :func:`merge_kv_at_evicted_positions` without recomputing.

    Parameters
    ----------
    seq_len
        Total number of token positions in the current attention
        view (prompt + drafts so far).
    sink_size
        Number of attention sinks at the head of the sequence (ADR
        0001 + ADR 0008 §2.3 v0.3 default: 4).
    window_size
        Width of the trailing sliding window (ADR 0001 + ADR 0008
        §2.3 v0.3 default: 64).

    Returns
    -------
    Sorted-ascending list of evicted position indices. Empty when
    ``seq_len <= sink_size + window_size`` (everything fits in the
    cache, no evictions needed). Always contiguous: positions
    ``[sink_size, seq_len - window_size)``.

    Raises
    ------
    ValueError
        If any of ``seq_len``, ``sink_size``, ``window_size`` is
        negative.

    Notes
    -----
    Position ranges:

    * ``[0, sink_size)`` — sink (kept in cache, NOT evicted)
    * ``[sink_size, seq_len - window_size)`` — middle (EVICTED)
    * ``[seq_len - window_size, seq_len)`` — window (kept in cache)

    When sink and window overlap (``sink_size + window_size >=
    seq_len``), nothing is evicted. The function returns ``[]`` and
    the v0.4 architecture degenerates to standard full-attention
    inference at that step.
    """
    if seq_len < 0 or sink_size < 0 or window_size < 0:
        raise ValueError(
            f"seq_len={seq_len}, sink_size={sink_size}, "
            f"window_size={window_size} must all be non-negative"
        )
    if seq_len <= sink_size + window_size:
        # Sink + window covers the whole sequence; nothing to evict.
        return []
    return list(range(sink_size, seq_len - window_size))
