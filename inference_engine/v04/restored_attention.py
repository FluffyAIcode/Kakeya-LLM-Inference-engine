"""Per-attention-layer K/V preparation for v0.4 dLM K/V Restoration.

Given a Gemma3-style attention layer's locally-computed K and V
tensors at all positions (post-norm post-RoPE for K, raw for V) plus
a K/V capture from the dLM proposer at evicted positions (pre-norm
pre-RoPE per K1.A's contract), produce the merged K/V tensors that
the verifier's attention should consume as if the verifier had run
full attention over the entire prompt.

ADR 0008 §11.5 — the verifier attends to the union of (its own
sink+window K/V from cache) ⊕ (reconstructed K/V from proposer
transient). This module implements that "⊕" operation in the
post-RoPE shape that HF's attention internals expect.

The architectural contract
--------------------------

Gemma3Attention.forward computes K and V like this::

    key_states = self.k_proj(hidden_states).view(...).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(...).transpose(1, 2)
    # ↑ pre-norm, pre-RoPE, shape [B, num_kv_heads, T, head_dim]
    key_states = self.k_norm(key_states)              # post-norm
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(  # post-norm, post-RoPE
        query_states, key_states, cos, sin,
    )
    # ↑ key_states now in the form attention consumes

For the v0.4 architecture, **at evicted positions** we want
key_states / value_states to come from the proposer (whose forward
already saw full attention) instead of from the verifier (whose
local hidden state was computed under the bounded sink+window mask).

The proposer's K/V are captured in K1.A *before* k_norm and
*before* RoPE — that's the most stable hook point on the Gemma3
attention forward (k_proj is a clean nn.Linear, post-RoPE is buried
inside the forward). So this module is responsible for:

1. Apply ``k_norm`` to the captured K (V doesn't go through k_norm).
2. Apply RoPE to the captured K *with the cos/sin slices for the
   captured K's own positions* (not the query positions).
3. Merge the K/V into the verifier's all-position K/V at the
   evicted positions.

The merged K and V are returned in the post-RoPE attention shape
``[B, num_kv_heads, T, head_dim]`` ready for HF's attention_interface
to consume. K1.D's verifier integration patches Gemma3Attention.forward
to call this module right before ``attention_interface(...)``.

Why apply RoPE here and not at K1.A capture time
------------------------------------------------

* RoPE is position-dependent. The captured K at proposer-position p
  must have RoPE for position p applied. That means we need access
  to the cos/sin tables that the verifier's attention layer is
  using at that step. K1.A runs as part of a *different* model
  forward (the proposer's), where the verifier's per-step
  position_embeddings are not yet known.

* In K2 / K3 (cross-model), a learned projection ``f_θ`` will sit
  between captured K (proposer space) and merged K (verifier space).
  That projection runs more naturally in the pre-RoPE space because
  RoPE is verifier-specific. Applying RoPE *after* ``f_θ`` keeps the
  projection's training surface clean.

* In same-model identity case (K1), the captured K passed through
  the verifier's k_norm and apply_rotary_pos_emb is **bit-exact**
  what the verifier's own forward would have produced at that
  position under full attention. So the same-model identity gate
  in K1.D becomes "K_local at sink+window matches verifier-full at
  those positions, and reconstructed K matches verifier-full at
  evicted" — both are bit-exact, and the merged K is exactly the
  verifier-with-full-attention's K. ADR 0008 §11.5 §"Five properties
  v0.4 architecture realises" item 2 ("intelligence approximates
  full attention") becomes a constructive equality in this case.

Linux-side unit tests use a synthetic ``k_norm`` (an
nn.RMSNorm-shaped module) and synthetic cos/sin tables. The RoPE
implementation is the standard interleaved-half formulation that
HF transformers also uses (and we cross-check against HF's
``apply_rotary_pos_emb`` import-path-conditionally so the test
behaviour stays identical to production).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from inference_engine.v04.kv_merge import merge_kv_at_evicted_positions


# ---------------------------------------------------------------------------
# RoPE primitives — standard interleaved-half formulation, kept local so
# Linux unit tests do not depend on HF transformers' RoPE function. The
# math is identical; we cross-check in the integration layer.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE half-rotation: split last dim in halves, swap with
    sign flip on the second half. Matches HF transformers' ``rotate_half``.

    For input x of shape ``[..., 2D]`` returns ``[-x_2, x_1]`` where
    x_1, x_2 are the two halves of the last dimension.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_k_at_positions(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to K-only at the positions implied by ``cos`` / ``sin``.

    ``k`` shape: ``[B, num_heads, T, head_dim]`` — the standard HF
    attention K shape after ``.transpose(1, 2)``.

    ``cos`` / ``sin`` shape: ``[B, T, head_dim]`` — same convention as
    HF's ``apply_rotary_pos_emb``. They are unsqueezed to
    ``[B, 1, T, head_dim]`` before the multiply so the head dimension
    broadcasts.

    Note: HF's ``apply_rotary_pos_emb`` rotates Q and K together. We
    only rotate K here because the v0.4 K/V Restoration injects K/V
    at evicted positions while Q is the verifier's standard query
    (already computed by the verifier's attention forward). This
    function returns just the rotated K.

    Returns
    -------
    The RoPE-applied K of the same shape as the input.

    Notes
    -----
    The math is bit-identical to the HF formulation::

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        k_embed = (k * cos) + (rotate_half(k) * sin)

    Tests in ``tests/inference_engine/v04/test_restored_attention.py``
    exercise this against a hand-rolled scalar reference for ``T=2,
    head_dim=4``, plus a deterministic synthetic ``cos/sin`` to verify
    multi-position multi-head correctness.
    """
    if k.dim() != 4:
        raise ValueError(
            "k must be 4-D [B, num_heads, T, head_dim]; got shape "
            f"{tuple(k.shape)}"
        )
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError(
            "cos and sin must be 3-D [B, T, head_dim]; got shapes "
            f"cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos shape {tuple(cos.shape)} != sin shape {tuple(sin.shape)}"
        )
    # RoPE cos/sin are position-dependent but batch-independent, so a batch-1
    # table broadcasts across B>1 (multi-tenant batched restore). Accept either
    # cos.shape[0] == k.shape[0] or cos.shape[0] == 1.
    if (cos.shape[0] not in (1, k.shape[0])
            or cos.shape[1] != k.shape[2] or cos.shape[2] != k.shape[3]):
        raise ValueError(
            f"cos shape {tuple(cos.shape)} incompatible with k shape "
            f"{tuple(k.shape)}: expected [B in (1,{k.shape[0]}), T={k.shape[2]}, "
            f"head_dim={k.shape[3]}]"
        )
    cos_b = cos.unsqueeze(1)  # [B, 1, T, head_dim]
    sin_b = sin.unsqueeze(1)
    return (k * cos_b) + (_rotate_half(k) * sin_b)


def slice_position_embeddings(
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice the (cos, sin) RoPE tables at a given list of token positions.

    ``cos`` / ``sin`` shape: ``[B, T, head_dim]``.
    ``positions``: sorted-ascending list of integer indices in ``[0, T)``.

    Returns ``(cos_slice, sin_slice)`` of shape
    ``[B, len(positions), head_dim]``.

    Used to compute the cos/sin for the captured K's *own* positions
    (proposer position p), separate from the verifier's query
    position. Validates the position list per the same contract as
    :func:`merge_kv_at_evicted_positions` (sorted, deduped, in range).
    """
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError(
            f"cos / sin must be 3-D [B, T, head_dim]; got cos="
            f"{tuple(cos.shape)} sin={tuple(sin.shape)}"
        )
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos shape {tuple(cos.shape)} != sin shape {tuple(sin.shape)}"
        )
    seq_len = cos.shape[1]
    positions_list = list(positions)
    sorted_positions = sorted(set(positions_list))
    if sorted_positions != positions_list:
        raise ValueError(
            "positions must be sorted ascending with no duplicates; "
            f"got {positions_list}"
        )
    if not sorted_positions:
        raise ValueError("positions must be non-empty")
    if sorted_positions[0] < 0 or sorted_positions[-1] >= seq_len:
        raise ValueError(
            f"positions must lie in [0, {seq_len}); got "
            f"[{sorted_positions[0]}, {sorted_positions[-1]}]"
        )
    idx = torch.tensor(sorted_positions, device=cos.device, dtype=torch.long)
    return cos.index_select(dim=1, index=idx), sin.index_select(dim=1, index=idx)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_restored_attention_kv(
    *,
    K_local: torch.Tensor,
    V_local: torch.Tensor,
    captured_K_pre_norm: torch.Tensor,
    captured_V: torch.Tensor,
    evicted_positions: Sequence[int],
    k_norm: nn.Module,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce the merged ``(K, V)`` an attention layer should consume
    when running the v0.4 verifier with K/V Restoration enabled.

    Parameters
    ----------
    K_local
        Verifier's locally-computed K tensor at all positions, **after**
        ``k_norm`` and **after** RoPE — i.e., the tensor that
        ``Gemma3Attention.forward`` would normally pass into
        ``attention_interface``. Shape ``[B, num_kv_heads, T, head_dim]``.
    V_local
        Verifier's locally-computed V tensor (no norm, no RoPE — V
        does not go through either in standard transformer attention).
        Shape ``[B, num_kv_heads, T, head_dim]``.
    captured_K_pre_norm
        Proposer's K at the evicted positions only, **pre-norm and
        pre-RoPE**, as captured by K1.A
        (``KVCapture.select_positions(evicted_positions)``). Shape
        ``[B, len(evicted_positions), num_kv_heads, head_dim]`` — note
        the layout difference from ``K_local`` (T comes before
        num_kv_heads). This module handles the transpose internally.
    captured_V
        Proposer's V at evicted positions, same shape as
        ``captured_K_pre_norm``.
    evicted_positions
        Sorted-ascending list of positions in ``[0, T)`` whose K/V
        should come from the captured (proposer) branch instead of
        the verifier's local branch.
    k_norm
        The verifier attention layer's ``k_norm`` module
        (``Gemma3RMSNorm`` in HF Gemma3). Applied to ``captured_K_pre_norm``
        in the *same way* the verifier's normal forward applies it
        to ``key_states`` after ``k_proj`` — i.e., on the
        ``[..., head_dim]`` last dimension.
    position_embeddings
        ``(cos, sin)`` tuple from the verifier attention layer's
        forward. Shapes ``[B, T, head_dim]`` each. The captured K is
        rotated using the slices at ``evicted_positions``.

    Returns
    -------
    A tuple ``(K_merged, V_merged)`` of shape ``[B, num_kv_heads, T,
    head_dim]``, ready to pass into HF's attention_interface as the
    K and V the verifier should consume.

    Behaviour at the ``evicted_positions`` slots:
        * ``K_merged[..., p, :]`` comes from
          ``apply_rope( k_norm(captured_K_pre_norm[..., p, :]) )``.
        * ``V_merged[..., p, :]`` comes from
          ``captured_V[..., p, :]`` directly (V does not get norm or
          RoPE in standard transformer attention).

    At all other positions K_merged / V_merged equal K_local / V_local.

    Raises
    ------
    ValueError
        If shapes / dtypes / devices are inconsistent. ADR 0008 §6.2
        forbids silent fallback.

    Same-model identity case (K1)
    ------------------------------
    When the proposer and verifier are the same checkpoint and the
    proposer ran with the same RoPE / k_norm modules:
        ``apply_rope( k_norm(captured_K_pre_norm) )`` at position p
        is **bit-identical** to what the verifier-with-full-attention
        would have computed for K at position p. Cross-checked in
        the K1.D Mac M4 reviewer.

    Cross-model case (K2 / K3)
    --------------------------
    When the proposer is a different model, ``captured_K_pre_norm``
    is fed through a learned projection ``f_θ`` *before* being
    passed to this function. The k_norm / RoPE applied here are the
    **verifier's** norm and RoPE — the captured K is being
    transformed into the verifier's K-space.
    """
    # Empty evicted: identity (no merge needed). We still validate
    # K_local / V_local rank to surface contract violations early.
    if K_local.dim() != 4:
        raise ValueError(
            f"K_local must be 4-D [B, num_kv_heads, T, head_dim]; got "
            f"shape {tuple(K_local.shape)}"
        )
    if V_local.dim() != 4:
        raise ValueError(
            f"V_local must be 4-D [B, num_kv_heads, T, head_dim]; got "
            f"shape {tuple(V_local.shape)}"
        )
    if K_local.shape != V_local.shape:
        raise ValueError(
            f"K_local shape {tuple(K_local.shape)} != V_local shape "
            f"{tuple(V_local.shape)}"
        )
    if not evicted_positions:
        return K_local.clone(), V_local.clone()

    if captured_K_pre_norm.dim() != 4:
        raise ValueError(
            "captured_K_pre_norm must be 4-D [B, n_evicted, "
            "num_kv_heads, head_dim]; got shape "
            f"{tuple(captured_K_pre_norm.shape)}"
        )
    if captured_V.dim() != 4:
        raise ValueError(
            "captured_V must be 4-D [B, n_evicted, num_kv_heads, "
            f"head_dim]; got shape {tuple(captured_V.shape)}"
        )

    # ------------------------------------------------------------------
    # Step 1: apply k_norm to captured K (V does not get norm).
    # k_norm operates on the last dim (head_dim), which is consistent
    # across both [B, n_evicted, num_kv_heads, head_dim] and
    # [B, num_kv_heads, n_evicted, head_dim] layouts. We apply it on
    # the captured layout for clarity.
    # ------------------------------------------------------------------
    captured_K_post_norm = k_norm(captured_K_pre_norm)

    # ------------------------------------------------------------------
    # Step 2: apply RoPE to captured K at the evicted positions. RoPE
    # expects the [B, num_heads, T, head_dim] layout, so transpose
    # captured_K from [B, n_evicted, num_kv_heads, head_dim] to
    # [B, num_kv_heads, n_evicted, head_dim] for the operation, then
    # transpose back so the merge can re-use the same layout K1.B
    # already validates.
    # ------------------------------------------------------------------
    cos, sin = position_embeddings
    cos_at_evicted, sin_at_evicted = slice_position_embeddings(
        cos, sin, evicted_positions,
    )
    captured_K_for_rope = captured_K_post_norm.transpose(1, 2)
    captured_K_post_rope_attn_layout = apply_rope_to_k_at_positions(
        captured_K_for_rope, cos_at_evicted, sin_at_evicted,
    )
    # Back to merge-friendly layout [B, T_evicted, num_kv_heads, head_dim].
    captured_K_post_rope = captured_K_post_rope_attn_layout.transpose(1, 2)

    # ------------------------------------------------------------------
    # Step 3: merge into the verifier's K_local / V_local. K1.B's
    # ``merge_kv_at_evicted_positions`` expects [B, T, num_kv_heads,
    # head_dim] layout; transpose K_local / V_local in and back.
    # ------------------------------------------------------------------
    K_local_merge_layout = K_local.transpose(1, 2)
    V_local_merge_layout = V_local.transpose(1, 2)
    K_merged_merge_layout, V_merged_merge_layout = merge_kv_at_evicted_positions(
        K_local_merge_layout,
        V_local_merge_layout,
        captured_K_post_rope,
        captured_V,
        evicted_positions,
    )
    K_merged = K_merged_merge_layout.transpose(1, 2)
    V_merged = V_merged_merge_layout.transpose(1, 2)
    return K_merged, V_merged
