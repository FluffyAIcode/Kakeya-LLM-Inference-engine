"""End-to-end v0.4 dLM K/V Restoration verifier wrapper.

ADR 0008 §11.7 phase K1.D — ties together the K1.A capture, K1.B
merge, and K1.C per-attention-layer K/V preparation primitives into
a single :class:`DLMRestoredVerifier` callable that wraps an HF
Gemma3-class model and produces logits using the v0.4 architecture:

* The proposer (same model in K1, distinct dLM in K2/K3) runs a
  full-attention forward to capture K/V at every layer at every
  position. The captures are transient — discarded after the
  verifier forward.
* The verifier runs its standard forward, but at every attention
  layer the K/V tensors that go into ``attention_interface`` are
  *replaced at evicted positions* with the captured branch (after
  k_norm + RoPE applied for the captured position).
* Sustained memory: only the verifier's sink+window cache (≈ 3 MB
  per session). No proposer cache, no permanently-stored evicted
  K/V. This is the constant-memory property of ADR 0008 §11.5
  realised in code.

How the integration with HF Gemma3Attention works
-------------------------------------------------

Each layer's ``self_attn.forward`` is monkey-patched at restoration
time. The patched forward is a faithful copy of the upstream
``Gemma3Attention.forward`` (HF transformers ≥ 4.57) with one
insertion: between ``apply_rotary_pos_emb`` and
``attention_interface``, the post-RoPE K/V are passed through
:func:`prepare_restored_attention_kv` along with the layer's slice
of the captured K/V. The result replaces the local K/V at evicted
positions; non-evicted positions are unchanged.

This monkey-patch approach is chosen over subclassing
``Gemma3Attention`` because:

1. **Per-instance install/uninstall**. The patch is a context-manager
   pattern — entering :meth:`DLMRestoredVerifier.__call__` installs
   patches, exiting removes them. The base model is left unmodified
   for any other consumer.
2. **Cross-version friendliness within a major HF line**. The
   patched forward replicates the exact upstream signature and
   logic; if HF's forward signature changes within 4.x we get an
   immediate crash at install time, not silent corruption.
3. **No new modules in the model graph**. Saving / loading the
   underlying model is unaffected.

Limitations
-----------

* **No incremental KV cache**. K1.D runs every forward as if from
  scratch (``use_cache=False`` internally). The session-bound
  sink+window slab from ADR 0008 §2.3 is integrated in K-series
  Phase 2 — at that point the verifier's local K/V at sink+window
  positions come from the slab, and only evicted positions are
  reconstructed. This MVP keeps the focus on the load-bearing
  architectural mechanism.
* **Single-batch only**. The forward currently assumes
  ``input_ids.size(0) == 1``. Multi-batch routing through
  per-position-list selection requires care that's not in scope
  for K1.D's pipe-cleaning role.
* **Same-checkpoint proposer + verifier**. K2 introduces a separate
  proposer with a learned cross-model projection ``f_θ``; K1.D
  uses ``proposer = verifier`` for the identity case.

Tests
-----

Linux unit tests in ``tests/inference_engine/v04/test_dlm_restored_verifier.py``
exercise the wrapper on a synthetic Gemma3-shape surrogate that
ships with rotary embeddings, k_norm modules, and the same
attention forward signature as upstream HF. The synthetic surrogate
lets us validate the patch lifecycle, capture-merge-injection
pipeline, and the all-positions-evicted equivalence to
full-attention reference — all on Linux CI in <1 second with no HF
model download.

Real Gemma 3-1B-it integration smoke + NIAH validation lives on
the Mac M4 reviewer aid (``scripts/review_pr_k1d_on_mac.sh``) and
is not part of Linux CI.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from inference_engine.v04.kv_capture import (
    KVCapture,
    capture_proposer_kv,
)
from inference_engine.v04.kv_compressor import (
    IdentityCompressor,
    KVCompressor,
)
from inference_engine.v04.kv_merge import compute_evicted_positions
from inference_engine.v04.restored_attention import prepare_restored_attention_kv


# ---------------------------------------------------------------------------
# Per-step restoration context — what the patched forward needs to know
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SessionState:
    """Persistent per-session state for K2.A.2 stateful caching.

    Held on a :class:`DLMRestoredVerifier` instance across multiple
    ``forward()`` calls. Cleared by :meth:`DLMRestoredVerifier.reset_cache`
    to start a new session.

    Attributes
    ----------
    cache_token_count
        How many tokens of the prefix have already been processed and
        cached. ``forward(input_ids[1, T])`` is interpreted as: if
        T == 0, no-op; if T == cache_token_count, no-op (idempotent
        re-call of the previous forward); if T > cache_token_count,
        process tokens [cache_token_count, T) incrementally; if
        T < cache_token_count, raise (caller must
        ``reset_cache()`` first).
    compressors
        One :class:`KVCompressor` instance per attention layer. None
        before the first stateful forward; populated by
        ``_stateful_bootstrap`` and persisted by subsequent
        ``_stateful_incremental`` calls. Stores K/V at resident
        (sink + window) positions of the current prefix; positions
        that age out of the window are evicted (per §11.13.6.2 K1
        same-checkpoint property: cached K/V at resident positions
        in K2.A.2 stateful mode equals what K1.D / K2.A.1 stateless
        forward would compute fresh, modulo numerical noise).
    """

    cache_token_count: int = 0
    compressors: Optional[List[KVCompressor]] = None

    @classmethod
    def fresh(cls) -> "_SessionState":
        return cls(cache_token_count=0, compressors=None)


@dataclasses.dataclass
class _LayerRestorationContext:
    """The slice of restoration state needed by a single attention
    layer's patched forward. Attached to the attention module via a
    private attribute during a forward call and removed afterwards.

    Attributes
    ----------
    captured_K
        The proposer's K projection at evicted positions for *this
        layer*. Shape ``[B, len(evicted_positions), num_kv_heads,
        head_dim]``, pre-norm pre-RoPE per K1.A's contract.
    captured_V
        Same for V.
    evicted_positions
        Sorted-ascending list of token positions that should come
        from the captured branch. Shared across all layers (one
        list per forward).
    resident_positions
        Sorted-ascending list of token positions that come from
        the verifier's own forward (sink ∪ window slots). Disjoint
        from ``evicted_positions``; together they tile
        ``range(seq_len)``. Required by K2.A.1 to identify which
        positions go through the per-layer ``compressor``.
    compressor
        Per-layer :class:`KVCompressor` for K2.A.1 round-tripping
        the resident-window K/V through KakeyaLattice (or any
        other codec, including the no-op ``IdentityCompressor``
        which preserves K1 behaviour bit-for-bit).
    """

    captured_K: torch.Tensor
    captured_V: torch.Tensor
    evicted_positions: List[int]
    resident_positions: List[int] = dataclasses.field(default_factory=list)
    compressor: Optional[KVCompressor] = None


class _V04SessionCache:
    """K2.A.2 stateful K/V cache adapter implementing the HF
    ``Cache`` ``update()`` contract.

    HF Gemma3Attention.forward, when called with
    ``past_key_values=cache``, invokes
    ``cache.update(K_new, V_new, layer_idx, cache_kwargs)`` AFTER
    computing K, V for the current input via k_proj/v_proj/k_norm
    and applying RoPE. The return value is the (K, V) tensor used
    in the attention call.

    For K2.A.2 stateful incremental decode, the input is a small
    set of NEW tokens (typically 1 per decode step). The cache must
    return K, V at ALL preceding positions plus the new tokens, so
    the verifier's attention can attend across the full prefix.

    This cache assembles the returned K, V from three sources:

    1. **New tokens' K, V** — passed in directly to ``update``;
       already post-norm post-RoPE for the new positions.
    2. **Resident positions' K, V** — decompressed from per-layer
       :class:`KVCompressor` instances persisted on
       :class:`_SessionState`. These were stored in earlier
       forwards (or this forward's bootstrap path) and represent
       the verifier's own k_proj output for those positions,
       post-norm post-RoPE; under §11.13.6.2 (K1 same-checkpoint
       AR-causal proposer) these are bit-equivalent to what fresh
       computation would produce.
    3. **Evicted positions' K, V** — pre-computed before
       ``model.forward`` is called, by running
       :func:`prepare_restored_attention_kv` on the proposer's
       transient capture. Stored on this cache via
       :meth:`set_evicted_kv` per layer.

    After assembling and returning K, V, ``update`` ALSO writes the
    new K, V (for new positions that should be resident in the
    sink+window of the post-update prefix) into the per-layer
    compressor and evicts positions that age out of the window.

    The cache is single-batch only (B == 1); matches K1.D / K2.A.1
    constraint.
    """

    def __init__(
        self,
        compressors: List[KVCompressor],
        sink_size: int,
        window_size: int,
        cache_token_count_at_start: int,
        n_new_tokens: int,
    ) -> None:
        self._compressors = compressors
        self._sink_size = sink_size
        self._window_size = window_size
        # T_full = cache_token_count_at_start + n_new_tokens after this call.
        self._t_start = cache_token_count_at_start
        self._n_new = n_new_tokens
        self._t_full = cache_token_count_at_start + n_new_tokens
        # Per-layer pre-computed evicted K/V (post-norm post-RoPE).
        # Indexed [layer_idx] -> (K_evicted, V_evicted) where each tensor
        # has shape [B, num_kv_heads, len(evicted_positions), head_dim].
        self._evicted_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cached evicted positions list (set once per forward).
        self._evicted_positions: List[int] = []
        # Cached resident positions list (set once per forward).
        self._resident_positions: List[int] = []

    # -- Setup APIs called before model.forward --------------------------------

    def set_evicted_kv(
        self, layer_idx: int,
        K_evicted: torch.Tensor, V_evicted: torch.Tensor,
    ) -> None:
        """Store the layer's pre-computed evicted K/V (post-norm post-RoPE).

        Called by :meth:`DLMRestoredVerifier._stateful_incremental`
        for each layer before the verifier model.forward, so that
        when ``update`` fires per-layer it has the right evicted
        K/V to inject.
        """
        self._evicted_kv[layer_idx] = (K_evicted, V_evicted)

    def set_partition(
        self, evicted_positions: List[int], resident_positions: List[int],
    ) -> None:
        """Set evicted/resident position lists (same for all layers)."""
        self._evicted_positions = list(evicted_positions)
        self._resident_positions = list(resident_positions)

    # -- HF Cache contract: get_seq_length, update -----------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """How many tokens are in the cache. After the verifier
        forward consumes new tokens, the seq length is t_full."""
        return self._t_full

    def update(
        self,
        key_states: torch.Tensor,    # [B, kv_heads, n_new, head_dim]
        value_states: torch.Tensor,  # [B, kv_heads, n_new, head_dim]
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HF Cache.update contract.

        Called by HF Gemma3Attention.forward after it computes K, V
        for new tokens (post-k_proj, post-k_norm, post-RoPE).
        Returns assembled (K_full, V_full) at all positions
        ``[0..t_full)`` for the verifier's attention call.

        Side effect: stores new K, V at new resident positions in
        the per-layer compressor; evicts positions that age out.
        """
        if key_states.size(2) != self._n_new:
            raise RuntimeError(
                f"V04SessionCache.update at layer {layer_idx}: K_new has "
                f"{key_states.size(2)} positions but expected {self._n_new}"
            )

        compressor = self._compressors[layer_idx]
        device = key_states.device
        dtype = key_states.dtype

        # 1. Determine new positions and which of them should land in
        #    the resident cache after this update.
        new_positions = list(range(self._t_start, self._t_full))
        resident_set = set(self._resident_positions)

        new_resident_positions = [p for p in new_positions if p in resident_set]
        if new_resident_positions:
            # Slice K_new / V_new at the new-resident positions.
            new_resident_indices = torch.tensor(
                [p - self._t_start for p in new_resident_positions],
                dtype=torch.int64, device=device,
            )
            K_new_resident = key_states.index_select(-2, new_resident_indices)
            V_new_resident = value_states.index_select(-2, new_resident_indices)
            # Compress + store. Compressor's positions tensor must be on CPU
            # (per K2.A.1 KVCompressor protocol — IdentityCompressor uses
            # int keys; KakeyaLatticeCompressor also uses CPU position list).
            pos_cpu = torch.tensor(
                new_resident_positions, dtype=torch.int64,
            )
            compressor.evict(pos_cpu)  # idempotent overwrite safety
            compressor.compress(K_new_resident, V_new_resident, pos_cpu)

        # 2. Evict positions that were resident before this forward but
        #    are no longer resident at t_full. (For sliding-window aging.)
        # Positions that WERE in cache before this forward: those at
        # range(t_start) that landed in the previous step's resident set.
        # Easier: ask the compressor for ALL stored positions and evict
        # those NOT in the current resident_set.
        # We don't have a direct "list resident positions" API on
        # KVCompressor, so use the union of old-resident + new-resident
        # candidates and re-derive.
        # For now: compute aged-out as "positions in [0, t_start) that
        # were resident at the t_start prefix length but are not at t_full".
        # The sink positions are stable; the window slides.
        prev_window_start = max(self._sink_size, self._t_start - self._window_size)
        new_window_start = max(self._sink_size, self._t_full - self._window_size)
        if new_window_start > prev_window_start:
            aged_out = list(range(prev_window_start, new_window_start))
            if aged_out:
                aged_cpu = torch.tensor(aged_out, dtype=torch.int64)
                compressor.evict(aged_cpu)

        # 3. Assemble the full K, V tensor at all positions [0..t_full).
        #    Sources:
        #      * resident_set ∩ [0..t_start): from compressor (cached
        #        from earlier forwards or this forward's new_resident
        #        path)
        #      * resident_set ∩ [t_start..t_full): from compressor (just
        #        compressed via the new_resident path above)
        #      * evicted_set: from self._evicted_kv[layer_idx] (pre-
        #        computed post-norm post-RoPE)
        K_evicted, V_evicted = self._evicted_kv.get(
            layer_idx, (None, None),
        )
        if self._evicted_positions and (K_evicted is None or V_evicted is None):
            raise RuntimeError(
                f"V04SessionCache.update at layer {layer_idx}: evicted "
                f"positions {len(self._evicted_positions)} but no pre-"
                "computed evicted K/V was set; call set_evicted_kv before "
                "model.forward"
            )

        # Build a position->K/V dict for assembly.
        # Resident from compressor (decompress all in one call):
        if self._resident_positions:
            res_cpu = torch.tensor(
                self._resident_positions, dtype=torch.int64,
            )
            K_res, V_res = compressor.decompress(res_cpu)
            # K_res shape: [B, kv_heads, len(resident), head_dim]
            # Cast to working dtype if codec round-tripped to fp32 (per
            # K2.A.1 fix for bf16 mismatch).
            K_res = K_res.to(device=device, dtype=dtype)
            V_res = V_res.to(device=device, dtype=dtype)
        else:
            K_res = None
            V_res = None

        # Allocate full output tensor and scatter into positional slots.
        # Shape: [B, kv_heads, t_full, head_dim].
        B, kv_heads, _, head_dim = key_states.shape
        K_full = torch.empty(
            B, kv_heads, self._t_full, head_dim,
            dtype=dtype, device=device,
        )
        V_full = torch.empty_like(K_full)

        if self._resident_positions:
            res_idx = torch.tensor(
                self._resident_positions, dtype=torch.int64, device=device,
            )
            K_full.index_copy_(-2, res_idx, K_res)
            V_full.index_copy_(-2, res_idx, V_res)

        if self._evicted_positions:
            ev_idx = torch.tensor(
                self._evicted_positions, dtype=torch.int64, device=device,
            )
            K_full.index_copy_(
                -2, ev_idx,
                K_evicted.to(device=device, dtype=dtype),
            )
            V_full.index_copy_(
                -2, ev_idx,
                V_evicted.to(device=device, dtype=dtype),
            )

        return K_full, V_full


def _round_trip_resident_through_compressor(
    K: torch.Tensor,
    V: torch.Tensor,
    resident_positions: List[int],
    compressor: KVCompressor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round-trip the K/V slices at *resident* positions through the
    compressor.

    Per ADR 0008 §11.11.2, the codec applies ONLY to the resident
    local cache (sink ∪ window slots). The K/V slices at evicted
    positions are reconstructed from the proposer's transient K/V
    (`prepare_restored_attention_kv` in the patched attention
    forward) and are NOT compressed — they live one decode step
    and the compress→decompress round-trip would be pure overhead
    with no sustained-memory benefit.

    K2.A.1 stateless contract: the compressor's state is reset
    every call (via ``evict(positions)`` before compress). State
    accumulation across forward calls is K2.A.2's responsibility.

    Parameters
    ----------
    K, V
        Post-RoPE post-norm K/V tensors of shape
        ``[B, num_kv_heads, seq_len, head_dim]``. Evicted positions
        in K, V are already overwritten with reconstructed values.
    resident_positions
        Sorted ascending list of resident position indices. Disjoint
        from evicted positions; ``len(resident) + len(evicted) == seq_len``.
    compressor
        Per-layer :class:`KVCompressor` instance.

    Returns
    -------
    ``(K_out, V_out)`` with the same shape as the inputs. K/V at
    resident positions has passed through ``compressor.compress``
    + ``compressor.decompress``; K/V at evicted positions is
    untouched.
    """
    if not resident_positions:
        return K, V
    pos_tensor = torch.tensor(
        resident_positions, dtype=torch.int64, device=K.device,
    )
    # Slice resident K/V along the sequence dim (-2 in
    # [B, kv_heads, seq, head_dim] layout).
    K_resident = K.index_select(dim=-2, index=pos_tensor).contiguous()
    V_resident = V.index_select(dim=-2, index=pos_tensor).contiguous()
    # Stateless K2.A.1: clear any state from prior calls so the
    # round-trip reflects only this forward's K/V values.
    compressor.evict(pos_tensor.cpu())
    compressor.compress(K_resident, V_resident, pos_tensor.cpu())
    K_round_tripped, V_round_tripped = compressor.decompress(pos_tensor.cpu())
    # Reassemble: K/V at evicted positions are unchanged; K/V at
    # resident positions get the round-tripped values.
    #
    # KakeyaLattice's decompress returns fp32 on the lattice's compute
    # device (typically CPU); the verifier's K, V are typically
    # bf16/fp16 on MPS/CUDA. ``index_copy_`` requires self and source
    # to share dtype AND device. The K2.A.2 stateful path
    # (``_V04SessionCache.update``) handles this at line 380; the
    # K2.A.1 stateless path here was missing the cast — surfaced as
    # ``RuntimeError: index_copy_(): self and source expected to have
    # the same dtype, but got (self) BFloat16 and (source) Float`` on
    # the first Mac M4 production-smoke (2026-06-09). This cast
    # mirrors line 380 + 408 to close the regression.
    K_out = K.clone()
    V_out = V.clone()
    K_out.index_copy_(
        -2, pos_tensor,
        K_round_tripped.to(device=K_out.device, dtype=K_out.dtype),
    )
    V_out.index_copy_(
        -2, pos_tensor,
        V_round_tripped.to(device=V_out.device, dtype=V_out.dtype),
    )
    return K_out, V_out


# ---------------------------------------------------------------------------
# Patched attention forward — replicates Gemma3Attention.forward + merge
# ---------------------------------------------------------------------------


def _restored_attention_forward(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position=None,
    apply_rotary_pos_emb=None,
    eager_attention_forward=None,
    all_attention_functions=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """A faithful copy of Gemma3Attention.forward (HF 4.57+) with one
    insertion: between ``apply_rotary_pos_emb`` and
    ``attention_interface``, post-RoPE K/V are merged with captured
    proposer K/V at evicted positions.

    Imported function pointers (``apply_rotary_pos_emb``,
    ``eager_attention_forward``, ``all_attention_functions``) are
    passed in as parameters rather than imported at module load time
    so this module remains importable on systems without HF
    transformers (Linux CI does have it; runtime systems certainly
    do; this just keeps the import surface explicit).

    The ``_v04_layer_context`` private attribute on ``attn_module``
    provides the per-layer captured K/V; if it is ``None`` the
    function falls through to a verbatim copy of upstream forward
    (no v0.4 modification). This means partial activation is
    permitted: a verifier could run some layers with restoration
    and others without, though the K-series MVPs always activate
    all layers uniformly.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)

    query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = attn_module.q_norm(query_states)
    key_states = attn_module.k_norm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin,
    )

    # ▼▼▼ v0.4 K/V Restoration injection point ▼▼▼
    ctx: Optional[_LayerRestorationContext] = getattr(
        attn_module, "_v04_layer_context", None,
    )
    if ctx is not None and ctx.evicted_positions:
        # K1.D: replace K/V at evicted positions with proposer-restored values.
        key_states, value_states = prepare_restored_attention_kv(
            K_local=key_states,
            V_local=value_states,
            captured_K_pre_norm=ctx.captured_K,
            captured_V=ctx.captured_V,
            evicted_positions=ctx.evicted_positions,
            k_norm=attn_module.k_norm,
            position_embeddings=(cos, sin),
        )
    # K2.A.1: round-trip resident-window K/V through the per-layer
    # KV compressor. Applies regardless of whether evicted_positions
    # is empty — at short context (T <= sink+window) the entire
    # sequence is resident and the compressor still applies. The
    # IdentityCompressor (K1 default) round-trips bit-for-bit so
    # this is a no-op when KL is off.
    if ctx is not None and ctx.compressor is not None and ctx.resident_positions:
        key_states, value_states = _round_trip_resident_through_compressor(
            key_states, value_states, ctx.resident_positions, ctx.compressor,
        )
    # ▲▲▲ end v0.4 injection ▲▲▲

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, attn_module.layer_idx, cache_kwargs,
        )

    attention_interface = eager_attention_forward
    impl = attn_module.config._attn_implementation
    if impl != "eager" and all_attention_functions is not None:
        attention_interface = all_attention_functions[impl]

    attn_output, attn_weights = attention_interface(
        attn_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=attn_module.attention_dropout if attn_module.training else 0.0,
        scaling=attn_module.scaling,
        sliding_window=attn_module.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_module.o_proj(attn_output)
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------


class DLMRestoredVerifier:
    """Run an HF Gemma3-class model under the v0.4 K/V Restoration
    architecture (ADR 0008 §11).

    A single :meth:`forward` call:

    1. Runs the proposer model (same as the verifier in K1) to
       capture pre-norm pre-RoPE K and V at every layer at every
       position. Captures are transient.
    2. Computes the evicted-position list from
       ``(seq_len, sink_size, window_size)``.
    3. Slices the captures to evicted positions only.
    4. Installs a monkey-patched ``self_attn.forward`` on every
       decoder layer that calls
       :func:`prepare_restored_attention_kv` between RoPE and the
       attention_interface call.
    5. Runs the verifier model's forward.
    6. Removes the patches and discards the captures.

    Returned tensor: the verifier's logits, same shape as a standard
    HF ``model(input_ids=...).logits`` call.

    Parameters
    ----------
    model
        An HF Gemma3-class causal LM. Used for both the proposer and
        verifier roles in K1; K2 will introduce a separate proposer.
    sink_size
        Number of attention sinks at the head of the sequence
        (ADR 0001 + ADR 0008 §2.3 v0.3 default: 4).
    window_size
        Width of the trailing sliding window (default: 64). The
        union ``[0, sink) ∪ [seq_len - window, seq_len)`` is
        considered "kept"; everything else is "evicted" and routed
        through the proposer's captured K/V.

    Notes
    -----
    Currently single-batch only (``input_ids.size(0) == 1``). The
    monkey-patch is a context-manager — patches are removed in a
    ``finally`` block so even an exception during the verifier
    forward leaves the model unmodified.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        sink_size: int = 4,
        window_size: int = 64,
        kv_compressor_factory: Optional[Callable[[int], KVCompressor]] = None,
        stateful: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model
            HF Gemma3-class causal LM (proposer + verifier in K1).
        sink_size, window_size
            Local cache shape; ADR 0008 §2.3 v0.3 default 4 + 64.
        kv_compressor_factory
            K2.A.1 hook (added per ADR 0008 §11.11.4). A callable
            ``head_dim -> KVCompressor`` invoked once per attention
            module at the start of each forward. ``None`` (the
            default) preserves K1 behaviour bit-for-bit by using
            :class:`IdentityCompressor` (round-trip is exact).

            For K2.A KakeyaLattice integration::

                from inference_engine.v04 import KakeyaLatticeCompressor
                v04 = DLMRestoredVerifier(
                    model,
                    kv_compressor_factory=lambda hd: KakeyaLatticeCompressor(
                        head_dim=hd,
                        device=next(model.parameters()).device,
                        lattice="D4",
                        q_range=38,
                    ),
                )

            K2.A.1 (stateless): the factory is re-invoked at every
            ``forward()`` call (when ``stateful=False``, the default),
            so each forward constructs fresh compressor instances.
            K2.A.2 (stateful, when ``stateful=True``): the factory
            is invoked ONCE per session at the first forward; the
            same compressor instances persist across forwards until
            :meth:`reset_cache` is called.
        stateful
            K2.A.2 mode (added per ADR 0008 §11.11.12 + §11.13.6).
            Default ``False`` preserves K1.D / K2.A.1 stateless
            behaviour bit-for-bit. When ``True``:

            * Compressors are constructed once per session and
              persist across ``forward()`` calls; their state
              accumulates the K/V at resident (sink + window)
              positions of the current prefix.
            * Subsequent forwards process **only the new tokens**
              (the verifier's per-step forward becomes O(1) in T,
              not O(T)) — this is what closes the §11.8 throughput
              gate (c) target of ≥ 1.3× over K2.A.1 at long context.
            * Per §11.13.6.2, at K1 / K2.A same-checkpoint setup
              the cached K/V are bit-equivalent to fresh
              computation (the proposer is AR-causal, no suffix
              drift). At K2.B+ the §11.13.6.4 freshness escalation
              paths apply if recall regresses beyond gate (b).
            * :meth:`reset_cache` MUST be called between distinct
              prompts (sessions); otherwise the second prompt is
              interpreted as a continuation of the first.
        """
        if sink_size < 0 or window_size < 0:
            raise ValueError(
                f"sink_size={sink_size}, window_size={window_size} must "
                "both be non-negative"
            )
        self.model = model
        self.sink_size = sink_size
        self.window_size = window_size
        # K2.A.1 hook. None → IdentityCompressor (K1 backward compat).
        self._kv_compressor_factory = kv_compressor_factory or (
            lambda head_dim: IdentityCompressor()
        )
        # K2.A.2 hook.
        self._stateful = bool(stateful)
        self._session_state: _SessionState = _SessionState.fresh()

    # -- K2.A.2 session lifecycle --------------------------------------------

    @property
    def stateful(self) -> bool:
        """K2.A.2 stateful caching mode flag (read-only after construction)."""
        return self._stateful

    @property
    def cache_token_count(self) -> int:
        """How many tokens have been processed in the current session.
        Returns 0 when no session is active or when ``stateful`` is False."""
        return self._session_state.cache_token_count

    def reset_cache(self) -> None:
        """Clear all session state. Required between distinct prompts in
        ``stateful`` mode; no-op in stateless mode (kept for symmetry)."""
        self._session_state = _SessionState.fresh()

    # -- Discovery helpers ----------------------------------------------------

    def _decoder_layers(self) -> List[nn.Module]:
        """Return the list of decoder layers on the wrapped model.
        Supports HF Gemma3 / Llama / Qwen / Mistral shape
        (``model.model.layers``)."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise RuntimeError(
            "DLMRestoredVerifier could not locate decoder layers; "
            "expected model.model.layers (HF Gemma3 / Llama / Qwen / "
            "Mistral shape). Add a binding for the new architecture."
        )

    def _attention_modules(self) -> List[nn.Module]:
        """Return the list of self-attention modules, one per layer."""
        out = []
        for layer in self._decoder_layers():
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(
                    f"layer {type(layer).__name__} does not expose "
                    ".self_attn; cannot install K/V Restoration patch"
                )
            out.append(layer.self_attn)
        return out

    # -- Patch lifecycle ------------------------------------------------------

    @contextlib.contextmanager
    def _restoration_active(
        self,
        capture: KVCapture,
        evicted_positions: List[int],
        resident_positions: Optional[List[int]] = None,
        *,
        apply_rotary_pos_emb,
        eager_attention_forward,
        all_attention_functions,
    ):
        """Context manager that installs the v0.4 patched forward on
        every attention module, attaches the per-layer restoration
        context (including the per-layer K2.A.1 compressor instance),
        and removes everything on exit. Exception-safe.
        """
        attn_modules = self._attention_modules()
        if len(attn_modules) != capture.num_layers:
            raise RuntimeError(
                f"capture has {capture.num_layers} layers but model has "
                f"{len(attn_modules)} attention modules"
            )

        # Default resident_positions when called without (legacy
        # test sites that pre-date K2.A.1): empty list — the
        # patched forward then skips the round-trip step entirely
        # and behaves bit-for-bit like K1.D.
        if resident_positions is None:
            resident_positions = []

        # K2.A.1 / K2.A.2: build one compressor per attention module.
        # In K2.A.1 stateless mode, the factory is invoked every forward
        # → fresh instances. In K2.A.2 stateful mode, the persistent
        # session state holds compressors across forwards; this branch
        # creates them on first call and reuses them thereafter.
        if self._stateful and self._session_state.compressors is not None:
            compressors = self._session_state.compressors
            if len(compressors) != len(attn_modules):
                raise RuntimeError(
                    f"Stateful session has {len(compressors)} compressors but "
                    f"model has {len(attn_modules)} attention modules; "
                    "session may have been switched to a different model. "
                    "Call reset_cache() before changing models."
                )
        else:
            compressors = []
            for attn_module in attn_modules:
                head_dim = int(attn_module.head_dim)
                compressors.append(self._kv_compressor_factory(head_dim))
            if self._stateful:
                # First stateful forward: persist into session state.
                self._session_state.compressors = compressors

        original_forwards = []
        try:
            # 1. Slice capture to evicted-only and attach per-layer
            #    context to each attention module.
            if evicted_positions:
                evicted_capture = capture.select_positions(evicted_positions)
            else:
                # No evictions: pass empty captures; the patched
                # forward will short-circuit.
                evicted_capture = capture
            for layer_idx, attn_module in enumerate(attn_modules):
                if evicted_positions:
                    layer_K = evicted_capture.keys[layer_idx]
                    layer_V = evicted_capture.values[layer_idx]
                else:
                    # Empty placeholders; patched forward checks
                    # evicted_positions and skips merge if empty.
                    layer_K = torch.empty(
                        0, device=capture.keys[layer_idx].device,
                        dtype=capture.keys[layer_idx].dtype,
                    )
                    layer_V = torch.empty_like(layer_K)
                attn_module._v04_layer_context = _LayerRestorationContext(
                    captured_K=layer_K,
                    captured_V=layer_V,
                    evicted_positions=evicted_positions,
                    resident_positions=resident_positions,
                    compressor=compressors[layer_idx],
                )
                original_forwards.append(attn_module.forward)

                # 2. Install the patched forward.
                #    Bind the imported function pointers via closure.
                module_ref = attn_module

                def _make_patched(_module, _aprp, _eaf, _aaf):
                    def patched(
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values=None,
                        cache_position=None,
                        **kwargs,
                    ):
                        return _restored_attention_forward(
                            _module,
                            hidden_states,
                            position_embeddings,
                            attention_mask,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            apply_rotary_pos_emb=_aprp,
                            eager_attention_forward=_eaf,
                            all_attention_functions=_aaf,
                            **kwargs,
                        )
                    return patched

                attn_module.forward = _make_patched(
                    module_ref,
                    apply_rotary_pos_emb,
                    eager_attention_forward,
                    all_attention_functions,
                )

            yield
        finally:
            # 3. Restore original forwards and clear contexts.
            for attn_module, original in zip(attn_modules, original_forwards):
                attn_module.forward = original
                if hasattr(attn_module, "_v04_layer_context"):
                    delattr(attn_module, "_v04_layer_context")

    # -- Public API -----------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        apply_rotary_pos_emb,
        eager_attention_forward,
        all_attention_functions=None,
        rotary_emb_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Run the v0.4 K/V Restoration forward and return logits.

        The three function pointers from HF transformers'
        ``modeling_gemma3`` are passed in explicitly to keep this
        module HF-import-free at module load time and to make the
        binding visible at call sites for ease of mocking in unit
        tests:

            * ``apply_rotary_pos_emb`` —
              ``transformers.models.gemma3.modeling_gemma3.apply_rotary_pos_emb``.
            * ``eager_attention_forward`` —
              ``transformers.models.gemma3.modeling_gemma3.eager_attention_forward``.
            * ``all_attention_functions`` (optional) —
              ``transformers.models.gemma3.modeling_gemma3.ALL_ATTENTION_FUNCTIONS``
              dict, used when the model's ``_attn_implementation`` is
              not ``"eager"``. Pass ``None`` if ``"eager"`` is the
              only implementation in play (recommended for K1.D for
              determinism).

        Parameters
        ----------
        input_ids
            Token-id tensor ``[1, T]`` (single-batch only in K1.D).

        Returns
        -------
        Logits tensor of shape ``[1, T, vocab_size]`` matching the
        wrapped model's standard ``forward(input_ids).logits``
        contract.

        Raises
        ------
        ValueError
            On batch != 1 inputs (single-batch only in K1.D).
        RuntimeError
            On model-shape mismatch (decoder layers / attention
            modules cannot be located).
        """
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                "input_ids must have shape [1, T] (single-batch only "
                f"in K1.D); got {tuple(input_ids.shape)}"
            )
        seq_len = int(input_ids.size(1))

        # K2.A.2: route to stateful path if enabled. The bootstrap branch
        # (cache_token_count == 0) reuses the K1.D / K2.A.1 stateless code
        # path below, just persisting compressors at the end. The
        # incremental branch (cache_token_count > 0) processes only new
        # tokens via the V04SessionCache.
        if self._stateful and self._session_state.cache_token_count > 0:
            return self._stateful_incremental_forward(
                input_ids,
                apply_rotary_pos_emb=apply_rotary_pos_emb,
                eager_attention_forward=eager_attention_forward,
                all_attention_functions=all_attention_functions,
                rotary_emb_fn=rotary_emb_fn,
            )

        # Stateless path (K1.D / K2.A.1) and K2.A.2 bootstrap (first
        # forward of a stateful session). When stateful is True, the
        # _restoration_active context manager persists compressors into
        # self._session_state.compressors at first invocation.

        # Step 1: capture proposer's K/V at every layer at every position.
        capture = capture_proposer_kv(self.model, input_ids)

        # Step 2: compute evicted + resident positions (disjoint, tile [0, seq_len)).
        evicted = compute_evicted_positions(
            seq_len, self.sink_size, self.window_size,
        )
        evicted_set = set(evicted)
        resident = [p for p in range(seq_len) if p not in evicted_set]

        # Step 3-5: install patches, run verifier forward, remove patches.
        with self._restoration_active(
            capture,
            evicted,
            resident,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            eager_attention_forward=eager_attention_forward,
            all_attention_functions=all_attention_functions,
        ):
            outputs = self.model(input_ids=input_ids, use_cache=False)

        # Step 6 (K2.A.2 bootstrap): record cache_token_count for next
        # incremental call. The compressors are already persisted via
        # _restoration_active's stateful branch above.
        if self._stateful:
            self._session_state.cache_token_count = seq_len

        # Step 6: capture is dropped (out of scope after this function);
        #         original model state is restored by the context
        #         manager's finally block.
        return outputs.logits

    @torch.no_grad()
    def _stateful_incremental_forward(
        self,
        input_ids: torch.Tensor,
        *,
        apply_rotary_pos_emb,
        eager_attention_forward,
        all_attention_functions=None,
        rotary_emb_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """K2.A.2 incremental forward: process only new tokens.

        Called when stateful mode is on AND the session already has
        ``cache_token_count > 0``. Validates that input_ids extends
        the cached prefix, runs the proposer over the full prefix
        (still O(T) — no proposer cache by §11.3), runs the verifier
        only over new tokens via :class:`_V04SessionCache`, and
        updates session state.

        Per §11.13.6.2, at K1 / K2.A same-checkpoint setup the
        cached resident K/V at any preceding position equals what
        a fresh K1.D / K2.A.1 forward would compute (proposer is
        AR-causal, no suffix drift). So the output of incremental
        forward at a session prefix length T is bit-equivalent
        (modulo numerical noise from compressor codec) to running
        a fresh K1.D forward on input_ids[1, T].

        Returns the verifier's logits for the **full prefix**
        ``[1, T_full, vocab]`` — for compatibility with the
        stateless return shape, the cache_token_count slots are
        filled with zeros (the caller in the K1.E NIAH harness
        only uses the LAST query position's logits anyway, so the
        zero-fill at earlier positions is benign; see
        ``inference_engine.v04.greedy_decode_v04`` which does
        ``logits[:, -1, :]``).
        """
        T_full = int(input_ids.size(1))
        T_start = self._session_state.cache_token_count
        if T_full == T_start:
            raise ValueError(
                f"stateful incremental forward called with input_ids of "
                f"length {T_full} == cache_token_count; nothing new to "
                "process. Did you mean to extend input_ids?"
            )
        if T_full < T_start:
            raise ValueError(
                f"stateful input_ids length {T_full} is shorter than the "
                f"cached prefix length {T_start}. Call reset_cache() to "
                "start a new session."
            )

        n_new = T_full - T_start
        new_input_ids = input_ids[:, T_start:T_full]

        # Step 1: capture proposer's K/V at every layer at every position
        # of the FULL prefix (proposer has no cache by §11.3 — must
        # re-encode every step).
        capture = capture_proposer_kv(self.model, input_ids)

        # Step 2: compute evicted + resident over T_full.
        evicted_positions = compute_evicted_positions(
            T_full, self.sink_size, self.window_size,
        )
        evicted_set = set(evicted_positions)
        resident_positions = [p for p in range(T_full) if p not in evicted_set]

        # Step 3: build V04SessionCache and pre-compute evicted K/V per layer.
        if self._session_state.compressors is None:
            raise RuntimeError(
                "stateful incremental forward called but no compressors "
                "are persisted on session state; bootstrap forward "
                "(cache_token_count == 0) was not run first"
            )
        compressors = self._session_state.compressors

        cache = _V04SessionCache(
            compressors=compressors,
            sink_size=self.sink_size,
            window_size=self.window_size,
            cache_token_count_at_start=T_start,
            n_new_tokens=n_new,
        )
        cache.set_partition(evicted_positions, resident_positions)

        # For each layer, pre-compute the post-norm post-RoPE K/V at
        # evicted positions using the layer's k_norm + the standard
        # apply_rotary_pos_emb helper. The result goes into
        # cache.set_evicted_kv(layer_idx, K_evicted, V_evicted).
        attn_modules = self._attention_modules()
        if evicted_positions:
            evicted_capture = capture.select_positions(evicted_positions)
            # Get the position embeddings for ALL evicted positions.
            # We need cos/sin at those positions; obtain by calling the
            # model's rotary embedding for the full prefix.
            cos_full, sin_full = self._get_full_position_embeddings(
                input_ids, rotary_emb_fn=rotary_emb_fn,
            )
            ev_idx_dev = torch.tensor(
                evicted_positions, dtype=torch.int64, device=cos_full.device,
            )
            cos_evicted = cos_full.index_select(-2, ev_idx_dev)
            sin_evicted = sin_full.index_select(-2, ev_idx_dev)

            for layer_idx, attn_module in enumerate(attn_modules):
                # KVCapture's natural layout is [B, T, num_kv_heads, head_dim]
                # (see kv_capture.py line 478 / module docstring lines 102-103).
                # apply_rotary_pos_emb and the HF attention pipeline expect
                # [B, num_kv_heads, T, head_dim] — same layout K_local uses
                # after `k_proj(...).view(...).transpose(1, 2)` in standard
                # Gemma3Attention.forward. Transpose BEFORE k_norm + RoPE.
                #
                # Without this transpose, q.shape=[1, n_ev, kv, head_dim]
                # multiplied by cos.unsqueeze(1).shape=[1, 1, n_ev, head_dim]
                # broadcasts to [1, n_ev, n_ev, head_dim] — quadratic in
                # n_ev. At ctx280 (n_ev≈6345) this is ~20 GB and crashes
                # MPS with "Invalid buffer size: 19.20 GiB" (2026-06-09
                # Mac M4 production-smoke v3 failure). prepare_restored_
                # attention_kv (K2.A.1 stateless path) handles the same
                # transpose at its line ~370 (see its docstring lines
                # 257-259); the K2.A.2 stateful path was missing it.
                K_pre = evicted_capture.keys[layer_idx].transpose(1, 2).contiguous()
                V_pre = evicted_capture.values[layer_idx].transpose(1, 2).contiguous()
                # K_pre / V_pre now: [B, num_kv_heads, n_ev, head_dim]
                # Apply k_norm — last-dim normalisation, layout-invariant.
                K_normed = attn_module.k_norm(K_pre)
                # Apply RoPE on K (Q is irrelevant here).
                _, K_roped = apply_rotary_pos_emb(
                    K_normed, K_normed, cos_evicted, sin_evicted,
                )
                cache.set_evicted_kv(layer_idx, K_roped, V_pre)

        # Step 4: run the verifier over NEW tokens only with the cache.
        position_ids = torch.arange(
            T_start, T_full,
            device=input_ids.device, dtype=torch.int64,
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=new_input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        # Step 5: update session state (cache_token_count). The compressors
        # have been mutated in-place by V04SessionCache.update; no extra
        # work needed.
        self._session_state.cache_token_count = T_full

        # Step 6: assemble logits in the [1, T_full, vocab] shape expected
        # by the stateless return contract. The verifier's outputs.logits
        # has shape [1, n_new, vocab]; we zero-fill the [0..T_start) prefix
        # since stateless callers only use the LAST query's logits anyway.
        new_logits = outputs.logits  # [1, n_new, vocab]
        vocab = new_logits.size(-1)
        full_logits = torch.zeros(
            1, T_full, vocab,
            dtype=new_logits.dtype, device=new_logits.device,
        )
        full_logits[:, T_start:T_full, :] = new_logits
        return full_logits

    def _get_full_position_embeddings(
        self,
        input_ids: torch.Tensor,
        *,
        rotary_emb_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin position embeddings for the full prefix.

        Two resolution paths:

        1. **Caller-provided** ``rotary_emb_fn``: a callable
           ``(input_ids, position_ids) -> (cos, sin)``. Lets test
           harnesses inject a stub without needing a real HF model
           with a rotary embedding submodule.
        2. **Auto-discovery** on the wrapped model: looks for
           ``self.model.model.rotary_emb`` then ``self.model.rotary_emb``
           (HF Gemma3 / Llama / Qwen / Mistral typical paths).
           Calls it with a dummy hidden_state of shape
           ``[1, T, hidden_size]`` per HF's convention.

        Either way, the return is ``(cos, sin)`` of shape
        ``[1, T, head_dim]`` each.
        """
        T = int(input_ids.size(1))
        device = input_ids.device
        position_ids = torch.arange(
            T, device=device, dtype=torch.int64,
        ).unsqueeze(0)

        if rotary_emb_fn is not None:
            cos, sin = rotary_emb_fn(input_ids, position_ids)
            return cos, sin

        rotary_emb = None
        for path in ("model.rotary_emb", "rotary_emb"):
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                rotary_emb = obj
                break
            except AttributeError:
                continue
        if rotary_emb is None:
            raise RuntimeError(
                "Could not locate the model's rotary embedding module. "
                "Expected model.model.rotary_emb or model.rotary_emb. "
                "Pass rotary_emb_fn=... to forward() for non-HF models, "
                "or add a binding for the new architecture."
            )
        hidden_dim = self.model.config.hidden_size
        dummy = torch.zeros(
            1, T, hidden_dim,
            dtype=next(self.model.parameters()).dtype, device=device,
        )
        cos, sin = rotary_emb(dummy, position_ids)
        return cos, sin
