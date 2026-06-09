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
from typing import Callable, List, Optional, Sequence, Tuple

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
    K_out = K.clone()
    V_out = V.clone()
    # The compressor round-trip may upcast to fp32 (KakeyaLattice's
    # quantize/dequantize math runs in fp32 for numerical fidelity),
    # whereas the resident K/V cache is the model's compute dtype
    # (bf16 on CUDA). index_copy_ requires matching dtype (and device),
    # so cast the round-tripped tensors back before writing them in.
    K_out.index_copy_(
        -2, pos_tensor, K_round_tripped.to(device=K_out.device, dtype=K_out.dtype),
    )
    V_out.index_copy_(
        -2, pos_tensor, V_round_tripped.to(device=V_out.device, dtype=V_out.dtype),
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

            K2.A.1 (this build) is **stateless**: the factory is
            re-invoked at every ``forward()`` call, so each forward
            constructs fresh compressor instances. This guarantees
            no state leakage across decode steps but precludes the
            cross-step caching savings of K2.A.2 (a future PR).
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

        # K2.A.1: build one compressor per attention module via the
        # factory. This is stateless across forwards — fresh instances
        # every call. Reads head_dim from the layer's K projection.
        compressors = []
        for attn_module in attn_modules:
            head_dim = int(attn_module.head_dim)
            compressors.append(self._kv_compressor_factory(head_dim))

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

        # Step 6: capture is dropped (out of scope after this function);
        #         original model state is restored by the context
        #         manager's finally block.
        return outputs.logits
