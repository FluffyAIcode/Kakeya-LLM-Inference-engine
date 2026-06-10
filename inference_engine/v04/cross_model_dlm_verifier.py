"""K3 Block B — Cross-model `DLMRestoredVerifier`.

The integrated Kakeya inference architecture per ADR 0008 §11.3:

  verifier (Gemma 4 26B-A4B):
      ├─ holds only sink+window local KV cache
      └─ at evicted positions, takes K/V supplied by the proposer
         (via `f_θ` projection) — verifier attends over full context
         despite holding O(sink+window) memory

  drafter (DFlash 0.4B, K3 alignment-trained baseline):
      ├─ runs full forward over committed prefix per step (no cache)
      ├─ K/V at every layer at every position captured via
      │   `inference_engine.v04.capture_proposer_kv`
      └─ K/V projected through `f_θ` into verifier K/V space, injected
         at evicted positions

This module implements the cross-model integration. The same-checkpoint
`DLMRestoredVerifier` (`inference_engine.v04.dlm_restored_verifier.
DLMRestoredVerifier`) covers the K1 / K2.A path; this module covers K3.

Differences from same-checkpoint DLMRestoredVerifier
-----------------------------------------------------

1. **Drafter ≠ verifier**: drafter is a separate model object
   (a `DFlashDrafter` or any `nn.Module` whose attention layers
   expose `k_proj` / `v_proj`). Same-checkpoint version assumes
   drafter is the verifier itself.

2. **`f_θ` projection mediates**: drafter K/V dim ≠ verifier K/V dim
   in cross-model setup. `f_θ` projects drafter K/V into verifier
   K/V space at every (layer, position) before injection.

3. **Layer-count mismatch handled**: drafter typically has fewer
   layers than verifier (DFlash 5 vs Gemma 4 26B-A4B 30). `f_θ`
   handles the projection from `drafter_num_layers`-concat input
   to `verifier_num_layers` outputs.

4. **K/V are pre-norm pre-RoPE on capture**: same as same-checkpoint
   path. The verifier's attention forward is patched to call
   `prepare_restored_attention_kv` which applies `k_norm` + RoPE
   to the projected K/V at evicted positions — matching the
   standard verifier's own K/V transformation pipeline.

What this module does NOT do (deliberately, scope-out)
------------------------------------------------------

* **MLX verifier path**: this module patches HF transformers
  attention modules. Mac MLX integration requires a separate
  approach (instrument mlx_lm Gemma 4 model directly). Tracked as
  follow-up PR after CUDA evidence.

* **Speculative decoding accept/reject loop**: that's a higher-level
  inference engine concern. This module produces a verifier with
  K/V Restoration; the spec decode loop wraps it. PR #93's
  `DFlashProposer` + `mlx_verify_block` is the spec decode side;
  combining with this module's K/V Restoration is a separate
  integration.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from inference_engine.v04.f_theta import FThetaProjection
from inference_engine.v04.kv_capture import KVCapture, capture_proposer_kv
from inference_engine.v04.kv_merge import compute_evicted_positions
from inference_engine.v04.restored_attention import prepare_restored_attention_kv


@dataclasses.dataclass
class CrossModelLayerMapping:
    """How drafter K/V layers project to verifier K/V layers under f_θ.

    f_θ takes ALL drafter layers' K/V (concat) per position and
    outputs ALL verifier layers' K/V per position. So the layer
    mapping is fixed by the f_θ architecture; this dataclass is
    informational only — it records which drafter / verifier layer
    counts the f_θ was trained against, so we can validate at
    construction time.
    """
    drafter_num_layers: int
    verifier_num_layers: int


class CrossModelDLMRestoredVerifier:
    """K3 cross-model verifier wrapper with f_θ-mediated K/V Restoration.

    Construction
    ------------

    >>> verifier = CrossModelDLMRestoredVerifier(
    ...     verifier_model=hf_gemma_4,         # transformers Gemma4ForCausalLM
    ...     drafter=dflash_drafter,            # DFlashDrafter from PR #93
    ...     f_theta=FThetaProjection(...),     # trained f_θ
    ...     sink_size=4,
    ...     window_size=64,
    ... )

    Forward
    -------

    >>> output = verifier.forward(
    ...     input_ids=...,
    ...     apply_rotary_pos_emb=...,    # transformers Gemma 4 RoPE helper
    ...     eager_attention_forward=..., # transformers Gemma 4 eager attn
    ... )

    Each forward:
      1. Drafter runs full forward over input_ids → KVCapture (per
         drafter layer, per position, pre-norm pre-RoPE).
      2. f_θ projects drafter K/V to verifier K/V at every (verifier
         layer, position).
      3. Verifier attention modules patched: at every layer, at every
         evicted position, the attention takes injected K/V from f_θ
         output (via prepare_restored_attention_kv to apply k_norm +
         RoPE) instead of computing K/V from the verifier's local
         hidden state.
      4. Verifier sink+window cache holds only resident K/V; evicted
         K/V come from f_θ each forward (transient, no memory cost).
    """

    def __init__(
        self,
        *,
        verifier_model: nn.Module,
        drafter: Any,                   # DFlashDrafter or any nn.Module with .model
        f_theta: FThetaProjection,
        sink_size: int = 4,
        window_size: int = 64,
    ) -> None:
        if sink_size < 0 or window_size < 0:
            raise ValueError("sink_size and window_size must be non-negative")
        self.verifier_model = verifier_model
        self.drafter = drafter
        self.f_theta = f_theta
        self.sink_size = sink_size
        self.window_size = window_size
        self._validate_dimensions()

    # -----------------------------------------------------------------
    # Dimension validation at construction time
    # -----------------------------------------------------------------

    def _validate_dimensions(self) -> None:
        cfg = self.f_theta.config
        # Verifier dimensions
        v_cfg = self.verifier_model.config
        v_layers = getattr(v_cfg, "num_hidden_layers", None)
        v_kv_heads = getattr(v_cfg, "num_key_value_heads", None)
        v_head_dim = getattr(v_cfg, "head_dim", None)
        if v_head_dim is None:
            hidden = getattr(v_cfg, "hidden_size", None)
            num_q_heads = getattr(v_cfg, "num_attention_heads", None)
            if hidden is not None and num_q_heads:
                v_head_dim = hidden // num_q_heads

        if v_layers is not None and v_layers != cfg.verifier_num_layers:
            raise ValueError(
                f"f_θ trained for verifier_num_layers={cfg.verifier_num_layers} "
                f"but verifier has {v_layers} layers"
            )
        if v_kv_heads is not None and v_kv_heads != cfg.verifier_num_kv_heads:
            raise ValueError(
                f"f_θ trained for verifier_num_kv_heads={cfg.verifier_num_kv_heads} "
                f"but verifier has {v_kv_heads}"
            )
        if v_head_dim is not None and v_head_dim != cfg.verifier_head_dim:
            raise ValueError(
                f"f_θ trained for verifier_head_dim={cfg.verifier_head_dim} "
                f"but verifier has {v_head_dim}"
            )

        # Drafter dimensions
        drafter_cfg = getattr(self.drafter, "cfg", None) or getattr(self.drafter, "config", None)
        if drafter_cfg is None:
            return  # cannot validate; trust the caller
        d_layers = getattr(drafter_cfg, "num_hidden_layers", None)
        d_kv_heads = getattr(drafter_cfg, "num_key_value_heads", None)
        d_head_dim = getattr(drafter_cfg, "head_dim", None)

        if d_layers is not None and d_layers != cfg.drafter_num_layers:
            raise ValueError(
                f"f_θ trained for drafter_num_layers={cfg.drafter_num_layers} "
                f"but drafter has {d_layers}"
            )
        if d_kv_heads is not None and d_kv_heads != cfg.drafter_num_kv_heads:
            raise ValueError(
                f"f_θ trained for drafter_num_kv_heads={cfg.drafter_num_kv_heads} "
                f"but drafter has {d_kv_heads}"
            )
        if d_head_dim is not None and d_head_dim != cfg.drafter_head_dim:
            raise ValueError(
                f"f_θ trained for drafter_head_dim={cfg.drafter_head_dim} "
                f"but drafter has {d_head_dim}"
            )

    # -----------------------------------------------------------------
    # Drafter capture + f_θ projection
    # -----------------------------------------------------------------

    @torch.no_grad()
    def project_drafter_kv(
        self, input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the drafter forward over input_ids, project K/V through f_θ.

        Returns
        -------
        (verifier_k, verifier_v) tensors of shape
        ``[B, T, verifier_num_layers, verifier_num_kv_heads, verifier_head_dim]``
        on the f_θ device.

        These are the per-position-per-verifier-layer K/V that the
        cross-model verifier injects at evicted positions during its
        attention forward.
        """
        capture = _capture_drafter_kv(
            verifier_model=self.verifier_model,
            drafter=self.drafter,
            input_ids=input_ids,
        )
        # capture.keys[i] shape: [B, T, num_d_kv_heads, head_dim]
        verifier_k, verifier_v = self.f_theta.forward_kv_pack(
            capture.keys, capture.values,
        )
        return verifier_k, verifier_v

    # -----------------------------------------------------------------
    # Forward (with K/V Restoration)
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        apply_rotary_pos_emb: Callable,
        eager_attention_forward: Callable,
        all_attention_functions: Optional[Any] = None,
    ):
        """Run a verifier forward with f_θ-mediated K/V Restoration.

        Steps:
          1. Compute evicted positions from sink+window per ADR §11.7.
          2. Drafter forward + f_θ projection → verifier K/V at every
             evicted position at every verifier layer.
          3. Patch verifier attention: at evicted positions, K/V come
             from the f_θ output (via prepare_restored_attention_kv);
             at resident positions, K/V come from the verifier's own
             k_proj / v_proj on its hidden state.
          4. Run verifier forward; collect logits.
          5. Restore original attention forwards.

        Returns the verifier's output (typically with .logits).
        """
        T = int(input_ids.size(1))
        evicted_positions = compute_evicted_positions(
            T, self.sink_size, self.window_size,
        )

        # If nothing is evicted (T <= sink+window), no K/V Restoration
        # needed — run the verifier directly. This is the trivial case
        # for short prompts, e.g. T=8 with sink=4 + window=64.
        if not evicted_positions:
            return self.verifier_model(input_ids=input_ids, use_cache=False)

        # f_θ projection
        verifier_k_full, verifier_v_full = self.project_drafter_kv(input_ids)
        # verifier_k_full shape: [B, T, L_v, num_kv_heads_v, head_dim_v]

        # Patch verifier attention forwards to inject K/V at evicted
        # positions. Restore originals after the forward.
        layers = self.verifier_model.model.layers
        originals: List[Callable] = []
        try:
            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                originals.append(attn.forward)
                attn.forward = self._make_patched_forward(
                    attn,
                    layer_idx=layer_idx,
                    evicted_positions=evicted_positions,
                    verifier_k_at_layer=verifier_k_full[:, :, layer_idx],
                    verifier_v_at_layer=verifier_v_full[:, :, layer_idx],
                    apply_rotary_pos_emb=apply_rotary_pos_emb,
                    eager_attention_forward=eager_attention_forward,
                    all_attention_functions=all_attention_functions,
                )
            return self.verifier_model(input_ids=input_ids, use_cache=False)
        finally:
            for layer_idx, layer in enumerate(layers):
                layer.self_attn.forward = originals[layer_idx]

    def _make_patched_forward(
        self, attn_module: nn.Module, *,
        layer_idx: int,
        evicted_positions: List[int],
        verifier_k_at_layer: torch.Tensor,
        verifier_v_at_layer: torch.Tensor,
        apply_rotary_pos_emb: Callable,
        eager_attention_forward: Callable,
        all_attention_functions: Optional[Any] = None,
    ) -> Callable:
        """Build a patched attention forward that injects K/V at evicted
        positions from `verifier_k_at_layer` / `verifier_v_at_layer`
        instead of using the verifier's own k_proj / v_proj at those
        positions.

        The patched forward replicates the standard verifier attention
        layer (Q, K, V projections + RoPE + GQA + softmax) with one
        change: after K, V are computed at every position, K and V at
        evicted positions are OVERWRITTEN with the f_θ-projected values
        (after k_norm + RoPE applied to match the standard pipeline).
        """
        def _patched_forward(
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, T, _ = hidden_states.shape

            input_shape = (B, T)
            hidden_shape = (*input_shape, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(key_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

            # Inject f_θ K/V at evicted positions.
            # verifier_k_at_layer shape: [B, T, num_kv_heads_v, head_dim_v]
            # K/V from k_proj also at all T positions; we overwrite the
            # evicted slice with f_θ output (after k_norm + RoPE).
            if evicted_positions:
                key_states, value_states = prepare_restored_attention_kv(
                    K_local=key_states,
                    V_local=value_states,
                    captured_K_pre_norm=verifier_k_at_layer,
                    captured_V=verifier_v_at_layer,
                    evicted_positions=evicted_positions,
                    k_norm=attn_module.k_norm,
                    position_embeddings=(cos, sin),
                )

            # Standard attention path
            attn_impl = getattr(
                attn_module.config, "_attn_implementation", "eager",
            )
            if attn_impl == "eager" or all_attention_functions is None:
                attention_interface = eager_attention_forward
            else:
                attention_interface = all_attention_functions[attn_impl]

            attn_output, attn_weights = attention_interface(
                attn_module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=getattr(attn_module, "attention_dropout", 0.0),
                scaling=getattr(attn_module, "scaling", attn_module.head_dim ** -0.5),
                sliding_window=getattr(attn_module, "sliding_window", None),
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        return _patched_forward


# ---------------------------------------------------------------------------
# Drafter K/V capture (DFlashDrafter-aware variant of capture_proposer_kv)
# ---------------------------------------------------------------------------


def _capture_drafter_kv(
    *, verifier_model: Any, drafter: Any, input_ids: torch.Tensor,
) -> KVCapture:
    """Capture pre-norm pre-RoPE K/V from the DFlash drafter at every
    drafter layer at every position.

    DFlashDrafter (PR #93) has a non-standard structure: it doesn't
    follow the embed → layers → norm → lm_head pattern. It's a flat
    ``nn.Module`` with ``.layers`` directly + an architectural choice
    that **embed_tokens are shared with the verifier** (DFlash design:
    no own embeddings or lm_head).

    Capture strategy:

      1. ``verifier_model.get_input_embeddings()(input_ids) * scale``
         → real embedded hiddens (Gemma scaling: ``× sqrt(hidden_size)``).
      2. Pass these embedded hiddens through ``drafter.layers`` with
         ``ctx_k = ctx_v = None`` per layer (no aux conditioning).
      3. Forward hooks on each layer's ``self_attn.k_proj`` /
         ``self_attn.v_proj`` capture pre-norm pre-RoPE K, V values
         per layer per position.

    This produces K/V values from drafter layers operating on REAL
    embedded hiddens (not synthetic zero) but WITHOUT aux conditioning
    on verifier mid-layer hiddens. For f_θ first-iteration training,
    this is the correct level: f_θ learns to project drafter K/V
    (computed without aux) into verifier K/V space. Adding aux
    conditioning is a follow-up that can be plumbed into both training
    and inference paths once first-iteration f_θ validates the
    architecture.

    Required because :func:`capture_proposer_kv` doesn't support the
    DFlashDrafter shape — it looks for ``model.model.layers`` or
    ``model.transformer.h`` and DFlashDrafter has neither.
    """
    # Capture K, V via forward hooks on each drafter layer's k_proj / v_proj.
    layers = list(drafter.layers)
    num_layers = len(layers)
    k_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    v_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    handles = []

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        def _make_k_hook(idx):
            def hook(_mod, _inp, output):
                k_capture[idx] = output.detach()
            return hook

        def _make_v_hook(idx):
            def hook(_mod, _inp, output):
                v_capture[idx] = output.detach()
            return hook

        handles.append(attn.k_proj.register_forward_hook(_make_k_hook(i)))
        handles.append(attn.v_proj.register_forward_hook(_make_v_hook(i)))

    try:
        # Embed input_ids using the verifier's embed_tokens (DFlash
        # design: shares verifier embeddings, no own lookup table).
        # Apply Gemma's × sqrt(hidden) scaling per the alignment
        # training pipeline convention.
        cfg = drafter.cfg
        verifier_embed = verifier_model.get_input_embeddings()
        embed_scale = math.sqrt(cfg.hidden_size)

        drafter_param = next(drafter.parameters())
        drafter_dtype = drafter_param.dtype
        drafter_device = drafter_param.device

        with torch.no_grad():
            input_ids_for_embed = input_ids.to(verifier_embed.weight.device)
            embedded = verifier_embed(input_ids_for_embed) * embed_scale
            embedded = embedded.to(device=drafter_device, dtype=drafter_dtype)
            T = embedded.size(1)
            query_positions = torch.arange(T, device=drafter_device)
            # Run each drafter layer with NO aux conditioning (ctx_k =
            # ctx_v = None). The k_proj / v_proj hooks fire on the
            # query hidden states' projection, capturing pre-norm
            # pre-RoPE K/V at every layer at every position.
            h = embedded
            for layer in layers:
                h = layer(h, query_positions, ctx_k=None, ctx_v=None)
    finally:
        for h in handles:
            h.remove()

    if any(k is None for k in k_capture):
        raise RuntimeError("drafter K capture missing some layers")
    if any(v is None for v in v_capture):
        raise RuntimeError("drafter V capture missing some layers")

    keys = []
    values = []
    for k_raw, v_raw in zip(k_capture, v_capture):
        # k_raw shape: [B, T, num_d_kv_heads * head_dim] (k_proj output)
        b, t, last = k_raw.shape
        if last != cfg.num_key_value_heads * cfg.head_dim:
            raise RuntimeError(
                f"drafter k_proj output last-dim {last} != "
                f"num_kv_heads * head_dim "
                f"({cfg.num_key_value_heads * cfg.head_dim})"
            )
        keys.append(k_raw.view(b, t, cfg.num_key_value_heads, cfg.head_dim))
        values.append(v_raw.view(b, t, cfg.num_key_value_heads, cfg.head_dim))

    return KVCapture(
        keys=keys,
        values=values,
        num_layers=len(keys),
        seq_len=keys[0].shape[1] if keys else 0,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
