"""K3 Block B MLX variant — Cross-model `DLMRestoredVerifier` for Mac M4.

CUDA path (`inference_engine.v04.cross_model_dlm_verifier`) instruments
HF transformers attention modules via Python attribute monkey-patching.
Mac M4 verifier is mlx-lm 4-bit which uses a different runtime
(`mlx.core` arrays, mlx.nn.Module functional style) so the same
monkey-patch approach works **but** the K/V injection logic must be
written in MLX-native ops.

This module implements the integrated K3 Kakeya inference architecture
on Mac M4:

  verifier (Gemma 4 26B-A4B-it, mlx_lm 4-bit):
    └─ holds only sink+window local KV cache (sink=4 + window=64)
    └─ at evicted positions, K/V come from drafter (PyTorch) via f_θ

  drafter (DFlash 0.4B, PyTorch on MPS):
    └─ runs full forward over input_ids
    └─ K/V at every layer at every position captured (numpy bridge)
    └─ projected to verifier K/V space via trained f_θ (PyTorch fp32)

  bridge (numpy intermediate):
    └─ torch.Tensor → numpy → mx.array (per ADR §11.7.0 Mac MLX path
       in PR #102 + this PR's MLXVerifierAuxProvider extension)

Differences from CUDA path
--------------------------

1. **MLX K/V injection**: per-layer K/V at evicted positions is
   bridged from PyTorch (f_θ output) → MLX, then injected via
   `mx.where(evicted_mask, injected_kv, original_kv)`. MLX is
   functional so we can't do in-place index assignment; the where-
   based scatter is the idiomatic MLX pattern.

2. **k_norm + RoPE on bridged K**: f_θ output K is RAW (pre-norm
   pre-RoPE). MLX side applies the layer's k_norm + RoPE to the
   bridged K BEFORE scattering — matches what the verifier's own
   k_proj path does. V gets v_norm (no RoPE — V doesn't go through
   RoPE in standard transformer attention).

3. **KV-shared layer handling**: some Gemma 4 layers (the last
   `num_kv_shared_layers` layers) reuse K/V from earlier layers.
   For these, no k_proj/v_proj exists; injection is skipped and
   the verifier's normal "shared_kv" path runs unmodified.

4. **K-eq-V handling**: full-attention layers with
   `attention_k_eq_v=True` have K and V as the SAME tensor (memory
   savings). f_θ injection uses K's prepared (post-norm post-RoPE)
   value for both K and V at evicted positions in those layers,
   matching the verifier's own "K = V" semantics.

5. **mlx.nn.Module monkey-patch**: per-instance `__call__` override.
   Works because mlx.nn.Module is Python; instance attributes can be
   set freely. The override stores the original method and restores
   it in a `try/finally` block, same pattern as CUDA path.

Validation gates
----------------

CUDA equivalent (`inference_engine.v04.cross_model_dlm_verifier`) is
fully unit-tested + has scripts/research/k3_integrated_niah_eval.py
for vast.ai product evidence. This MLX variant has Linux CI tests
for the **bridge utilities + dimension validation** (mlx-touching
paths require Mac M4 hardware to validate end-to-end). Final
validation gate: end-to-end run of
``scripts/research/k3_integrated_niah_eval_mac.py`` on Mac M4
producing acceptance + recall + memory evidence.

Architectural caveats
---------------------

* The numpy bridge adds a synchronous CPU round-trip per per-layer
  K/V injection. For T=6413 (ctx280 measured 2026-06-09), per-layer
  bridge cost ≈ 8 layers × 6413 × 2048 × 4 bytes (fp32 numpy) ≈
  400 MB transient per forward step. Acceptable for first-iteration
  evidence; future optimisation could pin numpy buffers or use MLX
  DLPack interop when both runtimes support it.

* This module imports `mlx.core` lazily inside method bodies to
  keep the module importable on Linux CI (where mlx is unavailable).
  Linux CI tests exercise the dimension-validation + bridge code
  path via numpy stand-ins (matching PR #102's pattern).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from inference_engine.v04.f_theta import FThetaProjection
from inference_engine.v04.kv_merge import compute_evicted_positions


@dataclasses.dataclass
class _MLXLayerWiring:
    """Per-layer wiring info derived from the mlx_lm Gemma 4 model
    structure. Computed once at __init__; used by the monkey-patched
    Attention.__call__ on each forward.
    """
    layer_idx: int
    has_kv: bool                # whether this layer has its own k_proj/v_proj
    is_sliding: bool            # sliding vs full attention layer
    use_k_eq_v: bool            # K-eq-V optimisation (K and V share tensor)
    n_kv_heads: int
    head_dim: int


class MLXCrossModelDLMRestoredVerifier:
    """Cross-model DLMRestoredVerifier for the Mac MLX runtime.

    Construction
    ------------

    >>> verifier = MLXCrossModelDLMRestoredVerifier(
    ...     mlx_verifier=mlx_lm_model,        # mlx_lm.load() result
    ...     drafter=dflash_drafter,           # PyTorch DFlashDrafter
    ...     f_theta=f_theta_projection,       # PyTorch FThetaProjection
    ...     sink_size=4,
    ...     window_size=64,
    ... )

    Forward
    -------

    >>> outputs = verifier.forward(input_ids)
    >>> # outputs is the mlx_lm verifier's logits ([B, T, vocab])
    """

    def __init__(
        self,
        *,
        mlx_verifier: Any,                # mlx_lm.load() result
        drafter: Any,                     # DFlashDrafter (PyTorch)
        f_theta: FThetaProjection,
        sink_size: int = 4,
        window_size: int = 64,
        bridge_dtype: Any = None,
        bridge_device: Any = "cpu",
    ) -> None:
        if sink_size < 0 or window_size < 0:
            raise ValueError("sink_size and window_size must be non-negative")
        self.mlx_verifier = mlx_verifier
        self.drafter = drafter
        self.f_theta = f_theta
        self.sink_size = sink_size
        self.window_size = window_size
        self.bridge_dtype = bridge_dtype
        self.bridge_device = bridge_device

        self._validate_dimensions()
        self._wirings = self._derive_layer_wirings()

    # -----------------------------------------------------------------
    # Dimension validation — mlx-aware
    # -----------------------------------------------------------------

    def _validate_dimensions(self) -> None:
        cfg = self.f_theta.config

        # mlx_lm wraps the text model; resolve via the standard pattern
        # (similar to scripts/research/k3_dflash_mlx_bridge.py).
        outer = getattr(self.mlx_verifier, "language_model", self.mlx_verifier)
        text_model = getattr(outer, "model", None) or outer
        layers = getattr(text_model, "layers", None)
        if layers is None or not hasattr(layers, "__len__"):
            raise AttributeError(
                "Could not locate MLX verifier text-model layers; "
                "expected mlx_verifier.model.layers or .language_model.model.layers"
            )

        if len(layers) != cfg.verifier_num_layers:
            raise ValueError(
                f"f_θ trained for verifier_num_layers={cfg.verifier_num_layers} "
                f"but mlx verifier has {len(layers)}"
            )

        # Drafter
        drafter_cfg = getattr(self.drafter, "cfg", None) or getattr(self.drafter, "config", None)
        if drafter_cfg is not None:
            if getattr(drafter_cfg, "num_hidden_layers", None) != cfg.drafter_num_layers:
                raise ValueError(
                    f"f_θ trained for drafter_num_layers={cfg.drafter_num_layers} "
                    f"but drafter has {drafter_cfg.num_hidden_layers}"
                )
            if getattr(drafter_cfg, "num_key_value_heads", None) != cfg.drafter_num_kv_heads:
                raise ValueError(
                    f"f_θ trained for drafter_num_kv_heads={cfg.drafter_num_kv_heads} "
                    f"but drafter has {drafter_cfg.num_key_value_heads}"
                )

    def _derive_layer_wirings(self) -> List[_MLXLayerWiring]:
        """Walk the mlx verifier layers, record per-layer KV wiring
        info needed by the patched Attention.__call__."""
        outer = getattr(self.mlx_verifier, "language_model", self.mlx_verifier)
        text_model = getattr(outer, "model", None) or outer
        layers = list(text_model.layers)
        wirings: List[_MLXLayerWiring] = []
        for idx, layer in enumerate(layers):
            attn = layer.self_attn
            wirings.append(_MLXLayerWiring(
                layer_idx=idx,
                has_kv=bool(getattr(attn, "has_kv", True)),
                is_sliding=bool(getattr(attn, "is_sliding", False)),
                use_k_eq_v=bool(getattr(attn, "use_k_eq_v", False)),
                n_kv_heads=int(attn.n_kv_heads),
                head_dim=int(attn.head_dim),
            ))
        return wirings

    # -----------------------------------------------------------------
    # Drafter capture + f_θ projection (re-uses CUDA path's logic with
    # minor adaptation: drafter could be on MPS/CPU; verifier is MLX so
    # verifier_model.get_input_embeddings() goes through the MLX bridge).
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _project_drafter_kv(
        self, input_ids_torch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drafter forward + f_θ → verifier K, V at every (layer, position).

        Returns torch tensors of shape
        ``[B, T, verifier_num_layers, verifier_num_kv_heads, verifier_head_dim]``
        each (K and V).

        The drafter is PyTorch; embed_tokens come from the MLX verifier
        via the bridge. Drafter K/V capture follows the CUDA path's
        approach (no aux conditioning; first-iteration f_θ training
        target).
        """
        from scripts.research.k3_dflash_mlx_bridge import mx_to_torch  # noqa
        import mlx.core as mx  # type: ignore
        import math

        cfg = self.drafter.cfg

        # Resolve mlx text model and embed_tokens
        outer = getattr(self.mlx_verifier, "language_model", self.mlx_verifier)
        text_model = getattr(outer, "model", None) or outer
        mlx_embed = text_model.embed_tokens

        # Get drafter device/dtype
        drafter_param = next(self.drafter.parameters())
        drafter_dtype = drafter_param.dtype
        drafter_device = drafter_param.device

        # Embed via MLX, bridge to torch
        ids_mx = mx.array([input_ids_torch[0].cpu().tolist()])
        embedded_mx = mlx_embed(ids_mx)
        # Apply Gemma scaling (× sqrt(hidden)) — matches DFlash drafter
        # alignment training convention.
        scale = math.sqrt(cfg.hidden_size)
        embedded_mx = embedded_mx * scale
        embedded_torch = mx_to_torch(
            embedded_mx, dtype=drafter_dtype, device=drafter_device,
        )

        # Run drafter layers with no aux conditioning, capture K/V via
        # forward hooks on each layer's k_proj/v_proj.
        layers = list(self.drafter.layers)
        num_d_layers = len(layers)
        k_capture: List[Optional[torch.Tensor]] = [None] * num_d_layers
        v_capture: List[Optional[torch.Tensor]] = [None] * num_d_layers
        handles = []

        try:
            for i, layer in enumerate(layers):
                def _make_k_hook(idx):
                    def hook(_mod, _inp, output):
                        k_capture[idx] = output.detach()
                    return hook

                def _make_v_hook(idx):
                    def hook(_mod, _inp, output):
                        v_capture[idx] = output.detach()
                    return hook

                handles.append(
                    layer.self_attn.k_proj.register_forward_hook(_make_k_hook(i))
                )
                handles.append(
                    layer.self_attn.v_proj.register_forward_hook(_make_v_hook(i))
                )

            T = embedded_torch.size(1)
            query_positions = torch.arange(T, device=drafter_device)
            h = embedded_torch
            with torch.no_grad():
                for layer in layers:
                    h = layer(h, query_positions, ctx_k=None, ctx_v=None)
        finally:
            for h in handles:
                h.remove()

        if any(k is None for k in k_capture):
            raise RuntimeError("drafter K capture missing some layers (Mac MLX path)")
        if any(v is None for v in v_capture):
            raise RuntimeError("drafter V capture missing some layers (Mac MLX path)")

        # Reshape: each k_raw is [B, T, num_kv_heads * head_dim]
        cfg = self.drafter.cfg
        k_per_layer = []
        v_per_layer = []
        for k_raw, v_raw in zip(k_capture, v_capture):
            b, t, last = k_raw.shape
            expected = cfg.num_key_value_heads * cfg.head_dim
            if last != expected:
                raise RuntimeError(
                    f"drafter k_proj output last-dim {last} != expected {expected}"
                )
            k_per_layer.append(k_raw.view(b, t, cfg.num_key_value_heads, cfg.head_dim))
            v_per_layer.append(v_raw.view(b, t, cfg.num_key_value_heads, cfg.head_dim))

        # Run f_θ
        verifier_k, verifier_v = self.f_theta.forward_kv_pack(k_per_layer, v_per_layer)
        # Shape: [B, T, verifier_num_layers, verifier_num_kv_heads, verifier_head_dim]
        return verifier_k, verifier_v

    # -----------------------------------------------------------------
    # Forward (with K/V Restoration)
    # -----------------------------------------------------------------

    def forward(self, input_ids_torch: torch.Tensor) -> Any:
        """Run a verifier forward with f_θ-mediated K/V Restoration.

        Returns the mlx verifier's logits (mx.array) of shape
        ``[B, T, vocab]``, same as a plain ``mlx_verifier(input_ids)``
        call. This module's contribution: at evicted positions in
        every layer, the verifier's K/V come from f_θ instead of from
        the verifier's own k_proj/v_proj.
        """
        import mlx.core as mx  # type: ignore

        T = int(input_ids_torch.size(1))
        evicted_positions = compute_evicted_positions(
            T, self.sink_size, self.window_size,
        )

        if not evicted_positions:
            # Trivial path: T <= sink+window, no K/V Restoration needed.
            ids_mx = mx.array([input_ids_torch[0].cpu().tolist()])
            return self.mlx_verifier(ids_mx)

        # Drafter forward + f_θ projection (PyTorch)
        verifier_k_torch, verifier_v_torch = self._project_drafter_kv(input_ids_torch)
        # verifier_k_torch shape: [B, T, L_v, num_kv_heads_v, head_dim_v]
        # All on torch (drafter_device); we'll bridge to MLX per layer
        # inside the patched Attention.__call__.

        # Bridge to numpy ONCE up front (saves per-layer bridge cost).
        verifier_k_np = verifier_k_torch.detach().to(torch.float32).cpu().numpy()
        verifier_v_np = verifier_v_torch.detach().to(torch.float32).cpu().numpy()
        # Shape: [B, T, L_v, num_kv_heads_v, head_dim_v]

        # Build evicted-position mask as mx.array (T,)
        evicted_mask_list = [False] * T
        for p in evicted_positions:
            evicted_mask_list[p] = True
        evicted_mask = mx.array(evicted_mask_list)

        # Patch each layer's Attention.__call__ to inject f_θ K/V.
        outer = getattr(self.mlx_verifier, "language_model", self.mlx_verifier)
        text_model = getattr(outer, "model", None) or outer
        layers = list(text_model.layers)

        originals: List[Callable] = []
        try:
            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                wiring = self._wirings[layer_idx]
                originals.append(attn.__call__)
                attn.__call__ = self._make_patched_call(
                    attn=attn,
                    wiring=wiring,
                    verifier_k_np=verifier_k_np[:, :, layer_idx],
                    verifier_v_np=verifier_v_np[:, :, layer_idx],
                    evicted_mask=evicted_mask,
                )
            ids_mx = mx.array([input_ids_torch[0].cpu().tolist()])
            return self.mlx_verifier(ids_mx)
        finally:
            for layer_idx, layer in enumerate(layers):
                # Restore original __call__
                # (mlx.nn.Module: instance attribute override works the
                # same as PyTorch — delete the override to expose the
                # class method again.)
                try:
                    delattr(layer.self_attn, "__call__")
                except AttributeError:
                    pass

    def _make_patched_call(
        self, *, attn: Any, wiring: _MLXLayerWiring,
        verifier_k_np: Any, verifier_v_np: Any,
        evicted_mask: Any,
    ) -> Callable:
        """Build a patched Attention.__call__ that injects f_θ K/V at
        evicted positions before the cache.update_and_fetch step.

        Mirrors the structure of mlx_lm.models.gemma4_text.Attention.__call__
        (read 2026-06-10) with one inserted step:

          ... (q_proj, k_proj, v_proj, q_norm, k_norm, v_norm, RoPE) ...
          # NEW: at evicted positions, REPLACE keys/values with f_θ output
          if cache is not None:
              keys, values = cache.update_and_fetch(keys, values)
          ... (scaled_dot_product_attention) ...

        For KV-shared layers (has_kv=False), no injection needed —
        those layers don't compute their own K/V (use shared_kv from
        a previous layer). Pass-through to the original __call__.
        """
        if not wiring.has_kv:
            # No injection on KV-shared layers; pass-through.
            return attn.__class__.__call__.__get__(attn)

        import mlx.core as mx  # type: ignore
        from scripts.research.k3_dflash_mlx_bridge import mx_to_torch  # noqa

        verifier_k_mx = mx.array(verifier_k_np)  # [B, T, num_kv_heads, head_dim]
        verifier_v_mx = mx.array(verifier_v_np)

        def _patched_call(
            x: Any,                        # mx.array, [B, L, hidden]
            mask: Any = None,
            cache: Any = None,
            shared_kv: Any = None,
            offset: Any = None,
        ) -> Any:
            from mlx_lm.models.gemma4_text import scaled_dot_product_attention  # type: ignore

            B, L, _ = x.shape
            queries = attn.q_proj(x).reshape(B, L, attn.n_heads, attn.head_dim)
            queries = attn.q_norm(queries)

            if shared_kv is not None:
                keys, values = shared_kv
            else:
                # Standard K/V from the verifier's own projections
                keys = attn.k_proj(x).reshape(
                    B, L, attn.n_kv_heads, attn.head_dim,
                )
                values = keys
                if not wiring.use_k_eq_v:
                    values = attn.v_proj(x).reshape(
                        B, L, attn.n_kv_heads, attn.head_dim,
                    )

                offset = mx.array(cache.offset) if cache is not None else 0

                # ---- EVICTED-POSITION INJECTION ----
                # f_θ output is per-position raw K/V (no norm, no RoPE).
                # Apply k_norm + RoPE to the f_θ output BEFORE scattering
                # so it matches the verifier's k_proj output's
                # post-norm post-RoPE state.
                #
                # Note: f_θ output shape [B, T, num_kv_heads, head_dim]
                # already matches the verifier's pre-transpose K shape.
                # We slice to current L (assume L == T here; spec decode
                # incremental cache scenarios are handled by mlx_lm
                # downstream of __call__).
                injected_k_normed = attn.k_norm(verifier_k_mx)
                if wiring.use_k_eq_v:
                    injected_v_normed = injected_k_normed
                else:
                    injected_v_normed = attn.v_norm(verifier_v_mx)

                # Scatter at evicted positions via where on the L axis.
                # Mask shape needs broadcasting: [L] → [1, L, 1, 1]
                if L == evicted_mask.shape[0]:
                    mask_b = evicted_mask.reshape(1, L, 1, 1)
                    keys = mx.where(mask_b, injected_k_normed, attn.k_norm(keys))
                    values = mx.where(mask_b, injected_v_normed, attn.v_norm(values))
                    # Above pre-norm K/V; now move to post-RoPE K layout
                    keys = keys.transpose(0, 2, 1, 3)
                    keys = attn.rope(keys, offset=offset)
                    values = values.transpose(0, 2, 1, 3)
                else:
                    # Defensive: L != T (e.g. incremental decode). Don't
                    # inject; fall back to standard path.
                    keys = attn.k_norm(keys)
                    keys = keys.transpose(0, 2, 1, 3)
                    keys = attn.rope(keys, offset=offset)
                    values = attn.v_norm(values)
                    values = values.transpose(0, 2, 1, 3)

            queries = queries.transpose(0, 2, 1, 3)
            queries = attn.rope(queries, offset=offset)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=attn.scale, mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return attn.o_proj(output), (keys, values), offset

        return _patched_call


__all__ = ["MLXCrossModelDLMRestoredVerifier"]
