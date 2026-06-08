"""K/V capture from a frozen dLM proposer's forward pass.

ADR 0008 §11.5 — the proposer's parallel forward computes K, V tensors
at every position transiently. v0.4 K/V Restoration uses these to fill
in the verifier's evicted-position cache slots **at compute time, not
permanent storage** (§11.3).

This module implements the capture half: hook every attention layer's
``k_proj`` and ``v_proj`` Linear modules, harvest their outputs after a
single forward, and return them as a structured object that the
injection layer (K1.B, separate PR) can consume.

What is captured
----------------

For each transformer layer *L* in the model, the K and V projections
produce tensors of shape ``[B, T, num_kv_heads * head_dim]``. We capture
exactly that — pre-`k_norm`, pre-RoPE, in the projection's natural
shape. The injection layer is responsible for re-applying RoPE for the
**target query position** when the captured K/V are reused inside the
verifier's attention.

Why pre-RoPE rather than post-RoPE
----------------------------------

RoPE encodes the absolute position of the K vector inside the attention
key space; the dot product Q · K depends only on the *relative* position
gap between query and key. If we captured post-RoPE K at proposer
position ``p`` and re-used it inside the verifier's attention at any
later position ``q``, the relative gap would be ``q - p`` — which is
exactly what we want. *In principle*, post-RoPE capture is also correct.

We chose pre-RoPE capture for three concrete engineering reasons:

1. **Stable hook point.** ``k_proj`` and ``v_proj`` are clean
   ``nn.Linear`` modules with a single output tensor. The post-RoPE
   tensor only exists ephemerally inside ``Gemma3Attention.forward`` and
   has no clean PyTorch hook surface — capturing it requires either
   subclassing or monkey-patching the attention module, both of which
   couple us to specific HF transformers versions.
2. **Re-projection flexibility.** For the cross-model case (K2), the
   captured K/V will pass through a learned projection ``f_θ`` before
   being used. RoPE is much cheaper to apply *after* ``f_θ`` than to
   strip-and-reapply it.
3. **Same-model identity check is exact.** When proposer and verifier
   share the same checkpoint (K1 setup), the captured raw K and the
   verifier's own raw K at any position are bit-identical (modulo
   nondeterministic kernel ordering, which we control by forcing
   ``attn_implementation="eager"``). This makes K1's first sanity
   gate — "round-trip K is preserved when ``f_θ = id``" — falsifiable
   on Linux without GPU.

What is NOT captured (yet)
--------------------------

* RoPE. Applied at injection time per target query position.
* Q tensors. The verifier computes its own Q from its own hidden state;
  proposer's Q is irrelevant for the restoration architecture.
* Attention weights. Out of scope; this module is a forward-only
  capture, not an interpretability instrument.
* Hidden states. Out of scope; we capture the outputs of the K/V
  projections specifically because those are what the attention
  consumes downstream.

API contract
------------

The single public entry point :func:`capture_proposer_kv` returns a
:class:`KVCapture` containing per-layer ``[B, T, num_kv_heads,
head_dim]`` tensors for both K and V. The :class:`KVCapture` object is
the input contract for K1.B (injection); it is intentionally narrow
(no model reference, no inference-time state) so that it can be
serialized, cached, or shipped across process boundaries without
dragging the proposer model along.

Linux-side unit tests in ``tests/inference_engine/v04/test_kv_capture.py``
exercise the contract on a synthetic mini-model with the same hook
surface as Gemma3Attention but no HF dependency.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KVCapture:
    """Per-layer K, V captures from a single proposer forward.

    Attributes
    ----------
    keys
        Per-layer K projections. Shape: list of length ``num_layers``;
        each element is ``[B, T, num_kv_heads, head_dim]``.
    values
        Per-layer V projections. Same shape as :attr:`keys`.
    num_layers
        ``len(self.keys)``. Provided as an attribute for downstream
        callers that prefer not to take ``len()``.
    seq_len
        ``T`` from the captured tensors. All layers share the same T.
    num_kv_heads
        Number of key/value heads on each layer (Gemma 3-1B uses GQA
        with num_kv_heads ≪ num_attention_heads).
    head_dim
        Per-head dimensionality.

    Invariants enforced at construction:

    * ``keys`` and ``values`` are non-empty and have the same length.
    * Every tensor in ``keys`` has the same ``[B, T, num_kv_heads,
      head_dim]`` shape; same for ``values``.
    * ``keys[i]`` and ``values[i]`` share the same dtype and device.

    The tensors stored here are detached from the proposer's autograd
    graph by :func:`capture_proposer_kv`; they are safe to free the
    proposer model while still using a :class:`KVCapture`.
    """

    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    num_layers: int
    seq_len: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if not self.keys or not self.values:
            raise ValueError("keys and values must be non-empty")
        if len(self.keys) != len(self.values):
            raise ValueError(
                f"keys ({len(self.keys)}) and values ({len(self.values)}) "
                f"must have the same number of layers"
            )
        ref_shape = self.keys[0].shape
        if len(ref_shape) != 4:
            raise ValueError(
                f"keys[0] must be 4-D [B, T, num_kv_heads, head_dim]; "
                f"got shape {tuple(ref_shape)}"
            )
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            if k.shape != ref_shape:
                raise ValueError(
                    f"keys[{i}] shape {tuple(k.shape)} != keys[0] "
                    f"shape {tuple(ref_shape)}"
                )
            if v.shape != ref_shape:
                raise ValueError(
                    f"values[{i}] shape {tuple(v.shape)} != keys[0] "
                    f"shape {tuple(ref_shape)}"
                )
            if k.dtype != v.dtype:
                raise ValueError(
                    f"layer {i} dtype mismatch: K={k.dtype} V={v.dtype}"
                )
            if k.device != v.device:
                raise ValueError(
                    f"layer {i} device mismatch: K={k.device} V={v.device}"
                )

    def select_positions(self, positions: Sequence[int]) -> "KVCapture":
        """Return a new :class:`KVCapture` containing only the listed
        token positions across all layers.

        ``positions`` must be sorted ascending and within ``[0, seq_len)``.
        Used by K1.B (injection) to extract K/V at evicted positions.
        """
        if not positions:
            raise ValueError("positions must be non-empty")
        sorted_positions = sorted(set(positions))
        if sorted_positions != list(positions):
            raise ValueError(
                "positions must be sorted ascending with no duplicates; "
                f"got {list(positions)}"
            )
        if sorted_positions[0] < 0 or sorted_positions[-1] >= self.seq_len:
            raise ValueError(
                f"positions must lie in [0, {self.seq_len}); "
                f"got [{sorted_positions[0]}, {sorted_positions[-1]}]"
            )
        idx = torch.tensor(sorted_positions, device=self.keys[0].device)
        new_keys = [k.index_select(dim=1, index=idx) for k in self.keys]
        new_values = [v.index_select(dim=1, index=idx) for v in self.values]
        return KVCapture(
            keys=new_keys,
            values=new_values,
            num_layers=self.num_layers,
            seq_len=len(sorted_positions),
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )


# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------


def _locate_attention_layers(model: nn.Module) -> List[nn.Module]:
    """Return the list of decoder layer modules whose ``self_attn``
    sub-module exposes ``k_proj`` and ``v_proj`` Linear projections.

    Currently supports the HF Gemma3 / Llama / Qwen / Mistral family
    (``model.model.layers[*].self_attn`` shape) and the GPT-2 family
    (``model.transformer.h[*].attn`` shape). Raises ``RuntimeError``
    on any other shape — silent fallbacks per ADR 0008 §6.2 are
    forbidden.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
    else:
        raise RuntimeError(
            "could not locate decoder layers on this model; expected "
            "model.model.layers (HF Gemma3 / Llama / Qwen / Mistral) "
            "or model.transformer.h (GPT-2 family). Add a binding for "
            "the new architecture."
        )
    if not layers:
        raise RuntimeError("model has zero decoder layers")
    return layers


def _attention_module(layer: nn.Module) -> nn.Module:
    """Return the self-attention sub-module from a decoder layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attn"):
        return layer.attn
    raise RuntimeError(
        f"could not locate self-attention sub-module on layer "
        f"{type(layer).__name__}; expected .self_attn or .attn"
    )


def register_kv_capture_hooks(
    model: nn.Module,
    *,
    layer_indices: Optional[Iterable[int]] = None,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List]:
    """Register forward hooks on each layer's ``k_proj`` and ``v_proj``
    that append their outputs to per-layer accumulator lists. The
    accumulators and the hook handles are returned so the caller can
    extract the captured tensors after a forward and ``handle.remove()``
    each hook.

    This is the lower-level primitive used internally by
    :func:`capture_proposer_kv`. Public callers should prefer the
    high-level function unless they need fine-grained control over
    when hooks fire (e.g., capturing across multiple forwards into the
    same accumulator).

    Parameters
    ----------
    model
        Any model whose decoder layers expose ``k_proj`` / ``v_proj``
        Linear projections inside their self-attention sub-module.
    layer_indices
        Optional subset of layer indices to capture. ``None`` captures
        all layers.

    Returns
    -------
    A tuple ``(k_acc, v_acc, handles)`` where ``k_acc[i]`` is a list of
    K-projection outputs collected at the i-th captured layer (across
    however many forwards fired the hooks), ``v_acc`` is the same for
    V, and ``handles`` is the list of hook handles to ``remove()``.

    Raises
    ------
    RuntimeError
        If decoder layers cannot be located, or any selected layer's
        attention module does not expose ``k_proj`` and ``v_proj``.
    ValueError
        If ``layer_indices`` contains an index outside ``[0, num_layers)``.
    """
    layers = _locate_attention_layers(model)
    if layer_indices is None:
        selected = list(range(len(layers)))
    else:
        selected = sorted(set(layer_indices))
        for idx in selected:
            if idx < 0 or idx >= len(layers):
                raise ValueError(
                    f"layer_indices contains {idx}, out of range "
                    f"[0, {len(layers)})"
                )

    k_acc: List[List[torch.Tensor]] = [[] for _ in selected]
    v_acc: List[List[torch.Tensor]] = [[] for _ in selected]
    handles = []

    for slot, layer_idx in enumerate(selected):
        attn = _attention_module(layers[layer_idx])
        if not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):
            raise RuntimeError(
                f"layer {layer_idx} self-attention "
                f"({type(attn).__name__}) does not expose k_proj / "
                "v_proj. K/V capture currently requires the standard "
                "fused-K-projection-Linear layout."
            )

        def make_k_hook(target_slot: int):
            def hook(_module, _inputs, output):
                k_acc[target_slot].append(output.detach())
            return hook

        def make_v_hook(target_slot: int):
            def hook(_module, _inputs, output):
                v_acc[target_slot].append(output.detach())
            return hook

        handles.append(attn.k_proj.register_forward_hook(make_k_hook(slot)))
        handles.append(attn.v_proj.register_forward_hook(make_v_hook(slot)))

    return k_acc, v_acc, handles


# ---------------------------------------------------------------------------
# High-level capture entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def capture_proposer_kv(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    layer_indices: Optional[Iterable[int]] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> KVCapture:
    """Run a single forward of ``model`` over ``input_ids`` and return
    a :class:`KVCapture` of the K and V projections at every (or selected)
    layer.

    Parameters
    ----------
    model
        Any model whose decoder layers' self-attention modules expose
        ``k_proj`` and ``v_proj`` Linear projections (HF Gemma3 / Llama
        / Qwen / Mistral / GPT-2 family).
    input_ids
        ``[B, T]`` token-id tensor. Currently only ``B=1`` is exercised
        in tests; the capture machinery is per-position-shape-agnostic
        but downstream injection code (K1.B) may still assume single-
        batch.
    attention_mask
        Optional attention mask passed straight through to the model.
        For proposer-role capture the mask is typically ``None``
        (full attention) or a batch padding mask. ADR 0008 §11.5
        specifies the proposer runs full attention; pass an explicit
        bounded mask only if you intentionally want to capture a
        bounded view.
    layer_indices
        Optional subset of layer indices to capture. ``None`` captures
        all layers (the production case).
    num_kv_heads, head_dim
        Optional explicit shape overrides. If omitted, derived from
        the model's config (``model.config.num_key_value_heads``,
        ``model.config.head_dim`` or ``hidden_size //
        num_attention_heads``). Useful for synthetic / surrogate models
        whose configs do not match the HF convention.

    Returns
    -------
    A :class:`KVCapture` whose ``keys[i]`` and ``values[i]`` are
    ``[B, T, num_kv_heads, head_dim]`` tensors detached from the
    forward graph.

    Notes
    -----
    The returned tensors are **pre-norm and pre-RoPE**, i.e., the raw
    output of ``k_proj`` / ``v_proj`` reshaped to per-head form. K1.B
    is responsible for re-applying RoPE for the verifier's target
    query position when injecting these into the verifier's attention.
    See the module docstring for the rationale (RoPE strip-and-
    reapply is cheaper after the K2 cross-model projection ``f_θ``;
    same-model round-trip is bit-exact in eager attention).

    The function is decorated with ``@torch.no_grad()`` because the
    proposer is frozen by design (ADR 0008 §11.5). If you need a
    differentiable path through the proposer's K/V — for example, to
    train ``f_θ`` end-to-end — call :func:`register_kv_capture_hooks`
    directly without the no-grad wrapper.
    """
    layers = _locate_attention_layers(model)
    if layer_indices is None:
        selected_indices = list(range(len(layers)))
    else:
        selected_indices = sorted(set(layer_indices))

    if num_kv_heads is None or head_dim is None:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError(
                "model has no .config; pass num_kv_heads and head_dim "
                "explicitly"
            )
        if num_kv_heads is None:
            num_kv_heads = getattr(cfg, "num_key_value_heads", None)
            if num_kv_heads is None:
                num_kv_heads = getattr(cfg, "num_attention_heads", None)
            if num_kv_heads is None:
                raise ValueError(
                    "could not derive num_kv_heads from config; pass "
                    "explicitly"
                )
        if head_dim is None:
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None:
                hidden = getattr(cfg, "hidden_size", None)
                num_q_heads = getattr(cfg, "num_attention_heads", None)
                if hidden is None or num_q_heads is None:
                    raise ValueError(
                        "could not derive head_dim from config; pass "
                        "explicitly"
                    )
                head_dim = hidden // num_q_heads

    k_acc, v_acc, handles = register_kv_capture_hooks(
        model, layer_indices=selected_indices,
    )

    try:
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        model(input_ids=input_ids, use_cache=False, **kwargs)
    finally:
        for h in handles:
            h.remove()

    if any(len(buf) != 1 for buf in k_acc):
        raise RuntimeError(
            "expected exactly one k_proj output per captured layer "
            "after a single forward; got "
            f"{[len(buf) for buf in k_acc]}. Did the model's forward "
            "fire the attention layers an unexpected number of times?"
        )
    if any(len(buf) != 1 for buf in v_acc):
        raise RuntimeError(
            "expected exactly one v_proj output per captured layer "
            "after a single forward; got "
            f"{[len(buf) for buf in v_acc]}"
        )

    keys: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    seq_len: Optional[int] = None
    for k_outputs, v_outputs in zip(k_acc, v_acc):
        k_raw = k_outputs[0]  # [B, T, num_kv_heads * head_dim]
        v_raw = v_outputs[0]
        if k_raw.dim() != 3:
            raise RuntimeError(
                f"unexpected k_proj output rank {k_raw.dim()}; "
                "expected 3 ([B, T, num_kv_heads * head_dim])"
            )
        b, t, last = k_raw.shape
        if last != num_kv_heads * head_dim:
            raise RuntimeError(
                f"k_proj output last-dim {last} != num_kv_heads "
                f"({num_kv_heads}) * head_dim ({head_dim}) = "
                f"{num_kv_heads * head_dim}; check the head-shape "
                "overrides"
            )
        k_reshaped = k_raw.view(b, t, num_kv_heads, head_dim)
        v_reshaped = v_raw.view(b, t, num_kv_heads, head_dim)
        if seq_len is None:
            seq_len = t
        elif t != seq_len:
            raise RuntimeError(
                f"layer captured T={t} but earlier layer captured "
                f"T={seq_len}; capture inconsistency"
            )
        keys.append(k_reshaped)
        values.append(v_reshaped)

    return KVCapture(
        keys=keys,
        values=values,
        num_layers=len(keys),
        seq_len=seq_len if seq_len is not None else 0,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
