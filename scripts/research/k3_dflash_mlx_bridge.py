"""K3 Step 3b — MLX↔PyTorch bridge for cross-runtime DFlash speculative
decoding on Mac M4.

Background
----------

PR #93 ships ``scripts/research/k3_dflash_specdecode_eval.py`` for CUDA:

  * verifier   = google/gemma-4-26B-A4B-it     (PyTorch bf16 via transformers)
  * drafter    = z-lab/gemma-4-26B-A4B-it-DFlash (PyTorch bf16, native impl)
  * spec loop  = (verifier forward + aux hiddens) → (DFlash propose_block)
                 → (verifier verify_block) → accept/reject → repeat

On Mac M4 24 GB the verifier MUST be 4-bit MLX (PyTorch MPS bf16 won't fit
the 26B model). The drafter stays PyTorch (PR #93's ``DFlashDrafter`` is
pure torch and runs comfortably on MPS). This means the spec decode loop
crosses runtime boundaries: MLX verifier ↔ PyTorch drafter.

This module supplies the bridge primitives needed for the cross-runtime
loop, **without touching anything in inference_engine/v04/**:

  1. :class:`MLXVerifierAuxProvider` — implements PR #93's
     :class:`AuxHiddenProvider` contract by running the MLX Gemma 4
     verifier with intermediate-layer hidden capture, converting
     captured hiddens (mx.array) → torch tensors via numpy.

  2. :func:`build_mlx_verifier_callbacks` — builds the
     ``embed_fn`` / ``lm_head_fn`` callbacks the DFlash drafter needs
     (``DFlashDrafter.draft_block(..., embed_fn, lm_head_fn, ...)``).
     The drafter calls these with torch token ids / hidden states, the
     callbacks bridge into MLX, run the MLX embed / lm_head, bridge
     back to torch.

  3. :func:`mlx_verify_block` — Mac equivalent of PR #93's
     ``verify_block()``: takes (committed, draft, mlx_verifier) and runs
     a forward over the concatenated sequence on the MLX side, returns
     accepted_count + correction_token (Python ints, no torch needed).

  4. Bridge utilities — ``mx_to_torch(x, dtype, device)`` and
     ``torch_to_mx(t)`` with explicit dtype/device handling.

Validation gates
----------------

This module is unit-tested for the bridge utilities on Linux CI using
numpy stand-ins (no mlx import on Linux). The MLX-touching paths
(:class:`MLXVerifierAuxProvider`, :func:`build_mlx_verifier_callbacks`,
:func:`mlx_verify_block`) require Mac M4 hardware to validate end-to-end
— their correctness is proven by ``scripts/research/
k3_dflash_specdecode_eval_mac.py`` running on real hardware and
producing acceptance evidence comparable to PR #93's CUDA evidence.

Architectural caveats (recorded for future contributors)
--------------------------------------------------------

* The numpy intermediate adds a synchronous CPU round-trip per bridge
  call. For a DFlash spec loop that runs ``aux_hidden_context`` every
  block (not every token), the per-block bridge cost is negligible
  compared to the ~10s/8-token MLX verifier forward measured 2026-06-09.
  If the bridge becomes a bottleneck under future optimisation, MLX
  ↔ torch.utils.dlpack zero-copy is a future option (when both
  runtimes support DLPack on the same buffer).

* MLX uses fp32 for hidden states by default; the DFlash drafter (per
  PR #93's alignment training scope) expects fp32 input to its
  ``fc`` projection. Bridge defaults to ``dtype=torch.float32`` to
  match this. Drafter K/V tensors are bf16 (drafter's own working
  dtype); the bridge respects whatever ``dtype`` the caller passes.

* The Gemma 4 verifier uses tied embeddings (``tie_word_embeddings:
  True`` per the production checkpoint config). This means the lm_head
  is ``embed_tokens.as_linear()`` not a separate matrix. The
  callback factory handles both tied and untied cases.

* MLX Gemma 4 logit softcapping (``final_logit_softcapping: 30.0``) is
  applied inside the MLX ``Model.__call__``. Our ``lm_head_fn`` mirrors
  this on the bridge boundary so the drafter sees logits with the same
  softcap as the CUDA path.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Bridge utilities (no mlx_lm dependency on Linux CI; numpy-only fallback
# for the dtype/shape/device test surface).
# ---------------------------------------------------------------------------


def mx_to_torch(
    x: Any, *, dtype: Any = None, device: Any = "cpu",
) -> Any:
    """Convert an mlx array to a torch tensor via numpy intermediate.

    Parameters
    ----------
    x
        ``mlx.core.array`` (Apple Silicon) or any object with an
        ``__array__`` method (numpy fallback for unit tests on Linux).
    dtype
        Optional torch dtype to cast to. ``None`` keeps the bridged
        dtype as-is.
    device
        Optional torch device string ('cpu' / 'mps' / 'cuda'). Defaults
        to 'cpu' which is the safest cross-platform default.
    """
    import numpy as np
    import torch
    arr = np.asarray(x)
    t = torch.from_numpy(arr.copy())  # copy: numpy array may be on a
                                       # buffer the MLX runtime owns
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None and str(t.device) != str(device):
        t = t.to(device)
    return t


def torch_to_mx(t: Any) -> Any:
    """Convert a torch tensor to an mlx array.

    Tensors on accelerator devices are first moved to CPU; tensors are
    materialised to numpy then handed to ``mx.array``. ``bfloat16`` is
    upcast to ``float32`` because mlx_lm's verifier hidden states /
    embeddings live in fp32 and bfloat16 lacks a numpy equivalent.
    """
    import mlx.core as mx  # type: ignore
    import torch
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    if t.device.type != "cpu":
        t = t.detach().cpu()
    return mx.array(t.numpy())


# ---------------------------------------------------------------------------
# MLX verifier aux-hidden capture
# ---------------------------------------------------------------------------


class MLXVerifierAuxProvider:
    """Aux-hidden provider that runs the MLX Gemma 4 verifier and
    captures hidden states at the ``aux_layer_ids`` indices.

    Conforms to PR #93's :class:`AuxHiddenProvider` contract:

        aux_hidden_context(committed_token_ids: List[int])
            -> Tuple[List[torch.Tensor], int]

    Returns ``(aux_list, bonus_token_id)`` where:

      * ``aux_list`` is a list of ``len(aux_layer_ids)`` torch tensors
        of shape ``[1, C, hidden]`` and dtype ``torch.float32``
        (matches PR #93's CUDA VerifierAuxProvider which calls
        ``.float()`` on the captured hiddens).

      * ``bonus_token_id`` is the verifier's greedy next token at
        position ``C`` (Python int).

    The aux capture works by manually replaying the MLX
    ``Gemma4TextModel.__call__`` layer loop while recording the
    output of the requested layer indices. This duplicates ~30 LOC
    of the upstream forward but avoids monkey-patching and works
    across mlx_lm minor versions as long as the layer-loop structure
    is preserved (verified against mlx_lm 0.31.3).
    """

    def __init__(
        self,
        mlx_model: Any,
        aux_layer_ids: Sequence[int],
        *,
        bridge_dtype: Any = None,
        bridge_device: Any = "cpu",
    ) -> None:
        self.mlx_model = mlx_model
        self.aux_layer_ids = tuple(aux_layer_ids)
        self.bridge_dtype = bridge_dtype  # default: keep fp32
        self.bridge_device = bridge_device
        self.forward_calls = 0

    def aux_hidden_context(
        self, committed_token_ids: List[int],
    ) -> Tuple[List[Any], int]:
        import mlx.core as mx  # type: ignore
        import torch

        if not committed_token_ids:
            raise ValueError("committed_token_ids must be non-empty")

        # mlx_lm.load returns Model (the wrapper); the inner text model
        # is at .model. The verifier-side Gemma4TextModel layer loop is
        # what we need to instrument.
        outer = self.mlx_model           # Model (lm_head wrapper)
        inner = outer.model              # Gemma4TextModel

        # Build input_ids as mx.array.
        ids_mx = mx.array([committed_token_ids])  # shape [1, C]

        # Replay the layer-loop logic from
        # mlx_lm.models.gemma4_text.Gemma4TextModel.__call__ (verified
        # against 0.31.3). We need:
        #   1. embed_tokens(ids) → h
        #   2. (per-layer-input setup, masks, cache=None for fresh forward)
        #   3. for each layer: h = layer(h, mask, c, ...)
        #   4. capture h at aux_layer_ids
        #   5. final norm
        #   6. lm_head (or tied embed) → logits
        #   7. softcap on last-position logits → bonus

        # Step 1: embed_tokens.
        h = inner.embed_tokens(ids_mx)

        # Step 2: per_layer_inputs + masks + cache.
        per_layer_inputs = inner.get_per_layer_inputs(ids_mx) if hasattr(
            inner, "get_per_layer_inputs",
        ) else [None] * len(inner.layers)
        cache = [None] * len(inner.layers)
        masks = inner._make_masks(h, cache)

        # Step 3: layer loop with intermediate capture.
        captured: dict = {}
        intermediates = [(None, None)] * len(inner.layers)
        for idx, (layer, c, mask, prev_idx, per_layer_input) in enumerate(
            zip(
                inner.layers,
                cache,
                masks,
                inner.previous_kvs,
                per_layer_inputs,
            )
        ):
            kvs, offset = intermediates[prev_idx]
            h, kvs, offset = layer(
                h, mask, c,
                per_layer_input=per_layer_input,
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (kvs, offset)
            if idx in self.aux_layer_ids:
                captured[idx] = h  # mx.array [1, C, hidden]

        # Step 5: final norm.
        h_final = inner.norm(h)

        # Step 6 + 7: logits + softcap → bonus.
        if outer.tie_word_embeddings:
            logits_mx = outer.model.embed_tokens.as_linear(h_final)
        else:
            logits_mx = outer.lm_head(h_final)
        if outer.final_logit_softcapping is not None:
            cap = outer.final_logit_softcapping
            logits_mx = cap * mx.tanh(logits_mx / cap)
        # Bonus = argmax of last-position logits.
        bonus_arr = mx.argmax(logits_mx[0, -1])
        bonus = int(bonus_arr.item())

        # Bridge captured hiddens → torch.float32 (matches CUDA path).
        aux_list: List[Any] = []
        for layer_idx in self.aux_layer_ids:
            if layer_idx not in captured:
                raise RuntimeError(
                    f"aux_layer_ids contains layer {layer_idx} but only "
                    f"{len(inner.layers)} layers exist in the verifier"
                )
            t = mx_to_torch(
                captured[layer_idx],
                dtype=self.bridge_dtype if self.bridge_dtype is not None
                else torch.float32,
                device=self.bridge_device,
            )
            aux_list.append(t)

        self.forward_calls += 1
        return aux_list, bonus


# ---------------------------------------------------------------------------
# Verifier embed / lm_head callbacks for the DFlash drafter
# ---------------------------------------------------------------------------


def build_mlx_verifier_callbacks(
    mlx_model: Any, hidden_size: int, softcap: Optional[float],
    *, bridge_dtype: Any = None, bridge_device: Any = "cpu",
) -> Tuple[Callable, Callable]:
    """Build ``(embed_fn, lm_head_fn)`` callbacks for the DFlash drafter.

    ``embed_fn(token_ids: torch.Tensor) -> torch.Tensor`` of shape
    ``[*, hidden]`` × ``sqrt(hidden)`` (Gemma scaling).

    ``lm_head_fn(h: torch.Tensor) -> torch.Tensor`` of shape ``[*, vocab]``
    with optional softcapping applied.

    Both bridge into MLX, run the verifier's embed_tokens / lm_head, then
    bridge back to torch with the requested dtype/device.

    The Gemma 4 verifier uses tied embeddings (lm_head = embed.as_linear).
    The factory handles both tied and untied cases.
    """
    import torch

    inner = mlx_model.model
    tied = mlx_model.tie_word_embeddings
    scale = math.sqrt(hidden_size)

    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        ids_mx = torch_to_mx(ids)
        emb_mx = inner.embed_tokens(ids_mx)
        # Gemma 4 multiplies token embeddings by sqrt(hidden) — match
        # k3_dflash_specdecode_eval.py's _build_embed_lm_head.
        emb_mx = emb_mx * scale
        return mx_to_torch(
            emb_mx,
            dtype=bridge_dtype if bridge_dtype is not None else torch.float32,
            device=bridge_device,
        )

    def lm_head_fn(h: torch.Tensor) -> torch.Tensor:
        h_mx = torch_to_mx(h)
        if tied:
            logits_mx = inner.embed_tokens.as_linear(h_mx)
        else:
            logits_mx = mlx_model.lm_head(h_mx)
        if softcap is not None:
            import mlx.core as mx  # type: ignore
            logits_mx = softcap * mx.tanh(logits_mx / softcap)
        return mx_to_torch(
            logits_mx,
            dtype=bridge_dtype if bridge_dtype is not None else torch.float32,
            device=bridge_device,
        )

    return embed_fn, lm_head_fn


# ---------------------------------------------------------------------------
# verify_block (Mac equivalent of CUDA verify_block in PR #93)
# ---------------------------------------------------------------------------


def mlx_verify_block(
    mlx_model: Any, committed: List[int], draft: List[int],
) -> Tuple[int, int]:
    """Run the verifier over ``committed + draft`` and return
    ``(accepted_count, correction_token)`` matching PR #93's
    ``k3_dflash_specdecode_eval.verify_block`` semantics.

    Greedy accept: walk through draft positions, at each position
    compare the verifier's argmax to the draft token; accept the
    longest contiguous matching prefix. Correction is the verifier's
    argmax at the first mismatched position (which the spec-decode
    loop commits as the "corrected" token regardless).
    """
    import mlx.core as mx  # type: ignore

    seq = committed + draft
    inp = mx.array([seq])
    logits_mx = mlx_model(inp)  # [1, C+L, V]
    if mlx_model.final_logit_softcapping is not None:
        # Already applied inside __call__ for the wrapper, so skip;
        # but if a future mlx_lm version moves softcap elsewhere,
        # add a re-apply here.
        pass
    # logits_mx is [1, C+L, V]; greedy argmax over vocab dim per position.
    preds_mx = mx.argmax(logits_mx[0], axis=-1)  # [C+L]
    preds = preds_mx.tolist()
    C = len(committed)
    accepted = 0
    for i in range(len(draft)):
        if preds[C - 1 + i] == draft[i]:
            accepted += 1
        else:
            break
    correction = int(preds[C - 1 + accepted])
    return accepted, correction


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "mx_to_torch",
    "torch_to_mx",
    "MLXVerifierAuxProvider",
    "build_mlx_verifier_callbacks",
    "mlx_verify_block",
]
