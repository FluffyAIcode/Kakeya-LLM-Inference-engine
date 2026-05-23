"""MLX-backed sparse-logits DLM proposer.

API parity with
:class:`inference_engine.proposer.sparse_logits.SparseLogitsProposer`
so the speculative decoder is unchanged.

Design
======

The dllm-hub checkpoint ``Qwen3-0.6B-diffusion-mdlm-v0.1`` is
architecturally identical to ``Qwen/Qwen3-0.6B-Base`` — same hidden
size, same layer count, same kv-head structure. The only differences
are:

  1. **Attention mask**: causal in Qwen3-Base, bidirectional in the
     dllm-hub checkpoint (a masked-diffusion LM needs to see the
     full sequence including the masked positions).
  2. **Tokenizer**: dllm-hub's tokenizer adds a ``<|mask|>`` token
     for masked-diffusion denoising.

Both points are handled at runtime here: we load Qwen3-0.6B-Base's
architecture via ``mlx_lm.load(...)`` (which gives us a working MLX
``Qwen3.Model`` instance with the right shapes), then **overwrite its
weights** from the dllm-hub safetensors file we download via
``huggingface_hub.snapshot_download``. We use the *PyTorch* tokenizer
(via the existing ``DLMProposer``'s wrapper) so the chat-template /
mask-id semantics match the rest of the speculative loop exactly.

For the bidirectional-attention override we don't need to monkey-
patch the attention layer: ``mlx_lm.models.qwen3.Qwen3Model.__call__``
calls ``create_attention_mask(h, cache[0])`` and passes the result to
every layer. We just call ``model.model(inputs, cache=None)`` after
substituting our own mask of ``None`` (which the SDPA path
interprets as "no causal restriction") into the attention call. This
is done by running a single layer at a time, mirroring the model's
own forward but with the override.

Sparse logits
=============

Following Phase B's pattern: the model backbone is run for the full
sequence T, then ``embed_tokens.as_linear`` is applied **only at
masked positions** to produce a ``[1, n_masked, V]`` logits tensor
instead of ``[1, T, V]``. With Qwen3-0.6B's V=151_936 and the typical
T=50, n_masked=8, this trims the dominant transient tensor ~10×.

The module imports ``mlx.core`` and ``mlx_lm`` at top level. Non-Apple-
Silicon hosts cannot load this file, by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx_lm

from kv_cache_proposer.proposer import (
    BlockProposal,
    DLMProposer,
    ProposerConfig,
    ProposerStats,
)

from ._torch_bridge import mx_to_torch
from .env import require_environment


_BASE_MODEL_FOR_ARCHITECTURE = "Qwen/Qwen3-0.6B-Base"


def _load_dllm_safetensors_as_mx(repo_id: str) -> dict:
    """Download the dllm-hub checkpoint and load its safetensors into mx.array.

    The dllm-hub checkpoint ships its weights under the same key names
    as Qwen3-0.6B-Base (it's a fine-tune), so the resulting dict can be
    fed straight into ``mlx_lm`` Qwen3.Model.load_weights(...)``.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import torch

    snap_dir = snapshot_download(repo_id=repo_id)
    snap = Path(snap_dir)
    weight_files = sorted(snap.glob("*.safetensors"))
    if not weight_files:  # pragma: no cover - defensive: valid HF snapshot has safetensors
        raise RuntimeError(
            f"no .safetensors found in {snap_dir} for {repo_id!r}"
        )
    weights: dict = {}
    for wf in weight_files:
        with safe_open(str(wf), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key).detach().cpu()
                if tensor.dtype is torch.bfloat16:
                    weights[key] = mx.array(
                        tensor.float().numpy(),
                        dtype=mx.bfloat16,
                    )
                else:  # pragma: no cover - current dllm checkpoint stores bf16 tensors
                    weights[key] = mx.array(tensor.numpy())
    return weights


def _filter_weights_to_architecture(
    weights: dict, model_param_keys: set
) -> dict:
    """Drop any keys the MLX architecture doesn't have.

    The dllm-hub safetensors may include the head/extra weights that
    aren't part of the tied-embedding base model (e.g. an explicit
    ``lm_head.weight`` even when ``tie_word_embeddings=True``). We
    silently drop those — mlx_lm's ``Model.sanitize`` does the same
    for tied embeddings.
    """
    return {k: v for k, v in weights.items() if k in model_param_keys}


def _flat_param_keys(model) -> set:
    """Return the set of dotted-path parameter names the model expects."""
    from mlx.utils import tree_flatten

    return {k for k, _ in tree_flatten(model.parameters())}


class MLXSparseLogitsProposer(DLMProposer):
    """MLX drop-in replacement for SparseLogitsProposer.

    Construction:
        cfg = ProposerConfig(
            model_id="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
            dtype=torch.bfloat16, device="cpu",
        )
        proposer = MLXSparseLogitsProposer(cfg)

    The class subclasses DLMProposer to inherit the tokenizer wrapper
    and stats bookkeeping (the dllm-hub HF tokenizer is the source of
    truth for chat-template + mask_id; we don't need a separate MLX
    one). Only the model and `propose_block` are MLX-native.
    """

    def __init__(
        self,
        config: Optional[ProposerConfig] = None,
        *,
        compile_backbone: bool = True,
    ) -> None:
        """Construct the MLX proposer.

        Parameters
        ----------
        config
            ProposerConfig. ``model_id`` defaults to the dllm-hub
            checkpoint; ``device`` is ignored (MLX picks Metal).
        compile_backbone
            When True (default), the bidirectional backbone forward is
            wrapped in ``mx.compile``. mx.compile caches a graph per
            unique input shape, so the K-step diffusion loop pays JIT
            once per (T, batch=1) pair and amortizes across all
            subsequent steps with the same shape. Pass ``False`` to
            run the uncompiled path — used by tests to verify
            output-equivalence with the compiled path.
        """
        require_environment()
        # Skip DLMProposer.__init__: it would load the full HF PyTorch
        # model (1.5 GB) just to be discarded. Instead we set up only
        # the fields the speculative loop reads — tokenizer, mask_id,
        # pad_id, stats — and then build the MLX model from scratch.
        from transformers import AutoTokenizer

        self.config = config or ProposerConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:  # pragma: no cover - checkpoint contract
            raise RuntimeError(
                "Proposer tokenizer does not declare a mask_token_id; "
                "this is required for masked-diffusion denoising."
            )
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if self.pad_id is None:  # pragma: no cover - checkpoint contract
            raise RuntimeError("Tokenizer has neither pad nor eos token id.")

        # Build the MLX architecture (from Qwen3-0.6B-Base since the
        # dllm-hub checkpoint registers an unknown model_type and would
        # fail mlx_lm.load), then load the dllm-hub weights into it.
        base_model, _ = mlx_lm.load(_BASE_MODEL_FOR_ARCHITECTURE)
        weights = _load_dllm_safetensors_as_mx(self.config.model_id)
        weights = _filter_weights_to_architecture(
            weights, _flat_param_keys(base_model)
        )
        base_model.load_weights(list(weights.items()))
        mx.eval(base_model.parameters())
        self.model = base_model

        if not hasattr(self.model, "model"):  # pragma: no cover - mlx_lm Qwen3 contract
            raise RuntimeError(
                "loaded MLX Qwen3 model does not expose backbone as `.model`"
            )
        self._backbone = self.model.model
        if not hasattr(self._backbone, "embed_tokens"):  # pragma: no cover - mlx_lm Qwen3 contract
            raise RuntimeError(
                "MLX Qwen3 backbone has no `embed_tokens`; cannot apply "
                "tied-embedding head"
            )
        self._embed_tokens = self._backbone.embed_tokens

        from .verifier import _model_weight_bytes
        self.stats = ProposerStats(
            weight_bytes=_model_weight_bytes(self.model)
        )

        # Compile the bidirectional backbone if requested. The closure
        # captures `self._backbone` (an mlx.nn.Module). mx.compile traces
        # the function, recording each leaf tensor op against the
        # nn.Module's parameter mx.arrays — so the resulting compiled
        # graph executes purely as MLX kernels (no Python in the inner
        # loop, no per-call kernel-launch overhead).
        self._compile_backbone = compile_backbone
        self._backbone_forward_compiled = None
        if compile_backbone:
            backbone = self._backbone

            def _bidirectional_impl(x: "mx.array") -> "mx.array":
                h = backbone.embed_tokens(x)
                for layer in backbone.layers:
                    h = layer(h, None, None)
                return backbone.norm(h)

            self._backbone_forward_compiled = mx.compile(_bidirectional_impl)

    # ------------------------------------------------------------------ #
    # Bidirectional backbone forward
    # ------------------------------------------------------------------ #
    def _backbone_forward(self, x: "mx.array") -> "mx.array":
        """Run the Qwen3 backbone with a NULL (bidirectional) attention
        mask. Returns the last hidden state of shape [1, T, hidden].

        When constructed with ``compile_backbone=True`` (default), this
        dispatches to the ``mx.compile``-cached implementation. The
        compiled graph is keyed on input shape, so a stable
        (T_prompt + L_block) pair across diffusion steps reuses the
        same graph for K-1 of the K calls per block.
        """
        if self._backbone_forward_compiled is not None:
            return self._backbone_forward_compiled(x)
        # Uncompiled fallback path (used by the
        # output-equivalence test below, and as a debugging aid if
        # mx.compile interactions ever regress on a future mlx version).
        h = self._backbone.embed_tokens(x)
        for layer in self._backbone.layers:
            h = layer(h, None, None)
        return self._backbone.norm(h)

    # ------------------------------------------------------------------ #
    # Override: propose_block with sparse lm_head application
    # ------------------------------------------------------------------ #
    def propose_block(
        self,
        committed_token_ids: List[int],
        block_size: int,
        num_steps: int,
    ) -> BlockProposal:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if num_steps > block_size:
            num_steps = block_size

        prefix_len = len(committed_token_ids)
        total_len = prefix_len + block_size

        # Build the [1, T] input array, with mask_id at the new positions.
        seed = [self.pad_id] * total_len
        seed[:prefix_len] = list(committed_token_ids)
        seed[prefix_len:] = [self.mask_id] * block_size
        x = mx.array([seed], dtype=mx.int32)

        # Track the same front-loaded transfer schedule the parent uses
        # so the sparse path commits the same positions in the same
        # order as the dense (PyTorch) path.
        num_masked = block_size
        base = num_masked // num_steps
        remainder = num_masked % num_steps
        num_transfer = [
            base + (1 if i < remainder else 0) for i in range(num_steps)
        ]

        # mask_id_arr: a 1-D int array of length 1, used by `==` against x.
        mask_id_arr = mx.array(self.mask_id, dtype=mx.int32)

        peak_act_bytes = 0
        for step_idx in range(num_steps):
            # Recompute current block-mask each step (some positions are
            # unmasked as we proceed).
            block_mask_now = (x[0, prefix_len:total_len] == mask_id_arr)
            # `block_mask_now` is [block_size]; we need the indices into
            # the full T-sized x.
            n_masked_now = int(block_mask_now.sum().item())
            if n_masked_now == 0:  # pragma: no cover - schedule guarantees
                break

            # Indices of masked positions WITHIN THE BLOCK [0..block_size-1].
            # mlx 0.31.x does not support boolean indexing, so materialize this
            # tiny block mask through Python and build an integer index array.
            block_mask_list = [
                bool(v) for v in mx_to_torch(block_mask_now).tolist()
            ]
            mask_block_idx = mx.array(
                [i for i, is_masked in enumerate(block_mask_list) if is_masked],
                dtype=mx.int32,
            )
            # Convert to global positions within the full sequence
            mask_global_idx = mask_block_idx + prefix_len

            # Backbone forward (no causal mask)
            hidden = self._backbone_forward(x)  # [1, T, hidden]

            # Sparse lm_head: only at masked positions
            hidden_at_mask = hidden[:, mask_global_idx, :]  # [1, n_masked_now, hidden]
            mask_logits = self._embed_tokens.as_linear(hidden_at_mask)
            # Force evaluation before reading shapes / running argmax.
            mx.eval(mask_logits)
            peak_act_bytes = max(
                peak_act_bytes, int(mask_logits.size) * int(mask_logits.dtype.size)
            )

            # argmax + softmax confidence on the sparse slice
            mask_x0 = mx.argmax(mask_logits, axis=-1)  # [1, n_masked_now]
            mask_probs = mx.softmax(mask_logits.astype(mx.float32), axis=-1)
            # gather along last axis: equivalent to torch.gather
            mask_x0_probs = mx.take_along_axis(
                mask_probs, mask_x0[..., None], axis=-1
            ).squeeze(-1)  # [1, n_masked_now]

            k = int(num_transfer[step_idx])
            if k <= 0:  # pragma: no cover - num_steps clamp guarantees k >= 1
                continue
            k_eff = min(k, n_masked_now)

            # Top-k by confidence: get the top-k_eff highest-confidence
            # positions. mx.argpartition(-x, k) puts the largest k at the
            # front when given negative values; we need the top by value.
            # Equivalent to torch.topk: use argsort and take last k_eff.
            ranking = mx.argsort(mask_x0_probs[0])  # ascending
            top_idx_in_mask = ranking[-k_eff:]  # indices INTO mask_block_idx
            top_global_idx = mask_global_idx[top_idx_in_mask]
            top_tokens = mask_x0[0][top_idx_in_mask]

            # Scatter: x[0, top_global_idx] = top_tokens. mx.scatter_set
            # (a.k.a. indexed assignment) requires constructing a new
            # array since mx.array is immutable from a pure functional
            # standpoint. We use boolean masking to write at the
            # selected positions.
            #
            # Build a one-hot indicator of the global positions to write.
            # Shape [T]; True where this step should commit.
            full_indicator = mx.zeros(total_len, dtype=mx.bool_)
            full_indicator = _scatter_true(full_indicator, top_global_idx)
            # Build the "would-be-replaced" tokens at every position:
            # at each masked-in-block position, the argmax token; else
            # the existing x value.
            full_x0 = mx.array(seed, dtype=mx.int32)  # template (all pad/mask)
            # Place argmax at the global mask positions:
            full_x0 = _scatter_values(full_x0, mask_global_idx, mask_x0[0])
            # Write at top positions only.
            new_row = mx.where(full_indicator, full_x0, x[0])
            x = new_row[None, :]
            mx.eval(x)

        # If anything is still <mask>, fail loudly.
        if int((x[0, prefix_len:] == mask_id_arr).sum().item()) > 0:  # pragma: no cover - defensive
            leftover = int((x[0, prefix_len:] == mask_id_arr).sum().item())
            raise RuntimeError(
                f"MLXSparseLogitsProposer left {leftover} masked positions after "
                f"{num_steps} denoising steps; refusing to emit mask tokens."
            )

        block_tokens = [int(t) for t in mx_to_torch(x[0, prefix_len:]).tolist()]

        self.stats.total_blocks += 1
        self.stats.total_diffusion_steps += num_steps
        self.stats.total_forward_passes += num_steps
        self.stats.peak_activation_bytes = max(
            self.stats.peak_activation_bytes, peak_act_bytes
        )
        return BlockProposal(
            tokens=block_tokens,
            diffusion_steps=num_steps,
            forward_passes=num_steps,
            peak_activation_bytes=peak_act_bytes,
        )


def _scatter_true(indicator: "mx.array", positions: "mx.array") -> "mx.array":
    """Return an mx.array equal to `indicator` but with True at every
    index in `positions`. mx.array doesn't support in-place indexed
    assignment from a 1-D index array, so we build the new array via
    a one-hot reduction.
    """
    n = indicator.shape[0]
    # one-hot: shape [len(positions), n], one True per row at its index.
    eye = mx.arange(n) == positions[:, None]
    return indicator | mx.any(eye, axis=0)


def _scatter_values(
    base: "mx.array", positions: "mx.array", values: "mx.array"
) -> "mx.array":
    """Return `base` with `values[i]` written at `positions[i]`.

    Both `positions` and `values` are 1-D and have the same length;
    `base` is 1-D. Works by building a sparse contribution and adding
    it at the selected positions, with `where` to keep `base` elsewhere.
    """
    n = base.shape[0]
    indicator = mx.arange(n) == positions[:, None]  # [k, n]
    # which row contributed each position (rows are unique per position)
    chosen_row = mx.argmax(indicator.astype(mx.int32), axis=0)  # [n]
    any_hit = mx.any(indicator, axis=0)  # [n]
    overlay = values[chosen_row]  # [n]
    return mx.where(any_hit, overlay, base)
