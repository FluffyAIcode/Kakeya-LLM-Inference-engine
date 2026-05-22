"""Sparse-logits DLM proposer.

The vanilla `DLMProposer` computes a full ``[1, T, V]`` logits tensor on
every diffusion step. For Qwen3-0.6B-MDLM with V=151_936 and a typical
prompt of 50–100 tokens, the bf16 logits buffer alone is 15–30 MB, and
the subsequent ``F.softmax(... .float())`` doubles that. Across K
diffusion steps × ~20 blocks per generation, this is the dominant
contributor to wall-time AND peak activation on CPU/MLX backends.

Yet the algorithm only ever looks at logits at the **masked** positions
of the current block (selecting top-confidence positions to unmask, and
the argmax tokens at those positions). All work done at non-masked
positions is wasted.

This module replaces the dense forward with a two-step pattern:

    1. Run the model **backbone** to get last-layer hidden states
       ``[1, T, d]``  (this part cannot be sparsified — bidirectional
       attention requires every position).
    2. Apply the language-model head **only at the masked positions**,
       producing ``[1, n_masked, V]`` instead of ``[1, T, V]``.

For typical (T=84, n_masked=8) workloads this trims the logits buffer
~10×, reduces lm_head FLOPs by the same factor, and avoids the
bf16→fp32 softmax tax on T−n_masked positions.

The output token sequence is **identical** to the dense path under
greedy temperature-0 decoding (verified by tests on real Qwen3
weights — see `tests/inference_engine/proposer/test_sparse_logits.py`).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from kv_cache_proposer.proposer import (
    BlockProposal,
    DLMProposer,
    ProposerConfig,
)


class SparseLogitsProposer(DLMProposer):
    """DLM proposer that computes lm_head only at masked positions.

    Constructor signature is identical to :class:`DLMProposer`. Only
    :meth:`propose_block` is overridden; all other behavior (tokenizer,
    encode_chat, stats, validation) is inherited unchanged.

    The class deliberately re-implements ``propose_block`` rather than
    calling super and patching, because the dense version's local
    computation graph (full-sequence argmax, gather, confidence) cannot
    be cleanly intercepted from the outside without either monkey-
    patching torch or duplicating the loop. We accept the duplication
    in exchange for clarity; the diff against the parent is small and
    test-covered (token-equivalence is enforced).
    """

    def __init__(self, config: Optional[ProposerConfig] = None) -> None:
        super().__init__(config)
        # The dllm-hub `A2DQwen3LMHeadModel` exposes the backbone as
        # `.model` and the head as `.lm_head`. We resolve them once at
        # init and fail loudly if the upstream model factors them
        # differently — there is no fallback to the dense path.
        if not hasattr(self.model, "model"):
            raise RuntimeError(
                "SparseLogitsProposer requires the proposer model to "
                "expose its backbone as `.model`; this build's model is "
                f"a {type(self.model).__name__} which does not."
            )
        if not hasattr(self.model, "lm_head"):
            raise RuntimeError(
                "SparseLogitsProposer requires the proposer model to "
                "expose its language-model head as `.lm_head`; this "
                f"build's model is a {type(self.model).__name__} which "
                "does not."
            )
        self._backbone = self.model.model
        self._lm_head = self.model.lm_head

    # ------------------------------------------------------------------ #
    # Override: propose_block with sparse lm_head application            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
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

        device = self.config.device
        prefix_len = len(committed_token_ids)
        total_len = prefix_len + block_size

        x = torch.full((1, total_len), self.pad_id, dtype=torch.long, device=device)
        x[0, :prefix_len] = torch.tensor(committed_token_ids, dtype=torch.long, device=device)
        x[0, prefix_len:] = self.mask_id

        positions = torch.arange(total_len, device=device)
        block_start = prefix_len
        block_end = total_len

        # Same front-loaded transfer schedule as the dense parent so
        # the two paths produce identical commit orders.
        block_mask_init = (positions >= block_start) & (positions < block_end)
        num_masked = int(block_mask_init.sum().item())
        if num_masked == 0:  # pragma: no cover - block_size > 0 guarantees masks exist
            raise RuntimeError("Block initialization yielded zero masked tokens.")
        base = num_masked // num_steps
        remainder = num_masked % num_steps
        num_transfer = [base + (1 if i < remainder else 0) for i in range(num_steps)]
        assert sum(num_transfer) == num_masked

        peak_act_bytes = 0
        for step_idx in range(num_steps):
            block_mask_now = (
                (positions >= block_start)
                & (positions < block_end)
                & (x[0] == self.mask_id)
            )
            if not bool(block_mask_now.any()):  # pragma: no cover - schedule guarantees
                break

            mask_indices = torch.nonzero(block_mask_now, as_tuple=False).squeeze(-1)
            #          ^ shape [n_masked_now], values are absolute positions in [0, T)

            # ----- backbone forward (full T) ----- #
            backbone_out = self._backbone(x)
            hidden = backbone_out.last_hidden_state  # [1, T, d]

            # ----- sparse lm_head (only at mask positions) ----- #
            # Slicing along seq_len is a zero-copy view; lm_head sees a
            # [1, n_masked_now, d] tensor instead of [1, T, d].
            hidden_at_mask = hidden[:, mask_indices, :]
            mask_logits = self._lm_head(hidden_at_mask)  # [1, n_masked_now, V]

            # The logits buffer is the dominant transient; we record its
            # peak as the proposer's activation peak for parity with the
            # dense path's bookkeeping.
            peak_act_bytes = max(
                peak_act_bytes,
                mask_logits.numel() * mask_logits.element_size(),
            )

            # ----- argmax + low-confidence remasking on the sparse slice ----- #
            mask_x0 = torch.argmax(mask_logits, dim=-1)  # [1, n_masked_now]
            mask_probs = F.softmax(mask_logits.float(), dim=-1)
            mask_x0_probs = mask_probs.gather(-1, mask_x0.unsqueeze(-1)).squeeze(-1)
            #         shape: [1, n_masked_now]

            k = int(num_transfer[step_idx])
            if k <= 0:  # pragma: no cover - num_steps clamp guarantees k >= 1
                continue
            # Among currently-masked positions, pick the k highest-confidence ones
            # to commit in this step. `topk` operates on the sparse vector, then
            # we map back to absolute positions via `mask_indices`.
            k_eff = min(k, mask_x0_probs.shape[1])
            _, top_idx = torch.topk(mask_x0_probs[0], k=k_eff)
            top_positions_global = mask_indices[top_idx]
            top_tokens = mask_x0[0, top_idx]
            x[0, top_positions_global] = top_tokens

        if bool((x[0, prefix_len:] == self.mask_id).any()):  # pragma: no cover - defensive
            leftover = (x[0, prefix_len:] == self.mask_id).sum().item()
            raise RuntimeError(
                f"SparseLogitsProposer left {leftover} masked positions after "
                f"{num_steps} denoising steps; refusing to emit mask tokens."
            )

        block_tokens = x[0, prefix_len:].tolist()

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
