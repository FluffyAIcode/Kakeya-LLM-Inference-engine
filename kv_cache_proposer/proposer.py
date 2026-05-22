"""DLM Proposer.

Wraps `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (an MDLM-style masked diffusion
language model with the same tokenizer as the Qwen3 family) as a *block proposer*
for speculative decoding.

Block-generation contract used by the speculative loop:

    propose_block(committed_token_ids, block_size, num_steps) -> List[int]

Internally this runs `num_steps` denoising iterations of masked-diffusion over
the L masked positions appended after the committed prefix, following the
"low-confidence remasking" schedule documented on the model card. The
implementation is a faithful port of the reference `generate()` function from
the model card, restricted to:

  * `temperature == 0.0`        -> deterministic greedy proposal
  * `cfg_scale == 0.0`          -> no classifier-free guidance branch
  * `remasking == "low_confidence"` -> top-confidence positions get committed first
  * batch size 1                -> single ongoing speculative session

This is a Proposer wrapper, not a fallback. There is no path that bypasses the
diffusion forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class ProposerConfig:
    model_id: str = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"


@dataclass
class BlockProposal:
    """One block proposed by the DLM."""

    tokens: List[int]
    """The L proposed token ids."""

    diffusion_steps: int
    """Number of denoising iterations actually run."""

    forward_passes: int
    """Number of full Proposer forward passes consumed (= diffusion_steps)."""

    peak_activation_bytes: int
    """Peak transient activation bytes during this block's diffusion."""


@dataclass
class ProposerStats:
    total_blocks: int = 0
    total_diffusion_steps: int = 0
    total_forward_passes: int = 0
    peak_activation_bytes: int = 0
    weight_bytes: int = 0


class DLMProposer:
    def __init__(self, config: Optional[ProposerConfig] = None) -> None:
        self.config = config or ProposerConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        # The dLLM checkpoint uses a custom `A2DQwen3LMHeadModel` head.
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.config.model_id,
            dtype=self.config.dtype,
            trust_remote_code=True,
        )
        self.model.to(self.config.device).eval()
        self.mask_id: int = self.tokenizer.mask_token_id
        if self.mask_id is None:
            raise RuntimeError(
                "Proposer tokenizer does not declare a mask_token_id; "
                "this is required for masked-diffusion denoising."
            )
        self.pad_id: int = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if self.pad_id is None:
            raise RuntimeError("Tokenizer has neither pad nor eos token id.")

        self.stats = ProposerStats(
            weight_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters())
        )

    # ------------------------------------------------------------------ #
    # Core: propose a block of `block_size` tokens conditioned on prefix #
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
            # The reference recipe distributes mask-unmask actions across steps.
            # Allow num_steps == block_size (one unmask per step) but no more.
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

        # Schedule: at step i we unmask `num_transfer_tokens[i]` masked
        # positions inside the block, picking the highest-confidence ones.
        block_mask_init = (positions >= block_start) & (positions < block_end)
        num_masked = int(block_mask_init.sum().item())
        if num_masked == 0:
            raise RuntimeError("Block initialization yielded zero masked tokens.")

        base = num_masked // num_steps
        remainder = num_masked % num_steps
        # Front-load the remainder, matching the reference implementation.
        num_transfer = [base + (1 if i < remainder else 0) for i in range(num_steps)]
        assert sum(num_transfer) == num_masked

        peak_act_bytes = 0
        for step_idx in range(num_steps):
            block_mask_now = (positions >= block_start) & (positions < block_end) & (x[0] == self.mask_id)
            if not bool(block_mask_now.any()):
                break

            outputs = self.model(x)
            logits = outputs.logits  # [1, T, V]

            # Track activation footprint (logits buffer is the dominant transient).
            peak_act_bytes = max(
                peak_act_bytes,
                logits.numel() * logits.element_size(),
            )

            x0 = torch.argmax(logits, dim=-1)  # [1, T]

            # "low_confidence" remasking: rank candidate positions by the
            # softmax probability of the picked token.
            probs = F.softmax(logits.float(), dim=-1)
            x0_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [1, T]

            confidence = torch.full_like(x0_probs, float("-inf"))
            confidence[0, block_mask_now] = x0_probs[0, block_mask_now]

            x_candidate = torch.where(block_mask_now.unsqueeze(0), x0, x)

            k = int(num_transfer[step_idx])
            if k <= 0:
                continue
            _, top_idx = torch.topk(confidence[0], k=k)
            transfer_mask = torch.zeros_like(x0, dtype=torch.bool)
            transfer_mask[0, top_idx] = True
            # Only commit positions that are still masked (intersection with block_mask_now).
            transfer_mask = transfer_mask & block_mask_now.unsqueeze(0)
            x = torch.where(transfer_mask, x_candidate, x)

        if bool((x[0, prefix_len:] == self.mask_id).any()):
            # Final clean-up: any leftover mask collapses to its argmax. This
            # path is only reachable if num_transfer underflows for numerical
            # reasons; we fail loudly rather than silently emit <mask>.
            leftover = (x[0, prefix_len:] == self.mask_id).sum().item()
            raise RuntimeError(
                f"DLM proposer left {leftover} masked positions after "
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

    # ----------------------------------------- #
    # Tokenizer helpers (delegated, no fallback) #
    # ----------------------------------------- #
    def encode_chat(self, messages: List[dict]) -> List[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        if not isinstance(ids, list):
            ids = list(ids)
        return ids
