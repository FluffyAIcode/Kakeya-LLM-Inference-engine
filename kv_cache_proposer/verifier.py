"""AR Verifier with sink+window KV cache.

The verifier is an autoregressive Qwen3 model. Its KV cache is bounded to
`sink_size + window_size` tokens at all times. After every model call we
*physically* slice each layer's K and V tensors to enforce that bound
(StreamingLLM-style attention sink + sliding window).

Important correctness notes:
  * `position_ids` for new tokens always use the **global** sequence position,
    so RoPE on new queries and new keys is rotated at their true distance.
    The surviving sink/window K vectors retain the RoPE rotation they had at
    their original positions — attention dot-products use the correct
    relative-position phase per surviving token.
  * `cache_position` tracks where in the *trimmed* cache the new K/V land,
    which is what the causal mask uses to enforce ``q_i attends k_{<= i}``
    inside the trimmed cache layout.
  * No fallback. If the cache layout becomes inconsistent we raise.
  * Greedy decoding (argmax) is used. With block_size+sink+window covering the
    full sequence, this is bit-equivalent to vanilla greedy AR — the
    speculative loop's correctness test relies on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass
class VerifierConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"
    sink_size: int = 4
    window_size: int = 64


@dataclass
class VerifierStats:
    forward_calls: int = 0
    tokens_consumed: int = 0
    peak_kv_bytes: int = 0
    weight_bytes: int = 0
    peak_activation_bytes: int = 0
    """Largest single-forward activation footprint observed.

    We approximate activation peak by the size of the logits buffer
    (`[B, T, V_vocab]`), which is the dominant transient tensor of a
    Qwen3 forward at long contexts. Other intermediates (Q, attn probs,
    MLP buffers) are smaller per layer and overlapping in lifetime."""


class SinkWindowVerifier:
    def __init__(self, config: Optional[VerifierConfig] = None) -> None:
        self.config = config or VerifierConfig()
        if self.config.sink_size < 0 or self.config.window_size <= 0:
            raise ValueError("sink_size must be >= 0 and window_size must be > 0")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            dtype=self.config.dtype,
        )
        self.model.to(self.config.device).eval()

        self.cache: Optional[DynamicCache] = None
        # Number of K/V slots currently in cache (after trimming).
        self.cache_logical_size: int = 0
        # Next global token position to emit (== number of tokens the verifier
        # has *seen*, counting both prompt and generated tokens).
        self.next_global_position: int = 0
        # Logits at `next_global_position` predicting the next token. Updated
        # after every forward pass.
        self.next_token_logits: Optional[torch.Tensor] = None

        self.stats = VerifierStats(
            weight_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters())
        )

    # ---------------------------- public API ---------------------------- #
    def reset(self) -> None:
        self.cache = DynamicCache(config=self.model.config)
        self.cache_logical_size = 0
        self.next_global_position = 0
        self.next_token_logits = None

    @torch.no_grad()
    def prefill(self, prompt_ids: List[int]) -> None:
        """Run the prompt through the verifier, then trim KV cache."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        device = self.config.device
        L = len(prompt_ids)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        cache_position = torch.arange(L, dtype=torch.long, device=device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self.cache_logical_size = L
        self.next_global_position = L
        self.next_token_logits = outputs.logits[0, -1].clone()

        self._record_peak_activation(outputs.logits)
        self._trim_cache_in_place()
        self._record_peak_kv()
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L

    @torch.no_grad()
    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        """Forward `tokens` through the verifier with the trimmed cache.

        Returns a [len(tokens), V] tensor of next-token logits, where
        ``logits[i]`` is the verifier's prediction for the token *after*
        ``tokens[i]``. K/V for these tokens is appended to the cache
        (subject to subsequent trimming via :meth:`commit_or_truncate`).
        """
        if self.cache is None:
            raise RuntimeError("Verifier not prefilled.")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        device = self.config.device
        L = len(tokens)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        global_start = self.next_global_position
        position_ids = torch.arange(
            global_start, global_start + L, dtype=torch.long, device=device
        ).unsqueeze(0)
        cache_start = self.cache_logical_size
        cache_position = torch.arange(
            cache_start, cache_start + L, dtype=torch.long, device=device
        )

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        # Cache provisionally has cache_start + L slots until commit/truncate.
        self.cache_logical_size = cache_start + L
        self._record_peak_activation(outputs.logits)
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        # Don't trim yet — caller decides how many tokens were accepted.
        return outputs.logits[0].clone()  # [L, V]

    def commit_or_truncate(
        self,
        forwarded: int,
        accepted: int,
    ) -> None:
        """Reconcile cache state after a verification pass.

        ``forwarded`` is the number of tokens passed to :meth:`forward_block`.
        ``accepted`` is how many of those tokens speculative decoding kept.
        The unaccepted tail (``forwarded - accepted``) is sliced out of the
        cache; ``next_global_position`` is advanced by ``accepted``. The cache
        is then trimmed to sink+window.
        """
        if accepted < 0 or accepted > forwarded:
            raise ValueError("accepted must satisfy 0 <= accepted <= forwarded")
        drop = forwarded - accepted
        if drop > 0:
            self._truncate_tail_in_place(drop)
        self.cache_logical_size -= drop
        self.next_global_position += accepted
        self._trim_cache_in_place()
        self._record_peak_kv()

    @torch.no_grad()
    def append_token(self, token_id: int) -> torch.Tensor:
        """Forward a single token (e.g., correction or bonus) into the cache.

        Returns the logits predicting the token *after* `token_id`.
        """
        logits = self.forward_block([token_id])
        # commit it: this single token is accepted
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1].clone()
        return self.next_token_logits

    # ------------------------- internal helpers ------------------------- #
    def _budget(self) -> int:
        return self.config.sink_size + self.config.window_size

    def _trim_cache_in_place(self) -> None:
        if self.cache is None:
            raise RuntimeError("No cache to trim.")
        budget = self._budget()
        if self.cache_logical_size <= budget:
            return
        sink = self.config.sink_size
        keep_window = budget - sink

        for layer in self.cache.layers:
            keys: torch.Tensor = layer.keys
            values: torch.Tensor = layer.values
            if keys is None or values is None:
                continue
            if keys.shape[2] != self.cache_logical_size:
                raise RuntimeError(
                    f"Cache shape {keys.shape} inconsistent with logical "
                    f"size {self.cache_logical_size}; layout invariant violated."
                )
            sink_k = keys[:, :, :sink, :]
            sink_v = values[:, :, :sink, :]
            tail_k = keys[:, :, -keep_window:, :]
            tail_v = values[:, :, -keep_window:, :]
            # Allocate fresh contiguous tensors so the originals are GC'd —
            # otherwise CPython retains the trimmed slices' parent storage and
            # peak_kv_bytes would over-report.
            layer.keys = torch.cat([sink_k, tail_k], dim=2).contiguous()
            layer.values = torch.cat([sink_v, tail_v], dim=2).contiguous()
        self.cache_logical_size = budget

    def _truncate_tail_in_place(self, drop: int) -> None:
        if drop <= 0:
            return
        if self.cache is None:
            raise RuntimeError("No cache to truncate.")
        if drop > self.cache_logical_size:
            raise RuntimeError(
                f"Cannot drop {drop} tokens from cache of size {self.cache_logical_size}"
            )
        keep = self.cache_logical_size - drop
        for layer in self.cache.layers:
            keys: torch.Tensor = layer.keys
            values: torch.Tensor = layer.values
            if keys is None or values is None:
                continue
            layer.keys = keys[:, :, :keep, :].contiguous()
            layer.values = values[:, :, :keep, :].contiguous()

    def _record_peak_kv(self) -> None:
        if self.cache is None:
            return
        total = 0
        for layer in self.cache.layers:
            if layer.keys is not None:
                total += layer.keys.numel() * layer.keys.element_size()
            if layer.values is not None:
                total += layer.values.numel() * layer.values.element_size()
        self.stats.peak_kv_bytes = max(self.stats.peak_kv_bytes, total)

    def _record_peak_activation(self, logits: torch.Tensor) -> None:
        bytes_ = int(logits.numel() * logits.element_size())
        if bytes_ > self.stats.peak_activation_bytes:
            self.stats.peak_activation_bytes = bytes_
