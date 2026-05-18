"""Baseline: vanilla AR greedy decode with full KV cache.

Runs the same verifier model with an *unbounded* DynamicCache. Used to:

  1. Establish the per-token KV byte baseline that we are compressing
     against (dense, never trimmed).
  2. Provide ground-truth token sequences to verify that the speculative
     loop's output is bit-identical (under greedy decoding the speculative
     loop must produce exactly the same tokens as this baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass
class BaselineConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"


@dataclass
class BaselineRunResult:
    output_token_ids: List[int]
    forward_calls: int
    tokens_consumed: int
    peak_kv_bytes: int
    final_kv_bytes: int
    weight_bytes: int
    final_kv_token_count: int


class BaselineDecoder:
    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        self.config = config or BaselineConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            dtype=self.config.dtype,
        )
        self.model.to(self.config.device).eval()
        self._weight_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: Optional[List[int]] = None,
    ) -> BaselineRunResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        device = self.config.device
        cache = DynamicCache(config=self.model.config)
        L = len(prompt_ids)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        cache_position = torch.arange(L, dtype=torch.long, device=device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values
        forward_calls = 1
        tokens_consumed = L
        next_logits = outputs.logits[0, -1]
        peak_kv = self._kv_bytes(cache)

        eos_set = set(eos_token_ids or [])
        generated: List[int] = []
        cache_len = L
        for _ in range(max_new_tokens):
            tok = int(torch.argmax(next_logits).item())
            generated.append(tok)
            if tok in eos_set:
                break
            inp = torch.tensor([[tok]], dtype=torch.long, device=device)
            pos = torch.tensor([[cache_len]], dtype=torch.long, device=device)
            cpos = torch.tensor([cache_len], dtype=torch.long, device=device)
            outputs = self.model(
                input_ids=inp,
                position_ids=pos,
                cache_position=cpos,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            cache_len += 1
            forward_calls += 1
            tokens_consumed += 1
            next_logits = outputs.logits[0, -1]
            peak_kv = max(peak_kv, self._kv_bytes(cache))

        return BaselineRunResult(
            output_token_ids=generated,
            forward_calls=forward_calls,
            tokens_consumed=tokens_consumed,
            peak_kv_bytes=peak_kv,
            final_kv_bytes=self._kv_bytes(cache),
            weight_bytes=self._weight_bytes,
            final_kv_token_count=cache_len,
        )

    @staticmethod
    def _kv_bytes(cache: DynamicCache) -> int:
        total = 0
        for layer in cache.layers:
            if layer.keys is not None:
                total += layer.keys.numel() * layer.keys.element_size()
            if layer.values is not None:
                total += layer.values.numel() * layer.values.element_size()
        return total
