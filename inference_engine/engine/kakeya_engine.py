"""Kakeya Inference Engine runtime — chunked restoration prefill + bounded-KV decode.

Implements the v1 core of `docs/design/kakeya-inference-engine-architecture.md`:
the **NativeHybridBounded** restoration policy (§5) for hybrid-attention models
(e.g. Gemma-4), with **chunked restoration prefill** (§4) and **bounded-KV
decode** (§6). On a hybrid model the recall-critical full-attention layers are
kept exact while the remaining layers are bounded to ``sink + window`` — the
model's native sliding mechanism *is* the restoration, so no reconstruction
forward is needed. Prefill is consumed in fixed token blocks so the attention
mask / activations scale with the block size, never with O(T²); the
sliding-window-aware cache keeps the resident KV bounded.

``torch`` / ``transformers`` are imported lazily so the admission math
(`inference_engine.engine.admission`) stays importable on any host.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from inference_engine.engine.admission import (
    BoundedKVModel,
    exact_layer_indices_for_layer_types,
    max_concurrent_sessions,
)


class KakeyaEngine:
    """Bounded-KV-native inference engine (v1 core, NativeHybridBounded policy)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        sink: int = 4,
        window: int = 64,
        chunk_size: int = 2048,
        policy: str = "native-hybrid",
    ) -> None:
        if policy != "native-hybrid":
            raise NotImplementedError(
                "v1 ships the NativeHybridBounded policy; FThetaRestored "
                "(full-attention models) is design §5 / v1.2.")
        self.model = model
        self.tok = tokenizer
        self.sink = sink
        self.window = window
        self.chunk_size = chunk_size
        self.policy = policy

        tc = model.config.get_text_config()
        self._native_sliding_window = getattr(tc, "sliding_window", None)
        # Bounded-KV invariant (§2): non-exact (sliding) layers resident to
        # sink+window; exact (full-attention) layers stay full.
        bounded = sink + window
        tc.sliding_window = bounded
        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = bounded

        self.layer_types = list(getattr(tc, "layer_types", []) or [])
        self.exact_layer_indices = (
            exact_layer_indices_for_layer_types(self.layer_types)
            if self.layer_types else [])
        self.num_layers = int(getattr(tc, "num_hidden_layers"))
        self.num_kv_heads = int(getattr(tc, "num_key_value_heads"))
        self.head_dim = int(getattr(tc, "head_dim"))

    # ------------------------------------------------------------------ #

    def bounded_kv_model(self, dtype_bytes: int = 2) -> BoundedKVModel:
        """The per-session bounded-KV cost model (§2) for admission (§7)."""
        return BoundedKVModel(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            n_exact_layers=len(self.exact_layer_indices) or self.num_layers,
            sink=self.sink,
            window=self.window,
            dtype_bytes=dtype_bytes,
        )

    def max_concurrent(self, *, memory_budget_bytes: int, model_weight_bytes: int,
                       context_len: int, dtype_bytes: int = 2) -> int:
        """Peak-window admission ceiling (§7) at a given context length."""
        per_session = self.bounded_kv_model(dtype_bytes).resident_bytes(context_len)
        return max_concurrent_sessions(
            memory_budget_bytes=memory_budget_bytes,
            model_weight_bytes=model_weight_bytes,
            per_session_bytes=per_session,
        )

    # ------------------------------------------------------------------ #

    def generate_cohort(
        self,
        prompt_ids_2d: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        device: Optional[Any] = None,
    ) -> List[List[int]]:
        """Chunked restoration prefill (§4) + bounded-KV decode (§6) for a cohort.

        Equal-length prompts (a served cohort) are decoded as one batch; prefill
        is chunked via ``prefill_chunk_size`` so mask/activation memory scales
        with ``chunk_size`` not the prompt length, and the sliding-window-aware
        cache bounds the resident KV.
        """
        import torch

        if device is None:
            device = next(self.model.parameters()).device
        ids = torch.tensor([list(p) for p in prompt_ids_2d], device=device)
        T = ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                prefill_chunk_size=min(self.chunk_size, T),
            )
        return [out[i, T:].tolist() for i in range(ids.shape[0])]
