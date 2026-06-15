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
        evicting_cache: bool = True,
    ) -> List[List[int]]:
        """Chunked restoration prefill (§4) + bounded-KV decode (§6) for a cohort.

        Equal-length prompts (a served cohort) are decoded as one batch; prefill
        is chunked via ``prefill_chunk_size`` so mask/activation memory scales
        with ``chunk_size`` not the prompt length.

        ``evicting_cache`` (KIE-v1.1) realizes the bounded-KV bound at runtime:
        it builds a hybrid-aware ``StaticCache`` (sliding layers capped at
        ``sink+window``, full-attention layers exact) and passes the **cache
        object** to ``generate``. This evicts the sliding-layer KV (so resident
        memory is the bounded ~`resident_bytes`, not full KV) while avoiding
        ``cache_implementation="static"``, which triggers a CUDA-graph compile
        that segfaults with this model + chunked prefill. With it False the
        default growing ``DynamicCache`` is used (KV stored full; window applied
        only in the attention mask).
        """
        import torch
        from transformers import StaticCache

        if device is None:
            device = next(self.model.parameters()).device
        ids = torch.tensor([list(p) for p in prompt_ids_2d], device=device)
        N, T = ids.shape
        kwargs: dict = dict(
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            prefill_chunk_size=min(self.chunk_size, T),
        )
        if evicting_cache:
            # KIE-v1.1: graph capture OFF. A static cache makes `generate`
            # torch.compile the decode step (triton/inductor → CUDA-graph),
            # which segfaults with this model + chunked prefill (and writes a
            # .so that fails to load on noexec tmp). Run the evicting cache in
            # eager mode instead — correct + bounded, just ungraphed.
            torch._dynamo.config.disable = True
            kwargs["past_key_values"] = StaticCache(
                config=self.model.config.get_text_config(),
                max_cache_len=T + max_new_tokens, max_batch_size=N,
                device=device, dtype=self.model.dtype,
            )
        with torch.no_grad():
            out = self.model.generate(ids, **kwargs)
        return [out[i, T:].tolist() for i in range(N)]

    # ------------------------------------------------------------------ #
    # Decoupled prefill / decode (§4 + §6): prefill each session into its
    # bounded cache (transient = ONE session), then stack the cohort and
    # batch-decode — so the prefill working set is never held simultaneously
    # for all N sessions (the gate that caps batched generate). This is what
    # lets admitted concurrency approach the peak-window ceiling (§7).
    # ------------------------------------------------------------------ #

    def _new_static_cache(self, *, max_cache_len: int, batch: int, device: Any):
        from transformers import StaticCache
        return StaticCache(
            config=self.model.config.get_text_config(),
            max_cache_len=max_cache_len, max_batch_size=batch,
            device=device, dtype=self.model.dtype,
        )

    def prefill_session(self, prompt_ids, *, max_new_tokens, device=None):
        """Chunked prefill ONE session into its bounded StaticCache (eager).
        Returns (cache, last_logits[1,V], T)."""
        import torch

        torch._dynamo.config.disable = True
        if device is None:
            device = next(self.model.parameters()).device
        ids = torch.tensor([list(prompt_ids)], device=device)
        T = ids.shape[1]
        cache = self._new_static_cache(max_cache_len=T + max_new_tokens,
                                       batch=1, device=device)
        out = None
        with torch.no_grad():
            for s in range(0, T, self.chunk_size):
                e = min(s + self.chunk_size, T)
                out = self.model(
                    input_ids=ids[:, s:e], past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(s, e, device=device),
                    logits_to_keep=1 if e == T else 0,
                )
        return cache, out.logits[:, -1, :], T

    def _init_batched_layer(self, bl, ref, *, N, device):
        """Allocate a batched layer [N, …] from a prefilled single-session ref
        layer and copy the meta the lazy batched layer hasn't set."""
        import torch

        shp = list(ref.keys.shape)
        shp[0] = N
        bl.keys = torch.zeros(shp, dtype=ref.keys.dtype, device=device)
        bl.values = torch.zeros(shp, dtype=ref.values.dtype, device=device)
        bl.is_initialized = True
        bl.max_batch_size = N
        for attr in ("dtype", "device", "max_cache_len", "num_heads",
                     "v_head_dim", "k_head_dim", "cumulative_length_int"):
            if hasattr(ref, attr):
                setattr(bl, attr, getattr(ref, attr))
        if hasattr(ref, "cumulative_length"):
            cl = ref.cumulative_length  # lockstep scalar (all rows same T)
            bl.cumulative_length = cl.clone() if hasattr(cl, "clone") else cl

    def decode_cohort(self, prompt_ids_2d, *, max_new_tokens, device=None):
        """Decoupled serve: per-session prefill → row-copy into one batched
        bounded cache → batched bounded decode.

        Memory-efficient stacking: the batched cache is allocated once and each
        session's bounded cache is copied into its row then freed, so the peak is
        ``batched (N · resident) + one session`` — the prefill transient is never
        held for all N at once (the gate that caps batched generate). Resident
        memory at decode is the peak-window admission regime (§7).
        """
        import torch

        torch._dynamo.config.disable = True
        if device is None:
            device = next(self.model.parameters()).device
        N = len(prompt_ids_2d)

        # Session 0 sets the reference shapes for the batched cache.
        c0, last0, T = self.prefill_session(prompt_ids_2d[0],
                                            max_new_tokens=max_new_tokens,
                                            device=device)
        batched = self._new_static_cache(max_cache_len=T + max_new_tokens,
                                         batch=N, device=device)
        for li, bl in enumerate(batched.layers):
            self._init_batched_layer(bl, c0.layers[li], N=N, device=device)
            bl.keys[0].copy_(c0.layers[li].keys[0])
            bl.values[0].copy_(c0.layers[li].values[0])
        first_tokens = [int(last0.argmax(-1)[0])]
        del c0

        for i in range(1, N):
            ci, lasti, _ = self.prefill_session(prompt_ids_2d[i],
                                                max_new_tokens=max_new_tokens,
                                                device=device)
            for li, bl in enumerate(batched.layers):
                bl.keys[i].copy_(ci.layers[li].keys[0])
                bl.values[i].copy_(ci.layers[li].values[0])
            first_tokens.append(int(lasti.argmax(-1)[0]))
            del ci

        gen = [[first_tokens[i]] for i in range(N)]
        nxt = torch.tensor(first_tokens, device=device)
        with torch.no_grad():
            for step in range(max_new_tokens - 1):
                out = self.model(
                    input_ids=nxt.view(N, 1), past_key_values=batched,
                    use_cache=True,
                    cache_position=torch.tensor([T + step], device=device),
                    logits_to_keep=1,
                )
                nxt = out.logits[:, -1, :].argmax(-1)
                for i in range(N):
                    gen[i].append(int(nxt[i]))
        return gen
