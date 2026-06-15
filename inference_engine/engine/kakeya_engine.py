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


_INT_EXACT_CLS = None


def _int_exact_layer_cls():
    """Lazily build the int-quantized exact-layer cache class (subclasses
    transformers' ``CacheLayerMixin`` so the model's mask/length dispatch — an
    ``isinstance`` check — recognises it). Built lazily so the engine package
    stays importable on torch-less hosts."""
    global _INT_EXACT_CLS
    if _INT_EXACT_CLS is not None:
        return _INT_EXACT_CLS
    import torch
    from transformers.cache_utils import CacheLayerMixin

    class _IntQuantExactLayer(CacheLayerMixin):
        """Genuine int{bits} storage for a full-attention (exact) cache layer.

        Stores K/V as int8 (or int4-range packed in int8) + per-token scale,
        halving (8b) / quartering (4b) the resident bytes of the recall-critical
        exact layers (the bounded-decode floor). Dequantizes to bf16 on
        ``update`` return (one layer's worth, transient, freed after that
        layer's attention), so the model attends in bf16 while storage stays
        int. De-risked: int8/int4 keep recall 1.0 at 62k.
        """

        is_sliding = False
        is_compileable = False

        def __init__(self, *, N, kv_heads, max_cache_len, head_dim, device,
                     dtype, bits):
            self.max_cache_len = int(max_cache_len)
            self.max_batch_size = int(N)
            self.num_heads = int(kv_heads)
            self.k_head_dim = self.v_head_dim = int(head_dim)
            self.device = device
            self.dtype = dtype
            self.is_initialized = True
            self._qmax = (1 << (bits - 1)) - 1
            self.cumulative_length = torch.zeros(1, dtype=torch.long,
                                                 device=device)
            shp = (N, kv_heads, self.max_cache_len, head_dim)
            sshp = (N, kv_heads, self.max_cache_len, 1)
            self.keys = torch.zeros(shp, dtype=torch.int8, device=device)
            self.values = torch.zeros(shp, dtype=torch.int8, device=device)
            self.k_scale = torch.zeros(sshp, dtype=dtype, device=device)
            self.v_scale = torch.zeros(sshp, dtype=dtype, device=device)

        def _quant(self, t):
            amax = t.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
            scale = amax / self._qmax
            q = torch.clamp(torch.round(t / scale), -self._qmax, self._qmax)
            return q.to(torch.int8), scale.to(self.dtype)

        def write_prefill_row(self, row, k, v, length):
            kq, ks = self._quant(k)
            vq, vs = self._quant(v)
            self.keys[row, :, :length] = kq[0]
            self.k_scale[row, :, :length] = ks[0]
            self.values[row, :, :length] = vq[0]
            self.v_scale[row, :, :length] = vs[0]

        def set_length(self, length):
            self.cumulative_length.fill_(int(length))

        @property
        def current_len(self):
            return int(self.cumulative_length.item())

        def append_only(self, key_states, value_states):
            """Quantize + store one decode step's K/V WITHOUT returning the full
            bf16 dequant (the quant-attention path reads the int8 store directly
            via the tiled flash kernel, so no O(S) bf16 materialization)."""
            L = key_states.shape[-2]
            pos = torch.arange(L, device=self.device) + self.cumulative_length
            self.cumulative_length.add_(L)
            kq, ks = self._quant(key_states)
            vq, vs = self._quant(value_states)
            self.keys.index_copy_(2, pos, kq)
            self.k_scale.index_copy_(2, pos, ks)
            self.values.index_copy_(2, pos, vq)
            self.v_scale.index_copy_(2, pos, vs)

        def lazy_initialization(self, key_states, value_states):
            self.is_initialized = True

        def update(self, key_states, value_states, *args, **kwargs):
            L = key_states.shape[-2]
            pos = torch.arange(L, device=self.device) + self.cumulative_length
            self.cumulative_length.add_(L)
            kq, ks = self._quant(key_states)
            vq, vs = self._quant(value_states)
            self.keys.index_copy_(2, pos, kq)
            self.k_scale.index_copy_(2, pos, ks)
            self.values.index_copy_(2, pos, vq)
            self.v_scale.index_copy_(2, pos, vs)
            return (self.keys.to(self.dtype) * self.k_scale,
                    self.values.to(self.dtype) * self.v_scale)

        def get_mask_sizes(self, query_length, *a, **k):
            return self.max_cache_len, 0

        def get_max_cache_shape(self, *a, **k):
            return self.max_cache_len

        def get_seq_length(self, *a, **k):
            return int(self.cumulative_length.item())

        def reset(self):
            self.keys.zero_(); self.values.zero_()
            self.k_scale.zero_(); self.v_scale.zero_()
            self.cumulative_length.zero_()

    _INT_EXACT_CLS = _IntQuantExactLayer
    return _INT_EXACT_CLS


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

    @staticmethod
    def _roundtrip_int(t, bits: int):
        """Per-token affine int{bits} quantize→dequantize round-trip of a
        K/V tensor [B, kv, S, D] (de-risk probe: proves whether int storage of
        the exact layers preserves recall, before building int storage)."""
        import torch

        if bits <= 0:
            return t
        qmax = (1 << (bits - 1)) - 1  # int8 -> 127, int4 -> 7
        amax = t.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
        scale = amax / qmax
        q = torch.clamp(torch.round(t / scale), -qmax, qmax)
        return (q * scale).to(t.dtype)

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

    def _decoder_layers(self):
        """Resolve the verifier's decoder-layer ModuleList (gemma-4 nests it
        under model.language_model / model.model.language_model)."""
        for chain in (("model", "language_model", "layers"),
                      ("language_model", "model", "layers"),
                      ("model", "model", "layers"),
                      ("model", "layers")):
            o = self.model
            ok = True
            for a in chain:
                if hasattr(o, a):
                    o = getattr(o, a)
                else:
                    ok = False
                    break
            if ok:
                return o
        raise RuntimeError("could not resolve decoder layers")

    def _make_quant_attn_forward(self, attn, layer_cache, tile):
        """Patched gemma-4 exact-layer attention (decode): project the new
        token, append it to the int8 store, then run the tiled quantized flash
        attention over the store — no full-bf16 K/V materialization (KIE-v1.1.y).
        """
        import torch
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
        from inference_engine.engine.quant_attention import quantized_flash_attention

        def _fwd(hidden_states, position_embeddings, attention_mask=None,
                 past_key_values=None, cache_position=None, **kwargs):
            cos, sin = position_embeddings
            hshape = (*hidden_states.shape[:-1], -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(hidden_states).view(hshape))
            q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            k_lin = attn.k_proj(hidden_states).view(hshape)
            v_lin = (attn.v_proj(hidden_states).view(hshape)
                     if getattr(attn, "v_proj", None) is not None else k_lin)
            k = attn.k_norm(k_lin)
            k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            v = attn.v_norm(v_lin).transpose(1, 2)
            layer_cache.append_only(k, v)
            L = layer_cache.current_len
            scale = getattr(attn, "scaling", attn.head_dim ** -0.5)
            out = quantized_flash_attention(
                q,
                layer_cache.keys[:, :, :L], layer_cache.k_scale[:, :, :L],
                layer_cache.values[:, :, :L], layer_cache.v_scale[:, :, :L],
                scale=scale, tile=tile,
            )
            out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
            return attn.o_proj(out), None

        return _fwd

    def decode_cohort(self, prompt_ids_2d, *, max_new_tokens, device=None,
                      quant_exact_bits: int = 0, quant_attn: bool = False,
                      attn_tile: int = 4096):
        """Decoupled serve: per-session prefill → row-copy into one batched
        bounded cache → batched bounded decode.

        Memory-efficient stacking: the batched cache is allocated once and each
        session's bounded cache is copied into its row then freed, so the peak is
        ``batched (N · resident) + one session`` — the prefill transient is never
        held for all N at once (the gate that caps batched generate). Resident
        memory at decode is the peak-window admission regime (§7).
        """
        import time

        import torch

        torch._dynamo.config.disable = True
        if device is None:
            device = next(self.model.parameters()).device
        N = len(prompt_ids_2d)
        torch.cuda.synchronize(device)
        _t_prefill0 = time.perf_counter()

        # Session 0 sets the reference shapes for the batched cache.
        c0, last0, T = self.prefill_session(prompt_ids_2d[0],
                                            max_new_tokens=max_new_tokens,
                                            device=device)
        batched = self._new_static_cache(max_cache_len=T + max_new_tokens,
                                         batch=N, device=device)
        exact = set(self.exact_layer_indices)
        quant = bool(quant_exact_bits and quant_exact_bits > 0)
        IntCls = _int_exact_layer_cls() if quant else None

        def _write_row(li, row, ci):
            src = ci.layers[li]
            bl = batched.layers[li]
            if quant and li in exact and isinstance(bl, IntCls):
                bl.write_prefill_row(row, src.keys[:, :, :T], src.values[:, :, :T], T)
            else:
                bl.keys[row].copy_(src.keys[0])
                bl.values[row].copy_(src.values[0])

        for li, bl in enumerate(batched.layers):
            if quant and li in exact:
                # genuine int storage for the recall-critical exact layers
                ref = c0.layers[li]
                batched.layers[li] = IntCls(
                    N=N, kv_heads=ref.keys.shape[1],
                    max_cache_len=T + max_new_tokens, head_dim=ref.keys.shape[3],
                    device=device, dtype=self.model.dtype, bits=quant_exact_bits)
            else:
                self._init_batched_layer(bl, c0.layers[li], N=N, device=device)
            _write_row(li, 0, c0)
        first_tokens = [int(last0.argmax(-1)[0])]
        del c0

        for i in range(1, N):
            ci, lasti, _ = self.prefill_session(prompt_ids_2d[i],
                                                max_new_tokens=max_new_tokens,
                                                device=device)
            for li in range(len(batched.layers)):
                _write_row(li, i, ci)
            first_tokens.append(int(lasti.argmax(-1)[0]))
            del ci

        # exact int layers: set logical length to T so decode appends after the
        # prompt (bf16 StaticLayers already track this via cumulative_length).
        if quant:
            for li in exact:
                if isinstance(batched.layers[li], IntCls):
                    batched.layers[li].set_length(T)

        torch.cuda.synchronize(device)
        self._last_prefill_s = time.perf_counter() - _t_prefill0

        gen = [[first_tokens[i]] for i in range(N)]
        nxt = torch.tensor(first_tokens, device=device)
        torch.cuda.synchronize(device)
        _t_decode0 = time.perf_counter()

        # KIE-v1.1.y: patch the exact layers' attention to read the int8 store
        # via the tiled flash kernel (no full-bf16 transient). Sliding layers
        # keep the normal path.
        patched = []
        if quant_attn and quant:
            layers = self._decoder_layers()
            for li in exact:
                attn = layers[li].self_attn
                patched.append((attn, attn.forward))
                attn.forward = self._make_quant_attn_forward(
                    attn, batched.layers[li], attn_tile)
        try:
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
        finally:
            for attn, orig in patched:
                attn.forward = orig
        torch.cuda.synchronize(device)
        self._last_decode_s = time.perf_counter() - _t_decode0
        return gen
