"""MLX cross-model DLM-restored verifier (K3 Mac path).

Apple-Silicon (MLX) counterpart of
:class:`inference_engine.v04.cross_model_dlm_verifier.CrossModelDLMRestoredVerifier`
(the validated CUDA/transformers implementation). Same architecture:

  * verifier = Gemma 4 26B-A4B (MLX 4-bit, ``mlx_lm``)
  * proposer/drafter = DFlash 0.4B (PyTorch ``DFlashDrafter``, on MPS/CPU)
  * f_θ = trained K/V projection (PyTorch ``FThetaProjection``)

The verifier holds only a sink+window local cache; at *evicted* positions
its attention reads **restored** K/V:

  * **sliding-attention layers** → f_θ projection of the drafter's K/V
  * **full-attention (global) layers** → the verifier's OWN true K/V
    (**S5**). These are the recall-critical layers f_θ cannot reconstruct
    (CUDA eval rel_mse floor ~1.4). For long context the needle is outside
    the sliding window, so it reaches the output only through these layers
    — exact K/V there gives oracle-parity recall (CUDA ctx280: 10/10).

Cross-runtime design (mirrors ``scripts/research/k3_dflash_mlx_bridge.py``):
verifier in MLX; drafter + f_θ in PyTorch; tensors bridged at the
K/V-injection boundary. f_θ weights stay PyTorch (no MLX port for v1).

Mechanism: MLX ``nn.Module`` resolves ``__call__`` on the *class*, so we
temporarily patch ``Attention.__call__`` (verified against
mlx_lm.models.gemma4_text 0.31.3) with a dispatcher that, for layers
carrying a per-instance ``_kakeya_inject`` config, replaces evicted-position
K/V (via ``mx.where``) with restored pre-norm K/V before k_norm + RoPE;
other layers fall through to the original. This is the MLX analogue of the
CUDA ``_make_patched_forward`` (which patches ``layer.self_attn.forward``).
A clean capture pass supplies the S5 exact K/V for the full-attention
layers — mirroring CUDA ``capture_verifier_own_kv``.

KV sharing: Gemma 4 shares K/V across same-type layers for the last
``num_kv_shared_layers``. Sharing layers (``self_attn.has_kv == False``)
receive K/V via ``shared_kv`` from a source layer; injection happens only at
source (``has_kv``) layers and propagates to the sharers.

**Validation status**: the MLX path needs Apple Silicon and is validated by
``scripts/research/k3_integrated_niah_eval_mac.py`` on a Mac. The non-MLX
helpers here are importable + unit-tested on Linux (``mlx`` is imported
lazily inside the MLX-touching functions).
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from inference_engine.v04.kv_merge import compute_evicted_positions


# ---------------------------------------------------------------------------
# Model-structure helpers (no mlx import needed; unit-tested on Linux)
# ---------------------------------------------------------------------------


def resolve_mlx_text_model(mlx_model: Any) -> Any:
    """Return the ``Gemma4TextModel`` (exposes ``.layers`` / ``.embed_tokens``).

    Handles the multimodal wrapper (``model.language_model.model``) and the
    text-only wrapper (``model.model``), matching the bridge resolver.
    """
    logits_model = getattr(mlx_model, "language_model", mlx_model)
    text_model = getattr(logits_model, "model", None)
    if text_model is None and hasattr(logits_model, "embed_tokens"):
        text_model = logits_model
    if text_model is None or not hasattr(text_model, "embed_tokens"):
        raise AttributeError(
            "Could not locate MLX Gemma text model "
            "(expected model.language_model.model or model.model)"
        )
    return text_model


def mlx_full_attention_layer_indices(text_model: Any) -> List[int]:
    """Indices of the full-attention (global) layers — the S5 exact layers.

    Detected via each attention module's ``head_dim`` (full layers use
    ``global_head_dim`` > sliding ``head_dim``); falls back to the
    ``layer_type == 'full_attention'`` label. Returns [] if uniform.
    """
    layers = text_model.layers
    head_dims: List[int] = []
    types: List[str] = []
    for layer in layers:
        attn = layer.self_attn
        head_dims.append(int(getattr(attn, "head_dim", 0)))
        types.append(str(getattr(attn, "layer_type", "")))
    if len(set(head_dims)) > 1:
        max_hd = max(head_dims)
        return [i for i, hd in enumerate(head_dims) if hd == max_hd]
    return [i for i, t in enumerate(types) if t == "full_attention"]


def per_layer_kv_geometry(text_model: Any) -> List[Tuple[int, int, str]]:
    """Return ``[(n_kv_heads, head_dim, layer_type)]`` per layer."""
    out: List[Tuple[int, int, str]] = []
    for layer in text_model.layers:
        a = layer.self_attn
        out.append((
            int(getattr(a, "n_kv_heads", 0)),
            int(getattr(a, "head_dim", 0)),
            str(getattr(a, "layer_type", "")),
        ))
    return out


def kv_memory_report(
    text_model: Any,
    *,
    sink_size: int,
    window_size: int,
    seq_len: int,
    kv_dtype_bytes: int = 2,
    exact_layer_indices: Optional[Sequence[int]] = None,
    compress_full_bits_per_token_per_head: Optional[float] = None,
) -> Dict[str, Any]:
    """Analytical resident-KV-cache accounting for the Kakeya S5 engine.

    Models a *bounded* production engine (not the eval's full re-forward):

      * sliding layers   → resident = sink + window positions
      * exact full layers→ resident = ``seq_len`` positions (S5 keeps them
        exact). If ``compress_full_bits_per_token_per_head`` is given
        (KakeyaLattice), the per-token byte cost uses the compressed
        bits/head instead of ``head_dim * kv_dtype_bytes``.

    Returns per-layer-type bytes, total, and the per-token growth slope
    (the asymptotic linear term, dominated by the exact full layers).
    All quantities are pure arithmetic — unit-tested on Linux.
    """
    geom = per_layer_kv_geometry(text_model)
    exact = set(exact_layer_indices or [])
    resident_window = sink_size + window_size

    def kv_bytes_per_token(n_kv: int, hd: int, compressed: bool) -> int:
        # K + V (two tensors). attention_k_eq_v sharing is ignored here
        # (conservative: count both K and V).
        if compressed and compress_full_bits_per_token_per_head is not None:
            per_head_bytes = compress_full_bits_per_token_per_head / 8.0
            return int(round(2 * n_kv * per_head_bytes))
        return 2 * n_kv * hd * kv_dtype_bytes

    sliding_total = 0
    full_total = 0
    full_slope = 0          # bytes/token contributed by O(T) exact layers
    sliding_slope = 0       # bytes/token for sliding (0 once bounded)
    per_layer = []
    for i, (n_kv, hd, lt) in enumerate(geom):
        is_exact = i in exact
        bpt = kv_bytes_per_token(n_kv, hd, compressed=is_exact)
        if is_exact:
            positions = seq_len
            full_total += positions * bpt
            full_slope += bpt
        else:
            positions = min(resident_window, seq_len)
            sliding_total += positions * bpt
            sliding_slope += bpt if seq_len <= resident_window else 0
        per_layer.append({
            "layer": i, "layer_type": lt, "n_kv_heads": n_kv,
            "head_dim": hd, "exact": is_exact,
            "resident_positions": positions,
            "bytes_per_token": bpt,
            "resident_bytes": positions * bpt,
        })

    total = sliding_total + full_total
    return {
        "seq_len": seq_len,
        "kv_dtype_bytes": kv_dtype_bytes,
        "sink_window": resident_window,
        "exact_layer_indices": sorted(exact),
        "compress_full_bits_per_token_per_head": compress_full_bits_per_token_per_head,
        "sliding_resident_bytes": sliding_total,
        "full_resident_bytes": full_total,
        "total_resident_bytes": total,
        "total_resident_mb": round(total / 1e6, 2),
        "per_token_growth_bytes": full_slope + sliding_slope,
        "per_token_growth_kb": round((full_slope + sliding_slope) / 1024, 2),
        "per_layer": per_layer,
    }


def kv_source_layer_map(text_model: Any) -> List[int]:
    """Map layer index → the layer index that actually computes its K/V.

    For KV-shared layers (``has_kv == False``) the source is the earlier
    same-type layer in ``text_model.previous_kvs``; otherwise self.
    """
    n = len(text_model.layers)
    prev = list(getattr(text_model, "previous_kvs", list(range(n))))
    src: List[int] = []
    for i in range(n):
        attn = text_model.layers[i].self_attn
        has_kv = bool(getattr(attn, "has_kv", True))
        src.append(i if has_kv else int(prev[i]))
    return src


# ---------------------------------------------------------------------------
# MLX attention dispatcher (class-level patch with per-instance config)
# ---------------------------------------------------------------------------


def _build_dispatch(orig_call: Callable) -> Callable:
    """Build a replacement ``Attention.__call__`` (mlx_lm 0.31.3) that honors
    a per-instance ``self._kakeya_inject`` config when present, else delegates
    to ``orig_call``.

    Config dict keys: ``mode`` ("capture"|"inject"), ``restored_k``,
    ``restored_v`` (mx [B,T,n_kv,hd] pre-norm), ``evicted_mask`` (mx bool [T]),
    ``sink`` (dict for capture mode).
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm.models.base import scaled_dot_product_attention as _sdpa  # type: ignore

    def dispatch(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        cfg = getattr(self, "_kakeya_inject", None)
        if cfg is None:
            return orig_call(self, x, mask, cache, shared_kv, offset)

        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        elif not getattr(self, "has_kv", True):
            raise ValueError(
                f"Layer {self.layer_idx} is KV-shared but received no shared_kv"
            )
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            values = keys
            if not self.use_k_eq_v:
                values = self.v_proj(x).reshape(
                    B, L, self.n_kv_heads, self.head_dim
                )

            mode = cfg.get("mode")
            if mode == "capture":
                cfg["sink"][self.layer_idx] = (keys, values)
            elif mode == "inject":
                em = cfg.get("evicted_mask")
                if em is not None:
                    m = em.reshape(1, L, 1, 1)
                    rk = cfg.get("restored_k")
                    if rk is not None:
                        keys = mx.where(m, rk.astype(keys.dtype), keys)
                    if self.use_k_eq_v:
                        values = keys
                    else:
                        rv = cfg.get("restored_v")
                        if rv is not None:
                            values = mx.where(m, rv.astype(values.dtype), values)

            offset = mx.array(cache.offset) if cache is not None else 0
            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = _sdpa(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values), offset

    return dispatch


@contextlib.contextmanager
def _patched_attention_class(text_model: Any):
    """Temporarily replace the Attention class ``__call__`` with the
    Kakeya dispatcher; restore on exit. Clears any ``_kakeya_inject`` configs.
    """
    if not text_model.layers:
        yield
        return
    attn_cls = type(text_model.layers[0].self_attn)
    orig_call = attn_cls.__call__
    attn_cls.__call__ = _build_dispatch(orig_call)  # type: ignore[assignment]
    try:
        yield
    finally:
        attn_cls.__call__ = orig_call  # type: ignore[assignment]
        for layer in text_model.layers:
            if hasattr(layer.self_attn, "_kakeya_inject"):
                delattr(layer.self_attn, "_kakeya_inject")


def capture_own_kv(mlx_model: Any, input_ids: Sequence[int]) -> Dict[int, Tuple[Any, Any]]:
    """Clean forward recording each source layer's pre-norm K/V (mx arrays).

    Mirrors CUDA ``capture_verifier_own_kv``: ``{layer_idx: (k, v)}`` for
    ``has_kv`` layers, each ``[B, T, n_kv, head_dim]`` pre-norm. Supplies the
    S5 exact K/V for the full-attention layers.
    """
    import mlx.core as mx  # type: ignore

    text_model = resolve_mlx_text_model(mlx_model)
    sink: Dict[int, Tuple[Any, Any]] = {}
    with _patched_attention_class(text_model):
        for layer in text_model.layers:
            layer.self_attn._kakeya_inject = {"mode": "capture", "sink": sink}
        ids = mx.array([list(input_ids)])
        _ = text_model(ids, cache=None)
        mx.eval([t for kv in sink.values() for t in kv])
    return sink


def restored_logits(
    mlx_model: Any,
    input_ids: Sequence[int],
    *,
    restored_k_per_layer: Dict[int, Any],   # source_layer_idx -> mx [B,T,n_kv,hd] pre-norm
    restored_v_per_layer: Dict[int, Any],
    evicted_positions: Sequence[int],
    return_all: bool = False,
) -> Any:
    """Run the verifier with evicted-position K/V restoration.

    Returns the last-row logits (mx.array ``[V]``) by default, or all-position
    logits (``[T, V]``) when ``return_all=True`` (used by the teacher-forced
    single-forward recall eval). Injects only at ``has_kv`` source layers
    (sharers inherit via ``shared_kv``).
    """
    import mlx.core as mx  # type: ignore

    text_model = resolve_mlx_text_model(mlx_model)
    T = len(list(input_ids))
    mask_bool = [False] * T
    for p in evicted_positions:
        if 0 <= p < T:
            mask_bool[p] = True
    evicted_mask = mx.array(mask_bool)

    with _patched_attention_class(text_model):
        for idx, layer in enumerate(text_model.layers):
            attn = layer.self_attn
            if not bool(getattr(attn, "has_kv", True)):
                continue  # sharers inherit injected K/V via shared_kv
            rk = restored_k_per_layer.get(idx)
            rv = restored_v_per_layer.get(idx)
            if rk is None:
                continue
            attn._kakeya_inject = {
                "mode": "inject",
                "evicted_mask": evicted_mask,
                "restored_k": rk,
                "restored_v": rv,
            }
        ids = mx.array([list(input_ids)])
        logits = mlx_model(ids)            # full Model.__call__ → tied embed + softcap
        mx.eval(logits)
    return logits[0] if return_all else logits[0, -1]


# ---------------------------------------------------------------------------
# Incremental decode (MLX port of CUDA Gap-A) — kills the per-token re-forward
# throughput collapse. See docs/mlx-port-lessons.md.
# ---------------------------------------------------------------------------


def restored_prefill_cache(
    mlx_model: Any,
    input_ids: Sequence[int],
    *,
    restored_k_per_layer: Dict[int, Any],
    restored_v_per_layer: Dict[int, Any],
    evicted_positions: Sequence[int],
):
    """Prefill ONCE with restoration, capturing the restored K/V into a
    persistent mlx_lm prompt cache; return ``(cache, last_logits)``.

    Same injection as :func:`restored_logits`, but run **with a cache** so the
    patched attention's ``cache.update_and_fetch`` stores the post-injection
    K/V (full-attention/S5 layers → exact own K/V; sliding → f_θ-restored,
    window-bounded by the model's native RotatingKVCache). After this the
    verifier can decode new tokens incrementally over the cache — O(L)/step —
    instead of re-forwarding the whole sequence each token.

    Returns the model's native hybrid cache (full `KVCache` for global layers,
    `RotatingKVCache(sliding_window)` for sliding layers) populated to the
    prompt, plus the last-row logits (``mx [V]``) predicting the first token.
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore

    text_model = resolve_mlx_text_model(mlx_model)
    T = len(list(input_ids))
    evicted = set(int(p) for p in evicted_positions if 0 <= int(p) < T)
    evicted_mask = mx.array([p in evicted for p in range(T)])

    cache = make_prompt_cache(mlx_model)
    with _patched_attention_class(text_model):
        for idx, layer in enumerate(text_model.layers):
            attn = layer.self_attn
            if not bool(getattr(attn, "has_kv", True)):
                continue  # sharers inherit injected K/V via shared_kv
            rk = restored_k_per_layer.get(idx)
            rv = restored_v_per_layer.get(idx)
            if rk is None:
                continue
            attn._kakeya_inject = {
                "mode": "inject",
                "evicted_mask": evicted_mask,
                "restored_k": rk,
                "restored_v": rv,
            }
        ids = mx.array([list(input_ids)])
        logits = mlx_model(ids, cache=cache)
        mx.eval(logits)
    # Context manager restored the original Attention.__call__ → subsequent
    # decode steps run NATIVE incremental attention over this cache.
    return cache, logits[0, -1]


def restored_incremental_generate(
    mlx_model: Any,
    cache: Any,
    first_logits: Any,
    *,
    max_tokens: int,
    eos_ids: Sequence[int] = (),
) -> List[int]:
    """Greedy-decode up to ``max_tokens`` tokens over a restored prefill cache
    using mlx_lm's native ``generate_step`` (chunked + async-pipelined) — the
    throughput-critical incremental loop. Recall is carried by the cache's
    full-attention (S5) layers.
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm.generate import generate_step  # type: ignore

    eos = set(int(t) for t in eos_ids)
    nxt = int(mx.argmax(first_logits).item())
    out: List[int] = [nxt]
    if nxt in eos or max_tokens <= 1:
        return out
    # generate_step with a 1-token prompt + prefilled cache skips re-prefill
    # (its chunked-prefill loop needs >1 prompt token) and decodes incrementally.
    for tok, _ in generate_step(
        mx.array([nxt]), mlx_model, prompt_cache=cache, max_tokens=max_tokens - 1,
    ):
        t = int(tok)
        out.append(t)
        if t in eos:
            break
    return out
