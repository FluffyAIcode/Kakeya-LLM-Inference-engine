"""MLX **native** restored-cache primitive — the systemic fix for the Mac
token-throughput collapse (PR #109 stacked).

Problem (per the architecture review): the first MLX port made the *decode* hot
path native (`generate_step`), but the restored cache was still produced by a
Python attention-patch injection at prefill, plus a separate `capture_own_kv`
forward, plus an MLX↔PyTorch/MPS bridge for f_θ — so end-to-end cost piled up in
prefill materialization / lazy-eval sync / cross-runtime bridging rather than in
the attention kernel.

This module makes the **whole cache lifecycle native**:

1. **`build_native_prefill_cache`** — a single *native* prefill
   (`mlx_model(prompt, cache=make_prompt_cache(...))`) populates the model's own
   native cache with the **exact own K/V** for every layer (full-attention →
   unbounded `KVCache` carrying the needle for S5 recall; sliding → bounded
   `RotatingKVCache`). No attention patch, no second `capture_own_kv` forward,
   no Python cache reconstruction. (Optionally taps prompt aux in the *same*
   forward for the fused path.)
2. **`set_kv_cache_state` / `inject_restored_into_native_cache`** — write
   restored/own K/V **directly into the native cache layout** via the cache's
   own `.state` setter (no Python wrapper object).
3. **`quantize_full_attn_layers`** — convert the (few) full-attention layers'
   `KVCache` to a native **`QuantizedKVCache`** for *real* resident-memory
   reduction with native quantized decode (the MLX-native analog of the
   KakeyaLattice full-attn compression; sliding is already bounded natively).
4. Decode + trim/append stay on the **native prompt cache** (`generate_step`,
   `trim_prompt_cache`).

Recall is carried by S5's exact full-attention K/V, which the native prefill
produces for free — so this path needs **no f_θ and no drafter in the loop**,
hence **no bridge / no per-token patch**. That is why it fixes the collapse.

MLX execution requires Apple Silicon; the control flow + native `.state` write
are unit-tested on Linux with fakes, and the end-to-end path is validated on a
Mac via ``k3_integrated_niah_eval_mac.py --native-cache``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference_engine.backends.mlx.cross_model_dlm_verifier import (
    restored_incremental_generate,
)


# --------------------------------------------------------------------------- #
# (2) Direct native-layout writes.
# --------------------------------------------------------------------------- #
def set_kv_cache_state(layer_cache: Any, keys: Any, values: Any) -> Any:
    """Write ``(keys, values)`` (``mx [B, n_kv, T, head_dim]`` post-norm/RoPE)
    directly into a native ``KVCache`` via its ``.state`` setter (which also sets
    ``offset = T``). The native, wrapper-free way to place restored/own K/V into
    the MLX cache layout."""
    layer_cache.state = (keys, values)
    return layer_cache


def inject_restored_into_native_cache(
    cache: Sequence[Any],
    restored_k_per_layer: Dict[int, Any],
    restored_v_per_layer: Dict[int, Any],
    layer_indices: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Overwrite selected layers' native cache state with restored K/V (e.g. f_θ
    sliding restoration for the bounded-memory variant). Layers absent from the
    restored dicts are left as their native prefill K/V."""
    out = list(cache)
    idxs = layer_indices if layer_indices is not None else list(restored_k_per_layer)
    for i in idxs:
        if i in restored_k_per_layer and i < len(out):
            set_kv_cache_state(out[i], restored_k_per_layer[i], restored_v_per_layer[i])
    return out


# --------------------------------------------------------------------------- #
# (1) One native prefill -> native cache with exact own K/V (+ optional aux).
# --------------------------------------------------------------------------- #
def build_native_prefill_cache(
    mlx_model: Any,
    prompt_ids: Sequence[int],
    *,
    prefill_step_size: int = 512,
) -> Tuple[Any, Any]:
    """Native prefill into the model's own prompt cache, processing the prompt in
    **chunks of ``prefill_step_size``** (mirrors ``mlx_lm.generate_step``'s
    chunked prefill, lines ~430-443). Returns ``(cache, last_logits)``.

    Chunking is essential on Apple Silicon: a single full-prompt forward
    materializes an O(T_query × T_key) attention and OOMs on long NIAH prompts
    (the failure that made the naive Step-0 unusable). After this the cache holds
    the **exact own K/V** for every layer (full-attention `KVCache` → S5 recall;
    sliding `RotatingKVCache` → bounded), and decode runs via `generate_step`.
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore

    ids_list = [int(t) for t in prompt_ids]
    if not ids_list:
        raise ValueError("prompt_ids must be non-empty")
    step = max(int(prefill_step_size), 1)
    cache = make_prompt_cache(mlx_model)

    n = len(ids_list)
    pos = 0
    logits = None
    while pos < n:
        chunk = ids_list[pos:pos + step]
        logits = mlx_model(mx.array([chunk]), cache=cache)
        mx.eval([c.state for c in cache])   # materialize per chunk (bound peak)
        pos += len(chunk)
    mx.eval(logits)
    return cache, logits[0, -1]


# --------------------------------------------------------------------------- #
# (3) Native real memory compression of the full-attention layers.
# --------------------------------------------------------------------------- #
def quantize_full_attn_layers(
    cache: Sequence[Any],
    full_attn_layer_indices: Sequence[int],
    *,
    bits: int = 8,
    group_size: int = 64,
) -> List[Any]:
    """Convert the full-attention layers' native ``KVCache`` to a native
    ``QuantizedKVCache`` (real resident-memory reduction + native quantized
    decode). Sliding layers are already bounded by ``RotatingKVCache`` and are
    left untouched. S5 recall is preserved (the needle stays in the full-attn
    K/V, now quantized)."""
    out = list(cache)
    for i in full_attn_layer_indices:
        if i >= len(out):
            continue
        c = out[i]
        if hasattr(c, "to_quantized") and not c.empty():
            out[i] = c.to_quantized(group_size=group_size, bits=bits)
    return out


def cache_resident_bytes(cache: Sequence[Any]) -> int:
    """Sum of the live native cache layers' ``nbytes`` — the *actual* resident
    KV memory (quantized layers report their compressed size)."""
    return int(sum(int(getattr(c, "nbytes", 0)) for c in cache))


# --------------------------------------------------------------------------- #
# (4) Native decode (re-exported for a one-call collapse-fix path).
# --------------------------------------------------------------------------- #
def native_restored_decode(
    mlx_model: Any,
    cache: Any,
    first_logits: Any,
    *,
    max_tokens: int,
    eos_ids: Sequence[int] = (),
) -> List[int]:
    """Decode over a *pre-built* native cache via mlx_lm ``generate_step``
    (async-pipelined, O(L)/token). Thin alias of the validated incremental
    decoder. Used by the quantized path (prefill → quantize full-attn → decode)."""
    return restored_incremental_generate(
        mlx_model, cache, first_logits, max_tokens=max_tokens, eos_ids=eos_ids)


def native_generate(
    mlx_model: Any,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int,
    eos_ids: Sequence[int] = (),
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
) -> Tuple[List[int], Any]:
    """Pure-native end-to-end path: hand the **whole prompt** to mlx_lm's own
    ``generate_step`` (chunked prefill + async-pipelined decode + optional
    ``kv_bits`` quantization) over a native prompt cache. This is mlx_lm's
    validated loop verbatim — it cannot OOM on prefill (chunked) and is the
    safest collapse-fix path. Returns ``(tokens, cache)``.

    Recall rides S5's exact full-attention K/V (native `KVCache`); sliding stays
    bounded (`RotatingKVCache`). ``kv_bits`` (if set) quantizes every trimmable
    layer natively once prefilled (extra memory savings).
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm.generate import generate_step  # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore

    ids = [int(t) for t in prompt_ids]
    if not ids:
        raise ValueError("prompt_ids must be non-empty")
    eos = set(int(t) for t in eos_ids)
    cache = make_prompt_cache(mlx_model)
    out: List[int] = []
    for tok, _ in generate_step(
        mx.array(ids), mlx_model, prompt_cache=cache, max_tokens=max_tokens,
        prefill_step_size=prefill_step_size, kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    ):
        t = int(tok)
        out.append(t)
        if t in eos:
            break
    return out, cache
