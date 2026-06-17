"""MLX port of the #107 fused DFlash spec-decode engine (Components A+B+C).

Hybrid runtime: the **verifier is MLX** (Gemma-4 26B-A4B, 4-bit) and the
**DFlash drafter + f_θ are PyTorch** (MPS/CPU), bridged at the K/V-injection and
aux-hidden boundaries (one bridge per block, never a re-forward).

This mirrors ``scripts/research/k3_specdecode_gpu_bench.py:restored_specdecode_fused``
(CUDA) per-block O(L):

* **C (Gap-A)** — incremental restored verify: prefill captures restored K/V
  into the model's native hybrid cache (full-attn = exact own K/V, S5 → recall;
  sliding = f_θ-restored, window-bounded); each block verifies the candidate
  tokens against that cache and is rolled back on rejection via mlx_lm's native
  ``trim_prompt_cache`` (the same primitive mlx_lm's own spec-decode uses).
* **B** — drafter context K/V cache: built once from the prompt's aux hidden,
  then EXTENDED with each committed token's aux (no O(C) recompute per block).
* **A** — the committed tokens' aux hidden are captured FROM the verify forward
  (by patching the Gemma-4 decoder-layer ``__call__`` to record its output), so
  there is no separate per-block clean-aux forward.

The MLX-execution paths (forward/inject/aux-capture/trim) require Apple Silicon
and are validated on a Mac by ``k3_integrated_niah_eval_mac.py --fused-specdecode``;
the pure control flow (``fused_specdecode_generate`` loop, the verifier adapter's
prefill/verify/commit/truncate sequencing) is unit-tested on Linux with fakes.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from inference_engine.backends.mlx.cross_model_dlm_verifier import (
    resolve_mlx_text_model,
    restored_prefill_cache,
)

# #region agent log (Phase-1 long-gen degeneration debug; remove after fix)
import json as _kjson
import sys as _ksys


def _kdbg(ev: str, **kw: Any) -> None:
    """Emit one compact NDJSON line to stderr (captured by the git-bus bridge)."""
    try:
        rec = {"ev": ev, **kw}
        _ksys.stderr.write("KDBG " + _kjson.dumps(rec, separators=(",", ":")) + "\n")
        _ksys.stderr.flush()
    except Exception:
        pass


def _kdbg_rep(toks: List[int], k: int = 64) -> Dict[str, Any]:
    """Degeneration signal over the last ``k`` generated tokens.

    Phase-1 widened the window (32→64) AND added phrase-level cycle detection:
    ``max_run`` only sees single-token runs, so a structural repeat loop
    ("### 1. ...### 1. ...") reads ``max_run:1`` yet is fully degenerate. We
    also scan for the period ``p`` in ``[1, n//2]`` that maximises the fraction
    of positions equal to the token ``p`` steps back (``cyc_frac`` near 1.0 with
    ``cyc_p>1`` ⇒ phrase/sentence-level repetition the run-length metric misses).
    """
    w = toks[-k:]
    if not w:
        return {"win": 0}
    n = len(w)
    uniq = len(set(w))
    run = best = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a == b else 1
        if run > best:
            best = run
    cyc_p, cyc_frac = 0, 0.0
    for p in range(1, n // 2 + 1):
        m = sum(1 for i in range(p, n) if w[i] == w[i - p])
        frac = m / (n - p)
        if frac > cyc_frac:
            cyc_frac, cyc_p = frac, p
    return {"win": n, "uniq_frac": round(uniq / n, 3),
            "rep_frac": round(1.0 - uniq / n, 3), "max_run": best,
            "cyc_p": cyc_p, "cyc_frac": round(cyc_frac, 3)}


def _kdbg_sync(cache: Any, past_len: int) -> Dict[str, Any]:
    """Phase-1 H2: surface cache desync. The torch_ftheta loop rolls rejections
    back with ``trim_prompt_cache`` on the NATIVE hybrid cache. If the sliding
    ``RotatingKVCache`` and the full ``KVCache`` trim by different amounts (or
    one is non-trimmable after the ring wraps), their ``offset`` diverges from
    ``_past_len`` and from each other — the position misalignment that would
    corrupt subsequent logits. Compare both offsets to ``past_len``."""
    sl = fu = None
    for c in (cache or []):
        off = int(getattr(c, "offset", 0))
        if "Rotating" in type(c).__name__:
            if sl is None:
                sl = off
        elif fu is None:
            fu = off
    return {"past_len": int(past_len), "sliding_off": sl, "full_off": fu,
            "sliding_eq": (sl == past_len) if sl is not None else None,
            "full_eq": (fu == past_len) if fu is not None else None,
            "sliding_minus_full": (sl - fu) if (sl is not None and fu is not None) else None}


def _kdbg_cache(cache: Any) -> Dict[str, Any]:
    """Summarize per-layer cache state: pick the first sliding (RotatingKVCache)
    and first full (KVCache) layer and report global offset, physical resident
    seq-len, max_size and keep (sink) so we can correlate window-eviction with
    the restored-coverage boundary. Also returns layer-class counts."""
    sliding = full = None
    counts: Dict[str, int] = {}
    for c in (cache or []):
        cls = type(c).__name__
        counts[cls] = counts.get(cls, 0) + 1
        keys = getattr(c, "keys", None)
        info = {
            "cls": cls,
            "off": int(getattr(c, "offset", 0)),
            "phys": int(keys.shape[2]) if keys is not None else 0,
            "ms": (int(getattr(c, "max_size")) if getattr(c, "max_size", None) is not None else None),
            "keep": (int(getattr(c, "keep")) if getattr(c, "keep", None) is not None else None),
        }
        if "Rotating" in cls and sliding is None:
            sliding = info
        elif "Rotating" not in cls and full is None:
            full = info
    return {"counts": counts, "sliding": sliding, "full": full}


def _kdbg_lost(cache: Any, restored: Any, prompt_len: int) -> Optional[Dict[str, Any]]:
    """Phase-1 Q2: count sliding-layer positions evicted DURING decode that have
    NO restored K/V. For the first RotatingKVCache: positions [keep, evict_hi)
    are no longer resident, where evict_hi = offset - (max_size - keep). Of those,
    any not in the (prompt-only) restored coverage are 'lost' (no K/V anywhere)."""
    for c in (cache or []):
        if "Rotating" not in type(c).__name__:
            continue
        ms = getattr(c, "max_size", None)
        if ms is None:
            return None
        off = int(getattr(c, "offset", 0))
        keep = int(getattr(c, "keep", 0) or 0)
        ms = int(ms)
        evict_hi = off - (ms - keep)          # exclusive upper bound of evicted region
        evicted_n = max(0, evict_hi - keep)
        rset = restored if isinstance(restored, set) else set()
        lost = sum(1 for p in range(keep, evict_hi) if p not in rset)
        return {"off": off, "ms": ms, "keep": keep, "evict_hi": evict_hi,
                "evicted_n": evicted_n, "restored_in_evicted": evicted_n - lost,
                "lost": lost, "prompt_len": int(prompt_len),
                "window_slid_off_prompt": bool(evict_hi > prompt_len)}
    return None
# #endregion


# --------------------------------------------------------------------------- #
# Component A: capture verifier aux-layer hidden states (no transformers
# `output_hidden_states` on MLX → patch the decoder-layer __call__).
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _patched_decoder_layers(text_model: Any):
    """Enable aux hidden capture on Gemma-4 decoder layers.

    Patched local MLX-LM exposes a lightweight in-layer tap via
    ``_kakeya_aux_sink``. Use that when available so every verify forward stays
    on the normal ``DecoderLayer.__call__`` implementation. Fall back to the old
    class-level wrapper for unpatched MLX-LM installs.
    """
    if not text_model.layers:
        yield
        return
    if bool(getattr(text_model.layers[0], "_kakeya_native_aux_tap", False)):
        try:
            yield
        finally:
            for layer in text_model.layers:
                if hasattr(layer, "_kakeya_aux_sink"):
                    delattr(layer, "_kakeya_aux_sink")
                if hasattr(layer, "_aux_record"):
                    delattr(layer, "_aux_record")
        return
    layer_cls = type(text_model.layers[0])
    orig_call = layer_cls.__call__

    def dispatch(self, *args, **kwargs):
        out = orig_call(self, *args, **kwargs)   # (h, shared_kv, offset)
        rec = getattr(self, "_aux_record", None)
        if rec is not None:
            rec[int(self.layer_idx)] = out[0]
        return out

    layer_cls.__call__ = dispatch  # type: ignore[assignment]
    try:
        yield
    finally:
        layer_cls.__call__ = orig_call  # type: ignore[assignment]
        for layer in text_model.layers:
            if hasattr(layer, "_aux_record"):
                delattr(layer, "_aux_record")


def _build_aux(text_model: Any, ids_mx: Any, sink: Dict[int, Any],
               embed_scale: float, aux_layer_ids: Sequence[int]) -> List[Any]:
    """Assemble a transformers-style ``hidden_states`` list and index it.

    ``hs[0]`` = scaled token embeddings; ``hs[k]`` = output of decoder layer
    ``k-1`` (so ``hs[a]`` matches HF ``output_hidden_states[a]`` = input to
    layer ``a`` = output of layer ``a-1``). Returns ``[hs[a] for a in
    aux_layer_ids]``, each ``mx [1, L, hidden]``.
    """
    embeds = text_model.embed_tokens(ids_mx)
    embeds = embeds * embed_scale
    n = len(text_model.layers)
    hs = [embeds] + [sink[i] for i in range(n)]
    return [hs[a] for a in aux_layer_ids]


def capture_aux_hidden(
    mlx_model: Any,
    input_ids: Sequence[int],
    aux_layer_ids: Sequence[int],
    *,
    embed_scale: float,
) -> List[Any]:
    """Clean (no-cache) forward capturing the verifier's aux-layer hidden over
    ``input_ids``. Returns ``[mx [1, T, hidden]]`` for the prompt; used to seed
    the drafter context K/V cache (Component B)."""
    import mlx.core as mx  # type: ignore

    text_model = resolve_mlx_text_model(mlx_model)
    sink: Dict[int, Any] = {}
    with _patched_decoder_layers(text_model):
        for layer in text_model.layers:
            layer._kakeya_aux_sink = sink
            layer._aux_record = sink
        ids = mx.array([list(input_ids)])
        _ = mlx_model(ids)
        aux = _build_aux(text_model, ids, sink, embed_scale, aux_layer_ids)
        mx.eval(aux)
    return aux


# --------------------------------------------------------------------------- #
# Component C: incremental restored verifier (MLX analog of
# CrossModelRestoredSinkWindowVerifier(incremental=True)) with aux capture (A).
# --------------------------------------------------------------------------- #
class MLXRestoredIncrementalVerifier:
    """Stateful MLX restored verifier for the fused spec-decode loop.

    ``prefill`` builds the restored cache (Gap-A) and the first-token logits;
    ``forward_block`` verifies a candidate block incrementally (and, when
    ``_capture_aux``, records the per-token aux hidden bridged to torch);
    ``commit_or_truncate`` rolls the cache back by the rejected count via
    ``mlx_lm.trim_prompt_cache``; ``append_token`` commits the correction.
    """

    def __init__(
        self,
        mlx_model: Any,
        *,
        embed_scale: float,
        aux_layer_ids: Sequence[int] = (),
        bridge_to_torch: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.mlx_model = mlx_model
        self.text_model = resolve_mlx_text_model(mlx_model)
        self.embed_scale = float(embed_scale)
        self.aux_layer_ids = tuple(int(a) for a in aux_layer_ids)
        self._bridge = bridge_to_torch
        self._cache: Any = None
        self._past_len = 0
        self.next_token_logits: Any = None
        self._last_aux: Optional[List[Any]] = None
        self._last_aux_mx: Optional[List[Any]] = None
        self._capture_aux = False
        self._block_snapshot: Optional[List[Dict[str, Any]]] = None
        self._full_kv = False

    def reset(self) -> None:
        self._cache = None
        self._past_len = 0
        self.next_token_logits = None
        self._last_aux = None
        self._last_aux_mx = None
        self._block_snapshot = None

    def prefill(
        self,
        prompt_ids: Sequence[int],
        *,
        restored_k_per_layer: Dict[int, Any],
        restored_v_per_layer: Dict[int, Any],
        evicted_positions: Sequence[int],
        prefill_chunk_size: int = 0,
        full_kv: bool = False,
    ) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        # full_kv=True → all-`KVCache` layout so accept/reject rollback can use
        # SOUND native trim (keep accepted, drop rejected) with no re-forward.
        self._full_kv = bool(full_kv)
        factory = make_full_kv_prompt_cache if full_kv else None
        self._cache, self.next_token_logits = restored_prefill_cache(
            self.mlx_model, list(prompt_ids),
            restored_k_per_layer=restored_k_per_layer,
            restored_v_per_layer=restored_v_per_layer,
            evicted_positions=evicted_positions,
            prefill_chunk_size=prefill_chunk_size,
            cache_factory=factory,
        )
        self._past_len = len(prompt_ids)
        # #region agent log (Phase-1)
        try:
            ev = sorted(int(p) for p in evicted_positions)
            rk_layers = sorted(int(k) for k in restored_k_per_layer.keys())
            # Stash restored coverage for the decode loop's lost-position check.
            self._dbg_restored_positions = set(ev)
            self._dbg_prompt_len = int(len(prompt_ids))
            _kdbg(
                "prefill",
                prompt_len=len(prompt_ids),
                evicted_count=len(ev),
                evicted_lo=(ev[0] if ev else None),
                evicted_hi=(ev[-1] if ev else None),
                restored_layers=rk_layers,
                restored_layer_count=len(rk_layers),
                full_kv=bool(self._full_kv),
                cache=_kdbg_cache(self._cache),
            )
        except Exception:
            pass
        # #endregion

    def forward_block(self, tokens: Sequence[int]) -> Any:
        """Incremental verify of ``tokens`` against the restored cache. Returns
        ``mx [len(tokens), V]`` logits; captures aux hidden states in MX first
        and bridges to torch lazily via :meth:`last_aux_torch_slice`."""
        import mlx.core as mx  # type: ignore

        if self._cache is None:
            raise RuntimeError("verifier not prefilled")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        ids = mx.array([list(tokens)])
        want_aux = self._capture_aux and bool(self.aux_layer_ids)
        if want_aux:
            sink = {}
            with _patched_decoder_layers(self.text_model):
                for layer in self.text_model.layers:
                    layer._kakeya_aux_sink = sink
                    layer._aux_record = sink
                logits = self.mlx_model(ids, cache=self._cache)
                aux = _build_aux(self.text_model, ids, sink,
                                 self.embed_scale, self.aux_layer_ids)
                mx.eval(logits, *aux)
            self._last_aux_mx = [a[0] for a in aux]  # [L, hidden] each, mx
            self._last_aux = None
        else:
            logits = self.mlx_model(ids, cache=self._cache)
            mx.eval(logits)
            self._last_aux = None
            self._last_aux_mx = None
        return logits[0]

    def last_aux_torch_slice(self, start: int = 0, end: Optional[int] = None) -> List[Any]:
        """Bridge a token slice from the last captured aux hidden states."""
        if self._last_aux_mx is None:
            raise RuntimeError("aux hidden not captured for the last forward_block")
        bridge = self._bridge or (lambda a: a)
        return [bridge(a[start:end]) for a in self._last_aux_mx]

    def rollback_block(self) -> None:
        """O(1) full rollback of the last ``forward_block_lazy`` call.

        ``trim_prompt_cache`` is NOT a valid rollback on Gemma-4's hybrid
        cache once the sliding-window RotatingKVCache has wrapped
        (seq >> 512): rejected draft K/V linger in the ring and poison
        subsequent logits (observed live as stream divergence vs greedy +
        acceptance collapse; retroactively explains iterC's 23-token
        sample and the eager loop's silent post-answer divergence). MLX
        arrays are immutable, so a snapshot is just attribute references;
        restore rebinds them.
        """
        if self._block_snapshot is None:
            raise RuntimeError("no block snapshot to roll back to")
        for c, snap in zip(self._cache, self._block_snapshot):
            for attr, val in snap.items():
                setattr(c, attr, val)

    def forward_block_lazy(self, ids_mx: Any) -> Any:
        """LAZY incremental verify: ``ids_mx`` is an mx ``[1, L]`` (typically
        the in-graph concatenation of the carried bonus + lazy draft ids —
        lever ② of the single-sync loop). Returns ``mx [L, V]`` logits with
        NO evaluation; aux hidden (when ``_capture_aux``) stays lazy in
        ``_last_aux_mx`` and is consumed lazily by the drafter-context
        extension."""
        if self._cache is None:
            raise RuntimeError("verifier not prefilled")
        # Reference snapshot of every per-layer cache state (immutable mx
        # arrays → O(layers) attribute refs) for rollback_block().
        self._block_snapshot = [
            {attr: getattr(c, attr)
             for attr in ("keys", "values", "offset", "_idx")
             if hasattr(c, attr)}
            for c in self._cache
        ]
        want_aux = self._capture_aux and bool(self.aux_layer_ids)
        if want_aux:
            sink: Dict[int, Any] = {}
            with _patched_decoder_layers(self.text_model):
                for layer in self.text_model.layers:
                    layer._kakeya_aux_sink = sink
                    layer._aux_record = sink
                logits = self.mlx_model(ids_mx, cache=self._cache)
                aux = _build_aux(self.text_model, ids_mx, sink,
                                 self.embed_scale, self.aux_layer_ids)
            self._last_aux_mx = [a[0] for a in aux]  # [L, hidden] each, lazy
            self._last_aux = None
        else:
            logits = self.mlx_model(ids_mx, cache=self._cache)
            self._last_aux = None
            self._last_aux_mx = None
        return logits[0]

    def commit_or_truncate(self, *, forwarded: int, accepted: int) -> None:
        if accepted < 0 or accepted > forwarded:
            raise ValueError("accepted must satisfy 0 <= accepted <= forwarded")
        drop = forwarded - accepted
        if drop > 0 and self._cache is not None:
            from mlx_lm.models.cache import trim_prompt_cache  # type: ignore
            # #region agent log (Phase-1 H2: unsound trim on the hybrid cache)
            _before = _kdbg_cache(self._cache)
            trimmed = trim_prompt_cache(self._cache, drop)
            _kdbg(
                "trim",
                drop=int(drop),
                trimmed=(int(trimmed) if trimmed is not None else None),
                short=(trimmed is not None and int(trimmed) < int(drop)),
                past_len=int(self._past_len),
                before=_before,
                after=_kdbg_cache(self._cache),
            )
            # #endregion
        self._past_len += accepted

    def append_token(self, token_id: int) -> Any:
        logits = self.forward_block([int(token_id)])
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1]
        return self.next_token_logits


# --------------------------------------------------------------------------- #
# Bridge embed / lm_head for the hybrid path (drafter torch ↔ verifier MLX).
# --------------------------------------------------------------------------- #
def make_bridge_embed_lm_head(
    text_model: Any,
    *,
    mx_to_torch: Callable[..., Any],
    torch_to_mx: Callable[..., Any],
    device: Any,
    torch_dtype: Any,
    softcap: Optional[float] = None,
) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """Return ``(embed_fn, lm_head_fn)`` over the MLX verifier weights for the
    PyTorch drafter.

    * ``embed_fn`` — **Gap-B**: a *plain* shared-embedding lookup with **no
      ``×sqrt(hidden)`` scaling** (the drafting query embedding bug fixed in
      #107). Returns torch ``[*, hidden]``.
    * ``lm_head_fn`` — tied-embedding logits + Gemma ``final_logit_softcapping``
      (monotonic; preserves argmax). Returns torch ``[*, vocab]``.
    """
    import mlx.core as mx  # type: ignore

    def embed_fn(query_ids: Any) -> Any:
        ids_mx = mx.array(query_ids.detach().to("cpu").tolist())
        emb = text_model.embed_tokens(ids_mx)        # NO * embed_scale (Gap-B)
        return mx_to_torch(emb, dtype=torch_dtype, device=device)

    def lm_head_fn(h: Any) -> Any:
        h_mx = torch_to_mx(h)
        out = text_model.embed_tokens.as_linear(h_mx)
        if softcap:
            out = softcap * mx.tanh(out / softcap)
        return mx_to_torch(out, dtype=torch_dtype, device=device)

    return embed_fn, lm_head_fn


# --------------------------------------------------------------------------- #
# Single-sync all-MLX fused loop (levers ① ② ③ of the Step-2 throughput plan;
# docs/mlx-port-lessons.md "Step-2 rescue status").
# --------------------------------------------------------------------------- #
def make_full_kv_prompt_cache(mlx_model: Any) -> List[Any]:
    """Build a prompt cache that uses a full append-only ``KVCache`` for EVERY
    layer (including the sliding-attention ones, which the model's native
    ``make_cache`` would give a ``RotatingKVCache``).

    Why: ``RotatingKVCache`` is not trimmable once the ring has wrapped
    (``is_trimmable`` → ``offset < max_size``), so spec-decode accept/reject
    rollback cannot keep the accepted K/V via a cheap trim — it must re-forward
    (the v3 carry penalty). With an all-``KVCache`` layout, ``trim_prompt_cache``
    is a sound O(1) slice on every layer, so the loop keeps accepted K/V and
    drops only the rejected tail (CUDA `DynamicCache` parity). Sliding attention
    remains byte-exact because the per-layer window mask is applied regardless
    of cache capacity; the only cost is O(T) sliding KV during decode.
    """
    from mlx_lm.models.cache import make_prompt_cache, KVCache  # type: ignore

    n = len(make_prompt_cache(mlx_model))
    return [KVCache() for _ in range(n)]


def fused_specdecode_generate_mlx_trim(
    adapter: "MLXRestoredIncrementalVerifier",
    drafter: Any,
    *,
    aux_prompt: Sequence[Any],
    embed_fn: Callable[[Any], Any],
    lm_head_fn: Callable[[Any], Any],
    gen_tokens: int,
    block_size: int,
    eos_ids: Sequence[int] = (),
    single_fused: bool = False,
) -> Dict[str, Any]:
    """CUDA-parity fused spec decode: KEEP accepted K/V, TRIM only the rejected
    tail (no rollback, no carry re-forward). Requires the adapter to be
    prefilled with ``full_kv=True`` (all-``KVCache`` layout) so the native
    ``trim_prompt_cache`` is sound. Levers ①②③ retained (lazy draft+verify
    single graph, in-graph cumprod acceptance, carried correction).

    Per block: forward ``[bonus + drafts]`` (L tokens) → cache = base+L; accept
    the leading match count ``k`` (bonus always accepts); ``trim_prompt_cache``
    drops the L−k rejected tokens; advance ``_past_len`` by ``k``. The accepted
    tokens' K/V (computed in this forward) stay in the cache — never recomputed.
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm.models.cache import trim_prompt_cache  # type: ignore

    eos = set(int(t) for t in eos_ids)
    C = adapter._past_len
    ctx_kv = drafter.make_context_kv(list(aux_prompt), mx.arange(0, C))
    mx.async_eval([t for kv in ctx_kv for t in kv])
    timing = {"ctx_kv_build_s": 0.0, "build_s": 0.0, "eval_s": 0.0, "extend_s": 0.0}
    adapter._capture_aux = True

    generated: List[int] = []
    accepts: List[int] = []
    block_evals: List[float] = []
    ctx_len = C
    try:
        while len(generated) < gen_tokens:
            _kblk_t0 = time.perf_counter()  # agent log (Phase-1)
            L = min(block_size, gen_tokens - len(generated))
            base = adapter._past_len
            t_build = time.perf_counter()
            bonus_id = mx.argmax(adapter.next_token_logits)        # lazy scalar
            n_draft = max(L - 1, 0)
            if n_draft:
                drafts = drafter.draft_block_ids(
                    ctx_kv, bonus_id, embed_fn, lm_head_fn,
                    n_masks=n_draft, context_len=base)
                check_ids = mx.concatenate([bonus_id[None], drafts])   # [L]
                if not single_fused:
                    mx.eval(check_ids)   # two-phase (drafter graph before 26B)
                # single_fused=True → leave check_ids LAZY so the drafter and
                # 26B verify fuse into ONE graph (the path b876 found Metal-
                # pathological); this probe times it to classify the instability.
            else:
                check_ids = bonus_id[None]
            block_logits = adapter.forward_block_lazy(check_ids[None])  # [L, V]
            # in-graph greedy acceptance over the check region
            pred_rows = mx.concatenate(
                [adapter.next_token_logits[None], block_logits[:max(L - 1, 0)]],
                axis=0)
            matches = (mx.argmax(pred_rows, axis=-1) == check_ids)
            accepted_mx = mx.sum(mx.cumprod(matches.astype(mx.int32)))
            rows = mx.concatenate(
                [adapter.next_token_logits[None], block_logits], axis=0)  # [L+1,V]
            next_row = mx.take(rows, accepted_mx[None], axis=0)[0]        # [V]
            timing["build_s"] += time.perf_counter() - t_build
            t_eval = time.perf_counter()
            mx.eval(accepted_mx, check_ids)
            blk_eval = time.perf_counter() - t_eval
            timing["eval_s"] += blk_eval
            block_evals.append(round(blk_eval, 4))
            accepted = int(accepted_mx.item())
            check = [int(x) for x in check_ids.tolist()]
            commit = check[:accepted]
            generated += commit
            accepts.append(accepted)
            # #region agent log (Phase-1)
            _kdbg(
                "block",
                loop="mlx_trim",
                blk=len(accepts) - 1,
                gen=len(generated),
                past_len=adapter._past_len,
                accepted=accepted,
                L=int(check_ids.shape[0]),
                dt_ms=round((time.perf_counter() - _kblk_t0) * 1e3, 1),
                rep=_kdbg_rep(generated),
                lost=_kdbg_lost(adapter._cache,
                                getattr(adapter, "_dbg_restored_positions", set()),
                                getattr(adapter, "_dbg_prompt_len", 0)),
                cache=_kdbg_cache(adapter._cache),
            )
            # #endregion
            adapter.next_token_logits = next_row
            aux_rows = adapter._last_aux_mx
            # KEEP accepted (positions base..base+accepted-1), TRIM rejected.
            drop = L - accepted
            if drop > 0:
                trim_prompt_cache(adapter._cache, drop)
            adapter._past_len = base + accepted
            S_new = adapter._past_len
            lo, hi = ctx_len - base, S_new - base
            if hi > lo and aux_rows is not None:
                t_extend = time.perf_counter()
                new_aux = [a[lo:hi][None] for a in aux_rows]
                ctx_kv = drafter.extend_context_kv(
                    ctx_kv,
                    drafter.make_context_kv(new_aux, mx.arange(ctx_len, S_new)))
                mx.async_eval([t for kv in ctx_kv for t in kv])
                ctx_len = S_new
                timing["extend_s"] += time.perf_counter() - t_extend
            if any(t in eos for t in commit):
                break
    finally:
        adapter._capture_aux = False
    generated = generated[:gen_tokens]
    return {
        "tokens": generated,
        "blocks": len(accepts),
        "mean_accept_len": (round(sum(accepts) / len(accepts), 3)
                            if accepts else 0.0),
        "decode_tokens": len(generated),
        "loop": ("mlx_trim_single_fused_probe" if single_fused
                 else "mlx_trim_keep_accepted_cuda_parity"),
        "single_fused": bool(single_fused),
        "block_eval_s_first8": block_evals[:8],
        "block_eval_s_max": (round(max(block_evals), 4) if block_evals else None),
        "block_eval_s_mean": (round(sum(block_evals) / len(block_evals), 4)
                              if block_evals else None),
        "time_breakdown_s": {k: round(v, 3) for k, v in timing.items()},
    }


def fused_specdecode_generate_mlx(
    adapter: "MLXRestoredIncrementalVerifier",
    drafter: Any,
    *,
    aux_prompt: Sequence[Any],
    embed_fn: Callable[[Any], Any],
    lm_head_fn: Callable[[Any], Any],
    gen_tokens: int,
    block_size: int,
    eos_ids: Sequence[int] = (),
) -> Dict[str, Any]:
    """All-MLX fused spec decode with ONE host sync per block.

    * ② draft+verify single graph: the drafter's lazy draft ids
      (:meth:`MLXDFlashDrafter.draft_block_ids`) are concatenated with the
      carried bonus in-graph and fed straight into the verifier forward —
      no draft token ever crosses to python before verification.
    * ① in-graph acceptance: the leading-match count is
      ``sum(cumprod(argmax(pred_rows) == candidate))``; the next-position
      logits row is gathered with the lazy count (``mx.take``). The block's
      single ``mx.eval`` materialises exactly three things: the accept
      count, the candidate ids, and nothing else. Drafter-context
      extensions are pushed with ``mx.async_eval`` so Metal works while
      python does bookkeeping.
    * ③ carried correction: on rejection there is NO correction forward.
      ``next_token_logits`` is set to the gathered next-position row, so
      the verifier's own argmax (the correction) becomes the next block's
      bonus — guaranteed-accepted at position 0 of the next verify, where
      its K/V and aux are computed as part of the batched forward.

    Per-block commit = the accepted candidate prefix (position 0, the
    carried bonus/correction, always accepts by construction — every block
    commits >= 1 token, so the loop degrades to AR pace, never below).
    """
    import mlx.core as mx  # type: ignore

    eos = set(int(t) for t in eos_ids)
    C = adapter._past_len
    t_ctx = time.perf_counter()
    ctx_kv = drafter.make_context_kv(list(aux_prompt), mx.arange(0, C))
    mx.async_eval([t for kv in ctx_kv for t in kv])
    timing = {
        "ctx_kv_build_s": time.perf_counter() - t_ctx,
        "build_s": 0.0,   # lazy graph construction (python-side)
        "eval_s": 0.0,    # per-block syncs (Metal compute)
        "extend_s": 0.0,
    }
    adapter._capture_aux = True

    generated: List[int] = []
    accepts: List[int] = []
    # Rollback-carry state: rejected blocks roll the WHOLE forward back
    # (rollback_block — see its docstring for why trim is unsound on the
    # wrapped sliding ring) and carry the stream-committed-but-not-cached
    # tokens (`tail`) into the next candidate, where they are guaranteed
    # re-accepted and their K/V + aux recomputed correctly.
    tail: List[int] = []
    tail_logits = adapter.next_token_logits   # row predicting position S
    ctx_len = C                               # drafter context coverage
    try:
        while len(generated) < gen_tokens:
            L = min(block_size, gen_tokens - len(generated))
            base_fwd = adapter._past_len      # cache offset at forward start
            S = base_fwd + len(tail)          # committed stream length
            t_build = time.perf_counter()
            bonus_id = mx.argmax(tail_logits)            # lazy scalar
            n_draft = max(L - 1, 0)
            if n_draft:
                drafts = drafter.draft_block_ids(
                    ctx_kv, bonus_id, embed_fn, lm_head_fn,
                    n_masks=n_draft, context_len=S)
                check_ids = mx.concatenate([bonus_id[None], drafts])  # [L]
                # Two-phase evaluation: materialise the drafter graph
                # BEFORE building the 26B verify graph. A single fused
                # drafter+verifier graph proved pathological on Metal
                # (command-buffer blowups: 143 s evals in the first live
                # run); two small syncs per block are stable and still
                # ~3× fewer than the eager loop's 6+L.
                mx.eval(check_ids)
            else:
                check_ids = bonus_id[None]
            if tail:
                cand_full = mx.concatenate(
                    [mx.array(tail, dtype=check_ids.dtype), check_ids])
            else:
                cand_full = check_ids
            k = len(tail)
            block_logits = adapter.forward_block_lazy(cand_full[None])  # [k+L, V]
            # In-graph greedy acceptance over the CHECK region only
            # (the carried tail is already stream-committed): row i of
            # pred_rows predicts check_ids[i]; leading-match via cumprod.
            pred_rows = mx.concatenate(
                [tail_logits[None], block_logits[k:k + L - 1]], axis=0)
            matches = (mx.argmax(pred_rows, axis=-1) == check_ids)
            accepted_mx = mx.sum(mx.cumprod(matches.astype(mx.int32)))
            rows = mx.concatenate(
                [tail_logits[None], block_logits[k:]], axis=0)  # [L+1, V]
            next_row = mx.take(rows, accepted_mx[None], axis=0)[0]   # [V]
            timing["build_s"] += time.perf_counter() - t_build
            t_eval = time.perf_counter()
            mx.eval(accepted_mx, check_ids)
            timing["eval_s"] += time.perf_counter() - t_eval
            accepted = int(accepted_mx.item())
            check = [int(x) for x in check_ids.tolist()]
            commit = check[:accepted]
            generated += commit
            accepts.append(accepted)
            tail_logits = next_row
            adapter.next_token_logits = next_row
            aux_rows = adapter._last_aux_mx   # rows for positions base_fwd..base_fwd+k+L
            if accepted == L:
                # Whole forward (tail + check region) is now valid cache.
                adapter._past_len = base_fwd + k + L
                tail = []
            else:
                adapter.rollback_block()      # cache back to base_fwd
                tail = tail + commit          # re-verified next block
            S_new = adapter._past_len + len(tail)
            # Extend the drafter context with aux rows for newly committed
            # positions (ctx_len..S_new). Accepted rows are correct even
            # after rollback: causal attention means rejected positions
            # only sit AFTER them in the forward.
            lo, hi = ctx_len - base_fwd, S_new - base_fwd
            if hi > lo and aux_rows is not None:
                t_extend = time.perf_counter()
                new_aux = [a[lo:hi][None] for a in aux_rows]
                ctx_kv = drafter.extend_context_kv(
                    ctx_kv,
                    drafter.make_context_kv(new_aux, mx.arange(ctx_len, S_new)))
                mx.async_eval([t for kv in ctx_kv for t in kv])
                ctx_len = S_new
                timing["extend_s"] += time.perf_counter() - t_extend
            if any(t in eos for t in commit):
                break
    finally:
        adapter._capture_aux = False
    generated = generated[:gen_tokens]
    return {
        "tokens": generated,
        "blocks": len(accepts),
        "mean_accept_len": (round(sum(accepts) / len(accepts), 3)
                            if accepts else 0.0),
        "decode_tokens": len(generated),
        "loop": "mlx_rollback_carry_v3",
        "time_breakdown_s": {k: round(v, 3) for k, v in timing.items()},
    }


# --------------------------------------------------------------------------- #
# The fused spec-decode loop (control flow; MLX/torch ops via injected fns).
# --------------------------------------------------------------------------- #
def fused_specdecode_generate(
    adapter: Any,
    drafter: Any,
    *,
    aux_prompt: Sequence[Any],
    embed_fn: Callable[[Any], Any],
    lm_head_fn: Callable[[Any], Any],
    gen_tokens: int,
    block_size: int,
    eos_ids: Sequence[int] = (),
    argmax_fn: Callable[[Any], int],
    arange_fn: Callable[[int, int], Any],
    cat_aux_fn: Callable[[Sequence[Any]], Any],
    allow_greedy_fallback: bool = True,
) -> Dict[str, Any]:
    """Run the fused engine. ``adapter`` must already be prefilled. Per block:
    draft from the cached drafter context (B), verify+capture-aux incrementally
    (C+A), accept the longest correct prefix, commit the correction, and EXTEND
    the drafter context with the committed tokens' aux.

    ``argmax_fn`` (logits-row → int), ``arange_fn`` (start, stop → positions),
    and ``cat_aux_fn`` (parts → ``[1, k, hidden]``) abstract the MLX/torch ops so
    the loop is unit-testable.
    """
    n_aux = len(aux_prompt)
    eos = set(int(t) for t in eos_ids)
    C = adapter._past_len
    t_ctx = time.perf_counter()
    ctx_kv = drafter.make_context_kv(list(aux_prompt), arange_fn(0, C))
    timing = {
        "ctx_kv_build_s": time.perf_counter() - t_ctx,
        "draft_s": 0.0,
        "verify_s": 0.0,
        "append_s": 0.0,
        "extend_s": 0.0,
        "fallback_greedy_s": 0.0,
    }
    adapter._capture_aux = True

    generated: List[int] = []
    accepts: List[int] = []
    fallback_to_greedy = False
    try:
        while len(generated) < gen_tokens:
            _kblk_t0 = time.perf_counter()  # agent log (Phase-1)
            L = min(block_size, gen_tokens - len(generated))
            cstart = adapter._past_len
            bonus = int(argmax_fn(adapter.next_token_logits))
            t_draft = time.perf_counter()
            drafts = drafter.draft_block_cached(
                ctx_kv, bonus, embed_fn, lm_head_fn,
                block_size=max(L - 1, 1), context_len=cstart)
            timing["draft_s"] += time.perf_counter() - t_draft
            candidate = [bonus] + list(drafts[: max(L - 1, 0)])
            prev = adapter.next_token_logits
            t_verify = time.perf_counter()
            block_logits = adapter.forward_block(candidate)
            timing["verify_s"] += time.perf_counter() - t_verify
            accepted = 0
            for i in range(len(candidate)):
                if int(argmax_fn(prev)) == candidate[i]:
                    accepted += 1
                    prev = block_logits[i]
                else:
                    break
            adapter.commit_or_truncate(forwarded=len(candidate), accepted=accepted)
            if accepted == len(candidate):
                # The verifier cache already contains the whole accepted block.
                # Reuse the last block logit as the next-token distribution instead
                # of paying an extra correction-token forward.
                adapter.next_token_logits = block_logits[-1]
                new_positions = arange_fn(cstart, cstart + accepted)
                t_extend = time.perf_counter()
                cand_aux = adapter.last_aux_torch_slice(0, accepted)
                # cat_aux_fn of a single part == unsqueeze(0) in the torch
                # path; routing through it keeps this loop runtime-agnostic
                # (the all-MLX drafter path injects an mx-based cat_aux_fn).
                new_aux = [cat_aux_fn([cand_aux[li]]) for li in range(n_aux)]
                ctx_kv = drafter.extend_context_kv(
                    ctx_kv, drafter.make_context_kv(new_aux, new_positions))
                timing["extend_s"] += time.perf_counter() - t_extend
                commit = candidate
            else:
                correction = int(argmax_fn(prev))
                cand_aux = adapter.last_aux_torch_slice(0, accepted)
                t_append = time.perf_counter()
                adapter.append_token(correction)
                timing["append_s"] += time.perf_counter() - t_append
                corr_aux = adapter.last_aux_torch_slice(0, 1)
                new_positions = arange_fn(cstart, cstart + accepted + 1)
                t_extend = time.perf_counter()
                new_aux = [
                    cat_aux_fn([cand_aux[li], corr_aux[li]])
                    for li in range(n_aux)
                ]
                ctx_kv = drafter.extend_context_kv(
                    ctx_kv, drafter.make_context_kv(new_aux, new_positions))
                timing["extend_s"] += time.perf_counter() - t_extend
                commit = candidate[:accepted] + [correction]
            generated += commit
            accepts.append(accepted)
            # #region agent log (Phase-1)
            _kdbg(
                "block",
                loop="torch_ftheta",
                blk=len(accepts) - 1,
                gen=len(generated),
                gen_since_prompt=len(generated),
                past_len=adapter._past_len,
                accepted=accepted,
                L=len(candidate),
                commit_ids=[int(t) for t in commit],
                dt_ms=round((time.perf_counter() - _kblk_t0) * 1e3, 1),
                rep=_kdbg_rep(generated),
                sync=_kdbg_sync(adapter._cache, adapter._past_len),
                lost=_kdbg_lost(adapter._cache,
                                getattr(adapter, "_dbg_restored_positions", set()),
                                getattr(adapter, "_dbg_prompt_len", 0)),
                cache=_kdbg_cache(adapter._cache),
            )
            # #endregion
            if any(t in eos for t in commit):
                break
            if (allow_greedy_fallback and len(accepts) >= 2
                    and (sum(accepts) / len(accepts)) < 1.5):
                fallback_to_greedy = True
                break

        if allow_greedy_fallback and fallback_to_greedy and len(generated) < gen_tokens:
            # Low acceptance makes speculative control flow slower than AR. Finish
            # on the restored verifier cache with plain greedy decode and no aux
            # capture/bridge.
            adapter._capture_aux = False
            t_fb = time.perf_counter()
            while len(generated) < gen_tokens:
                tok = int(argmax_fn(adapter.next_token_logits))
                adapter.append_token(tok)
                generated.append(tok)
                if tok in eos:
                    break
            timing["fallback_greedy_s"] += time.perf_counter() - t_fb
    finally:
        adapter._capture_aux = False
    generated = generated[:gen_tokens]
    # #region agent log (Phase-1: full token dump for offline fused-vs-native
    # divergence comparison + final wide-window degeneration summary)
    _kdbg("final", loop="torch_ftheta", n=len(generated),
          blocks=len(accepts),
          mean_accept_len=(round(sum(accepts) / len(accepts), 3) if accepts else 0.0),
          rep_w128=_kdbg_rep(generated, k=128),
          tokens=[int(t) for t in generated])
    # #endregion
    return {
        "tokens": generated,
        "blocks": len(accepts),
        "mean_accept_len": (round(sum(accepts) / len(accepts), 3)
                            if accepts else 0.0),
        "decode_tokens": len(generated),
        "time_breakdown_s": {k: round(v, 3) for k, v in timing.items()},
    }
