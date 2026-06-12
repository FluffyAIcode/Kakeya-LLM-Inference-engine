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

    def reset(self) -> None:
        self._cache = None
        self._past_len = 0
        self.next_token_logits = None
        self._last_aux = None
        self._last_aux_mx = None

    def prefill(
        self,
        prompt_ids: Sequence[int],
        *,
        restored_k_per_layer: Dict[int, Any],
        restored_v_per_layer: Dict[int, Any],
        evicted_positions: Sequence[int],
        prefill_chunk_size: int = 0,
    ) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        self._cache, self.next_token_logits = restored_prefill_cache(
            self.mlx_model, list(prompt_ids),
            restored_k_per_layer=restored_k_per_layer,
            restored_v_per_layer=restored_v_per_layer,
            evicted_positions=evicted_positions,
            prefill_chunk_size=prefill_chunk_size,
        )
        self._past_len = len(prompt_ids)

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

    def forward_block_lazy(self, ids_mx: Any) -> Any:
        """LAZY incremental verify: ``ids_mx`` is an mx ``[1, L]`` (typically
        the in-graph concatenation of the carried bonus + lazy draft ids —
        lever ② of the single-sync loop). Returns ``mx [L, V]`` logits with
        NO evaluation; aux hidden (when ``_capture_aux``) stays lazy in
        ``_last_aux_mx`` and is consumed lazily by the drafter-context
        extension."""
        if self._cache is None:
            raise RuntimeError("verifier not prefilled")
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
            trim_prompt_cache(self._cache, drop)
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
        "eval_s": 0.0,    # the per-block single sync (Metal compute)
        "extend_s": 0.0,
    }
    adapter._capture_aux = True

    generated: List[int] = []
    accepts: List[int] = []
    next_logits = adapter.next_token_logits  # mx [V], may be lazy
    try:
        while len(generated) < gen_tokens:
            L = min(block_size, gen_tokens - len(generated))
            cstart = adapter._past_len
            t_build = time.perf_counter()
            bonus_id = mx.argmax(next_logits)            # lazy scalar
            n_draft = max(L - 1, 0)
            if n_draft:
                drafts = drafter.draft_block_ids(
                    ctx_kv, bonus_id, embed_fn, lm_head_fn,
                    n_masks=n_draft, context_len=cstart)
                candidate = mx.concatenate([bonus_id[None], drafts])  # [L]
            else:
                candidate = bonus_id[None]
            block_logits = adapter.forward_block_lazy(candidate[None])  # [L, V]
            # In-graph greedy acceptance: row i of pred_rows predicts
            # candidate[i]; leading-match count via cumprod.
            pred_rows = mx.concatenate(
                [next_logits[None], block_logits[:-1]], axis=0)  # [L, V]
            matches = (mx.argmax(pred_rows, axis=-1) == candidate)
            accepted_mx = mx.sum(mx.cumprod(matches.astype(mx.int32)))
            # Logits predicting position cstart+accepted (the carried
            # bonus/correction source for the next block).
            rows = mx.concatenate([next_logits[None], block_logits], axis=0)
            next_row = mx.take(rows, accepted_mx[None], axis=0)[0]   # [V]
            timing["build_s"] += time.perf_counter() - t_build
            # ---- the block's single host sync ----
            t_eval = time.perf_counter()
            mx.eval(accepted_mx, candidate)
            timing["eval_s"] += time.perf_counter() - t_eval
            accepted = int(accepted_mx.item())
            cand = [int(x) for x in candidate.tolist()]
            adapter.commit_or_truncate(forwarded=L, accepted=accepted)
            commit = cand[:accepted]
            generated += commit
            accepts.append(accepted)
            next_logits = next_row
            adapter.next_token_logits = next_row
            if accepted and adapter._last_aux_mx is not None:
                t_extend = time.perf_counter()
                new_aux = [a[0:accepted][None] for a in adapter._last_aux_mx]
                ctx_kv = drafter.extend_context_kv(
                    ctx_kv,
                    drafter.make_context_kv(
                        new_aux, mx.arange(cstart, cstart + accepted)))
                mx.async_eval([t for kv in ctx_kv for t in kv])
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
        "loop": "mlx_single_sync_v2",
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
    return {
        "tokens": generated,
        "blocks": len(accepts),
        "mean_accept_len": (round(sum(accepts) / len(accepts), 3)
                            if accepts else 0.0),
        "decode_tokens": len(generated),
        "time_breakdown_s": {k: round(v, 3) for k, v in timing.items()},
    }
