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
    """Temporarily wrap the Gemma-4 ``DecoderLayer.__call__`` so that, when a
    layer carries an ``_aux_record`` dict, its output hidden (the residual
    stream after the layer) is stored at ``_aux_record[layer_idx]``. Restores
    the original ``__call__`` and clears ``_aux_record`` on exit.
    """
    if not text_model.layers:
        yield
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
        self._capture_aux = False

    def reset(self) -> None:
        self._cache = None
        self._past_len = 0
        self.next_token_logits = None
        self._last_aux = None

    def prefill(
        self,
        prompt_ids: Sequence[int],
        *,
        restored_k_per_layer: Dict[int, Any],
        restored_v_per_layer: Dict[int, Any],
        evicted_positions: Sequence[int],
    ) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        self._cache, self.next_token_logits = restored_prefill_cache(
            self.mlx_model, list(prompt_ids),
            restored_k_per_layer=restored_k_per_layer,
            restored_v_per_layer=restored_v_per_layer,
            evicted_positions=evicted_positions,
        )
        self._past_len = len(prompt_ids)

    def forward_block(self, tokens: Sequence[int]) -> Any:
        """Incremental verify of ``tokens`` against the restored cache. Returns
        ``mx [len(tokens), V]`` logits; sets ``_last_aux`` (torch ``[L, hidden]``
        per aux layer) when ``_capture_aux`` and ``aux_layer_ids`` are set."""
        import mlx.core as mx  # type: ignore

        if self._cache is None:
            raise RuntimeError("verifier not prefilled")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        ids = mx.array([list(tokens)])
        want_aux = self._capture_aux and bool(self.aux_layer_ids)
        if want_aux:
            sink: Dict[int, Any] = {}
            with _patched_decoder_layers(self.text_model):
                for layer in self.text_model.layers:
                    layer._aux_record = sink
                logits = self.mlx_model(ids, cache=self._cache)
                aux = _build_aux(self.text_model, ids, sink,
                                 self.embed_scale, self.aux_layer_ids)
                mx.eval(logits)
                mx.eval(aux)
            bridge = self._bridge or (lambda a: a)
            self._last_aux = [bridge(a[0]) for a in aux]   # [L, hidden] each
        else:
            logits = self.mlx_model(ids, cache=self._cache)
            mx.eval(logits)
            self._last_aux = None
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
    ctx_kv = drafter.make_context_kv(list(aux_prompt), arange_fn(0, C))
    adapter._capture_aux = True

    generated: List[int] = []
    accepts: List[int] = []
    while len(generated) < gen_tokens:
        L = min(block_size, gen_tokens - len(generated))
        cstart = adapter._past_len
        bonus = int(argmax_fn(adapter.next_token_logits))
        drafts = drafter.draft_block_cached(
            ctx_kv, bonus, embed_fn, lm_head_fn,
            block_size=max(L - 1, 1), context_len=cstart)
        candidate = [bonus] + list(drafts[: max(L - 1, 0)])
        prev = adapter.next_token_logits
        block_logits = adapter.forward_block(candidate)
        cand_aux = adapter._last_aux
        accepted = 0
        for i in range(len(candidate)):
            if int(argmax_fn(prev)) == candidate[i]:
                accepted += 1
                prev = block_logits[i]
            else:
                break
        correction = int(argmax_fn(prev))
        adapter.commit_or_truncate(forwarded=len(candidate), accepted=accepted)
        adapter.append_token(correction)
        corr_aux = adapter._last_aux
        new_positions = arange_fn(cstart, cstart + accepted + 1)
        new_aux = [
            cat_aux_fn([cand_aux[li][:accepted], corr_aux[li][:1]])
            for li in range(n_aux)
        ]
        ctx_kv = drafter.extend_context_kv(
            ctx_kv, drafter.make_context_kv(new_aux, new_positions))
        commit = candidate[:accepted] + [correction]
        generated += commit
        accepts.append(accepted)
        if any(t in eos for t in commit):
            break

    adapter._capture_aux = False
    generated = generated[:gen_tokens]
    return {
        "tokens": generated,
        "blocks": len(accepts),
        "mean_accept_len": (round(sum(accepts) / len(accepts), 3)
                            if accepts else 0.0),
        "decode_tokens": len(generated),
    }
