"""Real-model glue for the distributed DFlash+f_θ path (ADR 0009 §4 F3).

Implements the two model-bound contracts the framework-agnostic distributed
machinery (``inference_engine.distributed.{dflash_service,fused_decode}``) needs:

* :class:`MLXRestorationDraftEngine` — host B: the torch DFlash drafter + f_θ
  projection + verifier embed/lm_head, behind ``RestorationDraftEngine``.
* :class:`MLXRestoringVerifierAdapter` — host A: wraps
  ``MLXRestoredIncrementalVerifier`` as a ``RestoringVerifier``.

Plus :class:`InProcessDFlashProposer`, a ``RemoteDFlashProposer``-shaped object
that calls a local engine directly (no gRPC) — used for the in-process
byte-identical check.

This module imports mlx + torch + the v04 stack, so it lives in the MLX backend
(not coverage-gated) and is validated end-to-end on-device, not by unit tests.
Reuses the exact fused-path helpers from
``scripts/research/k3_integrated_niah_eval_mac.py`` /
``inference_engine.backends.mlx.fused_specdecode`` so the distributed split is
numerically the same engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from inference_engine.distributed.dflash_service import DraftResult, RestoreResult
from inference_engine.distributed.fused_decode import CommitResult
from inference_engine.distributed.tensor_codec import (
    WireTensor,
    mlx_to_wire,
    torch_to_wire,
    wire_to_mlx,
    wire_to_torch,
)


# --------------------------------------------------------------------------- #
# Host B: DFlash drafter + f_θ engine
# --------------------------------------------------------------------------- #
@dataclass
class _Session:
    ctx_kv: Any = None


class MLXRestorationDraftEngine:
    """``RestorationDraftEngine`` backed by a torch DFlash drafter + f_θ, using
    the MLX verifier's embedding for ``embed_fn``/``lm_head_fn`` (host A and B
    share the verifier weights in-process; for a true split host B replicates the
    embedding). Per-session drafter context K/V is held here (host B)."""

    def __init__(
        self,
        *,
        mlx_model: Any,
        text_model: Any,
        drafter: Any,
        f_theta: Any,
        embed_scale: float,
        device: Any,
        sink: int,
        window: int,
        force_f_theta: bool = True,
    ) -> None:
        import torch

        from inference_engine.backends.mlx.cross_model_dlm_verifier import (
            kv_source_layer_map,
            mlx_full_attention_layer_indices,
        )
        from inference_engine.backends.mlx.fused_specdecode import (
            make_bridge_embed_lm_head,
        )
        from scripts.research.k3_dflash_mlx_bridge import mx_to_torch, torch_to_mx

        self._torch = torch
        self.mlx_model = mlx_model
        self.text_model = text_model
        self.drafter = drafter
        self.f_theta = f_theta
        self.fcfg = f_theta.config
        self.embed_scale = float(embed_scale)
        self.device = device
        self.sink = int(sink)
        self.window = int(window)
        self.force_f_theta = bool(force_f_theta)
        self.n_layers = len(text_model.layers)
        self.exact_set = set(mlx_full_attention_layer_indices(text_model))
        self.src_map = kv_source_layer_map(text_model)
        self._mx_to_torch = mx_to_torch
        self._torch_to_mx = torch_to_mx

        softcap = None
        for obj in (getattr(mlx_model, "language_model", None), mlx_model):
            cap = getattr(obj, "final_logit_softcapping", None) if obj is not None else None
            if cap:
                softcap = float(cap)
                break
        self._embed_fn, self._lm_head_fn = make_bridge_embed_lm_head(
            text_model, mx_to_torch=mx_to_torch, torch_to_mx=torch_to_mx,
            device=device, torch_dtype=torch.float32, softcap=softcap)
        self._sessions: Dict[str, _Session] = {}

    # --- prompt-time restoration (capture_drafter_kv + f_θ) ---------------- #
    def _capture_drafter_kv(self, ids: Sequence[int]):
        import mlx.core as mx

        torch = self._torch
        ids_mx = mx.array([list(ids)])
        emb_mx = self.text_model.embed_tokens(ids_mx)
        embedded = self._mx_to_torch(emb_mx, dtype=torch.float32, device=self.device)
        layers = list(self.drafter.layers)
        k_cap: List[Any] = [None] * len(layers)
        v_cap: List[Any] = [None] * len(layers)
        handles = []
        for i, layer in enumerate(layers):
            a = layer.self_attn
            handles.append(a.k_proj.register_forward_hook(
                lambda m, inp, out, i=i: k_cap.__setitem__(i, out.detach())))
            handles.append(a.v_proj.register_forward_hook(
                lambda m, inp, out, i=i: v_cap.__setitem__(i, out.detach())))
        try:
            with torch.no_grad():
                T = embedded.size(1)
                qpos = torch.arange(T, device=self.device)
                h = embedded
                for layer in layers:
                    h = layer(h, qpos, ctx_k=None, ctx_v=None)
        finally:
            for hh in handles:
                hh.remove()
        dh, ddim = self.fcfg.drafter_num_kv_heads, self.fcfg.drafter_head_dim
        d_k = [k_cap[i].view(1, -1, dh, ddim) for i in range(len(layers))]
        d_v = [v_cap[i].view(1, -1, dh, ddim) for i in range(len(layers))]
        return d_k, d_v

    def restore(
        self, session_id: str, prompt_ids: Sequence[int], *,
        sink: int, window: int, s5_exact_full_attn: bool, model_id: str,
    ) -> RestoreResult:
        from inference_engine.v04.kv_merge import compute_evicted_positions

        torch = self._torch
        self._sessions[session_id] = _Session()
        prompt_ids = list(prompt_ids)
        T = len(prompt_ids)
        evicted = compute_evicted_positions(T, self.sink, self.window)
        restored: List[Tuple[int, WireTensor, WireTensor]] = []
        # S5 free lunch: with native exact-layer prefill and no force, the
        # verifier owns all needed K/V and nothing is shipped.
        if not (s5_exact_full_attn and not self.force_f_theta):
            d_k, d_v = self._capture_drafter_kv(prompt_ids)
            with torch.no_grad():
                vk, vv = self.f_theta.forward_kv_pack(d_k, d_v)
            for li in range(self.n_layers):
                if self.src_map[li] != li:
                    continue
                if s5_exact_full_attn and li in self.exact_set:
                    continue  # native cache owns exact (full-attn) layers
                k_mx = self._torch_to_mx(vk[li])
                v_mx = self._torch_to_mx(vv[li])
                restored.append((li, mlx_to_wire(k_mx), mlx_to_wire(v_mx)))
        return RestoreResult(restored=restored, evicted_positions=list(evicted),
                             prompt_len=T)

    def seed_context(
        self, session_id: str, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int:
        torch = self._torch
        aux_t = [wire_to_torch(w).to(self.device) for w in aux]
        pos = torch.tensor(list(positions), device=self.device)
        ctx = self.drafter.make_context_kv(aux_t, pos)
        self._sessions[session_id].ctx_kv = ctx
        return len(positions)

    def draft_block(
        self, session_id: str, *, bonus_token_id: int, context_len: int,
        block_size: int,
    ) -> DraftResult:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        sess = self._sessions[session_id]
        drafts = self.drafter.draft_block_cached(
            sess.ctx_kv, int(bonus_token_id), self._embed_fn, self._lm_head_fn,
            block_size=block_size, context_len=int(context_len))
        return DraftResult(draft_token_ids=[int(t) for t in drafts],
                           forward_passes=1, peak_activation_bytes=0)

    def extend_context(
        self, session_id: str, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int:
        torch = self._torch
        sess = self._sessions[session_id]
        aux_t = [wire_to_torch(w).to(self.device) for w in aux]
        pos = torch.tensor(list(positions), device=self.device)
        new_kv = self.drafter.make_context_kv(aux_t, pos)
        sess.ctx_kv = self.drafter.extend_context_kv(sess.ctx_kv, new_kv)
        return int(positions[-1]) + 1 if len(positions) else context_len_unknown()

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def context_len_unknown() -> int:  # pragma: no cover - defensive; positions never empty
    return 0


# --------------------------------------------------------------------------- #
# Host A: MLX verifier adapter
# --------------------------------------------------------------------------- #
class MLXRestoringVerifierAdapter:
    """``RestoringVerifier`` over ``MLXRestoredIncrementalVerifier``."""

    def __init__(
        self, *, adapter: Any, mlx_model: Any, aux_layer_ids: Sequence[int],
        embed_scale: float, bridge: Any, prefill_chunk_size: int = 512,
    ) -> None:
        import mlx.core as mx

        self._mx = mx
        self.adapter = adapter
        self.mlx_model = mlx_model
        self.aux_layer_ids = tuple(int(a) for a in aux_layer_ids)
        self.embed_scale = float(embed_scale)
        self.bridge = bridge
        self.prefill_chunk_size = int(prefill_chunk_size)
        self._prompt: List[int] = []
        self._cstart = 0
        self._prev = None
        self._block_logits = None
        self._candidate: List[int] = []
        # gemma-4 shares K/V across layers; the MLX verifier injects restored K/V
        # only at "source" layers (src_map[li]==li). A torch host B ships every
        # non-exact layer; filter to what THIS verifier consumes.
        from inference_engine.backends.mlx.cross_model_dlm_verifier import (
            kv_source_layer_map,
            resolve_mlx_text_model,
        )
        _tm = resolve_mlx_text_model(mlx_model)
        _src = kv_source_layer_map(_tm)
        self._source_layers = {li for li in range(len(_src)) if _src[li] == li}

    @property
    def context_len(self) -> int:
        return self.adapter._past_len

    def prefill(
        self, prompt_ids: Sequence[int],
        restored: Sequence[Tuple[int, WireTensor, WireTensor]],
        evicted_positions: Sequence[int],
    ) -> None:
        self._prompt = list(prompt_ids)
        rk: Dict[int, Any] = {}
        rv: Dict[int, Any] = {}
        for layer, k_w, v_w in restored:
            if layer not in self._source_layers:
                continue  # non-source layer (shared K/V) — verifier doesn't inject it
            rk[layer] = wire_to_mlx(k_w)
            rv[layer] = wire_to_mlx(v_w)
        self.adapter.prefill(
            self._prompt, restored_k_per_layer=rk, restored_v_per_layer=rv,
            evicted_positions=list(evicted_positions),
            prefill_chunk_size=self.prefill_chunk_size, full_kv=False)
        self.adapter._capture_aux = True

    def aux_over_prompt(self) -> List[WireTensor]:
        from inference_engine.backends.mlx.fused_specdecode import capture_aux_hidden

        aux_mx = capture_aux_hidden(
            self.mlx_model, self._prompt, self.aux_layer_ids,
            embed_scale=self.embed_scale)
        return [torch_to_wire(self.bridge(a)) for a in aux_mx]

    def next_greedy(self) -> int:
        return int(self._mx.argmax(self.adapter.next_token_logits).item())

    def verify_block(self, candidate: Sequence[int]) -> int:
        mx = self._mx
        candidate = list(candidate)
        self._cstart = self.adapter._past_len
        self._prev = self.adapter.next_token_logits
        self._block_logits = self.adapter.forward_block(candidate)
        self._candidate = candidate
        accepted = 0
        running = self._prev
        for i, tok in enumerate(candidate):
            if int(mx.argmax(running).item()) != tok:
                break
            accepted += 1
            running = self._block_logits[i]
        self._running = running
        return accepted

    def commit(self, accepted: int) -> CommitResult:
        torch_cat = __import__("torch").cat
        cand = self._candidate
        n_aux = len(self.aux_layer_ids)
        self.adapter.commit_or_truncate(forwarded=len(cand), accepted=accepted)
        cand_aux = self.adapter.last_aux_torch_slice(0, accepted)
        if accepted == len(cand):
            self.adapter.next_token_logits = self._block_logits[-1]
            tokens = list(cand)
            new_aux = [torch_cat([cand_aux[li]], dim=0).unsqueeze(0) for li in range(n_aux)]
        else:
            correction = int(self._mx.argmax(self._running).item())
            self.adapter.append_token(correction)
            corr_aux = self.adapter.last_aux_torch_slice(0, 1)
            tokens = list(cand[:accepted]) + [correction]
            new_aux = [
                torch_cat([cand_aux[li], corr_aux[li]], dim=0).unsqueeze(0)
                for li in range(n_aux)
            ]
        positions = list(range(self._cstart, self._cstart + len(tokens)))
        aux_wires = [torch_to_wire(a) for a in new_aux]
        return CommitResult(tokens=tokens, aux=aux_wires, positions=positions, stop=False)


# --------------------------------------------------------------------------- #
# In-process proposer (no gRPC) for the byte-identical check
# --------------------------------------------------------------------------- #
class InProcessDFlashProposer:
    """``RemoteDFlashProposer``-shaped wrapper calling a local engine directly."""

    def __init__(self, engine: MLXRestorationDraftEngine, *, session_id: str = "inproc",
                 sink: int = 4, window: int = 64) -> None:
        self.engine = engine
        self.session_id = session_id
        self.sink = sink
        self.window = window

    def restore(self, prompt_ids, *, sink, window, s5_exact_full_attn=True) -> RestoreResult:
        return self.engine.restore(
            self.session_id, prompt_ids, sink=sink, window=window,
            s5_exact_full_attn=s5_exact_full_attn, model_id="")

    def seed_context(self, aux, positions) -> int:
        return self.engine.seed_context(self.session_id, aux, positions)

    def draft_block(self, *, bonus_token_id, context_len, block_size) -> DraftResult:
        return self.engine.draft_block(
            self.session_id, bonus_token_id=bonus_token_id,
            context_len=context_len, block_size=block_size)

    def extend_context(self, aux, positions) -> int:
        return self.engine.extend_context(self.session_id, aux, positions)

    def close(self) -> None:
        self.engine.close_session(self.session_id)
