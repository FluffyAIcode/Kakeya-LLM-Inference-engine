"""Torch/CUDA ``RestorationDraftEngine`` (ADR 0009 §4 F3, host B on a GPU).

The pure-torch twin of ``inference_engine.backends.mlx.dflash_distributed
.MLXRestorationDraftEngine``: a remote DFlash drafter + f_θ projection that runs
on a CUDA host (no MLX), feeding a gemma-4 MLX verifier on another host. Reuses
the CUDA fused-engine machinery (``CrossModelDLMRestoredVerifier.project_drafter_kv``,
``DFlashDrafter`` context K/V, the Gap-B torch embed/lm_head).

Imports torch + transformers + the v04 stack, so it lives in v04 (not coverage-
gated) and is validated on-device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from inference_engine.distributed.dflash_service import DraftResult, RestoreResult
from inference_engine.distributed.tensor_codec import (
    WireTensor,
    torch_to_wire,
    wire_to_torch,
)


def build_torch_embed_lm_head(verifier_model, softcap):
    """Gap-B torch embed/lm_head over the verifier's tied embedding (no
    ×sqrt(hidden) on embed; tied head + final-logit softcap). Mirrors
    scripts/research/k3_specdecode_gpu_bench._build_embed_lm_head."""
    import torch
    import torch.nn.functional as F

    emb_w = verifier_model.get_input_embeddings().weight.detach()
    head_w = verifier_model.get_output_embeddings().weight.detach()

    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, emb_w).float()

    def lm_head_fn(h: torch.Tensor) -> torch.Tensor:
        logits = (h.to(head_w.dtype) @ head_w.t()).float()
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    return embed_fn, lm_head_fn


@dataclass
class _Session:
    ctx_kv: Any = None


class TorchRestorationDraftEngine:
    """``RestorationDraftEngine`` on a CUDA host: torch DFlash + f_θ + a gemma-4
    verifier (used only for its embedding / drafter-KV capture)."""

    def __init__(
        self, *, verifier_model: Any, drafter: Any, f_theta: Any, device: Any,
        sink: int, window: int, force_f_theta: bool = True,
    ) -> None:
        import torch

        from inference_engine.v04.cross_model_dlm_verifier import (
            CrossModelDLMRestoredVerifier,
            full_attention_layer_indices,
        )

        self._torch = torch
        self.device = device
        self.sink = int(sink)
        self.window = int(window)
        self.force_f_theta = bool(force_f_theta)
        self.drafter = drafter
        self.exact_set = set(full_attention_layer_indices(verifier_model))
        self._restored = CrossModelDLMRestoredVerifier(
            verifier_model=verifier_model, drafter=drafter, f_theta=f_theta,
            sink_size=sink, window_size=window,
            exact_layer_indices=self.exact_set)
        softcap = None
        vcfg = getattr(verifier_model, "config", None)
        for attr in ("final_logit_softcapping",):
            cap = getattr(vcfg, attr, None) if vcfg is not None else None
            if cap is None and vcfg is not None:
                cap = getattr(getattr(vcfg, "text_config", None), attr, None)
            if cap:
                softcap = float(cap)
        self._embed_fn, self._lm_head_fn = build_torch_embed_lm_head(
            verifier_model, softcap)
        self._sessions: Dict[str, _Session] = {}

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
        if not (s5_exact_full_attn and not self.force_f_theta):
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                vk, vv = self._restored.project_drafter_kv(ids)
            for li in range(len(vk)):
                if s5_exact_full_attn and li in self.exact_set:
                    continue  # native cache owns exact (full-attn) layers
                restored.append((li, torch_to_wire(vk[li]), torch_to_wire(vv[li])))
        return RestoreResult(restored=restored, evicted_positions=list(evicted),
                             prompt_len=T)

    def seed_context(
        self, session_id: str, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int:
        torch = self._torch
        aux_t = [wire_to_torch(w).to(self.device) for w in aux]
        pos = torch.tensor(list(positions), device=self.device)
        self._sessions[session_id].ctx_kv = self.drafter.make_context_kv(aux_t, pos)
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
        return int(positions[-1]) + 1 if len(positions) else 0

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
