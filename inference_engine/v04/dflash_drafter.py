"""K3 — native DFlash drafter for Kakeya speculative decoding.

This module ports the DFlash block-diffusion drafter
(``z-lab/gemma-4-26B-A4B-it-DFlash``) into the Kakeya inference engine so
DFlash speculative decoding runs natively — *not* via vLLM/SGLang.

DFlash is an EAGLE-3-style drafter: a small (5-layer) Qwen3 transformer
that consumes the **verifier's** hidden states from several auxiliary
layers, projects them with a learned ``fc`` (concat of aux layers →
hidden) + ``hidden_norm``, and then drafts a *block* of tokens in
parallel by **block-diffusion** denoising (non-causal attention over the
block, ``mask_token_id`` initialisation). It owns **no embeddings and no
lm_head** — it shares the verifier's (with Gemma's ``×sqrt(hidden)``
embedding scaling and ``final_logit_softcapping``).

Architecture (reverse-engineered from the checkpoint weights + vLLM
PR #41703 ``qwen3_dflash.py``):

* weights: ``layers.{0..4}.*`` (Qwen3: q/k/v/o_proj, q_norm/k_norm,
  input_layernorm, post_attention_layernorm, mlp.gate/up/down_proj),
  plus top-level ``fc`` ``[hidden, num_aux*hidden]``, ``hidden_norm``
  ``[hidden]``, ``norm`` ``[hidden]``.
* aux layers: ``dflash_config.target_layer_ids`` are stored unshifted;
  the **HF/vLLM-correct semantics shift them +1** — PR #41703 measured
  44.7 % acceptance / 7.70 length with the ``+1`` shift vs 37.3 % / 6.60
  without. :attr:`DFlashConfig.aux_layer_ids` returns the shifted ids.
* draft attention is **non-causal** within the block (PR #41703:
  "DFlash uses non-causal attention").

**Fidelity status.** The backbone (Qwen3 layers + ``fc``/``hidden_norm``/
``norm``), weight loading, aux-layer selection, RoPE/GQA/RMSNorm, and the
non-causal block mask are faithful and unit-tested. The exact
EAGLE-3↔block fusion and the denoising schedule are a documented,
principled implementation (low-confidence remasking, like the MDLM
proposer, but non-causal and conditioned on the projected aux hidden);
matching the reference acceptance profile is the Stage-2 H200 validation
task and may require reconciliation against ``qwen3_dflash.py``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the engine's existing speculative-decoding block contract so the
# DFlash proposer is a drop-in for the SpeculativeDecoder loop.
from kv_cache_proposer.proposer import BlockProposal


# ===========================================================================
# Config
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class DFlashConfig:
    """DFlash drafter config, parsed from the checkpoint ``config.json``."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    # DFlash-specific
    block_size: int
    mask_token_id: int
    target_layer_ids: Tuple[int, ...]
    final_logit_softcapping: Optional[float]
    # Gemma-family target shares embeddings; scale by sqrt(hidden) on lookup.
    embed_scale_sqrt_hidden: bool = True

    @property
    def aux_layer_ids(self) -> Tuple[int, ...]:
        """Verifier layers whose hidden states feed the drafter.

        The checkpoint stores ``target_layer_ids`` unshifted, but the
        HF/vLLM-correct semantics use a ``+1`` shift (PR #41703). These are
        the *verifier* decoder-layer output indices to capture.
        """
        return tuple(t + 1 for t in self.target_layer_ids)

    @property
    def num_aux_layers(self) -> int:
        return len(self.target_layer_ids)

    @property
    def fc_in_features(self) -> int:
        return self.num_aux_layers * self.hidden_size

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "DFlashConfig":
        dfl = cfg.get("dflash_config", {}) or {}
        target_layer_ids = dfl.get("target_layer_ids")
        if not target_layer_ids:
            raise ValueError(
                "config.json has no dflash_config.target_layer_ids; this does "
                "not look like a DFlash drafter checkpoint."
            )
        block_size = dfl.get("block_size", cfg.get("block_size"))
        if block_size is None:
            raise ValueError("DFlash config missing block_size.")
        mask_token_id = dfl.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError("DFlash config missing dflash_config.mask_token_id.")
        head_dim = cfg.get("head_dim")
        if head_dim is None:
            head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        return cls(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(head_dim),
            intermediate_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 8192)),
            block_size=int(block_size),
            mask_token_id=int(mask_token_id),
            target_layer_ids=tuple(int(x) for x in target_layer_ids),
            final_logit_softcapping=(
                float(cfg["final_logit_softcapping"])
                if cfg.get("final_logit_softcapping") is not None
                else None
            ),
        )

    @classmethod
    def from_pretrained(cls, model_id_or_path: str) -> "DFlashConfig":
        """Load config from a local dir or an HF hub id (downloads only
        ``config.json``)."""
        local = Path(model_id_or_path) / "config.json"
        if local.is_file():
            cfg = json.loads(local.read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download

            cfg = json.loads(
                Path(hf_hub_download(model_id_or_path, "config.json")).read_text(
                    encoding="utf-8",
                )
            )
        return cls.from_hf_config(cfg)


# ===========================================================================
# Building blocks (Qwen3-style, self-contained so we can load the raw
# checkpoint by weight name without depending on transformers internals)
# ===========================================================================


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def _rope_cos_sin(
    position_ids: torch.Tensor, head_dim: int, theta: float,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard rotary embedding tables for ``position_ids`` ``[T]`` →
    ``(cos, sin)`` each ``[T, head_dim]``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    freqs = position_ids.float()[:, None] * inv_freq[None, :]  # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


# Query-chunk size for the drafter's non-causal attention. Bounds peak
# attention memory to O(q_chunk × (C+T)); tune down on tight-memory hosts
# (e.g. 24 GB Mac at long context). 0 ⇒ no chunking (single SDPA call).
# Override at runtime with KAKEYA_DFLASH_ATTN_QCHUNK (e.g. 256 on a 24 GB Mac).
import os as _os
_ATTN_Q_CHUNK = int(_os.environ.get("KAKEYA_DFLASH_ATTN_QCHUNK", "1024") or "1024")


def _chunked_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, scale: float, q_chunk: Optional[int] = None,
) -> torch.Tensor:
    """Non-causal SDPA computed in query-dimension chunks.

    ``q`` is ``[B, nh, T, hd]``; ``k``/``v`` ``[B, nh, C+T, hd]``. Returns
    ``[B, nh, T, hd]``. Chunking the query dim keeps the (possibly
    materialised) score tensor at ``[B, nh, q_chunk, C+T]`` so long-context
    attention does not OOM on hosts/kernels without a flash path (MPS).
    """
    T = q.shape[-2]
    if not q_chunk or q_chunk <= 0 or T <= q_chunk:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=scale,
        )
    outs = []
    for start in range(0, T, q_chunk):
        qc = q[:, :, start:start + q_chunk, :]
        outs.append(F.scaled_dot_product_attention(
            qc, k, v, attn_mask=None, is_causal=False, scale=scale,
        ))
    return torch.cat(outs, dim=2)


class _DFlashAttention(nn.Module):
    """DFlash draft attention (faithful to vLLM ``DFlashQwen3Attention``).

    Context K/V are precomputed from the target's combined hidden states
    (see :meth:`project_context_kv`) and prepended to the query tokens'
    own K/V; the query attends **non-causally** over ``[context ++ query]``.
    """

    def __init__(self, cfg: DFlashConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        # Qwen3 applies RMSNorm on the head_dim of q and k.
        self.q_norm = _RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = _RMSNorm(self.hd, cfg.rms_norm_eps)
        self.scale = self.hd ** -0.5

    def project_context_kv(
        self, ctx_normed: torch.Tensor, ctx_positions: torch.Tensor,
    ):
        """Project the (already ``hidden_norm``-ed) target context hidden to
        this layer's K/V, apply ``k_norm`` + RoPE at ``ctx_positions``.

        Mirrors ``precompute_and_store_context_kv``: context K/V come from
        the target hidden via each draft layer's k/v_proj (not from the
        draft tokens). Returns ``(ctx_k, ctx_v)`` each ``[B, nkv, C, hd]``.
        """
        B, C, _ = ctx_normed.shape
        k = self.k_proj(ctx_normed).view(B, C, self.nkv, self.hd)
        v = self.v_proj(ctx_normed).view(B, C, self.nkv, self.hd)
        k = self.k_norm(k).transpose(1, 2)  # [B, nkv, C, hd]
        v = v.transpose(1, 2)
        cos, sin = _rope_cos_sin(
            ctx_positions, self.hd, self.theta, ctx_normed.device, k.dtype,
        )
        k = _apply_rope(k, cos, sin)
        return k, v

    def forward(
        self, h: torch.Tensor, query_positions: torch.Tensor,
        ctx_k: torch.Tensor, ctx_v: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = h.shape
        q = self.q_proj(h).view(B, T, self.nh, self.hd)
        k = self.k_proj(h).view(B, T, self.nkv, self.hd)
        v = self.v_proj(h).view(B, T, self.nkv, self.hd)
        q = self.q_norm(q).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_norm(k).transpose(1, 2)  # [B, nkv, T, hd]
        v = v.transpose(1, 2)
        cos, sin = _rope_cos_sin(
            query_positions, self.hd, self.theta, h.device, q.dtype,
        )
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # Prepend the precomputed context K/V (from target hidden).
        if ctx_k is not None:
            k = torch.cat([ctx_k.to(k.dtype), k], dim=2)  # [B, nkv, C+T, hd]
            v = torch.cat([ctx_v.to(v.dtype), v], dim=2)
        # GQA: expand kv heads to query heads.
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # Non-causal (queries see all context + all query positions), no mask.
        # Use SDPA, and **chunk over the query dimension** so peak attention
        # memory stays O(chunk × (C+T)) instead of O(T × (C+T)). The full
        # materialisation OOMs at long context (≈5 GB at T≈6k, nh=32) — and
        # MPS's SDPA has no flash kernel for this shape, so it materialises
        # too; query-chunking bounds it on every device/kernel.
        out = _chunked_sdpa(q, k, v, scale=self.scale, q_chunk=_ATTN_Q_CHUNK)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o_proj(out)


class _DFlashMLP(nn.Module):
    def __init__(self, cfg: DFlashConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DFlashLayer(nn.Module):
    def __init__(self, cfg: DFlashConfig) -> None:
        super().__init__()
        self.input_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = _DFlashAttention(cfg)
        self.post_attention_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = _DFlashMLP(cfg)

    def forward(self, h, query_positions, ctx_k, ctx_v):
        h = h + self.self_attn(
            self.input_layernorm(h), query_positions, ctx_k, ctx_v,
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


# ===========================================================================
# DFlash drafter
# ===========================================================================


class DFlashDrafter(nn.Module):
    """The DFlash block-diffusion drafter backbone.

    Owns the 5 Qwen3 layers + ``fc`` (aux-layer projection) +
    ``hidden_norm`` + final ``norm``. Embeddings and the lm_head are
    supplied by the verifier at draft time (DFlash shares the target's).
    """

    def __init__(self, cfg: DFlashConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [_DFlashLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.fc = nn.Linear(cfg.fc_in_features, cfg.hidden_size, bias=False)
        self.hidden_norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    # -- aux fusion (EAGLE-3 combine_hidden_states) ------------------------
    def combine_aux(self, aux_hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concat the ``num_aux`` verifier hidden states ``[B, C, hidden]``
        along the feature dim → ``fc`` → ``[B, C, hidden]``.

        Faithful to ``DFlashQwen3ForCausalLM.combine_hidden_states``: this is
        ``fc`` only — ``hidden_norm`` is applied later in
        :meth:`precompute_context_kv` (mirroring
        ``precompute_and_store_context_kv``), NOT here.
        """
        if len(aux_hidden_states) != self.cfg.num_aux_layers:
            raise ValueError(
                f"expected {self.cfg.num_aux_layers} aux hidden states "
                f"(one per aux layer {self.cfg.aux_layer_ids}), got "
                f"{len(aux_hidden_states)}"
            )
        cat = torch.cat(list(aux_hidden_states), dim=-1)  # [B, C, num_aux*hidden]
        if cat.shape[-1] != self.cfg.fc_in_features:
            raise ValueError(
                f"aux concat feature dim {cat.shape[-1]} != fc_in_features "
                f"{self.cfg.fc_in_features}"
            )
        return self.fc(cat.to(self.fc.weight.dtype))

    # -- context K/V precompute (from target hidden) -----------------------
    def precompute_context_kv(
        self, context_states: torch.Tensor, ctx_positions: torch.Tensor,
    ):
        """Per-layer context K/V from the combined target hidden.

        ``hidden_norm`` is applied ONCE to ``context_states`` (matching the
        fused-buffer path in the reference), then each draft layer projects
        its own K/V (+ ``k_norm`` + RoPE at ``ctx_positions``). Returns a
        list of ``(ctx_k, ctx_v)`` per layer, each ``[B, nkv, C, hd]``.
        """
        ctx_normed = self.hidden_norm(context_states.to(self.hidden_norm.weight.dtype))
        return [
            layer.self_attn.project_context_kv(ctx_normed, ctx_positions)
            for layer in self.layers
        ]

    # -- transformer forward over query tokens -----------------------------
    def _run_layers(
        self, hidden: torch.Tensor, query_positions: torch.Tensor, ctx_kv,
    ) -> torch.Tensor:
        for layer, (ck, cv) in zip(self.layers, ctx_kv):
            hidden = layer(hidden, query_positions, ck, cv)
        return self.norm(hidden)

    # -- weight loading ----------------------------------------------------
    @torch.no_grad()
    def load_state_dict_from_hf(self, state: dict, *, strict: bool = True) -> None:
        """Load a checkpoint state dict whose keys match the HF DFlash
        layout (``layers.N.self_attn.q_proj.weight`` etc., plus ``fc``,
        ``hidden_norm``, ``norm``)."""
        own = dict(self.named_parameters())
        missing, unexpected = [], []
        for k in own:
            if k not in state:
                missing.append(k)
        for k in state:
            if k not in own:
                unexpected.append(k)
        if strict and (missing or unexpected):
            raise ValueError(
                f"DFlash weight mismatch: missing={missing[:6]} "
                f"unexpected={unexpected[:6]}"
            )
        for k, p in own.items():
            if k in state:
                p.copy_(state[k].to(p.dtype))

    @classmethod
    def from_pretrained(
        cls, model_id_or_path: str, *, dtype: torch.dtype = torch.bfloat16,
    ) -> "DFlashDrafter":
        """Build + load a DFlash drafter from a local dir or HF hub id."""
        cfg = DFlashConfig.from_pretrained(model_id_or_path)
        model = cls(cfg).to(dtype)
        local = Path(model_id_or_path) / "model.safetensors"
        if local.is_file():
            path = str(local)
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id_or_path, "model.safetensors")
        from safetensors.torch import load_file

        model.load_state_dict_from_hf(load_file(path), strict=True)
        model.eval()
        return model

    # -- parallel block drafting (single non-causal forward) ---------------
    @torch.no_grad()
    def draft_block(
        self,
        aux_hidden_context: Sequence[torch.Tensor],
        bonus_token_id: int,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        block_size: int,
    ) -> List[int]:
        """Draft ``block_size`` tokens in a **single non-causal forward**.

        Faithful to ``DFlashQwen3Model`` (vLLM PR #41703): the verifier's
        combined context hidden become per-layer context K/V (prewritten);
        the query tokens ``[last_token, mask×block_size]`` attend non-causally
        to that context + each other and predict the block in one pass — no
        in-model mask-denoise loop.

        Parameters
        ----------
        aux_hidden_context
            ``num_aux`` tensors ``[1, C, hidden]`` — verifier aux-layer hidden
            at **all** committed positions ``0..C-1``.
        bonus_token_id
            The verifier's greedy next token (``t_C``, guaranteed-correct
            first token). It is the bonus query at position ``C``; the
            block-diffusion masks then occupy positions ``C+1..C+block_size``.
        embed_fn
            Verifier token embedding ``[*, T] -> [*, T, hidden]`` (incl.
            Gemma ``×sqrt(hidden)`` scaling).
        lm_head_fn
            Verifier logits head ``[*, hidden] -> [*, vocab]`` (incl.
            ``final_logit_softcapping``).

        Returns ``block_size`` drafted tokens for positions
        ``C+1..C+block_size``. Per ``copy_and_expand_dflash_inputs_kernel``,
        the sampled (drafted) tokens are the MASK positions
        (``query_off > 0``), and the bonus query sits at ``last_pos+1 == C``.
        """
        logits = self.draft_logits(
            aux_hidden_context, bonus_token_id, embed_fn, lm_head_fn,
            block_size=block_size,
        ).clone()
        logits[..., self.cfg.mask_token_id] = float("-inf")  # never draft the sentinel
        return torch.argmax(logits[0], dim=-1).tolist()

    # -- fused-engine fast path: draft from a PRECOMPUTED context K/V cache --
    def make_context_kv(
        self, aux_hidden_context: Sequence[torch.Tensor], positions: torch.Tensor,
    ):
        """Per-layer context K/V for ``positions`` from their aux hidden.

        ``combine_aux`` (fc) + ``precompute_context_kv`` (hidden_norm + per-layer
        k/v_proj + k_norm + RoPE). Returns a per-layer list of ``(k, v)``, each
        ``[B, nkv, len(positions), hd]``. Use once at prefill, then
        :meth:`extend_context_kv` incrementally for newly-committed tokens — so
        the drafter never re-scans the whole committed prefix (O(L)/block, not
        O(C)). This is component B of the fused spec-decode engine.
        """
        ctx_states = self.combine_aux(aux_hidden_context)
        return self.precompute_context_kv(ctx_states, positions)

    @staticmethod
    def extend_context_kv(ctx_kv, new_kv):
        """Append per-layer ``new_kv`` (from :meth:`make_context_kv`) to the
        running ``ctx_kv`` cache along the sequence axis."""
        out = []
        for (ck, cv), (nk, nv) in zip(ctx_kv, new_kv):
            out.append((
                torch.cat([ck, nk.to(ck.dtype)], dim=2),
                torch.cat([cv, nv.to(cv.dtype)], dim=2),
            ))
        return out

    @torch.no_grad()
    def draft_block_cached(
        self,
        ctx_kv,
        bonus_token_id: int,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        block_size: int,
        context_len: int,
    ) -> List[int]:
        """Draft ``block_size`` tokens using a PRECOMPUTED per-layer context
        K/V cache (``ctx_kv`` covering positions ``0..context_len-1``).

        Same single non-causal pass as :meth:`draft_block`, but skips the
        O(C) context K/V recompute — the caller maintains ``ctx_kv``
        incrementally. Cost is O(block_size) on the drafter.
        """
        cfg = self.cfg
        device = ctx_kv[0][0].device
        query_ids = torch.tensor(
            [[int(bonus_token_id)] + [cfg.mask_token_id] * block_size],
            dtype=torch.long, device=device,
        )
        query_positions = torch.arange(
            context_len, context_len + 1 + block_size, device=device,
        )
        h = embed_fn(query_ids).to(self.fc.weight.dtype)
        h = self._run_layers(h, query_positions, ctx_kv)
        logits = lm_head_fn(h).clone()
        logits[..., cfg.mask_token_id] = float("-inf")
        return torch.argmax(logits[0, 1:1 + block_size], dim=-1).tolist()

    def draft_logits(
        self,
        aux_hidden_context: Sequence[torch.Tensor],
        bonus_token_id: int,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        block_size: int,
    ) -> torch.Tensor:
        """Grad-enabled forward → mask-position logits ``[1, block_size, vocab]``.

        Same single non-causal pass as :meth:`draft_block` but differentiable
        (no ``no_grad`` wrapper), so the projection / norms can be trained to
        align the drafter to a verifier (K3 ``f_θ`` alignment; see
        ``docs/design/k3-f-theta-training-pipeline.md``). Gradients flow into
        whichever drafter params are left trainable.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        cfg = self.cfg
        context_states = self.combine_aux(aux_hidden_context)  # [1, C, hidden]
        B, C, _ = context_states.shape
        device = context_states.device
        ctx_positions = torch.arange(C, device=device)
        ctx_kv = self.precompute_context_kv(context_states, ctx_positions)

        # Query = [bonus token (t_C), mask × block_size]; bonus at position C,
        # masks at C+1..C+block_size (kernel: query_pos = last_pos + 1 + off).
        query_ids = torch.tensor(
            [[int(bonus_token_id)] + [cfg.mask_token_id] * block_size],
            dtype=torch.long, device=device,
        )
        query_positions = torch.arange(C, C + 1 + block_size, device=device)
        h = embed_fn(query_ids).to(self.fc.weight.dtype)  # [1, 1+block_size, hidden]
        h = self._run_layers(h, query_positions, ctx_kv)  # [1, 1+block_size, hidden]
        logits = lm_head_fn(h)  # [1, 1+block_size, vocab]
        # Mask positions query_off=1..block_size are the drafts.
        return logits[:, 1:1 + block_size, :]


# ===========================================================================
# Proposer adapter (engine spec-decode `propose_block` contract)
# ===========================================================================


class AuxHiddenProvider:
    """Contract for the object that supplies verifier aux-layer hidden
    states at **all** committed positions.

    DFlash turns the target's context hidden into the draft layers'
    prewritten context K/V, so the drafter needs the aux hidden over the
    whole committed prefix (not just the last position). Wired to the
    gemma-4 verifier in Stage 2 (capturing decoder-layer outputs at
    :attr:`DFlashConfig.aux_layer_ids`); tests inject a synthetic provider.
    """

    def aux_hidden_context(
        self, committed_token_ids: List[int],
    ) -> Tuple[List[torch.Tensor], int]:
        """Return ``(aux_list, bonus_token_id)`` for ``committed``.

        ``aux_list``: ``num_aux`` tensors ``[1, C, hidden]``.
        ``bonus_token_id``: the verifier's greedy next token ``t_C`` (the
        guaranteed-correct first token of the upcoming block).
        """
        raise NotImplementedError


class DFlashProposer:
    """DFlash drafter as a Kakeya ``propose_block`` proposer.

    Drop-in for ``SpeculativeDecoder``: ``propose_block(committed, L, steps)
    -> BlockProposal``. Unlike the standalone MDLM ``DLMProposer``, DFlash is
    EAGLE-style and needs the verifier's aux hidden states, supplied via an
    :class:`AuxHiddenProvider` + the verifier's shared embed/lm_head.
    """

    def __init__(
        self,
        drafter: DFlashDrafter,
        aux_provider: AuxHiddenProvider,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.drafter = drafter
        self.aux_provider = aux_provider
        self.embed_fn = embed_fn
        self.lm_head_fn = lm_head_fn

    @torch.no_grad()
    def propose_block(
        self, committed_token_ids: List[int], block_size: int, num_steps: int,
    ) -> BlockProposal:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not committed_token_ids:
            raise ValueError("committed_token_ids must be non-empty (need a bonus token)")
        aux_ctx, bonus_token_id = self.aux_provider.aux_hidden_context(committed_token_ids)
        device = _detect_device(self.drafter)
        _reset_peak_memory(device)
        # DFlash drafts the whole block in ONE non-causal forward (parallel
        # drafting); num_steps is accepted for interface compatibility but
        # the reference uses a single pass. The returned tokens are the drafts
        # for positions C+1..C+block_size; the bonus (t_C) is the always-correct
        # first token handled by the spec-decode loop.
        tokens = self.drafter.draft_block(
            aux_ctx, bonus_token_id, self.embed_fn, self.lm_head_fn,
            block_size=block_size,
        )
        peak = _peak_memory_bytes(device)
        if len(tokens) != block_size:  # pragma: no cover - draft_block guarantees
            raise RuntimeError(
                f"DFlash drafted {len(tokens)} tokens; expected {block_size}."
            )
        return BlockProposal(
            tokens=tokens,
            diffusion_steps=1,
            forward_passes=1,
            peak_activation_bytes=peak,
        )


# ===========================================================================
# Platform-aware peak memory measurement
# ===========================================================================
#
# DFlashProposer.propose_block records peak activation bytes during
# the draft forward as part of BlockProposal — used by the engine's
# spec-decode harness for memory accounting. The original
# implementation called ``torch.cuda.reset_peak_memory_stats`` /
# ``torch.cuda.max_memory_allocated`` directly, which silently
# returned 0 on non-CUDA devices (Mac MPS, CPU). The Mac speculative-
# decoding eval (post-merge follow-up PR) needs honest peak memory
# numbers on Apple Silicon, so the measurement is now platform-aware:
#
#   CUDA  → torch.cuda.{reset_peak_memory_stats, max_memory_allocated}
#   MPS   → torch.mps.{driver_allocated_memory before/after} delta
#           (MPS has no peak counter; we measure the live allocation
#           delta around the forward, which is a tight upper bound
#           on activations released after the forward — close enough
#           for spec-decode-loop memory accounting)
#   CPU   → None (no transient-tensor memory accounting on CPU; the
#           field is left at 0 to signal "unmeasured" rather than
#           lying with a fake 0)


def _detect_device(model: nn.Module) -> str:
    """Detect which compute device the model's parameters live on.

    Returns one of ``"cuda"`` / ``"mps"`` / ``"cpu"``. Raises
    ``RuntimeError`` if the model has no parameters (defensive — every
    real DFlashDrafter has parameters).
    """
    try:
        p = next(model.parameters())
    except StopIteration:
        raise RuntimeError(
            "_detect_device: model has no parameters; cannot infer device"
        )
    return p.device.type


def _reset_peak_memory(device: str) -> None:
    """Reset the peak-memory counter for the device (CUDA only)."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # MPS: no peak counter exposed; we capture pre-forward allocation
    # in _peak_memory_bytes via driver_allocated_memory. Initialised
    # implicitly by the caller via the post-forward read minus a
    # snapshot taken here in a thread-local. To keep the API simple
    # and stateless, we do NOT snapshot here for MPS — the post-
    # forward read alone is the absolute peak under the assumption
    # that the proposer is the dominant memory consumer in its own
    # process (true for spec-decode loops where drafter + verifier
    # are the only large tensors). If a stricter delta is needed,
    # the caller can wrap propose_block with their own MPS allocator
    # snapshot via torch.mps.driver_allocated_memory before the call.
    # CPU: nothing to reset.


def _peak_memory_bytes(device: str) -> int:
    """Return the peak allocation since the last reset, in bytes.

    Returns 0 (unmeasured) on CPU and on devices where the runtime
    doesn't expose a peak counter. Returns int(driver_allocated_memory)
    on MPS — see :func:`_reset_peak_memory` docstring for the caveat
    that this is the post-forward live allocation rather than a true
    peak across the forward.
    """
    if device == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch, "mps"):
        try:
            return int(torch.mps.driver_allocated_memory())
        except Exception:
            return 0
    # CPU and unknown devices: no peak counter; return 0 to signal
    # "unmeasured". Callers that care about CPU peak measurement
    # should track it externally via psutil or tracemalloc.
    return 0
