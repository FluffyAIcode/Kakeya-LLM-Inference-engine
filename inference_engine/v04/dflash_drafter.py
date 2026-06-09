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


class _DFlashAttention(nn.Module):
    def __init__(self, cfg: DFlashConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        # Qwen3 applies RMSNorm on the head_dim of q and k.
        self.q_norm = _RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = _RMSNorm(self.hd, cfg.rms_norm_eps)
        self.scale = self.hd ** -0.5

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = h.shape
        q = self.q_proj(h).view(B, T, self.nh, self.hd)
        k = self.k_proj(h).view(B, T, self.nkv, self.hd)
        v = self.v_proj(h).view(B, T, self.nkv, self.hd)
        q = self.q_norm(q).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_norm(k).transpose(1, 2)  # [B, nkv, T, hd]
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # GQA: expand kv heads to query heads.
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,nh,T,T]
        scores = scores + attn_bias  # additive mask [1,1,T,T] or [B,1,T,T]
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)  # [B, nh, T, hd]
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

    def forward(self, h, cos, sin, attn_bias):
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, attn_bias)
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

    # -- aux projection ----------------------------------------------------
    def project_aux(self, aux_hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concat the ``num_aux`` verifier hidden states ``[B, T, hidden]``
        along the feature dim → ``fc`` → ``hidden_norm`` → ``[B, T, hidden]``."""
        if len(aux_hidden_states) != self.cfg.num_aux_layers:
            raise ValueError(
                f"expected {self.cfg.num_aux_layers} aux hidden states "
                f"(one per aux layer {self.cfg.aux_layer_ids}), got "
                f"{len(aux_hidden_states)}"
            )
        cat = torch.cat(list(aux_hidden_states), dim=-1)  # [B, T, num_aux*hidden]
        if cat.shape[-1] != self.cfg.fc_in_features:
            raise ValueError(
                f"aux concat feature dim {cat.shape[-1]} != fc_in_features "
                f"{self.cfg.fc_in_features}"
            )
        # The verifier hands aux hidden in its own dtype (often upcast to
        # fp32 for the capture); cast to the drafter's compute dtype.
        cat = cat.to(self.fc.weight.dtype)
        return self.hidden_norm(self.fc(cat))

    # -- transformer forward ----------------------------------------------
    def backbone(
        self, hidden: torch.Tensor, position_ids: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = _rope_cos_sin(
            position_ids, self.cfg.head_dim, self.cfg.rope_theta,
            hidden.device, hidden.dtype,
        )
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attn_bias)
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

    # -- block-diffusion drafting -----------------------------------------
    @torch.no_grad()
    def draft_block(
        self,
        aux_hidden_last: Sequence[torch.Tensor],
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        block_size: int,
        num_steps: int,
    ) -> List[int]:
        """Draft ``block_size`` tokens by block-diffusion denoising.

        Parameters
        ----------
        aux_hidden_last
            ``num_aux`` tensors ``[1, 1, hidden]`` — the verifier's aux-layer
            hidden states at the **last committed position** (the conditioning
            context for the next block).
        embed_fn
            Verifier token-embedding lookup ``[*, T] -> [*, T, hidden]``
            (must already include Gemma's ``×sqrt(hidden)`` scaling).
        lm_head_fn
            Verifier logits head ``[*, hidden] -> [*, vocab]`` (must apply
            ``final_logit_softcapping`` if configured).
        block_size, num_steps
            Draft length and number of denoising iterations.

        Returns the ``block_size`` drafted token ids.

        NOTE (Stage-2 fidelity): the EAGLE-3↔block fusion (conditioning the
        block on the projected aux hidden via a leading register token) and
        the low-confidence remasking schedule are a principled, documented
        implementation; matching the reference DFlash acceptance profile is
        validated/tuned on the H200.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        num_steps = min(num_steps, block_size)
        cfg = self.cfg
        aux_proj = self.project_aux(aux_hidden_last)  # [1, 1, hidden]
        device = aux_proj.device
        dtype = aux_proj.dtype

        # Sequence layout: position 0 is the aux-conditioned context
        # register; positions 1..block_size are the draft block.
        T = 1 + block_size
        position_ids = torch.arange(T, device=device)
        # Non-causal block mask: the register attends only to itself; block
        # positions attend to the register + all block positions (PR #41703:
        # DFlash draft attention is non-causal within the block).
        neg = torch.finfo(dtype).min
        bias = torch.full((T, T), neg, device=device, dtype=dtype)
        bias[0, 0] = 0.0
        bias[1:, 0] = 0.0
        bias[1:, 1:] = 0.0
        attn_bias = bias[None, None]  # [1,1,T,T]

        block = torch.full((1, block_size), cfg.mask_token_id, dtype=torch.long,
                           device=device)
        # Low-confidence remasking schedule (front-loaded), like the MDLM
        # proposer, but non-causal and aux-conditioned.
        base, rem = divmod(block_size, num_steps)
        per_step = [base + (1 if i < rem else 0) for i in range(num_steps)]

        for step in range(num_steps):
            masked = block[0] == cfg.mask_token_id
            if not bool(masked.any()):
                break
            block_embeds = embed_fn(block).to(aux_proj.dtype)  # [1, block_size, hidden]
            h = torch.cat([aux_proj, block_embeds], dim=1)
            h = self.backbone(h, position_ids, attn_bias)  # [1, T, hidden]
            block_h = h[:, 1:, :]  # drop the register
            logits = lm_head_fn(block_h)  # [1, block_size, vocab]
            # The drafter must never propose the mask sentinel as a real
            # token; forbid it before argmax.
            logits[..., cfg.mask_token_id] = float("-inf")
            x0 = torch.argmax(logits, dim=-1)  # [1, block_size]
            probs = torch.softmax(logits.float(), dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)[0]  # [block_size]
            conf = torch.where(masked, conf, torch.full_like(conf, float("-inf")))
            k = per_step[step]
            if k <= 0:
                continue
            _, top = torch.topk(conf, k=min(k, int(masked.sum().item())))
            commit = torch.zeros_like(masked)
            commit[top] = True
            commit &= masked
            block[0] = torch.where(commit, x0[0], block[0])

        if bool((block[0] == cfg.mask_token_id).any()):
            # Collapse any leftover masks to argmax rather than emit <mask>.
            block_embeds = embed_fn(block).to(aux_proj.dtype)
            h = torch.cat([aux_proj, block_embeds], dim=1)
            h = self.backbone(h, position_ids, attn_bias)
            logits = lm_head_fn(h[:, 1:, :])
            logits[..., cfg.mask_token_id] = float("-inf")
            x0 = torch.argmax(logits, dim=-1)[0]
            leftover = block[0] == cfg.mask_token_id
            block[0] = torch.where(leftover, x0, block[0])
        return block[0].tolist()


# ===========================================================================
# Proposer adapter (engine spec-decode `propose_block` contract)
# ===========================================================================


class AuxHiddenProvider:
    """Contract for the object that supplies verifier aux-layer hidden
    states for the last committed position.

    Stage 2 wires this to the gemma-4 verifier (capturing decoder-layer
    outputs at :attr:`DFlashConfig.aux_layer_ids`). Stage 1 tests inject a
    synthetic provider.
    """

    def aux_hidden_last(self, committed_token_ids: List[int]) -> List[torch.Tensor]:
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
        aux_last = self.aux_provider.aux_hidden_last(committed_token_ids)
        peak = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tokens = self.drafter.draft_block(
            aux_last, self.embed_fn, self.lm_head_fn,
            block_size=block_size, num_steps=num_steps,
        )
        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
        if len(tokens) != block_size:  # pragma: no cover - draft_block guarantees
            raise RuntimeError(
                f"DFlash drafted {len(tokens)} tokens; expected {block_size}."
            )
        steps = min(num_steps, block_size)
        return BlockProposal(
            tokens=tokens,
            diffusion_steps=steps,
            forward_passes=steps,
            peak_activation_bytes=peak,
        )
