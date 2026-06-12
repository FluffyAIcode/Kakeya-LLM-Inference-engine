"""All-MLX DFlash drafter — Step-2 rescue (eliminate the per-block bridge).

The gate-clean iterC evidence (PR #109) showed the hybrid fused engine is
correct (recall 5/5 @ ctx280, accept_len 2.1–2.9/4) but 0.028× decode-only
vs native AR: each 4-token block paid 4+ MLX↔numpy↔torch bridge crossings
plus a float32 CPU-torch drafter forward (~2.7–8.4 s/block). This module
is the fix: the same DFlash drafter, native MLX, sharing the verifier's
Metal stream — zero bridge crossings per block.

Fidelity contract: a 1:1 port of ``inference_engine/v04/dflash_drafter.py``
(``DFlashDrafter``) — same config parsing (``DFlashConfig`` is reused
directly), same weight names (loaded straight from the checkpoint
``model.safetensors`` via ``mx.load``), same math:

* Qwen3 blocks: q/k/v/o_proj (no bias), q_norm/k_norm RMSNorm on head_dim,
  pre/post-attention RMSNorm, SiLU-gated MLP;
* explicit float32 RoPE tables (cos/sin) with the rotate-half convention,
  applied at arbitrary positions (context positions and query positions
  are different ranges);
* non-causal attention over [context ++ query] with GQA via
  ``mx.fast.scaled_dot_product_attention`` (handles n_q != n_kv natively,
  no repeat_interleave materialisation);
* ``fc`` aux fusion → ``hidden_norm`` once over context → per-layer
  context K/V (k_norm + RoPE), prefill-built and extended per block
  (components B of the fused engine);
* drafts = argmax over mask-position logits with the mask sentinel
  excluded.

Parity with the torch reference is validated ON DEVICE by
``scripts/research/k3_mlx_drafter_parity.py`` (bridge preset
``k3-drafter-parity``) before any throughput claim — same evidence
discipline as everything else in PR #109.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

from inference_engine.v04.dflash_drafter import DFlashConfig


def _mx():
    import mlx.core as mx  # type: ignore

    return mx


def _nn():
    import mlx.nn as nn  # type: ignore

    return nn


def _rope_cos_sin(positions: Any, head_dim: int, theta: float):
    """Float32 rotary tables for arbitrary ``positions`` ``[T]`` →
    ``(cos, sin)`` each ``[T, head_dim]``. Mirrors the torch reference
    (full-precision tables, rotate-half pairing)."""
    mx = _mx()
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    )
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)  # [T, head_dim]
    return mx.cos(emb), mx.sin(emb)


def _apply_rope(x: Any, cos: Any, sin: Any) -> Any:
    """x: [B, H, T, D] (any dtype); cos/sin: [T, D] float32."""
    mx = _mx()
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    out = x.astype(mx.float32) * cos[None, None] + rotated.astype(mx.float32) * sin[None, None]
    return out.astype(x.dtype)


class _Attention:
    """DFlash draft attention, MLX-native (see torch ``_DFlashAttention``)."""

    def __init__(self, cfg: DFlashConfig) -> None:
        nn = _nn()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.theta = cfg.rope_theta
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps)

    def project_context_kv(self, ctx_normed: Any, ctx_positions: Any):
        """(hidden_norm-ed) context hidden → this layer's (k, v), k_norm +
        RoPE applied. Returns each ``[B, nkv, C, hd]``."""
        mx = _mx()
        B, C, _ = ctx_normed.shape
        k = self.k_proj(ctx_normed).reshape(B, C, self.nkv, self.hd)
        v = self.v_proj(ctx_normed).reshape(B, C, self.nkv, self.hd)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        cos, sin = _rope_cos_sin(ctx_positions, self.hd, self.theta)
        k = _apply_rope(k, cos, sin)
        return k, v

    def __call__(self, h: Any, query_positions: Any, ctx_k: Any, ctx_v: Any) -> Any:
        mx = _mx()
        B, T, _ = h.shape
        q = self.q_proj(h).reshape(B, T, self.nh, self.hd)
        k = self.k_proj(h).reshape(B, T, self.nkv, self.hd)
        v = self.v_proj(h).reshape(B, T, self.nkv, self.hd)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        cos, sin = _rope_cos_sin(query_positions, self.hd, self.theta)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if ctx_k is not None:
            k = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2)
            v = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2)
        # Non-causal; GQA handled natively (no repeat_interleave).
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=None,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.nh * self.hd)
        return self.o_proj(out)


class _MLP:
    def __init__(self, cfg: DFlashConfig) -> None:
        nn = _nn()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: Any) -> Any:
        nn = _nn()
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer:
    def __init__(self, cfg: DFlashConfig) -> None:
        nn = _nn()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = _Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = _MLP(cfg)

    def __call__(self, h, query_positions, ctx_k, ctx_v):
        h = h + self.self_attn(
            self.input_layernorm(h), query_positions, ctx_k, ctx_v,
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class MLXDFlashDrafter:
    """MLX-native DFlash drafter with the SAME method surface as the torch
    ``DFlashDrafter`` fast path (``make_context_kv`` / ``extend_context_kv``
    / ``draft_block_cached``), so ``fused_specdecode_generate`` drives either
    implementation unchanged."""

    def __init__(self, cfg: DFlashConfig) -> None:
        nn = _nn()
        self.cfg = cfg
        self.layers = [_Layer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.fc = nn.Linear(cfg.fc_in_features, cfg.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    # -- weights ------------------------------------------------------------
    def load_weights(self, weights: dict) -> None:
        """Assign checkpoint tensors (HF DFlash names) onto the modules.

        Same strictness as the torch loader: any missing/unexpected key is
        a hard error. Dtype is preserved from the checkpoint (bf16).
        """
        own: dict = {}

        def reg(prefix: str, mod: Any) -> None:
            for name in ("weight",):
                own[f"{prefix}.{name}"] = (mod, name)

        for i, layer in enumerate(self.layers):
            p = f"layers.{i}"
            reg(f"{p}.input_layernorm", layer.input_layernorm)
            reg(f"{p}.post_attention_layernorm", layer.post_attention_layernorm)
            for sub in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                reg(f"{p}.self_attn.{sub}", getattr(layer.self_attn, sub))
            for sub in ("gate_proj", "up_proj", "down_proj"):
                reg(f"{p}.mlp.{sub}", getattr(layer.mlp, sub))
        reg("fc", self.fc)
        reg("hidden_norm", self.hidden_norm)
        reg("norm", self.norm)

        missing = [k for k in own if k not in weights]
        unexpected = [k for k in weights if k not in own]
        if missing or unexpected:
            raise ValueError(
                f"DFlash MLX weight mismatch: missing={missing[:6]} "
                f"unexpected={unexpected[:6]}"
            )
        for key, (mod, attr) in own.items():
            setattr(mod, attr, weights[key])

    @classmethod
    def from_pretrained(cls, model_id_or_path: str) -> "MLXDFlashDrafter":
        mx = _mx()
        cfg = DFlashConfig.from_pretrained(model_id_or_path)
        local = Path(model_id_or_path) / "model.safetensors"
        if local.is_file():
            path = str(local)
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id_or_path, "model.safetensors")
        model = cls(cfg)
        model.load_weights(mx.load(path))
        return model

    # -- aux fusion + context K/V (components B) -----------------------------
    def combine_aux(self, aux_hidden_states: Sequence[Any]) -> Any:
        mx = _mx()
        if len(aux_hidden_states) != self.cfg.num_aux_layers:
            raise ValueError(
                f"expected {self.cfg.num_aux_layers} aux hidden states, got "
                f"{len(aux_hidden_states)}"
            )
        cat = mx.concatenate(list(aux_hidden_states), axis=-1)
        if cat.shape[-1] != self.cfg.fc_in_features:
            raise ValueError(
                f"aux concat feature dim {cat.shape[-1]} != fc_in_features "
                f"{self.cfg.fc_in_features}"
            )
        return self.fc(cat.astype(self.fc.weight.dtype))

    def precompute_context_kv(self, context_states: Any, ctx_positions: Any):
        ctx_normed = self.hidden_norm(
            context_states.astype(self.hidden_norm.weight.dtype))
        return [
            layer.self_attn.project_context_kv(ctx_normed, ctx_positions)
            for layer in self.layers
        ]

    def make_context_kv(self, aux_hidden_context: Sequence[Any], positions: Any):
        ctx_states = self.combine_aux(aux_hidden_context)
        return self.precompute_context_kv(ctx_states, positions)

    @staticmethod
    def extend_context_kv(ctx_kv, new_kv):
        mx = _mx()
        out = []
        for (ck, cv), (nk, nv) in zip(ctx_kv, new_kv):
            out.append((
                mx.concatenate([ck, nk.astype(ck.dtype)], axis=2),
                mx.concatenate([cv, nv.astype(cv.dtype)], axis=2),
            ))
        return out

    # -- drafting -------------------------------------------------------------
    def _run_layers(self, hidden: Any, query_positions: Any, ctx_kv) -> Any:
        for layer, (ck, cv) in zip(self.layers, ctx_kv):
            hidden = layer(hidden, query_positions, ck, cv)
        return self.norm(hidden)

    def draft_block_cached(
        self,
        ctx_kv,
        bonus_token_id: int,
        embed_fn: Callable[[Any], Any],
        lm_head_fn: Callable[[Any], Any],
        *,
        block_size: int,
        context_len: int,
    ) -> List[int]:
        """Single non-causal pass over ``[bonus, mask×block_size]`` against the
        cached context K/V → ``block_size`` draft token ids. All-MLX: the
        embed/lm_head fns are the verifier's native MLX weights — no bridge."""
        mx = _mx()
        cfg = self.cfg
        query_ids = mx.array(
            [[int(bonus_token_id)] + [cfg.mask_token_id] * block_size])
        query_positions = mx.arange(context_len, context_len + 1 + block_size)
        h = embed_fn(query_ids).astype(self.fc.weight.dtype)
        h = self._run_layers(h, query_positions, ctx_kv)
        logits = lm_head_fn(h)  # [1, 1+block, vocab]
        vocab = logits.shape[-1]
        never_mask = mx.arange(vocab) == cfg.mask_token_id
        logits = mx.where(never_mask, mx.array(-float("inf")), logits)
        drafts = mx.argmax(logits[0, 1:1 + block_size], axis=-1)
        mx.eval(drafts)
        return [int(t) for t in drafts.tolist()]


def make_native_embed_lm_head(
    text_model: Any, *, softcap: float | None = None,
) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """All-MLX ``(embed_fn, lm_head_fn)`` over the verifier's weights.

    Same semantics as ``make_bridge_embed_lm_head`` (Gap-B: plain lookup,
    NO ``×sqrt(hidden)``; tied-embed logits + softcapping) minus the
    mx↔torch conversions that made the hybrid path 0.028× AR.
    """
    mx = _mx()

    def embed_fn(query_ids: Any) -> Any:
        return text_model.embed_tokens(query_ids)  # no embed_scale (Gap-B)

    def lm_head_fn(h: Any) -> Any:
        out = text_model.embed_tokens.as_linear(h)
        if softcap:
            out = softcap * mx.tanh(out / softcap)
        return out

    return embed_fn, lm_head_fn
