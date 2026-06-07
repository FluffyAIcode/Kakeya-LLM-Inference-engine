"""ADR 0011 toy prototype — cross-attention proposer/verifier coupling.

Phase 1 (G-X1) feasibility study: validate that a bounded-KV verifier
with a cross-attention bridge to a full-attention proposer's hidden
bank can recover long-context recall lost to KV trimming.

Architecture (single-modality text in Phase 1; multimodal extension
hooks documented inline for Phase 2 video):

    proposer (full attention over T tokens)
        └── hidden_p[0..T-1]          : memory bank, shape [T, hidden_p]
                                ▼
    verifier (bounded KV at layer ≤ K)
        └── self-attention            : sink+window-style
                                +
            cross-attention(Q←verifier, K,V←hidden_p)   ← THE NEW LAYER
                                ▼
            output logits

The single-layer cross-attention bridge is initialized with zero
output projection so that at training step 0 the cross-attention
contributes nothing and the verifier behaves identically to its
pre-ADR-0011 self. Gradients gradually mix the cross-attention output
into the verifier's residual stream.

Usage::

    # Smallest viable toy: single-batch text NIAH on Apple Silicon.
    # R1c defaults are heavier (2000 steps, 16 heads × 128 dim); pass
    # --train-steps 200 for a quick smoke run.
    PYTHONPATH=. python3 scripts/research/cross_attn_toy_prototype.py \\
        --model google/gemma-3-1b-it \\
        --device mps \\
        --train-steps 200 \\
        --eval-every 50

    # R1c full capacity-bumped run (Gate G-X1, ~2000 steps):
    PYTHONPATH=. python3 scripts/research/cross_attn_toy_prototype.py \\
        --model google/gemma-3-1b-it \\
        --device auto \\
        --train-steps 2000 \\
        --o-proj-init-std 0.01 \\
        --needle-debug-mode off

    # R1c easy-target debug probe (does the bridge work at all?):
    PYTHONPATH=. python3 scripts/research/cross_attn_toy_prototype.py \\
        --model google/gemma-3-1b-it \\
        --device auto \\
        --train-steps 2000 \\
        --needle-debug-mode small

    # Larger toy (Mac M4 24 GB, careful with memory):
    PYTHONPATH=. python3 scripts/research/cross_attn_toy_prototype.py \\
        --model google/gemma-3-2b-it \\
        --device mps \\
        --train-steps 500

    # Multimodal-ready (Phase 2 — substitute Gemma 4 multimodal):
    PYTHONPATH=. python3 scripts/research/cross_attn_toy_prototype.py \\
        --model google/gemma-4-2b-mm \\
        --multimodal-tokens video \\
        --train-steps 500

Phase 1 acceptance (Gate G-X1 per ADR 0011 §4):
  bounded baseline NIAH recall ≈ 20 %
  bounded + cross-attention NIAH recall ≥ 80 %
  full-attention reference NIAH recall ≈ 100 %

Per the project's CLI-plumbing convention this script is exempt from
the unit-test coverage gate. The cross-attention layer's invariants
are validated by the toy training itself + Gate G-X2/3 production
benchmarks.

This file is research-grade; it is intentionally NOT part of the
v0.3 inference engine. ADR 0011's Phase 4 will productionize the
verified parts under ``inference_engine.backends.*``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Cross-attention layer (the load-bearing new module)
# ============================================================================


class CrossAttentionBridge(nn.Module):
    """Single multi-head cross-attention layer inserted into the verifier.

    Q from verifier hidden state at the chosen depth.
    K, V from proposer hidden bank (full-attention representation of the
    same prefix).

    Initialized so that ``W_o`` is small (``o_proj_init_std``) — at step 0
    the layer contributes (near-)zero to the verifier's residual stream.
    Stability: the verifier behaves (almost) identically to its baseline
    for the first few gradient steps, then progressively incorporates
    cross-attention as the output projection learns larger weights.

    R1c note — ``o_proj_init_std``: PR-R1b used a strict ``W_o = 0``
    init. The R1b run showed loss decreasing but plateauing slowly
    (step 50→2.975 … 200→2.046), with per-token answer probability only
    ~13 % — far from the < ~0.7 loss where ``cross_attn_recall`` first
    turns non-zero. A strict-zero ``W_o`` makes the bridge's *initial*
    gradient signal flow only through ``ctx`` (the ``W_o`` row gradient
    is ``ctx``-shaped but the residual sees zero contribution on the
    forward pass for many steps). Seeding ``W_o`` with a small non-zero
    std lets the bridge contribute — and therefore be shaped by the loss
    — from step 1, trading a little step-0 stability for faster escape
    from the plateau. ``o_proj_init_std=0.0`` recovers the exact R1b
    zero-init behaviour (and keeps the step-0 ``output == 0`` invariant).
    """

    def __init__(
        self,
        verifier_hidden_dim: int,
        proposer_hidden_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        attn_dropout: float = 0.0,
        o_proj_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.o_proj_init_std = o_proj_init_std

        self.q_proj = nn.Linear(
            verifier_hidden_dim, num_heads * head_dim, bias=False,
        )
        self.k_proj = nn.Linear(
            proposer_hidden_dim, num_heads * head_dim, bias=False,
        )
        self.v_proj = nn.Linear(
            proposer_hidden_dim, num_heads * head_dim, bias=False,
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim, verifier_hidden_dim, bias=False,
        )
        self.attn_dropout = attn_dropout

        # NEAR-IDENTITY INITIALIZATION — the most important training-
        # stability trick in this prototype. W_q/W_k/W_v get the usual
        # small-normal init so gradient flow is non-zero. W_o is seeded
        # with std=``o_proj_init_std``: at exactly 0.0 the bridge output
        # is zero at step 0 (strict R1b behaviour); at a small positive
        # value (R1c default 0.01) the bridge contributes a small delta
        # immediately so the loss can shape W_o from step 1.
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        if o_proj_init_std > 0.0:
            nn.init.normal_(self.o_proj.weight, std=o_proj_init_std)
        else:
            nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        verifier_hidden: torch.Tensor,        # [B, T_v, hidden_v]
        proposer_hidden_bank: torch.Tensor,   # [B, T_p, hidden_p]
        proposer_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention; returns delta to add to verifier residual.

        ``proposer_attention_mask``: optional [B, T_p] mask where 0
        indicates padding to ignore. Modality-agnostic — for text it
        masks pad tokens; for video it would mask out absent frames
        in a fixed-size buffer.
        """
        B, T_v, _ = verifier_hidden.shape
        _, T_p, _ = proposer_hidden_bank.shape

        Q = self.q_proj(verifier_hidden).view(
            B, T_v, self.num_heads, self.head_dim,
        ).transpose(1, 2)  # [B, H, T_v, D]
        K = self.k_proj(proposer_hidden_bank).view(
            B, T_p, self.num_heads, self.head_dim,
        ).transpose(1, 2)  # [B, H, T_p, D]
        V = self.v_proj(proposer_hidden_bank).view(
            B, T_p, self.num_heads, self.head_dim,
        ).transpose(1, 2)  # [B, H, T_p, D]

        # [B, H, T_v, T_p]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if proposer_attention_mask is not None:
            # mask: [B, T_p]; broadcast to [B, 1, 1, T_p]
            mask = proposer_attention_mask[:, None, None, :].to(
                attn_scores.dtype,
            )
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        if self.training and self.attn_dropout > 0:
            attn = F.dropout(attn, p=self.attn_dropout)

        out = torch.matmul(attn, V)  # [B, H, T_v, D]
        out = out.transpose(1, 2).contiguous().view(
            B, T_v, self.num_heads * self.head_dim,
        )
        out = self.o_proj(out)        # [B, T_v, hidden_v]
        return out


# ============================================================================
# Bounded-KV mask — simulates the v0.3 sink+window verifier in this toy
# ============================================================================


def make_sink_window_attention_mask(
    seq_len: int,
    sink: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal attention mask that emulates v0.3 sink+window KV trimming.

    For each query position q:
      - allowed key positions = {0, 1, ..., sink-1}              (sink)
                              ∪ {q-window+1, ..., q}             (window, capped at q)
      - all other positions are masked out (finfo.min for floats)

    Returns a [seq_len, seq_len] additive bias where masked positions
    are dtype's finite minimum (== effective -inf inside attention
    softmax) and allowed positions are 0.

    bf16/fp16: ``finfo(dtype).min`` is used instead of ``float("-inf")``
    because some attention kernels (notably bf16 SDPA on MPS) propagate
    NaN through ``-inf + 0`` when the entire query row is masked. The
    finite minimum has identical numerical effect through softmax.
    """
    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else float("-inf")
    mask = torch.full(
        (seq_len, seq_len), neg_inf, device=device, dtype=dtype,
    )
    for q in range(seq_len):
        sink_end = min(sink, q + 1)
        mask[q, :sink_end] = 0.0
        window_start = max(sink, q - window + 1)
        mask[q, window_start : q + 1] = 0.0
    return mask


def make_sink_window_attention_mask_4d(
    seq_len: int,
    sink: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """4D additive sink+window mask of shape ``[1, 1, seq_len, seq_len]``.

    Same semantics as :func:`make_sink_window_attention_mask` but
    pre-shaped so it can be passed directly into HuggingFace decoder
    layers' eager attention path (``attn_weights + attention_mask``).

    For Gemma3-class models that distinguish full vs sliding attention
    layers, wrap this tensor in a dict
    ``{"full_attention": mask, "sliding_attention": mask}`` and pass
    that as the model's ``attention_mask`` keyword.
    """
    mask_2d = make_sink_window_attention_mask(
        seq_len, sink, window, device=device, dtype=dtype,
    )
    return mask_2d.unsqueeze(0).unsqueeze(0)


# ============================================================================
# Verifier wrapper with cross-attention injected at chosen depth
# ============================================================================


class CrossAttentionVerifier(nn.Module):
    """Wraps a HuggingFace causal LM with a cross-attention bridge.

    The bridge is inserted as a **residual addition on the OUTPUT of
    decoder layer ``cross_attn_depth - 1``** (1-indexed), so the
    augmented hidden state then propagates through the remaining
    layers (K, K+1, ..., N-1), the final layernorm, and the language
    modeling head — i.e., proper architectural integration, not a
    bypassed lm_head shortcut. (R1b Bug A.2 fix.)

    The verifier's self-attention is constrained to a sink+window
    pattern by passing a 4D additive ``attention_mask`` through to
    the base model's eager attention path. For Gemma3-class models
    the mask is wrapped in the
    ``{"full_attention": ..., "sliding_attention": ...}`` dict
    convention so both layer types are bounded identically. For
    other model families a raw 4D tensor is used. (R1b Bug B fix.)

    The wrapper leaves the underlying model's weights frozen by
    default; only the cross-attention bridge is trainable.

    Modality-agnostic: ``input_ids`` for Phase 1 text; Phase 2
    multimodal substitutes Gemma 4-class checkpoints; the bridge and
    masking machinery are unchanged.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cross_attn: CrossAttentionBridge,
        cross_attn_depth: int,
        sink: int = 4,
        window: int = 64,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.cross_attn = cross_attn
        self.cross_attn_depth = cross_attn_depth
        self.sink = sink
        self.window = window
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        layers = self._layers_module()
        if not 1 <= cross_attn_depth <= len(layers):
            raise ValueError(
                f"cross_attn_depth={cross_attn_depth} is out of range "
                f"[1, {len(layers)}] for this base model"
            )

    @property
    def config(self):
        return self.base.config

    def _layers_module(self) -> nn.ModuleList:
        """Locate the decoder ``nn.ModuleList`` on the base model.

        Supports HF Gemma3 / Llama / Qwen / Mistral families
        (``base.model.layers``) and GPT-2 family (``base.transformer.h``).
        Raises if neither shape is detected.
        """
        if hasattr(self.base, "model") and hasattr(self.base.model, "layers"):
            return self.base.model.layers
        if hasattr(self.base, "transformer") and hasattr(self.base.transformer, "h"):
            return self.base.transformer.h
        raise RuntimeError(
            "CrossAttentionVerifier could not find decoder layers on the "
            "base model (looked for base.model.layers and "
            "base.transformer.h). Add a binding for the new model family."
        )

    def _build_sink_window_mask_kwarg(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Return the attention_mask kwarg for the base forward.

        For Gemma3-class models that have separate full and sliding
        layer types, returns a dict so BOTH categories use the same
        sink+window restriction. For other families returns the bare
        4D tensor.
        """
        mask_4d = make_sink_window_attention_mask_4d(
            seq_len, self.sink, self.window, device=device, dtype=dtype,
        )
        cfg_arch = (getattr(self.config, "model_type", "") or "").lower()
        if cfg_arch.startswith("gemma3"):
            return {"full_attention": mask_4d, "sliding_attention": mask_4d}
        return mask_4d

    def _forward_with_bridge(
        self,
        input_ids: torch.Tensor,
        proposer_hidden_bank: torch.Tensor,
        proposer_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run base model with a forward hook injecting cross-attention.

        We register a hook on decoder layer ``K-1`` (0-indexed) that:

          1. Receives the layer's natural output ``h_K = layer_{K-1}(h_{K-1})``.
          2. Computes ``delta = cross_attn(h_K, proposer_hidden_bank)``.
          3. Returns ``h_K + delta`` so layers K..N-1, the final norm,
             and the lm_head all receive the augmented hidden.

        This is architecturally faithful to ADR 0011's design (cross-
        attention residual injected mid-stack, propagating through
        the rest of the verifier) without manually reimplementing HF's
        layer dispatch (which is fragile across model families and
        version pins).
        """
        layers = self._layers_module()
        target_layer = layers[self.cross_attn_depth - 1]
        cross_attn = self.cross_attn

        def hook(module, layer_inputs, layer_output):
            if isinstance(layer_output, tuple):
                hidden_out = layer_output[0]
                rest = layer_output[1:]
            else:
                hidden_out = layer_output
                rest = None
            delta = cross_attn(
                verifier_hidden=hidden_out,
                proposer_hidden_bank=proposer_hidden_bank,
                proposer_attention_mask=proposer_attention_mask,
            )
            modified = hidden_out + delta
            if rest is not None:
                return (modified,) + rest
            return modified

        seq_len = input_ids.size(1)
        attention_mask_kwarg = self._build_sink_window_mask_kwarg(
            seq_len, device=input_ids.device, dtype=self._base_dtype(),
        )

        handle = target_layer.register_forward_hook(hook)
        try:
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask_kwarg,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        return out.logits

    def _base_dtype(self) -> torch.dtype:
        """Best-effort dtype detection (HF doesn't always expose .dtype)."""
        if hasattr(self.base, "dtype") and isinstance(self.base.dtype, torch.dtype):
            return self.base.dtype
        for p in self.base.parameters():
            return p.dtype
        return torch.float32

    def forward(
        self,
        input_ids: torch.Tensor,
        proposer_hidden_bank: torch.Tensor,
        proposer_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bounded verifier + cross-attention bridge → logits."""
        return self._forward_with_bridge(
            input_ids=input_ids,
            proposer_hidden_bank=proposer_hidden_bank,
            proposer_attention_mask=proposer_attention_mask,
        )

    def forward_bounded_no_bridge(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Bounded baseline: same sink+window mask, no cross-attention.

        Used by the eval loop as the apples-to-apples baseline for the
        bridge — same KV restriction, no rescue path. Should perform
        like the v0.3 ``SinkWindowVerifier``.
        """
        seq_len = input_ids.size(1)
        attention_mask_kwarg = self._build_sink_window_mask_kwarg(
            seq_len, device=input_ids.device, dtype=self._base_dtype(),
        )
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask_kwarg,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    def forward_full_attention(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full-attention oracle: no sink+window, no bridge.

        This is the eval pipeline's sanity anchor — if oracle recall is
        not ~100% on these synthetic NIAH samples, the prompt format /
        chat template / answer matching is broken and the bridge
        comparison is uninformative. (R1b Bug D fix.)
        """
        out = self.base(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


# ============================================================================
# Toy data: needle-in-haystack
# ============================================================================


@dataclasses.dataclass
class NIAHSample:
    """One needle-in-haystack training example."""

    prompt_text: str
    answer_text: str
    needle_position: int  # token index where the needle lives


@dataclasses.dataclass(frozen=True)
class NeedleVocab:
    """Closed vocabulary the needle's secret code is drawn from.

    The answer string is ``"{prefix}-{code}"`` with ``prefix`` drawn
    uniformly from :attr:`prefixes` and ``code`` an integer in
    ``[code_min, code_max]``. Shrinking this vocabulary lowers the
    per-token entropy of the answer, which is the entire point of the
    R1c ``--needle-debug-mode`` knob: it lets us check whether the
    cross-attention bridge mechanism works *at all* on an easy target
    before asking it to memorise the full 15-prefix × 4-digit space
    (~135 k distinct answers) that R1b struggled to reach inside 200
    steps.
    """

    prefixes: Tuple[str, ...]
    code_min: int
    code_max: int

    def size(self) -> int:
        """Number of distinct answers this vocabulary can produce."""
        return len(self.prefixes) * (self.code_max - self.code_min + 1)


# Full-difficulty vocabulary — identical to the R1/R1b hard-coded set so
# that ``--needle-debug-mode off`` reproduces prior runs bit-for-bit
# under a fixed seed (same RNG draw order).
DEFAULT_NEEDLE_VOCAB = NeedleVocab(
    prefixes=(
        "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA",
        "ETA", "THETA", "IOTA", "KAPPA", "ORCHID", "PINE",
        "MAPLE", "OAK", "BIRCH",
    ),
    code_min=1000,
    code_max=9999,
)


def needle_vocab_for_mode(mode: str) -> Optional[NeedleVocab]:
    """Map a ``--needle-debug-mode`` value to a :class:`NeedleVocab`.

    * ``off``    → ``None`` (full default vocabulary, ~135 k answers).
    * ``small``  → 2 prefixes × single digit (20 answers) — the easiest
      probe: a single answer token range the bridge can plausibly nail
      inside a couple thousand steps, so a *non-zero* ``cross_attn``
      recall here isolates "is the bridge mechanism working?" from "is
      the target simply too hard?".
    * ``medium`` → 4 prefixes × two digits (400 answers) — an
      intermediate difficulty between ``small`` and ``off``.

    Raises ``ValueError`` on an unknown mode (ADR 0008 §6.2: no silent
    fallback).
    """
    if mode == "off":
        return None
    if mode == "small":
        return NeedleVocab(prefixes=("ALPHA", "BETA"), code_min=0, code_max=9)
    if mode == "medium":
        return NeedleVocab(
            prefixes=("ALPHA", "BETA", "GAMMA", "DELTA"),
            code_min=0,
            code_max=99,
        )
    raise ValueError(
        f"unknown needle_debug_mode={mode!r}; expected one of "
        "{'off', 'small', 'medium'}"
    )


def make_niah_dataset(
    *,
    tokenizer,
    n_samples: int = 200,
    haystack_min_tokens: int = 256,
    haystack_max_tokens: int = 1024,
    seed: int = 42,
    needle_vocab: Optional[NeedleVocab] = None,
) -> List[NIAHSample]:
    """Synthetic NIAH samples: hide a fact in random padding, ask for it.

    Each sample is structured as::

        <padding>... <NEEDLE>: the secret code is XXX-9999. <padding>...
        Question: what is the secret code? Answer:

    ``needle_vocab`` selects the closed set the secret code is drawn
    from; ``None`` uses :data:`DEFAULT_NEEDLE_VOCAB` (full difficulty,
    bit-for-bit identical to R1/R1b). A smaller vocabulary (see
    :func:`needle_vocab_for_mode`) lowers answer entropy for the R1c
    debug modes.
    """
    vocab = needle_vocab if needle_vocab is not None else DEFAULT_NEEDLE_VOCAB
    rng = random.Random(seed)
    samples: List[NIAHSample] = []
    for _ in range(n_samples):
        haystack_len = rng.randint(haystack_min_tokens, haystack_max_tokens)
        # Synthetic codes: e.g., ALPHA-1234, BETA-5678, etc. RNG draw
        # order (haystack_len, prefix, code) is preserved from R1b so a
        # fixed seed reproduces the full-vocab dataset exactly.
        prefix = rng.choice(vocab.prefixes)
        code = f"{prefix}-{rng.randint(vocab.code_min, vocab.code_max)}"
        needle = f"\nIMPORTANT: the secret code is {code}.\n"

        padding_lines = []
        for i in range(haystack_len // 16):
            padding_lines.append(
                f"Note {i:04d}: this paragraph is unrelated padding "
                "and does not contain the answer."
            )
        # Insert needle at a random position so it's neither in sink
        # nor in the late window.
        insert_at = rng.randint(2, max(2, len(padding_lines) - 4))
        padding_lines.insert(insert_at, needle)
        prompt = "\n".join(padding_lines) + (
            "\nQuestion: what is the secret code? Answer:"
        )
        samples.append(
            NIAHSample(
                prompt_text=prompt,
                answer_text=" " + code,
                needle_position=insert_at,
            )
        )
    return samples


# ============================================================================
# Training loop
# ============================================================================


def _encode_prompt_with_chat_template(
    tokenizer, prompt_text: str, device: torch.device,
) -> torch.Tensor:
    """Tokenize ``prompt_text`` as a single user turn ready for generation.

    Uses ``apply_chat_template(..., add_generation_prompt=True)`` so the
    model receives the SFT-correct framing it was trained on. Required
    for instruction-tuned checkpoints (e.g., ``gemma-3-1b-it``); raw
    text prompts cause IT models to emit control tokens or refuse.
    (R1b Bug C fix.)

    Raises ``RuntimeError`` if the tokenizer has no chat template — the
    project's ADR 0008 §6.2 forbids silent fallbacks.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "tokenizer has no chat_template; pass an instruction-tuned "
            "checkpoint (e.g., google/gemma-3-1b-it) or set a template "
            "explicitly. ADR 0008 §6.2 forbids silent fallbacks."
        )
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(enc, list):
        enc = torch.tensor([enc])
    return enc.to(device)


def _encode_answer_continuation(
    tokenizer, answer_text: str, device: torch.device,
) -> torch.Tensor:
    """Tokenize the answer as a raw continuation (no special tokens).

    This is concatenated with the chat-template-prefixed prompt to form
    the supervised target for cross-entropy loss on answer positions.
    """
    enc = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=128,
    )
    return enc.input_ids.to(device)


def train_step(
    *,
    proposer,
    verifier_with_bridge: CrossAttentionVerifier,
    sample: NIAHSample,
    tokenizer,
    optimizer,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """One gradient step.

    1. Tokenize sample's prompt via chat template; tokenize answer
       as a raw continuation.
    2. Run proposer in full-attention mode over the chat-template-
       wrapped prompt to produce the hidden bank.
    3. Run verifier (bounded local attention + cross-attention bridge)
       on prompt+answer.
    4. Loss: cross-entropy at answer positions only.
    """
    prompt_ids = _encode_prompt_with_chat_template(
        tokenizer, sample.prompt_text, device,
    )
    if prompt_ids.size(1) < 8:
        return 0.0
    answer_ids = _encode_answer_continuation(
        tokenizer, sample.answer_text, device,
    )
    if answer_ids.numel() == 0:
        return 0.0
    full_input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    if full_input_ids.size(1) > 4096:
        return 0.0

    with torch.no_grad():
        proposer_out = proposer(
            input_ids=prompt_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    proposer_hidden_bank = proposer_out.hidden_states[-1]  # [1, T_p, hidden_p]

    logits = verifier_with_bridge(
        input_ids=full_input_ids,
        proposer_hidden_bank=proposer_hidden_bank,
    )

    answer_start = prompt_ids.size(1)
    target = full_input_ids[:, answer_start:].contiguous()
    pred = logits[:, answer_start - 1 : -1, :].contiguous()
    if target.numel() == 0 or pred.size(1) != target.size(1):
        return 0.0
    # Upcast logits to fp32 for numerically stable cross-entropy under
    # bf16 base model weights — bf16 cross_entropy on MPS occasionally
    # returns NaN on extreme logit magnitudes.
    loss = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)).float(),
        target.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        verifier_with_bridge.cross_attn.parameters(), max_norm=1.0,
    )
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_recall(
    *,
    proposer,
    verifier_with_bridge: CrossAttentionVerifier,
    samples: List[NIAHSample],
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 24,
    include_oracle: bool = True,
) -> Tuple[float, float, float]:
    """Measure NIAH recall under three regimes.

    1. **cross_attn**: bounded verifier (sink+window mask) + bridge.
    2. **bounded_baseline**: same bounded verifier, no bridge.
    3. **full_attention oracle**: verifier with no sink+window mask,
       no bridge — sanity anchor. If oracle ≪ 100 %, the prompt
       template / answer matching is broken and the bridge comparison
       cannot be trusted (Bug D fix).

    Returns ``(cross_attn_recall, bounded_baseline_recall, oracle_recall)``.
    If ``include_oracle=False``, the third value is ``-1.0`` to signal
    "not measured" without inventing data.
    """
    cross_attn_correct = 0
    bounded_correct = 0
    oracle_correct = 0
    for sample in samples:
        prompt_ids = _encode_prompt_with_chat_template(
            tokenizer, sample.prompt_text, device,
        )

        proposer_out = proposer(
            input_ids=prompt_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_bank = proposer_out.hidden_states[-1]

        cross_attn_text = _greedy_decode(
            forward_fn=lambda ids: verifier_with_bridge(
                input_ids=ids, proposer_hidden_bank=hidden_bank,
            ),
            input_ids=prompt_ids, tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if sample.answer_text.strip() in cross_attn_text:
            cross_attn_correct += 1

        bounded_text = _greedy_decode(
            forward_fn=lambda ids: verifier_with_bridge.forward_bounded_no_bridge(ids),
            input_ids=prompt_ids, tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if sample.answer_text.strip() in bounded_text:
            bounded_correct += 1

        if include_oracle:
            oracle_text = _greedy_decode(
                forward_fn=lambda ids: verifier_with_bridge.forward_full_attention(ids),
                input_ids=prompt_ids, tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            if sample.answer_text.strip() in oracle_text:
                oracle_correct += 1

    n = max(len(samples), 1)
    return (
        cross_attn_correct / n,
        bounded_correct / n,
        (oracle_correct / n) if include_oracle else -1.0,
    )


@torch.no_grad()
def _greedy_decode(
    *,
    forward_fn,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
) -> str:
    """Generic greedy decoder: takes a callable ``forward_fn(input_ids)``
    that returns logits ``[B, T, V]`` and unrolls argmax up to
    ``max_new_tokens`` or EOS.
    """
    cur = input_ids
    for _ in range(max_new_tokens):
        logits = forward_fn(cur)
        next_token = int(torch.argmax(logits[:, -1, :]).item())
        cur = torch.cat(
            [cur, torch.tensor([[next_token]], device=cur.device)], dim=1,
        )
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(
        cur[0, input_ids.size(1):], skip_special_tokens=True,
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default="google/gemma-3-1b-it",
        help="HF model id; same model used for both proposer and verifier "
             "in this toy. Phase 2: substitute Gemma 4 multimodal here.",
    )
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
        help="auto picks mps on Mac, cuda on Linux+NVIDIA, else cpu",
    )
    ap.add_argument("--cross-attn-depth", type=int, default=20,
                    help="verifier decoder layer index (1-indexed) on whose "
                         "output the cross-attention residual is injected; "
                         "the augmented hidden then flows through layers "
                         "K+1..N, the final layernorm, and lm_head. Default "
                         "20 is appropriate for Gemma 3-1B (26 layers); for "
                         "other base models pick a depth ≈ 0.7 × num_layers.")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=16,
                    help="R1c: bumped 8→16. More heads give the bridge "
                         "finer-grained routing over the proposer bank.")
    ap.add_argument("--head-dim", type=int, default=128,
                    help="R1c: bumped 64→128. Wider per-head dim gives "
                         "the bridge more capacity to memorise needle "
                         "codes (the R1b bottleneck was capacity, not "
                         "gradient flow).")
    ap.add_argument("--train-steps", type=int, default=2000,
                    help="R1c: bumped 200→2000. The R1b loss curve was "
                         "still descending at step 200 (2.046, slope ~ "
                         "-0.001/step); extrapolation puts the < ~0.7 "
                         "loss where cross_attn_recall turns non-zero at "
                         "~1500-2500 steps.")
    ap.add_argument("--o-proj-init-std", type=float, default=0.01,
                    help="std for the cross-attention output projection "
                         "init. R1c default 0.01 (R1b used strict 0.0). "
                         "0.0 keeps the step-0 zero-contribution "
                         "invariant; a small positive value lets the "
                         "bridge contribute — and be shaped by the loss "
                         "— from step 1.")
    ap.add_argument(
        "--needle-debug-mode",
        choices=["off", "small", "medium"],
        default="off",
        help="shrink the needle answer vocabulary to lower per-token "
             "entropy. 'off' = full 15-prefix × 4-digit set (~135 k "
             "answers, the real task); 'small' = 2 prefixes × 1 digit "
             "(20 answers); 'medium' = 4 prefixes × 2 digits (400). Use "
             "'small' to check the bridge mechanism works on an easy "
             "target before debugging the hard one.",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--haystack-min-tokens", type=int, default=256)
    ap.add_argument("--haystack-max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--multimodal-tokens",
        choices=["none", "image", "video"],
        default="none",
        help="Phase 2 hook (not active in Phase 1): when set, the dataset "
             "loader switches to the multimodal NIAH variant. Phase 1 "
             "validates text-only; Phase 2 substitutes Gemma 4 MM model + "
             "this flag.",
    )
    ap.add_argument(
        "--output", default="results/research/cross_attn_toy_run.json",
        help="JSON report path",
    )
    args = ap.parse_args()

    if args.multimodal_tokens != "none":
        # ADR 0008 §6.2: no silent fallback. If the user explicitly
        # asked for a Phase 2 mode, refuse rather than degrade silently.
        print(
            f"[toy] --multimodal-tokens={args.multimodal_tokens} is "
            "reserved for Phase 2 (Gemma 4 multimodal). Phase 1 toy "
            "validates text-only; aborting. Pass --multimodal-tokens "
            "none to continue.",
            file=sys.stderr,
        )
        return 2

    print(f"[toy] loading {args.model}", file=sys.stderr, flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[toy] device={device}", file=sys.stderr)

    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Same checkpoint for proposer + verifier in Phase 1; verifier
    # gets the cross-attention bridge added on top, proposer is frozen
    # full-attention reference. Phase 2 substitutes a Gemma 4 MM
    # checkpoint (same shape, different weights).
    #
    # attn_implementation="eager" is required so we can pass a 4D
    # additive sink+window mask through to the attention forward.
    # SDPA / FlashAttention paths normalize the mask in ways that
    # silently drop our restriction.
    proposer = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(device)
    proposer.eval()
    for p in proposer.parameters():
        p.requires_grad = False

    verifier_base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(device)
    verifier_base.eval()

    config = verifier_base.config
    verifier_hidden_dim = config.hidden_size
    proposer_hidden_dim = proposer.config.hidden_size

    cross_attn = CrossAttentionBridge(
        verifier_hidden_dim=verifier_hidden_dim,
        proposer_hidden_dim=proposer_hidden_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        o_proj_init_std=args.o_proj_init_std,
    ).to(device).to(dtype)

    verifier = CrossAttentionVerifier(
        base_model=verifier_base,
        cross_attn=cross_attn,
        cross_attn_depth=args.cross_attn_depth,
        sink=args.sink,
        window=args.window,
        freeze_base=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in cross_attn.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Data
    needle_vocab = needle_vocab_for_mode(args.needle_debug_mode)
    vocab_desc = (
        f"{args.needle_debug_mode} "
        f"({(needle_vocab or DEFAULT_NEEDLE_VOCAB).size()} distinct answers)"
    )
    print(
        f"[toy] generating {args.n_train} train + {args.n_eval} eval "
        f"NIAH samples; needle vocab={vocab_desc}", file=sys.stderr,
    )
    train_data = make_niah_dataset(
        tokenizer=tokenizer, n_samples=args.n_train,
        haystack_min_tokens=args.haystack_min_tokens,
        haystack_max_tokens=args.haystack_max_tokens,
        seed=args.seed,
        needle_vocab=needle_vocab,
    )
    eval_data = make_niah_dataset(
        tokenizer=tokenizer, n_samples=args.n_eval,
        haystack_min_tokens=args.haystack_min_tokens,
        haystack_max_tokens=args.haystack_max_tokens,
        seed=args.seed + 1,
        needle_vocab=needle_vocab,
    )

    # Pre-train evaluation: with o_proj_init_std=0 the bridge
    # contributes nothing → cross_attn ≡ bounded_baseline at step 0;
    # with the R1c default (0.01) the bridge already perturbs the
    # residual slightly, so cross_attn may differ marginally from the
    # bounded baseline even before training. Oracle measures the
    # prompt/answer pipeline itself either way.
    print("[toy] pre-train eval (full eval set, all 3 baselines)",
          file=sys.stderr)
    pre_xa, pre_bounded, pre_oracle = evaluate_recall(
        proposer=proposer,
        verifier_with_bridge=verifier,
        samples=eval_data,
        tokenizer=tokenizer,
        device=device,
    )
    print(
        f"[toy] pre-train: oracle={pre_oracle:.3f}  "
        f"bounded_baseline={pre_bounded:.3f}  "
        f"cross_attn={pre_xa:.3f}",
        file=sys.stderr,
    )

    history = []
    rng = random.Random(args.seed)
    print(f"[toy] training {args.train_steps} steps", file=sys.stderr)
    t0 = time.perf_counter()
    losses = []
    avg = 0.0
    for step in range(1, args.train_steps + 1):
        sample = rng.choice(train_data)
        loss = train_step(
            proposer=proposer,
            verifier_with_bridge=verifier,
            sample=sample,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
        )
        losses.append(loss)
        if step % 10 == 0:
            avg = sum(losses[-10:]) / max(len(losses[-10:]), 1)
            print(
                f"[toy] step={step}  loss(avg10)={avg:.4f}",
                file=sys.stderr, flush=True,
            )
        if step % args.eval_every == 0:
            # Periodic eval skips oracle (it's invariant — saves time).
            xa, bounded, _ = evaluate_recall(
                proposer=proposer,
                verifier_with_bridge=verifier,
                samples=eval_data[: max(20, len(eval_data) // 4)],
                tokenizer=tokenizer,
                device=device,
                include_oracle=False,
            )
            print(
                f"[toy] step={step}  cross_attn={xa:.3f}  "
                f"bounded_baseline={bounded:.3f}",
                file=sys.stderr,
            )
            history.append({
                "step": step,
                "cross_attn_recall": xa,
                "baseline_recall": bounded,
                "loss_avg10": avg,
            })

    elapsed = time.perf_counter() - t0
    print(f"[toy] training done in {elapsed:.1f}s", file=sys.stderr)

    print("[toy] final eval on full eval set (all 3 baselines)",
          file=sys.stderr)
    final_xa, final_bounded, final_oracle = evaluate_recall(
        proposer=proposer,
        verifier_with_bridge=verifier,
        samples=eval_data,
        tokenizer=tokenizer,
        device=device,
    )
    print(
        f"[toy] FINAL: oracle={final_oracle:.3f}  "
        f"bounded_baseline={final_bounded:.3f}  "
        f"cross_attn={final_xa:.3f}",
        file=sys.stderr,
    )

    # Gate G-X1 acceptance — three predicates:
    #   (a) cross_attn_recall  ≥ 0.80  (the hypothesis)
    #   (b) bounded_baseline   ≤ 0.30  (memory bound is real)
    #   (c) oracle             ≥ 0.80  (sanity: eval pipeline works)
    # All three must hold. (R1b adds (c) as the new sanity gate;
    # G-X1 attempt #1 silently failed (c) because the lm_head was
    # consuming a layer-K hidden state.)
    gate_oracle_ok = final_oracle >= 0.80
    gate_bounded_ok = final_bounded <= 0.30
    gate_cross_ok = final_xa >= 0.80
    gate_g_x1_pass = gate_oracle_ok and gate_bounded_ok and gate_cross_ok
    print(
        f"[toy] Gate G-X1 predicates: "
        f"oracle≥0.80 ({final_oracle:.3f}) -> "
        f"{'OK' if gate_oracle_ok else 'FAIL'}  |  "
        f"bounded≤0.30 ({final_bounded:.3f}) -> "
        f"{'OK' if gate_bounded_ok else 'FAIL'}  |  "
        f"cross≥0.80 ({final_xa:.3f}) -> "
        f"{'OK' if gate_cross_ok else 'FAIL'}",
        file=sys.stderr,
    )
    print(
        f"[toy] Gate G-X1 overall: "
        f"{'PASS' if gate_g_x1_pass else 'FAIL'}",
        file=sys.stderr,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        # v1 (R1 #1) → v2 (R1b: oracle baseline + 3-predicate gate)
        # → v3 (R1c: o_proj_init_std + needle_debug_mode config fields,
        #   capacity bump defaults). Consumers keyed on v2 must handle
        #   the two new config keys.
        "schema_version": 3,
        "kind": "adr_0011_toy_prototype_g_x1",
        "config": {
            "model": args.model,
            "device": str(device),
            "attn_implementation": "eager",
            "cross_attn_depth": args.cross_attn_depth,
            "sink": args.sink,
            "window": args.window,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "train_steps": args.train_steps,
            "lr": args.lr,
            "o_proj_init_std": args.o_proj_init_std,
            "needle_debug_mode": args.needle_debug_mode,
            "needle_vocab_size": (
                needle_vocab or DEFAULT_NEEDLE_VOCAB
            ).size(),
            "n_train": args.n_train,
            "n_eval": args.n_eval,
            "haystack_min_tokens": args.haystack_min_tokens,
            "haystack_max_tokens": args.haystack_max_tokens,
            "seed": args.seed,
            "uses_chat_template": True,
            "verifier_layer_surgery": "forward_hook_on_layer_K_output",
        },
        "pre_train": {
            "cross_attn_recall": pre_xa,
            "baseline_recall": pre_bounded,
            "oracle_recall": pre_oracle,
        },
        "training_history": history,
        "final": {
            "cross_attn_recall": final_xa,
            "baseline_recall": final_bounded,
            "oracle_recall": final_oracle,
            "elapsed_s": elapsed,
        },
        "gate_predicates": {
            "oracle_ge_080": gate_oracle_ok,
            "bounded_le_030": gate_bounded_ok,
            "cross_attn_ge_080": gate_cross_ok,
        },
        "gate_g_x1_pass": gate_g_x1_pass,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[toy] report -> {args.output}", file=sys.stderr)
    return 0 if gate_g_x1_pass else 1


if __name__ == "__main__":
    sys.exit(main())
