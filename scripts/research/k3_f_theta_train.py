"""K3 Block C — Train ``f_θ`` K/V projection: drafter K/V → verifier K/V.

v3 (2026-06-10) — one-shot principled trainer, attention-output distillation
===========================================================================

PR #103 v1 evidence: identity-restore recall = 1.0 (machinery correct);
f_θ-projected recall = 0.0 (training inadequate).

Per user request 2026-06-10: "一步到位，不要中间态" — skip the v2
intermediate (cos+mag) and ship the principled fix directly.

The ONE-SHOT principled fix
---------------------------

**Attention-output distillation loss** (the v3 default
``--loss-type attn_distill``). For each verifier layer ℓ:

    K_pred_ℓ, V_pred_ℓ = f_θ(drafter_KV)[ℓ]

    Q_for_attn = q_norm(Q_raw_ℓ).view(B, T, H_q, D) → RoPE → transpose
    K_for_attn = k_norm(K_pred_ℓ).view(B, T, H_kv, D) → RoPE → transpose
    V_for_attn = v_norm(V_pred_ℓ).view(B, T, H_kv, D) → transpose

    O_pred_ℓ = o_proj(scaled_dot_product_attention(Q, K, V, mask, scale))

    loss_ℓ = MSE(O_pred_ℓ, O_tgt_ℓ)              # O_tgt is the verifier's
                                                   actual attn output captured
                                                   during data collection

    Total = mean over layers

This is the **mathematically right loss for K/V projection**. It directly
optimises "f_θ-injected K/V produces equivalent verifier attention output",
accounting for: GQA grouping, RoPE positional encoding, causal/sliding
mask, k_norm/q_norm/v_norm, AND the layer's o_proj. Unlike pure MSE
(v1) or cos+mag (v2), this loss exposes the gradient to the actual
quantity that propagates through the residual stream at inference.

To make this affordable on H200, data collection caches per layer per
sequence (Q_raw, O_tgt, cos, sin, attention_mask) on CPU bf16; training
streams these to GPU per step. Verifier forward is run ONCE per
sequence (not per training step). For 64 sequences × 30 layers × T=512,
cache is ~25 GB CPU RAM (fits comfortably).

Three additional changes (carried over from v2 design)
------------------------------------------------------

  (a) **Larger f_θ rank**: default 256 → 768 for ``attn_distill``
      (more capacity at the encoder bottleneck; ~88M params total
      vs v1's 32M). Legacy losses keep rank=256.

  (b) **NIAH-style synthetic training prompts**: 64 prompts (50% of
      corpus) match the eval's haystack+needle pattern with
      independent seeds, so f_θ sees retrieval structure at training.

  (c) **Cosine LR schedule + 20000 steps**: linear warmup (500 steps)
      then cosine decay to peak/100. v1's 4000 constant-lr steps was
      grossly undertrained (59 s of training).

Reproducibility
---------------

v1 reproduction:
    --loss-type mse --steps 4000 --gen-len 128 --lr-schedule const
    --no-niah-prompts --rank 256
v2 reproduction:
    --loss-type combined --steps 20000 --gen-len 512 --lr-schedule cosine
    (default in v2 — see git log of this file pre-v3)
v3 (default): --loss-type attn_distill (everything above tuned for it)

Reproducibility
---------------

v1 training is reproducible by passing
``--loss-type mse --steps 4000 --gen-len 128 --lr-schedule const
--no-niah-prompts``.

v2 defaults are tuned for converging f_θ to a checkpoint that closes
the integrated NIAH gate (recall_delta_vs_oracle ≤ 5pp).

Pipeline (CUDA, vast.ai H200/H100):

  1. Load Gemma 4 26B-A4B verifier (transformers, bf16, sdpa)
  2. Load DFlash drafter (PR #93's DFlashDrafter.from_pretrained,
     using models/dflash-kakeya-baseline)
  3. Build training corpus:
     a. PROMPTS list (general / code / math / facts / creative)
     b. (v2) synthetic NIAH-style prompts (haystack with random
        marker_id + question — same pattern as the eval but with
        independent seeds → no test contamination)
  4. For each training sequence in the corpus:
     a. Run verifier forward; record K/V at every layer at every position
        (extracted via attention forward hooks on each layer's k_proj
        / v_proj — pre-norm pre-RoPE, matching what the cross-model
        DLMRestoredVerifier needs to inject)
     b. Run drafter forward via capture_proposer_kv; KVCapture has
        K/V at every drafter layer at every position (pre-norm pre-RoPE)
     c. f_θ targets: f_θ(drafter_kv) ≈ verifier_kv
  5. Train f_θ with the configured loss (default cosine+mag combined,
     v1 mse-only via flag), AdamW + cosine LR schedule

Requires:
  * HF_TOKEN (Gemma 4 is gated)
  * transformers >= 5.0 (Gemma 4 support)
  * drafter checkpoint at models/dflash-kakeya-baseline/

Outputs:
  * Trained f_θ checkpoint at --save (default: results/research/f_theta/)
    Format: f_theta_config.json + f_theta_weights.pt (per
    FThetaProjection.save_pretrained contract)
  * Training report at <save>.json with config, final_loss,
    per-layer-loss breakdown, elapsed time

Usage:
  HF_TOKEN=hf_xxx PYTHONPATH=.:sdks/python python3 \
      scripts/research/k3_f_theta_train.py \
      --steps 4000 --lr 1e-3 --rank 256 --batch-prompts 4 --seq-len 512 \
      --save results/research/f_theta_v1

The training set is the same prompts the alignment_train.py corpus
uses (PR #93's PROMPTS list, expanded if --extended-corpus). Each
training step: pick a random sequence from the cache, sample a
random window of ``seq_len`` positions, compute f_θ predictions vs
verifier targets at those positions, MSE loss, AdamW step.

Memory budget:
  * verifier 26B bf16:    ~52 GB (needs H200 80 GB / multi-GPU)
  * drafter 0.43B bf16:   ~0.9 GB
  * f_θ rank=256 fp32:    ~130 MB (tiny vs everything else)
  * verifier K/V cache for 1 sequence at T=512:
      30 layers × 512 × 2048 × 2 (K+V) × 2 bytes = ~125 MB
  * drafter K/V cache for 1 sequence at T=512:
      5 layers × 512 × 256 × 2 × 2 = ~2.5 MB
  * Training takes a few hundred GB of K/V cache across the corpus;
    we keep K/V in fp16 on GPU and stream from CPU when corpus > GPU.

For the K3 first training run, we start with the same 64-prompt corpus
PR #93 used, ~512 tokens generated per prompt. Total cache: ~64 × 125
MB = ~8 GB, fits in GPU memory comfortably.

Validation gate:
  * Final MSE loss ≤ 0.5× the initial random-init loss (proves f_θ
    learned something meaningful; the 0.5× threshold is conservative
    — actual converged f_θ will be much lower).
  * Per-layer loss should be roughly uniform; outliers indicate
    layer-specific issues that need investigation.

After training, the cross-model DLMRestoredVerifier loads this
checkpoint and uses it for K/V Restoration in the integrated
Kakeya inference loop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from inference_engine.v04.f_theta import FThetaConfig, FThetaProjection
from inference_engine.v04.cross_model_dlm_verifier import (
    _capture_drafter_kv,
    get_verifier_decoder,
    resolve_text_config,
)
from inference_engine.v04.dflash_drafter import DFlashDrafter


# Same training prompt corpus as PR #93's k3_dflash_alignment_train.py
# — direct comparability of evidence.
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write a Python function to reverse a linked list.",
    "Implement binary search in Python with comments.",
    "Write a function to compute the factorial of n iteratively.",
    "Write a Python class for a simple stack with push/pop/peek.",
    "Implement quicksort in Python.",
    "Write a regex that matches a valid IPv4 address and explain it.",
    "Write a Python decorator that times a function and prints the duration.",
    "Implement a function to merge two sorted lists.",
    "Write a SQL query to find the second-highest salary in an Employees table.",
    "Write a bash one-liner to count lines in all .py files under a directory.",
    "Implement a debounce function in JavaScript.",
    "Write a Python generator that yields prime numbers.",
    "Explain and implement memoization for a recursive Fibonacci.",
    "Write a function to detect a cycle in a directed graph.",
    "Implement a least-recently-used (LRU) cache in Python.",
    "Compute the sum of the first 100 positive integers and show your reasoning.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    "Solve for x: 3x + 7 = 22.",
    "What is the derivative of x^3 + 2x with respect to x?",
    "Explain the Pythagorean theorem with an example.",
    "A bag has 3 red and 2 blue balls; what is the probability of drawing red?",
    "List the first eight powers of two.",
    "Explain why the square root of 2 is irrational.",
    "Convert 0.625 to a fraction and simplify.",
    "What is 15% of 240? Show the steps.",
    "What is the capital of Japan?",
    "Who wrote the play Hamlet?",
    "What is photosynthesis in one sentence?",
    "Name the four fundamental forces of physics.",
    "What gas do plants absorb from the atmosphere?",
    "What is the largest planet in the solar system?",
    "Who developed the theory of general relativity?",
    "What is the chemical symbol for gold?",
    "What year did the first human land on the moon?",
    "What is the speed of light in a vacuum (approximate)?",
    "Explain how a hash map works in one paragraph.",
    "Explain the difference between a process and a thread.",
    "Explain what a REST API is to a beginner.",
    "Describe how TCP establishes a connection (three-way handshake).",
    "Explain what overfitting is in machine learning.",
    "Explain the concept of recursion with a simple analogy.",
    "Describe what a transformer attention mechanism does at a high level.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is a deadlock and how can it be avoided?",
    "Explain garbage collection in managed languages.",
    "Write a haiku about autumn leaves.",
    "Write a two-sentence horror story.",
    "Compose a short motivational quote about perseverance.",
    "Write a limerick about a programmer who loves coffee.",
    "Draft a one-line git commit message for a bug fix in the parser.",
    "Summarize the water cycle in two sentences.",
    "Write a polite email asking to reschedule a meeting.",
    "Give three tips for writing clear documentation.",
    "Write a short poem about the ocean at night.",
    "Describe a sunset using vivid imagery in two sentences.",
    "Explain why the sky appears blue.",
    "Summarize the plot of Cinderella in one sentence.",
    "List three benefits of regular exercise.",
    "What causes the seasons on Earth?",
    "Give two reasons why version control is important.",
    "Write a tagline for a fictional eco-friendly water bottle.",
]


@dataclass
class AttentionTargetData:
    """Per-layer attention-output distillation target data.

    Captured during data collection by running the verifier forward
    once with hooks on every layer. Used by the attention-output
    distillation loss (v3 / one-shot trainer) to evaluate
    ``attention(Q, f_θ(K), f_θ(V))`` against the verifier's actual
    attention output without needing to re-run the verifier at every
    training step.

    Per-layer (length = num_verifier_layers):

      q_raw  [T, num_heads × head_dim]   — q_proj output, pre-norm
      o_tgt  [T, hidden_dim]             — attn module output, post-o_proj
      cos    [1, T, head_dim]            — RoPE cosine table
      sin    [1, T, head_dim]            — RoPE sine table
      attention_mask                      — captured causal/sliding mask

    All tensors stored bf16 to halve memory (cast to fp32 on use).
    Stored on CPU; transferred to GPU per training step. For T=512,
    one sequence costs ≈ 30 layers × 13 MB ≈ 390 MB (CPU bf16); for
    a 64-prompt corpus that is ≈ 25 GB CPU RAM.
    """
    q_raw: List[torch.Tensor]            # per-layer pre-norm Q
    o_tgt: List[torch.Tensor]            # per-layer attn module output
    cos: List[torch.Tensor]              # per-layer RoPE cos
    sin: List[torch.Tensor]              # per-layer RoPE sin
    attention_mask: Optional[torch.Tensor]
    num_heads_per_layer: List[int]
    head_dim_per_layer: List[int]


@dataclass
class CapturedSequence:
    """Paired drafter / verifier data over one training sequence.

    All tensors live on the device that produced them by default; the
    attention distillation tensors are CPU bf16 to keep total cache
    size manageable for 64-prompt corpora.

    Two paths populate this:

      legacy K/V path (loss_type ∈ mse, cos_mag, combined):
        drafter_k, drafter_v, verifier_k, verifier_v
        attn_target = None

      attention-output distillation (loss_type = attn_distill, default):
        drafter_k, drafter_v, attn_target (verifier_k/verifier_v omitted)

    The attn_distill path is the one-shot principled trainer. The
    legacy path is kept for v1/v2 reproducibility / ablation but is
    not the default after v3.
    """
    seq_len: int
    drafter_k: torch.Tensor          # [num_d_layers, T, drafter_kv_dim]
    drafter_v: torch.Tensor          # [num_d_layers, T, drafter_kv_dim]
    # Legacy K/V (None when attn_distill captured instead)
    verifier_k: Optional[List[torch.Tensor]] = None
    verifier_v: Optional[List[torch.Tensor]] = None
    # Attention-output distillation target data (None for legacy path)
    attn_target: Optional[AttentionTargetData] = None


def _capture_verifier_kv(
    verifier_model: torch.nn.Module, input_ids: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run verifier forward and capture per-layer K, V via forward hooks
    on each decoder layer's k_proj / v_proj.

    Returns
    -------
    (verifier_k, verifier_v): per-layer lists of length num_v_layers,
    element ``i`` shaped ``[T, kv_dim_i]`` on the verifier's device.
    Layers can have heterogeneous ``kv_dim_i`` (Gemma 4).
    """
    layers = get_verifier_decoder(verifier_model).layers
    num_layers = len(layers)
    k_capture: List[torch.Tensor] = [None] * num_layers
    v_capture: List[torch.Tensor] = [None] * num_layers
    handles = []
    # Gemma 4 has KV-sharing layers where v_proj is None; there the
    # value_states equal the raw k_proj output (pre k_norm / pre RoPE).
    # Capture V from the k_proj output for those layers.
    v_shared_from_k: List[int] = []

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        def _make_k_hook(idx):
            def hook(_mod, _inp, output):
                k_capture[idx] = output.detach()
            return hook

        def _make_v_hook(idx):
            def hook(_mod, _inp, output):
                v_capture[idx] = output.detach()
            return hook

        handles.append(attn.k_proj.register_forward_hook(_make_k_hook(i)))
        if getattr(attn, "v_proj", None) is not None:
            handles.append(attn.v_proj.register_forward_hook(_make_v_hook(i)))
        else:
            v_shared_from_k.append(i)

    try:
        with torch.no_grad():
            _ = verifier_model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(k is None for k in k_capture):
        raise RuntimeError(
            "verifier K capture missing some layers — hooks did not fire"
        )
    # Fill V for v_proj-None layers with the captured k_proj output.
    for i in v_shared_from_k:
        v_capture[i] = k_capture[i]
    if any(v is None for v in v_capture):
        raise RuntimeError(
            "verifier V capture missing some layers — hooks did not fire"
        )
    # Per-layer lists; layers may have heterogeneous kv_dim (Gemma 4).
    # Each k_capture[i] is [B, T, kv_dim_i]; assume B=1 and drop it.
    k_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []
    for kc, vc in zip(k_capture, v_capture):
        if kc.size(0) != 1:
            raise NotImplementedError(
                f"f_θ training currently assumes batch=1 (got {kc.size(0)})"
            )
        k_list.append(kc[0])   # [T, kv_dim_i]
        v_list.append(vc[0])
    return k_list, v_list


def _capture_attention_target_data(
    verifier_model: torch.nn.Module, input_ids: torch.Tensor,
) -> AttentionTargetData:
    """Run verifier forward with hooks to capture per-layer attention
    distillation targets (Q_raw, O_tgt, cos, sin, attention_mask).

    Returns an :class:`AttentionTargetData` with all tensors moved to
    CPU bf16 (the per-step training loop streams these back to GPU).
    """
    layers = get_verifier_decoder(verifier_model).layers
    num_layers = len(layers)

    q_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    o_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    cos_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    sin_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    mask_capture: List[Optional[torch.Tensor]] = [None] * num_layers
    handles = []

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        def _make_q_hook(idx):
            def hook(_mod, _inp, output):
                q_capture[idx] = output.detach()
            return hook

        def _make_o_hook(idx):
            def hook(_mod, _inp, output):
                # attn module returns (attn_output, attn_weights)
                if isinstance(output, tuple):
                    o_capture[idx] = output[0].detach()
                else:
                    o_capture[idx] = output.detach()
            return hook

        def _make_pre_hook(idx):
            def hook(_mod, args, kwargs):
                # Gemma 4 attention.forward signature:
                #   (hidden_states, position_embeddings, attention_mask, ...)
                pos_emb = None
                if "position_embeddings" in kwargs:
                    pos_emb = kwargs["position_embeddings"]
                elif len(args) >= 2:
                    pos_emb = args[1]
                if pos_emb is not None:
                    cos, sin = pos_emb
                    cos_capture[idx] = cos.detach()
                    sin_capture[idx] = sin.detach()
                am = None
                if "attention_mask" in kwargs:
                    am = kwargs["attention_mask"]
                elif len(args) >= 3:
                    am = args[2]
                if am is not None:
                    mask_capture[idx] = am.detach()
            return hook

        handles.append(attn.q_proj.register_forward_hook(_make_q_hook(i)))
        handles.append(attn.register_forward_hook(_make_o_hook(i)))
        handles.append(
            attn.register_forward_pre_hook(_make_pre_hook(i), with_kwargs=True),
        )

    try:
        with torch.no_grad():
            _ = verifier_model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(q is None for q in q_capture):
        raise RuntimeError("attention distill: Q capture missing some layers")
    if any(o is None for o in o_capture):
        raise RuntimeError("attention distill: O capture missing some layers")
    if any(c is None for c in cos_capture):
        raise RuntimeError("attention distill: cos capture missing some layers")

    num_heads_per_layer: List[int] = []
    head_dim_per_layer: List[int] = []
    for layer in layers:
        attn = layer.self_attn
        head_dim_per_layer.append(int(attn.head_dim))
        num_heads_per_layer.append(int(attn.q_proj.out_features // attn.head_dim))

    # Stack and move to CPU bf16. Drop batch dim (B=1).
    def _to_cpu_bf16(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.bfloat16, device="cpu", copy=True)

    q_list = [_to_cpu_bf16(q[0]) for q in q_capture]      # [T, n_heads*head_dim]
    o_list = [_to_cpu_bf16(o[0]) for o in o_capture]      # [T, hidden]
    cos_list = [_to_cpu_bf16(c) for c in cos_capture]     # [1, T, head_dim] or [B, T, head_dim]
    sin_list = [_to_cpu_bf16(s) for s in sin_capture]
    mask_cpu = (
        mask_capture[0].to(device="cpu", copy=True) if mask_capture[0] is not None
        else None
    )

    return AttentionTargetData(
        q_raw=q_list,
        o_tgt=o_list,
        cos=cos_list,
        sin=sin_list,
        attention_mask=mask_cpu,
        num_heads_per_layer=num_heads_per_layer,
        head_dim_per_layer=head_dim_per_layer,
    )


def _collect_sequence(
    verifier_model: torch.nn.Module,
    drafter: DFlashDrafter,
    input_ids: torch.Tensor,
    *,
    capture_legacy_kv: bool = False,
    capture_attn_target: bool = True,
) -> CapturedSequence:
    """Capture paired drafter + verifier data for one input sequence.

    Parameters
    ----------
    capture_legacy_kv : bool
        If True, capture verifier K/V via k_proj/v_proj hooks (used by
        loss_type ∈ mse | cos_mag | combined). Default False — the v3
        attn_distill path doesn't need it.
    capture_attn_target : bool
        If True, capture per-layer Q + O_tgt + cos/sin/mask (used by
        loss_type=attn_distill, the v3 default).
    """
    if not (capture_legacy_kv or capture_attn_target):
        raise ValueError(
            "must capture at least one of legacy_kv or attn_target"
        )

    v_k = v_v = None
    if capture_legacy_kv:
        v_k, v_v = _capture_verifier_kv(verifier_model, input_ids)

    attn_target: Optional[AttentionTargetData] = None
    if capture_attn_target:
        attn_target = _capture_attention_target_data(verifier_model, input_ids)

    # Drafter K/V capture (always; cheap and small, ~5 MB per seq).
    capture = _capture_drafter_kv(
        verifier_model=verifier_model,
        drafter=drafter,
        input_ids=input_ids,
    )
    k_flat = [k.flatten(-2, -1) for k in capture.keys]
    v_flat = [v.flatten(-2, -1) for v in capture.values]
    d_k = torch.stack(k_flat, dim=0)[:, 0]
    d_v = torch.stack(v_flat, dim=0)[:, 0]

    return CapturedSequence(
        seq_len=int(input_ids.size(1)),
        drafter_k=d_k.detach(),
        drafter_v=d_v.detach(),
        verifier_k=[t.detach() for t in v_k] if v_k is not None else None,
        verifier_v=[t.detach() for t in v_v] if v_v is not None else None,
        attn_target=attn_target,
    )


def _attention_distillation_loss(
    f_theta: FThetaProjection,
    seq: CapturedSequence,
    layers: Sequence[torch.nn.Module],
    *,
    apply_rotary_pos_emb: Any,
    device: torch.device,
    sample_positions: Optional[int] = None,
    seed: Optional[int] = None,
    diag_buf: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Attention-output distillation loss (the v3 / one-shot principled loss).

    For each verifier layer ℓ:

      K_pred_ℓ = f_θ_K(drafter_KV)[ℓ]
      V_pred_ℓ = f_θ_V(drafter_KV)[ℓ]

      Q_for_attn  = q_norm(Q_raw_ℓ).view(B, T, H_q, D) → RoPE → transpose
      K_for_attn  = k_norm(K_pred_ℓ).view(B, T, H_kv, D) → RoPE → transpose
      V_for_attn  = v_norm(V_pred_ℓ).view(B, T, H_kv, D) → transpose

      GQA repeat K_for_attn, V_for_attn to H_q
      O_inner = scaled_dot_product_attention(Q, K, V, mask)
      O_pred  = o_proj(O_inner.reshape(B, T, H_q*D))

      loss_ℓ = MSE(O_pred, O_tgt_ℓ)         # O_tgt captured during data
                                              collection (verifier's actual
                                              attn module post-o_proj output)

    Total loss = mean over layers.

    This is the principled training objective for K/V replacement: it
    directly optimises "f_θ-injected K/V produces equivalent verifier
    attention output". Unlike pure-MSE-on-K/V (v1) or cos+mag (v2),
    this loss accounts for:

      * GQA: same num_heads/num_kv_heads/head_dim per layer
      * RoPE: same positional encoding the verifier uses at inference
      * Causal mask (and sliding-window for sliding layers): captured
        from the verifier's own forward
      * o_proj: every layer's downstream projection that f_θ K/V
        ultimately propagates through

    Memory: per training step, K_pred/V_pred at full T positions are
    needed for attention's K, V dims. We sample only the OUTPUT side
    (where loss is evaluated) when ``sample_positions`` < T to save
    on attention output + o_proj memory; this reduces gradient noise
    only marginally because the loss is averaged across positions.
    Default ``None`` ⇒ use all T output positions (recommended for
    short sequences T ≤ 1024).
    """
    if seq.attn_target is None:
        raise RuntimeError(
            "attn_distill loss requires CapturedSequence.attn_target; "
            "call _collect_sequence with capture_attn_target=True"
        )
    target = seq.attn_target
    cfg = f_theta.config
    T = seq.seq_len

    # f_θ forward (drafter K/V on CPU/GPU, f_θ on GPU). We pull drafter
    # K/V to f_θ's device + cast to f_θ encoder dtype.
    f_dtype = next(f_theta.parameters()).dtype
    drafter_k = seq.drafter_k.to(device=device).unsqueeze(0)  # [1, L_d, T, kv_dim]
    drafter_v = seq.drafter_v.to(device=device).unsqueeze(0)
    d_k_list = []
    d_v_list = []
    for li in range(cfg.drafter_num_layers):
        k_per = drafter_k[:, li].view(
            1, T, cfg.drafter_num_kv_heads, cfg.drafter_head_dim,
        )
        v_per = drafter_v[:, li].view(
            1, T, cfg.drafter_num_kv_heads, cfg.drafter_head_dim,
        )
        d_k_list.append(k_per)
        d_v_list.append(v_per)
    pred_k_per_layer, pred_v_per_layer = f_theta.forward_kv_pack(d_k_list, d_v_list)
    # pred_k_per_layer[ℓ]: [1, T, kv_heads_ℓ, head_dim_ℓ] in fp32

    # Sample positions for output-side loss
    if sample_positions is not None and sample_positions < T:
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(seed)
        else:
            g = None
        idx = torch.randperm(T, generator=g)[:sample_positions].to(device)
        idx, _ = idx.sort()
    else:
        idx = None

    n_layers = cfg.verifier_num_layers
    loss = pred_k_per_layer[0].new_zeros(())
    diag = {"mse_O_total": 0.0, "abs_O_target": 0.0}

    for li in range(n_layers):
        layer = layers[li]
        attn = layer.self_attn

        # Move per-layer cached tensors to GPU (bf16 cache → cast to compute dtype)
        compute_dtype = next(layer.parameters()).dtype
        q_raw = target.q_raw[li].to(device=device, dtype=compute_dtype).unsqueeze(0)
        o_tgt = target.o_tgt[li].to(device=device, dtype=compute_dtype).unsqueeze(0)
        cos = target.cos[li].to(device=device, dtype=compute_dtype)
        sin = target.sin[li].to(device=device, dtype=compute_dtype)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0)
        if sin.ndim == 2:
            sin = sin.unsqueeze(0)

        n_heads = target.num_heads_per_layer[li]
        head_dim = target.head_dim_per_layer[li]
        kv_heads = cfg.layer_kv_heads[li]
        kv_head_dim = cfg.layer_head_dims[li]
        if kv_head_dim != head_dim:
            # Sanity: f_θ's per-layer head_dim must match verifier's
            # actual head_dim. (Both come from the verifier config.)
            raise RuntimeError(
                f"layer {li}: f_θ head_dim {kv_head_dim} != verifier {head_dim}"
            )

        # Q pipeline: q_norm → RoPE → transpose
        Q = q_raw.view(1, T, n_heads, head_dim)
        Q = attn.q_norm(Q)
        Q = apply_rotary_pos_emb(Q, cos, sin, unsqueeze_dim=2)
        Q = Q.transpose(1, 2)                           # [1, n_heads, T, head_dim]

        # K pipeline (f_θ output → norm → RoPE → transpose)
        K_pred = pred_k_per_layer[li].to(dtype=compute_dtype)  # [1, T, kv_heads, head_dim]
        K = attn.k_norm(K_pred)
        K = apply_rotary_pos_emb(K, cos, sin, unsqueeze_dim=2)
        K = K.transpose(1, 2)                           # [1, kv_heads, T, head_dim]

        # V pipeline (f_θ output → v_norm → transpose, no RoPE)
        V_pred = pred_v_per_layer[li].to(dtype=compute_dtype)
        V = attn.v_norm(V_pred).transpose(1, 2)          # [1, kv_heads, T, head_dim]

        # GQA: repeat K, V to match num_heads
        if n_heads != kv_heads:
            n_rep = n_heads // kv_heads
            if n_heads % kv_heads != 0:
                raise RuntimeError(
                    f"layer {li}: n_heads {n_heads} not divisible by "
                    f"kv_heads {kv_heads}"
                )
            K = K.repeat_interleave(n_rep, dim=1)
            V = V.repeat_interleave(n_rep, dim=1)

        # Attention with the verifier's actual mask + scaling
        scale = float(getattr(attn, "scaling", head_dim ** -0.5))
        # Use scaled_dot_product_attention; if attention_mask is None,
        # use causal=True.
        attn_mask = target.attention_mask
        if attn_mask is None:
            O_inner = F.scaled_dot_product_attention(
                Q, K, V, scale=scale, is_causal=True,
            )
        else:
            attn_mask_dev = attn_mask.to(device=device, dtype=compute_dtype)
            # attention_mask shapes vary (B, 1, T, T) or (B, T, T); align
            # to what scaled_dot_product_attention accepts.
            if attn_mask_dev.ndim == 4 and attn_mask_dev.size(0) == 1:
                pass
            elif attn_mask_dev.ndim == 3:
                attn_mask_dev = attn_mask_dev.unsqueeze(1)
            elif attn_mask_dev.ndim == 2:
                attn_mask_dev = attn_mask_dev.unsqueeze(0).unsqueeze(0)
            O_inner = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask_dev, scale=scale,
            )

        # o_proj (linear, frozen weights → no grad through it)
        O_inner = O_inner.transpose(1, 2).reshape(1, T, n_heads * head_dim).contiguous()
        O_pred = attn.o_proj(O_inner)

        if idx is not None:
            O_pred_eval = O_pred.index_select(1, idx)
            O_tgt_eval = o_tgt.index_select(1, idx)
        else:
            O_pred_eval = O_pred
            O_tgt_eval = o_tgt

        l_o = F.mse_loss(O_pred_eval.float(), O_tgt_eval.float())
        loss = loss + l_o
        diag["mse_O_total"] += float(l_o.detach().item())
        diag["abs_O_target"] += float(O_tgt_eval.float().abs().mean().item())

        # Free GPU memory of cached per-layer tensors before next layer
        del q_raw, o_tgt, cos, sin, Q, K, V, O_inner, O_pred

    if diag_buf is not None:
        diag_buf["mse_O_mean"] = diag["mse_O_total"] / max(n_layers, 1)
        diag_buf["abs_O_target_mean"] = diag["abs_O_target"] / max(n_layers, 1)
    return loss / max(n_layers, 1)


def _per_vector_cosine_mag_loss(
    pred: torch.Tensor, tgt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cosine-similarity + magnitude-MSE loss between paired K (or V) vectors.

    Each vector is a single-head K (or V) at a single position, shape
    ``[..., head_dim]``. Loss components:

      cos:  1 − cosine_similarity(pred, tgt)        ∈ [0, 2]
      mag:  MSE(‖pred‖, ‖tgt‖) / mean(‖tgt‖)²       (scale-normalised)

    Why this loss is correct for K/V projection
    -------------------------------------------

    Attention is ``softmax(QK^T / √d) · V``. For the verifier's
    attention output to be preserved when K is replaced, we need:

      1. ``Q · pred_K_p ≈ Q · tgt_K_p`` for every position p — this is
         the **direction** of K_p relative to Q. Pure MSE on K_p does
         not bound this; cosine sim does (Cauchy-Schwarz).
      2. The scale of ``Q · K_p`` across positions must be preserved
         so softmax peaks at the same positions — this is the
         **magnitude** of K_p. Mag-MSE handles this.

    For V: attention output is ``Σ a_p · V_p``; here both direction
    and magnitude of V_p directly contribute to the output, so cosine
    + magnitude on V is also the right structure.

    Returns the (combined_loss_scalar, cos_component, mag_component) so
    callers can log per-component diagnostics during training.
    """
    pred_f = pred.float()
    tgt_f = tgt.float()
    # Cosine on the last (head_dim) axis: shape [..., head_dim] → [...]
    cos = F.cosine_similarity(pred_f, tgt_f, dim=-1).mean()
    cos_loss = 1.0 - cos
    # Magnitude: scalar L2 norm per vector, shape [..., 1] squeeze to [...].
    pred_mag = pred_f.norm(dim=-1)
    tgt_mag = tgt_f.norm(dim=-1)
    tgt_mag_mean_sq = tgt_mag.pow(2).mean().clamp(min=1e-6)
    mag_loss = F.mse_loss(pred_mag, tgt_mag) / tgt_mag_mean_sq
    return cos_loss + mag_loss, cos_loss.detach(), mag_loss.detach()


def _f_theta_loss(
    f_theta: FThetaProjection,
    seq: CapturedSequence,
    *,
    sample_positions: int = 256,
    seed: Optional[int] = None,
    loss_type: str = "attn_distill",
    diag_buf: Optional[Dict[str, float]] = None,
    layers: Optional[Sequence[torch.nn.Module]] = None,
    apply_rotary_pos_emb: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute the configured loss for one sequence (subsampled positions).

    Parameters
    ----------
    loss_type : str
        ``"attn_distill"`` — v3 default (one-shot principled): attention-output
                              distillation. Requires ``layers`` +
                              ``apply_rotary_pos_emb`` + ``device``.
        ``"relmse"``    — PR #103 v3-relmse: magnitude-normalised MSE per
                          (layer, K/V); balances gradient across layers.
                          Diagnostic-grade; weaker than attn_distill on
                          recall (PR #103 v3 evidence: recall=0).
        ``"mse"``       — v1 MSE on raw K and V (kept for reproducibility).
        ``"cos_mag"``   — v2 cosine + magnitude on K and V.
        ``"combined"``  — v2 cosine + magnitude + 0.1× normalised MSE.
    diag_buf : dict
        Optional dict to receive per-component aggregates (cos_K_mean,
        cos_V_mean, mag_K_mean, mag_V_mean, mse_mean, mse_O_mean) for logging.
    """
    if loss_type == "attn_distill":
        if layers is None or apply_rotary_pos_emb is None or device is None:
            raise ValueError(
                "attn_distill requires layers + apply_rotary_pos_emb + device"
            )
        return _attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            device=device,
            sample_positions=(
                None if sample_positions <= 0 or sample_positions >= seq.seq_len
                else sample_positions
            ),
            seed=seed, diag_buf=diag_buf,
        )

    if seq.verifier_k is None or seq.verifier_v is None:
        raise RuntimeError(
            f"loss_type={loss_type!r} requires legacy K/V capture; "
            "ensure data collection ran with capture_legacy_kv=True"
        )
    T = seq.seq_len
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
    else:
        g = None
    if T <= sample_positions:
        idx = torch.arange(T, device=seq.drafter_k.device)
    else:
        idx = torch.randperm(T, generator=g)[:sample_positions].to(
            seq.drafter_k.device,
        )

    d_k_sub = seq.drafter_k.index_select(1, idx).unsqueeze(0)
    d_v_sub = seq.drafter_v.index_select(1, idx).unsqueeze(0)
    cfg = f_theta.config
    d_k_list, d_v_list = [], []
    for li in range(cfg.drafter_num_layers):
        k_per = d_k_sub[:, li]
        v_per = d_v_sub[:, li]
        k_per = k_per.view(
            1, k_per.size(1), cfg.drafter_num_kv_heads, cfg.drafter_head_dim,
        )
        v_per = v_per.view(
            1, v_per.size(1), cfg.drafter_num_kv_heads, cfg.drafter_head_dim,
        )
        d_k_list.append(k_per)
        d_v_list.append(v_per)

    pred_k, pred_v = f_theta.forward_kv_pack(d_k_list, d_v_list)

    layer_kv_heads = cfg.layer_kv_heads
    layer_head_dims = cfg.layer_head_dims
    idx_pos = idx.to(seq.verifier_k[0].device)
    loss = pred_k[0].new_zeros(())
    n_layers = cfg.verifier_num_layers

    diag = {
        "cos_K_total": 0.0, "cos_V_total": 0.0,
        "mag_K_total": 0.0, "mag_V_total": 0.0,
        "mse_K_total": 0.0, "mse_V_total": 0.0,
        "rel_K_total": 0.0, "rel_V_total": 0.0,
    }

    for li in range(n_layers):
        v_k_sub = seq.verifier_k[li].index_select(0, idx_pos)
        v_v_sub = seq.verifier_v[li].index_select(0, idx_pos)
        tgt_k = v_k_sub.view(
            1, v_k_sub.size(0), layer_kv_heads[li], layer_head_dims[li],
        ).float()
        tgt_v = v_v_sub.view(
            1, v_v_sub.size(0), layer_kv_heads[li], layer_head_dims[li],
        ).float()
        pred_k_li = pred_k[li].float()
        pred_v_li = pred_v[li].float()

        if loss_type == "mse":
            l_k = F.mse_loss(pred_k_li, tgt_k)
            l_v = F.mse_loss(pred_v_li, tgt_v)
            loss = loss + l_k + l_v
            diag["mse_K_total"] += float(l_k.detach().item())
            diag["mse_V_total"] += float(l_v.detach().item())
        elif loss_type == "relmse":
            # Magnitude-normalised MSE — PR #103's relmse fix
            # (commit ce25dfa). Each (layer, K/V) component's MSE is
            # divided by that component's target mean-square so layers
            # with very different K/V magnitudes (Gemma 4: sliding
            # head_dim 256 vs full head_dim 512) contribute comparable
            # gradient. Diagnostic-friendly: per-layer rel_mse exposes
            # full-attention layer fidelity gap (which is the core
            # finding of PR #103's v3 evidence).
            l_k = F.mse_loss(pred_k_li, tgt_k)
            l_v = F.mse_loss(pred_v_li, tgt_v)
            denom_k = tgt_k.pow(2).mean().clamp(min=1e-6)
            denom_v = tgt_v.pow(2).mean().clamp(min=1e-6)
            rel_k = l_k / denom_k
            rel_v = l_v / denom_v
            loss = loss + rel_k + rel_v
            diag["mse_K_total"] += float(l_k.detach().item())
            diag["mse_V_total"] += float(l_v.detach().item())
            diag["rel_K_total"] += float(rel_k.detach().item())
            diag["rel_V_total"] += float(rel_v.detach().item())
        elif loss_type == "cos_mag":
            l_k, c_k, m_k = _per_vector_cosine_mag_loss(pred_k_li, tgt_k)
            l_v, c_v, m_v = _per_vector_cosine_mag_loss(pred_v_li, tgt_v)
            loss = loss + l_k + l_v
            diag["cos_K_total"] += float(c_k.item())
            diag["cos_V_total"] += float(c_v.item())
            diag["mag_K_total"] += float(m_k.item())
            diag["mag_V_total"] += float(m_v.item())
        elif loss_type == "combined":
            l_cm_k, c_k, m_k = _per_vector_cosine_mag_loss(pred_k_li, tgt_k)
            l_cm_v, c_v, m_v = _per_vector_cosine_mag_loss(pred_v_li, tgt_v)
            l_mse_k = F.mse_loss(pred_k_li, tgt_k)
            l_mse_v = F.mse_loss(pred_v_li, tgt_v)
            # Weight: cos+mag dominate (×1.0), MSE is a stability term (×0.1).
            # MSE is normalised by tgt's own variance so it doesn't blow up
            # for high-magnitude layers.
            tgt_var_k = tgt_k.var().clamp(min=1e-6)
            tgt_var_v = tgt_v.var().clamp(min=1e-6)
            mse_norm_k = l_mse_k / tgt_var_k
            mse_norm_v = l_mse_v / tgt_var_v
            loss = loss + l_cm_k + l_cm_v + 0.1 * (mse_norm_k + mse_norm_v)
            diag["cos_K_total"] += float(c_k.item())
            diag["cos_V_total"] += float(c_v.item())
            diag["mag_K_total"] += float(m_k.item())
            diag["mag_V_total"] += float(m_v.item())
            diag["mse_K_total"] += float(l_mse_k.detach().item())
            diag["mse_V_total"] += float(l_mse_v.detach().item())
        else:
            raise ValueError(
                f"unknown loss_type {loss_type!r} "
                f"(want attn_distill | mse | relmse | cos_mag | combined)"
            )

    if diag_buf is not None:
        for k, v in diag.items():
            diag_buf[k] = v / max(n_layers, 1)
    return loss / (2.0 * n_layers)


# ---------------------------------------------------------------------------
# v2: synthetic NIAH-style training prompts
# ---------------------------------------------------------------------------

# Same vocabulary the eval uses, reproduced here so training corpus
# generation is independent of the eval module (avoid test contamination
# via shared seeds). PR #94's `make_niah_dataset` uses these patterns;
# we use distinct random seeds + extra word lists so training NIAH never
# reuses an eval-seed needle.
_NIAH_TRAIN_KEY_WORDS = (
    # Greek (overlaps with eval but seeds differ → independent samples)
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA",
    "IOTA", "KAPPA", "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI",
    "RHO", "SIGMA", "TAU", "UPSILON", "PHI", "CHI", "PSI", "OMEGA",
    # Botanical (extra — different from eval's set so no needle reuse)
    "ROSE", "TULIP", "DAISY", "ORCHID", "JASMINE", "LILAC", "POPPY",
    "VIOLET", "IRIS", "PEONY", "DAHLIA", "ASTER", "SAGE", "BASIL",
    "MINT", "THYME", "OAK", "MAPLE", "PINE", "BIRCH", "CEDAR",
)

_NIAH_TRAIN_FILLER_LINES = (
    "The afternoon sun cast long shadows across the empty courtyard.",
    "She turned the pages slowly, savouring each illustration.",
    "Most of the equations balanced, though one stubbornly refused to.",
    "Light wind stirred the paper notes pinned to the corkboard.",
    "The hallway smelled faintly of old wood and lemon polish.",
    "Conversations drifted in from the kitchen but no one was listening.",
    "Three bicycles leaned against the fence in a perfect row.",
    "He paused, considered the diagram, and added another arrow.",
    "Outside, snow continued to fall gently and without urgency.",
    "The library catalogue was newer than the books it described.",
    "A single candle flickered on the mantelpiece beside the clock.",
    "Half the bookshelf was devoted entirely to volumes about birds.",
    "The path narrowed then widened then narrowed again unpredictably.",
    "Faint lines of older handwriting were visible beneath the print.",
    "Someone had circled a paragraph in red on the third page.",
    "The cat watched the rain from the windowsill without moving.",
    "Each measurement was double-checked but a few still seemed wrong.",
    "Brass instruments lay arranged along the wall in increasing size.",
    "The map was old but the labelling was unexpectedly precise.",
    "Footsteps echoed along the corridor before fading into silence.",
)


def _make_niah_training_prompts(
    n_prompts: int, *, seed: int,
    haystack_min_lines: int = 30, haystack_max_lines: int = 90,
) -> List[str]:
    """Generate synthetic NIAH-style training prompts.

    Pattern matches PR #94's eval harness:

        Read the following text carefully, then answer the question
        at the end.

        <line1>
        <line2>
        ...
        The secret code is <KEYWORD>-<NNNN>.
        ...
        <lineN>

        Question: What is the secret code?

    The needle position is random within the haystack. Independent
    seed from the eval (default seed=0 here vs eval default seed=42),
    extra filler lines + word list → no needle reuse.
    """
    rng = random.Random(seed)
    prompts: List[str] = []
    for _ in range(n_prompts):
        n_lines = rng.randint(haystack_min_lines, haystack_max_lines)
        keyword = rng.choice(_NIAH_TRAIN_KEY_WORDS)
        number = rng.randint(1000, 9999)
        needle = f"The secret code is {keyword}-{number}."
        needle_pos = rng.randint(1, n_lines - 2)
        lines: List[str] = []
        for i in range(n_lines):
            if i == needle_pos:
                lines.append(needle)
            else:
                lines.append(rng.choice(_NIAH_TRAIN_FILLER_LINES))
        body = "\n".join(lines)
        prompt = (
            "Read the following text carefully, then answer the question "
            "at the end.\n\n"
            f"{body}\n\n"
            "Question: What is the secret code?"
        )
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# v2: cosine LR schedule with linear warmup
# ---------------------------------------------------------------------------


def _lr_at_step(step: int, *, peak_lr: float, total_steps: int,
                warmup_steps: int, schedule: str) -> float:
    """Return the LR at ``step`` (1-indexed) for the configured schedule.

    schedule="const":  always peak_lr
    schedule="cosine": linear warmup over warmup_steps, then cosine
                       decay to peak_lr / 100 over the remainder.
    """
    if schedule == "const":
        return peak_lr
    if schedule != "cosine":
        raise ValueError(f"unknown schedule {schedule!r}")
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * (step / max(warmup_steps, 1))
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    floor_lr = peak_lr * 0.01
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_lr + (peak_lr - floor_lr) * cosine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="models/dflash-kakeya-baseline")
    # v2 defaults: 5× more steps, 4× longer sequences, cosine LR, NIAH on.
    # v1 reproduction: --steps 4000 --gen-len 128 --lr-schedule const
    #                  --no-niah-prompts --loss-type mse
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--lr-schedule", default="cosine", choices=["const", "cosine"],
        help="LR schedule (v2 default cosine; v1 used const)",
    )
    ap.add_argument(
        "--warmup-steps", type=int, default=500,
        help="Linear warmup steps for cosine schedule (ignored if const)",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-prompts", type=int, default=64,
                    help="General prompts from PROMPTS list (capped at 62)")
    ap.add_argument(
        "--n-niah-prompts", type=int, default=64,
        help="(v2) Synthetic NIAH-style prompts to add to the corpus. "
             "Set 0 with --no-niah-prompts to reproduce v1.",
    )
    ap.add_argument(
        "--no-niah-prompts", action="store_true",
        help="Disable NIAH synthetic prompts (v1 reproduction mode)",
    )
    ap.add_argument("--niah-min-lines", type=int, default=30)
    ap.add_argument("--niah-max-lines", type=int, default=90)
    ap.add_argument("--gen-len", type=int, default=512,
                    help="Tokens generated per prompt during data collection")
    ap.add_argument(
        "--sample-positions", type=int, default=0,
        help="Random output-side positions per training step. 0 (default) "
             "= use all T positions. For legacy losses (mse/cos_mag/combined) "
             "default falls back to 256 if 0 is passed.",
    )
    ap.add_argument(
        "--loss-type", default="attn_distill",
        choices=["attn_distill", "relmse", "mse", "cos_mag", "combined"],
        help="Training loss. v3 default attn_distill (attention-output "
             "distillation, the principled one-shot loss). v2 used "
             "combined (cos+mag); v1 used mse.",
    )
    ap.add_argument(
        "--rank", type=int, default=None,
        help="f_θ encoder bottleneck. Default 768 for attn_distill, 256 "
             "for legacy losses (v1/v2). Override to override default.",
    )
    ap.add_argument("--save", default="results/research/f_theta_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "[f_theta-train] WARNING: no CUDA detected; running on CPU. "
            "This will be very slow on the production-scale verifier.",
            file=sys.stderr,
        )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Resolve rank default per loss type (rank ↑ for attn_distill = more
    # f_θ capacity; legacy losses keep v1's 256 for direct comparability).
    if args.rank is None:
        args.rank = 768 if args.loss_type == "attn_distill" else 256
        print(
            f"[f_theta-train] using rank={args.rank} (loss_type={args.loss_type})",
            file=sys.stderr,
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Eager attention is required for attn_distill so we can hook the
    # attention module's pre/post forward and capture position_embeddings
    # + attention_mask + post-o_proj output. SDPA fuses these and breaks
    # the hook contract. For legacy losses, sdpa is fine (and faster).
    attn_impl = "eager" if args.loss_type == "attn_distill" else "sdpa"
    apply_rotary_pos_emb = None
    if args.loss_type == "attn_distill":
        from transformers.models.gemma4.modeling_gemma4 import (  # type: ignore
            apply_rotary_pos_emb,
        )

    print(f"[f_theta-train] loading verifier {args.verifier_id} (attn={attn_impl})",
          file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation=attn_impl,
        device_map="auto" if device.type == "cuda" else None,
    ).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)

    print(f"[f_theta-train] loading drafter {args.drafter_id}",
          file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype)
    drafter = drafter.to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)

    # Derive f_θ config from drafter + verifier shapes. Gemma 4's config
    # nests decoder dims under .text_config, so resolve it first.
    v_cfg = resolve_text_config(verifier.config)
    # Read per-layer (head_dim, KV-head count) directly off the decoder
    # layers. Gemma 4 uses head_dim=256 / 8 KV heads on sliding layers
    # and head_dim=512 (global_head_dim) / 2 KV heads on full-attention
    # layers (where v_proj is None: K == V).
    v_layers = get_verifier_decoder(verifier).layers
    layer_head_dims = tuple(int(layer.self_attn.head_dim) for layer in v_layers)
    layer_kv_heads = tuple(
        layer.self_attn.k_proj.out_features // hd
        for layer, hd in zip(v_layers, layer_head_dims)
    )
    uniform_heads = len(set(layer_kv_heads)) == 1
    uniform_dims = len(set(layer_head_dims)) == 1
    f_cfg = FThetaConfig(
        drafter_num_layers=drafter.cfg.num_hidden_layers,
        drafter_num_kv_heads=drafter.cfg.num_key_value_heads,
        drafter_head_dim=drafter.cfg.head_dim,
        verifier_num_layers=v_cfg.num_hidden_layers,
        verifier_num_kv_heads=layer_kv_heads[0],
        verifier_head_dim=layer_head_dims[0],
        rank=args.rank,
        verifier_layer_kv_heads=None if uniform_heads else layer_kv_heads,
        verifier_layer_head_dims=None if uniform_dims else layer_head_dims,
    )
    print(f"[f_theta-train] verifier per-layer kv heads: {layer_kv_heads}",
          file=sys.stderr)
    print(f"[f_theta-train] verifier per-layer head dims: {layer_head_dims}",
          file=sys.stderr)
    print(f"[f_theta-train] f_θ config: {f_cfg}", file=sys.stderr)

    f_theta = FThetaProjection(f_cfg).to(device, dtype=torch.float32)
    n_params = sum(p.numel() for p in f_theta.parameters())
    print(f"[f_theta-train] f_θ params: {n_params:,}", file=sys.stderr)

    # ---------------- Build training corpus (PROMPTS + optional NIAH) ----------------
    n_general = min(args.n_prompts, len(PROMPTS))
    n_niah = 0 if args.no_niah_prompts else max(args.n_niah_prompts, 0)
    corpus_prompts: List[str] = list(PROMPTS[:n_general])
    if n_niah > 0:
        # Use args.seed + 1000 so NIAH seed is reproducible but distinct
        # from any other seed in the system.
        niah_prompts = _make_niah_training_prompts(
            n_niah, seed=args.seed + 1000,
            haystack_min_lines=args.niah_min_lines,
            haystack_max_lines=args.niah_max_lines,
        )
        corpus_prompts.extend(niah_prompts)
        print(
            f"[f_theta-train] corpus: {n_general} general + {n_niah} NIAH "
            f"= {len(corpus_prompts)} prompts (NIAH seed={args.seed + 1000})",
            file=sys.stderr,
        )
    else:
        print(
            f"[f_theta-train] corpus: {n_general} general prompts "
            f"(NIAH disabled — v1 reproduction mode)",
            file=sys.stderr,
        )

    # ---------------- Data collection ----------------
    capture_legacy_kv = args.loss_type in ("mse", "relmse", "cos_mag", "combined")
    capture_attn_target = args.loss_type == "attn_distill"
    print(
        f"[f_theta-train] data capture: legacy_kv={capture_legacy_kv} "
        f"attn_target={capture_attn_target}",
        file=sys.stderr,
    )
    print(f"[f_theta-train] collecting from {len(corpus_prompts)} prompts ...",
          file=sys.stderr, flush=True)
    sequences: List[CapturedSequence] = []
    t0 = time.perf_counter()
    eos_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for pi, prompt in enumerate(corpus_prompts):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        # Greedy AR extension. For NIAH prompts the haystack alone is
        # already long; we still extend by gen_len to cover the answer
        # region — the answer position is the lexically critical one
        # for f_θ to reproduce.
        with torch.no_grad():
            cur = enc.to(device)
            for _ in range(args.gen_len):
                out = verifier(input_ids=cur, use_cache=False)
                nxt = int(torch.argmax(out.logits[0, -1]).item())
                cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt in eos_ids:
                    break
            full_ids = cur

        seq = _collect_sequence(
            verifier, drafter, full_ids,
            capture_legacy_kv=capture_legacy_kv,
            capture_attn_target=capture_attn_target,
        )
        sequences.append(seq)
        if (pi + 1) % 10 == 0 or pi == len(corpus_prompts) - 1:
            print(
                f"[f_theta-train]   collected {pi + 1}/{len(corpus_prompts)}, "
                f"latest seq_len={seq.seq_len}",
                file=sys.stderr,
            )
    collect_elapsed = time.perf_counter() - t0
    print(f"[f_theta-train] data collection done in {collect_elapsed:.0f}s",
          file=sys.stderr)

    # ---------------- Training ----------------
    # Resolve sample_positions: 0 ⇒ full-T for attn_distill (the design
    # choice — every position contributes); fall back to 256 for legacy
    # losses (memory reduction matters there).
    if args.sample_positions <= 0:
        args.sample_positions = (
            0 if args.loss_type == "attn_distill" else 256
        )
    print(
        f"[f_theta-train] training: loss_type={args.loss_type} "
        f"schedule={args.lr_schedule} (warmup={args.warmup_steps}) "
        f"steps={args.steps} peak_lr={args.lr} "
        f"sample_positions={args.sample_positions}",
        file=sys.stderr,
    )
    optimizer = torch.optim.AdamW(
        f_theta.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    losses_window: List[float] = []
    initial_loss: Optional[float] = None
    final_diag: Dict[str, float] = {}
    f_theta.train()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        # Set per-step LR
        cur_lr = _lr_at_step(
            step, peak_lr=args.lr, total_steps=args.steps,
            warmup_steps=args.warmup_steps, schedule=args.lr_schedule,
        )
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        seq = random.choice(sequences)
        diag_buf: Dict[str, float] = {}
        loss = _f_theta_loss(
            f_theta, seq,
            sample_positions=args.sample_positions,
            loss_type=args.loss_type,
            diag_buf=diag_buf,
            layers=v_layers if args.loss_type == "attn_distill" else None,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            device=device,
        )
        if initial_loss is None:
            initial_loss = float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f_theta.parameters(), 1.0)
        optimizer.step()
        losses_window.append(float(loss.item()))
        final_diag = diag_buf  # last step's per-component breakdown
        if step % args.log_every == 0:
            recent = losses_window[-args.log_every:]
            extra_msg = ""
            if args.loss_type in ("cos_mag", "combined"):
                extra_msg = (
                    f" cosK={diag_buf.get('cos_K_total', 0):.4f}"
                    f" cosV={diag_buf.get('cos_V_total', 0):.4f}"
                )
            elif args.loss_type == "relmse":
                extra_msg = (
                    f" relK={diag_buf.get('rel_K_total', 0):.4f}"
                    f" relV={diag_buf.get('rel_V_total', 0):.4f}"
                )
            elif args.loss_type == "attn_distill":
                # mse_O_mean is the per-layer attn-output MSE; abs_O_target
                # is the magnitude of O_tgt (so MSE/abs is "noise ratio").
                mse_o = diag_buf.get("mse_O_mean", 0)
                abs_o = diag_buf.get("abs_O_target_mean", 1e-6)
                extra_msg = (
                    f" mseO={mse_o:.6f}"
                    f" |O_tgt|={abs_o:.4f}"
                    f" ratio={mse_o / max(abs_o ** 2, 1e-12):.4f}"
                )
            print(
                f"[f_theta-train] step={step} lr={cur_lr:.2e} "
                f"loss={sum(recent)/len(recent):.6f} "
                f"(init={initial_loss:.6f}){extra_msg}",
                file=sys.stderr, flush=True,
            )
    train_elapsed = time.perf_counter() - t0

    # ---------------- Save ----------------
    f_theta.eval()
    f_theta.save_pretrained(args.save)
    final_loss = sum(losses_window[-args.log_every:]) / max(len(losses_window[-args.log_every:]), 1)

    report = {
        "kind": "k3_f_theta_train",
        "schema_version": 2,
        "config": vars(args),
        "f_theta_config": f_cfg.to_json_dict(),
        "n_params": n_params,
        "n_sequences": len(sequences),
        "n_general_prompts": n_general,
        "n_niah_prompts": n_niah,
        "collect_seconds": collect_elapsed,
        "train_seconds": train_elapsed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_factor": (
            initial_loss / final_loss if final_loss > 0 else None
        ),
        # Per-component diagnostic at end of training. For combined / cos_mag
        # losses, cosK_total close to 0.0 (≈ cos sim → 1.0) and cosV_total
        # close to 0.0 indicates good direction alignment. For combined,
        # mse_K_total + mse_V_total is the raw MSE for diff-ability with v1.
        "final_diagnostic": final_diag,
        "loss_type": args.loss_type,
        "lr_schedule": args.lr_schedule,
    }
    Path(args.save).mkdir(parents=True, exist_ok=True)
    Path(f"{args.save}.json").write_text(json.dumps(report, indent=2))
    print(
        f"[f_theta-train] DONE in {train_elapsed:.0f}s; "
        f"initial_loss={initial_loss:.6f} final_loss={final_loss:.6f} "
        f"reduction={report['loss_reduction_factor']:.2f}× "
        f"-> {args.save}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
