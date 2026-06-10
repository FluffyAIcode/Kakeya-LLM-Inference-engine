"""K3 Block C — Train ``f_θ`` K/V projection: drafter K/V → verifier K/V.

v2 (2026-06-10) — fixes recall=0 from f_θ v1
============================================

PR #103 v1 evidence: identity-restore recall = 1.0 (machinery correct);
f_θ-projected recall = 0.0 (training inadequate). Root causes:

  (a) **Wrong loss objective**: pure MSE on raw K/V. Final MSE 3.70
      ≈ 2σ noise per element. Attention is exp(QK^T); 2σ noise on K
      destroys softmax peakedness → lexical content lost at evicted
      positions. Solution: cosine + magnitude per-vector loss
      (direction-preserving, scale-aware) replaces pure MSE. Cosine
      preserves attention scores; magnitude preserves softmax scale.

  (b) **Tiny corpus, no NIAH structure**: 62 prompts × ~600 tokens
      ≈ 37k unique tokens, zero needle-in-a-haystack patterns. The
      eval is 100% NIAH; training never saw retrieval structure.
      Solution: synthetic NIAH-style training prompts (haystack +
      needle line) generated alongside the existing corpus, default
      50% NIAH / 50% PROMPTS.

  (c) **Trivial training duration**: 4000 steps × ~15ms ≈ 59s. The
      LR=1e-3 AdamW had barely warmed. Solution: default 20000 steps
      with cosine LR schedule (warmup → peak → cosine decay), 5×
      more training compute.

  (d) **No LR schedule**: constant lr=1e-3, never anneals. Solution:
      cosine schedule with linear warmup.

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
from typing import Any, Dict, List, Optional, Tuple

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
class CapturedSequence:
    """Paired drafter / verifier K/V over one training sequence.

    All tensors are kept on the same device as the models that
    produced them (typically CUDA). Memory cost per sequence:

      drafter_k:  num_drafter_layers × T × drafter_kv_dim × 2 (bytes/bf16)
      drafter_v:  same
      verifier_k: num_verifier_layers × T × verifier_kv_dim × 2
      verifier_v: same

    For T=512, Gemma 4 26B-A4B + DFlash 0.4B at bf16:
      drafter K+V: 5 × 512 × 256 × 2 × 2 = ~2.5 MB
      verifier K+V: 30 × 512 × 2048 × 2 × 2 = ~125 MB
      total per sequence: ~128 MB
    """
    seq_len: int
    drafter_k: torch.Tensor          # [num_d_layers, T, drafter_kv_dim]
    drafter_v: torch.Tensor          # [num_d_layers, T, drafter_kv_dim]
    # Verifier K/V are per-layer lists because Gemma 4 uses heterogeneous
    # KV-head counts across layers (8 on sliding layers, 4 on full layers).
    verifier_k: List[torch.Tensor]   # num_v_layers × [T, kv_dim_i]
    verifier_v: List[torch.Tensor]   # num_v_layers × [T, kv_dim_i]


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


def _collect_sequence(
    verifier_model: torch.nn.Module,
    drafter: DFlashDrafter,
    input_ids: torch.Tensor,
) -> CapturedSequence:
    """Capture paired drafter + verifier K/V for one input sequence."""
    # Verifier — k_proj / v_proj forward hooks
    v_k, v_v = _capture_verifier_kv(verifier_model, input_ids)

    # Drafter — uses verifier embed_tokens (DFlash shares verifier's),
    # runs drafter layers without aux conditioning, captures K/V via
    # forward hooks on k_proj/v_proj. See _capture_drafter_kv docstring
    # in cross_model_dlm_verifier for the architectural choice.
    capture = _capture_drafter_kv(
        verifier_model=verifier_model,
        drafter=drafter,
        input_ids=input_ids,
    )
    # capture.keys[i] shape: [B, T, num_d_kv_heads, head_dim]
    # Flatten last two dims and stack across layers.
    k_flat = [k.flatten(-2, -1) for k in capture.keys]
    v_flat = [v.flatten(-2, -1) for v in capture.values]
    d_k = torch.stack(k_flat, dim=0)[:, 0]  # [L_d, T, drafter_kv_dim]
    d_v = torch.stack(v_flat, dim=0)[:, 0]

    return CapturedSequence(
        seq_len=int(input_ids.size(1)),
        drafter_k=d_k.detach(),
        drafter_v=d_v.detach(),
        verifier_k=[t.detach() for t in v_k],
        verifier_v=[t.detach() for t in v_v],
    )


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
    loss_type: str = "combined",
    diag_buf: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Compute the configured loss for one sequence (subsampled positions).

    Parameters
    ----------
    loss_type : str
        ``"mse"``       — v1 MSE on raw K and V (kept for reproducibility).
        ``"cos_mag"``   — v2 cosine + magnitude on K and V.
        ``"combined"``  — v2 default. Cosine + magnitude with a small
                          MSE weight (0.1) for stability when norms are
                          near zero.
    diag_buf : dict
        Optional dict to receive per-component aggregates (cos_K_mean,
        cos_V_mean, mag_K_mean, mag_V_mean, mse_mean) for logging.
    """
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
                f"(want mse | cos_mag | combined)"
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
    ap.add_argument("--rank", type=int, default=256)
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
    ap.add_argument("--sample-positions", type=int, default=256,
                    help="Random positions sampled per training step (memory reduction)")
    ap.add_argument(
        "--loss-type", default="combined",
        choices=["mse", "cos_mag", "combined"],
        help="Training loss (v2 default combined; v1 used mse)",
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[f_theta-train] loading verifier {args.verifier_id}",
          file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="sdpa",
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
    print(f"[f_theta-train] collecting K/V from {len(corpus_prompts)} prompts ...",
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

        seq = _collect_sequence(verifier, drafter, full_ids)
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
    print(
        f"[f_theta-train] training: loss_type={args.loss_type} "
        f"schedule={args.lr_schedule} (warmup={args.warmup_steps}) "
        f"steps={args.steps} peak_lr={args.lr}",
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
            cos_msg = ""
            if args.loss_type in ("cos_mag", "combined"):
                cos_msg = (
                    f" cosK={diag_buf.get('cos_K_total', 0):.4f}"
                    f" cosV={diag_buf.get('cos_V_total', 0):.4f}"
                )
            print(
                f"[f_theta-train] step={step} lr={cur_lr:.2e} "
                f"loss={sum(recent)/len(recent):.6f} "
                f"(init={initial_loss:.6f}){cos_msg}",
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
