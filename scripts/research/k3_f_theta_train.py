"""K3 Block C — Train ``f_θ`` K/V projection: drafter K/V → verifier K/V.

Pipeline (CUDA, vast.ai H200/H100):

  1. Load Gemma 4 26B-A4B verifier (transformers, bf16, sdpa)
  2. Load DFlash drafter (PR #93's DFlashDrafter.from_pretrained,
     using models/dflash-kakeya-baseline)
  3. For each training sequence in the corpus:
     a. Run verifier forward; record K/V at every layer at every position
        (extracted via attention forward hooks on each layer's k_proj
        / v_proj — pre-norm pre-RoPE, matching what the cross-model
        DLMRestoredVerifier needs to inject)
     b. Run drafter forward via capture_proposer_kv; KVCapture has
        K/V at every drafter layer at every position (pre-norm pre-RoPE)
     c. f_θ targets: f_θ(drafter_kv) ≈ verifier_kv
  4. Train f_θ with MSE loss across layers + positions, AdamW

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


def _f_theta_loss(
    f_theta: FThetaProjection,
    seq: CapturedSequence,
    *,
    sample_positions: int = 256,
    seed: Optional[int] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute MSE loss for one sequence (subsampled positions).

    Sampling positions reduces memory + adds stochastic regularisation.
    All ``sample_positions`` positions are used for both K and V.

    ``normalize=True`` (v3 ``relmse`` mode): each (layer, K/V)
    component's MSE is divided by that component's target mean-square
    (``|O_tgt|²``). Gemma 4's K/V magnitudes differ widely across layers
    (sliding layers head_dim 256 vs full-attention layers head_dim 512),
    so a plain MSE lets the few high-magnitude components dominate the
    gradient and starves the rest. Relative MSE balances every component
    to ~unit scale, which the per-component diagnostics
    (:func:`_compute_diagnostics`) make visible.
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

    # Drafter K/V at sampled positions → list of [1, T_sub, num_kv_heads_d, head_dim_d]
    d_k_sub = seq.drafter_k.index_select(1, idx).unsqueeze(0)  # [1, L_d, T_sub, kv_dim]
    d_v_sub = seq.drafter_v.index_select(1, idx).unsqueeze(0)
    cfg = f_theta.config
    d_k_list = []
    d_v_list = []
    for li in range(cfg.drafter_num_layers):
        k_per = d_k_sub[:, li]  # [1, T_sub, kv_dim]
        v_per = d_v_sub[:, li]
        k_per = k_per.view(
            1, k_per.size(1), cfg.drafter_num_kv_heads, cfg.drafter_head_dim,
        )
        v_per = v_per.view(
            1, v_per.size(1), cfg.drafter_num_kv_heads, cfg.drafter_head_dim,
        )
        d_k_list.append(k_per)
        d_v_list.append(v_per)

    # Per-layer predictions: list of [1, T_sub, kv_heads_i, head_dim]
    pred_k, pred_v = f_theta.forward_kv_pack(d_k_list, d_v_list)

    # Per-layer targets + MSE (layers can have heterogeneous kv_dim).
    layer_kv_heads = cfg.layer_kv_heads
    layer_head_dims = cfg.layer_head_dims
    idx_pos = idx.to(seq.verifier_k[0].device)
    loss = pred_k[0].new_zeros(())
    n_layers = cfg.verifier_num_layers
    for li in range(n_layers):
        v_k_sub = seq.verifier_k[li].index_select(0, idx_pos)  # [T_sub, kv_dim_i]
        v_v_sub = seq.verifier_v[li].index_select(0, idx_pos)
        tgt_k = v_k_sub.view(
            1, v_k_sub.size(0), layer_kv_heads[li], layer_head_dims[li],
        ).float()
        tgt_v = v_v_sub.view(
            1, v_v_sub.size(0), layer_kv_heads[li], layer_head_dims[li],
        ).float()
        mse_k = F.mse_loss(pred_k[li].float(), tgt_k)
        mse_v = F.mse_loss(pred_v[li].float(), tgt_v)
        if normalize:
            mse_k = mse_k / (tgt_k.pow(2).mean() + 1e-6)
            mse_v = mse_v / (tgt_v.pow(2).mean() + 1e-6)
        loss = loss + mse_k + mse_v
    return loss / (2.0 * n_layers)


def _compute_diagnostics(
    f_theta: FThetaProjection,
    sequences: List[CapturedSequence],
    *,
    sample_positions: int = 256,
    max_seqs: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-component fidelity diagnostics for the trained f_θ.

    For each verifier layer and each of K / V, reports:
      * ``mseO``         — mean squared error of the f_θ output vs target
      * ``tgt_sq_mean``  — ``|O_tgt|²`` (target mean-square)
      * ``tgt_abs_mean`` — ``|O_tgt|`` (target mean-abs)
      * ``rel_mse``      — ``mseO / |O_tgt|²`` (unit-free relative error;
                            ~1.0 means the prediction is no better than
                            predicting zero, ≪1.0 means good fidelity)

    Aggregates split by layer type (sliding head_dim vs full head_dim)
    so the heterogeneous-magnitude structure is visible. This is the
    "mseO + |O_tgt| 分量诊断" the K3 recall investigation needs.
    """
    cfg = f_theta.config
    n = cfg.verifier_num_layers
    lkh, lhd = cfg.layer_kv_heads, cfg.layer_head_dims
    k_se = [0.0] * n; k_tsq = [0.0] * n; k_tabs = [0.0] * n
    v_se = [0.0] * n; v_tsq = [0.0] * n; v_tabs = [0.0] * n
    cnt = [0] * n
    g = torch.Generator(device="cpu").manual_seed(seed)
    chosen = sequences[:max_seqs]
    f_theta.eval()
    with torch.no_grad():
        for seq in chosen:
            T = seq.seq_len
            if T <= sample_positions:
                idx = torch.arange(T)
            else:
                idx = torch.randperm(T, generator=g)[:sample_positions]
            idx = idx.to(seq.drafter_k.device)
            d_k_sub = seq.drafter_k.index_select(1, idx).unsqueeze(0)
            d_v_sub = seq.drafter_v.index_select(1, idx).unsqueeze(0)
            d_k_list, d_v_list = [], []
            for li in range(cfg.drafter_num_layers):
                d_k_list.append(d_k_sub[:, li].view(
                    1, -1, cfg.drafter_num_kv_heads, cfg.drafter_head_dim))
                d_v_list.append(d_v_sub[:, li].view(
                    1, -1, cfg.drafter_num_kv_heads, cfg.drafter_head_dim))
            pred_k, pred_v = f_theta.forward_kv_pack(d_k_list, d_v_list)
            idx_pos = idx.to(seq.verifier_k[0].device)
            for li in range(n):
                tk = seq.verifier_k[li].index_select(0, idx_pos).view(
                    1, -1, lkh[li], lhd[li]).float()
                tv = seq.verifier_v[li].index_select(0, idx_pos).view(
                    1, -1, lkh[li], lhd[li]).float()
                pk, pv = pred_k[li].float(), pred_v[li].float()
                k_se[li] += float(((pk - tk) ** 2).sum())
                v_se[li] += float(((pv - tv) ** 2).sum())
                k_tsq[li] += float((tk ** 2).sum())
                v_tsq[li] += float((tv ** 2).sum())
                k_tabs[li] += float(tk.abs().sum())
                v_tabs[li] += float(tv.abs().sum())
                cnt[li] += tk.numel()

    per_layer = []
    for li in range(n):
        c = max(cnt[li], 1)
        k_mse, v_mse = k_se[li] / c, v_se[li] / c
        k_tsqm, v_tsqm = k_tsq[li] / c, v_tsq[li] / c
        per_layer.append({
            "layer": li,
            "kv_heads": lkh[li],
            "head_dim": lhd[li],
            "k_mseO": k_mse,
            "k_tgt_sq_mean": k_tsqm,
            "k_tgt_abs_mean": k_tabs[li] / c,
            "k_rel_mse": k_mse / (k_tsqm + 1e-8),
            "v_mseO": v_mse,
            "v_tgt_sq_mean": v_tsqm,
            "v_tgt_abs_mean": v_tabs[li] / c,
            "v_rel_mse": v_mse / (v_tsqm + 1e-8),
        })

    def _mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    sliding = [p for p in per_layer if p["head_dim"] == min(lhd)]
    full = [p for p in per_layer if p["head_dim"] == max(lhd)] if len(set(lhd)) > 1 else []
    aggregate = {
        "n_sequences_used": len(chosen),
        "sample_positions": sample_positions,
        "k_mseO_mean": _mean(p["k_mseO"] for p in per_layer),
        "v_mseO_mean": _mean(p["v_mseO"] for p in per_layer),
        "k_rel_mse_mean": _mean(p["k_rel_mse"] for p in per_layer),
        "v_rel_mse_mean": _mean(p["v_rel_mse"] for p in per_layer),
        "overall_rel_mse_mean": _mean(
            [p["k_rel_mse"] for p in per_layer] + [p["v_rel_mse"] for p in per_layer]
        ),
        "sliding_rel_mse_mean": _mean(
            [p["k_rel_mse"] for p in sliding] + [p["v_rel_mse"] for p in sliding]
        ),
        "full_rel_mse_mean": _mean(
            [p["k_rel_mse"] for p in full] + [p["v_rel_mse"] for p in full]
        ) if full else None,
    }
    return {"per_layer": per_layer, "aggregate": aggregate}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="models/dflash-kakeya-baseline")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=64,
                    help="Sequences in the training corpus")
    ap.add_argument("--gen-len", type=int, default=128,
                    help="Tokens generated per prompt during data collection")
    ap.add_argument("--sample-positions", type=int, default=256,
                    help="Random positions sampled per training step (memory reduction)")
    ap.add_argument("--save", default="results/research/f_theta_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--loss-mode", choices=["mse", "relmse"], default="mse",
        help="'mse' = plain MSE (v1); 'relmse' = per-component "
             "magnitude-normalized MSE (v3), balancing Gemma 4's "
             "heterogeneous-magnitude K/V layers.",
    )
    args = ap.parse_args()
    normalize = args.loss_mode == "relmse"

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

    # ---------------- Data collection ----------------
    print(f"[f_theta-train] collecting training corpus ({args.n_prompts} prompts) ...",
          file=sys.stderr, flush=True)
    sequences: List[CapturedSequence] = []
    t0 = time.perf_counter()
    eos_ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for pi in range(min(args.n_prompts, len(PROMPTS))):
        prompt = PROMPTS[pi]
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        # Greedy AR extension to gen_len for richer K/V coverage
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
        if (pi + 1) % 10 == 0 or pi == args.n_prompts - 1:
            print(
                f"[f_theta-train]   collected {pi + 1}/{args.n_prompts}, "
                f"latest seq_len={seq.seq_len}",
                file=sys.stderr,
            )
    collect_elapsed = time.perf_counter() - t0
    print(f"[f_theta-train] data collection done in {collect_elapsed:.0f}s",
          file=sys.stderr)

    # ---------------- Training ----------------
    optimizer = torch.optim.AdamW(
        f_theta.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    losses_window: List[float] = []
    initial_loss: Optional[float] = None
    f_theta.train()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        seq = random.choice(sequences)
        loss = _f_theta_loss(
            f_theta, seq, sample_positions=args.sample_positions,
            normalize=normalize,
        )
        if initial_loss is None:
            initial_loss = float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f_theta.parameters(), 1.0)
        optimizer.step()
        losses_window.append(float(loss.item()))
        if step % args.log_every == 0:
            recent = losses_window[-args.log_every:]
            print(
                f"[f_theta-train] step={step} loss={sum(recent)/len(recent):.6f} "
                f"(init={initial_loss:.6f})",
                file=sys.stderr, flush=True,
            )
    train_elapsed = time.perf_counter() - t0

    # ---------------- Save ----------------
    f_theta.eval()
    f_theta.save_pretrained(args.save)
    final_loss = sum(losses_window[-args.log_every:]) / max(len(losses_window[-args.log_every:]), 1)

    # ---------------- Diagnostics (mseO + |O_tgt| per component) -------
    print("[f_theta-train] computing per-component diagnostics "
          "(mseO + |O_tgt|) ...", file=sys.stderr, flush=True)
    diagnostics = _compute_diagnostics(
        f_theta, sequences,
        sample_positions=args.sample_positions, max_seqs=8, seed=args.seed,
    )
    agg = diagnostics["aggregate"]
    print(
        f"[f_theta-train] diag: overall_rel_mse={agg['overall_rel_mse_mean']:.4f} "
        f"(K={agg['k_rel_mse_mean']:.4f} V={agg['v_rel_mse_mean']:.4f}); "
        f"sliding_rel_mse={agg['sliding_rel_mse_mean']:.4f} "
        f"full_rel_mse={agg['full_rel_mse_mean']}",
        file=sys.stderr,
    )

    report = {
        "kind": "k3_f_theta_train",
        "config": vars(args),
        "loss_mode": args.loss_mode,
        "f_theta_config": f_cfg.to_json_dict(),
        "n_params": n_params,
        "n_sequences": len(sequences),
        "collect_seconds": collect_elapsed,
        "train_seconds": train_elapsed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_factor": (
            initial_loss / final_loss if final_loss > 0 else None
        ),
        "diagnostics": diagnostics,
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
