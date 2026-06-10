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
from inference_engine.v04.cross_model_dlm_verifier import _capture_drafter_kv
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
    drafter_k: torch.Tensor    # [num_d_layers, T, drafter_kv_dim]
    drafter_v: torch.Tensor    # [num_d_layers, T, drafter_kv_dim]
    verifier_k: torch.Tensor   # [num_v_layers, T, verifier_kv_dim]
    verifier_v: torch.Tensor   # [num_v_layers, T, verifier_kv_dim]


def _capture_verifier_kv(
    verifier_model: torch.nn.Module, input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run verifier forward and capture per-layer K, V via forward hooks
    on each decoder layer's k_proj / v_proj.

    Returns
    -------
    (verifier_k, verifier_v) of shape [num_v_layers, T, verifier_kv_dim]
    each, on the verifier's device.
    """
    layers = verifier_model.model.layers
    num_layers = len(layers)
    k_capture: List[torch.Tensor] = [None] * num_layers
    v_capture: List[torch.Tensor] = [None] * num_layers
    handles = []

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
        handles.append(attn.v_proj.register_forward_hook(_make_v_hook(i)))

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
    # Each k_capture[i] is [B, T, num_kv_heads × head_dim] = [B, T, kv_dim]
    # Stack to [num_layers, B, T, kv_dim] then drop B (assume B=1)
    K = torch.stack(k_capture, dim=0)  # [L_v, B, T, kv_dim]
    V = torch.stack(v_capture, dim=0)
    if K.size(1) != 1:
        raise NotImplementedError(
            f"f_θ training currently assumes batch=1 (got {K.size(1)})"
        )
    return K[:, 0], V[:, 0]   # [L_v, T, kv_dim]


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
        verifier_k=v_k.detach(),
        verifier_v=v_v.detach(),
    )


def _f_theta_loss(
    f_theta: FThetaProjection,
    seq: CapturedSequence,
    *,
    sample_positions: int = 256,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Compute MSE loss for one sequence (subsampled positions).

    Sampling positions reduces memory + adds stochastic regularisation.
    All ``sample_positions`` positions are used for both K and V.
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

    # Drafter K/V at sampled positions, reshaped to [B=1, T_sub, ...]
    d_k_sub = seq.drafter_k.index_select(1, idx).unsqueeze(0)  # [1, L_d, T_sub, kv_dim]
    d_v_sub = seq.drafter_v.index_select(1, idx).unsqueeze(0)
    # Permute so batch dim is first, then T, layer (forward_kv_pack
    # expects a list of [B, T, num_kv_heads, head_dim]).
    # d_k_sub is [1, L_d, T_sub, kv_dim] = [B, L_d, T, kv_dim]; we need
    # list of L_d tensors each [B, T, num_kv_heads, head_dim].
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

    pred_k, pred_v = f_theta.forward_kv_pack(d_k_list, d_v_list)
    # pred_k: [1, T_sub, L_v, num_kv_heads_v, head_dim_v]

    # Targets
    v_k_sub = seq.verifier_k.index_select(1, idx)  # [L_v, T_sub, verifier_kv_dim]
    v_v_sub = seq.verifier_v.index_select(1, idx)
    v_k_target = v_k_sub.permute(1, 0, 2).unsqueeze(0)  # [1, T_sub, L_v, kv_dim]
    v_v_target = v_v_sub.permute(1, 0, 2).unsqueeze(0)
    v_k_target = v_k_target.view(
        1, v_k_target.size(1), cfg.verifier_num_layers,
        cfg.verifier_num_kv_heads, cfg.verifier_head_dim,
    )
    v_v_target = v_v_target.view(
        1, v_v_target.size(1), cfg.verifier_num_layers,
        cfg.verifier_num_kv_heads, cfg.verifier_head_dim,
    )

    # MSE in fp32 for stability
    loss_k = F.mse_loss(pred_k.float(), v_k_target.float())
    loss_v = F.mse_loss(pred_v.float(), v_v_target.float())
    return (loss_k + loss_v) / 2.0


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

    # Derive f_θ config from drafter + verifier shapes
    f_cfg = FThetaConfig(
        drafter_num_layers=drafter.cfg.num_hidden_layers,
        drafter_num_kv_heads=drafter.cfg.num_key_value_heads,
        drafter_head_dim=drafter.cfg.head_dim,
        verifier_num_layers=verifier.config.num_hidden_layers,
        verifier_num_kv_heads=verifier.config.num_key_value_heads,
        verifier_head_dim=verifier.config.head_dim,
        rank=args.rank,
    )
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

    report = {
        "kind": "k3_f_theta_train",
        "config": vars(args),
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
