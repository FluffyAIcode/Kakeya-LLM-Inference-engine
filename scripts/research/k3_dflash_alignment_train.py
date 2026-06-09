"""K3 alignment training — align the native DFlash drafter to the Gemma-4
verifier (treats the residual inference-fidelity gap as an f_θ-style
alignment task; ADR 0008 §11, docs/design/k3-f-theta-training-pipeline.md).

Rather than perfectly reconstructing vLLM's aux-hidden-tap semantics, we
freeze the verifier (and, by default, the drafter's Qwen3 backbone) and
train the projection ``fc`` + ``hidden_norm`` + ``norm`` so the drafter's
mask-position logits predict the verifier's greedy next tokens. The trained
drafter state is saved and re-evaluated by k3_dflash_specdecode_eval.py.

Efficiency: the verifier is causal, so ONE full-sequence forward yields the
aux hidden for every prefix; training is then drafter-only per step.

Usage:
  HF_TOKEN=hf_xxx PYTHONPATH=.:sdks/python python scripts/research/k3_dflash_alignment_train.py \
      --steps 600 --lr 1e-4 --block-size 16 --n-prompts 8 --gen-len 192 \
      --train-scope fc_norms --save results/research/dflash_aligned_fcnorms.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from inference_engine.v04.dflash_drafter import DFlashDrafter


PROMPTS = [
    # coding
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
    # math / reasoning
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
    # QA / factual
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
    # explanations
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
    # writing / misc
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


def _embed_lm_head(model, hidden_size, softcap):
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    scale = math.sqrt(hidden_size)

    def embed_fn(ids):
        return emb(ids).float() * scale

    def lm_head_fn(h):
        logits = head(h.to(head.weight.dtype)).float()
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    return embed_fn, lm_head_fn


@torch.no_grad()
def greedy_seq(model, prompt_ids, gen_len, device, eos_ids):
    # KV-cached greedy generation (fast) — data gen only, no grad.
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids=inp, max_new_tokens=gen_len, do_sample=False, use_cache=True,
    )
    return out[0].tolist()


@torch.no_grad()
def cache_aux(model, ids, aux_layer_ids, device):
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    # [1, T, hidden] per aux layer, kept on GPU in fp16 to save memory.
    return [hs[a].half() for a in aux_layer_ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--gen-len", type=int, default=192)
    ap.add_argument("--prompt-min-ctx", type=int, default=8)
    ap.add_argument("--train-scope", choices=["fc_norms", "full"], default="fc_norms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="results/research/dflash_aligned.pt")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[align] loading verifier {args.verifier_id}", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="sdpa", device_map="auto",
    ).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)
    print(f"[align] loading drafter {args.drafter_id} (fp32 for training)",
          file=sys.stderr, flush=True)
    # Train the (small, 0.43B) drafter entirely in fp32 so the forward is
    # uniformly fp32 (the verifier stays frozen bf16; its embed/lm_head/aux
    # outputs are upcast to fp32). Avoids mixed fp32-trainable / bf16-frozen
    # matmul dtype errors. Saved back to bf16 for inference.
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=torch.float32).to(device)
    cfg = drafter.cfg
    embed_fn, lm_head_fn = _embed_lm_head(verifier, cfg.hidden_size, cfg.final_logit_softcapping)
    eos_ids = {x for x in [tok.eos_token_id] if x is not None}

    # Trainable surface.
    for p in drafter.parameters():
        p.requires_grad_(False)
    if args.train_scope == "full":
        trainable = list(drafter.parameters())
        for p in trainable:
            p.requires_grad_(True)
    else:  # fc + hidden_norm + norm (the EAGLE-3 projection/norms)
        trainable = (
            list(drafter.fc.parameters())
            + list(drafter.hidden_norm.parameters())
            + list(drafter.norm.parameters())
        )
        for p in trainable:
            p.requires_grad_(True)
    # Drafter is already fp32 (loaded above) for stable optimisation.
    n_train = sum(p.numel() for p in trainable)
    print(f"[align] trainable params ({args.train_scope}): {n_train:,}", file=sys.stderr)
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    # Build greedy sequences + cache aux for every prefix (causal => one forward).
    L = args.block_size
    seqs, aux_cache, prompt_lens = [], [], []
    for i in range(min(args.n_prompts, len(PROMPTS))):
        msgs = [{"role": "user", "content": PROMPTS[i]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        pids = enc[0].tolist()
        seq = greedy_seq(verifier, pids, args.gen_len, device, eos_ids)
        if len(seq) < len(pids) + L + 2:
            continue
        seqs.append(seq)
        aux_cache.append(cache_aux(verifier, seq, cfg.aux_layer_ids, device))
        prompt_lens.append(len(pids))
        print(f"[align] seq {i}: len={len(seq)} (prompt {len(pids)})", file=sys.stderr)

    if not seqs:
        print("[align] no usable sequences", file=sys.stderr)
        return 1

    # Sampleable (seq_idx, C) windows.
    windows = []
    for si, seq in enumerate(seqs):
        for C in range(max(prompt_lens[si], args.prompt_min_ctx), len(seq) - L - 1):
            windows.append((si, C))
    print(f"[align] {len(windows)} training windows", file=sys.stderr)

    drafter.train()
    losses, matches = [], []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        si, C = random.choice(windows)
        seq = seqs[si]
        aux = [a[:, :C, :].float() for a in aux_cache[si]]  # [1, C, hidden]
        bonus = seq[C]
        targets = torch.tensor(seq[C + 1 : C + 1 + L], dtype=torch.long, device=device)
        logits = drafter.draft_logits(aux, bonus, embed_fn, lm_head_fn, block_size=L)
        loss = F.cross_entropy(logits[0].float(), targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            pred = torch.argmax(logits[0], dim=-1)
            matches.append(float((pred == targets).float().mean().item()))
        if step % args.log_every == 0:
            print(
                f"[align] step={step} loss={sum(losses[-args.log_every:])/args.log_every:.4f} "
                f"match={sum(matches[-args.log_every:])/args.log_every:.3f}",
                file=sys.stderr, flush=True,
            )

    elapsed = time.perf_counter() - t0
    # Save the (fp32-trained) params cast back to the model dtype.
    drafter.eval()
    state = {k: v.detach().to(dtype).cpu() for k, v in drafter.state_dict().items()}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.save)
    report = {
        "kind": "k3_dflash_alignment_train",
        "config": vars(args),
        "trainable_params": n_train,
        "n_windows": len(windows),
        "final_loss": sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1),
        "final_train_match": sum(matches[-args.log_every:]) / max(len(matches[-args.log_every:]), 1),
        "elapsed_s": elapsed,
    }
    Path(args.save).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(
        f"[align] DONE in {elapsed:.0f}s; final loss={report['final_loss']:.4f} "
        f"train_match={report['final_train_match']:.3f}; saved -> {args.save}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
