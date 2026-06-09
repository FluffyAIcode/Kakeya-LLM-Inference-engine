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
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a haiku about speculative decoding.",
    "Describe how a hash map works in one paragraph.",
    "Give three tips for writing clear commit messages.",
    "What causes the seasons on Earth?",
    "Write a short limerick about a cat who loves GPUs.",
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
    cur = list(prompt_ids)
    for _ in range(gen_len):
        inp = torch.tensor([cur], dtype=torch.long, device=device)
        out = model(input_ids=inp, use_cache=False)
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        cur.append(nxt)
        if nxt in eos_ids:
            break
    return cur


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
    print(f"[align] loading drafter {args.drafter_id}", file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype).to(device)
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
    # Train the projection/norms in fp32 for stable optimisation.
    for p in trainable:
        p.data = p.data.float()
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
