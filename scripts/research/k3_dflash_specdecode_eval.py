"""K3 Stage 2 — native DFlash speculative-decoding acceptance eval (CUDA).

Drives the engine's native DFlash drafter (`inference_engine/v04/
dflash_drafter.py`) against the real Gemma-4 26B-A4B verifier and measures
the speculative-decoding **acceptance length / acceptance rate** — the
metric that determines DFlash speedup (reference: ~7.7 length / ~44 % on
HumanEval, vLLM PR #41703).

Self-speculative loop (no KV cache; measures acceptance correctness, not
wall-clock):

  1. verifier forward over `committed`  → aux hidden at the last position
     (layers `aux_layer_ids` = target_layer_ids+1) + next-token logits.
  2. DFlashProposer.propose_block(committed, L, steps) → draft block.
  3. verifier forward over `committed + draft` → greedy-accept the longest
     prefix where the verifier's argmax matches the draft.
  4. commit accepted (+1 bonus/correction token), repeat.

Reports per-prompt and aggregate acceptance length/rate, and confirms the
spec output equals greedy AR (lossless). Requires transformers >= 5
(gemma4) and HF_TOKEN.

Usage:
  HF_TOKEN=hf_xxx PYTHONPATH=.:sdks/python python scripts/research/k3_dflash_specdecode_eval.py \
      --max-new-tokens 48 --block-size 16 --num-steps 8 --n-prompts 4 \
      --output results/research/k3_dflash_specdecode_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import torch

from inference_engine.v04.dflash_drafter import (
    AuxHiddenProvider,
    DFlashDrafter,
    DFlashProposer,
)


PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a haiku about speculative decoding.",
]

# HumanEval-style code-generation prompts — the regime the z-lab DFlash
# reference (~0.447 / 7.7) is measured on. DFlash drafts code/structured
# output far better than open-ended short Q&A, so this set characterizes
# acceptance on the reference's distribution.
CODE_PROMPTS = [
    "Complete this Python function:\n\ndef has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\"Return True if any two numbers are closer than threshold.\"\"\"\n",
    "Complete this Python function:\n\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s reads the same forwards and backwards, ignoring case and non-alphanumeric chars.\"\"\"\n",
    "Complete this Python function:\n\ndef merge_sort(arr: list[int]) -> list[int]:\n    \"\"\"Return a new list with the elements of arr sorted ascending using merge sort.\"\"\"\n",
    "Complete this Python function:\n\ndef gcd(a: int, b: int) -> int:\n    \"\"\"Return the greatest common divisor of a and b using the Euclidean algorithm.\"\"\"\n",
    "Complete this Python function:\n\ndef flatten(nested: list) -> list:\n    \"\"\"Flatten an arbitrarily nested list of integers into a single flat list.\"\"\"\n",
    "Complete this Python function:\n\ndef count_words(text: str) -> dict[str, int]:\n    \"\"\"Return a dict mapping each lowercased word in text to its frequency.\"\"\"\n",
]

# Disjoint from the alignment trainer's prompt corpus — for honest held-out
# acceptance after alignment training (no topic/phrasing near-duplicates).
HELD_OUT_PROMPTS = [
    "Write a Python function that counts vowels in a string.",
    "Name two differences between a list and a tuple in Python.",
    "What is the boiling point of water at sea level in Celsius?",
    "Write a short rhyming couplet about a robot learning to paint.",
    "Explain what a database index is and when to use one.",
    "Who painted the Mona Lisa?",
    "Write a function that returns whether a year is a leap year.",
    "Give two reasons code review is valuable.",
]


class VerifierAuxProvider(AuxHiddenProvider):
    """Wraps the Gemma-4 verifier: runs a forward over `committed` and serves
    the aux-layer hidden states at **all** positions to the drafter (DFlash
    turns them into the draft layers' prewritten context K/V)."""

    def __init__(self, model, aux_layer_ids, device):
        self.model = model
        self.aux_layer_ids = aux_layer_ids
        self.device = device
        self.forward_calls = 0

    @torch.no_grad()
    def aux_hidden_context(self, committed_token_ids):
        inp = torch.tensor([committed_token_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=inp, use_cache=False, output_hidden_states=True)
        self.forward_calls += 1
        hs = out.hidden_states  # tuple len = num_layers+1 (0 = embeddings)
        aux = [hs[a].float() for a in self.aux_layer_ids]  # [1, C, hidden] each
        # Bonus = verifier greedy next token t_C (guaranteed-correct first token).
        bonus = int(torch.argmax(out.logits[0, -1]).item())
        return aux, bonus


def _build_embed_lm_head(model, hidden_size, softcap, embed_scale=None):
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    # Reference DFlashQwen3Model.forward embeds with a PLAIN lookup
    # (``self.embed_tokens(input_ids)``) — no Gemma ``×sqrt(hidden)``
    # normalizer (that scale is applied inside the Gemma model body, not in
    # the shared embed the Qwen3 drafter consumes). Default to no scale to
    # match the reference; ``embed_scale`` overrides for A/B testing.
    scale = 1.0 if embed_scale is None else float(embed_scale)

    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        return emb(ids).float() * scale

    def lm_head_fn(h: torch.Tensor) -> torch.Tensor:
        logits = head(h.to(head.weight.dtype)).float()
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    return embed_fn, lm_head_fn


@torch.no_grad()
def verify_block(model, committed: List[int], draft: List[int], device):
    """Return the list of greedily-accepted draft tokens + the verifier's
    correction/bonus token, via one forward over committed+draft."""
    seq = committed + draft
    inp = torch.tensor([seq], dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=False)
    logits = out.logits[0].float()  # [C+L, V]
    C = len(committed)
    accepted = 0
    for i in range(len(draft)):
        pred = int(torch.argmax(logits[C - 1 + i]).item())
        if pred == draft[i]:
            accepted += 1
        else:
            break
    correction = int(torch.argmax(logits[C - 1 + accepted]).item())
    return accepted, correction


@torch.no_grad()
def greedy_ar(model, prompt_ids: List[int], max_new_tokens: int, device, eos_ids):
    cur = list(prompt_ids)
    forwards = 0
    for _ in range(max_new_tokens):
        inp = torch.tensor([cur], dtype=torch.long, device=device)
        out = model(input_ids=inp, use_cache=False)
        forwards += 1
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        cur.append(nxt)
        if nxt in eos_ids:
            break
    return cur[len(prompt_ids):], forwards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--drafter-state", default=None,
                    help="optional .pt state_dict to load over the drafter "
                         "(e.g. an alignment-trained checkpoint).")
    ap.add_argument("--embed-scale", type=float, default=None,
                    help="Scale applied to the shared embedding fed to the "
                         "drafter. Default None = 1.0 (reference, no Gemma "
                         "sqrt(hidden) normalizer). Pass e.g. 53.06 to A/B the "
                         "old (incorrect) sqrt(2816) scaling.")
    ap.add_argument("--held-out", action="store_true",
                    help="evaluate on HELD_OUT_PROMPTS (disjoint from the "
                         "alignment trainer's prompts) for honest generalization.")
    ap.add_argument("--prompt-set", choices=["default", "held-out", "code"],
                    default=None,
                    help="Which prompt set to use. 'code' = HumanEval-style "
                         "(the z-lab reference regime). Overrides --held-out.")
    ap.add_argument("--humaneval-jsonl", default=None,
                    help="Path to the canonical HumanEval .jsonl (each line a "
                         "problem with a 'prompt' field). Uses the first "
                         "--n-prompts problems' prompts. This is the exact "
                         "z-lab reference regime (~0.447 / 7.7).")
    ap.add_argument("--raw-completion", action="store_true",
                    help="Feed the raw prompt tokens (no chat template) — the "
                         "native HumanEval code-completion setup.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.humaneval_jsonl:
        with open(args.humaneval_jsonl) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        prompts = [r["prompt"] for r in rows[: args.n_prompts]]
    elif args.prompt_set == "code":
        prompts = CODE_PROMPTS
    elif args.prompt_set == "held-out" or args.held_out:
        prompts = HELD_OUT_PROMPTS
    else:
        prompts = PROMPTS

    device = torch.device("cuda")
    dtype = torch.bfloat16
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[k3-sd] loading verifier {args.verifier_id}", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="sdpa", device_map="auto",
    ).eval()
    print(f"[k3-sd] loading drafter {args.drafter_id}", file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype).to(device).eval()
    if args.drafter_state:
        sd = torch.load(args.drafter_state, map_location=device)
        drafter.load_state_dict({k: v.to(dtype) for k, v in sd.items()})
        drafter.eval()
        print(f"[k3-sd] loaded aligned drafter state from {args.drafter_state}",
              file=sys.stderr)
    cfg = drafter.cfg
    hidden = cfg.hidden_size
    softcap = cfg.final_logit_softcapping
    embed_fn, lm_head_fn = _build_embed_lm_head(
        verifier, hidden, softcap, embed_scale=args.embed_scale)
    provider = VerifierAuxProvider(verifier, cfg.aux_layer_ids, device)
    proposer = DFlashProposer(drafter, provider, embed_fn, lm_head_fn)

    eos_ids = set(
        x for x in [tok.eos_token_id, getattr(tok, "eot_token_id", None)] if x is not None
    )

    per_prompt = []
    tot_accepted = tot_drafted = tot_blocks = 0
    tot_spec_forwards = tot_ar_forwards = 0
    lossless = True

    for pi in range(min(args.n_prompts, len(prompts))):
        prompt = prompts[pi]
        if args.raw_completion:
            # Native HumanEval code-completion: feed the raw prompt tokens
            # (no chat template); the verifier continues the function body.
            ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        else:
            msgs = [{"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            )
            # transformers 5.x may return a Tensor or a BatchEncoding/dict.
            if hasattr(enc, "keys"):
                enc = enc["input_ids"]
            ids = enc[0].tolist()
        committed = list(ids)
        generated: List[int] = []
        blk_accepts = []
        provider.forward_calls = 0
        t0 = time.perf_counter()
        while len(generated) < args.max_new_tokens:
            L = min(args.block_size, args.max_new_tokens - len(generated))
            # aux hidden over committed + the verifier's greedy next token (bonus).
            aux_ctx, bonus = provider.aux_hidden_context(committed)
            drafts = drafter.draft_block(
                aux_ctx, bonus, embed_fn, lm_head_fn, block_size=L,
            )
            # Candidate = [bonus (always correct), drafts...]; verify greedily.
            candidate = [bonus] + drafts
            accepted, _ = verify_block(verifier, committed, candidate, device)
            tot_spec_forwards += 2  # 1 aux/prefill forward + 1 verify forward
            accepted = max(accepted, 1)  # bonus is guaranteed-correct
            committed += candidate[:accepted]
            generated += candidate[:accepted]
            draft_accepted = accepted - 1  # exclude the always-correct bonus
            blk_accepts.append(draft_accepted)
            tot_accepted += draft_accepted
            tot_drafted += L
            tot_blocks += 1
            if any(t in eos_ids for t in candidate[:accepted]):
                break
        spec_time = time.perf_counter() - t0
        spec_out = generated[: args.max_new_tokens]

        # AR reference (lossless check + forward count)
        ar_out, ar_forwards = greedy_ar(
            verifier, ids, len(spec_out), device, eos_ids,
        )
        tot_ar_forwards += ar_forwards
        match = spec_out[: len(ar_out)] == ar_out[: len(spec_out)]
        lossless = lossless and match
        mean_acc = sum(blk_accepts) / max(len(blk_accepts), 1)
        per_prompt.append({
            "prompt": prompt,
            "blocks": len(blk_accepts),
            "block_accepts": blk_accepts,
            "mean_accepted_per_block": mean_acc,
            "tokens_generated": len(spec_out),
            "verifier_forwards_spec": provider.forward_calls,
            "lossless_vs_ar": match,
            "decoded": tok.decode(spec_out, skip_special_tokens=True)[:200],
        })
        print(
            f"[k3-sd] prompt {pi}: blocks={len(blk_accepts)} "
            f"mean_accept={mean_acc:.2f} accepts={blk_accepts} lossless={match}",
            file=sys.stderr,
        )

    acc_rate = tot_accepted / max(tot_drafted, 1)
    # acceptance length = accepted + 1 bonus per block, the standard metric
    acc_length = (tot_accepted + tot_blocks) / max(tot_blocks, 1)
    report = {
        "schema_version": 1,
        "kind": "k3_dflash_specdecode_acceptance",
        "config": {
            "verifier_id": args.verifier_id,
            "drafter_id": args.drafter_id,
            "block_size": args.block_size,
            "num_steps": args.num_steps,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": min(args.n_prompts, len(prompts)),
            "aux_layer_ids": list(cfg.aux_layer_ids),
        },
        "aggregate": {
            "acceptance_rate": acc_rate,
            "acceptance_length": acc_length,
            "total_accepted": tot_accepted,
            "total_drafted": tot_drafted,
            "total_blocks": tot_blocks,
            "lossless_vs_ar": lossless,
            "reference_humaneval": {"acceptance_length": 7.7, "acceptance_rate": 0.447},
        },
        "per_prompt": per_prompt,
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_dflash_specdecode_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[k3-sd] AGGREGATE acceptance_rate={acc_rate:.3f} "
        f"acceptance_length={acc_length:.2f} lossless={lossless} "
        f"(ref ~0.447 / ~7.7)  -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
