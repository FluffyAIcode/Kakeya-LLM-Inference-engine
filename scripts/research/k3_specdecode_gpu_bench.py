"""K3 speculative-decoding GPU bench for the *restored* verifier.

Re-examines the spec-decode path for the Kakeya inference engine and
measures, on the same NIAH prompts:

  * **AR-incremental** — standalone Gemma 4 26B AR with the model's own KV
    cache (the throughput target).
  * **restored-pertoken** — the restored verifier decoded one token at a
    time (the naive baseline; what k3_e2e_gpu_bench used).
  * **restored-specdecode** — DFlash drafts a block, the **restored**
    verifier verifies it in one pass, greedily accepting the longest
    matching prefix (block-amortized verifier forwards).

Reports per path: decode tok/s, verifier forward passes, and (for
spec-decode) acceptance length; plus NIAH recall (correctness). This
quantifies how much the DFlash block-acceptance amortizes the (currently
O(T) re-forward) restored verifier, and isolates the two levers to reach
AR-parity: drafter acceptance and an incremental restored forward.

Run (transformers-5.x venv, CUDA)::

    HF_HOME=/workspace/.hf_home PYTHONPATH=.:sdks/python \
      .venv-k3/bin/python scripts/research/k3_specdecode_gpu_bench.py \
        --haystack-lines 60 --n-samples 3 --max-new-tokens 48 \
        --block-size 16 --output results/research/k3_specdecode_gpu_bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


# --------------------------------------------------------------------------- #
# DFlash wiring helpers (mirrors scripts/research/k3_dflash_specdecode_eval.py)
# --------------------------------------------------------------------------- #
def _build_embed_lm_head(model, hidden_size, softcap):
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    scale = math.sqrt(hidden_size)

    # Reference DFlash embeds the drafter query with a plain (unscaled)
    # lookup — NO Gemma ×sqrt(hidden) normalizer. The earlier ×sqrt scaling
    # was a fidelity bug (crippled original-DFlash acceptance ~0.05 → 0.16
    # once removed). Keep at 1.0 to match the reference.
    scale = 1.0

    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        return emb(ids).float() * scale

    def lm_head_fn(h: torch.Tensor) -> torch.Tensor:
        logits = head(h.to(head.weight.dtype)).float()
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    return embed_fn, lm_head_fn


@torch.no_grad()
def ar_incremental(model, ids, gen_tokens, device) -> Tuple[List[int], float, int]:
    """Standalone AR with the model's own KV cache. Returns (tokens, decode_s, fwds)."""
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    nxt = int(out.logits[0, -1].argmax().item())
    gen: List[int] = []
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    fwds = 0
    for _ in range(gen_tokens):
        gen.append(nxt)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        fwds += 1
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    torch.cuda.synchronize(device)
    return gen, time.perf_counter() - t0, fwds


@torch.no_grad()
def restored_pertoken(adapter, prompt, gen_tokens, device) -> Tuple[List[int], float, int]:
    adapter.prefill(prompt)
    nxt = int(adapter.next_token_logits.argmax().item())
    gen: List[int] = []
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        gen.append(nxt)
        adapter.append_token(nxt)
        nxt = int(adapter.next_token_logits.argmax().item())
    torch.cuda.synchronize(device)
    # forward count: prefill (1) + gen append_token (each 1 restored.forward)
    return gen, time.perf_counter() - t0, gen_tokens


@torch.no_grad()
def restored_specdecode(
    adapter, drafter, provider, embed_fn, lm_head_fn,
    prompt, gen_tokens, block_size, device, eos_ids,
) -> Dict[str, Any]:
    """DFlash drafts a block; the **incremental** (Gap-A) restored verifier
    verifies the block in one O(L) incremental forward, greedily accepting
    the matching prefix. The restored verifier is the source of truth
    (output == greedy restored decode). Reports a per-component time
    breakdown (aux / draft / verify) to expose the bottleneck."""
    assert adapter._incremental, "restored_specdecode needs incremental=True (Gap-A)"
    adapter.prefill(prompt)              # builds the restored KV cache once
    generated: List[int] = []
    accepts: List[int] = []
    t_aux = t_draft = t_verify = 0.0
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    while len(generated) < gen_tokens:
        L = min(block_size, gen_tokens - len(generated))
        # DFlash drafts using the *clean* verifier aux hidden (EAGLE) + bonus.
        ta = time.perf_counter()
        aux_ctx, bonus = provider.aux_hidden_context(adapter._committed)
        torch.cuda.synchronize(device); t_aux += time.perf_counter() - ta
        td = time.perf_counter()
        drafts = drafter.draft_block(aux_ctx, bonus, embed_fn, lm_head_fn, block_size=L)
        torch.cuda.synchronize(device); t_draft += time.perf_counter() - td
        candidate = [bonus] + drafts[: L - 1] if L > 1 else [bonus]
        # Verify with the INCREMENTAL restored verifier (O(L)).
        tv = time.perf_counter()
        prev = adapter.next_token_logits
        block_logits = adapter.forward_block(candidate)  # [len(candidate), V]
        accepted = 0
        for i in range(len(candidate)):
            if int(prev.argmax().item()) == candidate[i]:
                accepted += 1
                prev = block_logits[i]
            else:
                break
        correction = int(prev.argmax().item())
        adapter.commit_or_truncate(forwarded=len(candidate), accepted=accepted)
        adapter.append_token(correction)   # commit correction; updates next_token_logits
        torch.cuda.synchronize(device); t_verify += time.perf_counter() - tv
        commit = candidate[:accepted] + [correction]
        generated += commit
        accepts.append(accepted)
        if any(t in eos_ids for t in commit):
            break
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    generated = generated[:gen_tokens]
    return {
        "tokens": generated,
        "decode_s": dt,
        "decode_tokens_per_s": round(len(generated) / dt, 3) if dt > 0 else None,
        "time_breakdown_s": {
            "aux_clean_forward": round(t_aux, 3),
            "drafter": round(t_draft, 3),
            "incremental_verify": round(t_verify, 3),
        },
        "blocks": len(accepts),
        "mean_accept_len": round(sum(accepts) / len(accepts), 2) if accepts else 0.0,
        "decode_tokens": len(generated),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--f-theta-dir", default="results/research/f_theta_v5_s5_sliding")
    ap.add_argument("--haystack-lines", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[sd] CUDA required.", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    dtype = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import (  # type: ignore
        ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
    )
    from inference_engine.v04 import (
        CrossModelRestoredSinkWindowVerifier, DFlashDrafter, FThetaProjection,
        make_niah_dataset,
    )
    from inference_engine.v04.cross_model_dlm_verifier import (
        CrossModelDLMRestoredVerifier, full_attention_layer_indices,
    )
    from inference_engine.v04.dflash_drafter import DFlashProposer  # noqa: F401  (kept for parity)

    class VerifierAuxProvider:
        def __init__(self, model, aux_layer_ids, device):
            self.model = model
            self.aux_layer_ids = aux_layer_ids
            self.device = device

        @torch.no_grad()
        def aux_hidden_context(self, committed_token_ids):
            inp = torch.tensor([committed_token_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=inp, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            aux = [hs[a].float() for a in self.aux_layer_ids]
            bonus = int(torch.argmax(out.logits[0, -1]).item())
            return aux, bonus

    print(f"[sd] loading verifier {args.verifier_id}", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="eager", device_map="auto",
    ).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)
    print(f"[sd] loading drafter {args.drafter_id}", file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype).to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    print(f"[sd] loading f_θ {args.f_theta_dir}", file=sys.stderr, flush=True)
    f_theta = FThetaProjection.from_pretrained(args.f_theta_dir, dtype=torch.float32, device=device)

    exact_layers = full_attention_layer_indices(verifier)
    restored = CrossModelDLMRestoredVerifier(
        verifier_model=verifier, drafter=drafter, f_theta=f_theta,
        sink_size=args.sink, window_size=args.window, exact_layer_indices=exact_layers,
    )
    adapter = CrossModelRestoredSinkWindowVerifier(
        restored, apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS, device="cuda",
        incremental=True,   # Gap-A: O(L)/block incremental verify
    )
    cfg = drafter.cfg
    embed_fn, lm_head_fn = _build_embed_lm_head(verifier, cfg.hidden_size, cfg.final_logit_softcapping)
    provider = VerifierAuxProvider(verifier, cfg.aux_layer_ids, device)
    eos_ids = set(x for x in [tok.eos_token_id] if x is not None)

    def encode_chat(text):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids.to(device)

    samples = make_niah_dataset(
        n_samples=args.n_samples, haystack_min_lines=args.haystack_lines,
        haystack_max_lines=args.haystack_lines, seed=args.seed)
    ids_list = [encode_chat(s.prompt_text) for s in samples]
    seqlens = [int(t.size(1)) for t in ids_list]
    print(f"[sd] prompt tokens min={min(seqlens)} max={max(seqlens)}", file=sys.stderr)

    def recall(tokens, ans):
        return ans in tok.decode(tokens, skip_special_tokens=True)

    ar_tps: List[float] = []
    pt_tps: List[float] = []
    sd_rows: List[Dict[str, Any]] = []
    ar_hits = pt_hits = sd_hits = 0
    for i, ids in enumerate(ids_list):
        ans = samples[i].answer_text
        prompt = ids[0].tolist()
        g_ar, t_ar, _ = ar_incremental(verifier, ids, args.max_new_tokens, device)
        ar_tps.append(len(g_ar) / t_ar)
        ar_hits += int(recall(g_ar, ans))
        g_pt, t_pt, _ = restored_pertoken(adapter, prompt, args.max_new_tokens, device)
        pt_tps.append(len(g_pt) / t_pt)
        pt_hits += int(recall(g_pt, ans))
        sd = restored_specdecode(
            adapter, drafter, provider, embed_fn, lm_head_fn,
            prompt, args.max_new_tokens, args.block_size, device, eos_ids)
        sd_rows.append(sd)
        sd_hits += int(recall(sd["tokens"], ans))
        tb = sd["time_breakdown_s"]
        print(f"[sd] sample {i}: AR={ar_tps[-1]:.2f} tok/s | restored-pertoken="
              f"{pt_tps[-1]:.2f} tok/s | restored-specdecode={sd['decode_tokens_per_s']} tok/s "
              f"(accept_len={sd['mean_accept_len']}, blocks={sd['blocks']}, "
              f"aux={tb['aux_clean_forward']}s draft={tb['drafter']}s verify={tb['incremental_verify']}s) "
              f"| recall ar/pt/sd="
              f"{recall(g_ar, ans)}/{recall(g_pt, ans)}/{recall(sd['tokens'], ans)}",
              file=sys.stderr, flush=True)

    n = len(ids_list)
    report = {
        "kind": "k3_specdecode_gpu_bench",
        "config": vars(args),
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "prompt_tokens": {"min": min(seqlens), "max": max(seqlens)},
        "ar_incremental": {
            "decode_tokens_per_s_mean": round(sum(ar_tps) / n, 3), "recall": round(ar_hits / n, 3)},
        "restored_pertoken": {
            "decode_tokens_per_s_mean": round(sum(pt_tps) / n, 3), "recall": round(pt_hits / n, 3)},
        "restored_specdecode": {
            "decode_tokens_per_s_mean": round(
                sum(r["decode_tokens_per_s"] for r in sd_rows) / n, 3),
            "mean_accept_len": round(sum(r["mean_accept_len"] for r in sd_rows) / n, 2),
            "time_breakdown_s_mean": {
                k: round(sum(r["time_breakdown_s"][k] for r in sd_rows) / n, 3)
                for k in ("aux_clean_forward", "drafter", "incremental_verify")
            },
            "recall": round(sd_hits / n, 3),
            "per_sample": sd_rows,
        },
    }
    sd_tps = report["restored_specdecode"]["decode_tokens_per_s_mean"]
    pt_mean = report["restored_pertoken"]["decode_tokens_per_s_mean"]
    report["restored_specdecode"]["speedup_over_pertoken_x"] = (
        round(sd_tps / pt_mean, 2) if pt_mean else None)
    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_specdecode_gpu_bench_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[sd] AR={report['ar_incremental']['decode_tokens_per_s_mean']} | "
          f"restored-pertoken={pt_mean} | restored-specdecode={sd_tps} tok/s "
          f"(accept_len={report['restored_specdecode']['mean_accept_len']}, "
          f"spec-vs-pertoken {report['restored_specdecode']['speedup_over_pertoken_x']}x) | "
          f"recall ar/pt/sd={report['ar_incremental']['recall']}/"
          f"{report['restored_pertoken']['recall']}/{report['restored_specdecode']['recall']}",
          file=sys.stderr)
    print(f"[sd] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
