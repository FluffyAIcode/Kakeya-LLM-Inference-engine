"""PR-A3c end-to-end: per-session binding + true parallel multi-tenant decode.

Measures **parallel-inference throughput** for the recall-preserving restored S5
path, on CUDA. On one accelerator, true parallelism = a **batched** forward: N
sessions decoded in one pass, each session = one batch row with its own KV-cache
row (per-session binding). This is the capability v0.3's single-tenant served
path lacks (RPCs serialized on one verifier — PR-A3c).

For each batch size N it runs, on the SAME N prompts:
  * batched **AR** (native HF gemma)        — the parallel throughput ceiling
  * batched **restored S5** (Kakeya)        — recall-preserving bounded path

and reports aggregate decode tok/s (N rows in parallel), per-session recall
(must stay 1.0 — recall is the bottom line; the non-recall pure sink+window
config is intentionally NOT tested), and parallel scaling vs the N=1 rate.

Equal-length prompts (a modal-length NIAH bucket, tiled) keep the batch clean —
the restored forward has no attention-mask plumbing, so padding is avoided.
Recall-sacrificing configs are out of scope by request.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch


@torch.no_grad()
def _ar_batched(model, ids_bt, gen_tokens, device, eos_ids):
    """Batched AR decode. ids_bt: [N, T]. Returns (per_row_tokens, decode_s)."""
    N = ids_bt.size(0)
    out = model(input_ids=ids_bt, use_cache=True)
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1)              # [N]
    gen = [[int(nxt[i].item())] for i in range(N)]
    T = ids_bt.size(1)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for step in range(gen_tokens - 1):
        cur = nxt.view(N, 1)
        pos = torch.full((N, 1), T + step, device=device, dtype=torch.long)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                    cache_position=torch.tensor([T + step], device=device))
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        for i in range(N):
            gen[i].append(int(nxt[i].item()))
    torch.cuda.synchronize(device)
    return gen, time.perf_counter() - t0


@torch.no_grad()
def _restored_prefill_batched(restored, ids_bt, helpers):
    """Batched restored S5 prefill -> (DynamicCache, last_logits [N, V])."""
    from transformers.cache_utils import DynamicCache
    n_layers = len(_decoder_layers(restored.verifier_model))
    capture: list = [None] * n_layers
    out = restored.forward(ids_bt, capture_kv=capture, **helpers)
    logits = out.logits if hasattr(out, "logits") else out
    if any(c is None for c in capture):
        raise RuntimeError("restored prefill did not capture all layers "
                           "(prompt must exceed sink+window)")
    cache = DynamicCache()
    for li, (k, v) in enumerate(capture):
        cache.update(k, v, li)
    return cache, logits[:, -1, :]


@torch.no_grad()
def _restored_decode_batched(model, cache, last_logits, gen_tokens, T, device):
    N = last_logits.size(0)
    nxt = last_logits.argmax(-1)
    gen = [[int(nxt[i].item())] for i in range(N)]
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for step in range(gen_tokens - 1):
        cur = nxt.view(N, 1)
        pos = torch.full((N, 1), T + step, device=device, dtype=torch.long)
        cpos = torch.tensor([T + step], device=device)
        out = model(input_ids=cur, position_ids=pos, cache_position=cpos,
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        for i in range(N):
            gen[i].append(int(nxt[i].item()))
    torch.cuda.synchronize(device)
    return gen, time.perf_counter() - t0


def _decoder_layers(model):
    from inference_engine.v04.cross_model_dlm_verifier import get_verifier_decoder
    return get_verifier_decoder(model).layers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--f-theta-dir", default="results/research/f_theta_v5_s5_sliding")
    ap.add_argument("--haystack-lines", type=int, default=160)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[mt] CUDA required.", file=sys.stderr)
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

    print(f"[mt] loading verifier {args.verifier_id}", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype).to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    f_theta = FThetaProjection.from_pretrained(args.f_theta_dir, dtype=torch.float32, device=device)
    exact_layers = full_attention_layer_indices(verifier)
    restored = CrossModelDLMRestoredVerifier(
        verifier_model=verifier, drafter=drafter, f_theta=f_theta,
        sink_size=args.sink, window_size=args.window, exact_layer_indices=exact_layers,
    )
    helpers = dict(apply_rotary_pos_emb=apply_rotary_pos_emb,
                   eager_attention_forward=eager_attention_forward,
                   all_attention_functions=ALL_ATTENTION_FUNCTIONS)
    eos_ids = set(x for x in [tok.eos_token_id] if x is not None)

    def encode_chat(text):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids[0].tolist()

    # Build an equal-length prompt set: pick the modal token length so the batch
    # needs no padding (restored forward has no attention-mask path).
    pool = make_niah_dataset(n_samples=args.pool,
                             haystack_min_lines=args.haystack_lines,
                             haystack_max_lines=args.haystack_lines, seed=args.seed)
    enc = [(encode_chat(s.prompt_text), s.answer_text) for s in pool]
    lengths = Counter(len(e[0]) for e in enc)
    modal_len, _ = lengths.most_common(1)[0]
    bucket = [(ids, ans) for ids, ans in enc if len(ids) == modal_len]
    print(f"[mt] modal prompt len={modal_len}, {len(bucket)} equal-length prompts "
          f"(of {len(enc)})", file=sys.stderr, flush=True)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    need = max(batch_sizes)
    while len(bucket) < need:                  # tile distinct prompts up to N
        bucket += bucket[: need - len(bucket)]

    def recall(tokens, ans):
        return ans in tok.decode(tokens, skip_special_tokens=True)

    # warmup (kernels) at the largest batch
    print("[mt] warmup ...", file=sys.stderr, flush=True)
    wb = torch.tensor([b[0] for b in bucket[:max(batch_sizes)]], device=device)
    try:
        _ar_batched(verifier, wb, 4, device, eos_ids)
        c, ll = _restored_prefill_batched(restored, wb, helpers)
        _restored_decode_batched(verifier, c, ll, 4, modal_len, device)
    except Exception as e:  # noqa: BLE001
        print(f"[mt] warmup note: {e}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    single = {}
    for N in batch_sizes:
        sel = bucket[:N]
        ids_bt = torch.tensor([s[0] for s in sel], device=device)
        ans = [s[1] for s in sel]
        # AR
        g_ar, dt_ar = _ar_batched(verifier, ids_bt, args.gen_tokens, device, eos_ids)
        ar_tps = (N * args.gen_tokens) / dt_ar
        ar_rec = sum(recall(g, a) for g, a in zip(g_ar, ans)) / N
        # restored S5
        cache, last = _restored_prefill_batched(restored, ids_bt, helpers)
        g_rs, dt_rs = _restored_decode_batched(verifier, cache, last,
                                               args.gen_tokens, modal_len, device)
        rs_tps = (N * args.gen_tokens) / dt_rs
        rs_rec = sum(recall(g, a) for g, a in zip(g_rs, ans)) / N
        peak = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        if N == 1:
            single = {"ar": ar_tps, "restored": rs_tps}
        row = {
            "agents": N,
            "ar_aggregate_tps": round(ar_tps, 2),
            "restored_aggregate_tps": round(rs_tps, 2),
            "ar_recall": round(ar_rec, 3),
            "restored_recall": round(rs_rec, 3),
            "ar_parallel_speedup_vs_n1": round(ar_tps / single["ar"], 2) if single else None,
            "restored_parallel_speedup_vs_n1": round(rs_tps / single["restored"], 2) if single else None,
            "peak_gpu_gb": peak,
        }
        rows.append(row)
        print(f"[mt] N={N:3d} | AR {row['ar_aggregate_tps']} tok/s "
              f"(x{row['ar_parallel_speedup_vs_n1']}, recall {row['ar_recall']}) | "
              f"restored {row['restored_aggregate_tps']} tok/s "
              f"(x{row['restored_parallel_speedup_vs_n1']}, recall {row['restored_recall']}) "
              f"| peak {peak}GB", file=sys.stderr, flush=True)

    report = {
        "kind": "k3_cuda_multitenant_parallel",
        "schema_version": 1,
        "config": {
            "verifier_id": args.verifier_id, "drafter_id": args.drafter_id,
            "haystack_lines": args.haystack_lines, "modal_prompt_len": modal_len,
            "gen_tokens": args.gen_tokens, "sink": args.sink, "window": args.window,
            "batch_sizes": batch_sizes, "exact_layers": exact_layers,
            "note": ("per-session binding via batched decode (each row = a "
                     "session with its own KV-cache row); recall-preserving S5 "
                     "only — non-recall configs out of scope."),
        },
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "results": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[mt] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
