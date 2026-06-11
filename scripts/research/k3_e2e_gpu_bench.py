"""K3 end-to-end GPU benchmark — Kakeya restored verifier vs standalone AR.

Runs the served Kakeya inference path (the Gap 1 + Gap 2
``CrossModelRestoredSinkWindowVerifier``: f_θ + S5 K/V Restoration over a
bounded sink+window cache) and the standalone Gemma 4 26B-A4B AR model on
the *same* NIAH prompts, and reports, per context rung:

  * **Memory** — resident KV bytes (restored = bounded sink+window;
    AR = the model's own HF cache, which grows with context) + peak GPU.
  * **Throughput** — decode tokens/s (excludes prefill).
  * **Verifier attention context length** — restored: resident *window*
    (sink+window) vs *effective* context (full prompt+gen, reconstructed
    via restoration); AR: full resident context.
  * **Recall** — fraction of NIAH needles recalled (correctness check
    that the served restored path still answers).

Run on a CUDA host (e.g. H200) inside the transformers-5.x venv::

    HF_HOME=/workspace/.hf_home PYTHONPATH=.:sdks/python \
      .venv-k3/bin/python scripts/research/k3_e2e_gpu_bench.py \
        --verifier-id google/gemma-4-26B-A4B-it \
        --drafter-id models/dflash-kakeya-baseline \
        --f-theta-dir results/research/f_theta_v5_s5_sliding \
        --haystack-lines 60,160 --n-samples 3 --gen-tokens 16 \
        --output results/research/k3_e2e_gpu_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def _kv_bytes_from_cache(cache: Any) -> int:
    """Sum K+V bytes across an HF cache (DynamicCache / hybrid)."""
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is None:
        # Legacy tuple-of-tuples cache.
        try:
            for k, v in cache:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
            return total
        except TypeError:
            return 0
    for layer in layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if t is not None and hasattr(t, "numel"):
                total += t.numel() * t.element_size()
    return total


def _peak_gpu(device) -> int:
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return -1


@torch.no_grad()
def run_ar(model, ids_list, samples, gen_tokens, tokenizer, device) -> Dict[str, Any]:
    """Standalone AR: incremental decode with the model's own KV cache."""
    n = len(ids_list)
    tot_tok = 0
    tot_t = 0.0
    hits = 0
    kv_bytes = 0
    peak = 0
    prefill_t = 0.0
    for i, ids in enumerate(ids_list):
        torch.cuda.reset_peak_memory_stats(device)
        t_pf = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_t += time.perf_counter() - t_pf
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        gen_ids: List[int] = []
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
        t0 = time.perf_counter()
        for _ in range(gen_tokens):
            gen_ids.append(nxt)
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
        torch.cuda.synchronize(device)
        tot_t += time.perf_counter() - t0
        tot_tok += len(gen_ids)
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if samples[i].answer_text in txt:
            hits += 1
        kv_bytes = _kv_bytes_from_cache(cache)  # last sample (full context)
        peak = max(peak, _peak_gpu(device))
        print(f"[e2e]   AR sample {i}: gen={len(gen_ids)} "
              f"recall={'Y' if samples[i].answer_text in txt else 'N'} "
              f"out[:40]={txt[:40]!r}", file=sys.stderr, flush=True)
    return {
        "decode_tokens_per_s": round(tot_tok / tot_t, 3) if tot_t > 0 else None,
        "prefill_s_mean": round(prefill_t / n, 4),
        "kv_bytes_final": kv_bytes,
        "peak_mem_bytes": peak,
        "recall": round(hits / n, 3),
        "decode_tokens": tot_tok,
    }


@torch.no_grad()
def run_restored(adapter, ids_list, samples, gen_tokens, tokenizer, device) -> Dict[str, Any]:
    """Kakeya restored path: bounded sink+window cache + f_θ/S5 restoration."""
    n = len(ids_list)
    tot_tok = 0
    tot_t = 0.0
    hits = 0
    peak = 0
    resident_kv = 0
    eff_ctx = 0
    prefill_t = 0.0
    for i, ids in enumerate(ids_list):
        torch.cuda.reset_peak_memory_stats(device)
        prompt = ids[0].tolist()
        t_pf = time.perf_counter()
        adapter.prefill(prompt)
        torch.cuda.synchronize(device)
        prefill_t += time.perf_counter() - t_pf
        nxt = int(adapter.next_token_logits.argmax().item())
        gen_ids: List[int] = []
        t0 = time.perf_counter()
        for _ in range(gen_tokens):
            gen_ids.append(nxt)
            adapter.append_token(nxt)
            nxt = int(adapter.next_token_logits.argmax().item())
        torch.cuda.synchronize(device)
        tot_t += time.perf_counter() - t0
        tot_tok += len(gen_ids)
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if samples[i].answer_text in txt:
            hits += 1
        resident_kv = adapter.live_kv_bytes()
        eff_ctx = max(eff_ctx, len(adapter._committed))
        peak = max(peak, _peak_gpu(device))
        print(f"[e2e]   restored sample {i}: gen={len(gen_ids)} "
              f"recall={'Y' if samples[i].answer_text in txt else 'N'} "
              f"resident_kv_tok={adapter.sink_size + adapter.window_size} "
              f"eff_ctx={len(adapter._committed)} "
              f"out[:40]={txt[:40]!r}", file=sys.stderr, flush=True)
    return {
        "decode_tokens_per_s": round(tot_tok / tot_t, 3) if tot_t > 0 else None,
        "prefill_s_mean": round(prefill_t / n, 4),
        "resident_kv_bytes": resident_kv,
        "resident_window_tokens": adapter.sink_size + adapter.window_size,
        "effective_context_tokens": eff_ctx,
        "peak_mem_bytes": peak,
        "recall": round(hits / n, 3),
        "decode_tokens": tot_tok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="models/dflash-kakeya-baseline")
    ap.add_argument("--f-theta-dir", default="results/research/f_theta_v5_s5_sliding")
    ap.add_argument("--haystack-lines", default="60,160",
                    help="Comma-separated haystack line counts (context rungs).")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--incremental", action="store_true",
                    help="Use the incremental-decode restored path (capture "
                         "restored K/V at prefill, then native O(L)/block "
                         "decode) instead of the O(T) re-forward per step.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[e2e] CUDA not available — this benchmark requires a GPU.",
              file=sys.stderr)
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
        resolve_text_config,
    )

    print(f"[e2e] loading verifier {args.verifier_id}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.verifier_id)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="eager",
        device_map="auto",
    ).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)

    print(f"[e2e] loading drafter {args.drafter_id}", file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype).to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)

    print(f"[e2e] loading f_θ {args.f_theta_dir}", file=sys.stderr, flush=True)
    f_theta = FThetaProjection.from_pretrained(
        args.f_theta_dir, dtype=torch.float32, device=device,
    )

    exact_layers = full_attention_layer_indices(verifier)
    print(f"[e2e] S5 exact full-attention layers: {exact_layers}", file=sys.stderr)
    restored = CrossModelDLMRestoredVerifier(
        verifier_model=verifier, drafter=drafter, f_theta=f_theta,
        sink_size=args.sink, window_size=args.window,
        exact_layer_indices=exact_layers,
    )
    adapter = CrossModelRestoredSinkWindowVerifier(
        restored,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        device="cuda",
        incremental=args.incremental,
    )
    print(f"[e2e] restored adapter incremental={args.incremental}", file=sys.stderr)

    v_cfg = resolve_text_config(verifier.config)
    verifier_dims = {
        "num_hidden_layers": int(getattr(v_cfg, "num_hidden_layers", 0)),
        "num_key_value_heads": int(getattr(v_cfg, "num_key_value_heads", 0) or 0),
        "head_dim": int(getattr(v_cfg, "head_dim", 0) or 0),
        "sliding_window": int(getattr(v_cfg, "sliding_window", 0) or 0),
    }

    def encode_chat(text: str) -> torch.Tensor:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        elif isinstance(ids, list):
            ids = torch.tensor([ids])
        return ids.to(device)

    rungs = [int(x) for x in args.haystack_lines.split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for lines in rungs:
        samples = make_niah_dataset(
            n_samples=args.n_samples,
            haystack_min_lines=lines, haystack_max_lines=lines, seed=args.seed,
        )
        ids_list = [encode_chat(s.prompt_text) for s in samples]
        seqlens = [int(t.size(1)) for t in ids_list]
        print(f"\n[e2e] === rung: {lines} haystack lines | prompt tokens "
              f"min={min(seqlens)} max={max(seqlens)} ===", file=sys.stderr, flush=True)

        print("[e2e] running standalone AR baseline ...", file=sys.stderr, flush=True)
        ar = run_ar(verifier, ids_list, samples, args.gen_tokens, tokenizer, device)
        print("[e2e] running Kakeya restored path ...", file=sys.stderr, flush=True)
        rs = run_restored(adapter, ids_list, samples, args.gen_tokens, tokenizer, device)

        kv_saving = (ar["kv_bytes_final"] / rs["resident_kv_bytes"]
                     if rs["resident_kv_bytes"] else None)
        ctx_compression = (rs["effective_context_tokens"] / rs["resident_window_tokens"]
                           if rs["resident_window_tokens"] else None)
        row = {
            "haystack_lines": lines,
            "prompt_tokens": {"min": min(seqlens), "max": max(seqlens)},
            "ar": ar,
            "restored": rs,
            "comparison": {
                "kv_memory_saving_x": round(kv_saving, 1) if kv_saving else None,
                "ar_kv_mb": round(ar["kv_bytes_final"] / 1e6, 2),
                "restored_resident_kv_mb": round(rs["resident_kv_bytes"] / 1e6, 2),
                "context_compression_x": round(ctx_compression, 1) if ctx_compression else None,
                "throughput_ratio_restored_over_ar": (
                    round(rs["decode_tokens_per_s"] / ar["decode_tokens_per_s"], 3)
                    if ar["decode_tokens_per_s"] else None
                ),
            },
        }
        rows.append(row)
        c = row["comparison"]
        print(f"[e2e] rung {lines}: KV {c['ar_kv_mb']}MB(AR) vs "
              f"{c['restored_resident_kv_mb']}MB(restored) -> {c['kv_memory_saving_x']}x saving; "
              f"ctx {rs['effective_context_tokens']} tok over {rs['resident_window_tokens']}-tok window "
              f"({c['context_compression_x']}x); "
              f"tok/s restored={rs['decode_tokens_per_s']} ar={ar['decode_tokens_per_s']}; "
              f"recall restored={rs['recall']} ar={ar['recall']}", file=sys.stderr, flush=True)

    report = {
        "kind": "k3_e2e_gpu_bench",
        "config": {
            "verifier_id": args.verifier_id,
            "drafter_id": args.drafter_id,
            "f_theta_dir": args.f_theta_dir,
            "sink_size": args.sink, "window_size": args.window,
            "gen_tokens": args.gen_tokens, "n_samples": args.n_samples,
            "haystack_lines": rungs,
        },
        "verifier_dims": verifier_dims,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "results": rows,
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_e2e_gpu_bench_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[e2e] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
