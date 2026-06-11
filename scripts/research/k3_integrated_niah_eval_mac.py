"""K3 integrated NIAH eval — **Mac (MLX) path**.

Apple-Silicon counterpart of ``scripts/research/k3_integrated_niah_eval.py``
(the validated CUDA K3 product gate). Wires:

  * verifier = Gemma 4 26B-A4B  (MLX 4-bit, ``mlx_lm.load``)
  * drafter  = DFlash 0.4B       (PyTorch ``DFlashDrafter``, MPS/CPU)
  * f_θ      = trained K/V proj  (PyTorch ``FThetaProjection``)
  * S5       = full-attention layers kept exact (``--s5-exact-full-attn``)

Each generated token runs the **restored** verifier forward (sink+window
local cache + evicted-position K/V restoration), exactly mirroring the CUDA
``CrossModelDLMRestoredVerifier`` semantics. Cross-runtime tensors are
bridged via numpy (see ``scripts/research/k3_dflash_mlx_bridge.py``).

Run on the Mac mini (Apple Silicon, ~24 GB):

  HF_TOKEN unnecessary for the local MLX 4-bit verifier. From repo root:

    PYTHONPATH=.:sdks/python python3 scripts/research/k3_integrated_niah_eval_mac.py \\
        --verifier-path models/gemma-4-26B-A4B-it-mlx-4bit \\
        --drafter-id    models/dflash-kakeya-baseline \\
        --f-theta-dir   results/research/f_theta_v5_s5_sliding \\
        --s5-exact-full-attn \\
        --n-samples 10 --haystack-min-lines 238 --haystack-max-lines 322 \\
        --sink-size 4 --window-size 64 --max-new-tokens 24 \\
        --output results/research/k3_s5_niah_ctx280_mac.json

Quick sanity (smaller / faster):

    --n-samples 4 --haystack-min-lines 60 --haystack-max-lines 81 \\
    --max-new-tokens 16

Diagnostics:
  --identity-restore   restore ALL evicted K/V with the verifier's own true
                       K/V (should match oracle — validates the MLX
                       restoration machinery independent of f_θ / drafter).

Output JSON mirrors the CUDA harness gate schema (recall_cross_model,
recall_oracle, recall_delta_vs_oracle_pp, architectural_correctness).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", default="models/gemma-4-26B-A4B-it-mlx-4bit")
    ap.add_argument("--drafter-id", default="models/dflash-kakeya-baseline")
    ap.add_argument("--f-theta-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--haystack-min-lines", type=int, default=238)
    ap.add_argument("--haystack-max-lines", type=int, default=322)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drafter-device", default="mps",
                    help="torch device for the DFlash drafter + f_θ (mps|cpu)")
    ap.add_argument("--s5-exact-full-attn", action="store_true",
                    help="Keep full-attention layers' K/V exact (S5).")
    ap.add_argument("--identity-restore", action="store_true",
                    help="Restore ALL evicted K/V with the verifier's own "
                         "true K/V (machinery check; should match oracle).")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    import torch

    from inference_engine.v04 import (
        DFlashDrafter, FThetaProjection, NIAHSample,
        aggregate_recall, make_niah_dataset, recall_predicate,
    )
    from inference_engine.v04.kv_merge import compute_evicted_positions
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model, mlx_full_attention_layer_indices,
        kv_source_layer_map, capture_own_kv, restored_logits,
    )
    from scripts.research.k3_dflash_mlx_bridge import (
        mx_to_torch, torch_to_mx,
    )

    torch.manual_seed(args.seed)
    dev = torch.device(args.drafter_device if (
        args.drafter_device == "cpu" or torch.backends.mps.is_available()
    ) else "cpu")

    # ---------- Load verifier (MLX) ----------
    print(f"[mac] loading MLX verifier {args.verifier_path}", file=sys.stderr, flush=True)
    mlx_model, tokenizer = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    embed_scale = float(getattr(text_model, "embed_scale", 1.0))
    n_layers = len(text_model.layers)
    full_attn_idx = mlx_full_attention_layer_indices(text_model)
    src_map = kv_source_layer_map(text_model)
    print(f"[mac] verifier layers={n_layers} full_attn={full_attn_idx}", file=sys.stderr)

    # ---------- Load drafter + f_θ (PyTorch) ----------
    print(f"[mac] loading drafter {args.drafter_id} on {dev}", file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=torch.float32)
    drafter = drafter.to(dev).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    f_theta = FThetaProjection.from_pretrained(
        args.f_theta_dir, dtype=torch.float32, device=dev,
    )
    fcfg = f_theta.config

    # ---------- Drafter K/V capture (Mac): MLX embed → torch → drafter layers ----------
    def capture_drafter_kv(ids: List[int]):
        ids_mx = mx.array([ids])
        emb_mx = text_model.embed_tokens(ids_mx)
        emb_mx = emb_mx * embed_scale
        embedded = mx_to_torch(emb_mx, dtype=torch.float32, device=dev)  # [1,T,H]
        layers = list(drafter.layers)
        k_cap: List[Optional[torch.Tensor]] = [None] * len(layers)
        v_cap: List[Optional[torch.Tensor]] = [None] * len(layers)
        handles = []
        for i, layer in enumerate(layers):
            a = layer.self_attn
            handles.append(a.k_proj.register_forward_hook(
                lambda m, inp, out, i=i: k_cap.__setitem__(i, out.detach())))
            handles.append(a.v_proj.register_forward_hook(
                lambda m, inp, out, i=i: v_cap.__setitem__(i, out.detach())))
        try:
            with torch.no_grad():
                T = embedded.size(1)
                qpos = torch.arange(T, device=dev)
                h = embedded
                for layer in layers:
                    h = layer(h, qpos, ctx_k=None, ctx_v=None)
        finally:
            for hh in handles:
                hh.remove()
        dh, ddim = fcfg.drafter_num_kv_heads, fcfg.drafter_head_dim
        d_k = [k_cap[i].view(1, -1, dh, ddim) for i in range(len(layers))]
        d_v = [v_cap[i].view(1, -1, dh, ddim) for i in range(len(layers))]
        return d_k, d_v

    # ---------- Restored next-token logits ----------
    def restored_next_logits(ids: List[int]) -> int:
        T = len(ids)
        evicted = compute_evicted_positions(T, args.sink_size, args.window_size)
        if not evicted:
            out = mlx_model(mx.array([ids]))
            mx.eval(out)
            return int(mx.argmax(out[0, -1]).item())

        # f_θ projection of drafter K/V → per-verifier-layer K/V (torch)
        d_k, d_v = capture_drafter_kv(ids)
        with torch.no_grad():
            vk, vv = f_theta.forward_kv_pack(d_k, d_v)  # 30× [1,T,kv_i,hd_i]

        # S5 / identity: capture verifier's own true K/V (mx) when needed
        own = None
        if args.s5_exact_full_attn or args.identity_restore:
            own = capture_own_kv(mlx_model, ids)  # {src_idx: (k,v)} mx pre-norm

        exact_set = set(range(n_layers)) if args.identity_restore else set(full_attn_idx)

        rk: Dict[int, Any] = {}
        rv: Dict[int, Any] = {}
        for li in range(n_layers):
            src = src_map[li]
            if src != li:
                continue  # only inject at source (has_kv) layers
            if li in exact_set and own is not None and li in own:
                k_mx, v_mx = own[li]
                rk[li] = k_mx
                rv[li] = v_mx
            else:
                rk[li] = torch_to_mx(vk[li])   # [1,T,kv_i,hd_i] pre-norm
                rv[li] = torch_to_mx(vv[li])
        last = restored_logits(
            mlx_model, ids,
            restored_k_per_layer=rk, restored_v_per_layer=rv,
            evicted_positions=evicted,
        )
        return int(mx.argmax(last).item())

    def oracle_next_logits(ids: List[int]) -> int:
        out = mlx_model(mx.array([ids]))
        mx.eval(out)
        return int(mx.argmax(out[0, -1]).item())

    # ---------- Dataset ----------
    samples: List[NIAHSample] = make_niah_dataset(
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )

    def encode(prompt_text: str) -> List[int]:
        msgs = [{"role": "user", "content": prompt_text}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)

    sample_ids = [encode(s.prompt_text) for s in samples]
    seq_lens = [len(t) for t in sample_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    print(f"[mac] {len(samples)} samples, prompt len "
          f"min={min(seq_lens)} max={max(seq_lens)}", file=sys.stderr)

    def greedy(step_fn) -> Tuple[List[str], List[float], List[int]]:
        decoded, lats, toks = [], [], []
        for i, base in enumerate(sample_ids):
            cur = list(base)
            gen: List[int] = []
            t0 = time.perf_counter()
            for _ in range(args.max_new_tokens):
                nxt = step_fn(cur)
                gen.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
                cur.append(nxt)
            lats.append(time.perf_counter() - t0)
            decoded.append(tokenizer.decode(gen))
            toks.append(len(gen))
            print(f"[mac]   sample {i}: T={seq_lens[i]} -> {decoded[-1][:48]!r}",
                  file=sys.stderr)
        return decoded, lats, toks

    label = "identity" if args.identity_restore else (
        "s5" if args.s5_exact_full_attn else "f_theta_all")
    print(f"[mac] running restored cross-model verifier ({label})", file=sys.stderr, flush=True)
    cross_dec, cross_lat, cross_tok = greedy(restored_next_logits)
    cross_res = aggregate_recall("k3_cross_model_mac", samples, cross_dec, cross_lat, cross_tok)
    print(f"[mac] cross-model recall = {cross_res.recall:.3f} "
          f"({cross_res.samples_correct}/{cross_res.samples_total})", file=sys.stderr)

    oracle_res = None
    if not args.skip_oracle:
        print("[mac] running oracle (full MLX forward)", file=sys.stderr, flush=True)
        o_dec, o_lat, o_tok = greedy(oracle_next_logits)
        oracle_res = aggregate_recall("oracle_mac", samples, o_dec, o_lat, o_tok)
        print(f"[mac] oracle recall = {oracle_res.recall:.3f}", file=sys.stderr)

    delta = (abs(cross_res.recall - oracle_res.recall) if oracle_res else None)
    report = {
        "schema_version": 1,
        "kind": "k3_integrated_niah_acceptance_mac",
        "config": {
            "verifier_path": args.verifier_path,
            "drafter_id": args.drafter_id,
            "f_theta_dir": args.f_theta_dir,
            "n_samples": args.n_samples,
            "sink_size": args.sink_size,
            "window_size": args.window_size,
            "haystack_min_lines": args.haystack_min_lines,
            "haystack_max_lines": args.haystack_max_lines,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "s5_exact_full_attn": bool(args.s5_exact_full_attn),
            "identity_restore": bool(args.identity_restore),
            "full_attention_layers": full_attn_idx,
            "prompt_token_lens": seq_lens,
        },
        "results": {
            "k3_cross_model": dataclasses.asdict(cross_res),
            **({"oracle": dataclasses.asdict(oracle_res)} if oracle_res else {}),
        },
        "gate": {
            "recall_cross_model": cross_res.recall,
            "recall_oracle": oracle_res.recall if oracle_res else None,
            "recall_delta_vs_oracle_pp": (delta * 100 if delta is not None else None),
            "recall_delta_within_5pp": (delta is not None and delta <= 0.05),
        },
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_integrated_niah_mac_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[mac] DONE. cross={cross_res.recall:.3f} "
          f"oracle={oracle_res.recall if oracle_res else 'skipped'} "
          f"-> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
