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
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--teacher-forced", action="store_true",
                    help="DIAGNOSTIC ONLY (under-measures retrieval): single "
                         "restored forward per sample, check argmax at the "
                         "needle-code span. Note this misses the model's "
                         "preamble so it reads ~0 even for the oracle — use "
                         "the default free-generation for a real recall "
                         "number. Free-gen oracle uses mlx's fast native "
                         "incremental cache; the restored cross path does a "
                         "full forward per token (slow on M4 — see notes).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drafter-device", default="mps",
                    help="torch device for the DFlash drafter + f_θ (mps|cpu)")
    ap.add_argument("--s5-exact-full-attn", action="store_true",
                    help="Keep full-attention layers' K/V exact (S5).")
    ap.add_argument("--identity-restore", action="store_true",
                    help="Restore ALL evicted K/V with the verifier's own "
                         "true K/V (machinery check; should match oracle).")
    ap.add_argument("--compress-full-attn", action="store_true",
                    help="KakeyaLattice-compress the exact full-attention "
                         "layers' K/V (lossy round-trip) to shrink the O(T) "
                         "linear term. Reports the compression ratio + recall "
                         "under compression.")
    ap.add_argument("--kl-lattice", default="D4", choices=["D4", "E8"])
    ap.add_argument("--kl-q-range", type=int, default=38)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--chat-template", action="store_true",
                    help="Diagnostic mode: wrap NIAH completion prompts in "
                         "the model chat template. Default is raw completion "
                         "prompt because NIAH already ends with 'Answer:'.")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    import torch
    from mlx_lm.models.cache import KVCache  # type: ignore

    from inference_engine.v04 import (
        DFlashDrafter, FThetaProjection, NIAHSample,
        aggregate_recall, make_niah_dataset, recall_predicate,
    )
    from inference_engine.v04.kv_merge import compute_evicted_positions
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model, mlx_full_attention_layer_indices,
        kv_source_layer_map, capture_own_kv, restored_logits,
        per_layer_kv_geometry, kv_memory_report,
    )
    from inference_engine.backends.mlx.cache import make_sink_window_cache
    from inference_engine.v04.kv_compressor import make_default_compressor
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

    # ---------- Optional KakeyaLattice compression of full-attn layers ----------
    geom = per_layer_kv_geometry(text_model)
    compressors: Dict[int, Any] = {}
    kl_bits_per_head: Optional[float] = None
    if args.compress_full_attn:
        for li in full_attn_idx:
            n_kv, hd, _ = geom[li]
            comp = make_default_compressor(
                head_dim=hd, device=torch.device("cpu"),
                prefer_kakeya=True, lattice=args.kl_lattice, q_range=args.kl_q_range,
            )
            compressors[li] = comp
            codec = getattr(comp, "_codec", None)
            if codec is not None and kl_bits_per_head is None:
                kl_bits_per_head = float(getattr(codec, "bits_per_token_per_head", 0)) or None
        print(f"[mac] KakeyaLattice compression ON for full-attn layers "
              f"({args.kl_lattice} Q{args.kl_q_range}); "
              f"bits/token/head={kl_bits_per_head}", file=sys.stderr)

    def _compress_roundtrip(li: int, k_mx: Any, v_mx: Any):
        """Lossy KakeyaLattice round-trip of a full-attn layer's pre-norm K/V.
        mx [B,T,n_kv,hd] → torch [B,n_kv,T,hd] (positions=-2) → codec → back."""
        comp = compressors[li]
        kt = mx_to_torch(k_mx, dtype=torch.float32, device="cpu").transpose(1, 2).contiguous()
        vt = mx_to_torch(v_mx, dtype=torch.float32, device="cpu").transpose(1, 2).contiguous()
        T = kt.shape[-2]
        pos = torch.arange(T)
        comp.compress(kt, vt, pos)
        kh, vh = comp.decompress(pos)
        comp.evict(pos)  # keep state bounded between tokens
        kh = kh.transpose(1, 2).contiguous()   # [B,T,n_kv,hd]
        vh = vh.transpose(1, 2).contiguous()
        return torch_to_mx(kh), torch_to_mx(vh)

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

    # ---------- Per-sample restoration (amortized: captured ONCE over the
    # prompt, reused for all decode steps). The evicted positions are the
    # fixed prompt mid-context; with <= window generated tokens nothing else
    # is evicted, so the prompt's restored K/V cover every injected slot.
    exact_set = set(range(n_layers)) if args.identity_restore else set(full_attn_idx)

    def build_restoration(prompt_ids: List[int]):
        own = None
        if exact_set:
            own = capture_own_kv(mlx_model, prompt_ids)

        # S5 on Gemma 4: sliding-attention layers are local by design; adding
        # restored mid-context K/V to them only makes every decode step attend
        # over a long bank that the sliding mask would otherwise exclude. The
        # recall-critical long-range path is the full-attention layers, so the
        # product fast path restores only those exact layers.
        if args.s5_exact_full_attn and not args.identity_restore:
            rk: Dict[int, Any] = {}
            rv: Dict[int, Any] = {}
            for li in full_attn_idx:
                if own is None or li not in own:
                    continue
                k_mx, v_mx = own[li]
                if li in compressors:
                    k_mx, v_mx = _compress_roundtrip(li, k_mx, v_mx)
                rk[li], rv[li] = k_mx, v_mx
            return rk, rv, len(prompt_ids)

        d_k, d_v = capture_drafter_kv(prompt_ids)
        with torch.no_grad():
            vk, vv = f_theta.forward_kv_pack(d_k, d_v)
        rk: Dict[int, Any] = {}
        rv: Dict[int, Any] = {}
        for li in range(n_layers):
            if src_map[li] != li:
                continue
            if li in exact_set and own is not None and li in own:
                k_mx, v_mx = own[li]
                if li in compressors:
                    k_mx, v_mx = _compress_roundtrip(li, k_mx, v_mx)
                rk[li], rv[li] = k_mx, v_mx
            else:
                rk[li], rv[li] = torch_to_mx(vk[li]), torch_to_mx(vv[li])
        return rk, rv, len(prompt_ids)

    def _pad(rdict, t_src, t_dst):
        if t_dst <= t_src:
            return rdict
        out = {}
        for li, a in rdict.items():
            pad = mx.zeros((a.shape[0], t_dst - t_src, a.shape[2], a.shape[3]), dtype=a.dtype)
            out[li] = mx.concatenate([a, pad], axis=1)
        return out

    def restored_forward(ids: List[int], rk, rv, t_src, *, return_all: bool):
        T = len(ids)
        evicted = compute_evicted_positions(T, args.sink_size, args.window_size)
        if not evicted:
            out = mlx_model(mx.array([ids])); mx.eval(out)
            return out[0] if return_all else out[0, -1]
        return restored_logits(
            mlx_model, ids,
            restored_k_per_layer=_pad(rk, t_src, T),
            restored_v_per_layer=_pad(rv, t_src, T),
            evicted_positions=evicted, return_all=return_all,
        )

    def _post_rope_restored_bank(
        layer_idx: int, k_mx: Any, v_mx: Any, evicted: Sequence[int],
    ) -> Tuple[Any, Any]:
        """Convert restored pre-norm K/V into MLX cache-layout K/V.

        ``restored_logits`` injects pre-norm K/V during a full forward. The
        incremental path needs the equivalent post-norm/post-RoPE tensors in
        cache layout ``[B, n_kv, T_restored, head_dim]`` so decode can reuse
        MLX's native cache hot path instead of full re-forwarding the prompt.
        """
        layer = text_model.layers[layer_idx]
        attn = layer.self_attn
        start, end = int(evicted[0]), int(evicted[-1]) + 1
        k = k_mx[:, start:end, :, :]
        v = v_mx[:, start:end, :, :]
        k = attn.k_norm(k).transpose(0, 2, 1, 3)
        k = attn.rope(k, offset=start)
        v = attn.v_norm(v).transpose(0, 2, 1, 3)
        return k, v

    def attach_restored_banks(cache, rk, rv, prompt_len: int) -> None:
        evicted = compute_evicted_positions(
            prompt_len, args.sink_size, args.window_size,
        )
        if not evicted:
            return
        for li, k_mx in rk.items():
            c = cache[li]
            v_mx = rv[li]
            kb, vb = _post_rope_restored_bank(li, k_mx, v_mx, evicted)
            if li in full_attn_idx and getattr(c, "keys", None) is not None:
                # Full-attention layers are the long-context recall carriers in
                # S5. Hand them back to MLX's native KVCache so decode appends
                # into a preallocated buffer instead of re-concatenating the
                # restored bank on every generated token.
                full_cache = KVCache()
                full_cache.state = (
                    mx.concatenate([kb, c.keys], axis=2),
                    mx.concatenate([vb, c.values], axis=2),
                )
                cache[li] = full_cache
            elif hasattr(c, "set_restored_bank"):
                c.set_restored_bank(kb, vb)

    # ---------- Dataset ----------
    samples: List[NIAHSample] = make_niah_dataset(
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )

    def encode(prompt_text: str) -> List[int]:
        if not args.chat_template:
            ids = tokenizer.encode(prompt_text)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return list(ids)
        msgs = [{"role": "user", "content": prompt_text}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)

    def encode_answer(answer_text: str) -> List[int]:
        try:
            aid = tokenizer.encode(answer_text, add_special_tokens=False)
        except TypeError:
            aid = list(tokenizer.encode(answer_text))
            bos = getattr(tokenizer, "bos_token_id", None)
            if aid and bos is not None and aid[0] == bos:
                aid = aid[1:]
        return list(aid)

    sample_ids = [encode(s.prompt_text) for s in samples]
    answer_ids = [encode_answer(s.answer_text) for s in samples]
    seq_lens = [len(t) for t in sample_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    print(f"[mac] {len(samples)} samples, prompt len "
          f"min={min(seq_lens)} max={max(seq_lens)}", file=sys.stderr)

    def eval_teacher_forced(logits_all_fn) -> Tuple[List[str], List[float], List[int]]:
        """One restored forward per sample over [prompt + needle-code]; check
        the argmax at the answer span reproduces the code (substring predicate
        — same as CUDA). O(T) per sample, no autoregressive loop."""
        decoded, lats, toks = [], [], []
        for i, pid in enumerate(sample_ids):
            aid = answer_ids[i] or [eos_id or 0]
            full = pid + aid
            t0 = time.perf_counter()
            logits_all = logits_all_fn(pid, full)  # [T_full, V]
            Tp = len(pid)
            preds = [int(mx.argmax(logits_all[Tp - 1 + j]).item())
                     for j in range(len(aid))]
            lats.append(time.perf_counter() - t0)
            decoded.append(tokenizer.decode(preds))
            toks.append(len(aid))
            print(f"[mac]   sample {i}: T={seq_lens[i]} pred[:48]={decoded[-1][:48]!r}",
                  file=sys.stderr)
        return decoded, lats, toks

    def eval_free_gen_cross() -> Tuple[List[str], List[float], List[int]]:
        """Restored free generation on MLX's incremental cache hot path.

        The expensive cross-model restoration is built once over the prompt.
        Generation then uses ``make_sink_window_cache`` plus a decode-only
        restored K/V bank for evicted prompt positions, avoiding the previous
        full re-forward of the entire prompt on every generated token.
        """
        decoded, lats, toks = [], [], []
        stage_rows = []
        for i, pid in enumerate(sample_ids):
            t_build0 = time.perf_counter()
            rk, rv, tsrc = build_restoration(pid)
            build_s = time.perf_counter() - t_build0
            cache = make_sink_window_cache(
                mlx_model, sink_size=args.sink_size, window_size=args.window_size,
            )
            cur = list(pid); gen: List[int] = []
            t0 = time.perf_counter()
            t_prefill0 = time.perf_counter()
            out = mlx_model(mx.array([cur]), cache=cache)
            mx.eval(out)
            attach_restored_banks(cache, rk, rv, tsrc)
            prefill_attach_s = time.perf_counter() - t_prefill0
            nxt = int(mx.argmax(out[0, -1]).item())
            gen.append(nxt)
            if eos_id is not None and nxt == eos_id:
                lats.append(time.perf_counter() - t0)
                decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
                stage_rows.append({
                    "sample": i,
                    "build_restoration_s": round(build_s, 3),
                    "prefill_attach_s": round(prefill_attach_s, 3),
                    "decode_s": 0.0,
                })
                print(f"[mac]   sample {i}: T={seq_lens[i]} -> {decoded[-1][:48]!r}",
                      file=sys.stderr)
                continue
            cur.append(nxt)
            t_decode0 = time.perf_counter()
            for _ in range(args.max_new_tokens):
                if len(gen) >= args.max_new_tokens:
                    break
                out = mlx_model(mx.array([[nxt]]), cache=cache)
                mx.eval(out)
                nxt = int(mx.argmax(out[0, -1]).item()); gen.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
                cur.append(nxt)
            decode_s = time.perf_counter() - t_decode0
            lats.append(time.perf_counter() - t0)
            decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
            stage_rows.append({
                "sample": i,
                "build_restoration_s": round(build_s, 3),
                "prefill_attach_s": round(prefill_attach_s, 3),
                "decode_s": round(decode_s, 3),
            })
            print(f"[mac]   sample {i}: T={seq_lens[i]} -> {decoded[-1][:48]!r}",
                  file=sys.stderr)
        eval_free_gen_cross.stage_rows = stage_rows
        return decoded, lats, toks

    def eval_free_gen_oracle() -> Tuple[List[str], List[float], List[int]]:
        """Oracle free generation using mlx's NATIVE incremental KV cache
        (fast + correct reference; confirms the metric/dataset)."""
        decoded, lats, toks = [], [], []
        make_cache = getattr(mlx_model, "make_cache", None)
        for i, pid in enumerate(sample_ids):
            cache = make_cache() if make_cache is not None else None
            t0 = time.perf_counter()
            out = mlx_model(mx.array([pid]), cache=cache); mx.eval(out)
            tok = int(mx.argmax(out[0, -1]).item()); gen = [tok]
            for _ in range(args.max_new_tokens - 1):
                if eos_id is not None and tok == eos_id:
                    break
                out = mlx_model(mx.array([[tok]]), cache=cache); mx.eval(out)
                tok = int(mx.argmax(out[0, -1]).item()); gen.append(tok)
            lats.append(time.perf_counter() - t0)
            decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
            print(f"[mac]   oracle {i}: T={seq_lens[i]} -> {decoded[-1][:48]!r}",
                  file=sys.stderr)
        return decoded, lats, toks

    def cross_logits_all(prompt_ids, full_ids):
        rk, rv, tsrc = build_restoration(prompt_ids)
        return restored_forward(full_ids, rk, rv, tsrc, return_all=True)

    def oracle_logits_all(prompt_ids, full_ids):
        out = mlx_model(mx.array([full_ids])); mx.eval(out); return out[0]

    label = "identity" if args.identity_restore else (
        "s5" if args.s5_exact_full_attn else "f_theta_all")
    eval_mode = "teacher_forced" if args.teacher_forced else "free_gen"
    print(f"[mac] running restored cross-model verifier ({label}, {eval_mode})",
          file=sys.stderr, flush=True)
    if args.teacher_forced:
        cross_dec, cross_lat, cross_tok = eval_teacher_forced(cross_logits_all)
    else:
        cross_dec, cross_lat, cross_tok = eval_free_gen_cross()
    cross_res = aggregate_recall("k3_cross_model_mac", samples, cross_dec, cross_lat, cross_tok)
    print(f"[mac] cross-model recall = {cross_res.recall:.3f} "
          f"({cross_res.samples_correct}/{cross_res.samples_total})", file=sys.stderr)

    oracle_res = None
    if not args.skip_oracle:
        print("[mac] running oracle", file=sys.stderr, flush=True)
        if args.teacher_forced:
            o_dec, o_lat, o_tok = eval_teacher_forced(oracle_logits_all)
        else:
            o_dec, o_lat, o_tok = eval_free_gen_oracle()  # fast native incremental
        oracle_res = aggregate_recall("oracle_mac", samples, o_dec, o_lat, o_tok)
        print(f"[mac] oracle recall = {oracle_res.recall:.3f}", file=sys.stderr)

    # ---------- KV-memory accounting (bounded S5 engine) ----------
    T_max = max(seq_lens)
    exact_for_mem = full_attn_idx  # S5: full-attn layers kept exact / compressed
    mem_s5 = kv_memory_report(
        text_model, sink_size=args.sink_size, window_size=args.window_size,
        seq_len=T_max, exact_layer_indices=exact_for_mem,
        compress_full_bits_per_token_per_head=(
            kl_bits_per_head if args.compress_full_attn else None),
    )
    # Baselines for the savings story:
    mem_naive = kv_memory_report(   # all layers O(T), no bound, no compress
        text_model, sink_size=T_max, window_size=0, seq_len=T_max,
        exact_layer_indices=list(range(n_layers)))
    print(f"[mac] KV resident @T={T_max}: S5={mem_s5['total_resident_mb']} MB "
          f"(growth {mem_s5['per_token_growth_kb']} KB/tok); "
          f"naive-full={mem_naive['total_resident_mb']} MB", file=sys.stderr)

    # ---------- Throughput ----------
    def _tps(lats, toks):
        tot_t = sum(lats)
        tot_n = sum(toks)
        return {
            "tokens": tot_n, "wall_seconds": round(tot_t, 3),
            "tokens_per_second": round(tot_n / tot_t, 4) if tot_t > 0 else None,
            "mean_latency_per_sample_s": round(tot_t / max(len(lats), 1), 3),
        }
    cross_tps = _tps(cross_lat, cross_tok)
    cross_tps["eval_mode"] = eval_mode
    cross_tps["restored_forwards_per_sample"] = (
        1 if args.teacher_forced else 1)
    cross_tps["incremental_decode"] = (not args.teacher_forced)
    cross_tps["stage_timings"] = (
        getattr(eval_free_gen_cross, "stage_rows", [])
        if not args.teacher_forced else []
    )
    print(f"[mac] cross-model throughput ({eval_mode}): "
          f"{cross_tps['tokens_per_second']} tok/s "
          f"({cross_tps['tokens']} tok / {cross_tps['wall_seconds']} s, "
          f"{cross_tps['mean_latency_per_sample_s']} s/sample)", file=sys.stderr)

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
            "chat_template": bool(args.chat_template),
            "eval_mode": eval_mode,
            "teacher_forced": bool(args.teacher_forced),
            "s5_exact_full_attn": bool(args.s5_exact_full_attn),
            "identity_restore": bool(args.identity_restore),
            "compress_full_attn": bool(args.compress_full_attn),
            "kl_lattice": args.kl_lattice if args.compress_full_attn else None,
            "kl_q_range": args.kl_q_range if args.compress_full_attn else None,
            "kl_bits_per_token_per_head": kl_bits_per_head,
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
        "memory": {
            "s5": mem_s5,
            "naive_full_kv": {
                "total_resident_mb": mem_naive["total_resident_mb"],
                "per_token_growth_kb": mem_naive["per_token_growth_kb"],
            },
            "savings_vs_naive_pct": round(
                100 * (1 - mem_s5["total_resident_bytes"]
                       / max(mem_naive["total_resident_bytes"], 1)), 1),
        },
        "throughput": {"k3_cross_model": cross_tps},
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
