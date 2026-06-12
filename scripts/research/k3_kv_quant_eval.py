"""KV-quantization rate–distortion shoot-out: affine (mlx-native) vs KakeyaLattice.

Decision input for "is an MLX port of the KakeyaLattice codec worth it?"
(docs/mlx-port-lessons.md, K2 track): KL's value proposition over plain
affine quantization is better rate–distortion at equal bits. This eval
measures exactly that, on the only K/V that matter for the S5 memory
story — the 5 full-attention layers' exact own K/V (the 20 KB/token
linear term) — at ctx280 scale, with REAL recall as the end metric.

Arms (same captured K/V per sample, identical injection machinery):

  identity   — lossless round trip (machinery control; recall must match
               the S5 baseline)
  affine8    — mx.quantize/dequantize, 8-bit, group 64 (the storage
               format of mlx_lm's QuantizedKVCache; ~8.5 bits/value)
  affine4    — same, 4-bit (~4.5 bits/value)
  kl-d4      — KakeyaLattice D4 round trip (torch codec, eval-time only)
  kl-e8      — KakeyaLattice E8 round trip

Per arm and sample: bits/value (measured), energy-weighted rel_mse of
the lossy full-attn K/V vs the originals, then a REAL incremental
restored decode (lossy K/V injected at the evicted positions, sliding
layers native window-bounded) → NIAH recall.

Scope note: this measures STORAGE fidelity at matched rate. Decode in
every arm runs on bf16-materialised K/V, so per-arm decode timing is
not a codec-throughput claim (runtime decompression cost is a separate
question that only matters for codecs that win here).

Verdict rule printed at the end: KL justifies an MLX port only if, at
bits <= affine4's rate, it achieves BOTH lower full-attn rel_mse AND
recall >= affine4's. Otherwise native affine quantization wins by
default (zero porting cost, kernel-fused dequant).

Run on the Mac via bridge preset ``k3-kv-quant-eval`` or directly:

  PYTHONPATH=.:sdks/python python3 scripts/research/k3_kv_quant_eval.py \
      --verifier-path <mlx-4bit-dir> --n-samples 5 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", default="models/gemma-4-26B-A4B-it-mlx-4bit")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--haystack-min-lines", type=int, default=238)
    ap.add_argument("--haystack-max-lines", type=int, default=322)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prefill-chunk-size", type=int, default=512)
    ap.add_argument("--kl-q-range", type=int, default=38)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--output", default="results/research/k3_kv_quant_eval.json")
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    import torch

    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        capture_own_kv,
        mlx_full_attention_layer_indices,
        resolve_mlx_text_model,
        restored_incremental_generate,
        restored_prefill_cache,
    )
    from inference_engine.v04 import NIAHSample, aggregate_recall, make_niah_dataset
    from inference_engine.v04.kv_compressor import make_default_compressor
    from inference_engine.v04.kv_merge import compute_evicted_positions
    from scripts.research.k3_dflash_mlx_bridge import mx_to_torch, torch_to_mx

    print(f"[kvq] loading MLX verifier {args.verifier_path}", file=sys.stderr)
    mlx_model, tokenizer = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    full_attn_idx = mlx_full_attention_layer_indices(text_model)
    print(f"[kvq] full-attn layers: {full_attn_idx}", file=sys.stderr)

    # ---------- arms ----------
    GROUP = 64

    def affine_roundtrip(bits: int):
        def fn(k: Any, v: Any) -> Tuple[Any, Any, float]:
            outs = []
            for a in (k, v):
                shp = a.shape
                flat = a.reshape(-1, shp[-1]).astype(mx.float16)
                wq, scales, biases = mx.quantize(flat, group_size=GROUP, bits=bits)
                deq = mx.dequantize(
                    wq, scales, biases, group_size=GROUP, bits=bits)
                outs.append(deq.reshape(shp).astype(a.dtype))
            # bits/value: payload + fp16 scale & bias per group.
            rate = bits + 2 * 16.0 / GROUP
            return outs[0], outs[1], rate
        return fn

    def kl_roundtrip(lattice: str):
        comps: Dict[int, Any] = {}

        def fn(k: Any, v: Any, *, head_dim: int, layer: int) -> Tuple[Any, Any, float]:
            comp = comps.get(layer)
            if comp is None:
                comp = make_default_compressor(
                    head_dim=head_dim, device=torch.device("cpu"),
                    prefer_kakeya=True, lattice=lattice,
                    q_range=args.kl_q_range)
                comps[layer] = comp
            kt = mx_to_torch(k, dtype=torch.float32, device="cpu").transpose(1, 2).contiguous()
            vt = mx_to_torch(v, dtype=torch.float32, device="cpu").transpose(1, 2).contiguous()
            T = kt.shape[-2]
            pos = torch.arange(T)
            comp.compress(kt, vt, pos)
            kh, vh = comp.decompress(pos)
            comp.evict(pos)
            codec = getattr(comp, "_codec", None)
            bits_head = float(getattr(codec, "bits_per_token_per_head", 0.0) or 0.0)
            rate = bits_head / head_dim if bits_head else float("nan")
            kh = torch_to_mx(kh.transpose(1, 2).contiguous())
            vh = torch_to_mx(vh.transpose(1, 2).contiguous())
            return kh, vh, rate
        return fn

    ARMS: List[Tuple[str, Any]] = [
        ("identity", None),
        ("affine8", affine_roundtrip(8)),
        ("affine4", affine_roundtrip(4)),
        ("kl-d4", kl_roundtrip("D4")),
        ("kl-e8", kl_roundtrip("E8")),
    ]

    # ---------- dataset / prompts (mirrors the integrated harness) ----------
    samples: List[NIAHSample] = make_niah_dataset(
        n_samples=args.n_samples,
        haystack_min_lines=args.haystack_min_lines,
        haystack_max_lines=args.haystack_max_lines,
        seed=args.seed,
    )

    def encode(prompt_text: str) -> List[int]:
        prompt_text = prompt_text.replace(
            "and does not contain the answer.", "and is unrelated filler.")
        prompt_text += ("\n\nReturn only the secret code in PREFIX-NNNN "
                        "format. Do not explain, reason, or add any other text.")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        try:
            marker = tokenizer.encode(
                "<|channel>content\n<channel|>", add_special_tokens=False)
        except TypeError:
            marker = tokenizer.encode("<|channel>content\n<channel|>")
        if hasattr(marker, "tolist"):
            marker = marker.tolist()
        ids.extend(list(marker))
        return ids

    sample_ids = [encode(s.prompt_text) for s in samples]
    seq_lens = [len(t) for t in sample_ids]
    end_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        end_ids.add(int(tokenizer.eos_token_id))
    try:
        eot = tokenizer.encode("<turn|>", add_special_tokens=False)
    except TypeError:
        eot = tokenizer.encode("<turn|>")
    if hasattr(eot, "tolist"):
        eot = eot.tolist()
    if len(eot) == 1:
        end_ids.add(int(eot[0]))
    print(f"[kvq] {len(samples)} samples, prompt len {min(seq_lens)}..{max(seq_lens)}",
          file=sys.stderr)

    def rel_mse(orig_k, orig_v, lossy_k, lossy_v) -> float:
        num = den = 0.0
        for o, l in ((orig_k, lossy_k), (orig_v, lossy_v)):
            o32 = o.astype(mx.float32)
            d = l.astype(mx.float32) - o32
            num += float(mx.sum(d * d))
            den += float(mx.sum(o32 * o32))
        return num / max(den, 1e-12)

    # ---------- per-sample capture, per-arm roundtrip + decode ----------
    per_arm: Dict[str, Dict[str, Any]] = {
        name: {"rel_mse": [], "bits": [], "decoded": [], "lat": [], "tok": []}
        for name, _ in ARMS
    }
    oracle_decoded: List[str] = []
    oracle_lat: List[float] = []
    oracle_tok: List[int] = []

    for i, pid in enumerate(sample_ids):
        T = len(pid)
        evicted = compute_evicted_positions(T, args.sink_size, args.window_size)
        t0 = time.perf_counter()
        own = capture_own_kv(mlx_model, pid)
        print(f"[kvq] s{i}: T={T} capture {time.perf_counter()-t0:.1f}s",
              file=sys.stderr)

        if not args.skip_oracle:
            from mlx_lm.models.cache import make_prompt_cache  # noqa: F401
            cache = (getattr(mlx_model, "make_cache", lambda: None)())
            last = None
            for s in range(0, T, args.prefill_chunk_size):
                part = pid[s:s + args.prefill_chunk_size]
                last = mlx_model(mx.array([part]), cache=cache)
                mx.eval(last)
            t0 = time.perf_counter()
            gen = restored_incremental_generate(
                mlx_model, cache, last[0, -1],
                max_tokens=args.max_new_tokens, eos_ids=end_ids)
            oracle_lat.append(time.perf_counter() - t0)
            oracle_decoded.append(tokenizer.decode(gen))
            oracle_tok.append(len(gen))

        for name, roundtrip in ARMS:
            rk: Dict[int, Any] = {}
            rv: Dict[int, Any] = {}
            mses: List[float] = []
            rates: List[float] = []
            for li in full_attn_idx:
                k, v = own[li]
                if roundtrip is None:
                    lk, lv, rate = k, v, 16.0
                elif name.startswith("kl"):
                    lk, lv, rate = roundtrip(
                        k, v, head_dim=int(k.shape[-1]), layer=li)
                else:
                    lk, lv, rate = roundtrip(k, v)
                rk[li], rv[li] = lk, lv
                mses.append(rel_mse(k, v, lk, lv))
                rates.append(rate)
            arm_mse = sum(mses) / len(mses)
            t0 = time.perf_counter()
            cache, first = restored_prefill_cache(
                mlx_model, pid,
                restored_k_per_layer=rk, restored_v_per_layer=rv,
                evicted_positions=evicted,
                prefill_chunk_size=args.prefill_chunk_size)
            gen = restored_incremental_generate(
                mlx_model, cache, first,
                max_tokens=args.max_new_tokens, eos_ids=end_ids)
            elapsed = time.perf_counter() - t0
            per_arm[name]["rel_mse"].append(arm_mse)
            per_arm[name]["bits"].append(sum(rates) / len(rates))
            per_arm[name]["decoded"].append(tokenizer.decode(gen))
            per_arm[name]["lat"].append(elapsed)
            per_arm[name]["tok"].append(len(gen))
            print(f"[kvq] s{i} {name}: bits/val={sum(rates)/len(rates):.2f} "
                  f"rel_mse={arm_mse:.5f} -> {per_arm[name]['decoded'][-1][:32]!r}",
                  file=sys.stderr)

    # ---------- aggregate ----------
    results: Dict[str, Any] = {}
    for name, _ in ARMS:
        a = per_arm[name]
        rec = aggregate_recall(
            f"kvq_{name}", samples, a["decoded"], a["lat"], a["tok"])
        results[name] = {
            "recall": rec.recall,
            "samples_correct": rec.samples_correct,
            "bits_per_value_mean": round(sum(a["bits"]) / len(a["bits"]), 3),
            "full_attn_rel_mse_mean": round(
                sum(a["rel_mse"]) / len(a["rel_mse"]), 6),
            "per_sample_decoded": [d[:48] for d in a["decoded"]],
        }
    oracle_recall = None
    if oracle_decoded:
        orec = aggregate_recall(
            "kvq_oracle", samples, oracle_decoded, oracle_lat, oracle_tok)
        oracle_recall = orec.recall

    # ---------- verdict ----------
    aff4 = results["affine4"]
    verdicts = {}
    for klname in ("kl-d4", "kl-e8"):
        kl = results[klname]
        verdicts[klname] = {
            "bits_le_affine4": kl["bits_per_value_mean"] <= aff4["bits_per_value_mean"] + 0.25,
            "rel_mse_better": kl["full_attn_rel_mse_mean"] < aff4["full_attn_rel_mse_mean"],
            "recall_ge_affine4": kl["recall"] >= aff4["recall"],
        }
        verdicts[klname]["mlx_port_justified"] = all(verdicts[klname].values())

    report = {
        "kind": "k3_kv_quant_eval",
        "schema_version": 1,
        "config": vars(args),
        "full_attention_layers": full_attn_idx,
        "prompt_token_lens": seq_lens,
        "oracle_recall": oracle_recall,
        "results": results,
        "verdict": verdicts,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[kvq] DONE -> {out}", file=sys.stderr)
    for name in results:
        r = results[name]
        print(f"[kvq]   {name:9s} bits={r['bits_per_value_mean']:6.2f} "
              f"rel_mse={r['full_attn_rel_mse_mean']:.5f} "
              f"recall={r['recall']:.2f}", file=sys.stderr)
    print(f"[kvq] verdict: {json.dumps(verdicts)}", file=sys.stderr)
    # Machinery sanity: identity arm must not lose recall.
    if results["identity"]["recall"] < (oracle_recall or 1.0):
        print("[kvq] WARNING: identity arm below oracle — injection "
              "machinery issue, codec comparisons unreliable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
