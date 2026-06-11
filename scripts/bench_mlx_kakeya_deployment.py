"""Mac (MLX) high-performance deployment benchmark for the Kakeya engine.

Goal: demonstrate, on Apple Silicon (M-series), that the Kakeya **sink+window
bounded-KV** inference path delivers *sustained high throughput at constant
memory* as context grows — vs a vanilla full-KV baseline whose KV cache (and
per-token attention cost) grow with context.

This is the local-deployment benchmark for the **Gemma 4 26B-A4B** verifier on
a 24 GB M4: 4-bit weights are ~16 GB resident, leaving ~8 GB for KV + activations.
With a vanilla full-KV cache, per-token attention cost and KV memory grow with
context, so decode tok/s collapses and peak memory climbs toward the 24 GB
ceiling at long context. The Kakeya sink+window cache bounds both: persistent KV
is O(sink+window) and per-token attention is over the bounded window, so decode
throughput and peak memory stay ~flat as context grows. (Long-range *recall*
needs the separate K/V-Restoration path; this benchmark measures the throughput
+ memory envelope.)

For each context length L it runs, on the SAME model:

  * **Kakeya** — sink+window bounded cache (``make_sink_window_cache``):
    persistent KV is O(sink+window); per-token attention is over the bounded
    window. (Note: this is the bounded-KV / StreamingLLM-class fast path —
    long-range *recall* needs the separate, heavier K/V-Restoration; this
    benchmark measures the throughput + memory envelope.)
  * **Vanilla** — full KV cache (``make_prompt_cache``): KV grows with L,
    per-token attention is over all L keys.

Reports, per L: prefill time, decode tok/s, persistent KV bytes, peak memory.

Run on the Mac (Apple Silicon):

    source .venv-mac/bin/activate    # or your MLX venv
    PYTHONPATH=.:sdks/python python3 scripts/bench_mlx_kakeya_deployment.py \
        --model-id models/gemma-4-26B-A4B-it-mlx-4bit \
        --context-lengths 512,2048,8192 \
        --gen-tokens 64 --sink-size 4 --window-size 64 \
        --output results/platform-tests/bench_mlx_kakeya_deployment.json

The bounded-KV advantage grows with context: push --context-lengths higher
(e.g. 16384,32768) to widen the gap, as long as the vanilla full-KV prefill
still fits in memory. Use --skip-vanilla when vanilla would OOM at long context
so the Kakeya path can still be measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="models/gemma-4-26B-A4B-it-mlx-4bit",
                    help="MLX 4-bit model id or local path (default: the "
                         "Gemma 4 26B-A4B 4-bit verifier).")
    ap.add_argument("--context-lengths", default="512,2048,8192",
                    help="Comma-separated prompt token lengths to sweep.")
    ap.add_argument("--skip-kakeya", action="store_true",
                    help="Skip the sink+window path (measure vanilla only).")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--skip-vanilla", action="store_true",
                    help="Skip the full-KV baseline (e.g. when it would OOM).")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def _peak_memory_bytes(mx) -> int:
    for getter in ("get_peak_memory",):
        fn = getattr(mx, getter, None)
        if fn is not None:
            try:
                return int(fn())
            except Exception:
                pass
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "get_peak_memory"):
        try:
            return int(metal.get_peak_memory())
        except Exception:
            pass
    return -1


def _reset_peak_memory(mx) -> None:
    for name in ("reset_peak_memory",):
        fn = getattr(mx, name, None)
        if fn is not None:
            try:
                fn(); return
            except Exception:
                pass
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        try:
            metal.reset_peak_memory()
        except Exception:
            pass


def _decode(mx, model, cache, prompt_ids: List[int], gen_tokens: int,
            kv_bytes_fn) -> Dict[str, Any]:
    """Prefill prompt + greedy-decode gen_tokens with the given cache.
    Returns timing + memory metrics."""
    _reset_peak_memory(mx)
    ids = mx.array([prompt_ids])
    t0 = time.perf_counter()
    out = model(ids, cache=cache)
    mx.eval(out)
    prefill_s = time.perf_counter() - t0
    tok = int(mx.argmax(out[0, -1]).item())
    n = 1
    t1 = time.perf_counter()
    for _ in range(gen_tokens - 1):
        out = model(mx.array([[tok]]), cache=cache)
        mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        n += 1
    gen_s = time.perf_counter() - t1
    return {
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(gen_s, 4),
        "decode_tokens": n - 1,
        "decode_tokens_per_s": round((n - 1) / gen_s, 3) if gen_s > 0 else None,
        "kv_bytes": int(kv_bytes_fn(cache)),
        "peak_memory_bytes": _peak_memory_bytes(mx),
    }


def main() -> int:
    args = parse_args()

    import mlx.core as mx           # type: ignore
    import mlx_lm                   # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore
    from inference_engine.backends.mlx.cache import (
        make_sink_window_cache, total_kv_bytes,
    )

    ctx_lengths = [int(x) for x in args.context_lengths.split(",") if x.strip()]
    print(f"[bench] loading MLX model {args.model_id}", file=sys.stderr, flush=True)
    model, tokenizer = mlx_lm.load(args.model_id)

    # A deterministic synthetic prompt of a given length (content is
    # irrelevant for the throughput/memory envelope; we use a fixed filler
    # token so prefill length == L).
    bos = getattr(tokenizer, "bos_token_id", None)
    filler = tokenizer.encode("the ")
    filler_tok = filler[-1] if filler else 1

    def make_prompt(L: int) -> List[int]:
        ids = ([bos] if bos is not None else []) + [filler_tok] * (L - (1 if bos is not None else 0))
        return ids[:L] if len(ids) >= L else ids + [filler_tok] * (L - len(ids))

    rows: List[Dict[str, Any]] = []
    for L in ctx_lengths:
        prompt_ids = make_prompt(L)
        row: Dict[str, Any] = {"context_length": L}
        if not args.skip_kakeya:
            print(f"[bench] L={L}: Kakeya sink+window ...", file=sys.stderr, flush=True)
            try:
                kcache = make_sink_window_cache(
                    model, sink_size=args.sink_size, window_size=args.window_size)
                row["kakeya"] = _decode(
                    mx, model, kcache, prompt_ids, args.gen_tokens, total_kv_bytes)
            except Exception as e:
                row["kakeya"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[bench] L={L}: kakeya path failed: {e}", file=sys.stderr)

        if not args.skip_vanilla:
            print(f"[bench] L={L}: vanilla full-KV ...", file=sys.stderr, flush=True)
            try:
                vcache = make_prompt_cache(model)
                row["vanilla"] = _decode(
                    mx, model, vcache, prompt_ids, args.gen_tokens,
                    lambda c: _full_cache_bytes(c))
            except Exception as e:  # OOM or unsupported → record and continue
                row["vanilla"] = {"error": f"{type(e).__name__}: {e}"}

        k = row.get("kakeya", {})
        v = row.get("vanilla", {})
        k_ok = isinstance(k, dict) and "decode_tokens_per_s" in k
        v_ok = isinstance(v, dict) and "decode_tokens_per_s" in v
        if k_ok and v_ok:
            sp = (k["decode_tokens_per_s"] or 0) / max(v["decode_tokens_per_s"] or 1e-9, 1e-9)
            row["kakeya_vs_vanilla"] = {
                "decode_speedup_x": round(sp, 3),
                "kv_bytes_ratio_x": round(v.get("kv_bytes", 0) / max(k.get("kv_bytes", 1), 1), 1),
            }
        if k_ok:
            print(f"[bench] L={L}: kakeya {k['decode_tokens_per_s']} tok/s "
                  f"(prefill {k['prefill_s']}s, KV {k['kv_bytes']/1e6:.2f} MB, "
                  f"peak {k['peak_memory_bytes']/1e9:.2f} GB)", file=sys.stderr)
        if v_ok:
            print(f"[bench] L={L}: vanilla {v['decode_tokens_per_s']} tok/s "
                  f"(prefill {v['prefill_s']}s, KV {v['kv_bytes']/1e6:.2f} MB, "
                  f"peak {v['peak_memory_bytes']/1e9:.2f} GB)", file=sys.stderr)
        rows.append(row)

    report = {
        "kind": "mlx_kakeya_deployment_benchmark",
        "config": {
            "model_id": args.model_id,
            "context_lengths": ctx_lengths,
            "gen_tokens": args.gen_tokens,
            "sink_size": args.sink_size,
            "window_size": args.window_size,
        },
        "env": {"mlx_version": getattr(mx, "__version__", "?")},
        "results": rows,
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/platform-tests/bench_mlx_kakeya_deployment_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[bench] wrote {out_path}", file=sys.stderr)
    return 0


def _full_cache_bytes(cache: list) -> int:
    """Persistent KV bytes for an mlx_lm full-KV prompt cache.

    Per layer: K and V are ``[B, n_kv, S, head_dim]`` with logical length
    ``offset`` along the seq axis. Bytes ≈ 2 (K+V) × B×n_kv×offset×head_dim ×
    itemsize (2 for fp16/bf16).
    """
    total = 0
    for c in cache:
        off = int(getattr(c, "offset", 0) or 0)
        k = getattr(c, "keys", None)
        if k is None or off <= 0:
            continue
        shp = tuple(k.shape)  # [B, n_kv, S_buf, head_dim]
        if len(shp) != 4:
            continue
        b, n_kv, s_buf, hd = shp
        # Resident length = the actual stored buffer, capped (RotatingKVCache
        # keeps <= max_size even though .offset is the global position).
        seq = min(off, int(s_buf))
        itemsize = 2  # fp16/bf16 KV
        total += 2 * b * n_kv * seq * hd * itemsize
    return total


if __name__ == "__main__":
    sys.exit(main())
