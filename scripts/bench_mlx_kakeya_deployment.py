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

Both paths run through mlx_lm's **own** ``generate_step`` engine (chunked
prefill + pipelined async decode) — only the KV cache differs. This is the
apples-to-apples test: from first principles Kakeya is just MLX + a tighter
cache, so it should be *faster + lighter* than vanilla, never slower. If it is
slower, the cache implementation has a bug. For each context length L, on the
SAME model + SAME engine:

  * **Vanilla** — the model's native cache (``make_prompt_cache`` →
    ``model.make_cache()``: full ``KVCache`` for the 5 global layers,
    ``RotatingKVCache(sliding_window)`` for the 25 sliding layers). The 5
    global layers' KV grows with L; per-token attention there is over all L
    keys.
  * **Kakeya** — sink+window bounded cache (``make_sink_window_cache``) for
    every layer: persistent KV is O(sink+window) and per-token attention is
    over the bounded window for *all* layers (incl. the global ones). (Note:
    this is the bounded-KV / StreamingLLM-class fast path — long-range *recall*
    needs the separate K/V-Restoration; this benchmark measures the throughput
    + memory envelope.)

Reports, per L: time-to-first-token (incl. prefill), decode tok/s, resident KV
bytes, peak memory, and the kakeya/vanilla decode-speedup + KV-shrink ratios.

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


def _resident_kv_bytes(cache: list) -> int:
    """Actual resident K+V bytes across a per-layer cache list.

    Uses each tensor's real ``.nbytes`` (the allocated buffer), so it is
    correct and uniform across *every* cache type — full ``KVCache``
    (grows with context), ``RotatingKVCache`` (capped ring buffer) and our
    ``SinkWindowKVCache`` (sink+window). This is the honest comparison: it
    reflects what is physically held, not the unbounded global ``offset``.
    """
    total = 0
    for c in cache:
        for name in ("keys", "values"):
            t = getattr(c, name, None)
            nb = getattr(t, "nbytes", None) if t is not None else None
            if nb is not None:
                total += int(nb)
    return total


def _run(mx, generate_step, model, prompt_ids: List[int], gen_tokens: int,
         cache) -> Dict[str, Any]:
    """Prefill + greedy-decode ``gen_tokens`` using mlx_lm's *native*
    ``generate_step`` (chunked prefill + pipelined async decode), swapping
    only the KV cache. This isolates the cache's effect on the native engine.
    Returns timing + memory metrics.
    """
    _reset_peak_memory(mx)
    prompt = mx.array(prompt_ids)
    gen = generate_step(prompt, model, max_tokens=gen_tokens, prompt_cache=cache)
    t0 = time.perf_counter()
    first = next(gen)            # prefill + first decoded token
    _ = first[0]                 # already an int (generate_step yields y.item())
    ttft_s = time.perf_counter() - t0
    n = 0
    t1 = time.perf_counter()
    for _tok, _lp in gen:
        n += 1
    decode_s = time.perf_counter() - t1
    return {
        "ttft_s": round(ttft_s, 4),                 # time to first token (incl. prefill)
        "decode_s": round(decode_s, 4),
        "decode_tokens": n,                          # tokens after the first
        "decode_tokens_per_s": round(n / decode_s, 3) if decode_s > 0 and n > 0 else None,
        "kv_bytes": int(_resident_kv_bytes(cache)),
        "peak_memory_bytes": _peak_memory_bytes(mx),
    }


def main() -> int:
    args = parse_args()

    import mlx.core as mx           # type: ignore
    import mlx_lm                   # type: ignore
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore
    from mlx_lm.generate import generate_step  # type: ignore
    from inference_engine.backends.mlx.cache import make_sink_window_cache

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

    def make_vanilla_cache():
        return make_prompt_cache(model)

    def make_kakeya_cache():
        return make_sink_window_cache(
            model, sink_size=args.sink_size, window_size=args.window_size)

    # Warm up MLX kernel compilation for BOTH cache paths before timing.
    # MLX compiles graphs lazily on first use; without this, whichever path
    # runs first absorbs the (large, one-off) compile cost and looks slower.
    # The 1-token decode graph compiled here is shared across all context
    # lengths, so decode tok/s is measured fairly for both caches.
    warm_prompt = make_prompt(64)
    for label, mk in (("vanilla", make_vanilla_cache), ("kakeya", make_kakeya_cache)):
        if (label == "vanilla" and args.skip_vanilla) or (
            label == "kakeya" and args.skip_kakeya):
            continue
        print(f"[bench] warmup ({label}) ...", file=sys.stderr, flush=True)
        try:
            wc = mk()
            for _ in generate_step(mx.array(warm_prompt), model,
                                   max_tokens=8, prompt_cache=wc):
                pass
            wc = None
            mx.clear_cache()
        except Exception as e:
            print(f"[bench] warmup ({label}) failed: {e}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for L in ctx_lengths:
        prompt_ids = make_prompt(L)
        row: Dict[str, Any] = {"context_length": L}
        if not args.skip_vanilla:
            print(f"[bench] L={L}: vanilla (native make_prompt_cache) ...",
                  file=sys.stderr, flush=True)
            try:
                vcache = make_vanilla_cache()
                row["vanilla"] = _run(
                    mx, generate_step, model, prompt_ids, args.gen_tokens, vcache)
            except Exception as e:  # OOM or unsupported → record and continue
                row["vanilla"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[bench] L={L}: vanilla path failed: {e}", file=sys.stderr)
            finally:
                # Free the (possibly large) vanilla KV before measuring kakeya,
                # so its peak-memory reading isn't inflated by leftover state.
                vcache = None
                mx.clear_cache()

        if not args.skip_kakeya:
            print(f"[bench] L={L}: Kakeya sink+window ...", file=sys.stderr, flush=True)
            try:
                kcache = make_kakeya_cache()
                row["kakeya"] = _run(
                    mx, generate_step, model, prompt_ids, args.gen_tokens, kcache)
            except Exception as e:
                row["kakeya"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[bench] L={L}: kakeya path failed: {e}", file=sys.stderr)
            finally:
                kcache = None
                mx.clear_cache()

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
        if v_ok:
            print(f"[bench] L={L}: vanilla {v['decode_tokens_per_s']} tok/s "
                  f"(ttft {v['ttft_s']}s, KV {v['kv_bytes']/1e6:.2f} MB, "
                  f"peak {v['peak_memory_bytes']/1e9:.2f} GB)", file=sys.stderr)
        if k_ok:
            print(f"[bench] L={L}: kakeya  {k['decode_tokens_per_s']} tok/s "
                  f"(ttft {k['ttft_s']}s, KV {k['kv_bytes']/1e6:.2f} MB, "
                  f"peak {k['peak_memory_bytes']/1e9:.2f} GB)", file=sys.stderr)
        if k_ok and v_ok:
            r = row["kakeya_vs_vanilla"]
            print(f"[bench] L={L}: kakeya vs vanilla -> decode {r['decode_speedup_x']}x, "
                  f"KV {r['kv_bytes_ratio_x']}x smaller", file=sys.stderr)
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


if __name__ == "__main__":
    sys.exit(main())
