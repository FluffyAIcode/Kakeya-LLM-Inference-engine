"""verify(L) calibration sweep (Mac/MLX) — measure the verifier's per-block
forward cost vs block size L, to quantify the speculative-decoding kernel-dedup
headroom directly.

Definitions (all empirical, measured on-device):

* ``verify(L)`` — wall time of ONE decode forward processing L query tokens
  against a fixed prefilled cache at offset ``context_len`` (exactly what fused
  spec-decode's ``forward_block`` does per block). Measured as the median over
  ``reps`` repetitions; the cache is trimmed back by L after each rep so every
  measurement runs at the same cache offset.
* **measured kernel-dedup headroom** ``= L * verify(1) / verify(L)``. =L means
  verify(L) is as cheap as a single token (ideal spec-decode ceiling: a block of
  L verified for the price of 1). =1 means no batching benefit (spec-decode
  cannot help). This is the "real headroom" the sweep measures.
* **expert-union estimate** (MoE, best-effort) — across the L query tokens, the
  router activates a *set* of experts per layer; ``|union of top-k experts over
  the L tokens| / (L * top_k)`` is the theoretical FFN dedup factor. The
  expert-union-implied headroom for the MoE-FFN portion is its reciprocal. This
  is the analytical curve to compare the measured curve against.

Runs only on Apple Silicon (MLX). Invoked via the Mac bridge preset
``verify-l-sweep``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _parse_l_list(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("empty --l-list")
    return out


@contextlib.contextmanager
def _router_capture(text_model, sink: Dict[int, List[Any]]):
    """Patch the Gemma-4 MoE Router.__call__ to record top_k_indices per layer.
    Best-effort: if the model has no Router, this is a no-op."""
    router = None
    for layer in text_model.layers:
        r = getattr(layer, "router", None)
        if r is not None:
            router = r
            break
    if router is None:
        yield False
        return
    cls = type(router)
    orig = cls.__call__

    def dispatch(self, x):
        out = orig(self, x)
        rec = getattr(self, "_vl_sink", None)
        if rec is not None:
            idx = out[0] if isinstance(out, tuple) else out
            rec.append(idx)
        return out

    cls.__call__ = dispatch  # type: ignore[assignment]
    try:
        yield True
    finally:
        cls.__call__ = orig  # type: ignore[assignment]
        for layer in text_model.layers:
            r = getattr(layer, "router", None)
            if r is not None and hasattr(r, "_vl_sink"):
                delattr(r, "_vl_sink")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", required=True)
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--l-list", type=_parse_l_list, default="1,2,4,8,16")
    ap.add_argument("--prefill-chunk-size", type=int, default=512)
    ap.add_argument("--output", default="results/research/verify_l_sweep.json")
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # type: ignore

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (  # type: ignore
        resolve_mlx_text_model, per_layer_kv_geometry,
    )

    print(f"[vl] loading {args.verifier_path}", file=sys.stderr, flush=True)
    model, _tok = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(model)
    n_layers = len(text_model.layers)
    top_k = int(getattr(getattr(text_model, "config", object()), "top_k_experts", 0) or 0)
    n_experts = int(getattr(getattr(text_model, "config", object()), "num_experts", 0) or 0)

    # Vocab size for varied (non-degenerate) token ids.
    try:
        vocab = int(text_model.embed_tokens.weight.shape[0])
    except Exception:
        vocab = 256000
    C = int(args.context_len)
    ctx_ids = [(i * 1315423911) % vocab for i in range(C)]

    def fresh_cache():
        cache = make_prompt_cache(model)
        step = max(int(args.prefill_chunk_size), 1)
        for s in range(0, C, step):
            part = ctx_ids[s:s + step]
            out = model(mx.array([part]), cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    print(f"[vl] prefilling context_len={C} (chunk={args.prefill_chunk_size})",
          file=sys.stderr, flush=True)
    cache = fresh_cache()

    def block_ids(L: int) -> List[int]:
        return [(C + j) * 2654435761 % vocab for j in range(L)]

    def timed_verify(L: int) -> float:
        toks = mx.array([block_ids(L)])
        t0 = time.perf_counter()
        out = model(toks, cache=cache)
        mx.eval(out)
        dt = time.perf_counter() - t0
        trim_prompt_cache(cache, L)   # roll back to offset C
        return dt

    # Warmup the exact shapes we will time (kernel compilation off the clock).
    for L in sorted(set(args.l_list)):
        for _ in range(2):
            timed_verify(L)

    rows: List[Dict[str, Any]] = []
    for L in args.l_list:
        samples = [timed_verify(L) for _ in range(args.reps)]
        med = statistics.median(samples)

        # Expert-union (best-effort, one extra patched forward).
        union_ratio = None
        try:
            sink: List[Any] = []
            with _router_capture(text_model, {}) as ok:
                if ok:
                    for layer in text_model.layers:
                        r = getattr(layer, "router", None)
                        if r is not None:
                            r._vl_sink = sink
                    _ = model(mx.array([block_ids(L)]), cache=cache)
                    mx.eval([])
                    trim_prompt_cache(cache, L)
            if sink and top_k > 0:
                ratios = []
                for idx in sink:
                    arr = idx.tolist() if hasattr(idx, "tolist") else idx
                    flat = arr[0] if (arr and isinstance(arr[0], list) and arr[0]
                                      and isinstance(arr[0][0], list)) else arr
                    uniq = set()
                    for pos in flat:
                        for e in (pos if isinstance(pos, list) else [pos]):
                            uniq.add(int(e))
                    denom = max(L * top_k, 1)
                    ratios.append(min(len(uniq), denom) / denom)
                if ratios:
                    union_ratio = round(sum(ratios) / len(ratios), 4)
        except Exception as exc:  # pragma: no cover - device-only
            print(f"[vl] expert-union skipped for L={L}: {exc}", file=sys.stderr)

        rows.append({
            "L": L,
            "verify_s_median": round(med, 6),
            "verify_s_samples": [round(s, 6) for s in samples],
            "expert_union_ratio": union_ratio,
        })
        print(f"[vl] L={L}: verify={med*1e3:.2f} ms  union_ratio={union_ratio}",
              file=sys.stderr, flush=True)

    base = next((r["verify_s_median"] for r in rows if r["L"] == 1), None)
    for r in rows:
        if base and r["verify_s_median"] > 0:
            r["measured_headroom"] = round(r["L"] * base / r["verify_s_median"], 3)
        else:
            r["measured_headroom"] = None
        if r["expert_union_ratio"]:
            r["expert_union_headroom"] = round(1.0 / r["expert_union_ratio"], 3)
        else:
            r["expert_union_headroom"] = None

    report = {
        "schema_version": 1,
        "kind": "verify_l_sweep_mac",
        "config": {
            "verifier_path": args.verifier_path,
            "context_len": C,
            "reps": args.reps,
            "l_list": args.l_list,
            "n_layers": n_layers,
            "top_k_experts": top_k,
            "num_experts": n_experts,
            "vocab": vocab,
        },
        "rows": rows,
        "note": ("measured_headroom = L*verify(1)/verify(L) (kernel-dedup real "
                 "margin); expert_union_headroom = 1/(|union experts|/(L*top_k)) "
                 "(MoE-FFN theoretical dedup bound, router-measured)."),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[vl] DONE -> {out_path}", file=sys.stderr)
    print(json.dumps(report["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
