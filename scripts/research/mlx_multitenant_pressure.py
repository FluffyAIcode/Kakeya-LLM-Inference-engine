"""Multi-tenant resident-window pressure test — Kakeya vs MLX-native (A/B).

The single-tenant gRPC capacity test (grpc_agent_capacity_loadtest.py) measures
connection admission, not parallel-inference capacity. This harness measures the
metric that actually matters for many concurrent agents: **how many agents, each
with its own resident KV window, fit and run in a fixed memory budget** — and
A/Bs the Kakeya bounded sink+window cache against the MLX-native cache.

Per agent it builds an independent KV cache and prefills it to a context length:
  * ``native``  — ``mlx_model.make_cache()`` (gemma's own hybrid cache: the 5
    full-attention layers grow with context; sliding layers are bounded by the
    model's sliding_window, typically 1024).
  * ``kakeya``  — S5: the 5 full-attention layers exact + sliding layers bounded
    to ``sink+window`` (e.g. 68). The deployment config.

It then ramps the agent count (replicating the prefilled cache into N independent
allocations — real N× memory, ~1× prefill compute), measuring peak GPU memory
until a budget is hit → the **max concurrent agents** per mode. It also times a
single-agent decode for each mode (per-agent inference cost). Honest framing: a
single Mac GPU serializes/batches inference, so this reports the **memory-fit
capacity** (the dominant multi-tenant differentiator) + per-agent decode rate;
the *served* multi-tenant path needs per-session binding (PR-A3c / v0.4).

MLX-only → run on Apple Silicon (via the Mac bridge preset).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _peak_mb() -> Optional[float]:
    import mlx.core as mx
    for getter in ("get_peak_memory",):
        fn = getattr(mx, getter, None)
        if fn:
            try:
                return round(fn() / 1e6, 1)
            except Exception:  # noqa: BLE001
                pass
    metal = getattr(mx, "metal", None)
    if metal and getattr(metal, "get_peak_memory", None):
        try:
            return round(metal.get_peak_memory() / 1e6, 1)
        except Exception:  # noqa: BLE001
            return None
    return None


def _reset_peak() -> None:
    import mlx.core as mx
    for obj in (mx, getattr(mx, "metal", None)):
        fn = getattr(obj, "reset_peak_memory", None) if obj else None
        if fn:
            try:
                fn()
                return
            except Exception:  # noqa: BLE001
                pass


def _cache_kv_bytes(cache: list) -> int:
    total = 0
    for layer in cache:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None:
            total += int(k.nbytes)
        if v is not None:
            total += int(v.nbytes)
    return total


def _clone_cache(src: list, cls_native):
    """Deep-copy a per-layer cache list into independent allocations."""
    import mlx.core as mx
    out = []
    for layer in src:
        new = layer.__class__.__new__(layer.__class__)
        # copy the data + meta attributes we know about
        for attr in ("sink_size", "window_size", "offset", "step",
                     "max_size", "keep", "_idx"):
            if hasattr(layer, attr):
                setattr(new, attr, getattr(layer, attr))
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        new.keys = mx.array(k) if k is not None else None
        new.values = mx.array(v) if v is not None else None
        out.append(new)
    mx.eval([c.keys for c in out if c.keys is not None]
            + [c.values for c in out if c.values is not None])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", required=True)
    ap.add_argument("--mode", choices=["native", "s5", "sinkwin", "both"],
                    default="both",
                    help="'both' = native vs s5 (recall-preserving deployment "
                         "config). 'sinkwin' = pure sink+window memory floor "
                         "(does not preserve full-attn recall).")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--max-agents", type=int, default=256)
    ap.add_argument("--mem-budget-mb", type=float, default=21000.0,
                    help="stop ramping agents when peak memory exceeds this")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import mlx.core as mx  # noqa: F401
    import mlx_lm
    sys.path.insert(0, "sdks/python")
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model, mlx_full_attention_layer_indices,
    )
    from inference_engine.backends.mlx.cache import (
        make_sink_window_cache, SinkWindowKVCache,
    )

    def make_cache_for(mode: str):
        """Fresh per-agent cache for a mode.

        * native  — gemma's own hybrid cache (full-attn layers grow with ctx;
          sliding layers bounded by the model's sliding_window, ~1024).
        * sinkwin — pure sink+window on ALL layers (smallest, but the
          full-attn layers lose long-context recall — not a recall-preserving
          config; reported for the memory floor).
        * s5      — recall-preserving deployment config: the 5 full-attention
          layers keep exact KV (native KVCache), sliding layers bounded to
          sink+window.
        """
        if mode == "sinkwin":
            return make_sink_window_cache(text_model, sink_size=args.sink,
                                          window_size=args.window)
        if mode == "s5":
            native = mlx_model.make_cache()
            mixed = []
            for li in range(len(native)):
                if li in set(full_idx):
                    mixed.append(native[li])  # exact full-attn KV
                else:
                    mixed.append(SinkWindowKVCache(sink_size=args.sink,
                                                   window_size=args.window))
            return mixed
        return mlx_model.make_cache()

    print(f"[mt] loading {args.verifier_path}", flush=True)
    mlx_model, _tok = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    n_layers = len(text_model.layers)
    full_idx = mlx_full_attention_layer_indices(text_model)
    C = args.context_len
    prompt = [1 + (j % 64) for j in range(C)]
    weights_mb = _peak_mb()
    print(f"[mt] layers={n_layers} full_attn={full_idx} ctx={C} "
          f"weights/peak~{weights_mb}MB", flush=True)

    def prefill(cache) -> None:
        chunk = args.prefill_chunk
        for s in range(0, len(prompt), chunk):
            part = prompt[s:s + chunk]
            if part:
                out = mlx_model(mx.array([part]), cache=cache)
                mx.eval(out)

    def decode_tps(cache, logits) -> float:
        t0 = time.perf_counter()
        n = 0
        for _ in range(args.decode_steps):
            tok = int(mx.argmax(logits).item())
            out = mlx_model(mx.array([[tok]]), cache=cache)
            mx.eval(out)
            logits = out[0, -1]
            n += 1
        dt = time.perf_counter() - t0
        return round(n / dt, 3) if dt > 0 else 0.0

    def run_mode(mode: str) -> Dict[str, Any]:
        _reset_peak()
        # First agent: real prefill to context C, measure per-agent KV + decode.
        base = make_cache_for(mode)
        t0 = time.perf_counter()
        prefill(base)
        prefill_s = round(time.perf_counter() - t0, 2)
        out = mlx_model(mx.array([[1]]), cache=base)
        mx.eval(out)
        per_agent_kv = _cache_kv_bytes(base)
        tps = decode_tps(base, out[0, -1])
        # REAL ramp: each additional agent gets its own freshly-prefilled cache
        # (real N x memory — no copy-on-write shortcut). Keep all alive; stop at
        # the memory budget or the agent cap.
        agents = [base]
        peak = _peak_mb()
        rows = [{"agents": 1, "peak_mb": peak}]
        max_agents = 1
        budget_hit = False
        while len(agents) < args.max_agents:
            c = make_cache_for(mode)
            prefill(c)
            mx.eval(out)
            agents.append(c)
            peak = _peak_mb()
            n = len(agents)
            rows.append({"agents": n, "peak_mb": peak})
            if n in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128) or n == args.max_agents:
                print(f"[mt][{mode}] agents={n:4d} peak={peak}MB "
                      f"(per-agent KV {round(per_agent_kv/1e6,1)}MB)", flush=True)
            max_agents = n
            if peak and peak > args.mem_budget_mb:
                print(f"[mt][{mode}] budget {args.mem_budget_mb}MB hit at N={n}",
                      flush=True)
                budget_hit = True
                break
        # Derived capacity from the measured per-agent KV + a stated KV budget
        # (in case the cap was hit before the budget).
        kv_budget_mb = args.mem_budget_mb - (weights_mb or 0)
        derived_max = (int(kv_budget_mb / (per_agent_kv / 1e6))
                       if per_agent_kv else None)
        result = {
            "mode": mode,
            "per_agent_kv_mb": round(per_agent_kv / 1e6, 2),
            "max_agents_measured": max_agents,
            "max_agents_hit_budget": budget_hit,
            "derived_max_agents_in_kv_budget": derived_max,
            "kv_budget_mb": round(kv_budget_mb, 1),
            "prefill_s": prefill_s,
            "decode_tokens_per_s_per_agent": tps,
            "peak_mb_at_max": peak,
            "ramp": rows,
        }
        del agents
        return result

    if args.mode == "both":
        modes = ["native", "s5"]
    else:
        modes = [args.mode]
    results = {m: run_mode(m) for m in modes}
    report: Dict[str, Any] = {
        "kind": "mlx_multitenant_pressure",
        "schema_version": 1,
        "config": {
            "verifier_path": args.verifier_path, "context_len": C,
            "sink": args.sink, "window": args.window,
            "mem_budget_mb": args.mem_budget_mb, "n_layers": n_layers,
            "full_attn_layers": full_idx, "decode_steps": args.decode_steps,
        },
        "results": results,
    }
    kk_key = "s5" if "s5" in results else ("sinkwin" if "sinkwin" in results else None)
    if "native" in results and kk_key:
        nv, kk = results["native"], results[kk_key]
        report["ab"] = {
            "kakeya_config": kk_key,
            "per_agent_kv_mb": {"native": nv["per_agent_kv_mb"],
                                "kakeya": kk["per_agent_kv_mb"]},
            "kv_reduction_x": (round(nv["per_agent_kv_mb"] / kk["per_agent_kv_mb"], 2)
                               if kk["per_agent_kv_mb"] else None),
            "derived_max_agents": {"native": nv["derived_max_agents_in_kv_budget"],
                                   "kakeya": kk["derived_max_agents_in_kv_budget"]},
            "agent_capacity_x": (round(kk["derived_max_agents_in_kv_budget"]
                                       / nv["derived_max_agents_in_kv_budget"], 2)
                                 if nv["derived_max_agents_in_kv_budget"] else None),
            "decode_tps_per_agent": {"native": nv["decode_tokens_per_s_per_agent"],
                                     "kakeya": kk["decode_tokens_per_s_per_agent"]},
        }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[mt] wrote {args.output}", flush=True)
    if "ab" in report:
        ab = report["ab"]
        print(f"[mt] A/B @ctx{C} ({ab['kakeya_config']}): per-agent KV "
              f"native={ab['per_agent_kv_mb']['native']}MB vs "
              f"kakeya={ab['per_agent_kv_mb']['kakeya']}MB ({ab['kv_reduction_x']}x) | "
              f"derived max agents native={ab['derived_max_agents']['native']} "
              f"vs kakeya={ab['derived_max_agents']['kakeya']} "
              f"({ab['agent_capacity_x']}x)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
