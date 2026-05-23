"""Mac-only diagnostic: dump the MLX / mlx_lm API surface.

Run this on the Mac mini AFTER `setup_mac.sh` succeeded. It writes a
JSON report to `results/platform-tests/mlx_probe_<ts>.json` with:

  * MLX environment snapshot (from `inference_engine.backends.mlx.env`)
  * mlx_lm.load() return-type metadata (so the next backend commit can
    target the real API instead of assumptions)
  * Qwen3-1.7B model structure: layer count, hidden dim, num KV heads,
    head dim, vocab size, attribute paths to backbone vs lm_head
  * mlx_lm cache machinery: which `KVCache` class, what attributes it
    exposes, how `make_prompt_cache(model)` builds it
  * One end-to-end forward pass with timing, to establish a real
    wall-time baseline to compare future commits against

The purpose is to remove guesswork from the next commit. Bench-style
output is also printed to stdout so you can paste a quick summary.

Usage:
    source .venv-mac/bin/activate
    PYTHONPATH=. python3 scripts/probe_mlx.py \
        --model Qwen/Qwen3-1.7B \
        --prompt "Why is the sky blue?"

The probe does NOT modify any state; it only loads the model into MLX,
runs one forward, and writes a report.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_get(obj: Any, attr: str) -> Any:
    """Return getattr or a sentinel string describing why it failed."""
    try:
        return getattr(obj, attr)
    except Exception as e:  # pragma: no cover - probe diagnostic
        return f"<unavailable: {type(e).__name__}: {e}>"


def _describe_callable(fn: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        sig = inspect.signature(fn)
        out["signature"] = str(sig)
        out["parameters"] = list(sig.parameters.keys())
    except (TypeError, ValueError) as e:
        out["signature_error"] = f"{type(e).__name__}: {e}"
    out["doc"] = inspect.getdoc(fn) or ""
    out["module"] = getattr(fn, "__module__", "?")
    out["qualname"] = getattr(fn, "__qualname__", str(fn))
    return out


def _describe_object(obj: Any, *, max_attrs: int = 40) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "class": type(obj).__name__,
        "module": type(obj).__module__,
        "repr_head": repr(obj)[:160],
    }
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    out["public_attr_names"] = attrs[:max_attrs]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    report: Dict[str, Any] = {"model": args.model, "prompt": args.prompt}
    print(f"[probe] model={args.model}", flush=True)

    # --------- step 1: env snapshot ---------
    from inference_engine.backends.mlx.env import probe_environment, MLXEnvironmentError

    env = probe_environment()
    report["env"] = {
        "is_available": env.is_available,
        "mlx_version": env.mlx_version,
        "mlx_lm_version": env.mlx_lm_version,
        "metal_available": env.metal_available,
        "platform": env.platform_str,
        "machine": env.machine,
        "python": env.python_version,
        "failure_reason": env.failure_reason,
        "rendered": env.render(),
    }
    print(f"[probe] env: {env.render()}", flush=True)
    if not env.is_available:
        print("[probe] cannot proceed without working MLX/Metal", flush=True)
        report["aborted"] = True
        _write(report, args.report)
        return 2

    # --------- step 2: import mlx + mlx_lm and dump key API ---------
    import mlx.core as mx           # type: ignore[import-not-found]
    import mlx.nn as mnn            # type: ignore[import-not-found]
    print(f"[probe] mlx.core attrs (first 30): {sorted(dir(mx))[:30]}", flush=True)

    report["mlx.core"] = {
        "module": "mlx.core",
        "version_attr": getattr(mx, "__version__", None),
        "public_attrs_count": len([a for a in dir(mx) if not a.startswith("_")]),
        "default_device": str(mx.default_device()),
        "metal_is_available": mx.metal.is_available(),
    }
    # Snapshot mx.array dtypes
    report["mlx.dtype"] = {
        "bfloat16": str(getattr(mx, "bfloat16", None)),
        "float16": str(getattr(mx, "float16", None)),
        "float32": str(getattr(mx, "float32", None)),
        "int32": str(getattr(mx, "int32", None)),
    }

    try:
        import mlx_lm                   # type: ignore[import-not-found]
    except ImportError as e:
        report["mlx_lm_import_error"] = str(e)
        _write(report, args.report)
        return 3
    print(f"[probe] mlx_lm version: {getattr(mlx_lm, '__version__', '?')}", flush=True)

    report["mlx_lm"] = {
        "version_attr": getattr(mlx_lm, "__version__", None),
        "public_attrs": sorted(a for a in dir(mlx_lm) if not a.startswith("_")),
        "load_signature": _describe_callable(getattr(mlx_lm, "load", None))
            if hasattr(mlx_lm, "load") else None,
        "generate_signature": _describe_callable(getattr(mlx_lm, "generate", None))
            if hasattr(mlx_lm, "generate") else None,
    }
    # mlx_lm submodules of interest
    for sub in ("models", "utils", "tokenizer_utils", "sample_utils", "cache"):
        try:
            mod = __import__(f"mlx_lm.{sub}", fromlist=[sub])
        except Exception as e:
            report[f"mlx_lm.{sub}.error"] = f"{type(e).__name__}: {e}"
            continue
        report[f"mlx_lm.{sub}"] = {
            "module_path": getattr(mod, "__file__", None),
            "public_attrs": sorted(a for a in dir(mod) if not a.startswith("_"))[:40],
        }

    # --------- step 3: load Qwen3-1.7B and dump model structure ---------
    print(f"[probe] mlx_lm.load({args.model!r}) ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    load_time = time.perf_counter() - t0
    print(f"[probe] loaded in {load_time:.2f} s", flush=True)
    report["load_time_s"] = load_time

    report["model"] = _describe_object(model)
    report["tokenizer"] = _describe_object(tokenizer)

    # Specific attributes we expect / want to confirm
    cfg = _safe_get(model, "config") or _safe_get(model, "args")
    if cfg is not None and not isinstance(cfg, str):
        report["model.config"] = {
            k: getattr(cfg, k)
            for k in dir(cfg)
            if not k.startswith("_") and isinstance(getattr(cfg, k, None),
                                                     (int, float, str, bool, type(None)))
        }
    # Layer / backbone / lm_head paths
    for path in ["model", "model.layers", "lm_head", "embed_tokens",
                 "model.embed_tokens", "model.norm", "norm", "head"]:
        cur = model
        ok = True
        for part in path.split("."):
            cur = _safe_get(cur, part)
            if isinstance(cur, str) and cur.startswith("<unavailable"):
                ok = False
                break
        if ok:
            report[f"model.{path}"] = {
                "type": type(cur).__name__,
                "module": type(cur).__module__,
                "repr_head": repr(cur)[:160],
            }
            if isinstance(cur, list):
                report[f"model.{path}.length"] = len(cur)
                if cur:
                    report[f"model.{path}[0]"] = {
                        "type": type(cur[0]).__name__,
                        "public_attrs": sorted(
                            a for a in dir(cur[0]) if not a.startswith("_")
                        )[:30],
                    }

    # --------- step 4: introspect KV cache machinery ---------
    cache_obj = None
    cache_factory_path: Optional[str] = None
    for cand in (
        "make_prompt_cache",
        "models.cache.make_prompt_cache",
        "cache.make_prompt_cache",
    ):
        cur: Any = mlx_lm
        ok = True
        for part in cand.split("."):
            cur = _safe_get(cur, part)
            if isinstance(cur, str) and cur.startswith("<unavailable"):
                ok = False
                break
        if ok and callable(cur):
            cache_factory_path = cand
            try:
                cache_obj = cur(model)
            except Exception as e:
                report[f"cache_factory.{cand}.error"] = f"{type(e).__name__}: {e}"
                continue
            report[f"cache_factory.{cand}"] = _describe_callable(cur)
            break

    if cache_obj is not None:
        if isinstance(cache_obj, list) and cache_obj:
            report["cache_layout"] = {
                "is_list": True,
                "n_layers": len(cache_obj),
                "layer0_class": type(cache_obj[0]).__name__,
                "layer0_module": type(cache_obj[0]).__module__,
                "layer0_attrs": sorted(
                    a for a in dir(cache_obj[0]) if not a.startswith("_")
                )[:40],
            }
            # Probe whether the layer cache exposes K/V tensors directly
            for attr in ("keys", "values", "k", "v", "state", "offset"):
                v = _safe_get(cache_obj[0], attr)
                if isinstance(v, str) and v.startswith("<unavailable"):
                    continue
                report[f"cache.layer0.{attr}"] = {
                    "type": type(v).__name__,
                    "shape": getattr(v, "shape", None),
                }
        else:
            report["cache_layout"] = {
                "is_list": False,
                "type": type(cache_obj).__name__,
                "attrs": sorted(a for a in dir(cache_obj) if not a.startswith("_"))[:40],
            }

    # --------- step 5: one forward pass and timing ---------
    print(f"[probe] tokenize + forward ...", flush=True)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        except TypeError:
            ids = tokenizer.encode(args.prompt)
    else:
        ids = tokenizer.encode(args.prompt)
    if not isinstance(ids, list):
        ids = list(ids)
    report["prompt_tokens"] = len(ids)

    arr = mx.array([ids], dtype=mx.int32)
    cache = None
    if cache_factory_path is not None and cache_obj is not None:
        cache = cache_obj
    t0 = time.perf_counter()
    try:
        if cache is not None:
            try:
                out = model(arr, cache=cache)
            except TypeError:
                out = model(arr)
        else:
            out = model(arr)
        # Force evaluation (MLX is lazy)
        try:
            mx.eval(out)
        except Exception:
            pass
        forward_time = time.perf_counter() - t0
        report["forward_time_s"] = forward_time
        report["forward_output"] = {
            "type": type(out).__name__,
            "shape": tuple(out.shape) if hasattr(out, "shape") else None,
            "dtype": str(getattr(out, "dtype", None)),
        }
        print(f"[probe] forward took {forward_time:.3f} s, "
              f"output shape: {report['forward_output']['shape']}", flush=True)
    except Exception as e:
        report["forward_error"] = f"{type(e).__name__}: {e}"
        print(f"[probe] forward FAILED: {e}", flush=True)

    _write(report, args.report)
    return 0


def _write(report: Dict[str, Any], explicit_path: Optional[str]) -> None:
    if explicit_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "results" / "platform-tests"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"mlx_probe_{int(time.time())}.json"
    else:
        path = Path(explicit_path)
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[probe] wrote {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
