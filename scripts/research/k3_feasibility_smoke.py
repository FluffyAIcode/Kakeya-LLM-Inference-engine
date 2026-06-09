"""K3 hardware feasibility smoke: load DFlash drafter + Gemma 4 26B-A4B
verifier, run a smoke forward, report memory + latency.

This script answers a NARROW question: **on this hardware, can we
load both K3 models simultaneously and run one forward pass on a
non-trivial prompt?** It does NOT exercise:

* The cross-model DLMRestoredVerifier (that's K2.B/K3 implementation
  PR scope; see docs/design/k3-cross-model-dlmrestored-verifier-contract.md
  for the contract).
* Trained `f_θ` projection (no checkpoint exists yet; see
  docs/design/k3-f-theta-training-pipeline.md for how it's trained).
* NIAH ladder evidence (requires the above two).

What it DOES exercise:

* Model download + load (transformers AR for verifier in bf16 on
  CUDA, mlx_lm for verifier in 4-bit on Mac; transformers AR for
  drafter on both since DFlash 0.4B fits comfortably).
* Single forward pass with a representative prompt
* Memory snapshot during/after the forward
* Per-token latency for short generation

Output: JSON report at results/research/k3_feasibility_smoke_<stamp>.json.

The JSON contains enough information to answer "is K3 feasible on
this hardware?" before committing to the K2.B/K3 implementation
PR scope.

Two runtime paths:

  1. **CUDA (vast.ai H100/H200)**: bf16 verifier via
     transformers.AutoModelForCausalLM. DFlash drafter via
     transformers as well. Memory tracking via
     torch.cuda.max_memory_allocated.

  2. **Mac M4 MPS**: 4-bit MLX verifier (pre-quantized via
     scripts/research/k3_quantize_for_mac.py). DFlash drafter via
     transformers (PyTorch MPS). Memory tracking via
     torch.mps.driver_allocated_memory + per-process peak.

The Mac path requires the verifier to be pre-quantized; if the
--verifier-path doesn't exist or doesn't look like an MLX
directory, the script aborts with an actionable error pointing to
the quantize script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--platform", choices=["auto", "cuda", "mac"], default="auto",
        help="Force a specific runtime path. 'auto' = detect. 'cuda' = "
             "transformers bf16 (vast.ai H100/H200). 'mac' = MLX 4-bit "
             "(Mac M4, requires pre-quantized verifier).",
    )
    ap.add_argument(
        "--verifier-path",
        help="Path to verifier model. On CUDA: HF id (default "
             "google/gemma-4-26B-A4B-it). On Mac: local MLX directory "
             "(default models/gemma-4-26B-A4B-it-mlx-4bit, must exist).",
    )
    ap.add_argument(
        "--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash",
        help="HF id of the DFlash drafter (per ADR 0008 §11.14.3).",
    )
    ap.add_argument(
        "--prompt-tokens", type=int, default=512,
        help="Synthetic prompt length for the smoke forward. 512 is short "
             "enough to fit on Mac M4 24 GB at 4-bit + drafter; for longer "
             "context feasibility test, raise (e.g. 4096, 16384). Higher "
             "values stress KV cache memory.",
    )
    ap.add_argument(
        "--gen-tokens", type=int, default=8,
        help="Tokens to generate after the prompt forward. Just enough to "
             "measure per-token latency; not a quality test.",
    )
    ap.add_argument(
        "--seed", type=int, default=42)
    ap.add_argument(
        "--output", default=None,
        help="JSON report path. Default: "
             "results/research/k3_feasibility_smoke_<stamp>.json",
    )
    ap.add_argument(
        "--skip-drafter", action="store_true",
        help="Skip loading the drafter (verifier-only smoke). Useful for "
             "isolating verifier load from joint memory.",
    )
    ap.add_argument(
        "--use-dflash-loader",
        choices=["on", "off"], default="on",
        help="Use the DFlash custom loader from "
             "inference_engine.v04.dflash_loader (ADR 0008 §11.15.3 prereq 4) "
             "instead of plain AutoModelForCausalLM. Default 'on' validates "
             "Block B prereq 4 — checks expected_class, runs key remap, "
             "verifies embed_tokens.weight.var() above the trained-init "
             "threshold, and writes architectural_warnings into the JSON "
             "report. Set 'off' to reproduce the legacy plain-load behaviour "
             "for A/B comparison.",
    )
    ap.add_argument(
        "--inspect-only", action="store_true",
        help="Run inspect_dflash_checkpoint on the drafter and write its "
             "JSON to --output, then exit 0 without loading either model. "
             "Use this on vast as the diagnose-phase before the full smoke.",
    )
    return ap.parse_args()


def _detect_platform(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mac"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    print(
        "[k3-smoke] no GPU / MPS detected; falling back to CPU smoke "
        "(will likely OOM on a 26B verifier — abort if you can't accept "
        "this).",
        file=sys.stderr,
    )
    return "cpu"


def _record_memory(platform: str, label: str) -> Dict[str, Any]:
    """Memory snapshot per platform. Returns JSON-serialisable dict."""
    out: Dict[str, Any] = {"label": label, "platform": platform}
    if platform == "cuda":
        import torch
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        out["current_allocated_bytes"] = int(torch.cuda.memory_allocated())
        out["current_reserved_bytes"] = int(torch.cuda.memory_reserved())
        out["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        out["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        try:
            props = torch.cuda.get_device_properties(0)
            out["device_total_bytes"] = int(props.total_memory)
            out["device_name"] = props.name
        except Exception:
            pass
    elif platform == "mac":
        import torch
        try:
            out["current_allocated_bytes"] = int(
                torch.mps.current_allocated_memory()
            )
        except Exception:
            out["current_allocated_bytes"] = None
        try:
            out["driver_allocated_bytes"] = int(
                torch.mps.driver_allocated_memory()
            )
        except Exception:
            out["driver_allocated_bytes"] = None
        # macOS unified memory total — informational, not a hard limit per
        # se because of swap.
        try:
            import psutil
            out["device_total_bytes"] = int(psutil.virtual_memory().total)
        except Exception:
            out["device_total_bytes"] = None
    else:
        # CPU / unknown
        try:
            import psutil
            out["rss_bytes"] = int(psutil.Process().memory_info().rss)
        except Exception:
            out["rss_bytes"] = None
    return out


def _reset_peak(platform: str) -> None:
    if platform == "cuda":
        import torch
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    # MPS has no peak counter; CPU has nothing to reset.


def _load_verifier_cuda(verifier_id: str) -> Dict[str, Any]:
    """Load Gemma 4 26B-A4B-it via transformers in bf16 on CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(
        f"[k3-smoke] loading verifier (CUDA bf16): {verifier_id}",
        file=sys.stderr, flush=True,
    )
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(verifier_id)
    model = AutoModelForCausalLM.from_pretrained(
        verifier_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",  # auto-shard across GPUs if multi-GPU
    )
    model.eval()
    elapsed = time.perf_counter() - t0
    print(
        f"[k3-smoke] verifier loaded in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return {
        "kind": "transformers_bf16_cuda",
        "model": model,
        "tokenizer": tokenizer,
        "load_seconds": elapsed,
    }


def _load_verifier_mac(verifier_path: str) -> Dict[str, Any]:
    """Load 4-bit MLX-quantized Gemma 4 26B-A4B-it on Mac M4."""
    p = Path(verifier_path)
    if not p.exists() or not p.is_dir():
        print(
            f"ERROR: --verifier-path {verifier_path} is not an existing "
            "directory.\n"
            "On Mac, the verifier must be pre-quantized via:\n"
            "    PYTHONPATH=.:sdks/python python3 "
            "scripts/research/k3_quantize_for_mac.py "
            "--output models/gemma-4-26B-A4B-it-mlx-4bit\n"
            "Then re-run this smoke with --verifier-path pointing to that "
            "directory.",
            file=sys.stderr,
        )
        sys.exit(10)
    config = p / "config.json"
    if not config.exists():
        print(
            f"ERROR: {config} missing — {verifier_path} doesn't look like "
            "an MLX model directory. Re-run quantize.",
            file=sys.stderr,
        )
        sys.exit(11)
    try:
        import mlx_lm  # type: ignore
    except ImportError:
        print(
            "ERROR: mlx-lm not installed. On Mac:\n"
            "    pip install --upgrade mlx-lm",
            file=sys.stderr,
        )
        sys.exit(12)

    print(
        f"[k3-smoke] loading verifier (MLX 4-bit): {verifier_path}",
        file=sys.stderr, flush=True,
    )
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(verifier_path)
    elapsed = time.perf_counter() - t0
    print(
        f"[k3-smoke] verifier loaded in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return {
        "kind": "mlx_4bit_mac",
        "model": model,
        "tokenizer": tokenizer,
        "load_seconds": elapsed,
    }


def _load_drafter(
    drafter_id: str, platform: str, *, use_dflash_loader: bool,
) -> Dict[str, Any]:
    """Load DFlash drafter.

    Two paths:

      * use_dflash_loader=True (default; ADR 0008 §11.15.3 prereq 4):
        load via inference_engine.v04.dflash_loader.load_dflash_drafter.
        Asserts expected_class == Qwen3ForCausalLM, performs key remap,
        attaches fc + hidden_norm extras, and verifies
        embed_tokens.weight.var() above the trained-init threshold.

      * use_dflash_loader=False: legacy AutoModelForCausalLM path.
        For A/B comparison of the warnings the prereq-4 corrected
        loader emits vs HF's stock loader. The smoke will then NOT
        report the architectural_warnings field meaningfully.
    """
    import torch
    from transformers import AutoTokenizer

    print(
        f"[k3-smoke] loading drafter ({platform}, "
        f"loader={'dflash' if use_dflash_loader else 'plain'}): {drafter_id}",
        file=sys.stderr, flush=True,
    )
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(drafter_id, trust_remote_code=True)

    dtype_map = {
        "cuda": torch.bfloat16, "mac": torch.bfloat16, "cpu": torch.float32,
    }
    dtype = dtype_map.get(platform, torch.float32)
    device = {"cuda": "cuda", "mac": "mps", "cpu": "cpu"}.get(platform, "cpu")

    architectural_warnings: list[str] = []
    inspection_payload: Optional[Dict[str, Any]] = None
    embed_tokens_var: Optional[float] = None
    embed_tokens_trained: Optional[bool] = None
    extras_module = None
    expected_class: Optional[str] = None

    if use_dflash_loader:
        from inference_engine.v04.dflash_loader import load_dflash_drafter
        result = load_dflash_drafter(
            drafter_id,
            dtype=dtype,
            device=device if platform != "cuda" else None,
            trust_remote_code=True,
        )
        model = result.model
        if platform == "cuda":
            model = model.to("cuda")
        expected_class = result.expected_class_name
        embed_tokens_var = result.embed_tokens_var
        embed_tokens_trained = result.embed_tokens_trained
        architectural_warnings = list(result.architectural_warnings)
        extras_module = result.extras
        inspection_payload = {
            "qwen3_unmapped_count": len(result.inspection.qwen3_unmapped),
            "checkpoint_extras_count": len(result.inspection.checkpoint_extras),
            "fc_keys": result.inspection.fc_keys,
            "hidden_norm_keys": result.inspection.hidden_norm_keys,
            "warnings": result.inspection.warnings,
        }
    else:
        from transformers import AutoModelForCausalLM
        if platform == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                drafter_id, dtype=dtype,
                attn_implementation="sdpa",
                device_map="auto",
                trust_remote_code=True,
            )
        elif platform == "mac":
            model = AutoModelForCausalLM.from_pretrained(
                drafter_id, dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                drafter_id, dtype=dtype, trust_remote_code=True,
            )
        expected_class = type(model).__name__

    model.eval()
    elapsed = time.perf_counter() - t0
    print(
        f"[k3-smoke] drafter loaded in {elapsed:.1f}s "
        f"(class={expected_class}, embed_tokens_trained={embed_tokens_trained})",
        file=sys.stderr,
    )
    if architectural_warnings:
        print("[k3-smoke] architectural_warnings:", file=sys.stderr)
        for w in architectural_warnings:
            print(f"    * {w}", file=sys.stderr)
    return {
        "kind": f"transformers_{platform}",
        "model": model,
        "tokenizer": tokenizer,
        "load_seconds": elapsed,
        "expected_class": expected_class,
        "embed_tokens_var": embed_tokens_var,
        "embed_tokens_trained": embed_tokens_trained,
        "architectural_warnings": architectural_warnings,
        "inspection": inspection_payload,
        "extras_module": extras_module,
        "loader_path": "dflash" if use_dflash_loader else "plain",
    }


def _verifier_forward(state: Dict[str, Any], prompt: str, gen_tokens: int) -> Dict[str, Any]:
    """Single greedy forward + N tokens generation. Returns latency.

    Two paths matching the verifier loading kind:
      * transformers (CUDA): standard model.generate with greedy.
      * MLX (Mac): mlx_lm.generate with greedy via temp=0.0.
    """
    kind = state["kind"]
    model = state["model"]
    tokenizer = state["tokenizer"]
    if kind == "transformers_bf16_cuda":
        import torch
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(ids, use_cache=False)
        prefill_seconds = time.perf_counter() - t0
        # Gen
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                ids, max_new_tokens=gen_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id or 0,
            )
        gen_seconds = time.perf_counter() - t0
        gen_text = tokenizer.decode(
            out_ids[0, ids.size(1):], skip_special_tokens=True,
        )
        return {
            "prefill_seconds": prefill_seconds,
            "gen_seconds": gen_seconds,
            "gen_tokens": gen_tokens,
            "tokens_per_sec": gen_tokens / gen_seconds if gen_seconds > 0 else 0.0,
            "gen_text_head": gen_text[:120],
            "prompt_token_count": int(ids.size(1)),
        }
    if kind == "mlx_4bit_mac":
        from mlx_lm import generate  # type: ignore
        # mlx_lm tokenizer encodes via ``encode``; load() returns
        # mlx_lm tokenizer wrapper. Both APIs exist in 0.21+.
        t0 = time.perf_counter()
        text = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=gen_tokens, verbose=False,
            sampler=None,  # greedy via default temp=0
        )
        total_seconds = time.perf_counter() - t0
        # mlx_lm doesn't expose prefill vs decode split easily; report
        # combined and the count of generated tokens.
        return {
            "prefill_seconds": None,  # not separable in mlx_lm.generate
            "gen_seconds": total_seconds,
            "gen_tokens": gen_tokens,
            "tokens_per_sec": gen_tokens / total_seconds if total_seconds > 0 else 0.0,
            "gen_text_head": text[:120] if isinstance(text, str) else "",
            "prompt_token_count": None,  # mlx_lm doesn't return this directly
        }
    raise RuntimeError(f"unknown verifier kind: {kind}")


def _drafter_forward(state: Dict[str, Any], prompt_token_count: Optional[int]) -> Dict[str, Any]:
    """Single forward of the drafter on a same-length synthetic prompt
    to confirm the drafter loads + runs. Doesn't exercise DFlash's
    block-diffusion drafting protocol — just a plain transformers
    forward to confirm hooks would fire."""
    import torch
    model = state["model"]
    tokenizer = state["tokenizer"]
    n = prompt_token_count or 512
    # Resolve drafter vocab size robustly. DFlash uses
    # trust_remote_code=True with a custom tokenizer that may not
    # expose vocab_size as a simple attribute (the tokenizer's
    # vocab_size is sometimes a method, sometimes None, sometimes
    # 0 on the wrapped tokenizer object). Fall back through several
    # candidate attributes; if all yield <= 0, use a safe default
    # of 50000 (any real LLM tokeniser is far larger). The synthetic
    # input only needs valid token IDs in some valid range; the
    # smoke is checking forward-pass plumbing, not generation
    # quality, so bounding the random IDs at min(true vocab,
    # 50000) is fine.
    candidates = [
        getattr(tokenizer, "vocab_size", None),
        # Newer transformers tokenizers expose ``__len__`` returning
        # the full vocab size including added tokens.
        len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        # As a last resort, inspect the model's embedding matrix.
        (
            getattr(getattr(model, "get_input_embeddings", lambda: None)(),
                    "num_embeddings", None)
            if hasattr(model, "get_input_embeddings")
            else None
        ),
    ]
    vocab_size = None
    for c in candidates:
        try:
            iv = int(c) if c is not None else 0
            if iv > 1:
                vocab_size = iv
                break
        except (TypeError, ValueError):
            continue
    if vocab_size is None or vocab_size <= 1:
        vocab_size = 50000  # safe fallback for any real LM tokeniser
    # Use [1, vocab_size) so torch.randint always sees from < to.
    fake_ids = torch.randint(
        1, vocab_size,
        size=(1, n), device=model.device, dtype=torch.long,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(fake_ids, use_cache=False)
    elapsed = time.perf_counter() - t0
    logits_shape = tuple(out.logits.shape) if hasattr(out, "logits") else None
    return {
        "forward_seconds": elapsed,
        "input_tokens": n,
        "output_logits_shape": logits_shape,
        "drafter_vocab_size_used": vocab_size,
    }


def main() -> int:
    args = parse_args()
    platform = _detect_platform(args.platform)
    print(f"[k3-smoke] platform: {platform}", file=sys.stderr)

    # Pick default verifier path based on platform.
    if args.verifier_path is None:
        if platform == "mac":
            args.verifier_path = "models/gemma-4-26B-A4B-it-mlx-4bit"
        else:
            args.verifier_path = "google/gemma-4-26B-A4B-it"
    print(f"[k3-smoke] verifier:  {args.verifier_path}", file=sys.stderr)
    print(f"[k3-smoke] drafter:   {args.drafter_id}", file=sys.stderr)
    print(f"[k3-smoke] prompt n:  {args.prompt_tokens}", file=sys.stderr)
    print(f"[k3-smoke] gen n:     {args.gen_tokens}", file=sys.stderr)

    report: Dict[str, Any] = {
        "schema_version": 2,
        "kind": "k3_feasibility_smoke",
        "config": {
            "platform": platform,
            "verifier_path": args.verifier_path,
            "drafter_id": args.drafter_id,
            "prompt_tokens": args.prompt_tokens,
            "gen_tokens": args.gen_tokens,
            "seed": args.seed,
            "skip_drafter": bool(args.skip_drafter),
            "use_dflash_loader": bool(args.use_dflash_loader == "on"),
            "inspect_only": bool(args.inspect_only),
        },
        "stages": [],
    }

    if args.inspect_only:
        from inference_engine.v04.dflash_loader import inspect_dflash_checkpoint
        print(
            f"[k3-smoke] inspect-only mode: dumping DFlash checkpoint "
            f"inspection JSON for {args.drafter_id}",
            file=sys.stderr,
        )
        inspection = inspect_dflash_checkpoint(args.drafter_id)
        report["stages"].append({
            "stage": "drafter_inspection",
            "inspection": inspection.to_json(),
        })
        report["summary"] = {
            "status": "inspect_only",
            "qwen3_unmapped_count": len(inspection.qwen3_unmapped),
            "fc_keys_present": bool(inspection.fc_keys),
            "hidden_norm_keys_present": bool(inspection.hidden_norm_keys),
            "warnings_count": len(inspection.warnings),
        }
        _emit(report, args.output)
        for w in inspection.warnings:
            print(f"[k3-smoke] WARN: {w}", file=sys.stderr)
        return 0

    # Baseline memory snapshot before any model load.
    _reset_peak(platform)
    report["stages"].append({
        "stage": "baseline",
        "memory": _record_memory(platform, "baseline"),
    })

    # Verifier load.
    try:
        if platform == "mac":
            ver = _load_verifier_mac(args.verifier_path)
        else:
            ver = _load_verifier_cuda(args.verifier_path)
        report["stages"].append({
            "stage": "verifier_loaded",
            "memory": _record_memory(platform, "after_verifier_load"),
            "verifier_load_seconds": ver["load_seconds"],
            "verifier_kind": ver["kind"],
        })
    except Exception as e:
        report["stages"].append({
            "stage": "verifier_load_FAIL",
            "error": f"{type(e).__name__}: {e}",
        })
        report["summary"] = {"status": "fail_at_verifier_load"}
        _emit(report, args.output)
        return 20

    # Drafter load.
    drafter = None
    if not args.skip_drafter:
        try:
            drafter = _load_drafter(
                args.drafter_id, platform,
                use_dflash_loader=(args.use_dflash_loader == "on"),
            )
            stage = {
                "stage": "drafter_loaded",
                "memory": _record_memory(platform, "after_drafter_load"),
                "drafter_load_seconds": drafter["load_seconds"],
                "drafter_kind": drafter["kind"],
                "loader_path": drafter["loader_path"],
                "expected_class": drafter["expected_class"],
                "embed_tokens_var": drafter.get("embed_tokens_var"),
                "embed_tokens_trained": drafter.get("embed_tokens_trained"),
                "architectural_warnings": drafter.get("architectural_warnings", []),
                "inspection": drafter.get("inspection"),
                "extras_attached": drafter.get("extras_module") is not None,
            }
            report["stages"].append(stage)
            assert drafter["expected_class"] == "Qwen3ForCausalLM", (
                "ADR 0008 §11.15.3 prereq 4 expected_class assert failed: "
                f"got {drafter['expected_class']!r}, expected "
                "'Qwen3ForCausalLM' (DFlash declares model_type: qwen3 "
                "and ships no auto_map / modeling_dflash.py — Qwen3 dispatch "
                "is correct). If the loaded class differs, the upstream "
                "DFlash repo has likely changed its config.json — re-fetch "
                "and update §11.15.2.1 / §11.15.3."
            )
        except Exception as e:
            report["stages"].append({
                "stage": "drafter_load_FAIL",
                "error": f"{type(e).__name__}: {e}",
            })
            # Continue without drafter — still useful evidence.
            print(
                f"[k3-smoke] WARN: drafter load failed: {e}\n"
                "  Continuing with verifier-only smoke.",
                file=sys.stderr,
            )

    # Synthetic prompt — repeated short text to reach ~prompt_tokens.
    base = "The Kakeya inference engine validates v0.4 K/V Restoration on hardware. "
    repeats = max(1, args.prompt_tokens // 12)
    prompt = base * repeats

    # Verifier forward.
    try:
        ver_metrics = _verifier_forward(ver, prompt, args.gen_tokens)
        report["stages"].append({
            "stage": "verifier_forward",
            "memory": _record_memory(platform, "after_verifier_forward"),
            "metrics": ver_metrics,
        })
        print(
            f"[k3-smoke] verifier forward OK; "
            f"gen={ver_metrics['gen_tokens']} tokens in "
            f"{ver_metrics['gen_seconds']:.2f}s "
            f"({ver_metrics['tokens_per_sec']:.2f} tok/s)",
            file=sys.stderr,
        )
    except Exception as e:
        report["stages"].append({
            "stage": "verifier_forward_FAIL",
            "error": f"{type(e).__name__}: {e}",
        })
        report["summary"] = {"status": "fail_at_verifier_forward"}
        _emit(report, args.output)
        return 30

    # Drafter forward (if loaded).
    if drafter is not None:
        try:
            draft_metrics = _drafter_forward(
                drafter, ver_metrics.get("prompt_token_count"),
            )
            report["stages"].append({
                "stage": "drafter_forward",
                "memory": _record_memory(platform, "after_drafter_forward"),
                "metrics": draft_metrics,
            })
            print(
                f"[k3-smoke] drafter forward OK; "
                f"{draft_metrics['input_tokens']} tokens in "
                f"{draft_metrics['forward_seconds']:.3f}s",
                file=sys.stderr,
            )
        except Exception as e:
            report["stages"].append({
                "stage": "drafter_forward_FAIL",
                "error": f"{type(e).__name__}: {e}",
            })

    report["summary"] = {
        "status": "pass",
        "verifier_loadable": True,
        "verifier_forward_ok": True,
        "drafter_loadable": drafter is not None,
        "drafter_forward_ok": (
            drafter is not None
            and report["stages"][-1].get("stage") == "drafter_forward"
        ),
        "drafter_expected_class": (
            drafter["expected_class"] if drafter is not None else None
        ),
        "drafter_embed_tokens_trained": (
            drafter.get("embed_tokens_trained") if drafter is not None else None
        ),
        "drafter_architectural_warnings_count": (
            len(drafter.get("architectural_warnings", []))
            if drafter is not None else None
        ),
        "drafter_extras_attached": (
            drafter.get("extras_module") is not None
            if drafter is not None else None
        ),
        "loader_path": (
            drafter["loader_path"] if drafter is not None else None
        ),
    }
    _emit(report, args.output)
    print("[k3-smoke] PASS", file=sys.stderr)
    return 0


def _emit(report: Dict[str, Any], output: Optional[str]) -> None:
    out_path = (
        Path(output) if output
        else Path(f"results/research/k3_feasibility_smoke_{int(time.time())}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[k3-smoke] report -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
