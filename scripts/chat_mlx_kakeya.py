#!/usr/bin/env python3
"""Interactive CLI chat with gemma-4 on the Kakeya-for-Mac engine (MLX).

Runs the gemma-4 MLX verifier with **Kakeya Attention's bounded sink+window KV
cache (S5)**: the model's sliding-attention layers keep only ``sink + window``
tokens resident, while gemma-4's native full-attention layers keep full context
(the "S5 free lunch" — recall is carried by the full layers, so no f_θ/proposer
restoration is needed on gemma-4). This is single-stream (B=1) generation, which
sidesteps the MLX ``B>1, L=1`` batched-decode kernel bug entirely.

Usage (on the Mac, in the repo checkout):

    # interactive REPL — type a message, get gemma-4's reply, blank line/Ctrl-D quits
    PYTHONPATH=. python3 scripts/chat_mlx_kakeya.py \
        --verifier-path /Users/fluffy314/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit

    # non-interactive smoke (used by the Mac-bridge preset): fixed turns -> JSON transcript
    PYTHONPATH=. python3 scripts/chat_mlx_kakeya.py --verifier-path <dir> \
        --scripted "What is the capital of France?||Now multiply 6 by 7." \
        --output results/research/mac_gemma4_kakeya_chat.json

mlx_lm / mlx are imported lazily inside ``main`` so ``--help`` works off-Mac.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _log(msg: str) -> None:
    print(f"[kakeya-chat] {msg}", file=sys.stderr, flush=True)


def _resolve_eos(tok) -> set:
    """gemma stops a turn on <end_of_turn>; also honor the tokenizer EOS."""
    eos = set()
    if getattr(tok, "eos_token_id", None) is not None:
        eos.add(int(tok.eos_token_id))
    for marker in ("<end_of_turn>", "<eos>"):
        try:
            ids = tok.encode(marker, add_special_tokens=False)
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            if len(ids) == 1:
                eos.add(int(ids[0]))
        except Exception:
            pass
    return eos


def _is_degenerate_loop(s: str, unit: int = 16) -> bool:
    """True only on a TRUE consecutive loop: the same ``unit``-char block
    repeated 3x back-to-back at the tail. (Deliberately strict so an answer that
    merely echoes itself once — e.g. text + a json wrapper — is NOT cut.)"""
    if len(s) < unit * 3:
        return False
    a, b, c = s[-unit:], s[-2 * unit:-unit], s[-3 * unit:-2 * unit]
    return a == b == c and a.strip() != ""


def _apply_template(tok, history, *, thinking: bool) -> List[int]:
    """Encode the chat history. gemma-4 has a reasoning ("thought") channel; the
    clean way to get direct answers is the template's ``enable_thinking`` flag
    (NOT injecting a raw channel marker, which leaks 'thought' text and loops).
    Falls back gracefully if the template doesn't accept the kwarg."""
    try:
        ids = tok.apply_chat_template(
            history, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        ids = tok.apply_chat_template(history, add_generation_prompt=True)
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description="gemma-4 chat on the Kakeya-for-Mac (MLX) engine")
    ap.add_argument("--verifier-path", required=True,
                    help="Local MLX gemma-4 model dir.")
    ap.add_argument("--sink", type=int, default=4, help="Kakeya sink tokens.")
    ap.add_argument("--window", type=int, default=64,
                    help="Kakeya sliding-window tokens (S5; sliding layers).")
    ap.add_argument("--full-window", type=int, default=8192,
                    help="Resident window for the full-attention (exact) layers "
                         "— large = effectively full context (S5 recall carrier).")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="Generation cap. Long explanations can need 1500+; raise "
                         "this if answers get cut off ('断掉').")
    ap.add_argument("--repetition-penalty", type=float, default=1.3,
                    help="Penalize repeated tokens to stop greedy loops (1.0 = off).")
    ap.add_argument("--thinking", action="store_true",
                    help="Allow gemma-4's reasoning channel (default: direct answers).")
    ap.add_argument("--system", default=None, help="Optional system prompt.")
    ap.add_argument("--scripted", default=None,
                    help="Non-interactive: '||'-separated user turns; writes a transcript.")
    ap.add_argument("--output", default=None, help="Transcript JSON (scripted mode).")
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    from mlx_lm.generate import generate_step  # type: ignore
    from inference_engine.backends.mlx.cache import (
        SinkWindowKVCache, total_kv_bytes, cache_seq_length,
    )
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model, mlx_full_attention_layer_indices,
    )

    _log(f"loading MLX model: {args.verifier_path}")
    t_load = time.time()
    model, tok = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(model)
    n_layers = len(text_model.layers)
    full_idx = set(mlx_full_attention_layer_indices(text_model))
    eos = _resolve_eos(tok)
    _log(f"loaded in {time.time()-t_load:.1f}s | layers={n_layers} "
         f"exact(full-attn)={sorted(full_idx)} sink={args.sink} window={args.window} "
         f"eos={sorted(eos)}")
    _log("Kakeya Attention: sliding layers bounded to sink+window; "
         "exact layers keep full context (S5).")

    logits_processors = None
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        try:
            from mlx_lm.sample_utils import make_logits_processors  # type: ignore
            logits_processors = make_logits_processors(
                repetition_penalty=args.repetition_penalty)
            _log(f"repetition_penalty={args.repetition_penalty} enabled")
        except Exception as exc:  # noqa: BLE001
            _log(f"repetition penalty unavailable ({exc}); greedy")

    def new_cache() -> list:
        # S5 hybrid: exact (full-attn) layers get a large window (≈full context,
        # the recall carrier); sliding layers get the tight Kakeya window.
        return [
            SinkWindowKVCache(
                sink_size=args.sink,
                window_size=(args.full_window if li in full_idx else args.window),
            )
            for li in range(n_layers)
        ]

    def build_prompt_ids(history: List[Dict[str, str]]) -> List[int]:
        return _apply_template(tok, history, thinking=args.thinking)

    def generate_turn(prompt_ids: List[int], on_delta=None) -> Dict[str, Any]:
        """Single-stream greedy decode over a FRESH Kakeya bounded cache."""
        cache = new_cache()
        toks: List[int] = []
        shown = ""
        t0 = time.time()
        gkw: Dict[str, Any] = dict(prompt_cache=cache, max_tokens=args.max_new_tokens)
        if logits_processors is not None:
            gkw["logits_processors"] = logits_processors
        try:
            stream = generate_step(mx.array(prompt_ids), model, **gkw)
            first = next(stream)
        except TypeError:  # older mlx_lm without logits_processors kwarg
            gkw.pop("logits_processors", None)
            stream = generate_step(mx.array(prompt_ids), model, **gkw)
            first = next(stream)

        def _iter():
            yield first
            yield from stream

        stop_reason = "max"  # generator exhausts at max_tokens unless we break
        for tok_id, _ in _iter():
            t = int(tok_id)
            if t in eos:
                stop_reason = "eos"
                break
            toks.append(t)
            full = tok.decode(toks, skip_special_tokens=True)
            delta = full[len(shown):]
            if delta and on_delta is not None:
                on_delta(delta)
            shown = full
            if _is_degenerate_loop(full):  # true back-to-back repeat → stop
                stop_reason = "loop"
                break
        dt = max(time.time() - t0, 1e-9)
        return {
            "text": tok.decode(toks, skip_special_tokens=True),
            "n_tokens": len(toks),
            "stop_reason": stop_reason,
            "decode_tps": round(len(toks) / dt, 2),
            "resident_kv_bytes": int(total_kv_bytes(cache)),
            "resident_kv_seq_len_first_layer": int(cache_seq_length(cache)),
            "prompt_tokens": len(prompt_ids),
        }

    history: List[Dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    # ---- scripted (non-interactive) mode: for Mac-bridge verification ----
    if args.scripted is not None:
        turns = [t for t in args.scripted.split("||") if t.strip()]
        transcript: List[Dict[str, Any]] = []
        for user in turns:
            history.append({"role": "user", "content": user})
            info = generate_turn(build_prompt_ids(history))
            history.append({"role": "assistant", "content": info["text"]})
            transcript.append({"user": user, **info})
            _log(f"USER: {user!r}")
            _log(f"GEMMA-4: {info['text'][:200]!r}  "
                 f"({info['n_tokens']} tok, {info['decode_tps']} tok/s, "
                 f"resident_kv={info['resident_kv_bytes']/1e6:.1f}MB)")
        report = {
            "kind": "mac_gemma4_kakeya_chat", "schema_version": 1,
            "model_path": args.verifier_path,
            "engine": "Kakeya-for-Mac (MLX, S5 bounded sink+window, single-stream)",
            "sink": args.sink, "window": args.window, "full_window": args.full_window,
            "exact_layers": sorted(full_idx), "n_layers": n_layers,
            "turns": transcript,
        }
        if args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _log(f"wrote transcript -> {outp}")
        else:
            print(json.dumps(report, indent=2))
        return 0

    # ---- interactive REPL ----
    _log("ready. Type a message and press Enter. Blank line or Ctrl-D to quit.")
    while True:
        try:
            if sys.stdin.isatty():
                sys.stderr.write("\nyou> ")
                sys.stderr.flush()
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            _log("interrupted")
            break
        if not line:
            break
        user = line.strip()
        if not user:
            break
        history.append({"role": "user", "content": user})
        sys.stderr.write("gemma-4> ")
        sys.stderr.flush()
        info = generate_turn(
            build_prompt_ids(history),
            on_delta=lambda d: (sys.stdout.write(d), sys.stdout.flush()),
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
        history.append({"role": "assistant", "content": info["text"]})
        warn = ("  [WARN: hit --max-new-tokens; raise it for longer answers]"
                if info["stop_reason"] == "max" else f"  [stopped: {info['stop_reason']}]")
        _log(f"{info['n_tokens']} tok, {info['decode_tps']} tok/s, "
             f"resident bounded-KV {info['resident_kv_bytes']/1e6:.1f} MB "
             f"(sliding capped at sink+window={args.sink}+{args.window}){warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
