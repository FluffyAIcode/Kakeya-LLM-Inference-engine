"""Mac analog of the PR-A3c batched scheduler (§3.7) — MLX, on Mac mini.

The §3.7 BatchedDecodeScheduler is torch/CUDA (it ran on H200). On Apple Silicon
the equivalent is a batched MLX forward: N sessions decoded in one pass over the
MLX gemma verifier, each a batch row with its own KV-cache row. This bench
measures the served-path batching value on the Mac:

  * serialized — each session's decode run alone, summed (the §3.6 behaviour)
  * batched    — all N decoded in one batched forward per step

reporting aggregate decode tok/s, the speedup, and per-session recall (recall is
the bottom line — uses the gemma-native cache, which preserves recall). Equal-
length prompts (modal NIAH bucket) keep the batch clean.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", required=True)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--haystack-lines", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--kakeya-cache", action="store_true",
                    help="Build the batched cache from Kakeya's concat-based "
                         "SinkWindowKVCache (avoids mlx_lm's in-place "
                         "buffer-assignment decode path that breaks at batch>1). "
                         "S5: full-attn layers keep all, sliding bounded.")
    ap.add_argument("--window", type=int, default=64,
                    help="sliding-layer window when --kakeya-cache (S5).")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--full-window", type=int, default=100000,
                    help="full-attn-layer window when --kakeya-cache "
                         "(large = keep all, exact recall).")
    ap.add_argument("--manual-sdpa", action="store_true",
                    help="Replace mlx_lm gemma's mx.fast.scaled_dot_product_"
                         "attention with a manual batched matmul-softmax SDPA "
                         "(works around the suspected batch>1 + GQA fast-kernel "
                         "bug). The candidate fix.")
    ap.add_argument("--pad-decode", action="store_true",
                    help="L>=2 padded batched decode workaround: every decode "
                         "step feeds a length-2 query (the new token "
                         "duplicated) so the forward never enters mlx's L=1 "
                         "B>1 single-token (qmv) quantized-decode kernel "
                         "(the suspected core-kernel bug). The logits at query "
                         "position 0 give the next token; the duplicate at "
                         "position 1 is trimmed so the cache stays at the true "
                         "position. Stays batched/parallel over sessions "
                         "(the B dimension is untouched), Python-only. Requires "
                         "a trimmable cache, so it forces --kakeya-cache.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.pad_decode and not args.kakeya_cache:
        # The padded loop trims the duplicate KV each step; only the
        # concat SinkWindowKVCache is soundly trimmable (gemma's native
        # RotatingKVCache is not once the sliding ring wraps).
        args.kakeya_cache = True
        print("[mlx-mt] --pad-decode forces --kakeya-cache (trimmable cache "
              "needed to drop the padding token)", flush=True)

    import mlx.core as mx
    import mlx_lm
    sys.path.insert(0, "sdks/python")
    from inference_engine.v04 import make_niah_dataset
    from inference_engine.backends.mlx.cache import SinkWindowKVCache
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model, mlx_full_attention_layer_indices,
    )

    if args.manual_sdpa:
        import mlx_lm.models.gemma4_text as g4

        def _manual_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None,
                         sinks=None):
            # queries [B, n_heads, L, D]; keys/values [B, n_kv, S, D] (GQA).
            n_heads = queries.shape[1]
            n_kv = keys.shape[1]
            if n_kv != n_heads:
                rep = n_heads // n_kv
                keys = mx.repeat(keys, rep, axis=1)
                values = mx.repeat(values, rep, axis=1)
            scores = (queries * scale) @ mx.swapaxes(keys, -1, -2)  # [B,h,L,S]
            if mask is not None:
                if isinstance(mask, str):   # "causal"
                    qL, kL = scores.shape[-2], scores.shape[-1]
                    qi = mx.arange(kL - qL, kL)[:, None]
                    ki = mx.arange(kL)[None]
                    bmask = qi >= ki
                    scores = mx.where(bmask, scores, mx.finfo(scores.dtype).min)
                elif mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores = scores + mask
            scores = mx.softmax(scores, axis=-1, precise=True)
            return scores @ values

        g4.scaled_dot_product_attention = _manual_sdpa
        print("[mlx-mt] patched gemma SDPA -> manual batched matmul", flush=True)

    print(f"[mlx-mt] loading {args.verifier_path}", flush=True)
    model, tok = mlx_lm.load(args.verifier_path)
    N = args.sessions

    text_model = resolve_mlx_text_model(model)
    full_idx = set(mlx_full_attention_layer_indices(text_model))

    def new_cache():
        if not args.kakeya_cache:
            return model.make_cache()
        # S5 hybrid via concat-based caches: full-attn layers keep all (large
        # window = exact recall), sliding layers bounded to sink+window.
        return [
            SinkWindowKVCache(sink_size=args.sink,
                              window_size=(args.full_window if li in full_idx
                                           else args.window))
            for li in range(len(text_model.layers))
        ]

    def encode(text):
        # Match the working Mac NIAH harness: neutral filler + a direct-answer
        # instruction, and append Gemma-4's content-channel marker so short
        # completions don't spend tokens on the thought channel (else recall=0).
        text = text.replace("and does not contain the answer.",
                            "and is unrelated filler.")
        text = (text + "\n\nReturn only the secret code in PREFIX-NNNN format. "
                       "Do not explain, reason, or add any other text.")
        ids = list(tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True))
        try:
            marker = tok.encode("<|channel>content\n<channel|>",
                                add_special_tokens=False)
        except TypeError:
            marker = tok.encode("<|channel>content\n<channel|>")
        if hasattr(marker, "tolist"):
            marker = marker.tolist()
        ids.extend(list(marker))
        return ids

    pool = make_niah_dataset(n_samples=N * 3, haystack_min_lines=args.haystack_lines,
                             haystack_max_lines=args.haystack_lines, seed=0)
    enc = [(encode(s.prompt_text), s.answer_text) for s in pool]
    modal = Counter(len(e[0]) for e in enc).most_common(1)[0][0]
    bucket = [(i, a) for i, a in enc if len(i) == modal][:N]
    while len(bucket) < N:
        bucket += bucket[: N - len(bucket)]
    prompts = [b[0] for b in bucket]
    answers = [b[1] for b in bucket]
    print(f"[mlx-mt] {N} sessions, modal prompt len={modal}", flush=True)

    def recall(toks, ans):
        return ans in tok.decode(toks)

    def prefill_batched(ids_2d):
        """Chunked batched prefill -> (cache, last_logits[N,V])."""
        cache = new_cache()
        chunk = args.prefill_chunk
        T = len(ids_2d[0])
        last = None
        for s in range(0, T, chunk):
            part = [row[s:s + chunk] for row in ids_2d]
            last = model(mx.array(part), cache=cache)
            mx.eval(last)
        return cache, last[:, -1, :]

    def decode_batched(cache, logits, max_tokens):
        B = logits.shape[0]
        nxt = mx.argmax(logits, axis=-1)
        gen = [[int(nxt[i].item())] for i in range(B)]
        mx.eval(nxt)
        t0 = time.perf_counter()
        for _ in range(max_tokens - 1):
            cur = nxt.reshape(B, 1)
            out = model(cur, cache=cache)
            mx.eval(out)
            nxt = mx.argmax(out[:, -1, :], axis=-1)
            for i in range(B):
                gen[i].append(int(nxt[i].item()))
        dt = time.perf_counter() - t0
        return gen, dt

    def decode_batched_padded(cache, logits, max_tokens):
        """L>=2 padded decode: feed the new token duplicated so the
        forward routes through mlx's matrix-matrix (qmm) quantized
        kernel instead of the single-token (qmv) decode kernel suspected
        of the batch>1 bug. Position 0 is the real token (attends only to
        the cache + itself, so its logits == the L=1 decode result);
        position 1 is the duplicate, trimmed afterwards so the cache and
        global offset stay at the true position. B (sessions) untouched.
        """
        B = logits.shape[0]
        nxt = mx.argmax(logits, axis=-1)
        gen = [[int(nxt[i].item())] for i in range(B)]
        mx.eval(nxt)
        t0 = time.perf_counter()
        for _ in range(max_tokens - 1):
            cur = nxt.reshape(B, 1)
            pair = mx.concatenate([cur, cur], axis=1)  # [B, 2], L=2
            out = model(pair, cache=cache)
            mx.eval(out)
            # position 0 == the real next-token prediction (L=1-equivalent)
            nxt = mx.argmax(out[:, 0, :], axis=-1)
            for layer in cache:
                layer.trim(1)  # drop the duplicate (position 1)
            for i in range(B):
                gen[i].append(int(nxt[i].item()))
        dt = time.perf_counter() - t0
        return gen, dt

    decode = decode_batched_padded if args.pad_decode else decode_batched
    if args.pad_decode:
        print("[mlx-mt] decode path: L>=2 padded (qmm, avoids L=1 qmv kernel)",
              flush=True)

    # warmup
    try:
        c, l = prefill_batched([prompts[0]] * min(2, N))
        decode(c, l, 4)
    except Exception as e:  # noqa: BLE001
        print(f"[mlx-mt] warmup note: {e}", flush=True)

    # batched
    cache, logits = prefill_batched(prompts)
    g_b, dt_b = decode(cache, logits, args.max_new_tokens)
    batched_tps = round((N * args.max_new_tokens) / dt_b, 3) if dt_b > 0 else 0.0
    batched_recall = sum(recall(g_b[i], answers[i]) for i in range(N)) / N

    # serialized (one session at a time)
    t0 = time.perf_counter()
    g_s = []
    for i in range(N):
        c, l = prefill_batched([prompts[i]])
        gg, _ = decode(c, l, args.max_new_tokens)
        g_s.append(gg[0])
    # serialized decode-only time: re-time decode alone (prefill excluded for fair tps)
    ser_decode_s = 0.0
    for i in range(N):
        c, l = prefill_batched([prompts[i]])
        _, dt = decode(c, l, args.max_new_tokens)
        ser_decode_s += dt
    serial_tps = round((N * args.max_new_tokens) / ser_decode_s, 3) if ser_decode_s else 0.0
    serial_recall = sum(recall(g_s[i], answers[i]) for i in range(N)) / N

    # Diagnostic: per-row batched-vs-serialized first token + recall, to
    # localize whether batched PREFILL diverges from serialized (batch-1).
    print("[mlx-mt][diag] row | serial_tok0 | batched_tok0 | match | "
          "serial_recall | batched_recall", flush=True)
    for i in range(N):
        s0 = g_s[i][0] if g_s[i] else None
        b0 = g_b[i][0] if g_b[i] else None
        print(f"[mlx-mt][diag] {i:2d} | {s0} | {b0} | {s0 == b0} | "
              f"{recall(g_s[i], answers[i])} | {recall(g_b[i], answers[i])}",
              flush=True)

    speedup = round(batched_tps / serial_tps, 2) if serial_tps else None
    report = {
        "kind": "mlx_batched_multitenant",
        "config": {"sessions": N, "modal_prompt_len": modal,
                   "max_new_tokens": args.max_new_tokens,
                   "verifier_path": args.verifier_path,
                   "kakeya_cache": bool(args.kakeya_cache),
                   "manual_sdpa": bool(args.manual_sdpa),
                   "pad_decode": bool(args.pad_decode)},
        "serialized": {"aggregate_tps": serial_tps, "recall": round(serial_recall, 3)},
        "batched": {"aggregate_tps": batched_tps, "recall": round(batched_recall, 3)},
        "batched_speedup_vs_serialized": speedup,
    }
    print(f"[mlx-mt] N={N}: serialized {serial_tps} tok/s (recall {serial_recall}) | "
          f"batched {batched_tps} tok/s (recall {batched_recall}) | speedup {speedup}x",
          flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[mlx-mt] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
