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
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    sys.path.insert(0, "sdks/python")
    from inference_engine.v04 import make_niah_dataset

    print(f"[mlx-mt] loading {args.verifier_path}", flush=True)
    model, tok = mlx_lm.load(args.verifier_path)
    N = args.sessions

    def encode(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True)
        return list(ids)

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
        cache = model.make_cache()
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

    # warmup
    try:
        c, l = prefill_batched([prompts[0]] * min(2, N))
        decode_batched(c, l, 4)
    except Exception as e:  # noqa: BLE001
        print(f"[mlx-mt] warmup note: {e}", flush=True)

    # batched
    cache, logits = prefill_batched(prompts)
    g_b, dt_b = decode_batched(cache, logits, args.max_new_tokens)
    batched_tps = round((N * args.max_new_tokens) / dt_b, 3) if dt_b > 0 else 0.0
    batched_recall = sum(recall(g_b[i], answers[i]) for i in range(N)) / N

    # serialized (one session at a time)
    t0 = time.perf_counter()
    g_s = []
    for i in range(N):
        c, l = prefill_batched([prompts[i]])
        gg, _ = decode_batched(c, l, args.max_new_tokens)
        g_s.append(gg[0])
    # serialized decode-only time: re-time decode alone (prefill excluded for fair tps)
    ser_decode_s = 0.0
    for i in range(N):
        c, l = prefill_batched([prompts[i]])
        _, dt = decode_batched(c, l, args.max_new_tokens)
        ser_decode_s += dt
    serial_tps = round((N * args.max_new_tokens) / ser_decode_s, 3) if ser_decode_s else 0.0
    serial_recall = sum(recall(g_s[i], answers[i]) for i in range(N)) / N

    speedup = round(batched_tps / serial_tps, 2) if serial_tps else None
    report = {
        "kind": "mlx_batched_multitenant",
        "config": {"sessions": N, "modal_prompt_len": modal,
                   "max_new_tokens": args.max_new_tokens,
                   "verifier_path": args.verifier_path},
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
