"""Self-contained upstream MLX batched-decode probe (no inference_engine imports).

ADR 0014 root-caused the Mac batched-recall break to an MLX **core-kernel
bug at B>1, L=1** (4-bit quantized single-token decode): batched (batch>1)
decode over the gemma-4 verifier returned per-session recall 0.125 while
serialized was 1.0, and every Python-level cause (rotation / cache / shared-KV
/ embedding / mask / SDPA) was ruled out. PR #135's L>=2 padding workaround
confirmed it (recall 0.125 -> 1.0 by avoiding the L=1 path).

This probe re-tests the *plain* L=1 batched decode after an ``mlx``/``mlx-lm``
upgrade, using ONLY the mlx_lm public API (``mlx_lm.load`` +
``model.make_cache()``). It deliberately imports nothing from
``inference_engine`` so it is decoupled from our cache wrapper and isolates the
upstream runtime: if batched recall is now 1.0, the upstream kernel bug is
fixed and Mac can batch decode natively (parallel, no padding tax).

NIAH prompts are built inline (mirroring ``inference_engine.v04.niah_eval``)
to avoid the torch import that module carries.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

_NEEDLE_PREFIXES = (
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA",
    "ETA", "THETA", "IOTA", "KAPPA", "ORCHID", "PINE",
    "MAPLE", "OAK", "BIRCH",
)


def _dist_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_niah(n: int, lines: int, seed: int = 0) -> List[Tuple[str, str]]:
    """Inline NIAH samples: (prompt_text, answer_code). Mirrors
    ``inference_engine.v04.niah_eval.make_niah_dataset`` construction."""
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    for _ in range(n):
        prefix = rng.choice(_NEEDLE_PREFIXES)
        code = f"{prefix}-{rng.randint(1000, 9999)}"
        needle = f"\nIMPORTANT: the secret code is {code}.\n"
        padding = [
            f"Note {i:04d}: this paragraph is unrelated padding "
            "and is unrelated filler."
            for i in range(lines)
        ]
        padding.insert(rng.randint(4, lines - 4), needle)
        prompt = "\n".join(padding) + "\nQuestion: what is the secret code? Answer:"
        out.append((prompt, code))
    return out


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

    mlx_v = _dist_version("mlx")
    mlx_lm_v = _dist_version("mlx-lm") or _dist_version("mlx_lm")
    print(f"[probe] mlx={mlx_v} mlx_lm={mlx_lm_v}", flush=True)
    print(f"[probe] loading {args.verifier_path}", flush=True)
    model, tok = mlx_lm.load(args.verifier_path)
    N = args.sessions

    def encode(text: str) -> List[int]:
        text = (text + "\n\nReturn only the secret code in PREFIX-NNNN format. "
                       "Do not explain, reason, or add any other text.")
        ids = list(tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True))
        try:
            marker = tok.encode("<|channel>content\n<channel|>",
                                add_special_tokens=False)
        except TypeError:
            marker = tok.encode("<|channel>content\n<channel|>")
        if hasattr(marker, "tolist"):
            marker = marker.tolist()
        ids.extend(list(marker))
        return ids

    pool = make_niah(N * 3, args.haystack_lines, seed=0)
    enc = [(encode(p), a) for p, a in pool]
    modal = Counter(len(e[0]) for e in enc).most_common(1)[0][0]
    bucket = [(i, a) for i, a in enc if len(i) == modal][:N]
    while len(bucket) < N:
        bucket += bucket[: N - len(bucket)]
    prompts = [b[0] for b in bucket]
    answers = [b[1] for b in bucket]
    print(f"[probe] {N} sessions, modal prompt len={modal}", flush=True)

    def recall(toks, ans):
        return ans in tok.decode(toks)

    def prefill(ids_2d):
        cache = model.make_cache()
        T = len(ids_2d[0])
        last = None
        for s in range(0, T, args.prefill_chunk):
            part = [row[s:s + args.prefill_chunk] for row in ids_2d]
            last = model(mx.array(part), cache=cache)
            mx.eval(last)
        return cache, last[:, -1, :]

    def decode(cache, logits, max_tokens):
        B = logits.shape[0]
        nxt = mx.argmax(logits, axis=-1)
        gen = [[int(nxt[i].item())] for i in range(B)]
        mx.eval(nxt)
        t0 = time.perf_counter()
        for _ in range(max_tokens - 1):
            out = model(nxt.reshape(B, 1), cache=cache)  # L=1 batched decode
            mx.eval(out)
            nxt = mx.argmax(out[:, -1, :], axis=-1)
            for i in range(B):
                gen[i].append(int(nxt[i].item()))
        dt = time.perf_counter() - t0
        return gen, dt

    # warmup
    try:
        c, l = prefill([prompts[0]] * min(2, N))
        decode(c, l, 4)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] warmup note: {e}", flush=True)

    # batched (the path that broke: B>1, L=1)
    cache, logits = prefill(prompts)
    g_b, dt_b = decode(cache, logits, args.max_new_tokens)
    batched_tps = round((N * args.max_new_tokens) / dt_b, 3) if dt_b > 0 else 0.0
    batched_recall = sum(recall(g_b[i], answers[i]) for i in range(N)) / N

    # serialized ground truth (B=1)
    g_s = []
    ser_decode_s = 0.0
    for i in range(N):
        c, l = prefill([prompts[i]])
        gg, dt = decode(c, l, args.max_new_tokens)
        g_s.append(gg[0])
        ser_decode_s += dt
    serial_tps = round((N * args.max_new_tokens) / ser_decode_s, 3) if ser_decode_s else 0.0
    serial_recall = sum(recall(g_s[i], answers[i]) for i in range(N)) / N

    print("[probe][diag] row | serial_tok0 | batched_tok0 | match | "
          "serial_recall | batched_recall", flush=True)
    for i in range(N):
        s0 = g_s[i][0] if g_s[i] else None
        b0 = g_b[i][0] if g_b[i] else None
        print(f"[probe][diag] {i:2d} | {s0} | {b0} | {s0 == b0} | "
              f"{recall(g_s[i], answers[i])} | {recall(g_b[i], answers[i])}",
              flush=True)

    speedup = round(batched_tps / serial_tps, 2) if serial_tps else None
    upstream_fixed = bool(batched_recall >= serial_recall and batched_recall >= 0.99)
    report = {
        "kind": "mlx_upstream_batch_probe",
        "config": {"sessions": N, "modal_prompt_len": modal,
                   "max_new_tokens": args.max_new_tokens,
                   "verifier_path": args.verifier_path,
                   "mlx_version": mlx_v, "mlx_lm_version": mlx_lm_v,
                   "decode": "native L=1 batched (model.make_cache())"},
        "serialized": {"aggregate_tps": serial_tps, "recall": round(serial_recall, 3)},
        "batched": {"aggregate_tps": batched_tps, "recall": round(batched_recall, 3)},
        "batched_speedup_vs_serialized": speedup,
        "upstream_l1_batch_bug_fixed": upstream_fixed,
    }
    print(f"[probe] mlx={mlx_v} mlx_lm={mlx_lm_v}: serialized {serial_tps} tok/s "
          f"(recall {serial_recall}) | batched {batched_tps} tok/s "
          f"(recall {batched_recall}) | speedup {speedup}x | "
          f"upstream_fixed={upstream_fixed}", flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[probe] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
