"""Localize the mlx_lm gemma-4 batch>1 decode bug: per-layer hidden-state diff,
batched-row-i vs serialized-i, at decode step 1.

Prefill is known correct (first decoded token matches per row); the divergence
is in the batched decode forward. This dumps, per decoder layer, the max-abs
difference between the batched forward's row-i output and the serialized
single-row forward's output for the SAME session + SAME fed token. The first
layer whose diff jumps locates the bug (layer 0 → RoPE/embed/per-layer-input;
a sliding layer → sliding mask/cache; a full-attn layer → full path; the first
KV-shared layer → shared-KV plumbing).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--haystack-lines", type=int, default=15)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    sys.path.insert(0, "sdks/python")
    from inference_engine.v04 import make_niah_dataset
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model,
    )

    model, tok = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(model)
    layers = text_model.layers
    n_layers = len(layers)
    layer_types = [getattr(l, "layer_type", "?") for l in layers]
    first_shared = n_layers - getattr(model.args, "num_kv_shared_layers", 0)
    print(f"[diff] layers={n_layers} first_kv_shared_idx={first_shared}", flush=True)

    R = args.rows
    pool = make_niah_dataset(n_samples=R * 3, haystack_min_lines=args.haystack_lines,
                             haystack_max_lines=args.haystack_lines, seed=0)

    def encode(text):
        text = text.replace("and does not contain the answer.",
                            "and is unrelated filler.")
        text = text + "\n\nReturn only the secret code in PREFIX-NNNN format."
        ids = list(tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True))
        try:
            m = tok.encode("<|channel>content\n<channel|>", add_special_tokens=False)
        except TypeError:
            m = tok.encode("<|channel>content\n<channel|>")
        ids.extend(list(m if not hasattr(m, "tolist") else m.tolist()))
        return ids

    enc = [encode(s.prompt_text) for s in pool]
    modal = Counter(len(e) for e in enc).most_common(1)[0][0]
    prompts = [e for e in enc if len(e) == modal][:R]
    while len(prompts) < R:
        prompts += prompts[: R - len(prompts)]
    print(f"[diff] {R} rows, prompt len={modal}", flush=True)

    # capture per-layer output hidden by monkey-patching DecoderLayer.__call__
    DecoderLayer = type(layers[0])
    orig_call = DecoderLayer.__call__
    captured: List = []

    def patched(self, *a, **k):
        out = orig_call(self, *a, **k)
        captured.append(out[0])      # h: [B, L, D]
        return out

    def prefill(ids_2d):
        cache = model.make_cache()
        out = model(mx.array(ids_2d), cache=cache)
        mx.eval(out)
        return cache, out[:, -1, :]

    def decode_capture(token_ids_2d, cache):
        captured.clear()
        DecoderLayer.__call__ = patched
        try:
            out = model(mx.array(token_ids_2d), cache=cache)
            mx.eval(out)
        finally:
            DecoderLayer.__call__ = orig_call
        return list(captured), out

    # serialized: prefill + first token + capture decode step
    ser_caches, ser_tok0, ser_layers = [], [], []
    for i in range(R):
        c, lg = prefill([prompts[i]])
        t0 = int(mx.argmax(lg, axis=-1).item())
        ser_tok0.append(t0)
        caps, _ = decode_capture([[t0]], c)
        ser_layers.append(caps)        # n_layers x [1,1,D]

    # batched: prefill + first tokens + capture decode step (same tokens)
    cb, lgb = prefill(prompts)
    bat_tok0 = [int(mx.argmax(lgb[i], axis=-1).item()) for i in range(R)]
    caps_b, _ = decode_capture([[t] for t in bat_tok0], cb)  # n_layers x [R,1,D]

    print(f"[diff] tok0 serial={ser_tok0} batched={bat_tok0} "
          f"match={ser_tok0 == bat_tok0}", flush=True)
    print("[diff] layer | type | shared? | max|Δ| row0 | row1 ...", flush=True)
    first_div = None
    for li in range(n_layers):
        hb = caps_b[li]                # [R,1,D]
        diffs = []
        for i in range(R):
            d = float(mx.max(mx.abs(hb[i:i + 1] - ser_layers[i][li])).item())
            diffs.append(round(d, 4))
        shared = "shared" if li >= first_shared else ""
        mark = ""
        if first_div is None and max(diffs) > 1e-2:
            first_div = li
            mark = "  <-- FIRST DIVERGENCE"
        print(f"[diff] {li:2d} | {layer_types[li]:18s} | {shared:6s} | {diffs}{mark}",
              flush=True)
    print(f"[diff] FIRST DIVERGENT LAYER = {first_div} "
          f"(type={layer_types[first_div] if first_div is not None else None}, "
          f"shared={first_div is not None and first_div >= first_shared})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
