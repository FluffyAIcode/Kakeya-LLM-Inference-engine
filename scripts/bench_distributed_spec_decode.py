"""Benchmark the ADR 0009 distributed spec-decode path on three axes:

1. token throughput — local greedy baseline vs distributed spec-decode (tok/s);
2. bounded-KV footprint — the sink+window verifier's resident K/V bytes, which
   stay CONSTANT in context length, vs the equivalent full-attention K/V;
3. gRPC RTT — per-block ProposeBlock round-trip latency to the remote proposer
   (localhost vs cross-host shows the network cost of remote drafts).

Point ``--peer`` at a running ProposerService (see
scripts/demo_distributed_spec_decode.py --role proposer-node, or
scripts/run_distributed_bench.sh which starts one locally).

CLI plumbing around tested library code; exempt from unit-test coverage by the
same convention as start_grpc_runtime_server.py / demo_distributed_spec_decode.py.
"""
from __future__ import annotations
import argparse, time, statistics, json
import torch

from inference_engine.distributed.proposer_service import RemoteProposer
from inference_engine.distributed.capability import NGRAM_MODEL_ID
from inference_engine.distributed.spec_decode import DistributedSpeculativeDecoder
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

PROMPT = ("List the numbers from 1 to 30, separated by commas, then repeat "
          "the same list again:")

def pctl(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0; f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)

def greedy(verifier, prompt_ids, n):
    verifier.reset(); verifier.prefill(prompt_ids)
    out = [int(torch.argmax(verifier.next_token_logits).item())]
    while len(out) < n:
        verifier.append_token(out[-1])
        out.append(int(torch.argmax(verifier.next_token_logits).item()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rtt-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--long-tokens", type=int, default=1024)
    args = ap.parse_args()

    print(f"\n================  {args.label}  (peer={args.peer})  ================")

    # ---------- 1. gRPC RTT (single ProposeBlock round-trip) ----------
    rp = RemoteProposer(args.peer, model_id=NGRAM_MODEL_ID)
    ctx = [1, 2, 3, 4, 5, 6, 7, 8] * 16  # 128-token repetitive context
    for _ in range(15):  # warm up channel
        rp.propose_block(ctx, args.block_size, 1)
    lat = []
    for _ in range(args.rtt_samples):
        t = time.perf_counter(); rp.propose_block(ctx, args.block_size, 1)
        lat.append((time.perf_counter() - t) * 1000.0)
    rp.close()
    print(f"[RTT]  ProposeBlock n={len(lat)}  "
          f"mean={statistics.mean(lat):.3f}ms  p50={pctl(lat,50):.3f}ms  "
          f"p90={pctl(lat,90):.3f}ms  p99={pctl(lat,99):.3f}ms  "
          f"min={min(lat):.3f}ms  max={max(lat):.3f}ms")

    # ---------- 2. token throughput (baseline vs distributed) ----------
    verifier = SinkWindowVerifier(VerifierConfig(
        model_id=args.verifier_id, dtype=torch.bfloat16, device="cpu",
        sink_size=4, window_size=64))
    prompt = verifier.tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, tokenize=True, return_dict=False)

    t = time.perf_counter(); greedy(verifier, prompt, args.max_new_tokens)
    bt = time.perf_counter() - t
    verifier.reset()
    dec = DistributedSpeculativeDecoder(
        RemoteProposer(args.peer, model_id=NGRAM_MODEL_ID), verifier,
        block_size=args.block_size, num_diffusion_steps=1)
    t = time.perf_counter()
    res = dec.generate(prompt, max_new_tokens=args.max_new_tokens)
    dt = time.perf_counter() - t
    n = len(res.output_token_ids)
    dec.proposer.close()
    print(f"[THRUPUT]  baseline(local greedy)={args.max_new_tokens/bt:6.2f} tok/s "
          f"({bt:.2f}s)   distributed={n/dt:6.2f} tok/s ({dt:.2f}s)   "
          f"acceptance={res.acceptance_rate:.3f} ({res.total_accepted}/{res.total_proposed})")

    # ---------- 3. bounded-KV footprint (constant vs context length) ----------
    bpt = verifier._bytes_per_kv_token
    bound = verifier.config.sink_size + verifier.config.window_size
    long_prompt = (prompt * (args.long_tokens // len(prompt) + 1))[: args.long_tokens]
    verifier.reset(); verifier.prefill(long_prompt)
    nxt = int(torch.argmax(verifier.next_token_logits).item())
    for _ in range(args.max_new_tokens):  # decode further; cache must stay bounded
        verifier.append_token(nxt); nxt = int(torch.argmax(verifier.next_token_logits).item())
    total_ctx = len(long_prompt) + args.max_new_tokens
    live = verifier.cache_logical_size * bpt
    unbounded = total_ctx * bpt
    print(f"[BOUNDED-KV]  ctx={total_ctx} tok  sink+window={bound}  "
          f"cache_logical_size={verifier.cache_logical_size} slots  "
          f"bytes/kv-token={bpt}")
    print(f"[BOUNDED-KV]  resident KV={live/1e6:.3f} MB (peak={verifier.stats.peak_kv_bytes/1e6:.3f} MB)  "
          f"vs full-attention KV={unbounded/1e6:.3f} MB  "
          f"=> {unbounded/live:.1f}x smaller, CONSTANT in context length")

if __name__ == "__main__":
    main()
