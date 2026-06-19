"""End-to-end check for the distributed DFlash+f_θ path with REAL models.

Loads the gemma-4 MLX verifier + torch DFlash drafter + f_θ ONCE (avoids a 2x
26B load / OOM), then:

  1. runs a pure greedy baseline on the verifier, and
  2. runs the DistributedFusedDecoder over an InProcessDFlashProposer (the real
     MLXRestorationDraftEngine + MLXRestoringVerifierAdapter, exercising the full
     restore/seed/draft/verify/commit/extend protocol incl. WireTensor codec),

and asserts the distributed output is BYTE-IDENTICAL to greedy (correctness
containment), reporting acceptance + tok/s + per-block timing.

Use --grpc to instead run the proposer behind a real loopback gRPC server (same
process, two threads) to also exercise the wire + measure loopback RTT.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier-path", required=True)
    ap.add_argument("--drafter-id", required=True)
    ap.add_argument("--f-theta-dir", required=True)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grpc", action="store_true",
                    help="route the proposer through a real loopback gRPC server "
                         "(exercises the wire + measures loopback RTT)")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    import torch

    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model,
    )
    from inference_engine.backends.mlx.dflash_distributed import (
        InProcessDFlashProposer,
        MLXRestorationDraftEngine,
        MLXRestoringVerifierAdapter,
    )
    from inference_engine.backends.mlx.fused_specdecode import (
        MLXRestoredIncrementalVerifier,
    )
    from inference_engine.distributed.fused_decode import DistributedFusedDecoder
    from inference_engine.distributed.tensor_codec import wire_to_mlx
    from inference_engine.v04 import DFlashDrafter, FThetaProjection
    from inference_engine.v04.kv_merge import compute_evicted_positions
    from scripts.research.k3_dflash_mlx_bridge import mx_to_torch

    dev = torch.device(args.device)

    _log(f"[e2e] loading MLX verifier {args.verifier_path}")
    mlx_model, tok = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    embed_scale = float(getattr(text_model, "embed_scale", 1.0))

    _log(f"[e2e] loading drafter {args.drafter_id} + f_θ {args.f_theta_dir} on {dev}")
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=torch.float32).to(dev).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    f_theta = FThetaProjection.from_pretrained(args.f_theta_dir, dtype=torch.float32, device=dev)
    aux_layer_ids = tuple(drafter.cfg.aux_layer_ids)

    bridge = lambda a: mx_to_torch(a, dtype=torch.float32, device=dev)

    engine = MLXRestorationDraftEngine(
        mlx_model=mlx_model, text_model=text_model, drafter=drafter, f_theta=f_theta,
        embed_scale=embed_scale, device=dev, sink=args.sink, window=args.window,
        force_f_theta=True)

    raw = MLXRestoredIncrementalVerifier(
        mlx_model, embed_scale=embed_scale, aux_layer_ids=aux_layer_ids,
        bridge_to_torch=bridge)
    verifier = MLXRestoringVerifierAdapter(
        adapter=raw, mlx_model=mlx_model, aux_layer_ids=aux_layer_ids,
        embed_scale=embed_scale, bridge=bridge)

    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=True, return_dict=False)
    prompt_ids = [int(x) for x in prompt_ids]
    _log(f"[e2e] prompt_ids={len(prompt_ids)} tokens, max_new={args.max_new_tokens}, block={args.block_size}")

    # ---- 1. greedy baseline (verifier only, same f_θ restoration) ----------
    base_restore = engine.restore("base", prompt_ids, sink=args.sink, window=args.window,
                                  s5_exact_full_attn=True, model_id="")
    rk = {l: wire_to_mlx(k) for (l, k, v) in base_restore.restored}
    rv = {l: wire_to_mlx(v) for (l, k, v) in base_restore.restored}
    raw._capture_aux = False
    raw.prefill(prompt_ids, restored_k_per_layer=rk, restored_v_per_layer=rv,
                evicted_positions=base_restore.evicted_positions,
                prefill_chunk_size=512, full_kv=False)
    t0 = time.perf_counter()
    baseline: List[int] = [int(mx.argmax(raw.next_token_logits).item())]
    while len(baseline) < args.max_new_tokens:
        raw.append_token(baseline[-1])
        baseline.append(int(mx.argmax(raw.next_token_logits).item()))
    base_s = time.perf_counter() - t0
    engine.close_session("base")
    _log(f"[e2e] greedy baseline: {len(baseline)} tok in {base_s:.2f}s "
         f"({len(baseline)/base_s:.2f} tok/s)")

    # ---- 2. distributed in-process (or gRPC loopback) ----------------------
    if args.grpc:
        proposer, stop = _grpc_proposer(engine, sink=args.sink, window=args.window)
    else:
        proposer, stop = InProcessDFlashProposer(engine, session_id="dist",
                                                 sink=args.sink, window=args.window), (lambda: None)

    dec = DistributedFusedDecoder(proposer, verifier, block_size=args.block_size,
                                  sink=args.sink, window=args.window)
    t0 = time.perf_counter()
    res = dec.generate(prompt_ids, args.max_new_tokens)
    dist_s = time.perf_counter() - t0
    proposer.close()
    stop()

    n = len(res.output_token_ids)
    _log(f"[e2e] distributed: {n} tok in {dist_s:.2f}s ({n/dist_s:.2f} tok/s) "
         f"blocks={res.blocks} acceptance={res.acceptance_rate:.3f} "
         f"({res.total_accepted}/{res.total_proposed})")
    text = tok.decode(res.output_token_ids)
    _log(f"[e2e] output text:\n{text}")

    ok = res.output_token_ids == baseline[:n]
    if ok:
        print(f"[e2e] PASS byte-identical-to-greedy ({n} tokens, "
              f"acceptance={res.acceptance_rate:.3f}, "
              f"baseline={len(baseline)/base_s:.2f} tok/s, dist={n/dist_s:.2f} tok/s)")
        return 0
    print("[e2e] FAIL divergence from greedy", file=sys.stderr)
    print(f"  baseline={baseline[:n]}", file=sys.stderr)
    print(f"  dist    ={res.output_token_ids}", file=sys.stderr)
    return 1


def _grpc_proposer(engine, *, sink: int, window: int):
    """Start a loopback gRPC DFlashProposerService in a background event loop and
    return a (RemoteDFlashProposer, stop_fn) pair."""
    import asyncio
    import threading

    import grpc

    from inference_engine.distributed.dflash_service import (
        RemoteDFlashProposer,
        add_dflash_proposer_service,
    )

    holder = {}
    ready = threading.Event()

    async def _serve():
        server = grpc.aio.server(options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024)])
        add_dflash_proposer_service(server, engine)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        holder["addr"] = f"127.0.0.1:{port}"
        holder["server"] = server
        ready.set()
        await server.wait_for_termination()

    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve())

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    ready.wait(timeout=30)
    remote = RemoteDFlashProposer(holder["addr"], session_id="dist", timeout_s=120.0)

    def _stop():
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(holder["server"].stop(0)))

    return remote, _stop


if __name__ == "__main__":
    raise SystemExit(main())
