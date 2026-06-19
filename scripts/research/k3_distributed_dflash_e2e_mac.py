"""End-to-end check for the distributed DFlash+f_θ path with REAL models.

Host A is the gemma-4 MLX verifier (this script). The DFlash drafter + f_θ
proposer is either:
  * in-process (default): an MLXRestorationDraftEngine in this process (single
    model load — validates the protocol + codec without a 2x load), or
  * --grpc: a real loopback gRPC DFlashProposerService (same process, bg thread),
  * --remote-addr HOST:PORT: a REMOTE gRPC proposer (e.g. a torch DFlash+f_θ
    engine on a GPU) — the true cross-host run.

For each, it runs the SAME verifier with block_size=1 (pure greedy baseline) and
block_size=B (distributed spec-decode) and asserts byte-identical output, then
reports acceptance, tok/s, and per-RPC RTT + payload bytes.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _TimingProposer:
    """Wraps a proposer, timing each RPC + counting WireTensor payload bytes."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.t = {"restore": [], "seed_context": [], "draft_block": [], "extend_context": []}
        self.bytes = {"seed_context": 0, "extend_context": 0, "restore": 0}

    @staticmethod
    def _wbytes(aux) -> int:
        import numpy as np
        return int(sum(np.asarray(w.data).nbytes for w in aux))

    def restore(self, prompt_ids, **kw):
        import numpy as np
        t0 = time.perf_counter()
        r = self.inner.restore(prompt_ids, **kw)
        self.t["restore"].append((time.perf_counter() - t0) * 1000)
        self.bytes["restore"] += int(sum(
            np.asarray(k.data).nbytes + np.asarray(v.data).nbytes for (_, k, v) in r.restored))
        return r

    def seed_context(self, aux, positions):
        self.bytes["seed_context"] += self._wbytes(aux)
        t0 = time.perf_counter()
        r = self.inner.seed_context(aux, positions)
        self.t["seed_context"].append((time.perf_counter() - t0) * 1000)
        return r

    def draft_block(self, **kw):
        t0 = time.perf_counter()
        r = self.inner.draft_block(**kw)
        self.t["draft_block"].append((time.perf_counter() - t0) * 1000)
        return r

    def extend_context(self, aux, positions):
        self.bytes["extend_context"] += self._wbytes(aux)
        t0 = time.perf_counter()
        r = self.inner.extend_context(aux, positions)
        self.t["extend_context"].append((time.perf_counter() - t0) * 1000)
        return r

    def close(self):
        return self.inner.close()

    def report(self) -> str:
        import statistics
        out = []
        for name in ("restore", "seed_context", "draft_block", "extend_context"):
            v = self.t[name]
            if not v:
                continue
            p50 = sorted(v)[len(v) // 2]
            b = self.bytes.get(name, 0)
            out.append(f"{name}: n={len(v)} mean={statistics.mean(v):.2f}ms p50={p50:.2f}ms"
                       + (f" bytes={b/1e6:.2f}MB" if b else ""))
        return " | ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier-path", required=True)
    ap.add_argument("--drafter-id", default="")
    ap.add_argument("--f-theta-dir", default="")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grpc", action="store_true")
    ap.add_argument("--remote-addr", default="", help="HOST:PORT of a remote DFlashProposerService (cross-host)")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    import torch

    from inference_engine.backends.mlx.cross_model_dlm_verifier import resolve_mlx_text_model
    from inference_engine.backends.mlx.dflash_distributed import (
        InProcessDFlashProposer, MLXRestorationDraftEngine, MLXRestoringVerifierAdapter,
    )
    from inference_engine.backends.mlx.fused_specdecode import MLXRestoredIncrementalVerifier
    from inference_engine.distributed.dflash_service import RemoteDFlashProposer
    from inference_engine.distributed.fused_decode import DistributedFusedDecoder
    from scripts.research.k3_dflash_mlx_bridge import mx_to_torch

    dev = torch.device(args.device)
    _log(f"[e2e] loading MLX verifier {args.verifier_path}")
    mlx_model, tok = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    embed_scale = float(getattr(text_model, "embed_scale", 1.0))
    bridge = lambda a: mx_to_torch(a, dtype=torch.float32, device=dev)

    remote = bool(args.remote_addr)
    engine = None
    if remote:
        # aux_layer_ids must match the drafter; for a remote engine we still need
        # them on host A to capture aux. Use the drafter config (downloaded) or a
        # fixed gemma-4 DFlash value passed via --drafter-id (load cfg only).
        from inference_engine.v04 import DFlashDrafter
        aux_layer_ids = tuple(DFlashDrafter.from_pretrained(
            args.drafter_id, dtype=torch.float32).cfg.aux_layer_ids)
        _log(f"[e2e] REMOTE proposer at {args.remote_addr} (aux_layer_ids={aux_layer_ids})")
    else:
        from inference_engine.v04 import DFlashDrafter, FThetaProjection
        _log(f"[e2e] loading drafter {args.drafter_id} + f_θ {args.f_theta_dir} on {dev}")
        drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=torch.float32).to(dev).eval()
        for p in drafter.parameters():
            p.requires_grad_(False)
        f_theta = FThetaProjection.from_pretrained(args.f_theta_dir, dtype=torch.float32, device=dev)
        aux_layer_ids = tuple(drafter.cfg.aux_layer_ids)
        engine = MLXRestorationDraftEngine(
            mlx_model=mlx_model, text_model=text_model, drafter=drafter, f_theta=f_theta,
            embed_scale=embed_scale, device=dev, sink=args.sink, window=args.window,
            force_f_theta=True)

    raw = MLXRestoredIncrementalVerifier(
        mlx_model, embed_scale=embed_scale, aux_layer_ids=aux_layer_ids, bridge_to_torch=bridge)
    verifier = MLXRestoringVerifierAdapter(
        adapter=raw, mlx_model=mlx_model, aux_layer_ids=aux_layer_ids,
        embed_scale=embed_scale, bridge=bridge)

    prompt_ids = [int(x) for x in tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=True, return_dict=False)]
    _log(f"[e2e] prompt={len(prompt_ids)} tok, max_new={args.max_new_tokens}, block={args.block_size}")

    stop = lambda: None
    if args.grpc and not remote:
        addr, stop = _grpc_server(engine)
    elif remote:
        addr = args.remote_addr

    def make_proposer(session_id: str):
        if remote or args.grpc:
            return RemoteDFlashProposer(addr, session_id=session_id, timeout_s=300.0)
        return InProcessDFlashProposer(engine, session_id=session_id,
                                       sink=args.sink, window=args.window)

    def run(block_size: int, session_id: str):
        prop = make_proposer(session_id)
        timed = _TimingProposer(prop)
        dec = DistributedFusedDecoder(timed, verifier, block_size=block_size,
                                      sink=args.sink, window=args.window)
        t0 = time.perf_counter()
        res = dec.generate(prompt_ids, args.max_new_tokens)
        dt = time.perf_counter() - t0
        prop.close()
        return res, dt, timed

    base_res, base_s, _ = run(1, "base")
    baseline = base_res.output_token_ids
    _log(f"[e2e] greedy baseline (block=1): {len(baseline)} tok in {base_s:.2f}s "
         f"({len(baseline)/base_s:.2f} tok/s)")

    res, dist_s, timed = run(args.block_size, "dist")
    stop()
    n = len(res.output_token_ids)
    _log(f"[e2e] distributed (block={args.block_size}): {n} tok in {dist_s:.2f}s "
         f"({n/dist_s:.2f} tok/s) blocks={res.blocks} acceptance={res.acceptance_rate:.3f} "
         f"({res.total_accepted}/{res.total_proposed})")
    _log(f"[e2e] RTT/payload per RPC: {timed.report()}")
    _log(f"[e2e] output text:\n{tok.decode(res.output_token_ids)}")

    if res.output_token_ids == baseline[:n]:
        print(f"[e2e] PASS byte-identical-to-greedy ({n} tok, acceptance={res.acceptance_rate:.3f}, "
              f"baseline={len(baseline)/base_s:.2f} tok/s, dist={n/dist_s:.2f} tok/s)")
        return 0
    print("[e2e] FAIL divergence from greedy", file=sys.stderr)
    print(f"  baseline={baseline[:n]}\n  dist    ={res.output_token_ids}", file=sys.stderr)
    return 1


def _grpc_server(engine):
    import asyncio
    import threading

    import grpc

    from inference_engine.distributed.dflash_service import add_dflash_proposer_service

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
    threading.Thread(target=lambda: (asyncio.set_event_loop(loop), loop.run_until_complete(_serve())),
                     daemon=True).start()
    ready.wait(timeout=30)
    return holder["addr"], (lambda: loop.call_soon_threadsafe(
        lambda: asyncio.ensure_future(holder["server"].stop(0))))


if __name__ == "__main__":
    raise SystemExit(main())
