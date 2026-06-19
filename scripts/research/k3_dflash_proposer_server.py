"""Serve a remote DFlash+f_θ DFlashProposerService on a CUDA host (ADR 0009 F3).

Loads a torch gemma-4 verifier (for its embedding / drafter-KV capture), the
torch DFlash drafter, and f_θ, wraps them in a TorchRestorationDraftEngine, and
serves the gRPC DFlashProposerService. The gemma-4 MLX verifier on another host
drives it via RemoteDFlashProposer.
"""
from __future__ import annotations

import argparse
import asyncio
import sys


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--f-theta-dir", default="results/research/f_theta_v5_s5_sliding")
    ap.add_argument("--bind", default="0.0.0.0:6006")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import grpc
    import torch
    from transformers import AutoModelForCausalLM

    from inference_engine.distributed.dflash_service import add_dflash_proposer_service
    from inference_engine.v04 import DFlashDrafter, FThetaProjection
    from inference_engine.v04.dflash_distributed_engine import TorchRestorationDraftEngine

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[server] loading verifier {args.verifier_id} ({dtype}) on {dev}", file=sys.stderr, flush=True)
    verifier = AutoModelForCausalLM.from_pretrained(
        args.verifier_id, dtype=dtype, attn_implementation="eager").to(dev).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)
    print(f"[server] loading drafter {args.drafter_id} + f_θ {args.f_theta_dir}", file=sys.stderr, flush=True)
    drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=dtype).to(dev).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    f_theta = FThetaProjection.from_pretrained(args.f_theta_dir, dtype=torch.float32, device=dev)

    engine = TorchRestorationDraftEngine(
        verifier_model=verifier, drafter=drafter, f_theta=f_theta, device=dev,
        sink=args.sink, window=args.window, force_f_theta=True)

    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024)])
    add_dflash_proposer_service(server, engine)
    server.add_insecure_port(args.bind)
    await server.start()
    print(f"[server] DFlashProposerService serving on {args.bind} (ready)", file=sys.stderr, flush=True)
    await server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
