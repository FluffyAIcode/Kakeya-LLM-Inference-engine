"""Two-node distributed spec decode demo (ADR 0009, v0.5-M1).

Demonstrates the milestone end-to-end on two processes (two Mac minis
in production; two terminals on one host for the demo):

1. Both nodes serve ``kakeya.v1.CapabilityService`` and gossip cards.
2. The proposer node serves ``kakeya.v1.ProposerService`` with the
   model-free n-gram (prompt-lookup) proposer.
3. The verifier node discovers the fleet, plans placement
   (``plan_spec_decode_placement``), and runs
   ``DistributedSpeculativeDecoder`` — drafts come from the remote
   node, greedy verification stays local.
4. The verifier node also runs a plain greedy decode with the same
   verifier and asserts byte-identical output (the correctness-
   containment guarantee: remote drafts can change throughput, never
   tokens).

Terminal 1 (proposer node)::

    PYTHONPATH=. python3 scripts/demo_distributed_spec_decode.py \
        --role proposer-node --bind 127.0.0.1:50061 --node-id node-b

Terminal 2 (verifier node)::

    PYTHONPATH=. python3 scripts/demo_distributed_spec_decode.py \
        --role verifier-node --bind 127.0.0.1:50060 --node-id node-a \
        --peer 127.0.0.1:50061 --verifier-id Qwen/Qwen3-0.6B

CLI plumbing around tested library code; exempt from unit-test
coverage by the same convention as start_grpc_runtime_server.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import List

_LOG = logging.getLogger("kakeya.demo.distributed")


def _self_card(args: argparse.Namespace, *, roles: List[str]):
    import platform as _platform

    from inference_engine.distributed.capability import (
        NGRAM_MODEL_ID,
        CapabilityRole,
        ModelCapability,
        NodeCapability,
    )

    models = []
    if "verifier" in roles:
        models.append(
            ModelCapability(
                model_id=args.verifier_id,
                role=CapabilityRole.VERIFIER,
                quantization="bf16",
            ),
        )
    if "proposer" in roles:
        models.append(
            ModelCapability(
                model_id=NGRAM_MODEL_ID,
                role=CapabilityRole.PROPOSER,
                quantization="none",
            ),
        )
    return NodeCapability(
        node_id=args.node_id,
        grpc_address=args.advertise or args.bind,
        platform=f"{_platform.system()}-{_platform.machine()}".lower(),
        models=tuple(models),
        announced_at_unix=time.time(),
        ttl_seconds=args.capability_ttl_s,
    )


async def _run_proposer_node(args: argparse.Namespace) -> int:
    import grpc.aio

    from inference_engine.distributed.capability import (
        NGRAM_MODEL_ID,
        CapabilityRegistry,
    )
    from inference_engine.distributed.exchange import add_capability_service
    from inference_engine.distributed.ngram import NGramProposer
    from inference_engine.distributed.proposer_service import add_proposer_service

    registry = CapabilityRegistry(self_card=_self_card(args, roles=["proposer"]))
    server = grpc.aio.server()
    add_capability_service(server, registry)
    add_proposer_service(server, {NGRAM_MODEL_ID: NGramProposer()})
    server.add_insecure_port(args.bind)
    await server.start()
    _LOG.info(
        "proposer node %s serving CapabilityService + ProposerService(ngram) on %s",
        args.node_id, args.bind,
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=1.0)
    return 0


def _greedy_baseline(verifier, prompt_ids: List[int], max_new_tokens: int) -> List[int]:
    """Plain greedy AR decode with the same sink+window verifier."""
    import torch

    verifier.reset()
    verifier.prefill(prompt_ids)
    out: List[int] = []
    next_tok = int(torch.argmax(verifier.next_token_logits).item())
    out.append(next_tok)
    while len(out) < max_new_tokens:
        verifier.append_token(next_tok)
        next_tok = int(torch.argmax(verifier.next_token_logits).item())
        out.append(next_tok)
    return out


async def _run_verifier_node(args: argparse.Namespace) -> int:
    import grpc.aio
    import torch

    from inference_engine.distributed.capability import CapabilityRegistry
    from inference_engine.distributed.exchange import (
        add_capability_service,
        exchange_once,
    )
    from inference_engine.distributed.placement import plan_spec_decode_placement
    from inference_engine.distributed.spec_decode import (
        DistributedSpeculativeDecoder,
    )
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

    registry = CapabilityRegistry(self_card=_self_card(args, roles=["verifier"]))
    server = grpc.aio.server()
    add_capability_service(server, registry)
    server.add_insecure_port(args.bind)
    await server.start()
    _LOG.info("verifier node %s serving CapabilityService on %s",
              args.node_id, args.bind)

    # --- 1. Discover the fleet -------------------------------------
    report = await exchange_once(registry, args.peer)
    if report.errors:
        print(f"[demo] capability exchange errors: {report.errors}", file=sys.stderr)
        await server.stop(grace=0.5)
        return 2
    snapshot = registry.snapshot()
    print(f"[demo] fleet view after one gossip round ({len(snapshot)} node(s)):")
    for card in snapshot:
        print(f"  - {card.node_id} @ {card.grpc_address} "
              f"models={[(m.model_id, m.role.name) for m in card.models]}")

    # --- 2. Plan placement ------------------------------------------
    placement = plan_spec_decode_placement(
        snapshot, verifier_model_id=args.verifier_id,
    )
    print(f"[demo] placement: {placement.render()}")

    # --- 3. Load the verifier locally --------------------------------
    _LOG.info("loading verifier %s (cpu)", args.verifier_id)
    verifier = SinkWindowVerifier(VerifierConfig(
        model_id=args.verifier_id,
        dtype=torch.bfloat16, device="cpu",
        sink_size=args.sink, window_size=args.window,
    ))
    prompt_ids = verifier.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
    )

    # --- 4. Greedy baseline (same verifier, local only) --------------
    t0 = time.perf_counter()
    baseline = _greedy_baseline(verifier, prompt_ids, args.max_new_tokens)
    baseline_s = time.perf_counter() - t0
    print(f"[demo] greedy baseline: {len(baseline)} tokens in {baseline_s:.2f}s")

    # --- 5. Distributed spec decode ----------------------------------
    verifier.reset()
    decoder = DistributedSpeculativeDecoder.from_placement(
        placement, verifier,
        block_size=args.block_size, num_diffusion_steps=1,
    )
    t0 = time.perf_counter()
    result = decoder.generate(prompt_ids, max_new_tokens=args.max_new_tokens)
    spec_s = time.perf_counter() - t0
    decoder.proposer.close()

    print(f"[demo] distributed spec decode: {len(result.output_token_ids)} tokens "
          f"in {spec_s:.2f}s")
    print(f"[demo] remote proposer: node={placement.proposer_node.node_id} "
          f"blocks={result.proposer_forward_calls} "
          f"acceptance_rate={result.acceptance_rate:.3f} "
          f"accepted={result.total_accepted}/{result.total_proposed}")
    text = verifier.tokenizer.decode(result.output_token_ids)
    print(f"[demo] output text:\n{text}")

    # --- 6. Correctness containment ----------------------------------
    if result.output_token_ids == baseline[: len(result.output_token_ids)]:
        print("[demo] PASS: spec-decode output is byte-identical to local greedy")
        rc = 0
    else:
        print("[demo] FAIL: spec-decode output diverged from local greedy",
              file=sys.stderr)
        print(f"  baseline: {baseline}", file=sys.stderr)
        print(f"  specdec : {result.output_token_ids}", file=sys.stderr)
        rc = 1

    await server.stop(grace=0.5)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=["proposer-node", "verifier-node"],
                    required=True)
    ap.add_argument("--bind", default="127.0.0.1:50060")
    ap.add_argument("--advertise", default="",
                    help="Address peers should use; defaults to --bind.")
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--peer", action="append", default=[],
                    help="Seed peer address; repeatable (verifier role).")
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt",
                    default="List the numbers from 1 to 30, separated by "
                            "commas, then repeat the exact same list again.")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--capability-ttl-s", type=float, default=120.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.role == "proposer-node":
        return asyncio.run(_run_proposer_node(args))
    return asyncio.run(_run_verifier_node(args))


if __name__ == "__main__":
    sys.exit(main())
