"""Integration: distributed spec decode against a real verifier (ADR 0009).

The v0.5-M1 correctness gate: drafts served by a *remote* proposer
node (real ``grpc.aio`` ProposerService + CapabilityService over
loopback — the smallest honest model of a second Mac mini) must leave
the verifier's greedy output **byte-identical** to plain local greedy
decoding with the same sink+window verifier. This is the wire-level
analogue of INV-3: draft provenance must not influence committed
tokens.

Threading model: the gRPC services live on the test's asyncio loop;
the synchronous decoder (blocking RemoteProposer channel inside) runs
in a worker thread via ``asyncio.to_thread`` so the loop keeps
serving — the same split a real deployment has between the verifier
node's compute thread and the proposer node's server loop.

Requires real Qwen3-0.6B weights in the HF cache (Mac M4 gate, or any
host that ran ``scripts/kakeya_prewarm.py``).
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import grpc
import pytest
import torch

from inference_engine.distributed.capability import (
    NGRAM_MODEL_ID,
    CapabilityRegistry,
    CapabilityRole,
    ModelCapability,
    NodeCapability,
)
from inference_engine.distributed.exchange import add_capability_service, exchange_once
from inference_engine.distributed.ngram import NGramProposer
from inference_engine.distributed.placement import plan_spec_decode_placement
from inference_engine.distributed.proposer_service import (
    RemoteProposer,
    add_proposer_service,
)
from inference_engine.distributed.spec_decode import DistributedSpeculativeDecoder
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

VERIFIER_ID = "Qwen/Qwen3-0.6B"

# Window large enough that no trimming occurs over the test sequence:
# spec decode is then EXACTLY equivalent to greedy AR (speculative.py
# module docstring), making byte-identity a hard assertion.
SINK, WINDOW = 4, 512
MAX_NEW_TOKENS = 32
PROMPT = (
    "List the numbers from 1 to 30, separated by commas, then repeat "
    "the exact same list again."
)


@pytest.fixture(scope="module")
def verifier() -> SinkWindowVerifier:
    return SinkWindowVerifier(VerifierConfig(
        model_id=VERIFIER_ID,
        dtype=torch.bfloat16,
        device="cpu",
        sink_size=SINK,
        window_size=WINDOW,
    ))


@pytest.fixture(scope="module")
def prompt_ids(verifier) -> List[int]:
    # transformers 5.x returns a dict by default with tokenize=True; request the
    # legacy flat list-of-ids shape so it matches on 4.x and 5.x (same convention
    # as kv_cache_proposer.proposer.encode_chat).
    return verifier.tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


def _greedy_baseline(verifier, prompt_ids: List[int]) -> List[int]:
    verifier.reset()
    verifier.prefill(prompt_ids)
    out: List[int] = []
    next_tok = int(torch.argmax(verifier.next_token_logits).item())
    out.append(next_tok)
    while len(out) < MAX_NEW_TOKENS:
        verifier.append_token(next_tok)
        next_tok = int(torch.argmax(verifier.next_token_logits).item())
        out.append(next_tok)
    return out


async def _start_proposer_node():
    """Boot a full 'proposer node': CapabilityService + ProposerService."""
    server = grpc.aio.server()
    registry_stub_card = NodeCapability(
        node_id="proposer-node",
        grpc_address="placeholder",  # patched once the port is known
        models=(ModelCapability(NGRAM_MODEL_ID, CapabilityRole.PROPOSER),),
        announced_at_unix=time.time(),
    )
    registry = CapabilityRegistry(self_card=registry_stub_card)
    add_capability_service(server, registry)
    add_proposer_service(server, {NGRAM_MODEL_ID: NGramProposer()})
    port = server.add_insecure_port("127.0.0.1:0")
    address = f"127.0.0.1:{port}"
    registry.self_card = NodeCapability(
        node_id="proposer-node",
        grpc_address=address,
        models=registry_stub_card.models,
        announced_at_unix=time.time(),
    )
    await server.start()
    return server, address


def test_remote_drafts_never_change_greedy_output(verifier, prompt_ids):
    async def _run():
        server, address = await _start_proposer_node()
        try:
            # --- Discover + place via the capability plane ----------
            verifier_registry = CapabilityRegistry(self_card=NodeCapability(
                node_id="verifier-node",
                grpc_address="127.0.0.1:0",
                models=(ModelCapability(VERIFIER_ID, CapabilityRole.VERIFIER),),
                announced_at_unix=time.time(),
            ))
            report = await exchange_once(verifier_registry, [address])
            assert report.ok, report.errors
            placement = plan_spec_decode_placement(
                verifier_registry.snapshot(), verifier_model_id=VERIFIER_ID,
            )
            assert placement.proposer_node.node_id == "proposer-node"
            assert placement.verifier_node.node_id == "verifier-node"
            assert not placement.colocated

            # --- Local greedy baseline (worker thread) --------------
            baseline = await asyncio.to_thread(
                _greedy_baseline, verifier, prompt_ids,
            )

            # --- Distributed spec decode (worker thread) ------------
            verifier.reset()
            decoder = DistributedSpeculativeDecoder.from_placement(
                placement, verifier, block_size=8, num_diffusion_steps=1,
            )
            try:
                result = await asyncio.to_thread(
                    decoder.generate, prompt_ids, MAX_NEW_TOKENS,
                )
            finally:
                decoder.proposer.close()

            # Byte-identity: remote drafts must not change committed
            # tokens.
            assert result.output_token_ids == baseline[: len(result.output_token_ids)]
            assert len(result.output_token_ids) == MAX_NEW_TOKENS

            # The remote proposer actually served every block.
            assert decoder.proposer.stats.total_blocks == len(result.proposed_per_block)
            assert decoder.proposer.stats.total_blocks > 0
        finally:
            await server.stop(grace=0.5)

    asyncio.run(_run())


def test_repetitive_prompt_earns_nonzero_remote_acceptance(verifier, prompt_ids):
    """On a self-repeating prompt the n-gram proposer must land at
    least one accepted draft token — evidence the distributed path is
    actually speculating, not just falling through to corrections."""
    async def _run():
        server = grpc.aio.server()
        add_proposer_service(server, {NGRAM_MODEL_ID: NGramProposer()})
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            verifier.reset()
            proposer = RemoteProposer(
                f"127.0.0.1:{port}", model_id=NGRAM_MODEL_ID,
            )
            decoder = DistributedSpeculativeDecoder(
                proposer, verifier, block_size=8, num_diffusion_steps=1,
            )
            try:
                result = await asyncio.to_thread(
                    decoder.generate, prompt_ids, MAX_NEW_TOKENS,
                )
            finally:
                proposer.close()
            assert result.total_accepted > 0
            assert 0.0 < result.acceptance_rate <= 1.0
        finally:
            await server.stop(grace=0.5)

    asyncio.run(_run())
