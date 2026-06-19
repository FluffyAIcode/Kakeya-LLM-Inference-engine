"""ProposerService servicer + RemoteProposer client (design doc §4).

Server side: :class:`ProposerServiceServicer` exposes a map of
``{model_id: proposer}`` over gRPC. Any object satisfying the
``DLMProposer.propose_block`` contract serves — the PyTorch dLM, the
MLX sparse-logits proposer, the DFlash drafter glue, or the model-free
:class:`~inference_engine.distributed.ngram.NGramProposer`. Proposal
compute runs in a worker thread (``asyncio.to_thread``) so a long
diffusion block does not starve capability gossip on the same server.

Client side: :class:`RemoteProposer` is a drop-in ``DLMProposer``
substitute: same ``propose_block`` signature and semantics, same
``ProposerStats`` accounting, so ``SpeculativeDecoder`` drives it
without modification. It uses a synchronous gRPC channel because the
spec-decode loop itself is synchronous (one outstanding proposal at a
time, by construction of the accept/verify dependency).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Mapping, Protocol

import grpc
import grpc.aio

from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)
from kv_cache_proposer.proposer import BlockProposal, ProposerStats

_LOG = logging.getLogger("kakeya.distributed.proposer")

DEFAULT_PROPOSE_TIMEOUT_S = 60.0


class ProposerLike(Protocol):
    """Structural type of every block proposer (ADR 0001 contract)."""

    def propose_block(
        self,
        committed_token_ids: List[int],
        block_size: int,
        num_steps: int,
    ) -> BlockProposal: ...


class RemoteProposerError(RuntimeError):
    """A remote ProposeBlock call failed or returned a malformed block."""


class ProposerServiceServicer(distributed_pb2_grpc.ProposerServiceServicer):
    """Serve one or more in-process proposers to remote verifier loops."""

    def __init__(
        self,
        proposers: Mapping[str, ProposerLike],
        *,
        default_model_id: str = "",
    ) -> None:
        if not proposers:
            raise ValueError("proposers map must be non-empty")
        if default_model_id and default_model_id not in proposers:
            raise ValueError(
                f"default_model_id {default_model_id!r} is not in the proposers map"
            )
        self._proposers = dict(proposers)
        self._default_model_id = default_model_id or next(iter(self._proposers))

    @property
    def model_ids(self) -> List[str]:
        return list(self._proposers)

    async def ProposeBlock(  # noqa: N802 - gRPC casing
        self,
        request: distributed_pb2.ProposeBlockRequest,
        context: grpc.aio.ServicerContext,
    ) -> distributed_pb2.ProposeBlockResponse:
        model_id = request.model_id or self._default_model_id
        proposer = self._proposers.get(model_id)
        if proposer is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"no proposer for model_id {model_id!r}; "
                f"serving: {sorted(self._proposers)}",
            )
        try:
            proposal = await asyncio.to_thread(
                proposer.propose_block,
                list(request.committed_token_ids),
                int(request.block_size),
                int(request.num_steps),
            )
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return distributed_pb2.ProposeBlockResponse(
            token_ids=proposal.tokens,
            diffusion_steps=proposal.diffusion_steps,
            forward_passes=proposal.forward_passes,
            peak_activation_bytes=proposal.peak_activation_bytes,
        )


def add_proposer_service(
    server: grpc.aio.Server,
    proposers: Mapping[str, ProposerLike],
    *,
    default_model_id: str = "",
) -> ProposerServiceServicer:
    """Register a ProposerService for ``proposers`` on ``server``."""
    servicer = ProposerServiceServicer(
        proposers, default_model_id=default_model_id,
    )
    distributed_pb2_grpc.add_ProposerServiceServicer_to_server(servicer, server)
    return servicer


class RemoteProposer:
    """Drop-in ``DLMProposer`` substitute backed by a remote node.

    Owns a synchronous insecure channel to ``address`` (the proposer
    node's ``grpc_address`` from its capability card). Close with
    :meth:`close` or use as a context manager.
    """

    def __init__(
        self,
        address: str,
        *,
        model_id: str = "",
        timeout_s: float = DEFAULT_PROPOSE_TIMEOUT_S,
    ) -> None:
        if not address:
            raise ValueError("address must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.address = address
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.stats = ProposerStats(weight_bytes=0)
        self._channel = grpc.insecure_channel(address)
        self._stub = distributed_pb2_grpc.ProposerServiceStub(self._channel)

    def propose_block(
        self,
        committed_token_ids: List[int],
        block_size: int,
        num_steps: int,
    ) -> BlockProposal:
        request = distributed_pb2.ProposeBlockRequest(
            committed_token_ids=committed_token_ids,
            block_size=block_size,
            num_steps=num_steps,
            model_id=self.model_id,
        )
        try:
            response = self._stub.ProposeBlock(request, timeout=self.timeout_s)
        except grpc.RpcError as exc:
            raise RemoteProposerError(
                f"ProposeBlock to {self.address} failed: "
                f"{exc.code().name}: {exc.details()}"
            ) from exc
        tokens = list(response.token_ids)
        if len(tokens) != block_size:
            # Same malformed-block refusal as SpeculativeDecoder's
            # in-process check: never continue on a short/long draft.
            raise RemoteProposerError(
                f"remote proposer at {self.address} returned {len(tokens)} "
                f"tokens; expected exactly {block_size}"
            )
        self.stats.total_blocks += 1
        self.stats.total_diffusion_steps += int(response.diffusion_steps)
        self.stats.total_forward_passes += int(response.forward_passes)
        self.stats.peak_activation_bytes = max(
            self.stats.peak_activation_bytes, int(response.peak_activation_bytes),
        )
        return BlockProposal(
            tokens=tokens,
            diffusion_steps=int(response.diffusion_steps),
            forward_passes=int(response.forward_passes),
            peak_activation_bytes=int(response.peak_activation_bytes),
        )

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "RemoteProposer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
