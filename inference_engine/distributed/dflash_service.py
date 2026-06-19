"""DFlashProposerService servicer + RemoteDFlashProposer client (ADR 0009 §4 F3).

Splits the Kakeya engine across hosts: a gemma-4 verifier on host A drives a
remote DFlash drafter + f_θ projection on host B. This module is the wire glue;
the actual drafter/f_θ math lives behind the framework-neutral
:class:`RestorationDraftEngine` contract (the real torch engine is
``inference_engine.distributed.dflash_engine``; tests inject a fake).

Tensors cross the wire as framework-neutral :class:`~inference_engine.distributed
.tensor_codec.WireTensor` (proto ``Tensor``); the engine and the caller convert
to/from torch/mlx at their own boundaries. Correctness containment is unchanged:
the verifier's local greedy verify decides every token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import grpc

from inference_engine.distributed.tensor_codec import (
    WireTensor,
    from_proto_fields,
    to_proto_fields,
)
from inference_engine.server.proto_gen.kakeya.v1 import distributed_pb2
from inference_engine.server.proto_gen.kakeya.v1 import distributed_pb2_grpc

# Restored K/V banks and per-block aux are large; lift gRPC's 4 MiB default.
_MAX_MESSAGE_BYTES = 512 * 1024 * 1024
_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
]


class DFlashProposerError(RuntimeError):
    """A remote DFlash RPC failed or returned a malformed result."""


@dataclass(frozen=True)
class RestoreResult:
    """f_θ-projected verifier K/V for the prompt + the eviction plan."""

    restored: List[Tuple[int, WireTensor, WireTensor]]  # (verifier_layer, K, V)
    evicted_positions: List[int]
    prompt_len: int


@dataclass(frozen=True)
class DraftResult:
    """One block's drafts + accounting (mirrors BlockProposal)."""

    draft_token_ids: List[int]
    forward_passes: int
    peak_activation_bytes: int


class RestorationDraftEngine(Protocol):
    """Server-side contract: a stateful DFlash drafter + f_θ projection.

    All tensors are :class:`WireTensor` (framework-neutral) so the servicer
    never imports torch/mlx; the real engine converts internally.
    """

    def restore(
        self, session_id: str, prompt_ids: Sequence[int], *,
        sink: int, window: int, s5_exact_full_attn: bool, model_id: str,
    ) -> RestoreResult: ...

    def seed_context(
        self, session_id: str, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int: ...

    def draft_block(
        self, session_id: str, *, bonus_token_id: int, context_len: int,
        block_size: int,
    ) -> DraftResult: ...

    def extend_context(
        self, session_id: str, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int: ...

    def close_session(self, session_id: str) -> None: ...


# --------------------------------------------------------------------------- #
# proto <-> WireTensor
# --------------------------------------------------------------------------- #
def _tensor_to_wire(t: distributed_pb2.Tensor) -> WireTensor:
    return from_proto_fields(t.dtype, list(t.shape), t.data)


def _wire_to_tensor(w: WireTensor) -> distributed_pb2.Tensor:
    dtype, shape, data = to_proto_fields(w)
    return distributed_pb2.Tensor(dtype=dtype, shape=shape, data=data)


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
class DFlashProposerServicer(distributed_pb2_grpc.DFlashProposerServiceServicer):
    """gRPC servicer delegating to a :class:`RestorationDraftEngine`.

    Engine ``KeyError`` (unknown session) maps to NOT_FOUND; ``ValueError``
    (bad args) to INVALID_ARGUMENT. Both surface cleanly on the client.
    """

    def __init__(self, engine: RestorationDraftEngine) -> None:
        self._engine = engine

    async def Restore(self, request, context):  # noqa: N802 - gRPC casing
        with _grpc_errors(context):
            res = self._engine.restore(
                request.session_id, list(request.prompt_ids),
                sink=request.sink, window=request.window,
                s5_exact_full_attn=request.s5_exact_full_attn,
                model_id=request.model_id,
            )
        return distributed_pb2.RestoreResponse(
            restored=[
                distributed_pb2.LayerKV(
                    layer=layer, k=_wire_to_tensor(k), v=_wire_to_tensor(v))
                for layer, k, v in res.restored
            ],
            evicted_positions=res.evicted_positions,
            prompt_len=res.prompt_len,
        )

    async def SeedContext(self, request, context):  # noqa: N802
        with _grpc_errors(context):
            cl = self._engine.seed_context(
                request.session_id,
                [_tensor_to_wire(t) for t in request.aux],
                list(request.positions),
            )
        return distributed_pb2.SeedContextResponse(context_len=cl)

    async def DraftBlock(self, request, context):  # noqa: N802
        with _grpc_errors(context):
            dr = self._engine.draft_block(
                request.session_id, bonus_token_id=request.bonus_token_id,
                context_len=request.context_len, block_size=request.block_size,
            )
        return distributed_pb2.DraftBlockResponse(
            draft_token_ids=dr.draft_token_ids,
            forward_passes=dr.forward_passes,
            peak_activation_bytes=dr.peak_activation_bytes,
        )

    async def ExtendContext(self, request, context):  # noqa: N802
        with _grpc_errors(context):
            cl = self._engine.extend_context(
                request.session_id,
                [_tensor_to_wire(t) for t in request.aux],
                list(request.positions),
            )
        return distributed_pb2.ExtendContextResponse(context_len=cl)

    async def CloseSession(self, request, context):  # noqa: N802
        self._engine.close_session(request.session_id)
        return distributed_pb2.CloseDFlashSessionResponse()


class _grpc_errors:
    """Context manager mapping engine exceptions to gRPC status codes.

    Used as a plain object (not @contextmanager) so it works inside the async
    servicer methods without awaiting; ``context.set_code`` is synchronous.
    """

    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_grpc_errors":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            return False
        if issubclass(exc_type, KeyError):
            self._context.set_code(grpc.StatusCode.NOT_FOUND)
            self._context.set_details(f"unknown dflash session: {exc}")
            return True
        if issubclass(exc_type, ValueError):
            self._context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            self._context.set_details(str(exc))
            return True
        return False


def add_dflash_proposer_service(
    server, engine: RestorationDraftEngine,
) -> DFlashProposerServicer:
    """Register a DFlashProposerService for ``engine`` on ``server``."""
    servicer = DFlashProposerServicer(engine)
    distributed_pb2_grpc.add_DFlashProposerServiceServicer_to_server(servicer, server)
    return servicer


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class RemoteDFlashProposer:
    """Client for a remote DFlash+f_θ proposer, bound to one decode session.

    Caller passes/receives :class:`WireTensor` (it owns the mlx/torch bridge).
    Any RPC failure raises :class:`DFlashProposerError` with the gRPC status.
    """

    def __init__(
        self, address: str, *, session_id: str, model_id: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.address = address
        self.session_id = session_id
        self.model_id = model_id
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTIONS)
        self._stub = distributed_pb2_grpc.DFlashProposerServiceStub(self._channel)

    def _call(self, name: str, method, request):
        try:
            return method(request, timeout=self.timeout_s)
        except grpc.RpcError as exc:
            raise DFlashProposerError(
                f"{name} to {self.address} failed: "
                f"{exc.code().name}: {exc.details()}"
            ) from exc

    def restore(
        self, prompt_ids: Sequence[int], *, sink: int, window: int,
        s5_exact_full_attn: bool = True,
    ) -> RestoreResult:
        resp = self._call("Restore", self._stub.Restore, distributed_pb2.RestoreRequest(
            session_id=self.session_id, prompt_ids=list(prompt_ids),
            sink=sink, window=window, s5_exact_full_attn=s5_exact_full_attn,
            model_id=self.model_id,
        ))
        return RestoreResult(
            restored=[(lk.layer, _tensor_to_wire(lk.k), _tensor_to_wire(lk.v))
                      for lk in resp.restored],
            evicted_positions=list(resp.evicted_positions),
            prompt_len=resp.prompt_len,
        )

    def seed_context(
        self, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int:
        resp = self._call("SeedContext", self._stub.SeedContext, distributed_pb2.SeedContextRequest(
            session_id=self.session_id,
            aux=[_wire_to_tensor(w) for w in aux],
            positions=list(positions),
        ))
        return resp.context_len

    def draft_block(
        self, *, bonus_token_id: int, context_len: int, block_size: int,
    ) -> DraftResult:
        resp = self._call("DraftBlock", self._stub.DraftBlock, distributed_pb2.DraftBlockRequest(
            session_id=self.session_id, bonus_token_id=bonus_token_id,
            context_len=context_len, block_size=block_size,
        ))
        tokens = list(resp.draft_token_ids)
        if len(tokens) != block_size:
            raise DFlashProposerError(
                f"remote DFlash returned {len(tokens)} drafts; expected {block_size}")
        return DraftResult(
            draft_token_ids=tokens, forward_passes=resp.forward_passes,
            peak_activation_bytes=resp.peak_activation_bytes,
        )

    def extend_context(
        self, aux: Sequence[WireTensor], positions: Sequence[int],
    ) -> int:
        resp = self._call("ExtendContext", self._stub.ExtendContext, distributed_pb2.ExtendContextRequest(
            session_id=self.session_id,
            aux=[_wire_to_tensor(w) for w in aux],
            positions=list(positions),
        ))
        return resp.context_len

    def close(self) -> None:
        # Best-effort: free remote state if reachable, but never let a dead
        # channel mask the real error in the caller's `finally`.
        try:
            self._call("CloseSession", self._stub.CloseSession,
                       distributed_pb2.CloseDFlashSessionRequest(session_id=self.session_id))
        except DFlashProposerError:
            pass
        finally:
            self._channel.close()

    def __enter__(self) -> "RemoteDFlashProposer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
