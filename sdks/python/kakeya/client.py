"""Kakeya Python SDK — :class:`Client` (PR-B4 of ADR 0008 Phase B).

A thin sync wrapper around ``grpc.insecure_channel`` + the
generated ``RuntimeServiceStub``. Public surface matches the
ADR 0008 §3.1 example::

    client = Client("localhost:50051")
    session = client.create_session(eos_token_ids=[151645])
    session.append([10, 20, 30])
    for token_id in session.generate(max_tokens=64):
        print(token_id)
    session.close()
    client.close()

Or as a context manager::

    with Client("localhost:50051") as client:
        with client.create_session() as session:
            ...

The SDK is sync because (a) the ADR §3.1 example is sync, (b) the
target audience for the Python SDK in v0.3 is REPL / scripts /
agent harnesses where async adds friction, and (c) v0.3 is
single-tenant (``max_concurrent=1``) so the perf cost of a sync
call blocking is irrelevant. An async API can be added in a
follow-up PR without breaking this surface.

The runtime can be either sync- or async-backed (``grpc.aio.server``
or ``grpc.server``); the wire protocol is identical and our SDK
talks to both.
"""

from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

import grpc

from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from kakeya.errors import _wrap_grpc_error

if TYPE_CHECKING:
    from kakeya.session import Session


DEFAULT_ADDRESS = "localhost:50051"
"""Default gRPC bind address for a local Kakeya runtime, matching
``inference_engine.server.grpc_app.DEFAULT_BIND_ADDRESS``."""


class Client:
    """A connection to a Kakeya RuntimeService.

    Construction opens a gRPC channel; the channel is closed by
    :meth:`close` (also invoked by ``__exit__`` when used as a
    context manager). The connection is lazy — no RPC is made until
    a method like :meth:`create_session` is called.
    """

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        *,
        channel_options: Optional[list] = None,
    ) -> None:
        self._address = address
        self._channel = grpc.insecure_channel(
            address, options=channel_options or [],
        )
        self._stub = runtime_pb2_grpc.RuntimeServiceStub(self._channel)
        self._closed = False

    @property
    def address(self) -> str:
        return self._address

    @property
    def closed(self) -> bool:
        return self._closed

    def create_session(
        self,
        *,
        eos_token_ids: Iterable[int] = (),
        client_label: str = "",
    ) -> "Session":
        """Create a new session on the runtime.

        Returns a :class:`~kakeya.session.Session` bound to this
        client. The session is alive until ``session.close()`` is
        called or the runtime evicts it; until then the
        ``session.session_id`` is the stable handle.

        Raises:
          * :class:`~kakeya.errors.ResourceExhaustedError` if the
            runtime's slab pool is full.
          * :class:`~kakeya.errors.KakeyaError` (base) for any
            other gRPC failure.
        """
        from kakeya.session import Session  # local import: cycle avoidance

        request = runtime_pb2.CreateSessionRequest(
            eos_token_ids=list(eos_token_ids),
            client_label=client_label,
        )
        try:
            response = self._stub.CreateSession(request)
        except grpc.RpcError as exc:
            raise _wrap_grpc_error(exc) from exc
        return Session(client=self, session_id=response.session_id)

    def close(self) -> None:
        """Close the underlying gRPC channel.

        Idempotent: a second call is a no-op. The runtime's
        sessions are NOT closed by this method — call
        ``session.close()`` first if the runtime should free them.
        """
        if self._closed:
            return
        self._channel.close()
        self._closed = True

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
