"""Typed Python exceptions for the Kakeya Python SDK (PR-B4).

The SDK maps every gRPC status code raised by the Kakeya
RuntimeService into a typed :class:`KakeyaError` subclass. Callers
catch the typed exceptions; they should not need to import
``grpc`` to handle errors.

Mapping (per ADR 0008 §2.6 / §2.10):

==============================  =====================================
gRPC ``StatusCode``             SDK exception
==============================  =====================================
``NOT_FOUND``                   :class:`SessionNotFoundError`
``INVALID_ARGUMENT``            :class:`InvalidArgumentError`
``FAILED_PRECONDITION``         :class:`InvariantViolationError`
``RESOURCE_EXHAUSTED``          :class:`ResourceExhaustedError`
``UNIMPLEMENTED``               :class:`UnimplementedError`
``CANCELLED``                   :class:`RpcCancelledError`
(everything else)               :class:`KakeyaError` (base class)
==============================  =====================================

All SDK exceptions inherit from :class:`KakeyaError`, so users can
write a catch-all for "anything that came from the runtime" with
``except KakeyaError``. The ``rpc_code`` attribute carries the
underlying gRPC status code for callers that need it.

A separate :class:`SessionClosedError` is raised by the SDK itself
(client-side) when a method is invoked on a closed
:class:`~kakeya.session.Session` — this never reaches the runtime.
"""

from __future__ import annotations

from typing import Optional

import grpc


class KakeyaError(Exception):
    """Base for every typed exception raised by the Kakeya SDK."""

    def __init__(
        self,
        message: str,
        *,
        rpc_code: Optional[grpc.StatusCode] = None,
    ) -> None:
        super().__init__(message)
        self.rpc_code = rpc_code


class SessionNotFoundError(KakeyaError):
    """Raised when a ``session_id`` is not present on the runtime.

    The session may have been closed, evicted by LRU, evicted by
    TTL, or removed by an invariant violation; the caller cannot
    distinguish between these cases (per ADR 0008 §2.6 design).
    """


class InvalidArgumentError(KakeyaError):
    """Raised when the runtime rejects a request as malformed.

    Common triggers (per the runtime contract):
      * ``Generate`` called with sampling parameters set in v0.3
        greedy mode (``temperature != 0`` / ``top_p`` set /
        ``top_k != 1``).
      * ``Generate`` called before any ``AppendTokens`` for the
        session.
      * ``max_tokens`` < 1.
    """


class InvariantViolationError(KakeyaError):
    """Raised when the runtime detects an INV-1 / INV-2 violation
    on the session. Per ADR 0008 §2.8, the session has been
    removed from the runtime; subsequent calls referencing the
    same ``session_id`` return :class:`SessionNotFoundError`.
    """


class ResourceExhaustedError(KakeyaError):
    """Raised when the runtime cannot admit a new session because
    its slab pool is exhausted. The caller may retry after
    closing or evicting other sessions.
    """


class UnimplementedError(KakeyaError):
    """Raised when an RPC has not been implemented yet on the
    runtime (e.g., a Servicer constructed without the
    corresponding coordinator). This is distinct from Python's
    builtin :class:`NotImplementedError` to avoid silent collision.
    """


class RpcCancelledError(KakeyaError):
    """Raised on a ``CANCELLED`` gRPC status. Currently rare on the
    SDK surface (cancellation is observable inside the streaming
    iterator but does not raise on a fresh stream).
    """


class InterTokenTimeoutError(KakeyaError):
    """Client-side notification that no stream frame arrived in time."""


class SessionClosedError(KakeyaError):
    """Raised by the SDK itself — never crosses the wire — when a
    method is called on a :class:`~kakeya.session.Session` whose
    ``close()`` has already been invoked.

    This is a defensive client-side check, separate from
    :class:`SessionNotFoundError` (which means the runtime lost
    the session).
    """


def _wrap_grpc_error(exc: grpc.RpcError) -> KakeyaError:
    """Translate a gRPC error into the Kakeya typed equivalent.

    Used by the SDK's RPC helpers; not part of the public surface.
    """
    code = exc.code()
    details = exc.details() if hasattr(exc, "details") else str(exc)
    cls = _CODE_TO_EXCEPTION.get(code, KakeyaError)
    return cls(details or "", rpc_code=code)


_CODE_TO_EXCEPTION: dict = {
    grpc.StatusCode.NOT_FOUND: SessionNotFoundError,
    grpc.StatusCode.INVALID_ARGUMENT: InvalidArgumentError,
    grpc.StatusCode.FAILED_PRECONDITION: InvariantViolationError,
    grpc.StatusCode.RESOURCE_EXHAUSTED: ResourceExhaustedError,
    grpc.StatusCode.UNIMPLEMENTED: UnimplementedError,
    grpc.StatusCode.CANCELLED: RpcCancelledError,
}
