"""Kakeya Python SDK — :class:`Session` (PR-B4 of ADR 0008 Phase B).

A handle to one server-side session. Mirrors the ADR 0008 §2.2 RPC
surface as Python methods:

    session.append(token_ids)              -> AppendTokens
    session.generate(max_tokens=N)         -> Generate (server stream)
    session.info()                         -> GetSessionInfo
    session.close()                        -> CloseSession

After ``close()``, every method except ``session_id`` /
``info()`` raises :class:`~kakeya.errors.SessionClosedError`. The
runtime-side state may have been freed earlier (LRU / TTL eviction);
the SDK doesn't track that — the next RPC will raise
:class:`~kakeya.errors.SessionNotFoundError` if the runtime no
longer knows the id.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, TYPE_CHECKING

import grpc

from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from kakeya.errors import (
    SessionClosedError,
    _wrap_grpc_error,
)

if TYPE_CHECKING:
    from kakeya.client import Client


class SessionInfo:
    """Read-only snapshot of a session's server-side state.

    Returned by :meth:`Session.info`. Field names mirror
    :class:`runtime_pb2.GetSessionInfoResponse` for direct
    correspondence with the wire contract.
    """

    __slots__ = (
        "history_length",
        "kv_live_bytes",
        "cache_invariant_inv1_violations",
        "cache_invariant_inv2_violations",
        "idle_seconds",
    )

    def __init__(
        self,
        *,
        history_length: int,
        kv_live_bytes: int,
        cache_invariant_inv1_violations: int,
        cache_invariant_inv2_violations: int,
        idle_seconds: float,
    ) -> None:
        self.history_length = history_length
        self.kv_live_bytes = kv_live_bytes
        self.cache_invariant_inv1_violations = cache_invariant_inv1_violations
        self.cache_invariant_inv2_violations = cache_invariant_inv2_violations
        self.idle_seconds = idle_seconds

    def __repr__(self) -> str:
        return (
            f"SessionInfo(history_length={self.history_length}, "
            f"kv_live_bytes={self.kv_live_bytes}, "
            f"inv1={self.cache_invariant_inv1_violations}, "
            f"inv2={self.cache_invariant_inv2_violations}, "
            f"idle_seconds={self.idle_seconds:.3f})"
        )


class Session:
    """One server-side session, addressable by ``session_id``."""

    def __init__(self, *, client: "Client", session_id: str) -> None:
        self._client = client
        # Prefer the client's stub (already constructed) over making
        # a new one; this keeps a single channel per Client instance.
        self._stub: runtime_pb2_grpc.RuntimeServiceStub = client._stub
        self._session_id = session_id
        self._closed = False
        # Populated after the most recent generate() call returns.
        # Useful for callers that want the stop_reason / token count
        # without having to wrap iteration in their own bookkeeping.
        self._last_stop_reason: Optional[int] = None
        self._last_generated_token_count: int = 0
        self._last_prefill_duration_seconds: float = 0.0
        self._last_total_duration_seconds: float = 0.0
        self._last_history_truncated_dropped: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_stop_reason(self) -> Optional[int]:
        """``runtime_pb2.GenerateDone.StopReason`` enum value from the
        most recent :meth:`generate` call, or ``None`` if no call has
        been made yet."""
        return self._last_stop_reason

    @property
    def last_generated_token_count(self) -> int:
        return self._last_generated_token_count

    @property
    def last_prefill_duration_seconds(self) -> float:
        return self._last_prefill_duration_seconds

    @property
    def last_total_duration_seconds(self) -> float:
        return self._last_total_duration_seconds

    @property
    def last_history_truncated_dropped(self) -> Optional[int]:
        """If the most recent :meth:`generate` started in
        sink+window-truncated mode, the runtime emitted a
        ``HistoryTruncated`` event with the number of dropped
        tokens — this property exposes that count. ``None`` if the
        last call did not encounter truncation, or if no call has
        been made yet."""
        return self._last_history_truncated_dropped

    # ------------------------------------------------------------------
    # RPC methods
    # ------------------------------------------------------------------

    def append(self, token_ids: Iterable[int]) -> int:
        """Append raw token ids to the session's history.

        Returns the new ``history_length``. Empty input is a
        runtime-side no-op.
        """
        self._check_open()
        request = runtime_pb2.AppendTokensRequest(
            session_id=self._session_id,
            token_ids=list(token_ids),
        )
        try:
            response = self._stub.AppendTokens(request)
        except grpc.RpcError as exc:
            raise _wrap_grpc_error(exc) from exc
        return response.history_length

    def generate(
        self,
        *,
        max_tokens: int,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Iterator[int]:
        """Stream generated token ids.

        Yields ``int`` token ids in generation order. The iterator
        is exhausted when the server emits a ``GenerateDone``
        frame; metadata (stop reason, count, durations) is then
        available via :attr:`last_stop_reason` etc.

        v0.3 supports only greedy decoding. Setting
        ``temperature`` / ``top_p`` / ``top_k`` to anything other
        than the greedy no-op default raises
        :class:`~kakeya.errors.InvalidArgumentError` from the
        runtime (per ADR 0008 §2.10 "no graceful degradation").
        """
        self._check_open()
        # Reset metadata from any prior call before this one starts.
        self._last_stop_reason = None
        self._last_generated_token_count = 0
        self._last_prefill_duration_seconds = 0.0
        self._last_total_duration_seconds = 0.0
        self._last_history_truncated_dropped = None

        request = runtime_pb2.GenerateRequest(
            session_id=self._session_id,
            max_tokens=max_tokens,
        )
        if seed is not None:
            request.seed = seed
        if temperature is not None:
            request.temperature = temperature
        if top_p is not None:
            request.top_p = top_p
        if top_k is not None:
            request.top_k = top_k

        try:
            for response in self._stub.Generate(request):
                payload = response.WhichOneof("payload")
                if payload == "token_id":
                    yield response.token_id
                elif payload == "truncated":
                    self._last_history_truncated_dropped = (
                        response.truncated.dropped_token_count
                    )
                elif payload == "done":
                    done = response.done
                    self._last_stop_reason = done.stop_reason
                    self._last_generated_token_count = (
                        done.generated_token_count
                    )
                    self._last_prefill_duration_seconds = (
                        done.prefill_duration_seconds
                    )
                    self._last_total_duration_seconds = (
                        done.total_duration_seconds
                    )
                    return
        except grpc.RpcError as exc:
            raise _wrap_grpc_error(exc) from exc

    def info(self) -> SessionInfo:
        """Return a snapshot of the session's server-side state.

        Allowed even after :meth:`close` has been called locally —
        in that case the call goes to the runtime and most likely
        returns :class:`~kakeya.errors.SessionNotFoundError`.
        """
        request = runtime_pb2.GetSessionInfoRequest(
            session_id=self._session_id,
        )
        try:
            response = self._stub.GetSessionInfo(request)
        except grpc.RpcError as exc:
            raise _wrap_grpc_error(exc) from exc
        return SessionInfo(
            history_length=response.history_length,
            kv_live_bytes=response.kv_live_bytes,
            cache_invariant_inv1_violations=(
                response.cache_invariant_inv1_violations
            ),
            cache_invariant_inv2_violations=(
                response.cache_invariant_inv2_violations
            ),
            idle_seconds=response.idle_seconds,
        )

    def close(self) -> int:
        """Close the session on the runtime. Returns the final
        ``history_length``.

        Idempotent at the SDK level: a second call returns 0
        without contacting the runtime, the same way ``Client.close()``
        is idempotent. (A first close that fails on the wire still
        flips the local closed flag — the SDK assumes the runtime
        is unreachable rather than dribbling out further calls.)
        """
        if self._closed:
            return 0
        request = runtime_pb2.CloseSessionRequest(
            session_id=self._session_id,
        )
        try:
            response = self._stub.CloseSession(request)
        except grpc.RpcError as exc:
            self._closed = True
            raise _wrap_grpc_error(exc) from exc
        self._closed = True
        return response.final_history_length

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Best-effort close on context exit; swallow SessionNotFoundError
        # because the runtime may have evicted the session between the
        # last RPC and this exit.
        try:
            self.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise SessionClosedError(
                f"session {self._session_id!r} has been closed locally; "
                "create a new session to continue work",
            )
