"""SessionStore — in-memory data layer for ADR 0008 sessions.

This module is the **PR-A2** deliverable from ADR 0008 §6.1: pure
Python, no gRPC binding, no verifier wiring. It establishes the
data shape, the lifecycle methods, and the anomaly-invariant
enforcement that the rest of the runtime will build on.

Contracts enforced here (per ADR 0008 §2.2 / §2.6 / §2.8):

- **Server-issued session ids** (§2.2 item 1). Clients never supply
  the ``session_id``; this store is the only producer of fresh ids
  and uses ``uuid.uuid4`` to make collision impossible by construction.
- **Append-only history** (§2.2 item 3). The only public method that
  mutates a session's ``history_token_ids`` is :meth:`append_tokens`;
  it strictly extends, never rewrites prior entries. Tests verify
  this through the public API only.
- **INV-1 (parallel-sequence consistency, §2.8)**. Enforced via the
  optional :class:`CacheInspector` callback. When an inspector is
  registered, every history mutation triggers an invariant check;
  on violation the session is removed from the store, its counter
  is incremented on the local ``Session`` reference, and
  :class:`InvariantViolation` is raised. PR-A3 will inject the
  verifier as the inspector; PR-A2 leaves the slot empty by default
  (so INV-1 is "trivially holds: nothing to check").
- **INV-2 (position monotonicity, §2.8)**. Enforced inside
  :meth:`record_position_advance` — the only public mutator of
  ``Session.next_global_position``. Decreases raise
  :class:`InvariantViolation` and remove the session.
- **Capacity & eviction** (§2.6). When :meth:`create_session` is
  called at capacity, the least-recently-active session is evicted
  to admit the new one. :meth:`evict_idle` provides the public
  surface for TTL-based eviction; PR-A3 will call it from a
  background task at the configured interval.

This module is intentionally **not thread-safe**. The gRPC server
(PR-B1) runs all RPCs on a single asyncio event loop, serializing
access at that layer; multi-worker support is v0.4 scope (ADR 0008
§4.5). Direct concurrent access from threads is undefined behavior
and is documented as such here so reviewers can flag any future
caller that violates the contract.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Protocol


class SessionStoreError(Exception):
    """Base class for typed errors raised by :class:`SessionStore`.

    Subclasses correspond to ADR 0008's gRPC-status mapping (§2.6 /
    §2.8): :class:`SessionNotFoundError` -> ``NOT_FOUND``,
    :class:`InvariantViolation` -> ``FAILED_PRECONDITION``.
    """


class SessionNotFoundError(SessionStoreError):
    """Raised when a ``session_id`` is not present in the store.

    The session may have been closed via :meth:`SessionStore.close_session`,
    evicted by LRU pressure, evicted by TTL, or removed by an
    :class:`InvariantViolation` — from the caller's perspective these
    are indistinguishable, which is the intended observable behavior
    (ADR 0008 §2.6 "no implicit re-creation").
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session_id {session_id!r} not found")
        self.session_id = session_id


class InvariantViolation(SessionStoreError):
    """Raised when an anomaly invariant is detected on a session.

    Per ADR 0008 §2.8, an invariant violation is a **bug**, not a
    state. The store responds by:

    1. Incrementing the per-session counter on the local
       ``Session`` reference (so callers that hold the reference
       can introspect the violation kind).
    2. Removing the session from the store. Subsequent lookups on
       its id raise :class:`SessionNotFoundError`.
    3. Raising this exception with structured ``kind`` /
       ``session_id`` / ``detail`` fields.

    Slab freeing (the §2.6 lifecycle clause) is the verifier's job;
    PR-A3 will hook it to this raise site.
    """

    def __init__(self, *, kind: str, session_id: str, detail: str) -> None:
        super().__init__(
            f"INV-{kind} violation in session {session_id!r}: {detail}"
        )
        self.kind = kind
        self.session_id = session_id
        self.detail = detail


class CacheInspector(Protocol):
    """Protocol for INV-1 enforcement.

    PR-A3 will provide a verifier-backed implementation. PR-A2 has
    no real implementation; tests use synthetic inspectors to
    exercise the public contract.

    Implementations MUST return the K/V tensor sequence length
    that is observably consistent across all verifier layers; a
    layer-mismatch is an INV-1 violation that the verifier itself
    detects internally before this protocol is consulted.
    """

    def k_seq_length(self, session: "Session") -> int:
        """Return the K/V tensor sequence length for ``session``."""
        ...  # pragma: no cover - Protocol body, never executed


@dataclass
class Session:
    """One active inference session, owned by :class:`SessionStore`.

    Mutability rules (enforced by SessionStore methods, not by the
    dataclass itself — Python's dataclass doesn't model "mutable
    via specific methods only", so the contract lives in the store
    and is verified by tests):

    - ``session_id``, ``eos_token_ids``, ``client_label``,
      ``created_at`` — set once by ``SessionStore.create_session``
      and never modified.
    - ``history_token_ids`` — append-only via
      ``SessionStore.append_tokens``.
    - ``cached_token_sequence``, ``next_global_position``,
      ``last_active_at`` — mutated by SessionStore methods. PR-A3
      will additionally let the verifier mutate
      ``cached_token_sequence`` via documented helpers.
    - ``inv1_violations`` / ``inv2_violations`` — incremented by
      the store on detection. Always 0 in healthy operation
      (§2.8).

    ``kv_live_bytes()`` is a method (not a field) because PR-A3 will
    compute it from the slab on demand; a stale field would lie
    after a slab trim. PR-A2 returns 0 unconditionally as a
    placeholder.
    """

    session_id: str
    eos_token_ids: tuple[int, ...]
    client_label: str

    history_token_ids: list[int] = field(default_factory=list)
    cached_token_sequence: list[int] = field(default_factory=list)
    next_global_position: int = 0

    inv1_violations: int = 0
    inv2_violations: int = 0

    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)

    @property
    def history_length(self) -> int:
        """Number of tokens currently in the session's history."""
        return len(self.history_token_ids)

    @property
    def idle_seconds(self) -> float:
        """Wall seconds since the last RPC interaction with this
        session. Used by :meth:`SessionStore.evict_idle`."""
        return time.monotonic() - self.last_active_at

    def kv_live_bytes(self) -> int:
        """Live KV bytes held by this session's slab.

        PR-A2 placeholder: returns 0. PR-A3 will compute from the
        verifier's slab at call time.
        """
        return 0


class SessionStore:
    """In-memory store of active inference sessions.

    Capacity & eviction follow ADR 0008 §2.6: at capacity,
    :meth:`create_session` evicts the least-recently-active session
    to admit the new one. :meth:`evict_idle` provides the TTL-based
    eviction surface that PR-A3 will drive from a background task.

    The store is **not thread-safe** (see module docstring). All
    callers in v0.3 enter via the gRPC asyncio event loop, which
    serializes access.
    """

    def __init__(
        self,
        *,
        capacity: int,
        cache_inspector: Optional[CacheInspector] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(
                f"capacity must be >= 1, got {capacity}"
            )
        self._capacity = capacity
        self._cache_inspector = cache_inspector
        self._sessions: dict[str, Session] = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def total_kv_live_bytes(self) -> int:
        """Sum of live KV bytes across all active sessions.

        PR-A2: returns 0 (every session reports 0). PR-A3 will
        return the real aggregate, which is the §2.9
        ``session_kv_live_bytes`` Prometheus-gauge value.
        """
        return sum(s.kv_live_bytes() for s in self._sessions.values())

    def create_session(
        self,
        *,
        eos_token_ids: Iterable[int] = (),
        client_label: str = "",
    ) -> Session:
        """Allocate and return a new session.

        Server-issues the ``session_id`` (clients have no input on
        it, per §2.2 item 1). At capacity, evicts the
        least-recently-active session before admitting (§2.6 LRU).
        """
        if self.active_count >= self._capacity:
            self._evict_one_lru()
        session = Session(
            session_id=self._issue_id(),
            eos_token_ids=tuple(eos_token_ids),
            client_label=client_label,
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        """Return the live session for ``session_id``.

        Raises :class:`SessionNotFoundError` if no such session is
        currently in the store (closed, evicted, never existed —
        callers cannot distinguish, by design).
        """
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def append_tokens(
        self, session_id: str, token_ids: Iterable[int],
    ) -> int:
        """Append raw token ids to a session's history.

        Returns the new ``history_length``. The runtime treats token
        ids as opaque integers (§2.4 — no role markers, no chat
        template); only well-formedness checks (non-negative int)
        are applied here. Vocabulary-range validation is the
        verifier's responsibility and lands in PR-A3.

        On INV-1 violation (only possible when a
        :class:`CacheInspector` is registered), the session is
        removed and :class:`InvariantViolation` is raised. The
        history mutation is preserved (the violation is *detected*
        post-mutation, which is the contract — INV-1 is "after every
        cache mutation"); the session is no longer usable.
        """
        session = self.get_session(session_id)
        token_list = list(token_ids)
        for tid in token_list:
            if not isinstance(tid, int) or isinstance(tid, bool) or tid < 0:
                raise ValueError(
                    f"token_id must be a non-negative int, got {tid!r}"
                )
        session.history_token_ids.extend(token_list)
        session.last_active_at = time.monotonic()
        self._assert_inv1(session)
        return session.history_length

    def close_session(self, session_id: str) -> int:
        """Close a session and return its final ``history_length``.

        Raises :class:`SessionNotFoundError` if the session is not
        currently in the store. Idempotent close is **not** offered;
        a second close is an error so that double-close bugs in
        callers surface loudly.
        """
        session = self.get_session(session_id)
        final_length = session.history_length
        del self._sessions[session_id]
        return final_length

    def record_position_advance(
        self, session_id: str, new_position: int,
    ) -> None:
        """Update a session's ``next_global_position``.

        Enforces INV-2 (position monotonicity, §2.8): the new
        position MUST be >= the current ``next_global_position``.
        On violation the counter is incremented on the local
        ``Session`` reference, the session is removed, and
        :class:`InvariantViolation` is raised.

        Called by the verifier after every prefill / generation
        step (PR-A3 wiring); PR-A2 tests exercise it directly.
        """
        session = self.get_session(session_id)
        if new_position < session.next_global_position:
            session.inv2_violations += 1
            del self._sessions[session_id]
            raise InvariantViolation(
                kind="2",
                session_id=session_id,
                detail=(
                    f"position must be non-decreasing; "
                    f"current={session.next_global_position}, "
                    f"requested={new_position}"
                ),
            )
        session.next_global_position = new_position
        session.last_active_at = time.monotonic()

    def evict_idle(
        self,
        *,
        ttl_seconds: float,
        now: Optional[float] = None,
    ) -> list[Session]:
        """Evict every session whose idle time meets or exceeds
        ``ttl_seconds``.

        Returns the evicted sessions for observability (PR-A3 will
        emit a counter increment per evicted session; this method
        itself is silent so it can be called from tests without a
        metrics dependency).

        ``now`` is exposed for testability — callers can pass an
        explicit clock value to drive deterministic eviction tests
        without monkey-patching ``time.monotonic``. Production
        callers omit it to use the current monotonic clock.
        """
        clock = now if now is not None else time.monotonic()
        evicted: list[Session] = []
        for sid in list(self._sessions.keys()):
            session = self._sessions[sid]
            idle = clock - session.last_active_at
            if idle >= ttl_seconds:
                del self._sessions[sid]
                evicted.append(session)
        return evicted

    def _assert_inv1(self, session: Session) -> None:
        """Enforce INV-1 for ``session``.

        When no :class:`CacheInspector` is registered, INV-1 is
        treated as trivially holding (no cache to compare against).
        With an inspector, ``len(cached_token_sequence)`` MUST equal
        ``inspector.k_seq_length(session)``; otherwise the session
        is removed and :class:`InvariantViolation` is raised.
        """
        if self._cache_inspector is None:
            return
        actual_k_seq_len = self._cache_inspector.k_seq_length(session)
        cached_len = len(session.cached_token_sequence)
        if actual_k_seq_len != cached_len:
            session.inv1_violations += 1
            del self._sessions[session.session_id]
            raise InvariantViolation(
                kind="1",
                session_id=session.session_id,
                detail=(
                    f"cached_token_sequence length ({cached_len}) "
                    f"!= K/V tensor sequence length ({actual_k_seq_len})"
                ),
            )

    def _evict_one_lru(self) -> Session:
        """Evict the least-recently-active session.

        Caller must have confirmed ``active_count >= capacity``;
        the store guarantees ``capacity >= 1`` so the dict is
        non-empty here.
        """
        evicted_id, evicted = min(
            self._sessions.items(),
            key=lambda kv: kv[1].last_active_at,
        )
        del self._sessions[evicted_id]
        return evicted

    @staticmethod
    def _issue_id() -> str:
        """Generate a fresh server-issued opaque session id."""
        return f"sess-{uuid.uuid4().hex}"
