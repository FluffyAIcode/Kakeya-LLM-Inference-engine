"""Session-bound runtime state — ADR 0008 Phase A.

This package owns the in-memory representation of inference sessions
(``SessionStore`` + ``Session``) and the anomaly invariants INV-1 and
INV-2 (see ADR 0008 §2.8).

PR-A2 ships the pure-Python data layer with no verifier wiring and
no gRPC binding. Subsequent PRs add:

- **PR-A3** — Wires the verifier KV cache. ``Session.cached_token_sequence``
  is then synchronously maintained against the K/V tensors, the
  verifier acts as the ``CacheInspector`` for INV-1, and slab
  allocation moves into ``SessionStore.create_session``.
- **PR-B1** — Adds the gRPC server that maps ``CreateSession``,
  ``AppendTokens``, ``CloseSession``, ``GetSessionInfo`` onto this
  store.
"""

from inference_engine.session.store import (
    CacheInspector,
    InvariantViolation,
    Session,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)

__all__ = [
    "CacheInspector",
    "InvariantViolation",
    "Session",
    "SessionNotFoundError",
    "SessionStore",
    "SessionStoreError",
]
