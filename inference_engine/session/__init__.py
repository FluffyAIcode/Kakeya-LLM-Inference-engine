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

from inference_engine.session.coordinator import (
    AppendTokensCoordinator,
    OperationCancelledError,
    VerifierProtocol,
)
from inference_engine.session.generator import (
    DoneEvent,
    GenerateEvent,
    GenerationCoordinator,
    HistoryTruncatedEvent,
    SessionGenerationBusyError,
    STOP_REASON_CANCELLED,
    STOP_REASON_EOS,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_TRUNCATED,
    TokenEvent,
)
from inference_engine.session.store import (
    CacheInspector,
    InvariantViolation,
    Session,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)

__all__ = [
    "AppendTokensCoordinator",
    "CacheInspector",
    "DoneEvent",
    "GenerateEvent",
    "GenerationCoordinator",
    "HistoryTruncatedEvent",
    "InvariantViolation",
    "OperationCancelledError",
    "STOP_REASON_CANCELLED",
    "STOP_REASON_EOS",
    "STOP_REASON_MAX_TOKENS",
    "STOP_REASON_TRUNCATED",
    "Session",
    "SessionGenerationBusyError",
    "SessionNotFoundError",
    "SessionStore",
    "SessionStoreError",
    "TokenEvent",
    "VerifierProtocol",
]
