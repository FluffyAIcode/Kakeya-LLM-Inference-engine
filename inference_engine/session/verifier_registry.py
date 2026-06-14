"""Per-session verifier registry — PR-A3c (multi-tenant served path).

v0.3 was single-tenant: one verifier instance, its cache *is* the (only)
session's state, so concurrent sessions would corrupt each other's KV. This
registry binds **one verifier adapter per session**, all sharing the model
weights (via the adapter's :meth:`spawn`), so the gRPC served path can serve
N sessions with **isolated** KV state.

The registry doubles as the :class:`SessionStore` ``cache_inspector`` and as
the coordinators' verifier *resolver* (``get(session_id) -> verifier``), so a
session's K/V accounting and decode both route to the same per-session adapter.

It does NOT batch concurrent decodes into one forward (that is a scheduler
concern; the batched-parallel throughput capability is validated separately in
ADR 0014 §3.5). It provides **correct per-session isolation** — the
prerequisite for any multi-tenant serving.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict


class PerSessionVerifierRegistry:
    """Maps ``session_id -> verifier adapter`` (lazily created, shared weights).

    Parameters
    ----------
    factory
        Zero-arg callable returning a fresh verifier adapter that shares the
        model weights (e.g. ``base_adapter.spawn``). Called once per new
        session id.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._verifiers: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- resolver (coordinators call this) ------------------------------- #
    def get(self, session_id: str) -> Any:
        v = self._verifiers.get(session_id)
        if v is None:
            with self._lock:
                v = self._verifiers.get(session_id)
                if v is None:
                    v = self._factory()
                    self._verifiers[session_id] = v
        return v

    def remove(self, session_id: str) -> None:
        """Drop a session's adapter (frees its cache). Idempotent."""
        with self._lock:
            self._verifiers.pop(session_id, None)

    def active_sessions(self) -> int:
        return len(self._verifiers)

    # -- CacheInspector protocol (SessionStore calls these) -------------- #
    def k_seq_length(self, session: Any) -> int:
        return int(self.get(session.session_id).k_seq_length(session))

    def kv_live_bytes(self, session: Any) -> int:
        return int(self.get(session.session_id).kv_live_bytes(session))
