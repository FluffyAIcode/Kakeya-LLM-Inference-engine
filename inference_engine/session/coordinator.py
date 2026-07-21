"""AppendTokensCoordinator — ADR 0008 PR-B2 (Phase B).

Ties :class:`SessionStore` to a verifier so the gRPC ``AppendTokens``
RPC can implement the §2.3 byte-exact prefill-incremental contract:

  *For the same (session_id, history_token_ids) pair, the verifier's
  KV cache state at the end of N AppendTokens calls is bit-identical
  regardless of how the tokens were grouped.*

The coordinator is the single orchestrator of the dispatch logic
(first-call → full prefill; subsequent-call → forward + commit). It
intentionally lives outside the verifier (which post-PR-A3 has no
session knowledge) and outside the SessionStore (which is the data
layer with no verifier dependency). PR-A3's removal of
``verifier.path_select`` was the architectural prerequisite for
this — the dispatch belongs at this protocol-aware layer, not on
the verifier itself.

Layering note: this module imports a :class:`VerifierProtocol`
defined here, not the concrete ``SinkWindowVerifier`` from
``kv_cache_proposer``. Both the CPU and MLX verifiers satisfy the
protocol structurally (they already implement ``prefill``,
``forward_block``, ``commit_or_truncate``, and ``k_seq_length``
post-PR-A3b). Tests use a ``FakeVerifier`` that implements the same
protocol without loading model weights (Linux-runnable; the
real-verifier integration test lives under ``tests/core/`` and runs
on Mac M4).

Anomaly invariants:

  * INV-1 (parallel-sequence consistency): enforced by
    :meth:`SessionStore._assert_inv1` at the end of every
    :meth:`SessionStore.append_tokens` call. The coordinator syncs
    ``session.cached_token_sequence`` from
    ``verifier.cached_token_sequence`` *before* triggering INV-1
    so the check is meaningful.

  * INV-2 (position monotonicity): enforced by
    :meth:`SessionStore.record_position_advance`. The coordinator
    calls it with ``verifier.next_global_position`` after every
    append.

  * INV-3 (continuation-path determinism): the byte-exact contract
    above. Tested in :mod:`tests.inference_engine.session.test_coordinator`
    (Linux, FakeVerifier) and in :mod:`tests.core.test_coordinator_real_verifier`
    (Mac, real Qwen3 verifier).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, List, Protocol

import torch

from inference_engine.session.store import (
    InvariantViolation,
    Session,
    SessionNotFoundError,
    SessionStore,
)


class VerifierProtocol(Protocol):
    """Subset of the verifier API the coordinator needs.

    Both ``kv_cache_proposer.verifier.SinkWindowVerifier`` and
    ``inference_engine.backends.mlx.verifier.MLXSinkWindowVerifier``
    satisfy this structurally as of PR-A3 / PR-A3b.

    Mutability contract for fields used by the coordinator:

      * ``cached_token_sequence``: read-only from the coordinator's
        perspective; the verifier mutates it inside ``prefill``,
        ``forward_block``, and ``commit_or_truncate``.
      * ``next_global_position``: the verifier mutates it inside
        ``prefill`` (sets to ``len(prompt_ids)``) and
        ``commit_or_truncate`` (advances by ``accepted``).
      * ``next_token_logits``: the verifier sets it inside ``prefill``;
        the coordinator sets it after ``forward_block`` +
        ``commit_or_truncate`` to ``block_logits[-1].clone()``.
    """

    cached_token_sequence: List[int]
    next_global_position: int
    next_token_logits: torch.Tensor

    def prefill(self, prompt_ids: List[int]) -> None:
        ...  # pragma: no cover - Protocol body, never executed

    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        ...  # pragma: no cover - Protocol body, never executed

    def commit_or_truncate(self, *, forwarded: int, accepted: int) -> None:
        ...  # pragma: no cover - Protocol body, never executed

    def k_seq_length(self, session: Session) -> int:
        ...  # pragma: no cover - Protocol body, never executed

    def kv_live_bytes(self, session: Session) -> int:
        ...  # pragma: no cover - Protocol body, never executed


class PrefillCacheHookProtocol(Protocol):
    """Optional cold-prefill accelerator used by distributed cache nodes."""

    def prepare(self, verifier: Any, token_ids: List[int]) -> int:
        ...  # pragma: no cover


class OperationCancelledError(RuntimeError):
    """Raised when a blocking coordinator observes client cancellation."""


def _sync_slab_bytes(session: Session, verifier: "VerifierProtocol") -> None:
    """Mirror the verifier's current KV byte count onto the session's
    slab placeholder (PR-E1c).

    The slab's ``live_kv_bytes`` is the source of truth for
    :meth:`Session.kv_live_bytes`, which in turn feeds
    ``GetSessionInfo.kv_live_bytes`` over gRPC. The verifier owns
    the actual K/V tensors; the slab is a placeholder that holds
    one capacity unit per active session. Without this sync the
    gauge reads 0 forever (PR-E1b's 4h bench surfaced this).

    No-op when the session has no slab (pool-less SessionStore — the
    test / pure-data-layer mode the coordinator unit tests use).
    """
    if session.slab is None:
        return
    session.slab.live_kv_bytes_override = int(verifier.kv_live_bytes(session))


class AppendTokensCoordinator:
    """Orchestrator for the §2.3 byte-exact prefill-incremental contract.

    Construction wires a :class:`SessionStore` to a verifier. The
    store should have been constructed with the same verifier as its
    ``cache_inspector`` so INV-1 enforcement reads consistent state
    across the two sides.

    The coordinator does not own either component's lifecycle —
    callers (typically the gRPC ``RuntimeServiceServicer``)
    construct and destroy them.
    """

    def __init__(
        self,
        store: SessionStore,
        verifier: VerifierProtocol,
        resolver=None,
        prefill_cache: PrefillCacheHookProtocol | None = None,
        on_first_append: Callable[[Session, list[int]], None] | None = None,
        liveness=None,
    ) -> None:
        self._store = store
        self._verifier = verifier
        # PR-A3c: optional ``session_id -> verifier`` resolver for per-session
        # binding (multi-tenant). When None, the single ``verifier`` is used
        # for every session (v0.3 single-tenant behaviour, unchanged).
        self._resolver = resolver
        self._prefill_cache = prefill_cache
        self._on_first_append = on_first_append
        self._liveness = liveness

    def _verifier_for(self, session_id: str) -> "VerifierProtocol":
        return self._resolver(session_id) if self._resolver else self._verifier

    def append_tokens(
        self,
        session_id: str,
        token_ids: Iterable[int],
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Run the §2.3 byte-exact prefill-incremental for ``session_id``.

        Returns the new ``history_length``.

        Raises:
          * :class:`SessionNotFoundError` — unknown / closed / evicted id.
          * :class:`ValueError` — well-formedness violation
            (negative or non-integer token_id; surfaces from
            :meth:`SessionStore.append_tokens`).
          * :class:`InvariantViolation` — INV-1 mismatch between
            ``session.cached_token_sequence`` and
            ``verifier.k_seq_length(session)``, or INV-2 violation
            in the position-advance step. The session has been
            removed from the store; the verifier still holds the
            now-orphaned cache state and the caller must reset it
            before re-using the verifier instance.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelledError("AppendTokens cancelled")

        # Lookup early to surface NOT_FOUND before we touch the verifier.
        session = self._store.get_session(session_id)
        verifier = self._verifier_for(session_id)

        token_list = list(token_ids)

        if not token_list:
            # Empty append is a no-op: the cache, the history, and
            # the position are all unchanged. We still touch
            # last_active_at so TTL eviction doesn't fire on a
            # session that just made an empty (but legitimate) RPC.
            import time
            session.last_active_at = time.monotonic()
            return session.history_length

        # Drive the verifier through the prefill-incremental dispatch.
        # First call (cold cache) -> full prefill; subsequent calls ->
        # forward_block + commit_or_truncate. Both paths leave the
        # verifier in a state where:
        #   - cached_token_sequence is the post-trim parallel sequence
        #   - next_global_position = sum of all tokens ever appended
        #   - next_token_logits predicts position == next_global_position
        first_append = session.next_global_position == 0
        if self._liveness is not None:
            self._liveness.update(
                "prefill" if first_append else "decode", session_id, 0,
            )
        if first_append:
            if self._prefill_cache is not None:
                self._prefill_cache.prepare(verifier, token_list)
            else:
                verifier.prefill(token_list)
        else:
            append_accepted = getattr(
                verifier, "append_accepted_tokens", None,
            )
            if append_accepted is not None:
                # MLX primary append needs only the final logits row;
                # speculative verification continues to call
                # forward_block and keeps full [L,V] semantics.
                append_accepted(token_list)
            else:
                block_logits = verifier.forward_block(token_list)
                verifier.commit_or_truncate(
                    forwarded=len(token_list),
                    accepted=len(token_list),
                )
                verifier.next_token_logits = block_logits[-1].clone()

        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelledError("AppendTokens cancelled")

        # Mirror the verifier's post-trim cached_token_sequence onto the
        # session BEFORE the store's INV-1 assertion runs (it compares
        # session.cached_token_sequence length against
        # verifier.k_seq_length(session)).
        session.cached_token_sequence = list(
            verifier.cached_token_sequence,
        )

        # Extend history + run INV-1. The store's append_tokens does
        # both atomically; on INV-1 violation it removes the session
        # and raises before returning a length.
        new_history_length = self._store.append_tokens(
            session_id, token_list,
        )

        # Advance position with INV-2 enforcement. The verifier has
        # already advanced its own next_global_position; we sync the
        # session's via the store so the INV-2 check on subsequent
        # calls reads consistent state.
        self._store.record_position_advance(
            session_id, verifier.next_global_position,
        )

        # Mirror the verifier's current KV byte count onto the slab
        # so GetSessionInfo.kv_live_bytes reports physical bytes
        # rather than the slab's placeholder zero. PR-E1c.
        _sync_slab_bytes(session, verifier)
        if first_append and self._on_first_append is not None:
            self._on_first_append(session, token_list)

        return new_history_length
