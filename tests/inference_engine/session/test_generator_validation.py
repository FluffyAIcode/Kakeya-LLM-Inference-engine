"""Linux-side validation tests for :class:`GenerationCoordinator`.

The argument-validation paths in ``generator.generate`` (max_tokens,
temperature, top_p, top_k, AppendTokens-must-precede-Generate, unknown
session) reject **before** the coordinator touches the verifier. They
need no verifier instance at all — pass ``None`` and the assertion
that ``self._verifier`` is never accessed becomes part of the test.

This file replaces the verifier-double-driven validation tests that
previously lived in ``test_generator.py``. PR-N1 split the latter
into:

  * Validation paths (this file) — Linux-runnable, no test doubles.
  * Real-numerics paths (``tests/integration/test_generator_real.py``)
    — Mac M4 / CUDA only.

The split honors the architectural rule from PR-N1: the Linux gate
runs only verifier-independent code; runtime correctness moves to
the integration suite.
"""

from __future__ import annotations

import pytest
import threading

from inference_engine.session import (
    GenerationCoordinator,
    SessionGenerationBusyError,
    SessionNotFoundError,
    SessionStore,
)


# ---------------------------------------------------------------------------
# Setup helpers — neither needs a verifier instance.
# ---------------------------------------------------------------------------


def _store_and_session_with_history():
    """Create a SessionStore + Session whose
    ``next_global_position`` is non-zero so the
    ``AppendTokens-must-precede-Generate`` check is past, but no
    verifier was ever involved. We monkey-set the position to fake
    a "session has history" state without doing a real prefill;
    this is legitimate because the tests below all reject in the
    arg-validation block which fires BEFORE that position check.
    """
    store = SessionStore(capacity=1)
    sess = store.create_session()
    sess.next_global_position = 1  # any non-zero will do
    return store, sess


def _store_with_empty_session():
    """A session whose next_global_position is still 0 (cold)."""
    store = SessionStore(capacity=1)
    sess = store.create_session()
    return store, sess


# ---------------------------------------------------------------------------
# max_tokens
# ---------------------------------------------------------------------------


class TestMaxTokensValidation:
    def test_max_tokens_zero_rejected(self):
        store, sess = _store_and_session_with_history()
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            list(coord.generate(sess.session_id, max_tokens=0))

    def test_max_tokens_negative_rejected(self):
        store, sess = _store_and_session_with_history()
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            list(coord.generate(sess.session_id, max_tokens=-3))


# ---------------------------------------------------------------------------
# Sampling parameters — v0.3 greedy-only
# ---------------------------------------------------------------------------


class TestSamplingParamValidation:
    def test_temperature_nonzero_rejected(self):
        store, sess = _store_and_session_with_history()
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(ValueError, match="greedy"):
            list(coord.generate(
                sess.session_id, max_tokens=1, temperature=0.5,
            ))

    def test_top_p_set_rejected(self):
        store, sess = _store_and_session_with_history()
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(ValueError, match="top_p"):
            list(coord.generate(
                sess.session_id, max_tokens=1, top_p=0.9,
            ))

    def test_top_k_other_than_one_rejected(self):
        store, sess = _store_and_session_with_history()
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(ValueError, match="top_k"):
            list(coord.generate(
                sess.session_id, max_tokens=1, top_k=50,
            ))


# ---------------------------------------------------------------------------
# Session-state precondition
# ---------------------------------------------------------------------------


class TestSessionStateValidation:
    def test_no_appendtokens_first_rejected(self):
        store, sess = _store_with_empty_session()
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(ValueError, match="AppendTokens must precede"):
            list(coord.generate(sess.session_id, max_tokens=1))

    def test_unknown_session_raises_session_not_found(self):
        store = SessionStore(capacity=1)
        coord = GenerationCoordinator(store, verifier=None)
        with pytest.raises(SessionNotFoundError):
            list(coord.generate("sess-unknown", max_tokens=1))


# ---------------------------------------------------------------------------
# Constructor / event types — pure data; no verifier needed.
# ---------------------------------------------------------------------------


class TestEventDataclassesAreFrozen:
    def test_token_event_is_frozen(self):
        from inference_engine.session import TokenEvent
        e = TokenEvent(token_id=5)
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            e.token_id = 6  # type: ignore[misc]

    def test_history_truncated_event_is_frozen(self):
        from inference_engine.session import HistoryTruncatedEvent
        e = HistoryTruncatedEvent(dropped_token_count=3)
        with pytest.raises(Exception):
            e.dropped_token_count = 4  # type: ignore[misc]

    def test_done_event_is_frozen(self):
        from inference_engine.session import DoneEvent, STOP_REASON_MAX_TOKENS
        e = DoneEvent(
            stop_reason=STOP_REASON_MAX_TOKENS,
            generated_token_count=1,
            prefill_seconds=0.0, total_seconds=0.0,
        )
        with pytest.raises(Exception):
            e.generated_token_count = 2  # type: ignore[misc]


class TestConstructorAcceptsReferences:
    def test_constructor_stores_references_without_calling_them(self):
        sentinel_store = object()
        sentinel_verifier = object()
        coord = GenerationCoordinator(sentinel_store, sentinel_verifier)
        # Check internals — neither object was poked.
        assert coord._store is sentinel_store
        assert coord._verifier is sentinel_verifier


def test_generate_honors_pre_set_cancellation_at_token_boundary():
    store, sess = _store_and_session_with_history()
    cancelled = threading.Event()
    cancelled.set()
    events = list(GenerationCoordinator(store, verifier=None).generate(
        sess.session_id,
        max_tokens=1,
        cancel_event=cancelled,
    ))
    assert len(events) == 1
    assert events[0].stop_reason == "cancelled"


def test_generate_rejects_concurrent_stream_for_same_session():
    store, sess = _store_and_session_with_history()
    cancelled = threading.Event()
    cancelled.set()
    coordinator = GenerationCoordinator(store, verifier=None)
    first = coordinator.generate(
        sess.session_id, max_tokens=1, cancel_event=cancelled,
    )
    assert coordinator.active_count == 0
    assert next(first).stop_reason == "cancelled"
    assert coordinator.active_count == 1
    second = coordinator.generate(
        sess.session_id, max_tokens=1, cancel_event=cancelled,
    )
    with pytest.raises(SessionGenerationBusyError):
        next(second)
    first.close()
    assert coordinator.active_count == 0


def test_generate_prefers_on_device_argmax_and_last_logits_fast_path():
    class Verifier:
        cached_token_sequence = [10]
        next_global_position = 1
        next_token_logits = None
        argmax_calls = 0
        append_calls = []

        def greedy_next_token_id(self):
            self.argmax_calls += 1
            return 11

        def append_accepted_tokens(self, tokens):
            self.append_calls.append(list(tokens))
            self.cached_token_sequence.extend(tokens)
            self.next_global_position += len(tokens)

        def forward_block(self, _tokens):
            raise AssertionError("full-block path should not run")

        def k_seq_length(self, _session):
            return len(self.cached_token_sequence)

        def kv_live_bytes(self, _session):
            return 0

    verifier = Verifier()
    store = SessionStore(capacity=1, cache_inspector=verifier)
    session = store.create_session()
    session.cached_token_sequence = [10]
    store.append_tokens(session.session_id, [10])
    store.record_position_advance(session.session_id, 1)

    events = list(GenerationCoordinator(store, verifier).generate(
        session.session_id,
        max_tokens=1,
    ))

    assert verifier.argmax_calls == 1
    assert verifier.append_calls == [[11]]
    assert events[0].token_id == 11
