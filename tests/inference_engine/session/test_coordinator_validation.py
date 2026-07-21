"""Linux-side validation tests for :class:`AppendTokensCoordinator`.

The coordinator dispatches argument validation through pre-verifier
code paths that do not require model weights:

  * Unknown session id  → ``SessionNotFoundError`` from
    ``self._store.get_session(...)`` before the verifier is touched.
  * Empty token list    → early return without invoking the verifier.
  * Constructor         → just stores references; no verifier access.
  * ``_sync_slab_bytes`` None-branch (PR-E1c) → no-op when
    ``session.slab is None``.

These tests use ``verifier=None`` and assert by structure that the
coordinator never touches it. They run on the Linux gate.

Tests that require real verifier numerics (dispatch, state mirror,
INV-1/2/3 propagation) live in ``tests/integration/test_coordinator_real.py``
per PR-N1's no-doubles split.
"""

from __future__ import annotations

import time

import pytest

from inference_engine.session import (
    AppendTokensCoordinator,
    SessionNotFoundError,
    SessionStore,
)


def test_unknown_session_raises_session_not_found_without_verifier():
    """`AppendTokensCoordinator.append_tokens` does the
    ``self._store.get_session`` lookup BEFORE touching the verifier;
    an unknown session id surfaces ``SessionNotFoundError`` while
    ``verifier`` is never accessed."""
    store = SessionStore(capacity=1)
    coord = AppendTokensCoordinator(store, verifier=None)
    with pytest.raises(SessionNotFoundError):
        coord.append_tokens("sess-unknown", [1, 2, 3])


def test_empty_token_list_is_noop_without_verifier():
    """Empty token list returns early; verifier is never accessed."""
    store = SessionStore(capacity=1)
    sess = store.create_session()
    coord = AppendTokensCoordinator(store, verifier=None)
    new_len = coord.append_tokens(sess.session_id, [])
    assert new_len == 0
    assert sess.history_token_ids == []
    assert sess.next_global_position == 0


def test_empty_append_advances_last_active_at_without_verifier():
    """The empty-append no-op still touches ``last_active_at`` so
    a TTL-evicting store doesn't drop a session that just made a
    legitimate (but empty) RPC. No verifier needed."""
    store = SessionStore(capacity=1)
    sess = store.create_session()
    coord = AppendTokensCoordinator(store, verifier=None)
    before = sess.last_active_at
    time.sleep(0.001)
    coord.append_tokens(sess.session_id, [])
    assert sess.last_active_at > before


def test_constructor_stores_references_without_calling_them():
    """Constructor just assigns; nothing on either argument is
    invoked. Sentinel objects round-trip cleanly."""
    sentinel_store = object()
    sentinel_verifier = object()
    coord = AppendTokensCoordinator(sentinel_store, sentinel_verifier)
    assert coord._store is sentinel_store
    assert coord._verifier is sentinel_verifier


def test_first_append_callback_receives_session_and_tokens_once():
    class Verifier:
        cached_token_sequence = []
        next_global_position = 0
        next_token_logits = None

        def prefill(self, tokens):
            self.cached_token_sequence = list(tokens)
            self.next_global_position = len(tokens)

        def forward_block(self, tokens):
            self.cached_token_sequence.extend(tokens)
            self.next_global_position += len(tokens)
            return [type("Row", (), {"clone": lambda self: self})() for _ in tokens]

        def commit_or_truncate(self, *, forwarded, accepted):
            assert forwarded == accepted

        def k_seq_length(self, _session):
            return len(self.cached_token_sequence)

        def kv_live_bytes(self, _session):
            return 0

    verifier = Verifier()
    store = SessionStore(capacity=1, cache_inspector=verifier)
    session = store.create_session(client_label="live")
    observed = []
    coord = AppendTokensCoordinator(
        store,
        verifier,
        on_first_append=lambda sess, tokens: observed.append(
            (sess.client_label, list(tokens)),
        ),
    )
    coord.append_tokens(session.session_id, [1, 2])
    coord.append_tokens(session.session_id, [3])
    assert observed == [("live", [1, 2])]


def test_incremental_append_prefers_last_logits_only_fast_path():
    class Verifier:
        cached_token_sequence = []
        next_global_position = 0
        next_token_logits = None
        fast_path_calls = []

        def prefill(self, tokens):
            self.cached_token_sequence = list(tokens)
            self.next_global_position = len(tokens)

        def append_accepted_tokens(self, tokens):
            self.fast_path_calls.append(list(tokens))
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
    coord = AppendTokensCoordinator(store, verifier)
    coord.append_tokens(session.session_id, [1, 2])
    coord.append_tokens(session.session_id, [3, 4])

    assert verifier.fast_path_calls == [[3, 4]]
    assert session.history_token_ids == [1, 2, 3, 4]


# Note: tests for the ``_sync_slab_bytes`` helper (PR-E1c addition)
# live in PR-E1c's own commit. PR-N1 is branched off main; once
# PR-E1c merges, a follow-up will add the helper's None-branch test
# here. The non-None branch is already exercised in
# ``tests/integration/test_coordinator_real.py`` against the real
# Qwen3 verifier.
