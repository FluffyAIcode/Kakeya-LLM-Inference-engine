"""Unit tests for :mod:`inference_engine.session.coordinator` (PR-B2).

Coverage target: 100% on ``inference_engine/session/coordinator.py``.

Test strategy:

  * The coordinator's contract is **dispatch + state-mirroring +
    invariant enforcement**, not the verifier's actual computation.
    Tests use a :class:`FakeVerifier` that satisfies
    :class:`VerifierProtocol` deterministically without loading
    model weights. That makes the entire suite Linux-runnable in
    seconds and isolates coordinator logic from verifier-numerics
    flakiness.

  * The byte-exact §2.3 contract — and INV-3 in particular — is
    tested *here* as a structural property of the coordinator: same
    total token sequence delivered through different chunkings
    produces the same final ``cached_token_sequence``,
    ``next_global_position``, and ``next_token_logits``. The
    FakeVerifier's deterministic state-mutation (sink+window trim
    semantics matching the real verifier) makes this a meaningful
    test even without real attention weights.

  * The end-to-end byte-exact contract against the real Qwen3
    verifier lives under ``tests/core/`` (Mac-only) and runs on
    integration test rather than in this Linux unit suite.
"""

from __future__ import annotations

import time
from typing import List

import pytest
import torch

from inference_engine.session import (
    AppendTokensCoordinator,
    InvariantViolation,
    Session,
    SessionNotFoundError,
    SessionStore,
    VerifierProtocol,
)


# ---------------------------------------------------------------------------
# FakeVerifier — Linux-runnable VerifierProtocol implementation.
# ---------------------------------------------------------------------------


class FakeVerifier:
    """Deterministic VerifierProtocol implementation for unit tests.

    Mirrors the real ``SinkWindowVerifier`` state-mutation contract
    closely enough to verify INV-1 / INV-2 / INV-3 dispatch correctness:

      * ``prefill(prompt_ids)`` resets, sets ``cached_token_sequence``
        to the sink+window slice of ``prompt_ids``, sets
        ``next_global_position = len(prompt_ids)``, and produces
        deterministic ``next_token_logits`` derived from the cached
        suffix.
      * ``forward_block(tokens)`` extends ``cached_token_sequence``
        in-place (mirrors the real verifier — trim happens in
        ``commit_or_truncate``, not here) and returns deterministic
        per-position logits.
      * ``commit_or_truncate(forwarded, accepted)`` drops
        ``forwarded - accepted`` tokens from the tail, advances
        ``next_global_position`` by ``accepted``, then applies
        sink+window trim.
      * ``k_seq_length(session)`` returns
        ``len(cached_token_sequence)`` (single-tenant scope, the
        ``session`` argument is ignored — exactly mirrors the real
        v0.3 verifier behavior).

    A ``call_log`` attribute records every method call so tests can
    assert on the dispatch pattern directly.
    """

    def __init__(self, sink_size: int = 2, window_size: int = 4,
                 vocab_size: int = 16) -> None:
        self.sink_size = sink_size
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.cached_token_sequence: List[int] = []
        self.next_global_position: int = 0
        self.next_token_logits: torch.Tensor = torch.zeros(
            vocab_size, dtype=torch.float32,
        )
        self.call_log: List[tuple] = []

    @property
    def _budget(self) -> int:
        return self.sink_size + self.window_size

    def _logits_for(self, history: List[int]) -> torch.Tensor:
        """Deterministic logits derived from the last 3 history tokens."""
        out = torch.zeros(self.vocab_size, dtype=torch.float32)
        if history:
            recent = history[-3:]
            argmax = sum(recent) % self.vocab_size
            out[argmax] = 1.0
        return out

    def _sink_window_trim(self, sequence: List[int]) -> List[int]:
        if len(sequence) <= self._budget:
            return list(sequence)
        return (
            list(sequence[: self.sink_size])
            + list(sequence[-self.window_size :])
        )

    def k_seq_length(self, session: object) -> int:  # noqa: ARG002 — protocol
        del session
        return len(self.cached_token_sequence)

    def prefill(self, prompt_ids: List[int]) -> None:
        self.call_log.append(("prefill", tuple(prompt_ids)))
        self.cached_token_sequence = self._sink_window_trim(prompt_ids)
        self.next_global_position = len(prompt_ids)
        self.next_token_logits = self._logits_for(self.cached_token_sequence)

    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        self.call_log.append(("forward_block", tuple(tokens)))
        L = len(tokens)
        out = torch.zeros((L, self.vocab_size), dtype=torch.float32)
        running = list(self.cached_token_sequence)
        for i, t in enumerate(tokens):
            running.append(t)
            out[i] = self._logits_for(running)
        # Mirror real verifier: cached_token_sequence is extended in
        # forward_block (pre-trim); commit_or_truncate applies trim.
        self.cached_token_sequence = list(self.cached_token_sequence) + list(tokens)
        return out

    def commit_or_truncate(self, *, forwarded: int, accepted: int) -> None:
        self.call_log.append(("commit_or_truncate", forwarded, accepted))
        drop = forwarded - accepted
        if drop > 0:
            self.cached_token_sequence = self.cached_token_sequence[:-drop]
        self.next_global_position += accepted
        self.cached_token_sequence = self._sink_window_trim(
            self.cached_token_sequence,
        )


# ---------------------------------------------------------------------------
# Confirm FakeVerifier satisfies VerifierProtocol structurally.
# ---------------------------------------------------------------------------


def test_fake_verifier_is_structurally_a_verifier_protocol():
    fv = FakeVerifier()
    # Smoke: every protocol member resolves on the fake.
    assert callable(fv.prefill)
    assert callable(fv.forward_block)
    assert callable(fv.commit_or_truncate)
    assert callable(fv.k_seq_length)
    assert isinstance(fv.cached_token_sequence, list)
    assert isinstance(fv.next_global_position, int)
    assert isinstance(fv.next_token_logits, torch.Tensor)
    # Confirms the public name re-exports cleanly.
    _: VerifierProtocol = fv  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dispatch logic: cold start vs. incremental
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_first_call_uses_prefill(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20, 30])
        kinds = [c[0] for c in fv.call_log]
        assert kinds == ["prefill"]
        assert fv.call_log[0] == ("prefill", (10, 20, 30))

    def test_second_call_uses_forward_block_then_commit(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20, 30])
        coord.append_tokens(sess.session_id, [40, 50])
        kinds = [c[0] for c in fv.call_log]
        assert kinds == ["prefill", "forward_block", "commit_or_truncate"]
        assert fv.call_log[1] == ("forward_block", (40, 50))
        # commit_or_truncate(forwarded=2, accepted=2) — full accept
        assert fv.call_log[2] == ("commit_or_truncate", 2, 2)

    def test_third_call_again_uses_incremental(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1])
        coord.append_tokens(sess.session_id, [2])
        coord.append_tokens(sess.session_id, [3])
        kinds = [c[0] for c in fv.call_log]
        # 1 prefill, then 2 incremental pairs of (forward_block, commit_or_truncate)
        assert kinds == [
            "prefill",
            "forward_block", "commit_or_truncate",
            "forward_block", "commit_or_truncate",
        ]


# ---------------------------------------------------------------------------
# State mirroring: session ↔ verifier consistency
# ---------------------------------------------------------------------------


class TestStateMirroring:
    def test_session_history_extends(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        new_len = coord.append_tokens(sess.session_id, [10, 20, 30])
        assert new_len == 3
        assert sess.history_token_ids == [10, 20, 30]

    def test_session_history_grows_across_calls(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20])
        coord.append_tokens(sess.session_id, [30, 40, 50])
        assert sess.history_token_ids == [10, 20, 30, 40, 50]

    def test_session_cached_token_sequence_mirrors_verifier(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier(sink_size=2, window_size=4)
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Sink+window = 6; verifier trims to [1,2] + [6,7,8,9] = [1,2,6,7,8,9]
        assert fv.cached_token_sequence == [1, 2, 6, 7, 8, 9]
        assert sess.cached_token_sequence == [1, 2, 6, 7, 8, 9]

    def test_session_position_mirrors_verifier(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20, 30])
        coord.append_tokens(sess.session_id, [40])
        assert sess.next_global_position == 4
        assert fv.next_global_position == 4

    def test_next_token_logits_set_after_incremental_call(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10])
        # After first call: next_token_logits is set inside prefill.
        before = fv.next_token_logits.clone()
        coord.append_tokens(sess.session_id, [20])
        # After second call: next_token_logits has been re-assigned to
        # block_logits[-1].clone() — it must NOT be the prefill-time
        # tensor (incremental path didn't fall through to prefill).
        assert not torch.equal(fv.next_token_logits, before)


# ---------------------------------------------------------------------------
# INV-3 byte-exactness: same input via different chunkings → same final state
# ---------------------------------------------------------------------------


class TestInv3ByteExactDispatch:
    def test_one_call_vs_two_calls_produce_same_cache(self):
        full = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        # Path A: one big call
        store_a = SessionStore(capacity=1)
        fv_a = FakeVerifier(sink_size=2, window_size=4)
        coord_a = AppendTokensCoordinator(store_a, fv_a)
        sess_a = store_a.create_session()
        coord_a.append_tokens(sess_a.session_id, full)
        # Path B: split into two calls (5 + 4)
        store_b = SessionStore(capacity=1)
        fv_b = FakeVerifier(sink_size=2, window_size=4)
        coord_b = AppendTokensCoordinator(store_b, fv_b)
        sess_b = store_b.create_session()
        coord_b.append_tokens(sess_b.session_id, full[:5])
        coord_b.append_tokens(sess_b.session_id, full[5:])
        # INV-3: byte-equal final state
        assert fv_a.cached_token_sequence == fv_b.cached_token_sequence
        assert fv_a.next_global_position == fv_b.next_global_position
        assert torch.equal(fv_a.next_token_logits, fv_b.next_token_logits)
        # And the session state mirrors that, by construction.
        assert sess_a.cached_token_sequence == sess_b.cached_token_sequence
        assert sess_a.next_global_position == sess_b.next_global_position

    def test_chunking_invariance_across_many_splits(self):
        full = list(range(100, 130))  # 30 tokens
        # Three chunkings: one big, four medium, fifteen tiny pairs.
        chunkings = [
            [full],
            [full[:7], full[7:14], full[14:21], full[21:]],
            [full[i:i + 2] for i in range(0, len(full), 2)],
        ]
        results = []
        for chunks in chunkings:
            store = SessionStore(capacity=1)
            fv = FakeVerifier(sink_size=3, window_size=5)
            coord = AppendTokensCoordinator(store, fv)
            sess = store.create_session()
            for c in chunks:
                coord.append_tokens(sess.session_id, c)
            results.append((
                tuple(fv.cached_token_sequence),
                fv.next_global_position,
                tuple(fv.next_token_logits.tolist()),
            ))
        # All three chunkings produce the same final state.
        assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Empty / boundary cases
# ---------------------------------------------------------------------------


class TestEmptyAppend:
    def test_empty_token_list_is_noop(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        new_len = coord.append_tokens(sess.session_id, [])
        assert new_len == 0
        assert fv.call_log == []  # verifier untouched
        assert sess.history_token_ids == []
        assert sess.next_global_position == 0

    def test_empty_append_after_real_append_is_noop_on_verifier(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1, 2, 3])
        log_before = list(fv.call_log)
        new_len = coord.append_tokens(sess.session_id, [])
        assert new_len == 3
        # Verifier was not called again — no extra forward_block.
        assert fv.call_log == log_before

    def test_empty_append_advances_last_active_at(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        before = sess.last_active_at
        time.sleep(0.001)
        coord.append_tokens(sess.session_id, [])
        assert sess.last_active_at > before


# ---------------------------------------------------------------------------
# Error mapping: SessionNotFoundError, ValueError, InvariantViolation
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_session_raises_session_not_found(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        with pytest.raises(SessionNotFoundError):
            coord.append_tokens("sess-unknown", [1, 2, 3])

    def test_negative_token_id_raises_value_error(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        with pytest.raises(ValueError, match="non-negative"):
            coord.append_tokens(sess.session_id, [10, -1])

    def test_inv1_violation_propagates_through_coordinator(self):
        # Use a verifier that misreports cached state -> INV-1 fires
        # when store._assert_inv1 compares len(session.cached_token_sequence)
        # against the cache_inspector's k_seq_length.
        class _LyingVerifier(FakeVerifier):
            def k_seq_length(self, session):
                # Always lie: report a length the session never has.
                del session
                return 999

        fv = _LyingVerifier()
        store = SessionStore(capacity=1, cache_inspector=fv)
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        with pytest.raises(InvariantViolation) as exc:
            coord.append_tokens(sess.session_id, [1, 2, 3])
        assert exc.value.kind == "1"
        # Session was removed from the store — follow-up RPCs surface NOT_FOUND.
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)

    def test_inv2_violation_propagates_through_coordinator(self):
        # Use a verifier that returns a regressing next_global_position
        # so the store.record_position_advance INV-2 check fires.
        class _RegressingVerifier(FakeVerifier):
            def __init__(self):
                super().__init__()
                self._calls = 0

            def commit_or_truncate(self, *, forwarded, accepted):
                super().commit_or_truncate(forwarded=forwarded, accepted=accepted)
                self._calls += 1
                if self._calls == 1:
                    # On the SECOND coordinator append (= first
                    # commit_or_truncate), regress position to trip INV-2.
                    self.next_global_position = 0

        fv = _RegressingVerifier()
        store = SessionStore(capacity=1)
        coord = AppendTokensCoordinator(store, fv)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1, 2, 3])  # first call ok
        with pytest.raises(InvariantViolation) as exc:
            coord.append_tokens(sess.session_id, [4])
        assert exc.value.kind == "2"
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)


# ---------------------------------------------------------------------------
# Constructor / repr surface
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_stores_store_and_verifier_references(self):
        store = SessionStore(capacity=1)
        fv = FakeVerifier()
        coord = AppendTokensCoordinator(store, fv)
        # Coordinator can use both — there are no public accessors,
        # so we exercise via append_tokens.
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1])
        assert fv.next_global_position == 1
        assert sess.history_length == 1
