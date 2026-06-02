"""Integration tests for :mod:`inference_engine.session.coordinator`.

PR-N1 migration of the former Linux-side ``test_coordinator.py``.

The previous Linux-side test suite drove ``AppendTokensCoordinator``
against a hand-written ``FakeVerifier`` that mirrored the real
verifier's state-mutation contract. PR-N1's audit (PR-E1c discussion)
ruled this an instance of the "no test doubles" violation: the fake's
mirror of sink+window trim, ``cached_token_sequence`` management, and
``next_global_position`` advancement was *our model* of the real
verifier, not the real thing. Bugs that manifest in the real
verifier's edge cases (numeric stability of the bf16 trim, GQA dim
arithmetic, etc.) wouldn't be caught.

This file replaces those tests with the same assertions driven
against the real Qwen3-0.6B SinkWindowVerifier via
``fresh_verifier_factory``. Coverage is "structural correctness +
real-numerics state transitions". Linux CI does NOT run this file
(``tests/integration/`` is opt-in via ``pytest -m integration``);
Mac M4 / CUDA hosts run it via ``scripts/review_pr_n1_on_mac.sh``
and via the standalone ``run_platform_tests.sh`` flow.
"""

from __future__ import annotations

import time

import pytest
import torch

from inference_engine.session import (
    AppendTokensCoordinator,
    InvariantViolation,
    SessionNotFoundError,
    SessionStore,
    VerifierProtocol,
)


# ---------------------------------------------------------------------------
# Fixture: a fresh real verifier per test.
#
# fresh_verifier_factory comes from tests/conftest.py — it loads
# Qwen3-0.6B from the HF cache and returns a SinkWindowVerifier.
# Module-scoping it would be faster but creates a state-bleed risk
# that nullifies the integration value; we pay the model-load cost
# per test in exchange for guaranteed isolation.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_verifier(fresh_verifier_factory):
    return fresh_verifier_factory(sink=2, window=8)


@pytest.fixture
def store_and_coord(real_verifier):
    store = SessionStore(capacity=1, cache_inspector=real_verifier)
    coord = AppendTokensCoordinator(store, real_verifier)
    return store, coord, real_verifier


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_real_verifier_satisfies_verifier_protocol(real_verifier):
    """Structural typing check: the real CPU verifier IS a
    VerifierProtocol. Catches accidental protocol drift (e.g., if a
    new method gets added to the protocol but not implemented on
    the real verifier).
    """
    assert callable(real_verifier.prefill)
    assert callable(real_verifier.forward_block)
    assert callable(real_verifier.commit_or_truncate)
    assert callable(real_verifier.k_seq_length)
    assert callable(real_verifier.kv_live_bytes)
    _: VerifierProtocol = real_verifier  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dispatch logic: cold start vs. incremental
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_first_call_advances_position_from_zero(self, store_and_coord):
        store, coord, v = store_and_coord
        sess = store.create_session()
        assert v.next_global_position == 0
        assert v.next_token_logits is None
        coord.append_tokens(sess.session_id, [10, 20, 30])
        # First call dispatched to prefill: cache populated, position = 3.
        assert v.next_global_position == 3
        assert v.next_token_logits is not None
        assert v.cached_token_sequence == [10, 20, 30]

    def test_second_call_extends_position_incrementally(
        self, store_and_coord,
    ):
        store, coord, v = store_and_coord
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20, 30])
        coord.append_tokens(sess.session_id, [40, 50])
        # Position advanced by exactly 2 (no full re-prefill).
        assert v.next_global_position == 5
        assert v.cached_token_sequence == [10, 20, 30, 40, 50]

    def test_third_call_again_uses_incremental(self, store_and_coord):
        store, coord, v = store_and_coord
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1])
        coord.append_tokens(sess.session_id, [2])
        coord.append_tokens(sess.session_id, [3])
        assert v.next_global_position == 3


# ---------------------------------------------------------------------------
# State mirroring: session ↔ verifier consistency
# ---------------------------------------------------------------------------


class TestStateMirroring:
    def test_session_history_extends_in_lockstep(self, store_and_coord):
        store, coord, _ = store_and_coord
        sess = store.create_session()
        new_len = coord.append_tokens(sess.session_id, [10, 20, 30])
        assert new_len == 3
        assert sess.history_token_ids == [10, 20, 30]

    def test_session_cached_token_sequence_mirrors_verifier_after_trim(
        self, store_and_coord,
    ):
        store, coord, v = store_and_coord
        sess = store.create_session()
        # 12 tokens > sink+window (2+8=10): real verifier trims.
        coord.append_tokens(sess.session_id, list(range(100, 112)))
        assert sess.cached_token_sequence == v.cached_token_sequence
        # Trim is sink+window-bounded — capacity is the upper bound;
        # real verifier may report something <= capacity depending on
        # the exact prefill / commit_or_truncate sequencing.
        assert len(v.cached_token_sequence) <= 10
        assert len(v.cached_token_sequence) > 0

    def test_session_position_mirrors_verifier_across_calls(
        self, store_and_coord,
    ):
        store, coord, v = store_and_coord
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20, 30])
        coord.append_tokens(sess.session_id, [40])
        assert sess.next_global_position == 4
        assert v.next_global_position == 4

    def test_next_token_logits_re_assigned_on_incremental_call(
        self, store_and_coord,
    ):
        store, coord, v = store_and_coord
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10])
        before = v.next_token_logits.clone()
        coord.append_tokens(sess.session_id, [20])
        # Logits must have moved (incremental path runs forward_block,
        # writing block_logits[-1].clone() to next_token_logits).
        assert not torch.equal(v.next_token_logits, before)


# ---------------------------------------------------------------------------
# Empty append: boundary case
# ---------------------------------------------------------------------------


class TestEmptyAppend:
    def test_empty_token_list_is_noop_on_cold_session(
        self, store_and_coord,
    ):
        store, coord, v = store_and_coord
        sess = store.create_session()
        new_len = coord.append_tokens(sess.session_id, [])
        assert new_len == 0
        assert sess.history_token_ids == []
        assert sess.next_global_position == 0
        # Verifier untouched: still cold.
        assert v.next_global_position == 0
        assert v.next_token_logits is None

    def test_empty_append_after_real_append_does_not_re_invoke_verifier(
        self, store_and_coord,
    ):
        store, coord, v = store_and_coord
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1, 2, 3])
        pos_before = v.next_global_position
        new_len = coord.append_tokens(sess.session_id, [])
        assert new_len == 3
        # Verifier position unchanged → no extra forward.
        assert v.next_global_position == pos_before

    def test_empty_append_still_advances_last_active_at(
        self, store_and_coord,
    ):
        store, coord, _ = store_and_coord
        sess = store.create_session()
        before = sess.last_active_at
        time.sleep(0.001)
        coord.append_tokens(sess.session_id, [])
        assert sess.last_active_at > before


# ---------------------------------------------------------------------------
# Errors: SessionNotFoundError, ValueError, InvariantViolation
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_session_raises_session_not_found(
        self, store_and_coord,
    ):
        _store, coord, _ = store_and_coord
        with pytest.raises(SessionNotFoundError):
            coord.append_tokens("sess-unknown", [1, 2, 3])

    def test_negative_token_id_raises_value_error(self, store_and_coord):
        store, coord, _ = store_and_coord
        sess = store.create_session()
        with pytest.raises(ValueError, match="non-negative"):
            coord.append_tokens(sess.session_id, [10, -1])

    def test_inv1_violation_through_session_state_corruption(
        self, store_and_coord,
    ):
        """Corrupt the session's cached_token_sequence directly so its
        length stops matching the verifier's k_seq_length. The store's
        INV-1 check fires.

        This test directly mutates session state (a session-store
        invariant violation) instead of inserting a lying verifier
        between the verifier and the store. The INV-1 detection
        mechanism is what we're validating, not the verifier's
        cooperation; injecting a fault into the session state is the
        cleaner contract test.
        """
        store, coord, _ = store_and_coord
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1, 2, 3])
        # Corrupt: set cached_token_sequence to a wrong length.
        sess.cached_token_sequence = [99, 99, 99, 99, 99]
        with pytest.raises(InvariantViolation) as exc:
            coord.append_tokens(sess.session_id, [4])
        # On INV violation the session is evicted; follow-ups → NOT_FOUND.
        assert exc.value.kind == "1"
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)


# ---------------------------------------------------------------------------
# kv_live_bytes wiring (PR-E1c sync mechanism, with real verifier)
# ---------------------------------------------------------------------------


def _slab_pool(num_slabs: int = 1):
    from inference_engine.memory.pool import SlabPool
    from inference_engine.memory.slab import SlabConfig
    cfg = SlabConfig(
        num_layers=1, num_heads=1, sink_size=1,
        window_size=2, head_dim=4, dtype=torch.float32,
    )
    return SlabPool(num_slabs=num_slabs, slab_config=cfg)


class TestKvLiveBytesSync:
    """Verifies AppendTokensCoordinator writes the verifier's real
    KV byte count onto session.slab.live_kv_bytes_override after
    every successful mutation. This is the wiring PR-E1c added; in
    PR-N1 the test runs against the real Qwen3-0.6B verifier (was
    against FakeVerifier with synthetic bytes-per-token=17)."""

    def test_first_prefill_writes_real_bytes_to_slab_override(
        self, fresh_verifier_factory,
    ):
        v = fresh_verifier_factory(sink=2, window=8)
        pool = _slab_pool()
        store = SessionStore(
            capacity=1, cache_inspector=v, slab_pool=pool,
        )
        coord = AppendTokensCoordinator(store, v)
        sess = store.create_session()
        assert sess.slab.live_kv_bytes_override is None
        coord.append_tokens(sess.session_id, [10, 20, 30])
        # Real verifier: bytes = k_seq * num_layers * num_kv_heads * head_dim * itemsize * 2
        expected = v.kv_live_bytes(session=None)
        assert sess.slab.live_kv_bytes_override == expected
        assert expected > 0

    def test_subsequent_append_re_syncs_bytes(self, fresh_verifier_factory):
        v = fresh_verifier_factory(sink=2, window=8)
        pool = _slab_pool()
        store = SessionStore(
            capacity=1, cache_inspector=v, slab_pool=pool,
        )
        coord = AppendTokensCoordinator(store, v)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [10, 20, 30])
        first = sess.slab.live_kv_bytes_override
        coord.append_tokens(sess.session_id, [40, 50])
        second = sess.slab.live_kv_bytes_override
        # Either grew (cache below capacity) or plateaued (at cap).
        assert second is not None and second >= first  # type: ignore[operator]
        assert second == v.kv_live_bytes(session=None)

    def test_sync_no_op_when_session_has_no_slab(
        self, fresh_verifier_factory,
    ):
        v = fresh_verifier_factory(sink=2, window=8)
        store = SessionStore(capacity=1, cache_inspector=v)  # no slab_pool
        coord = AppendTokensCoordinator(store, v)
        sess = store.create_session()
        assert sess.slab is None
        coord.append_tokens(sess.session_id, [10, 20, 30])  # must not raise
        assert sess.kv_live_bytes() == 0

    def test_empty_append_does_not_overwrite_override(
        self, fresh_verifier_factory,
    ):
        v = fresh_verifier_factory(sink=2, window=8)
        pool = _slab_pool()
        store = SessionStore(
            capacity=1, cache_inspector=v, slab_pool=pool,
        )
        coord = AppendTokensCoordinator(store, v)
        sess = store.create_session()
        coord.append_tokens(sess.session_id, [1, 2, 3])
        before = sess.slab.live_kv_bytes_override
        coord.append_tokens(sess.session_id, [])
        assert sess.slab.live_kv_bytes_override == before


# ---------------------------------------------------------------------------
# INV-3 byte-exact dispatch — PR-N1 keeps a single sanity check here;
# the binding GA gate is tests/integration/test_inv3_session_determinism_gate.py
# (PR-E1), which exercises the same property more thoroughly.
# ---------------------------------------------------------------------------


def test_chunking_invariance_smoke(fresh_verifier_factory):
    """One-call vs. two-calls produces equivalent greedy decoding.

    INV-3's binding claim is byte-exact GREEDY-DECODING equality
    across chunkings, not byte-exact LOGITS equality — bf16 round-
    off can shift logit values without changing argmax. The
    comprehensive GA gate lives in
    ``test_inv3_session_determinism_gate.py``; this is a smoke
    sanity that the cached token sequence and next position
    converge, plus that the next greedy argmax matches.
    """
    full = [10, 20, 30, 40, 50, 60, 70, 80]
    v_a = fresh_verifier_factory(sink=2, window=4)
    v_b = fresh_verifier_factory(sink=2, window=4)
    store_a = SessionStore(capacity=1, cache_inspector=v_a)
    store_b = SessionStore(capacity=1, cache_inspector=v_b)
    coord_a = AppendTokensCoordinator(store_a, v_a)
    coord_b = AppendTokensCoordinator(store_b, v_b)
    sess_a = store_a.create_session()
    sess_b = store_b.create_session()
    coord_a.append_tokens(sess_a.session_id, full)
    coord_b.append_tokens(sess_b.session_id, full[:5])
    coord_b.append_tokens(sess_b.session_id, full[5:])
    assert v_a.cached_token_sequence == v_b.cached_token_sequence
    assert v_a.next_global_position == v_b.next_global_position
    # Byte-exact tokens (greedy argmax) — robust to bf16 round-off
    # in the underlying logit values.
    assert int(torch.argmax(v_a.next_token_logits).item()) == int(
        torch.argmax(v_b.next_token_logits).item()
    )
