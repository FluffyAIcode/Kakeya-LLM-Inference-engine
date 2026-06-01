"""Unit tests for inference_engine.session.store (ADR 0008 PR-A2).

Coverage target: 100%. No mocks of the SUT. Synthetic
:class:`CacheInspector` implementations are used **only** to drive
INV-1 enforcement through the public contract (the inspector is the
documented PR-A3 injection point).
"""

from __future__ import annotations

import time

import pytest

from inference_engine.memory.pool import PoolExhausted, SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.session import (
    CacheInspector,
    InvariantViolation,
    Session,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)


def _tiny_slab_pool(num_slabs: int = 4) -> SlabPool:
    """Construct a minimal SlabPool for SessionStore-with-pool tests.

    Dimensions are deliberately small so the test suite stays fast on
    Linux CI runners; we are testing the SessionStore <-> pool wiring,
    not the slab tensors themselves (those have their own coverage in
    tests/inference_engine/memory/test_pool.py).
    """
    cfg = SlabConfig(
        num_layers=1,
        num_heads=1,
        sink_size=1,
        window_size=2,
        head_dim=4,
    )
    return SlabPool(num_slabs=num_slabs, slab_config=cfg)


class _SyntheticInspector:
    """Minimal :class:`CacheInspector` that reports a configurable
    K/V sequence length. Used to drive INV-1 enforcement tests
    through the documented public protocol."""

    def __init__(self, k_seq_length: int) -> None:
        self._k_seq_length = k_seq_length

    def k_seq_length(self, session: Session) -> int:
        return self._k_seq_length


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_capacity_zero_rejected(self):
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            SessionStore(capacity=0)

    def test_capacity_negative_rejected(self):
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            SessionStore(capacity=-3)

    def test_capacity_property(self):
        assert SessionStore(capacity=7).capacity == 7

    def test_active_count_starts_at_zero(self):
        assert SessionStore(capacity=2).active_count == 0

    def test_total_kv_live_bytes_starts_at_zero(self):
        assert SessionStore(capacity=2).total_kv_live_bytes == 0


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_session_with_server_issued_id(self):
        store = SessionStore(capacity=2)
        sess = store.create_session()
        assert sess.session_id.startswith("sess-")
        assert len(sess.session_id) == len("sess-") + 32

    def test_two_sessions_get_distinct_ids(self):
        store = SessionStore(capacity=2)
        a = store.create_session()
        b = store.create_session()
        assert a.session_id != b.session_id

    def test_active_count_increments(self):
        store = SessionStore(capacity=3)
        store.create_session()
        store.create_session()
        assert store.active_count == 2

    def test_eos_token_ids_recorded_as_immutable_tuple(self):
        store = SessionStore(capacity=1)
        sess = store.create_session(eos_token_ids=[1, 2, 3])
        assert sess.eos_token_ids == (1, 2, 3)
        assert isinstance(sess.eos_token_ids, tuple)

    def test_eos_token_ids_default_empty_tuple(self):
        sess = SessionStore(capacity=1).create_session()
        assert sess.eos_token_ids == ()

    def test_client_label_recorded(self):
        sess = SessionStore(capacity=1).create_session(client_label="demo-1")
        assert sess.client_label == "demo-1"

    def test_client_label_default_empty_string(self):
        assert SessionStore(capacity=1).create_session().client_label == ""

    def test_history_starts_empty(self):
        sess = SessionStore(capacity=1).create_session()
        assert sess.history_token_ids == []
        assert sess.history_length == 0

    def test_cached_token_sequence_starts_empty(self):
        sess = SessionStore(capacity=1).create_session()
        assert sess.cached_token_sequence == []

    def test_initial_position_is_zero(self):
        sess = SessionStore(capacity=1).create_session()
        assert sess.next_global_position == 0

    def test_initial_violation_counters_are_zero(self):
        sess = SessionStore(capacity=1).create_session()
        assert sess.inv1_violations == 0
        assert sess.inv2_violations == 0

    def test_eos_token_ids_iterable_input_accepted(self):
        sess = SessionStore(capacity=1).create_session(
            eos_token_ids=iter([7, 11]),
        )
        assert sess.eos_token_ids == (7, 11)


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_returns_session_by_id(self):
        store = SessionStore(capacity=2)
        sess = store.create_session()
        assert store.get_session(sess.session_id) is sess

    def test_unknown_id_raises_session_not_found(self):
        store = SessionStore(capacity=2)
        with pytest.raises(SessionNotFoundError) as exc:
            store.get_session("sess-unknown")
        assert exc.value.session_id == "sess-unknown"

    def test_closed_session_id_raises_session_not_found(self):
        store = SessionStore(capacity=2)
        sess = store.create_session()
        store.close_session(sess.session_id)
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)


# ---------------------------------------------------------------------------
# append_tokens
# ---------------------------------------------------------------------------


class TestAppendTokens:
    def test_grows_history(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        new_len = store.append_tokens(sess.session_id, [10, 20, 30])
        assert new_len == 3
        assert sess.history_token_ids == [10, 20, 30]

    def test_returns_new_history_length(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.append_tokens(sess.session_id, [10, 20])
        new_len = store.append_tokens(sess.session_id, [30])
        assert new_len == 3

    def test_append_only_extends_never_rewrites(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.append_tokens(sess.session_id, [10, 20])
        store.append_tokens(sess.session_id, [30, 40])
        assert sess.history_token_ids == [10, 20, 30, 40]

    def test_empty_token_list_is_noop(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.append_tokens(sess.session_id, [10])
        new_len = store.append_tokens(sess.session_id, [])
        assert new_len == 1
        assert sess.history_token_ids == [10]

    def test_iterable_input_accepted(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.append_tokens(sess.session_id, iter([10, 20, 30]))
        assert sess.history_token_ids == [10, 20, 30]

    def test_unknown_session_raises(self):
        store = SessionStore(capacity=1)
        with pytest.raises(SessionNotFoundError):
            store.append_tokens("sess-unknown", [10])

    def test_negative_token_id_rejected(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        with pytest.raises(ValueError, match="non-negative"):
            store.append_tokens(sess.session_id, [10, -1])

    def test_non_int_token_id_rejected(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        with pytest.raises(ValueError, match="non-negative"):
            store.append_tokens(sess.session_id, [10, "20"])

    def test_bool_token_id_rejected(self):
        # bool is a subclass of int in Python; passing True as a
        # token id is almost always a caller bug.
        store = SessionStore(capacity=1)
        sess = store.create_session()
        with pytest.raises(ValueError, match="non-negative"):
            store.append_tokens(sess.session_id, [10, True])

    def test_validation_failure_does_not_mutate_history(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.append_tokens(sess.session_id, [10])
        with pytest.raises(ValueError):
            store.append_tokens(sess.session_id, [20, -1, 30])
        assert sess.history_token_ids == [10]

    def test_advances_last_active_at(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        before = sess.last_active_at
        time.sleep(0.001)
        store.append_tokens(sess.session_id, [10])
        assert sess.last_active_at > before


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------


class TestCloseSession:
    def test_returns_final_history_length(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.append_tokens(sess.session_id, [10, 20, 30])
        assert store.close_session(sess.session_id) == 3

    def test_returns_zero_for_empty_session(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        assert store.close_session(sess.session_id) == 0

    def test_removes_from_store(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.close_session(sess.session_id)
        assert store.active_count == 0

    def test_unknown_session_raises(self):
        store = SessionStore(capacity=1)
        with pytest.raises(SessionNotFoundError):
            store.close_session("sess-unknown")

    def test_double_close_raises(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.close_session(sess.session_id)
        with pytest.raises(SessionNotFoundError):
            store.close_session(sess.session_id)


# ---------------------------------------------------------------------------
# LRU eviction at capacity
# ---------------------------------------------------------------------------


class TestLruEviction:
    def test_creating_at_capacity_evicts_lru(self):
        store = SessionStore(capacity=2)
        a = store.create_session()
        b = store.create_session()
        # Touch b so a becomes the LRU.
        time.sleep(0.001)
        store.append_tokens(b.session_id, [1])
        c = store.create_session()
        assert store.active_count == 2
        with pytest.raises(SessionNotFoundError):
            store.get_session(a.session_id)
        assert store.get_session(b.session_id) is b
        assert store.get_session(c.session_id) is c

    def test_eviction_picks_least_recently_active_strictly(self):
        store = SessionStore(capacity=3)
        a = store.create_session()
        time.sleep(0.001)
        b = store.create_session()
        time.sleep(0.001)
        c = store.create_session()
        # Touch a so b becomes LRU.
        store.append_tokens(a.session_id, [1])
        d = store.create_session()
        assert store.get_session(a.session_id) is a
        with pytest.raises(SessionNotFoundError):
            store.get_session(b.session_id)
        assert store.get_session(c.session_id) is c
        assert store.get_session(d.session_id) is d


# ---------------------------------------------------------------------------
# evict_idle (TTL eviction)
# ---------------------------------------------------------------------------


class TestEvictIdle:
    def test_no_eviction_when_all_below_ttl(self):
        store = SessionStore(capacity=3)
        store.create_session()
        store.create_session()
        evicted = store.evict_idle(ttl_seconds=10.0)
        assert evicted == []
        assert store.active_count == 2

    def test_evicts_session_above_ttl(self):
        store = SessionStore(capacity=3)
        sess = store.create_session()
        future = sess.last_active_at + 100.0
        evicted = store.evict_idle(ttl_seconds=10.0, now=future)
        assert evicted == [sess]
        assert store.active_count == 0

    def test_evicts_only_those_above_ttl(self):
        store = SessionStore(capacity=3)
        a = store.create_session()
        b = store.create_session()
        a.last_active_at = 0.0
        b.last_active_at = 1000.0
        evicted = store.evict_idle(ttl_seconds=10.0, now=1001.0)
        assert evicted == [a]
        assert store.active_count == 1
        assert store.get_session(b.session_id) is b

    def test_now_default_uses_monotonic_clock(self):
        store = SessionStore(capacity=3)
        sess = store.create_session()
        sess.last_active_at = time.monotonic() - 100.0
        evicted = store.evict_idle(ttl_seconds=10.0)
        assert evicted == [sess]

    def test_at_threshold_is_evicted(self):
        # idle == ttl_seconds — boundary case. ">=", so it counts.
        store = SessionStore(capacity=3)
        sess = store.create_session()
        sess.last_active_at = 1000.0
        evicted = store.evict_idle(ttl_seconds=5.0, now=1005.0)
        assert evicted == [sess]


# ---------------------------------------------------------------------------
# record_position_advance / INV-2
# ---------------------------------------------------------------------------


class TestRecordPositionAdvance:
    def test_advances_position(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 5)
        assert sess.next_global_position == 5

    def test_monotonic_advance_succeeds(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 5)
        store.record_position_advance(sess.session_id, 10)
        assert sess.next_global_position == 10

    def test_equal_position_succeeds(self):
        # Non-decreasing means equal is OK.
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 5)
        store.record_position_advance(sess.session_id, 5)
        assert sess.next_global_position == 5

    def test_decreasing_position_raises_invariant_violation(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 10)
        with pytest.raises(InvariantViolation) as exc:
            store.record_position_advance(sess.session_id, 5)
        assert exc.value.kind == "2"
        assert exc.value.session_id == sess.session_id
        assert "must be non-decreasing" in exc.value.detail
        assert "current=10" in exc.value.detail
        assert "requested=5" in exc.value.detail

    def test_inv2_violation_increments_counter_on_session_ref(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 10)
        with pytest.raises(InvariantViolation):
            store.record_position_advance(sess.session_id, 5)
        # The store removes the session on violation, but the local
        # reference is still valid and shows the counter increment.
        assert sess.inv2_violations == 1

    def test_inv2_violation_removes_session_from_store(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 10)
        with pytest.raises(InvariantViolation):
            store.record_position_advance(sess.session_id, 5)
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)

    def test_unknown_session_raises_not_found(self):
        store = SessionStore(capacity=1)
        with pytest.raises(SessionNotFoundError):
            store.record_position_advance("sess-unknown", 5)

    def test_advances_last_active_at(self):
        store = SessionStore(capacity=1)
        sess = store.create_session()
        before = sess.last_active_at
        time.sleep(0.001)
        store.record_position_advance(sess.session_id, 1)
        assert sess.last_active_at > before


# ---------------------------------------------------------------------------
# INV-1 with cache inspector
# ---------------------------------------------------------------------------


class TestInv1WithCacheInspector:
    def test_no_inspector_makes_inv1_trivial(self):
        # Even if cached_token_sequence is "wrong" shape (synthetically
        # mismatched), no inspector means there is no cache to compare
        # against, so INV-1 cannot be violated.
        store = SessionStore(capacity=1)
        sess = store.create_session()
        sess.cached_token_sequence.append(99)
        store.append_tokens(sess.session_id, [1])
        # No exception raised.
        assert sess.history_token_ids == [1]

    def test_inspector_agreeing_passes(self):
        # Inspector reports K/V seq len = 0; cached_token_sequence = [].
        store = SessionStore(
            capacity=1,
            cache_inspector=_SyntheticInspector(k_seq_length=0),
        )
        sess = store.create_session()
        store.append_tokens(sess.session_id, [1, 2, 3])
        # cached_token_sequence is still [] in PR-A2 (PR-A3 wires it);
        # inspector says 0; so INV-1 holds.
        assert sess.cached_token_sequence == []

    def test_inspector_disagreeing_raises_invariant_violation(self):
        store = SessionStore(
            capacity=1,
            cache_inspector=_SyntheticInspector(k_seq_length=5),
        )
        sess = store.create_session()
        with pytest.raises(InvariantViolation) as exc:
            store.append_tokens(sess.session_id, [1])
        assert exc.value.kind == "1"
        assert exc.value.session_id == sess.session_id
        assert "cached_token_sequence length (0)" in exc.value.detail
        assert "K/V tensor sequence length (5)" in exc.value.detail

    def test_inv1_violation_increments_counter_on_session_ref(self):
        store = SessionStore(
            capacity=1,
            cache_inspector=_SyntheticInspector(k_seq_length=5),
        )
        sess = store.create_session()
        with pytest.raises(InvariantViolation):
            store.append_tokens(sess.session_id, [1])
        assert sess.inv1_violations == 1

    def test_inv1_violation_removes_session_from_store(self):
        store = SessionStore(
            capacity=1,
            cache_inspector=_SyntheticInspector(k_seq_length=5),
        )
        sess = store.create_session()
        with pytest.raises(InvariantViolation):
            store.append_tokens(sess.session_id, [1])
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)

    def test_cache_inspector_protocol_is_structural(self):
        # The Protocol is structural; any object with a k_seq_length
        # method that takes a Session and returns int satisfies it.
        # Verifies the public extension point is usable from external
        # code without subclassing.
        class _AnotherInspector:
            def k_seq_length(self, session):
                return 0

        # mypy-equivalent runtime check: assignment to the Protocol-typed
        # parameter must succeed.
        store = SessionStore(
            capacity=1, cache_inspector=_AnotherInspector(),
        )
        sess = store.create_session()
        store.append_tokens(sess.session_id, [1])
        assert sess.history_length == 1

    def test_protocol_class_is_publicly_exported(self):
        # Confirms CacheInspector is reachable from the package root.
        assert hasattr(CacheInspector, "k_seq_length")


# ---------------------------------------------------------------------------
# total_kv_live_bytes (PR-A2 semantics)
# ---------------------------------------------------------------------------


class TestTotalKvLiveBytes:
    def test_zero_when_no_sessions(self):
        assert SessionStore(capacity=2).total_kv_live_bytes == 0

    def test_zero_for_pool_less_store_even_with_sessions(self):
        # When SessionStore was constructed with no slab_pool,
        # session.slab stays None and kv_live_bytes() returns 0
        # by definition. PR-A3b makes the with-pool case return
        # real bytes (see TestSlabOwnership below).
        store = SessionStore(capacity=2)
        store.create_session()
        store.create_session()
        assert store.total_kv_live_bytes == 0


# ---------------------------------------------------------------------------
# Slab ownership (PR-A3b)
# ---------------------------------------------------------------------------


class TestSlabOwnership:
    def test_default_store_has_no_pool(self):
        store = SessionStore(capacity=2)
        assert store.slab_pool is None

    def test_store_with_pool_records_it(self):
        pool = _tiny_slab_pool()
        store = SessionStore(capacity=2, slab_pool=pool)
        assert store.slab_pool is pool

    def test_create_session_acquires_slab_when_pool_present(self):
        pool = _tiny_slab_pool(num_slabs=2)
        store = SessionStore(capacity=2, slab_pool=pool)
        sess = store.create_session()
        assert sess.slab is not None
        assert pool.in_use_count == 1
        assert pool.available_count == 1

    def test_create_session_pool_less_leaves_slab_none(self):
        store = SessionStore(capacity=2)
        sess = store.create_session()
        assert sess.slab is None

    def test_close_session_releases_slab_to_pool(self):
        pool = _tiny_slab_pool(num_slabs=2)
        store = SessionStore(capacity=2, slab_pool=pool)
        sess = store.create_session()
        store.close_session(sess.session_id)
        assert pool.in_use_count == 0
        assert pool.available_count == 2

    def test_lru_eviction_releases_slab_before_admitting(self):
        # capacity=1, num_slabs=1 — admitting a second session must
        # evict the first AND release its slab so the new session
        # can acquire one. Without the eviction-before-acquire
        # ordering, the pool would be exhausted at the moment of
        # acquire() and create_session would raise PoolExhausted.
        pool = _tiny_slab_pool(num_slabs=1)
        store = SessionStore(capacity=1, slab_pool=pool)
        a = store.create_session()
        b = store.create_session()  # must evict a, reuse a's slab
        assert b.slab is not None
        assert a.slab is None  # released
        assert pool.in_use_count == 1
        with pytest.raises(SessionNotFoundError):
            store.get_session(a.session_id)

    def test_evict_idle_releases_slabs(self):
        pool = _tiny_slab_pool(num_slabs=3)
        store = SessionStore(capacity=3, slab_pool=pool)
        a = store.create_session()
        b = store.create_session()
        a.last_active_at = 0.0
        b.last_active_at = 1000.0
        evicted = store.evict_idle(ttl_seconds=10.0, now=1001.0)
        assert evicted == [a]
        assert a.slab is None  # released
        assert pool.in_use_count == 1  # only b remains
        assert b.slab is not None

    def test_inv1_violation_releases_slab(self):
        pool = _tiny_slab_pool(num_slabs=2)
        store = SessionStore(
            capacity=2,
            cache_inspector=_SyntheticInspector(k_seq_length=5),
            slab_pool=pool,
        )
        sess = store.create_session()
        assert sess.slab is not None
        with pytest.raises(InvariantViolation):
            store.append_tokens(sess.session_id, [1])
        # Slab released back to pool even though session was failed.
        assert sess.slab is None
        assert pool.in_use_count == 0

    def test_inv2_violation_releases_slab(self):
        pool = _tiny_slab_pool(num_slabs=2)
        store = SessionStore(capacity=2, slab_pool=pool)
        sess = store.create_session()
        store.record_position_advance(sess.session_id, 10)
        assert sess.slab is not None
        with pytest.raises(InvariantViolation):
            store.record_position_advance(sess.session_id, 5)
        assert sess.slab is None
        assert pool.in_use_count == 0

    def test_kv_live_bytes_reads_through_slab(self):
        # KVSlab exposes a live_kv_bytes_override field that the
        # verifier wiring (PooledVerifier) currently uses to publish
        # real KV bytes. Session.kv_live_bytes() must read through
        # to the slab's live_kv_bytes property.
        pool = _tiny_slab_pool(num_slabs=2)
        store = SessionStore(capacity=2, slab_pool=pool)
        sess = store.create_session()
        sess.slab.live_kv_bytes_override = 12345
        assert sess.kv_live_bytes() == 12345
        assert store.total_kv_live_bytes == 12345

    def test_total_kv_live_bytes_aggregates_across_sessions(self):
        pool = _tiny_slab_pool(num_slabs=4)
        store = SessionStore(capacity=4, slab_pool=pool)
        a = store.create_session()
        b = store.create_session()
        a.slab.live_kv_bytes_override = 100
        b.slab.live_kv_bytes_override = 200
        assert store.total_kv_live_bytes == 300

    def test_pool_exhausted_at_create_when_capacity_exceeds_pool(self):
        # capacity=4 but num_slabs=1: after the first session, the
        # second create wants to admit (capacity allows) but the pool
        # is empty (no eviction triggered, since active_count <
        # capacity). PoolExhausted must propagate — silent fall-back
        # to a None slab would corrupt the §2.3 byte-exact contract.
        pool = _tiny_slab_pool(num_slabs=1)
        store = SessionStore(capacity=4, slab_pool=pool)
        store.create_session()
        with pytest.raises(PoolExhausted):
            store.create_session()


# ---------------------------------------------------------------------------
# Session dataclass (direct surface)
# ---------------------------------------------------------------------------


class TestSessionDataclass:
    def test_history_length_property(self):
        sess = Session(session_id="s", eos_token_ids=(), client_label="")
        sess.history_token_ids = [1, 2, 3]
        assert sess.history_length == 3

    def test_idle_seconds_grows(self):
        sess = Session(session_id="s", eos_token_ids=(), client_label="")
        before = sess.idle_seconds
        time.sleep(0.005)
        after = sess.idle_seconds
        assert after > before

    def test_kv_live_bytes_zero_when_no_slab(self):
        sess = Session(session_id="s", eos_token_ids=(), client_label="")
        assert sess.slab is None
        assert sess.kv_live_bytes() == 0


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class TestErrors:
    def test_session_not_found_records_id(self):
        err = SessionNotFoundError("sess-xyz")
        assert err.session_id == "sess-xyz"
        assert "sess-xyz" in str(err)

    def test_invariant_violation_records_fields(self):
        err = InvariantViolation(
            kind="1", session_id="sess-xyz", detail="too short",
        )
        assert err.kind == "1"
        assert err.session_id == "sess-xyz"
        assert err.detail == "too short"
        assert "INV-1" in str(err)
        assert "sess-xyz" in str(err)
        assert "too short" in str(err)

    def test_session_not_found_is_subclass_of_session_store_error(self):
        assert issubclass(SessionNotFoundError, SessionStoreError)

    def test_invariant_violation_is_subclass_of_session_store_error(self):
        assert issubclass(InvariantViolation, SessionStoreError)
