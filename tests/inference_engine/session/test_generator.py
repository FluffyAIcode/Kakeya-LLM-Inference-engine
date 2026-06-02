"""Unit tests for :mod:`inference_engine.session.generator` (PR-B3).

Coverage target: 100% on ``inference_engine/session/generator.py``.

Test strategy mirrors :mod:`tests.inference_engine.session.test_coordinator`:
the dispatch + state-mirroring + error-mapping logic is tested with
the deterministic :class:`FakeVerifier` (Linux-runnable, no model
weights). Real Qwen3 verifier integration lives under
:mod:`tests.core` (Mac-only) and runs on the §9 Mac M4 gate.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.session import (
    AppendTokensCoordinator,
    DoneEvent,
    GenerationCoordinator,
    HistoryTruncatedEvent,
    InvariantViolation,
    SessionNotFoundError,
    SessionStore,
    STOP_REASON_EOS,
    STOP_REASON_MAX_TOKENS,
    TokenEvent,
)

# Reuse the FakeVerifier from PR-B2's test module rather than
# re-defining it. It already mirrors the real verifier's mutation
# contract (sink+window trim in commit_or_truncate, parallel-sequence
# growth in forward_block, deterministic logits).
from tests.inference_engine.session.test_coordinator import FakeVerifier


def _build(
    *,
    sink_size: int = 2,
    window_size: int = 4,
    eos_token_ids=(),
    initial_tokens=(1, 2, 3),
):
    """Construct (store, fv, gen_coord, session) ready for Generate.

    Runs an AppendTokens via the PR-B2 coordinator first so the
    session has prefilled state — Generate against an empty session
    is a documented ValueError, tested separately.
    """
    fv = FakeVerifier(
        sink_size=sink_size, window_size=window_size, vocab_size=16,
    )
    store = SessionStore(capacity=2, cache_inspector=fv)
    append_coord = AppendTokensCoordinator(store, fv)
    gen_coord = GenerationCoordinator(store, fv)
    sess = store.create_session(eos_token_ids=eos_token_ids)
    if initial_tokens:
        append_coord.append_tokens(sess.session_id, list(initial_tokens))
    return store, fv, gen_coord, sess


# ---------------------------------------------------------------------------
# Greedy dispatch + happy path
# ---------------------------------------------------------------------------


class TestGreedyHappyPath:
    def test_yields_token_then_done(self):
        store, fv, coord, sess = _build()
        events = list(coord.generate(sess.session_id, max_tokens=1))
        assert len(events) == 2
        assert isinstance(events[0], TokenEvent)
        assert isinstance(events[1], DoneEvent)
        assert events[1].stop_reason == STOP_REASON_MAX_TOKENS
        assert events[1].generated_token_count == 1

    def test_max_tokens_caps_token_emission(self):
        store, fv, coord, sess = _build()
        events = list(coord.generate(sess.session_id, max_tokens=3))
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(token_events) == 3
        assert len(done_events) == 1
        assert done_events[0].stop_reason == STOP_REASON_MAX_TOKENS
        assert done_events[0].generated_token_count == 3

    def test_done_is_terminal_and_unique(self):
        store, fv, coord, sess = _build()
        events = list(coord.generate(sess.session_id, max_tokens=2))
        # Done event is exactly one and is the last.
        done_indices = [
            i for i, e in enumerate(events) if isinstance(e, DoneEvent)
        ]
        assert len(done_indices) == 1
        assert done_indices[0] == len(events) - 1

    def test_done_includes_total_seconds(self):
        store, fv, coord, sess = _build()
        events = list(coord.generate(sess.session_id, max_tokens=1))
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.total_seconds >= 0.0
        # PR-B3 has no separate prefill phase.
        assert done.prefill_seconds == 0.0


class TestGreedyAdvancesVerifier:
    def test_each_token_advances_position_by_one(self):
        store, fv, coord, sess = _build()
        pos_before = fv.next_global_position
        list(coord.generate(sess.session_id, max_tokens=4))
        assert fv.next_global_position == pos_before + 4

    def test_session_history_grows(self):
        store, fv, coord, sess = _build()
        list(coord.generate(sess.session_id, max_tokens=3))
        assert len(sess.history_token_ids) == 3 + 3  # initial + generated

    def test_session_cached_token_sequence_mirrors_verifier(self):
        store, fv, coord, sess = _build(sink_size=2, window_size=4)
        list(coord.generate(sess.session_id, max_tokens=10))
        assert sess.cached_token_sequence == fv.cached_token_sequence

    def test_each_token_calls_forward_then_commit(self):
        store, fv, coord, sess = _build()
        fv.call_log.clear()
        list(coord.generate(sess.session_id, max_tokens=2))
        # 2 (forward_block, commit_or_truncate) pairs.
        kinds = [c[0] for c in fv.call_log]
        assert kinds == [
            "forward_block", "commit_or_truncate",
            "forward_block", "commit_or_truncate",
        ]


# ---------------------------------------------------------------------------
# EOS handling
# ---------------------------------------------------------------------------


class TestEos:
    def test_eos_token_terminates_with_eos_stop_reason(self):
        # FakeVerifier._logits_for hashes recent tokens to an argmax.
        # We pre-load history that makes the next argmax a known
        # token, then put that token in eos_token_ids.
        # The FakeVerifier formula: argmax = sum(history[-3:]) % 16.
        # With initial=[1, 2, 3], next argmax = 6.
        store, fv, coord, sess = _build(
            initial_tokens=(1, 2, 3), eos_token_ids=(6,),
        )
        events = list(coord.generate(sess.session_id, max_tokens=10))
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        # Exactly one TokenEvent, then Done with EOS.
        assert len(token_events) == 1
        assert token_events[0].token_id == 6
        assert done_events[0].stop_reason == STOP_REASON_EOS
        assert done_events[0].generated_token_count == 1

    def test_no_eos_runs_to_max_tokens(self):
        # Use a token id that cannot be produced (vocab size 16; eos
        # set to 99 cannot match any argmax).
        store, fv, coord, sess = _build(eos_token_ids=(99,))
        events = list(coord.generate(sess.session_id, max_tokens=4))
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert done_events[0].stop_reason == STOP_REASON_MAX_TOKENS

    def test_empty_eos_set_runs_to_max_tokens(self):
        store, fv, coord, sess = _build(eos_token_ids=())
        events = list(coord.generate(sess.session_id, max_tokens=2))
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.stop_reason == STOP_REASON_MAX_TOKENS


# ---------------------------------------------------------------------------
# HistoryTruncated event
# ---------------------------------------------------------------------------


class TestHistoryTruncated:
    def test_emitted_at_start_when_already_truncated(self):
        # sink+window = 2+4 = 6 capacity. Append 8 tokens → cache
        # holds 6, history holds 8 → already truncated state.
        store, fv, coord, sess = _build(
            sink_size=2, window_size=4,
            initial_tokens=(10, 20, 30, 40, 50, 60, 70, 80),
        )
        events = list(coord.generate(sess.session_id, max_tokens=1))
        # First non-token event should be HistoryTruncated, before
        # any TokenEvent.
        assert isinstance(events[0], HistoryTruncatedEvent)
        assert events[0].dropped_token_count == 8 - 6  # 2 dropped
        # A TokenEvent must follow before Done.
        assert isinstance(events[1], TokenEvent)

    def test_not_emitted_when_under_capacity(self):
        # sink+window = 6; initial = 3 tokens; cache == history.
        store, fv, coord, sess = _build(
            sink_size=2, window_size=4, initial_tokens=(1, 2, 3),
        )
        events = list(coord.generate(sess.session_id, max_tokens=2))
        # No HistoryTruncated event present.
        assert not any(
            isinstance(e, HistoryTruncatedEvent) for e in events
        )

    def test_at_most_one_per_call(self):
        # Even after generation pushes well past the boundary, only
        # one HistoryTruncated per Generate call (per the proto
        # contract: "Emitted at most once per Generate call").
        store, fv, coord, sess = _build(
            sink_size=2, window_size=4,
            initial_tokens=(10, 20, 30, 40, 50, 60, 70, 80),
        )
        events = list(coord.generate(sess.session_id, max_tokens=10))
        truncated_events = [
            e for e in events if isinstance(e, HistoryTruncatedEvent)
        ]
        assert len(truncated_events) == 1


# ---------------------------------------------------------------------------
# Validation: max_tokens, sampling params, no AppendTokens prior
# ---------------------------------------------------------------------------


class TestValidation:
    def test_max_tokens_zero_rejected(self):
        store, fv, coord, sess = _build()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            list(coord.generate(sess.session_id, max_tokens=0))

    def test_max_tokens_negative_rejected(self):
        store, fv, coord, sess = _build()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            list(coord.generate(sess.session_id, max_tokens=-3))

    def test_temperature_nonzero_rejected(self):
        store, fv, coord, sess = _build()
        with pytest.raises(ValueError, match="greedy"):
            list(coord.generate(
                sess.session_id, max_tokens=1, temperature=0.5,
            ))

    def test_temperature_zero_accepted(self):
        store, fv, coord, sess = _build()
        # Temperature=0 is greedy's no-op default; accept silently.
        events = list(coord.generate(
            sess.session_id, max_tokens=1, temperature=0.0,
        ))
        assert any(isinstance(e, TokenEvent) for e in events)

    def test_top_p_set_rejected(self):
        store, fv, coord, sess = _build()
        with pytest.raises(ValueError, match="top_p"):
            list(coord.generate(
                sess.session_id, max_tokens=1, top_p=0.9,
            ))

    def test_top_k_other_than_one_rejected(self):
        store, fv, coord, sess = _build()
        with pytest.raises(ValueError, match="top_k"):
            list(coord.generate(
                sess.session_id, max_tokens=1, top_k=50,
            ))

    def test_top_k_one_accepted(self):
        store, fv, coord, sess = _build()
        events = list(coord.generate(
            sess.session_id, max_tokens=1, top_k=1,
        ))
        assert any(isinstance(e, TokenEvent) for e in events)

    def test_seed_accepted_and_ignored_in_greedy(self):
        store, fv, coord, sess = _build()
        # Seed shouldn't affect greedy output. Two runs with
        # different seeds must produce identical token streams.
        store_a, fv_a, coord_a, sess_a = _build()
        store_b, fv_b, coord_b, sess_b = _build()
        events_a = [
            e for e in coord_a.generate(
                sess_a.session_id, max_tokens=4, seed=1,
            )
            if isinstance(e, TokenEvent)
        ]
        events_b = [
            e for e in coord_b.generate(
                sess_b.session_id, max_tokens=4, seed=999,
            )
            if isinstance(e, TokenEvent)
        ]
        assert [e.token_id for e in events_a] == [
            e.token_id for e in events_b
        ]

    def test_no_appendtokens_first_rejected(self):
        # Session created but no AppendTokens called — no prefill,
        # so next_token_logits is meaningless. Reject loudly.
        fv = FakeVerifier()
        store = SessionStore(capacity=1, cache_inspector=fv)
        coord = GenerationCoordinator(store, fv)
        sess = store.create_session()
        with pytest.raises(ValueError, match="AppendTokens must precede"):
            list(coord.generate(sess.session_id, max_tokens=1))

    def test_unknown_session_raises_session_not_found(self):
        store, fv, coord, _ = _build()
        with pytest.raises(SessionNotFoundError):
            list(coord.generate("sess-unknown", max_tokens=1))


# ---------------------------------------------------------------------------
# INV-1 / INV-2 propagation through Generate
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_inv1_violation_propagates(self):
        # Drive AppendTokens with an honest inspector, then patch the
        # inspector to lie just before Generate. The lying inspector
        # makes the first generation step's INV-1 check fail because
        # session.cached_token_sequence (mirrored from verifier) won't
        # match the lie's reported k_seq_length.
        fv = FakeVerifier()
        store = SessionStore(capacity=1, cache_inspector=fv)
        append_coord = AppendTokensCoordinator(store, fv)
        gen_coord = GenerationCoordinator(store, fv)
        sess = store.create_session()
        # AppendTokens with the honest FakeVerifier — works.
        append_coord.append_tokens(sess.session_id, [1, 2, 3])
        # Now monkey-patch the inspector to lie. Note: SessionStore's
        # _assert_inv1 calls self._cache_inspector.k_seq_length(session),
        # which dispatches to the patched bound method.
        fv.k_seq_length = lambda session: 999  # type: ignore[assignment]
        with pytest.raises(InvariantViolation) as exc:
            list(gen_coord.generate(sess.session_id, max_tokens=1))
        assert exc.value.kind == "1"
        with pytest.raises(SessionNotFoundError):
            store.get_session(sess.session_id)

    def test_inv2_violation_propagates(self):
        # AppendTokens uses verifier.prefill, NOT commit_or_truncate,
        # so the FIRST commit_or_truncate the Verifier sees is from
        # the first generation step. Trip the regress on call #1.
        class _RegressingVerifier(FakeVerifier):
            def __init__(self):
                super().__init__()
                self._calls = 0

            def commit_or_truncate(self, *, forwarded, accepted):
                super().commit_or_truncate(
                    forwarded=forwarded, accepted=accepted,
                )
                self._calls += 1
                if self._calls == 1:  # first generation step's commit
                    self.next_global_position = 0  # regress

        fv = _RegressingVerifier()
        store = SessionStore(capacity=1, cache_inspector=fv)
        append_coord = AppendTokensCoordinator(store, fv)
        gen_coord = GenerationCoordinator(store, fv)
        sess = store.create_session()
        append_coord.append_tokens(sess.session_id, [1, 2, 3])
        with pytest.raises(InvariantViolation) as exc:
            list(gen_coord.generate(sess.session_id, max_tokens=1))
        assert exc.value.kind == "2"


# ---------------------------------------------------------------------------
# Determinism (INV-3 byte-exact under greedy)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_runs_with_same_history_produce_same_tokens(self):
        # INV-3 byte-exact at the GenerationCoordinator level: two
        # parallel sessions with identical history produce identical
        # token sequences under greedy decoding.
        store_a, fv_a, coord_a, sess_a = _build(
            initial_tokens=(7, 11, 13, 17, 19),
        )
        store_b, fv_b, coord_b, sess_b = _build(
            initial_tokens=(7, 11, 13, 17, 19),
        )
        tokens_a = [
            e.token_id
            for e in coord_a.generate(sess_a.session_id, max_tokens=8)
            if isinstance(e, TokenEvent)
        ]
        tokens_b = [
            e.token_id
            for e in coord_b.generate(sess_b.session_id, max_tokens=8)
            if isinstance(e, TokenEvent)
        ]
        assert tokens_a == tokens_b


# ---------------------------------------------------------------------------
# Constructor / event types
# ---------------------------------------------------------------------------


class TestConstructorAndEventDataclasses:
    def test_constructor_stores_references(self):
        fv = FakeVerifier()
        store = SessionStore(capacity=1, cache_inspector=fv)
        coord = GenerationCoordinator(store, fv)
        # Coordinator accepts the references; we verify by exercising.
        sess = store.create_session()
        AppendTokensCoordinator(store, fv).append_tokens(
            sess.session_id, [1],
        )
        events = list(coord.generate(sess.session_id, max_tokens=1))
        assert any(isinstance(e, TokenEvent) for e in events)

    def test_token_event_is_frozen(self):
        e = TokenEvent(token_id=5)
        with pytest.raises(Exception):  # FrozenInstanceError
            e.token_id = 6  # type: ignore[misc]

    def test_history_truncated_event_is_frozen(self):
        e = HistoryTruncatedEvent(dropped_token_count=3)
        with pytest.raises(Exception):
            e.dropped_token_count = 4  # type: ignore[misc]

    def test_done_event_is_frozen(self):
        e = DoneEvent(
            stop_reason=STOP_REASON_MAX_TOKENS,
            generated_token_count=1,
            prefill_seconds=0.0, total_seconds=0.0,
        )
        with pytest.raises(Exception):
            e.generated_token_count = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PR-E1c — Generation also syncs the slab byte count after each call.
# Mirrors test_coordinator.py::TestSlabBytesSync for the AppendTokens path.
# ---------------------------------------------------------------------------


def _slab_pool_for_test():
    import torch as _torch
    from inference_engine.memory.pool import SlabPool
    from inference_engine.memory.slab import SlabConfig
    cfg = SlabConfig(
        num_layers=1, num_heads=1, sink_size=1,
        window_size=2, head_dim=4, dtype=_torch.float32,
    )
    return SlabPool(num_slabs=1, slab_config=cfg)


class TestGenerationSyncsSlabBytes:
    def test_max_tokens_path_syncs_slab_bytes(self):
        pool = _slab_pool_for_test()
        fv = FakeVerifier()
        store = SessionStore(
            capacity=1, cache_inspector=fv, slab_pool=pool,
        )
        AppendTokensCoordinator(store, fv).append_tokens(
            store._sessions[next(iter(store._sessions))].session_id,
            [1, 2, 3],
        ) if False else None  # noqa: E501 - placeholder; real dispatch below

        sess = list(store._sessions.values())[0] if store._sessions else None
        if sess is None:
            sess = store.create_session()
        # Drive append → generate → expect override updated.
        AppendTokensCoordinator(store, fv).append_tokens(
            sess.session_id, [1, 2, 3],
        )
        before = sess.slab.live_kv_bytes_override
        gen = GenerationCoordinator(store, fv)
        events = list(gen.generate(sess.session_id, max_tokens=4))
        assert any(isinstance(e, TokenEvent) for e in events)
        # After generate, override has been re-synced (k_seq grew).
        after = sess.slab.live_kv_bytes_override
        assert after is not None and after >= before  # type: ignore[operator]
        # Concrete value: equals current k_seq * per-token bytes.
        assert after == (
            len(fv.cached_token_sequence) * FakeVerifier.BYTES_PER_KV_TOKEN
        )

    def test_eos_path_syncs_slab_bytes(self):
        pool = _slab_pool_for_test()
        fv = FakeVerifier()
        # FakeVerifier's vocab_size is 16; pick an EOS within range.
        eos_id = 7
        store = SessionStore(
            capacity=1, cache_inspector=fv, slab_pool=pool,
        )
        sess = store.create_session(eos_token_ids=(eos_id,))
        AppendTokensCoordinator(store, fv).append_tokens(
            sess.session_id, [1, 2, 3],
        )
        # Force the FakeVerifier to emit eos_id as the next token.
        fv.next_token_logits = torch.zeros_like(fv.next_token_logits)
        fv.next_token_logits[eos_id] = 1.0
        gen = GenerationCoordinator(store, fv)
        events = list(gen.generate(sess.session_id, max_tokens=4))
        # EOS path was hit.
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert done and done[0].stop_reason == "eos"
        # Slab override is set to the verifier's reported live bytes.
        assert sess.slab.live_kv_bytes_override == (
            len(fv.cached_token_sequence) * FakeVerifier.BYTES_PER_KV_TOKEN
        )
