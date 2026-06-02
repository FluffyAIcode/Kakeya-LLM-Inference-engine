"""Integration tests for :mod:`inference_engine.session.generator`.

PR-N1 migration of the former Linux-side ``test_generator.py``,
replacing FakeVerifier-driven tests with real-Qwen3-driven ones.

Validation tests (``max_tokens < 1``, sampling-param rejection,
seed-acceptance, AppendTokens-must-precede-Generate) DO NOT require
verifier numerics — those reject before touching the verifier — and
remain on the Linux gate as ``test_generator_validation.py``.
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
    SessionStore,
    STOP_REASON_EOS,
    STOP_REASON_MAX_TOKENS,
    TokenEvent,
)


@pytest.fixture
def real_verifier(fresh_verifier_factory):
    return fresh_verifier_factory(sink=2, window=8)


def _setup(verifier, *, eos_token_ids=(), initial_tokens=(1, 2, 3)):
    """Build (store, generator, session) pre-loaded with a prefill."""
    store = SessionStore(capacity=2, cache_inspector=verifier)
    append_coord = AppendTokensCoordinator(store, verifier)
    gen_coord = GenerationCoordinator(store, verifier)
    sess = store.create_session(eos_token_ids=eos_token_ids)
    if initial_tokens:
        append_coord.append_tokens(sess.session_id, list(initial_tokens))
    return store, gen_coord, sess


# ---------------------------------------------------------------------------
# Greedy happy path
# ---------------------------------------------------------------------------


class TestGreedyHappyPath:
    def test_yields_token_then_done(self, real_verifier):
        _store, gen_coord, sess = _setup(real_verifier)
        events = list(gen_coord.generate(sess.session_id, max_tokens=1))
        # Single TokenEvent + single DoneEvent (no HistoryTruncated for
        # short prefill).
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(token_events) == 1
        assert len(done_events) == 1
        assert done_events[0].generated_token_count == 1

    def test_max_tokens_yields_n_tokens_then_done(self, real_verifier):
        _store, gen_coord, sess = _setup(real_verifier)
        events = list(gen_coord.generate(sess.session_id, max_tokens=4))
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        # Either reaches max_tokens (stop_reason=max_tokens, count=4) or
        # emits EOS earlier (token_count < 4). Both legal outcomes.
        assert len(done_events) == 1
        if done_events[0].stop_reason == STOP_REASON_MAX_TOKENS:
            assert len(token_events) == 4
            assert done_events[0].generated_token_count == 4

    def test_each_token_advances_position(self, real_verifier):
        _store, gen_coord, sess = _setup(real_verifier)
        pos_before = real_verifier.next_global_position
        events = list(gen_coord.generate(sess.session_id, max_tokens=3))
        n_tokens = sum(1 for e in events if isinstance(e, TokenEvent))
        assert real_verifier.next_global_position == pos_before + n_tokens


# ---------------------------------------------------------------------------
# EOS stops generation
# ---------------------------------------------------------------------------


class TestEos:
    def test_eos_stops_generation_and_reports_eos_stop_reason(
        self, real_verifier,
    ):
        # Pick whatever the real verifier emits as its first greedy
        # token, then declare THAT token as EOS for a fresh session
        # — the second session will stop on the very first emit.
        _, gen_coord_a, sess_a = _setup(real_verifier)
        first_emitted = next(
            e.token_id for e in gen_coord_a.generate(
                sess_a.session_id, max_tokens=1,
            )
            if isinstance(e, TokenEvent)
        )

        # Reset and run again with that token as EOS.
        real_verifier.reset()
        _, gen_coord_b, sess_b = _setup(
            real_verifier, eos_token_ids=(first_emitted,),
        )
        events = list(gen_coord_b.generate(sess_b.session_id, max_tokens=8))
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        # First emitted token == EOS → stop after exactly one token.
        assert len(token_events) == 1
        assert token_events[0].token_id == first_emitted
        assert done_events[0].stop_reason == STOP_REASON_EOS
        assert done_events[0].generated_token_count == 1


# ---------------------------------------------------------------------------
# HistoryTruncated emission
# ---------------------------------------------------------------------------


class TestHistoryTruncated:
    def test_no_truncated_event_when_cache_holds_full_history(
        self, real_verifier,
    ):
        _store, gen_coord, sess = _setup(real_verifier)
        events = list(gen_coord.generate(sess.session_id, max_tokens=1))
        truncated = [e for e in events if isinstance(e, HistoryTruncatedEvent)]
        assert truncated == []

    def test_truncated_event_when_cache_is_in_truncated_mode(
        self, fresh_verifier_factory,
    ):
        # Use a tight sink+window so a moderate prefill triggers trim.
        v = fresh_verifier_factory(sink=2, window=4)
        store = SessionStore(capacity=1, cache_inspector=v)
        AppendTokensCoordinator(store, v).append_tokens(
            (sess := store.create_session()).session_id,
            list(range(100, 120)),  # 20 tokens > 6 = sink+window
        )
        # History is 20, cached is 6 → drops 14.
        events = list(GenerationCoordinator(store, v).generate(
            sess.session_id, max_tokens=1,
        ))
        truncated = [
            e for e in events if isinstance(e, HistoryTruncatedEvent)
        ]
        assert len(truncated) == 1
        # Exact value: history_length - len(cached_token_sequence).
        assert truncated[0].dropped_token_count == (
            len(sess.history_token_ids) - len(sess.cached_token_sequence)
        )


# ---------------------------------------------------------------------------
# INV propagation through Generate
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_inv1_violation_propagates_through_generate(
        self, real_verifier,
    ):
        # Drive a clean AppendTokens, then corrupt session state
        # before Generate runs — the per-step INV-1 check fires.
        store = SessionStore(capacity=1, cache_inspector=real_verifier)
        AppendTokensCoordinator(store, real_verifier).append_tokens(
            (sess := store.create_session()).session_id,
            [1, 2, 3],
        )
        sess.cached_token_sequence = [99, 99, 99, 99, 99]  # corrupt
        gen_coord = GenerationCoordinator(store, real_verifier)
        with pytest.raises(InvariantViolation):
            list(gen_coord.generate(sess.session_id, max_tokens=4))


# ---------------------------------------------------------------------------
# kv_live_bytes wiring (PR-E1c) — run against real verifier
# ---------------------------------------------------------------------------


def _slab_pool():
    from inference_engine.memory.pool import SlabPool
    from inference_engine.memory.slab import SlabConfig
    cfg = SlabConfig(
        num_layers=1, num_heads=1, sink_size=1,
        window_size=2, head_dim=4, dtype=torch.float32,
    )
    return SlabPool(num_slabs=1, slab_config=cfg)


class TestGenerationSyncsSlabBytes:
    def test_max_tokens_path_writes_real_bytes_to_slab_override(
        self, real_verifier,
    ):
        pool = _slab_pool()
        store = SessionStore(
            capacity=1, cache_inspector=real_verifier, slab_pool=pool,
        )
        AppendTokensCoordinator(store, real_verifier).append_tokens(
            (sess := store.create_session()).session_id,
            [1, 2, 3],
        )
        before = sess.slab.live_kv_bytes_override
        events = list(GenerationCoordinator(store, real_verifier).generate(
            sess.session_id, max_tokens=4,
        ))
        assert any(isinstance(e, TokenEvent) for e in events)
        after = sess.slab.live_kv_bytes_override
        assert after is not None
        assert after >= before  # type: ignore[operator]
        assert after == real_verifier.kv_live_bytes(session=None)

    def test_eos_path_writes_real_bytes_to_slab_override(
        self, real_verifier,
    ):
        # Discover the first emitted token, then restart with that as EOS.
        pool = _slab_pool()
        store_a = SessionStore(
            capacity=1, cache_inspector=real_verifier, slab_pool=pool,
        )
        AppendTokensCoordinator(store_a, real_verifier).append_tokens(
            (sess_a := store_a.create_session()).session_id,
            [1, 2, 3],
        )
        first_emitted = next(
            e.token_id for e in GenerationCoordinator(
                store_a, real_verifier,
            ).generate(sess_a.session_id, max_tokens=1)
            if isinstance(e, TokenEvent)
        )

        real_verifier.reset()
        pool2 = _slab_pool()
        store_b = SessionStore(
            capacity=1, cache_inspector=real_verifier, slab_pool=pool2,
        )
        AppendTokensCoordinator(store_b, real_verifier).append_tokens(
            (sess_b := store_b.create_session(
                eos_token_ids=(first_emitted,),
            )).session_id,
            [1, 2, 3],
        )
        events = list(GenerationCoordinator(store_b, real_verifier).generate(
            sess_b.session_id, max_tokens=8,
        ))
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert done and done[0].stop_reason == STOP_REASON_EOS
        assert sess_b.slab.live_kv_bytes_override == (
            real_verifier.kv_live_bytes(session=None)
        )
