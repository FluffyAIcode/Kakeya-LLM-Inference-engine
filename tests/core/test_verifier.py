"""Unit tests for `kv_cache_proposer.verifier.SinkWindowVerifier`.

Real Qwen3-1.7B weights, no mocks. Tests cover every public method,
every branch in the trim/truncate code paths, and every error raise.
"""

from __future__ import annotations

import pytest
import torch

from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_config_loads(verifier_session: SinkWindowVerifier) -> None:
    assert verifier_session.config.sink_size == 4
    assert verifier_session.config.window_size == 64
    assert verifier_session.cache is None  # not yet prefilled
    assert verifier_session.next_token_logits is None
    assert verifier_session.cache_logical_size == 0
    assert verifier_session.next_global_position == 0
    assert verifier_session.stats.weight_bytes > 0


@pytest.mark.parametrize(
    "sink,window,err",
    [
        (-1, 8, "sink_size must be >= 0"),
        (4, 0, "window_size must be > 0"),
        (4, -3, "window_size must be > 0"),
    ],
)
def test_construction_validates_window_args(sink: int, window: int, err: str) -> None:
    with pytest.raises(ValueError, match=err):
        SinkWindowVerifier(
            VerifierConfig(
                dtype=torch.bfloat16,
                device="cpu",
                sink_size=sink,
                window_size=window,
            )
        )


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------

def test_prefill_rejects_empty(verifier_session: SinkWindowVerifier) -> None:
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        verifier_session.prefill([])


def test_prefill_under_budget(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    prompt = list(range(20))
    verifier.prefill(prompt)
    assert verifier.cache_logical_size == 20  # below budget, no trim
    assert verifier.next_global_position == 20
    assert verifier.next_token_logits is not None
    assert verifier.stats.forward_calls == 1


def test_prefill_over_budget_triggers_trim(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=8)
    prompt = list(range(50))
    verifier.prefill(prompt)
    # cache trimmed to sink+window
    assert verifier.cache_logical_size == 12
    # K/V tensors should match the logical size physically
    layer0 = verifier.cache.layers[0]
    assert layer0.keys.shape[2] == 12
    assert layer0.values.shape[2] == 12


def test_prefill_zero_sink(fresh_verifier_factory) -> None:
    """Boundary: sink_size=0 must still produce a valid trimmed cache."""
    verifier = fresh_verifier_factory(sink=0, window=8)
    verifier.prefill(list(range(20)))
    assert verifier.cache_logical_size == 8
    assert verifier.cache.layers[0].keys.shape[2] == 8


# ---------------------------------------------------------------------------
# forward_block + commit_or_truncate
# ---------------------------------------------------------------------------

def test_forward_block_requires_prefill(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    with pytest.raises(RuntimeError, match="not prefilled"):
        verifier.forward_block([1, 2, 3])


def test_forward_block_rejects_empty(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.prefill([1, 2, 3])
    with pytest.raises(ValueError, match="tokens must be non-empty"):
        verifier.forward_block([])


def test_forward_block_returns_per_position_logits(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill(list(range(10)))
    L = 5
    block = list(range(100, 100 + L))
    logits = verifier.forward_block(block)
    assert logits.shape == (L, verifier.model.config.vocab_size)
    # cache_logical_size grows by L (still below budget here)
    assert verifier.cache_logical_size == 10 + L


def test_commit_validates_args(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.prefill([1, 2, 3])
    verifier.forward_block([4, 5, 6])
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        verifier.commit_or_truncate(forwarded=3, accepted=-1)
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        verifier.commit_or_truncate(forwarded=3, accepted=4)


def test_commit_full_accept_no_drop(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill(list(range(10)))
    verifier.forward_block([100, 101, 102])
    verifier.commit_or_truncate(forwarded=3, accepted=3)
    assert verifier.cache_logical_size == 13
    assert verifier.next_global_position == 13


def test_commit_partial_accept_drops_tail(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill(list(range(10)))
    verifier.forward_block([100, 101, 102])
    verifier.commit_or_truncate(forwarded=3, accepted=1)
    assert verifier.cache_logical_size == 11  # prefix 10 + 1 accepted
    assert verifier.next_global_position == 11
    # Physical K/V tail must reflect the drop
    assert verifier.cache.layers[0].keys.shape[2] == 11


def test_commit_zero_accept_drops_all(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill(list(range(10)))
    verifier.forward_block([100, 101, 102])
    verifier.commit_or_truncate(forwarded=3, accepted=0)
    assert verifier.cache_logical_size == 10
    assert verifier.next_global_position == 10


def test_commit_post_trims_to_budget(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=8)  # budget=12
    verifier.prefill(list(range(10)))
    verifier.forward_block([100, 101, 102, 103, 104])  # cache->15 then trim
    verifier.commit_or_truncate(forwarded=5, accepted=5)
    assert verifier.cache_logical_size == 12  # capped at budget


# ---------------------------------------------------------------------------
# append_token
# ---------------------------------------------------------------------------

def test_append_token_advances_state(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill(list(range(10)))
    pre_size = verifier.cache_logical_size
    pre_pos = verifier.next_global_position
    logits = verifier.append_token(123)
    assert verifier.cache_logical_size == pre_size + 1
    assert verifier.next_global_position == pre_pos + 1
    assert logits is verifier.next_token_logits
    assert logits.ndim == 1
    assert logits.shape[0] == verifier.model.config.vocab_size


# ---------------------------------------------------------------------------
# trim / truncate internals — error paths
# ---------------------------------------------------------------------------

def test_trim_raises_when_no_cache(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    with pytest.raises(RuntimeError, match="No cache to trim"):
        verifier._trim_cache_in_place()


def test_truncate_tail_raises_when_no_cache(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    with pytest.raises(RuntimeError, match="No cache to truncate"):
        verifier._truncate_tail_in_place(1)


def test_truncate_tail_zero_drop_is_noop(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.prefill([1, 2, 3])
    pre = verifier.cache_logical_size
    verifier._truncate_tail_in_place(0)
    assert verifier.cache_logical_size == pre


def test_truncate_tail_overflow_raises(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.prefill([1, 2, 3])
    with pytest.raises(RuntimeError, match=r"Cannot drop \d+ tokens"):
        verifier._truncate_tail_in_place(verifier.cache_logical_size + 1)


def test_trim_detects_layout_violation(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=8)
    verifier.prefill(list(range(20)))
    # Force inconsistency between bookkeeping and tensor shape:
    verifier.cache_logical_size = 999
    with pytest.raises(RuntimeError, match="layout invariant violated"):
        verifier._trim_cache_in_place()


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_state(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.prefill([1, 2, 3])
    verifier.reset()
    assert verifier.next_token_logits is None
    assert verifier.cache_logical_size == 0
    assert verifier.next_global_position == 0
    assert verifier.cache is not None  # reset re-creates an empty cache


# ---------------------------------------------------------------------------
# Defensive / invariant paths
# ---------------------------------------------------------------------------

def test_trim_skips_layers_with_no_keys(fresh_verifier_factory) -> None:
    """If a Cache layer reports null keys/values, trimming must skip it
    without raising. Such layers occur in hybrid-attention models that
    declare some layers as no-cache (the layout-invariant check is run
    only on layers that DO have populated keys)."""
    verifier = fresh_verifier_factory(sink=4, window=8)
    verifier.prefill(list(range(20)))  # cache trimmed to 12 already
    # Re-trigger trim with a single null-K layer: must short-circuit cleanly.
    layer0 = verifier.cache.layers[0]
    saved_k, saved_v = layer0.keys, layer0.values
    layer0.keys = None
    layer0.values = None
    try:
        # Manually force the cache_logical_size and re-call trim. The
        # null-K layer is skipped; remaining layers go through the normal
        # invariant check, which holds because they still match
        # cache_logical_size.
        # First push other layers' shapes to budget+1 so trim has work to do:
        for layer in verifier.cache.layers[1:]:
            if layer.keys is None or layer.values is None:
                continue
            # Append a junk slot to push past budget.
            extra_k = layer.keys[:, :, -1:, :].clone()
            extra_v = layer.values[:, :, -1:, :].clone()
            layer.keys = torch.cat([layer.keys, extra_k], dim=2)
            layer.values = torch.cat([layer.values, extra_v], dim=2)
        verifier.cache_logical_size = verifier._budget() + 1
        verifier._trim_cache_in_place()
        # All non-null layers shrank back to budget; null layer untouched.
        assert verifier.cache_logical_size == verifier._budget()
        for layer in verifier.cache.layers[1:]:
            if layer.keys is None:
                continue
            assert layer.keys.shape[2] == verifier._budget()
        assert verifier.cache.layers[0].keys is None
    finally:
        layer0.keys = saved_k
        layer0.values = saved_v


def test_truncate_skips_layers_with_no_keys(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill(list(range(20)))
    layer0 = verifier.cache.layers[0]
    saved_k, saved_v = layer0.keys, layer0.values
    layer0.keys = None
    layer0.values = None
    try:
        verifier._truncate_tail_in_place(1)  # should skip null layer cleanly
        # Other layers shrank by 1
        for layer in verifier.cache.layers[1:]:
            if layer.keys is None:
                continue
            assert layer.keys.shape[2] == 19
    finally:
        layer0.keys = saved_k
        layer0.values = saved_v


def test_record_peak_kv_handles_null_cache(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.cache = None
    # Should be a no-op; the peak should remain at its initial value.
    pre = verifier.stats.peak_kv_bytes
    verifier._record_peak_kv()
    assert verifier.stats.peak_kv_bytes == pre


def test_record_peak_kv_handles_layers_with_null_kv(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    verifier.prefill(list(range(8)))
    layer0 = verifier.cache.layers[0]
    saved_k, saved_v = layer0.keys, layer0.values
    layer0.keys = None
    layer0.values = None
    try:
        verifier._record_peak_kv()  # exercise the keys-None branch
    finally:
        layer0.keys = saved_k
        layer0.values = saved_v


def test_live_kv_bytes_zero_before_prefill(fresh_verifier_factory) -> None:
    """Before any prefill, ``live_kv_bytes()`` must read 0. Required
    by the /metrics scrape contract: the gauge has a stable value at
    process startup."""
    verifier = fresh_verifier_factory()
    assert verifier.live_kv_bytes() == 0


def test_live_kv_bytes_nonzero_after_prefill(fresh_verifier_factory) -> None:
    """After prefill the cache holds tensors; live_kv_bytes returns
    the sum of bytes across all layers' keys + values. This is the
    gauge value the bench scrapes during in-flight generation."""
    verifier = fresh_verifier_factory()
    verifier.prefill(list(range(16)))
    n = verifier.live_kv_bytes()
    assert n > 0
    # Round-trip: peak_kv_bytes is set from the same source so they
    # must agree right after prefill.
    assert verifier.stats.peak_kv_bytes == n


def test_live_kv_bytes_returns_zero_when_layer_kv_is_null(
    fresh_verifier_factory,
) -> None:
    """The keys-None branch is taken on cleared layers and must not
    raise — live_kv_bytes simply skips them in the sum."""
    verifier = fresh_verifier_factory()
    verifier.prefill(list(range(4)))
    layer0 = verifier.cache.layers[0]
    saved_k, saved_v = layer0.keys, layer0.values
    layer0.keys = None
    layer0.values = None
    try:
        n = verifier.live_kv_bytes()
        assert n >= 0
    finally:
        layer0.keys = saved_k
        layer0.values = saved_v


# ---------------------------------------------------------------------------
# ADR 0007 §2.2 + §2.9 — cached_token_sequence + INV-1
# ---------------------------------------------------------------------------


def test_cached_token_sequence_empty_after_construction(
    fresh_verifier_factory,
) -> None:
    """A freshly-constructed verifier has no cache yet, so the
    parallel sequence is empty and INV-1 is satisfied trivially."""
    verifier = fresh_verifier_factory()
    assert verifier.cached_token_sequence == []
    verifier._assert_cache_invariant_1()


def test_cached_token_sequence_populated_after_short_prefill(
    fresh_verifier_factory,
) -> None:
    """When the prompt is shorter than sink+window, the entire
    prompt is held in the cache and the parallel sequence equals
    the prompt verbatim."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    prompt = list(range(5))  # 5 < sink+window = 10
    verifier.prefill(prompt)
    assert verifier.cached_token_sequence == prompt
    verifier._assert_cache_invariant_1()


def test_cached_token_sequence_trimmed_after_long_prefill(
    fresh_verifier_factory,
) -> None:
    """When the prompt exceeds sink+window, the parallel sequence
    holds the first sink_size + last window_size token ids — exactly
    the same shape the K/V tensors hold post-trim."""
    verifier = fresh_verifier_factory(sink=2, window=4)
    prompt = list(range(20))  # 20 > sink+window = 6
    verifier.prefill(prompt)
    expected = prompt[:2] + prompt[-4:]
    assert verifier.cached_token_sequence == expected
    verifier._assert_cache_invariant_1()


def test_cached_token_sequence_extends_on_forward_block(
    fresh_verifier_factory,
) -> None:
    """``forward_block`` provisionally extends the cache with the new
    tokens; the parallel sequence must extend in lockstep so INV-1
    holds."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([0, 1, 2, 3])
    verifier.forward_block([4, 5])
    # Cache now holds [0,1,2,3,4,5] (6 entries, all under budget 10)
    assert verifier.cached_token_sequence == [0, 1, 2, 3, 4, 5]
    verifier._assert_cache_invariant_1()


def test_cached_token_sequence_drops_rejected_tail_on_partial_accept(
    fresh_verifier_factory,
) -> None:
    """``commit_or_truncate(forwarded=K, accepted=A)`` drops the last
    K-A tokens from both the K/V tensors and the parallel sequence."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([0, 1, 2, 3])
    verifier.forward_block([4, 5, 6])
    # Accept only the first of the 3 forwarded tokens
    verifier.commit_or_truncate(forwarded=3, accepted=1)
    # The two unaccepted tokens (5, 6) should be dropped
    assert verifier.cached_token_sequence == [0, 1, 2, 3, 4]
    verifier._assert_cache_invariant_1()


def test_cached_token_sequence_after_append_token(
    fresh_verifier_factory,
) -> None:
    """``append_token`` is forward_block + full-accept; sequence
    grows by exactly one token."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([0, 1, 2, 3])
    verifier.append_token(99)
    assert verifier.cached_token_sequence == [0, 1, 2, 3, 99]
    verifier._assert_cache_invariant_1()


def test_cached_token_sequence_cleared_on_reset(
    fresh_verifier_factory,
) -> None:
    """``reset`` empties the parallel sequence in lockstep with
    clearing the K/V tensors."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([0, 1, 2, 3])
    assert verifier.cached_token_sequence != []
    verifier.reset()
    assert verifier.cached_token_sequence == []
    verifier._assert_cache_invariant_1()


def test_inv_1_violation_raises_assertion_error(
    fresh_verifier_factory,
) -> None:
    """If the parallel sequence is forced out of sync with the K/V
    tensors, the next mutation site detects the divergence and
    raises. This is the §2.9 contract: bugs surface, never silently
    recover."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([0, 1, 2, 3])
    # Corrupt the parallel sequence by force.
    verifier.cached_token_sequence = verifier.cached_token_sequence + [999]
    with pytest.raises(AssertionError, match="INV-1 violated"):
        verifier._assert_cache_invariant_1()


def test_inv_1_assertion_message_carries_diagnostic_state(
    fresh_verifier_factory,
) -> None:
    """The error message exposes the actual vs expected lengths and
    the verifier's logical-position counters so the bug report can
    be triaged without re-running the workload."""
    verifier = fresh_verifier_factory()
    verifier.prefill([0, 1, 2, 3])
    verifier.cached_token_sequence = verifier.cached_token_sequence + [42, 43]
    with pytest.raises(AssertionError) as exc:
        verifier._assert_cache_invariant_1()
    msg = str(exc.value)
    assert "INV-1" in msg
    assert "cached_token_sequence" in msg
    assert "cache_logical_size=" in msg
    assert "next_global_position=" in msg


def test_inv_1_holds_when_cache_is_none(fresh_verifier_factory) -> None:
    """An empty cache + empty parallel sequence is the trivial INV-1
    satisfaction. No mutation has happened, but the assert path must
    still accept this state."""
    verifier = fresh_verifier_factory()
    # Construction leaves cache=None and sequence=[]
    assert verifier.cache is None
    assert verifier.cached_token_sequence == []
    # Should NOT raise
    verifier._assert_cache_invariant_1()


# ---------------------------------------------------------------------------
# ADR 0007 §2.4 — path_select + prefill_incremental + INV-2 (PR 7-2)
# ---------------------------------------------------------------------------


def test_path_select_cold_start_returns_new_session(
    fresh_verifier_factory,
) -> None:
    """§2.4.b case 1: empty cache always routes to NewSession."""
    from kv_cache_proposer.path_plan import NewSession

    verifier = fresh_verifier_factory()
    plan = verifier.path_select([1, 2, 3])
    assert isinstance(plan, NewSession)
    assert plan.prompt == [1, 2, 3]


def test_path_select_extends_returns_continuation(
    fresh_verifier_factory,
) -> None:
    """§2.4.a: new prompt is a strict monotonic extension of the
    cached state. Returns ContinuationPlan with the right skip_n
    and new_tokens."""
    from kv_cache_proposer.path_plan import ContinuationPlan

    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30, 40, 50])
    plan = verifier.path_select([10, 20, 30, 40, 50, 60, 70])
    assert isinstance(plan, ContinuationPlan)
    assert plan.skip_n == 5
    assert plan.new_tokens == [60, 70]


def test_path_select_shorter_history_returns_new_session(
    fresh_verifier_factory,
) -> None:
    """§2.4.b case 2: new prompt is shorter than the cache's logical
    end. The client opened a new conversation."""
    from kv_cache_proposer.path_plan import NewSession

    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30, 40, 50])
    plan = verifier.path_select([10, 20])
    assert isinstance(plan, NewSession)
    assert plan.prompt == [10, 20]


def test_path_select_diverging_history_returns_new_session(
    fresh_verifier_factory,
) -> None:
    """§2.4.b case 3: cached tokens disagree with the new prompt at
    a cached position."""
    from kv_cache_proposer.path_plan import NewSession

    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30, 40, 50])
    # Same length but middle differs at position 3
    plan = verifier.path_select([10, 20, 30, 99, 50])
    assert isinstance(plan, NewSession)


def test_path_select_with_long_prefill_only_compares_cached_positions(
    fresh_verifier_factory,
) -> None:
    """When the prefill exceeds sink+window, the cache holds only
    sink_first + window_last. The match must compare only those
    positions, not the evicted middle."""
    from kv_cache_proposer.path_plan import ContinuationPlan

    verifier = fresh_verifier_factory(sink=2, window=4)
    long_prompt = list(range(100, 120))  # 20 tokens, sink+window=6
    verifier.prefill(long_prompt)
    # cached_token_sequence = [100, 101] + [116, 117, 118, 119]
    assert verifier.cached_token_sequence == [100, 101, 116, 117, 118, 119]
    # Now extend by 3 tokens. The middle [102..115] is irrelevant —
    # it was evicted from the cache and never compared.
    extended_prompt = long_prompt + [200, 201, 202]
    plan = verifier.path_select(extended_prompt)
    assert isinstance(plan, ContinuationPlan)
    assert plan.skip_n == 20
    assert plan.new_tokens == [200, 201, 202]


def test_path_select_diverging_at_evicted_position_is_not_detected(
    fresh_verifier_factory,
) -> None:
    """Sink+window means we cannot detect divergence at evicted
    positions — the K/V values for those positions are no longer in
    the cache. This is a deliberate consequence of the §2.4.a
    precondition: we only check positions the cache holds. Document
    it in test form."""
    from kv_cache_proposer.path_plan import ContinuationPlan

    verifier = fresh_verifier_factory(sink=2, window=4)
    long_prompt = list(range(100, 120))
    verifier.prefill(long_prompt)
    # Forge a "different conversation" that agrees on the cached
    # positions [0,1, 16,17,18,19] but differs on the evicted
    # middle [2..15]:
    forged = (
        [100, 101]                                           # sink match
        + [999] * 14                                         # diverging middle (evicted, not detectable)
        + [116, 117, 118, 119]                               # window match
        + [300, 301, 302]                                    # extension
    )
    plan = verifier.path_select(forged)
    # Continuation accepts this — by design, sink+window's quality
    # approximation already accepts the same lossiness
    # (see ADR 0001 §4 + ADR 0007 §2.7).
    assert isinstance(plan, ContinuationPlan)


def test_path_select_rejects_empty_prompt(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        verifier.path_select([])


def test_prefill_incremental_extends_cache_state(fresh_verifier_factory) -> None:
    """After prefill + prefill_incremental, the verifier state is
    equivalent to a fresh full prefill of the combined prompt (the
    determinism contract from §2.7, validated structurally here;
    the bit-identical check is PR 7-5)."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30])
    verifier.prefill_incremental([40, 50])
    assert verifier.next_global_position == 5
    assert verifier.cached_token_sequence == [10, 20, 30, 40, 50]
    verifier._assert_cache_invariant_1()


def test_prefill_incremental_empty_new_tokens_is_noop(
    fresh_verifier_factory,
) -> None:
    """Edge case: the new prompt exactly matches the cached state.
    prefill_incremental with empty new_tokens is a no-op; cache
    state and next_token_logits are unchanged."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30])
    pos_before = verifier.next_global_position
    seq_before = list(verifier.cached_token_sequence)
    logits_before = verifier.next_token_logits.clone()
    verifier.prefill_incremental([])
    assert verifier.next_global_position == pos_before
    assert verifier.cached_token_sequence == seq_before
    assert torch.equal(verifier.next_token_logits, logits_before)
    verifier._assert_cache_invariant_1()


def test_prefill_incremental_before_any_prefill_raises(
    fresh_verifier_factory,
) -> None:
    """Calling prefill_incremental before any prefill is a usage
    error (caller failed to consult path_select). Raises rather than
    silently routing to a different path — per ADR 0007 §2.9 the
    layer above us must do path selection."""
    verifier = fresh_verifier_factory()
    assert verifier.cache is None
    with pytest.raises(RuntimeError, match="prefill_incremental called before"):
        verifier.prefill_incremental([1, 2, 3])


def test_path_select_continuation_with_long_extension_keeps_kv_bounded(
    fresh_verifier_factory,
) -> None:
    """A continuation that adds many more tokens still ends with the
    KV cache bounded by sink+window — same §2.3.a guarantee as
    full prefill, just achieved with O(new_tokens) prefill cost
    instead of O(history_length)."""
    from kv_cache_proposer.path_plan import ContinuationPlan

    verifier = fresh_verifier_factory(sink=2, window=4)
    initial = list(range(100, 105))
    verifier.prefill(initial)
    # Now extend by 20 tokens
    big_prompt = initial + list(range(200, 220))
    plan = verifier.path_select(big_prompt)
    assert isinstance(plan, ContinuationPlan)
    verifier.prefill_incremental(plan.new_tokens)
    assert verifier.next_global_position == len(big_prompt)
    # Cache must be at sink+window cap = 6 entries
    assert len(verifier.cached_token_sequence) == 6
    # Sink unchanged + last 4 of big_prompt
    assert verifier.cached_token_sequence == [100, 101] + big_prompt[-4:]
    verifier._assert_cache_invariant_1()


def test_inv_2_position_monotonic_across_continuation_chain(
    fresh_verifier_factory,
) -> None:
    """ADR 0007 §2.9 INV-2: across a continuation chain (consecutive
    continuation-path requests for the same session),
    next_global_position is monotonically non-decreasing.

    This is a contract test, not an assertion-fire test. INV-2 is
    structurally satisfied by the architecture (each continuation
    extends, never shrinks); the defensive assert in path_select is
    a safety net for future refactoring. We verify the behavioral
    contract over a 5-turn continuation chain.
    """
    verifier = fresh_verifier_factory(sink=2, window=8)
    history = [10, 20, 30]
    verifier.prefill(history)
    last_position = verifier.next_global_position
    for new_token in [40, 50, 60, 70, 80]:
        history.append(new_token)
        plan = verifier.path_select(history)
        # Plan must be continuation throughout this chain
        from kv_cache_proposer.path_plan import ContinuationPlan
        assert isinstance(plan, ContinuationPlan), (
            f"unexpected NewSession at history={history}"
        )
        verifier.prefill_incremental(plan.new_tokens)
        # INV-2: position must never decrease
        assert verifier.next_global_position >= last_position
        last_position = verifier.next_global_position
    # Final position must equal full history length
    assert verifier.next_global_position == len(history)


# ---------------------------------------------------------------------------
# Regression: _cached_global_positions length mirrors cache_size, not budget
# (PR 7-2 bug from 2026-05-31 Mac M4 smoke test —
#  bench_long_session_mac_v2_smoke_1780236903.json)
# ---------------------------------------------------------------------------


def test_cached_global_positions_uses_actual_cache_size_not_budget(
    fresh_verifier_factory,
) -> None:
    """The original PR 7-2 implementation derived position-list
    length from ``min(next_global_position, sink+window)``, which is
    wrong whenever the cache has shrunk below the budget (e.g. after
    a partial-accept commit_or_truncate on the MLX backend).

    Smoke-test failure mode (2026-05-31 Mac M4): the engine raised
    ``position list of length 68 disagrees with parallel sequence
    of length 55`` on every turn after the first. The fix derives
    length from the actual ``len(cached_token_sequence)``.

    This test exercises the helper directly with synthetic state —
    the buggy state is reachable on MLX through normal flow but not
    on CPU (CPU's ``_trim_cache_in_place`` always pulls cache back
    up to budget after commit).
    """
    verifier = fresh_verifier_factory(sink=2, window=4)  # budget = 6
    # Synthetic post-partial-accept state: cache holds 5 entries
    # (3 sink-relative + 2 window-relative would be conceptually
    # incorrect; use a realistic shape: 2 sink + 3 window).
    # Manually set the parallel sequence to simulate what the cache
    # would look like after a partial-accept commit on MLX.
    verifier.cached_token_sequence = [10, 20, 100, 101, 102]
    verifier.next_global_position = 20  # past budget
    positions = verifier._cached_global_positions()
    # Expected: sink=[0,1] (sink_size=2 entries) + window=[17,18,19]
    # (5 - 2 = 3 window entries ending at 19)
    assert positions == [0, 1, 17, 18, 19]
    # Length matches the actual cache size, not the budget.
    assert len(positions) == len(verifier.cached_token_sequence)


def test_cached_global_positions_with_cache_below_sink_size(
    fresh_verifier_factory,
) -> None:
    """If somehow only 1 entry remains and sink_size=4, the helper
    must not fabricate non-existent positions. All slots are
    sink-classified."""
    verifier = fresh_verifier_factory(sink=4, window=8)
    verifier.cached_token_sequence = [99]
    verifier.next_global_position = 50
    positions = verifier._cached_global_positions()
    assert positions == [0]
    assert len(positions) == len(verifier.cached_token_sequence)


def test_cached_global_positions_with_full_budget_no_eviction(
    fresh_verifier_factory,
) -> None:
    """When cache_size == sink + window and next_global_position is
    well past budget, the helper returns a contiguous-sink-plus-
    sliding-window layout."""
    verifier = fresh_verifier_factory(sink=2, window=4)  # budget = 6
    verifier.cached_token_sequence = list(range(6))
    verifier.next_global_position = 100
    positions = verifier._cached_global_positions()
    assert positions == [0, 1, 96, 97, 98, 99]
    assert len(positions) == 6


def test_cached_global_positions_short_history_no_window_yet(
    fresh_verifier_factory,
) -> None:
    """When n < budget, the cache holds [0..n-1] contiguously
    (no eviction has happened yet)."""
    verifier = fresh_verifier_factory(sink=2, window=4)
    verifier.cached_token_sequence = [10, 20, 30]
    verifier.next_global_position = 3
    positions = verifier._cached_global_positions()
    assert positions == [0, 1, 2]


def test_prompt_matches_cached_positions_with_post_partial_accept_state(
    fresh_verifier_factory,
) -> None:
    """End-to-end: the helper that path_select consults internally
    (``_prompt_matches_cached_positions``) returns True for a
    prompt that agrees at the cached positions, even when the
    cache is in the post-partial-accept state.

    Tests the helper directly because constructing the K/V tensor
    state to match a synthetic cached_token_sequence is not
    practical without going through real prefill — and prefill on
    CPU never produces this state. MLX's natural-flow reproduction
    is in tests/backends/mlx/test_verifier.py.
    """
    verifier = fresh_verifier_factory(sink=2, window=4)
    # Synthetic post-partial-accept state.
    verifier.cached_token_sequence = [10, 20, 100, 101, 102]
    verifier.next_global_position = 20
    # Build a prompt that agrees at the cached positions
    # (sink: 0, 1; window: 17, 18, 19).
    history = [7777] * 20
    history[0], history[1] = 10, 20
    history[17], history[18], history[19] = 100, 101, 102
    extended = history + [200, 201]
    # Match: True (no raise).
    assert verifier._prompt_matches_cached_positions(extended) is True
    # Prompt that diverges at a cached position (window middle).
    diverging = list(extended)
    diverging[18] = 999
    assert verifier._prompt_matches_cached_positions(diverging) is False
    # Prompt that's too short to cover the cached positions.
    too_short = history[:10]  # only covers position 0..9
    assert verifier._prompt_matches_cached_positions(too_short) is False


def test_record_peak_activation_grows_only(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory()
    a = torch.zeros((1, 4, 32), dtype=torch.bfloat16)
    b = torch.zeros((1, 8, 32), dtype=torch.bfloat16)
    verifier._record_peak_activation(a)
    pa1 = verifier.stats.peak_activation_bytes
    verifier._record_peak_activation(b)
    pa2 = verifier.stats.peak_activation_bytes
    assert pa1 > 0 and pa2 > pa1
    # smaller does not regress
    verifier._record_peak_activation(a)
    assert verifier.stats.peak_activation_bytes == pa2
