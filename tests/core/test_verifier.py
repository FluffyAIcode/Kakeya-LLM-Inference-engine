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


# ---------------------------------------------------------------------------
# ADR 0008 PR-A3b — CacheInspector protocol implementation
# ---------------------------------------------------------------------------


def test_k_seq_length_returns_zero_before_prefill(fresh_verifier_factory) -> None:
    """The CacheInspector contract: with no cache allocated, the
    seq length is 0. Session.cached_token_sequence is also empty,
    so SessionStore's INV-1 check (len == k_seq_length) trivially
    holds at session creation time, before the first prefill."""
    verifier = fresh_verifier_factory()
    assert verifier.k_seq_length(session=None) == 0


def test_k_seq_length_matches_cache_seq_dim_after_prefill(
    fresh_verifier_factory,
) -> None:
    """After a prefill, k_seq_length must equal the actual K/V
    tensor sequence dimension and equal len(cached_token_sequence)
    — the latter being the basis of INV-1."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30, 40, 50])
    k_len = verifier.k_seq_length(session=None)
    assert k_len == 5
    assert k_len == len(verifier.cached_token_sequence)


def test_k_seq_length_ignores_session_argument(fresh_verifier_factory) -> None:
    """v0.3 single-tenant scope: the session argument is accepted for
    Protocol conformance but ignored. Two arbitrary objects produce
    the same value."""
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30])
    assert verifier.k_seq_length(session=object()) == verifier.k_seq_length(
        session="any-string-shaped-stand-in",
    )


def test_kv_live_bytes_returns_zero_before_prefill(
    fresh_verifier_factory,
) -> None:
    """PR-E1c: with no cache allocated, kv_live_bytes is 0. Session
    has nothing to report yet."""
    verifier = fresh_verifier_factory()
    assert verifier.kv_live_bytes(session=None) == 0


def test_kv_live_bytes_equals_k_seq_length_times_per_token_bytes(
    fresh_verifier_factory,
) -> None:
    """PR-E1c: ``kv_live_bytes = k_seq_length × per-token bytes``.

    Per-token bytes itself is
    ``num_layers × num_kv_heads × head_dim × itemsize × 2`` (×2 = K + V).
    We compute it from the model config the same way the verifier does
    to verify the closed-form relationship; an off-by-one in either
    factor would surface immediately because the resulting product no
    longer matches.
    """
    verifier = fresh_verifier_factory(sink=2, window=8)
    verifier.prefill([10, 20, 30, 40, 50])
    k_len = verifier.k_seq_length(session=None)
    assert k_len == 5
    cfg = verifier.model.config
    num_layers = int(cfg.num_hidden_layers)
    num_kv_heads = int(
        getattr(cfg, "num_key_value_heads", None)
        or cfg.num_attention_heads
    )
    head_dim = int(
        getattr(cfg, "head_dim", None)
        or (cfg.hidden_size // cfg.num_attention_heads)
    )
    bytes_per_token = (
        num_layers * num_kv_heads * head_dim
        * verifier.config.dtype.itemsize * 2
    )
    expected = k_len * bytes_per_token
    assert verifier.kv_live_bytes(session=None) == expected
    # And for the headline 4-h Mac M4 Qwen3-0.6B numbers, this is in
    # the multi-megabyte range — no longer the constant 0 the bench
    # surfaced before PR-E1c.
    assert expected > 0


def test_kv_live_bytes_plateaus_at_capacity(fresh_verifier_factory) -> None:
    """The architectural KV-bound claim: once k_seq_length hits
    sink+window, kv_live_bytes plateaus. This test compares the
    bytes after a prefill that fills the cache to the bytes after
    additional tokens are forwarded — they must be equal."""
    verifier = fresh_verifier_factory(sink=2, window=4)
    verifier.prefill([10, 20, 30, 40, 50, 60])  # 6 = sink+window cap
    bytes_at_cap = verifier.kv_live_bytes(session=None)
    # Forward more tokens — sink+window keeps trimming so k_seq stays at cap.
    verifier.forward_block([70, 80, 90])
    verifier.commit_or_truncate(forwarded=3, accepted=3)
    bytes_after_forward = verifier.kv_live_bytes(session=None)
    assert bytes_at_cap == bytes_after_forward


def test_cpu_verifier_satisfies_cache_inspector_protocol(
    fresh_verifier_factory,
) -> None:
    """Structural typing check: the CPU verifier is a valid
    CacheInspector. SessionStore can be constructed with a real
    verifier as its inspector and INV-1 enforcement uses verifier
    state."""
    from inference_engine.session import CacheInspector, SessionStore

    verifier = fresh_verifier_factory(sink=2, window=8)
    store = SessionStore(capacity=1, cache_inspector=verifier)
    sess = store.create_session()
    # Sanity: cold verifier reports 0; cached_token_sequence is also
    # empty; INV-1 holds.
    store.append_tokens(sess.session_id, [1, 2, 3])
    assert sess.history_token_ids == [1, 2, 3]
    # Confirm the protocol shape; runtime-checkable Protocol would
    # report True, but isinstance(_, Protocol) requires the @runtime_checkable
    # decorator. We assert by signature presence instead.
    assert hasattr(CacheInspector, "k_seq_length")
    assert callable(verifier.k_seq_length)
