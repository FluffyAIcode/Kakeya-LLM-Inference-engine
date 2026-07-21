"""Tests for `inference_engine.backends.mlx.verifier.MLXSinkWindowVerifier`.

Mac-only: every test requires `mlx`, `mlx_lm`, and a working Metal
device. The real Qwen3-1.7B weights are loaded once per session via
the `mlx_verifier_session` fixture (model load is the slow step on
M-series, ~1.5 s; subsequent tests reuse it).

The headline correctness check is
`test_mlx_argmax_matches_pytorch_baseline`, which prefills both the
PyTorch and MLX verifiers with the same prompt and asserts their
argmax-of-next-token agrees. bf16 numerical noise across the two
backends could in principle flip an argmax — if it does we want to
know, so the test asserts equality (not approximate).
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

import torch

from kv_cache_proposer.verifier import VerifierConfig
from inference_engine.backends.mlx.verifier import (
    MLXSinkWindowVerifier,
    _model_weight_bytes,
    _map_torch_dtype_to_mx,
)
from inference_engine.backends.mlx.prefill_snapshot import (
    export_mlx_prefill_snapshot,
)
from inference_engine.distributed.capability import CacheCompatibility


@pytest.fixture(scope="session")
def mlx_verifier_session() -> MLXSinkWindowVerifier:
    return MLXSinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=4, window_size=64,
        )
    )


def _build_mlx_verifier(sink: int = 4, window: int = 64) -> MLXSinkWindowVerifier:
    return MLXSinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=sink, window_size=window,
        )
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_config_loads(mlx_verifier_session: MLXSinkWindowVerifier) -> None:
    v = mlx_verifier_session
    assert v.config.sink_size == 4
    assert v.config.window_size == 64
    assert v.cache is None
    assert v.next_token_logits is None
    assert v.cache_logical_size == 0
    assert v.next_global_position == 0
    assert v.stats.weight_bytes > 0


@pytest.mark.parametrize(
    "sink,window,err",
    [
        (-1, 8, "sink_size must be >= 0"),
        (4, 0, "window_size must be > 0"),
    ],
)
def test_construction_validates_window_args(sink, window, err) -> None:
    with pytest.raises(ValueError, match=err):
        MLXSinkWindowVerifier(
            VerifierConfig(
                dtype=torch.bfloat16, device="cpu",
                sink_size=sink, window_size=window,
            )
        )


def test_construction_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="no MLX equivalent"):
        MLXSinkWindowVerifier(
            VerifierConfig(
                dtype=torch.float64, device="cpu",  # fp64 not in our table
                sink_size=4, window_size=8,
            )
        )


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------

def test_prefill_rejects_empty(mlx_verifier_session: MLXSinkWindowVerifier) -> None:
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        mlx_verifier_session.prefill([])


def test_prefill_under_budget() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    prompt = list(range(20))
    v.prefill(prompt)
    assert v.cache_logical_size == 20
    assert v.next_global_position == 20
    assert v.next_token_logits is not None
    assert v.next_token_logits.shape[-1] > 1000  # vocab size
    assert v.stats.forward_calls == 1


def test_prefill_over_budget_triggers_trim() -> None:
    v = _build_mlx_verifier(sink=4, window=8)
    v.prefill(list(range(50)))
    assert v.cache_logical_size == 12
    # All non-null layers must reflect the trimmed size physically.
    for layer in v.cache:
        if layer.keys is None:
            continue
        assert int(layer.keys.shape[2]) == 12


def test_prefill_zero_sink() -> None:
    v = _build_mlx_verifier(sink=0, window=8)
    v.prefill(list(range(20)))
    assert v.cache_logical_size == 8


# ---------------------------------------------------------------------------
# forward_block + commit_or_truncate
# ---------------------------------------------------------------------------

def test_forward_block_requires_prefill() -> None:
    v = _build_mlx_verifier()
    with pytest.raises(RuntimeError, match="not prefilled"):
        v.forward_block([1, 2, 3])


def test_forward_block_rejects_empty() -> None:
    v = _build_mlx_verifier()
    v.prefill([1, 2, 3])
    with pytest.raises(ValueError, match="tokens must be non-empty"):
        v.forward_block([])


def test_forward_block_returns_per_position_logits() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    L = 5
    block = list(range(100, 100 + L))
    logits = v.forward_block(block)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == L
    assert logits.shape[1] > 1000  # vocab


def test_commit_validates_args() -> None:
    v = _build_mlx_verifier()
    v.prefill([1, 2, 3])
    v.forward_block([4, 5, 6])
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        v.commit_or_truncate(forwarded=3, accepted=-1)
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        v.commit_or_truncate(forwarded=3, accepted=4)


def test_commit_full_accept_no_drop() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102])
    v.commit_or_truncate(forwarded=3, accepted=3)
    assert v.cache_logical_size == 13
    assert v.next_global_position == 13


def test_commit_partial_accept_drops_tail() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102])
    v.commit_or_truncate(forwarded=3, accepted=1)
    assert v.cache_logical_size == 11
    assert v.next_global_position == 11


def test_commit_zero_accept_drops_all() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102])
    v.commit_or_truncate(forwarded=3, accepted=0)
    assert v.cache_logical_size == 10


def test_commit_post_trims_to_budget() -> None:
    v = _build_mlx_verifier(sink=4, window=8)  # budget = 12
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102, 103, 104])  # logical -> 15, then trim
    v.commit_or_truncate(forwarded=5, accepted=5)
    assert v.cache_logical_size == 12


# ---------------------------------------------------------------------------
# append_token
# ---------------------------------------------------------------------------

def test_append_token_advances_state() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    pre_size = v.cache_logical_size
    pre_pos = v.next_global_position
    logits = v.append_token(123)
    assert v.cache_logical_size == pre_size + 1
    assert v.next_global_position == pre_pos + 1
    assert logits is v.next_token_logits
    assert logits.ndim == 1


def test_on_device_argmax_exactly_matches_legacy_torch_argmax(
    mlx_verifier_session: MLXSinkWindowVerifier,
) -> None:
    """The scalar-only MLX reduction must select the exact same token."""
    v = mlx_verifier_session
    v.prefill([1, 2, 3, 4])
    on_device = v.greedy_next_token_id()
    legacy = int(torch.argmax(v.next_token_logits).item())
    assert on_device == legacy


def test_last_logits_path_has_exact_logits_and_kv_parity(
    mlx_verifier_session: MLXSinkWindowVerifier,
) -> None:
    """Optimized all-accepted append is bit-exact to full-block commit."""
    v = mlx_verifier_session
    prompt = [1, 2, 3, 4]
    block = [5, 6, 7]

    v.prefill(prompt)
    v.append_accepted_tokens(block)
    optimized_logits = v.next_token_logits.clone()
    optimized_kv = [
        (layer.keys, layer.values, layer.offset) for layer in v.cache
    ]
    mx.eval(*[
        tensor
        for keys, values, _offset in optimized_kv
        for tensor in (keys, values)
    ])
    optimized_tokens = list(v.cached_token_sequence)
    optimized_position = v.next_global_position
    compatibility = CacheCompatibility(model_id="step4-parity")
    optimized_snapshot = export_mlx_prefill_snapshot(
        v.cache,
        token_count=optimized_position,
        cached_token_ids=optimized_tokens,
        compatibility=compatibility,
    )

    v.prefill(prompt)
    full_logits = v.forward_block(block)
    v.commit_or_truncate(forwarded=len(block), accepted=len(block))
    full_snapshot = export_mlx_prefill_snapshot(
        v.cache,
        token_count=v.next_global_position,
        cached_token_ids=v.cached_token_sequence,
        compatibility=compatibility,
    )

    assert torch.equal(optimized_logits, full_logits[-1])
    assert optimized_snapshot == full_snapshot
    assert optimized_tokens == v.cached_token_sequence
    assert optimized_position == v.next_global_position
    for (expected_k, expected_v, expected_offset), layer in zip(
        optimized_kv, v.cache,
    ):
        assert expected_offset == layer.offset
        assert bool(mx.array_equal(expected_k, layer.keys).item())
        assert bool(mx.array_equal(expected_v, layer.values).item())


def test_greedy_path_avoids_vocab_bridge_and_full_block_keeps_it(
    mlx_verifier_session: MLXSinkWindowVerifier,
    monkeypatch,
) -> None:
    """Focused sync regression: greedy crosses zero vocabulary rows."""
    import inference_engine.backends.mlx.verifier as verifier_module

    bridged_shapes = []
    real_bridge = verifier_module.mx_to_torch

    def recording_bridge(arr):
        bridged_shapes.append(tuple(arr.shape))
        return real_bridge(arr)

    monkeypatch.setattr(verifier_module, "mx_to_torch", recording_bridge)
    v = mlx_verifier_session
    v.prefill([1, 2, 3, 4])
    assert v._next_token_logits_torch is None
    token = v.greedy_next_token_id()
    v.append_accepted_tokens([token])
    assert bridged_shapes == []

    # Speculative semantics are deliberately unchanged: full [L,V]
    # crosses the compatibility boundary once.
    block_logits = v.forward_block([5, 6])
    assert bridged_shapes == [tuple(block_logits.shape)]
    assert block_logits.ndim == 2 and block_logits.shape[0] == 2


def test_last_logits_memory_accounting_scales_as_one_vocab_row(
    mlx_verifier_session: MLXSinkWindowVerifier,
) -> None:
    """Peak output accounting distinguishes [V] from full [L,V]."""
    v = mlx_verifier_session
    v.stats.peak_activation_bytes = 0
    v.prefill([1, 2, 3, 4])
    row_bytes = v.stats.peak_activation_bytes
    expected_row_bytes = (
        int(v._next_token_logits_mx.size)
        * int(v._next_token_logits_mx.dtype.size)
    )
    assert row_bytes == expected_row_bytes

    v.append_accepted_tokens([5, 6, 7])
    assert v.stats.peak_activation_bytes == row_bytes

    v.stats.peak_activation_bytes = 0
    full_logits = v.forward_block([8, 9, 10])
    assert v.stats.peak_activation_bytes == (
        int(full_logits.numel()) * int(full_logits.element_size())
    )
    assert v.stats.peak_activation_bytes == 3 * row_bytes


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def test_cache_buffer_size_zero_when_no_cache() -> None:
    v = _build_mlx_verifier()
    assert v._cache_buffer_size() == 0


def test_commit_per_layer_trim_mismatch_raises(monkeypatch) -> None:
    """If, somehow, layers trim by inconsistent amounts (a real bug we
    want surfaced), commit_or_truncate must raise rather than silently
    proceed."""
    from inference_engine.backends.mlx.cache import SinkWindowKVCache
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(20)))
    v.forward_block([100, 101, 102])
    # Force one layer to claim it trimmed less than asked.
    real_trim = SinkWindowKVCache.trim
    seen = {"first": True}

    def _bad_trim(self, n):
        if seen["first"]:
            seen["first"] = False
            return n - 1  # off by one
        return real_trim(self, n)

    monkeypatch.setattr(SinkWindowKVCache, "trim", _bad_trim)
    with pytest.raises(RuntimeError, match="per-layer trim mismatch"):
        v.commit_or_truncate(forwarded=3, accepted=1)


def test_record_peak_kv_handles_null_cache() -> None:
    v = _build_mlx_verifier()
    pre = v.stats.peak_kv_bytes
    v._record_peak_kv()  # cache is None
    assert v.stats.peak_kv_bytes == pre


def test_live_kv_bytes_zero_before_prefill() -> None:
    """The /metrics gauge must read 0 before any prefill."""
    v = _build_mlx_verifier()
    assert v.live_kv_bytes() == 0


def test_live_kv_bytes_nonzero_after_prefill() -> None:
    """During in-flight generation the gauge must read the actual
    bytes — this is what bench_long_session.py polls on each turn
    to verify the ADR 0006 §2.3 KV-bounded claim."""
    v = _build_mlx_verifier()
    v.prefill(list(range(16)))
    n = v.live_kv_bytes()
    assert n > 0
    # Right after prefill, peak == live.
    assert v.stats.peak_kv_bytes == n


# ---------------------------------------------------------------------------
# ADR 0007 §2.2 + §2.9 — cached_token_sequence + INV-1
# ---------------------------------------------------------------------------


def test_mlx_cached_token_sequence_empty_after_construction() -> None:
    v = _build_mlx_verifier()
    assert v.cached_token_sequence == []
    v._assert_cache_invariant_1()


def test_mlx_cached_token_sequence_populated_after_short_prefill() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    prompt = list(range(5))  # 5 < sink+window = 10
    v.prefill(prompt)
    assert v.cached_token_sequence == prompt
    v._assert_cache_invariant_1()


def test_mlx_cached_token_sequence_trimmed_after_long_prefill() -> None:
    v = _build_mlx_verifier(sink=2, window=4)
    prompt = list(range(20))  # 20 > sink+window = 6
    v.prefill(prompt)
    expected = prompt[:2] + prompt[-4:]
    assert v.cached_token_sequence == expected
    v._assert_cache_invariant_1()


def test_mlx_cached_token_sequence_extends_on_forward_block() -> None:
    """``forward_block`` extends the cache; the parallel sequence
    extends in lockstep, then the same sink+window slice that the
    K/V tensors apply is applied here too."""
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([0, 1, 2, 3])
    v.forward_block([4, 5])
    # 6 entries, all under budget=10
    assert v.cached_token_sequence == [0, 1, 2, 3, 4, 5]
    v._assert_cache_invariant_1()


def test_mlx_cached_token_sequence_drops_rejected_tail_on_partial_accept() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([0, 1, 2, 3])
    v.forward_block([4, 5, 6])
    v.commit_or_truncate(forwarded=3, accepted=1)
    assert v.cached_token_sequence == [0, 1, 2, 3, 4]
    v._assert_cache_invariant_1()


def test_mlx_cached_token_sequence_after_append_token() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([0, 1, 2, 3])
    v.append_token(99)
    assert v.cached_token_sequence == [0, 1, 2, 3, 99]
    v._assert_cache_invariant_1()


def test_mlx_cached_token_sequence_cleared_on_reset() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([0, 1, 2, 3])
    assert v.cached_token_sequence != []
    v.reset()
    assert v.cached_token_sequence == []
    v._assert_cache_invariant_1()


def test_mlx_inv_1_violation_raises_assertion_error() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([0, 1, 2, 3])
    v.cached_token_sequence = v.cached_token_sequence + [999]
    with pytest.raises(AssertionError, match="INV-1 violated"):
        v._assert_cache_invariant_1()


def test_mlx_inv_1_assertion_message_carries_diagnostic_state() -> None:
    """The error message must expose actual vs expected lengths plus
    the verifier's logical-position counters so a bug report can be
    triaged from the message alone."""
    v = _build_mlx_verifier()
    v.prefill([0, 1, 2, 3])
    v.cached_token_sequence = v.cached_token_sequence + [42, 43]
    with pytest.raises(AssertionError) as exc:
        v._assert_cache_invariant_1()
    msg = str(exc.value)
    assert "INV-1" in msg
    assert "cached_token_sequence" in msg
    assert "cache_logical_size=" in msg
    assert "next_global_position=" in msg


def test_mlx_inv_1_holds_when_cache_is_none() -> None:
    """The pre-prefill state (cache None, sequence []) is the trivial
    INV-1 satisfaction — must not raise."""
    v = _build_mlx_verifier()
    assert v.cache is None
    assert v.cached_token_sequence == []
    v._assert_cache_invariant_1()


def test_mlx_sink_window_slice_below_budget_returns_input_unchanged() -> None:
    """The internal helper short-circuits when sequence fits in
    sink+window."""
    v = _build_mlx_verifier(sink=2, window=4)
    seq = [10, 20, 30]
    out = v._sink_window_slice(seq)
    assert out == seq
    assert out is not seq  # returns a copy


def test_mlx_sink_window_slice_above_budget_keeps_sink_plus_tail() -> None:
    v = _build_mlx_verifier(sink=2, window=4)
    seq = list(range(20))
    out = v._sink_window_slice(seq)
    assert out == seq[:2] + seq[-4:]


def test_record_peak_activation_grows_only() -> None:
    v = _build_mlx_verifier()
    a = mx.zeros((1, 4, 32), dtype=mx.bfloat16)
    b = mx.zeros((1, 8, 32), dtype=mx.bfloat16)
    v._record_peak_activation(a)
    pa1 = v.stats.peak_activation_bytes
    v._record_peak_activation(b)
    pa2 = v.stats.peak_activation_bytes
    v._record_peak_activation(a)
    assert pa1 > 0 and pa2 > pa1
    assert v.stats.peak_activation_bytes == pa2  # smaller doesn't regress


# ---------------------------------------------------------------------------
# ADR 0008 PR-A3b — CacheInspector protocol on MLX verifier
# ---------------------------------------------------------------------------


def test_mlx_k_seq_length_zero_before_prefill() -> None:
    """Cold MLX verifier reports k_seq_length = 0; matches
    Session.cached_token_sequence empty initial state, so INV-1
    trivially holds at session creation."""
    v = _build_mlx_verifier()
    assert v.k_seq_length(session=None) == 0


def test_mlx_k_seq_length_matches_cache_buffer_after_prefill() -> None:
    """After prefill, k_seq_length equals the SinkWindowKVCache's
    post-trim buffer size and equals len(cached_token_sequence) —
    the basis of INV-1 enforcement at the session layer."""
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30, 40, 50])
    k_len = v.k_seq_length(session=None)
    assert k_len == 5
    assert k_len == len(v.cached_token_sequence)


def test_mlx_k_seq_length_after_long_prefill_is_sink_plus_window() -> None:
    """When the prefill exceeds sink+window, the cache is trimmed
    and k_seq_length reflects the post-trim size, not the prompt
    length."""
    v = _build_mlx_verifier(sink=2, window=4)
    v.prefill(list(range(100, 120)))  # 20 tokens, budget = 6
    assert v.k_seq_length(session=None) == 6


def test_mlx_verifier_satisfies_cache_inspector_protocol() -> None:
    """The MLX verifier can be passed as the cache_inspector to
    SessionStore; INV-1 enforcement uses the live MLX cache state."""
    from inference_engine.session import SessionStore

    v = _build_mlx_verifier(sink=2, window=8)
    store = SessionStore(capacity=1, cache_inspector=v)
    sess = store.create_session()
    # Cold verifier: k_seq_length=0 matches empty cached_token_sequence.
    store.append_tokens(sess.session_id, [1, 2, 3])
    assert sess.history_token_ids == [1, 2, 3]
    assert callable(v.k_seq_length)


# ---------------------------------------------------------------------------
# ADR 0008 PR-E1c — kv_live_bytes accessor on the MLX verifier
# ---------------------------------------------------------------------------


def test_mlx_kv_live_bytes_zero_before_prefill() -> None:
    v = _build_mlx_verifier()
    assert v.kv_live_bytes(session=None) == 0


def test_mlx_kv_live_bytes_equals_k_seq_length_times_per_token() -> None:
    """kv_live_bytes = k_seq_length × resolved per-token bytes.

    The verifier may resolve geometry from an HF config or directly from a
    multimodal mlx-lm text-model wrapper (Gemma 4).
    """
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30, 40, 50])
    k_len = v.k_seq_length(session=None)
    assert k_len == 5
    bytes_per_token = v._bytes_per_kv_token
    expected = k_len * bytes_per_token
    assert v.kv_live_bytes(session=None) == expected
    assert expected > 0


def test_mlx_kv_live_bytes_plateaus_at_capacity() -> None:
    v = _build_mlx_verifier(sink=2, window=4)
    v.prefill(list(range(100, 106)))  # exactly sink+window
    bytes_at_cap = v.kv_live_bytes(session=None)
    v.forward_block([200, 201, 202])
    v.commit_or_truncate(forwarded=3, accepted=3)
    bytes_after = v.kv_live_bytes(session=None)
    assert bytes_at_cap == bytes_after


def test_reset_clears_state() -> None:
    v = _build_mlx_verifier()
    v.prefill([1, 2, 3])
    v.reset()
    assert v.next_token_logits is None
    assert v.cache_logical_size == 0
    assert v.next_global_position == 0
    assert v.cache is not None  # reset re-creates an empty cache list


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_model_weight_bytes_positive(mlx_verifier_session) -> None:
    bytes_ = _model_weight_bytes(mlx_verifier_session.model)
    # Qwen3-1.7B ≈ 3.4 GB at bf16 ≈ 3.4e9 bytes
    assert bytes_ > 1_000_000_000


def test_verifier_exposes_quantization_attribute(mlx_verifier_session) -> None:
    """The verifier records its quantization info on construction.

    For the bf16 baseline this is ``is_quantized=False`` plus a
    sensible total_weight_bytes; for a 4-bit checkpoint (not exercised
    here — it would require a separate fixture pulling from
    mlx-community) it would carry bits / group_size / effective bits.
    Either way, the attribute exists and is consistent with stats.
    """
    info = mlx_verifier_session.quantization
    assert info is not None
    assert info.is_quantized is False
    assert info.total_weight_bytes > 1_000_000_000
    assert info.full_precision_weight_bytes == info.total_weight_bytes
    assert info.quantized_weight_bytes == 0


def test_verifier_stats_weight_bytes_matches_quantization_total(
    mlx_verifier_session,
) -> None:
    """The legacy ``stats.weight_bytes`` is the same number as the
    quantization-aware ``quantization.total_weight_bytes``. Both
    reporting paths must agree exactly so existing dashboards keep
    working."""
    v = mlx_verifier_session
    assert v.stats.weight_bytes == v.quantization.total_weight_bytes


@pytest.mark.parametrize(
    "torch_dtype,mx_dtype",
    [
        (torch.bfloat16, mx.bfloat16),
        (torch.float16, mx.float16),
        (torch.float32, mx.float32),
    ],
)
def test_dtype_mapping_supported(torch_dtype, mx_dtype) -> None:
    assert _map_torch_dtype_to_mx(torch_dtype) == mx_dtype


def test_dtype_mapping_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="no MLX equivalent"):
        _map_torch_dtype_to_mx(torch.float64)


# ---------------------------------------------------------------------------
# Headline cross-backend correctness
# ---------------------------------------------------------------------------

def _greedy_decode(verifier, prompt_ids, max_new_tokens, eos_set):
    """Pure greedy AR generation (no proposer). Used by the regression
    tests below to exercise the verifier across many forwards."""
    verifier.prefill(prompt_ids)
    out = []
    while len(out) < max_new_tokens:
        tok = int(torch.argmax(verifier.next_token_logits).item())
        out.append(tok)
        if tok in eos_set:
            break
        verifier.append_token(tok)
    return out


def test_mlx_argmax_matches_pytorch_baseline() -> None:
    """MLX verifier's first-token argmax must equal the PyTorch
    verifier's first-token argmax for the same prompt. Primary
    correctness gate.
    """
    from kv_cache_proposer.verifier import SinkWindowVerifier

    cpu_v = SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=4, window_size=64,
        )
    )
    mlx_v = _build_mlx_verifier(sink=4, window=64)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    cpu_ids = cpu_v.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    cpu_v.prefill(cpu_ids)
    mlx_v.prefill(cpu_ids)

    cpu_argmax = int(torch.argmax(cpu_v.next_token_logits).item())
    mlx_argmax = int(torch.argmax(mlx_v.next_token_logits).item())
    assert cpu_argmax == mlx_argmax, (
        f"first-token argmax differs across backends: "
        f"cpu={cpu_argmax}  mlx={mlx_argmax}"
    )


def test_mlx_long_generation_matches_pytorch_below_budget() -> None:
    """**Regression test for the MLX-1b divergence-after-trim bug.**

    Drives both the PyTorch CPU and MLX verifier through 50 greedy
    decode steps using a config where the cache budget (sink+window)
    is LARGER than the full sequence — i.e. no trim is ever triggered.
    Outputs must be bit-identical: there's no eviction, so any drift
    is pure backend numerical noise, and Qwen3-1.7B's argmax margins
    are well above bf16 reduction error in this regime.
    """
    from kv_cache_proposer.verifier import SinkWindowVerifier

    SINK, WINDOW = 4, 256  # 260-slot budget >> 36 prompt + 50 gen
    cpu_v = SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=SINK, window_size=WINDOW,
        )
    )
    mlx_v = _build_mlx_verifier(sink=SINK, window=WINDOW)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
    prompt_ids = cpu_v.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    eos_set = {cpu_v.tokenizer.convert_tokens_to_ids("<|im_end|>")}
    cpu_out = _greedy_decode(cpu_v, prompt_ids, 50, eos_set)
    mlx_out = _greedy_decode(mlx_v, prompt_ids, 50, eos_set)
    assert cpu_out == mlx_out, (
        f"CPU and MLX diverged in long generation (no-trim regime):\n"
        f"  cpu: {cpu_out}\n"
        f"  mlx: {mlx_out}"
    )


def test_mlx_long_generation_with_trim_matches_pytorch() -> None:
    """**The headline regression test.**

    Exercises the post-trim code path that broke in MLX-1b: drives
    both the PyTorch CPU and MLX verifier with sink+window deliberately
    smaller than the full sequence so that trim *does* fire mid-
    generation. The two verifiers must produce identical token
    sequences — both implement the same StreamingLLM-style sink+window
    semantics, so once they're at parity the only surviving difference
    is bf16 reduction order, which is below the argmax margin for
    Qwen3-1.7B in chat-style continuations.

    Caveat: bf16 noise *can* eventually flip an argmax in pathological
    inputs. We assert exact equality for the first 32 tokens (well
    within the typical safe range) and at least 95% prefix agreement
    over the full 50.
    """
    from kv_cache_proposer.verifier import SinkWindowVerifier

    SINK, WINDOW = 4, 32  # 36-slot budget — smaller than 30 prompt + 50 gen
    cpu_v = SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=SINK, window_size=WINDOW,
        )
    )
    mlx_v = _build_mlx_verifier(sink=SINK, window=WINDOW)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
    prompt_ids = cpu_v.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    eos_set = {cpu_v.tokenizer.convert_tokens_to_ids("<|im_end|>")}
    cpu_out = _greedy_decode(cpu_v, prompt_ids, 50, eos_set)
    mlx_out = _greedy_decode(mlx_v, prompt_ids, 50, eos_set)

    # Find the first divergence
    common = 0
    for a, b in zip(cpu_out, mlx_out):
        if a == b:
            common += 1
        else:
            break

    n = max(len(cpu_out), len(mlx_out))
    assert common >= 32, (
        f"CPU and MLX diverged earlier than position 32 in trim regime "
        f"(common={common}/{n}). This is the divergence pattern the "
        f"MLX-1b cache mutation bug produced — investigate "
        f"SinkWindowKVCache.update_and_fetch / make_mask:\n"
        f"  cpu: {cpu_out[:common+5]}\n"
        f"  mlx: {mlx_out[:common+5]}"
    )
    # Allow up to 5% disagreement after position 32 (bf16 noise tolerance)
    assert common >= int(0.95 * n), (
        f"CPU/MLX prefix agreement {common}/{n} below 95% threshold; "
        f"unexpected drift accumulating across backends."
    )
