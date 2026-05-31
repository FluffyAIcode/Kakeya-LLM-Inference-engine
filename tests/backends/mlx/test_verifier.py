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


# ---------------------------------------------------------------------------
# ADR 0007 §2.4 — MLX path_select + prefill_incremental + INV-2
# ---------------------------------------------------------------------------


def test_mlx_path_select_cold_start_returns_new_session() -> None:
    from kv_cache_proposer.path_plan import NewSession
    v = _build_mlx_verifier()
    plan = v.path_select([1, 2, 3])
    assert isinstance(plan, NewSession)
    assert plan.prompt == [1, 2, 3]


def test_mlx_path_select_extends_returns_continuation() -> None:
    from kv_cache_proposer.path_plan import ContinuationPlan
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30, 40, 50])
    plan = v.path_select([10, 20, 30, 40, 50, 60, 70])
    assert isinstance(plan, ContinuationPlan)
    assert plan.skip_n == 5
    assert plan.new_tokens == [60, 70]


def test_mlx_path_select_shorter_history_returns_new_session() -> None:
    from kv_cache_proposer.path_plan import NewSession
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30, 40, 50])
    plan = v.path_select([10, 20])
    assert isinstance(plan, NewSession)


def test_mlx_path_select_diverging_history_returns_new_session() -> None:
    from kv_cache_proposer.path_plan import NewSession
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30, 40, 50])
    plan = v.path_select([10, 20, 30, 99, 50])
    assert isinstance(plan, NewSession)


def test_mlx_path_select_with_long_prefill_only_compares_cached_positions() -> None:
    from kv_cache_proposer.path_plan import ContinuationPlan
    v = _build_mlx_verifier(sink=2, window=4)
    long_prompt = list(range(100, 120))
    v.prefill(long_prompt)
    extended = long_prompt + [200, 201, 202]
    plan = v.path_select(extended)
    assert isinstance(plan, ContinuationPlan)
    assert plan.skip_n == 20


def test_mlx_path_select_rejects_empty_prompt() -> None:
    v = _build_mlx_verifier()
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        v.path_select([])


def test_mlx_prefill_incremental_extends_cache_state() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30])
    v.prefill_incremental([40, 50])
    assert v.next_global_position == 5
    assert v.cached_token_sequence == [10, 20, 30, 40, 50]
    v._assert_cache_invariant_1()


def test_mlx_prefill_incremental_empty_new_tokens_is_noop() -> None:
    v = _build_mlx_verifier(sink=2, window=8)
    v.prefill([10, 20, 30])
    pos_before = v.next_global_position
    seq_before = list(v.cached_token_sequence)
    logits_before = v.next_token_logits.clone()
    v.prefill_incremental([])
    assert v.next_global_position == pos_before
    assert v.cached_token_sequence == seq_before
    assert torch.equal(v.next_token_logits, logits_before)
    v._assert_cache_invariant_1()


def test_mlx_prefill_incremental_before_any_prefill_raises() -> None:
    v = _build_mlx_verifier()
    assert v.cache is None
    with pytest.raises(RuntimeError, match="prefill_incremental called before"):
        v.prefill_incremental([1, 2, 3])


def test_mlx_inv_2_position_monotonic_across_continuation_chain() -> None:
    """Mirror of the CPU INV-2 contract test (no contrived
    assertion firing — the guard is defensively coded and
    structurally unreachable; we verify the behavioral contract)."""
    from kv_cache_proposer.path_plan import ContinuationPlan
    v = _build_mlx_verifier(sink=2, window=8)
    history = [10, 20, 30]
    v.prefill(history)
    last_position = v.next_global_position
    for new_token in [40, 50, 60, 70, 80]:
        history.append(new_token)
        plan = v.path_select(history)
        assert isinstance(plan, ContinuationPlan)
        v.prefill_incremental(plan.new_tokens)
        assert v.next_global_position >= last_position
        last_position = v.next_global_position
    assert v.next_global_position == len(history)


# ---------------------------------------------------------------------------
# MLX regression: partial-accept state (PR 7-2 bug, 2026-05-31 smoke)
# Real-flow reproduction (CPU's _trim pulls back to budget on every
# commit, so this state is unreachable there; MLX trims per-layer
# during forward and naturally drops below budget after partial accept).
# ---------------------------------------------------------------------------


def test_mlx_cached_global_positions_after_partial_accept() -> None:
    """Reproduces the smoke-test failure mode through real flow.

    Pre-fix: this state caused _cached_global_positions to return
    a list of length budget=6 even though cache had shrunk below.
    Post-fix: positions list length matches actual cache size.
    """
    v = _build_mlx_verifier(sink=2, window=4)
    v.prefill(list(range(50, 56)))
    v.forward_block([70, 80])
    v.commit_or_truncate(forwarded=2, accepted=2)
    v.forward_block([90, 91, 92])
    v.commit_or_truncate(forwarded=3, accepted=1)
    cache_size = len(v.cached_token_sequence)
    positions = v._cached_global_positions()
    assert len(positions) == cache_size
    v._assert_cache_invariant_1()


def test_mlx_path_select_after_partial_accept_does_not_raise() -> None:
    """End-to-end MLX reproduction: after partial-accept, path_select
    produces a valid plan instead of raising the smoke-test error."""
    from kv_cache_proposer.path_plan import ContinuationPlan
    v = _build_mlx_verifier(sink=2, window=4)
    v.prefill(list(range(50, 56)))
    v.forward_block([70, 80])
    v.commit_or_truncate(forwarded=2, accepted=2)
    v.forward_block([90, 91, 92])
    v.commit_or_truncate(forwarded=3, accepted=1)
    cache_size = len(v.cached_token_sequence)
    n = v.next_global_position
    sink_eff = min(v.config.sink_size, cache_size)
    window_eff = cache_size - sink_eff
    history = [-1] * n
    for i in range(sink_eff):
        history[i] = v.cached_token_sequence[i]
    for j, global_pos in enumerate(range(n - window_eff, n)):
        history[global_pos] = v.cached_token_sequence[sink_eff + j]
    for i in range(n):
        if history[i] == -1:
            history[i] = 7777
    extended = history + [200, 201]
    plan = v.path_select(extended)
    assert isinstance(plan, ContinuationPlan)
    assert plan.skip_n == n
    assert plan.new_tokens == [200, 201]


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
