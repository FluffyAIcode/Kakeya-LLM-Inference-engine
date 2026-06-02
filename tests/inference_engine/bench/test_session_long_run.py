"""Unit tests for the gRPC long-session bench's aggregation logic.

Covers :mod:`inference_engine.bench.session_long_run` to 100%. The
bench's CLI driver under ``scripts/bench_agentic/bench_session_long_run.py``
is exempt from coverage by the same convention as ``serve.py``.
"""

from __future__ import annotations

import math

import pytest

from inference_engine.bench.session_long_run import (
    _bucketize_10min,
    _kv_bounded,
    _latency_drift_p50_s,
    _percentile,
    _prefill_bounded,
    aggregate_run,
)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_returns_none(self):
        assert _percentile([], 0.5) is None

    def test_single_value_returns_that_value(self):
        assert _percentile([4.2], 0.5) == 4.2
        assert _percentile([4.2], 0.0) == 4.2
        assert _percentile([4.2], 1.0) == 4.2

    def test_multiple_values_p50_is_median(self):
        # 5 values -> p50 is the middle one (index 2).
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0

    def test_p95_is_close_to_max(self):
        # For 10 evenly-spaced points 1..10, p95 = 9.55.
        values = [float(i) for i in range(1, 11)]
        result = _percentile(values, 0.95)
        assert result is not None
        assert math.isclose(result, 9.55, abs_tol=1e-9)

    def test_unsorted_input_is_handled(self):
        # Internal sort means caller doesn't have to sort.
        assert _percentile([5, 1, 3, 2, 4], 0.5) == 3

    def test_invalid_pct_raises(self):
        with pytest.raises(ValueError, match="pct must be in"):
            _percentile([1.0, 2.0], 1.5)
        with pytest.raises(ValueError, match="pct must be in"):
            _percentile([1.0, 2.0], -0.1)


# ---------------------------------------------------------------------------
# _kv_bounded
# ---------------------------------------------------------------------------


class TestKvBounded:
    def test_empty_returns_none(self):
        assert _kv_bounded([]) is None

    def test_single_sample_returns_none(self):
        assert _kv_bounded([100]) is None

    def test_within_tolerance_returns_true(self):
        # min=100, max=105 -> 5% drift, under default 10% tolerance.
        assert _kv_bounded([100, 102, 105, 100]) is True

    def test_outside_tolerance_returns_false(self):
        # min=100, max=130 -> 30% drift, over default 10% tolerance.
        assert _kv_bounded([100, 110, 120, 130]) is False

    def test_zero_minimum_uses_div_protect(self):
        # If min is 0, denominator falls back to 1 to avoid div/0;
        # the test thus reduces to "max < tolerance * 1 = 0.10 bytes".
        # For [0, 0, 0] -> max=0 < 0.10 = True (trivially bounded).
        assert _kv_bounded([0, 0, 0]) is True
        # For [0, 5, 10] -> max=10, not < 0.10 -> False.
        assert _kv_bounded([0, 5, 10]) is False

    def test_custom_tolerance(self):
        # 30% drift; with tolerance=0.50 should be True.
        assert _kv_bounded([100, 130], tolerance=0.50) is True
        # Same series with tolerance=0.20 should be False.
        assert _kv_bounded([100, 130], tolerance=0.20) is False


# ---------------------------------------------------------------------------
# _prefill_bounded
# ---------------------------------------------------------------------------


class TestPrefillBounded:
    def test_too_short_returns_none(self):
        # With default head=5, tail=5, anything under 10 returns None.
        assert _prefill_bounded([1.0, 2.0, 3.0, 4.0, 5.0]) is None

    def test_flat_latency_is_bounded(self):
        # 20 samples, all ~1.0s. tail_p50 - head_p50 ~= 0 < 5.
        latencies = [1.0 + 0.01 * i for i in range(20)]
        assert _prefill_bounded(latencies) is True

    def test_growing_latency_above_threshold_unbounded(self):
        # Linear growth from 1.0 to 20.0. head_p50 ~= 1.2, tail_p50 ~= 19.0.
        latencies = [1.0 + i for i in range(20)]
        assert _prefill_bounded(latencies) is False

    def test_growing_latency_within_threshold_bounded(self):
        # Drift of 3 seconds, threshold of 5 seconds.
        latencies = [1.0] * 5 + [2.0] * 5 + [3.0] * 5 + [4.0] * 5
        # head_p50 = 1, tail_p50 = 4, drift = 3. Default threshold = 5.
        assert _prefill_bounded(latencies) is True

    def test_custom_threshold(self):
        latencies = [1.0] * 10 + [4.0] * 10  # drift = 3
        assert _prefill_bounded(latencies, drift_threshold_s=2.0) is False
        assert _prefill_bounded(latencies, drift_threshold_s=10.0) is True

    def test_custom_windows(self):
        latencies = [1.0] * 3 + [10.0] * 3
        # With head=2, tail=2, drift = 9, exceeds default threshold.
        assert _prefill_bounded(
            latencies, head_window=2, tail_window=2,
        ) is False


# ---------------------------------------------------------------------------
# _latency_drift_p50_s
# ---------------------------------------------------------------------------


class TestLatencyDriftP50:
    def test_too_short_returns_none(self):
        assert _latency_drift_p50_s([1.0, 2.0]) is None

    def test_flat_latency_is_zero(self):
        latencies = [1.0] * 20
        result = _latency_drift_p50_s(latencies)
        assert result is not None
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_growth_is_positive(self):
        latencies = [1.0] * 5 + [2.0] * 5 + [3.0] * 5 + [4.0] * 5
        result = _latency_drift_p50_s(latencies)
        assert result is not None
        # head_p50 = 1.0, tail_p50 = 4.0
        assert math.isclose(result, 3.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# _bucketize_10min
# ---------------------------------------------------------------------------


class TestBucketize10min:
    def test_empty_returns_empty(self):
        assert _bucketize_10min([]) == []

    def test_all_errors_returns_empty(self):
        turns = [
            {"ok": False, "t_relative_s": 0, "error_class": "X"},
            {"ok": False, "t_relative_s": 60, "error_class": "Y"},
        ]
        assert _bucketize_10min(turns) == []

    def test_single_bucket(self):
        # All turns under 10 min -> bucket 0.
        turns = [
            {"ok": True, "t_relative_s": 0, "latency_s": 1.0,
             "kv_live_bytes": 100},
            {"ok": True, "t_relative_s": 300, "latency_s": 2.0,
             "kv_live_bytes": 200},
        ]
        out = _bucketize_10min(turns)
        assert len(out) == 1
        assert out[0]["bucket_index"] == 0
        assert out[0]["n_turns"] == 2
        assert out[0]["p50_latency_s"] == 1.5
        assert out[0]["mean_kv_live_bytes"] == 150

    def test_multiple_buckets(self):
        turns = [
            {"ok": True, "t_relative_s": 0, "latency_s": 1.0,
             "kv_live_bytes": 100},
            # Bucket 0 (0-10 min)
            {"ok": True, "t_relative_s": 599, "latency_s": 2.0,
             "kv_live_bytes": 110},
            # Bucket 1 (10-20 min)
            {"ok": True, "t_relative_s": 700, "latency_s": 3.0,
             "kv_live_bytes": 120},
            # Bucket 3 (30-40 min) — gap is intentional
            {"ok": True, "t_relative_s": 1900, "latency_s": 4.0,
             "kv_live_bytes": 130},
        ]
        out = _bucketize_10min(turns)
        assert [b["bucket_index"] for b in out] == [0, 1, 3]
        assert [b["n_turns"] for b in out] == [2, 1, 1]

    def test_skips_kv_none_in_mean(self):
        turns = [
            {"ok": True, "t_relative_s": 0, "latency_s": 1.0,
             "kv_live_bytes": 100},
            {"ok": True, "t_relative_s": 100, "latency_s": 1.0,
             "kv_live_bytes": None},
        ]
        out = _bucketize_10min(turns)
        assert len(out) == 1
        # Only the first turn has a KV value; mean is just that value.
        assert out[0]["mean_kv_live_bytes"] == 100
        # Both turns counted for n_turns.
        assert out[0]["n_turns"] == 2

    def test_all_kv_none_in_bucket_returns_none_mean(self):
        turns = [
            {"ok": True, "t_relative_s": 0, "latency_s": 1.0,
             "kv_live_bytes": None},
        ]
        out = _bucketize_10min(turns)
        assert len(out) == 1
        assert out[0]["mean_kv_live_bytes"] is None

    def test_errors_are_skipped(self):
        turns = [
            {"ok": True, "t_relative_s": 0, "latency_s": 1.0,
             "kv_live_bytes": 100},
            {"ok": False, "t_relative_s": 60, "error_class": "X"},
            {"ok": True, "t_relative_s": 120, "latency_s": 2.0,
             "kv_live_bytes": 110},
        ]
        out = _bucketize_10min(turns)
        # Only the 2 successes counted.
        assert out[0]["n_turns"] == 2


# ---------------------------------------------------------------------------
# aggregate_run
# ---------------------------------------------------------------------------


class TestAggregateRun:
    def test_empty_input(self):
        out = aggregate_run([], duration_s=0.0)
        assert out["n_turns"] == 0
        assert out["n_errors"] == 0
        assert out["duration_s"] == 0.0
        assert out["p50_latency_s"] is None
        assert out["p95_latency_s"] is None
        assert out["min_kv_live_bytes"] is None
        assert out["mean_kv_live_bytes"] is None
        assert out["max_kv_live_bytes"] is None
        assert out["kv_bounded"] is None
        assert out["prefill_bounded"] is None
        assert out["latency_drift_p50_s"] is None
        assert out["buckets_10min"] == []

    def test_all_errors(self):
        turns = [
            {"ok": False, "t_relative_s": 0, "error_class": "TimeoutError"},
            {"ok": False, "t_relative_s": 60, "error_class": "TimeoutError"},
        ]
        out = aggregate_run(turns, duration_s=120.0)
        assert out["n_turns"] == 0
        assert out["n_errors"] == 2
        assert out["p50_latency_s"] is None
        assert out["kv_bounded"] is None

    def test_happy_path_with_kv_bounded_and_prefill_bounded(self):
        # 12 successful turns, flat latency ~1.0s, kv ~ 100 bytes.
        turns = [
            {"ok": True, "t_relative_s": float(i * 30),
             "latency_s": 1.0 + 0.05 * (i % 3),
             "kv_live_bytes": 100 + (i % 3),
             "history_length": 10 + i, "n_emitted": 16,
             "user_message_tokens": 10}
            for i in range(12)
        ]
        out = aggregate_run(turns, duration_s=12 * 30.0)
        assert out["n_turns"] == 12
        assert out["n_errors"] == 0
        assert out["p50_latency_s"] is not None
        assert out["p95_latency_s"] is not None
        assert out["kv_bounded"] is True
        assert out["prefill_bounded"] is True
        assert out["min_kv_live_bytes"] == 100
        assert out["max_kv_live_bytes"] == 102

    def test_unbounded_run_reports_false(self):
        # Latency grows linearly -> prefill_bounded False.
        # KV grows linearly too -> kv_bounded False.
        turns = []
        for i in range(20):
            turns.append({
                "ok": True,
                "t_relative_s": float(i * 30),
                "latency_s": 1.0 + i * 1.0,
                "kv_live_bytes": 100 + i * 100,
            })
        out = aggregate_run(turns, duration_s=600.0)
        assert out["kv_bounded"] is False
        assert out["prefill_bounded"] is False
        assert out["latency_drift_p50_s"] is not None
        assert out["latency_drift_p50_s"] > 0

    def test_mixed_success_and_error(self):
        turns = [
            {"ok": True, "t_relative_s": 0, "latency_s": 1.0,
             "kv_live_bytes": 100},
            {"ok": False, "t_relative_s": 30, "error_class": "X"},
            {"ok": True, "t_relative_s": 60, "latency_s": 1.1,
             "kv_live_bytes": 102},
        ]
        out = aggregate_run(turns, duration_s=90.0)
        assert out["n_turns"] == 2
        assert out["n_errors"] == 1

    def test_custom_thresholds_pass_through(self):
        # Build a run that passes default 10% kv tolerance but fails 1%.
        turns = [
            {"ok": True, "t_relative_s": float(i),
             "latency_s": 1.0,
             "kv_live_bytes": 100 + i}
            for i in range(5)
        ]
        out_default = aggregate_run(turns, duration_s=5.0)
        # 100 -> 104 = 4% drift. Default 10%, so True.
        assert out_default["kv_bounded"] is True
        out_strict = aggregate_run(turns, duration_s=5.0, kv_tolerance=0.01)
        assert out_strict["kv_bounded"] is False

    def test_drift_window_pass_through(self):
        turns = [
            {"ok": True, "t_relative_s": float(i),
             "latency_s": 1.0 + i * 0.1,
             "kv_live_bytes": 100}
            for i in range(20)
        ]
        # With small windows, drift becomes meaningful.
        out = aggregate_run(
            turns, duration_s=20.0,
            drift_head_window=2, drift_tail_window=2,
            drift_threshold_s=0.1,
        )
        # head_p50 ~= 1.05, tail_p50 ~= 2.85, drift ~= 1.8 > 0.1.
        assert out["prefill_bounded"] is False
