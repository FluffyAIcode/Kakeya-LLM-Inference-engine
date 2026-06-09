"""Linux CI unit tests for inference_engine/v04/niah_eval.py.

These tests exercise the NIAH harness logic — dataset generation,
recall scoring, the sink+window 4D mask helper — without any HF
model dependency. The actual model evaluations (oracle / v0.3 /
v0.4 greedy decode on real Gemma 3-1B-it) are covered by the K1.E
Mac M4 reviewer aid.

Test classes:

* TestMakeNIAHDataset — sample generation correctness; reproducibility
  under fixed seed; needle position invariants; validation raises.
* TestRecallPredicate — substring match semantics.
* TestAggregateRecall — recall computation, latency stats, mismatch
  raises.
* TestSinkWindow4DMask — sink+window region validity for several
  configurations; finfo.min vs -inf for float dtypes.
* TestEvaluateOrchestration — the high-level evaluate() loop with a
  fake decode function.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.v04.niah_eval import (
    NIAHEvalResult,
    NIAHSample,
    aggregate_recall,
    evaluate,
    make_niah_dataset,
    make_sink_window_4d_mask,
    recall_predicate,
)


# ---------------------------------------------------------------------------
# make_niah_dataset
# ---------------------------------------------------------------------------


class TestMakeNIAHDataset:
    def test_default_n_samples(self):
        samples = make_niah_dataset(n_samples=5, seed=0)
        assert len(samples) == 5

    def test_each_sample_has_needle_and_question(self):
        samples = make_niah_dataset(n_samples=3, seed=0)
        for s in samples:
            assert "IMPORTANT: the secret code is" in s.prompt_text
            assert s.answer_text in s.prompt_text
            assert "Question: what is the secret code?" in s.prompt_text

    def test_answer_text_format(self):
        samples = make_niah_dataset(n_samples=10, seed=0)
        for s in samples:
            # Format: PREFIX-NNNN
            parts = s.answer_text.split("-")
            assert len(parts) == 2, f"unexpected format: {s.answer_text}"
            assert parts[0].isalpha()
            assert parts[1].isdigit()
            code_num = int(parts[1])
            assert 1000 <= code_num <= 9999

    def test_needle_text_contains_answer(self):
        samples = make_niah_dataset(n_samples=10, seed=0)
        for s in samples:
            assert s.answer_text in s.needle_text

    def test_needle_position_inside_safe_range(self):
        """Needle must be inserted with at least 4 padding lines before
        and 4 after, so neither sink (4 lines) nor a small trailing
        window can plausibly catch it from positional luck alone."""
        samples = make_niah_dataset(
            n_samples=20, haystack_min_lines=20, haystack_max_lines=30,
            seed=0,
        )
        for s in samples:
            assert s.needle_line_index >= 4

    def test_reproducible_under_fixed_seed(self):
        a = make_niah_dataset(n_samples=10, seed=42)
        b = make_niah_dataset(n_samples=10, seed=42)
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            assert sa.prompt_text == sb.prompt_text
            assert sa.answer_text == sb.answer_text
            assert sa.needle_line_index == sb.needle_line_index

    def test_different_seeds_produce_different_data(self):
        a = make_niah_dataset(n_samples=10, seed=42)
        b = make_niah_dataset(n_samples=10, seed=43)
        # Probability they're identical is ~0
        any_diff = any(sa.answer_text != sb.answer_text for sa, sb in zip(a, b))
        assert any_diff

    def test_zero_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            make_niah_dataset(n_samples=0)

    def test_negative_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            make_niah_dataset(n_samples=-1)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="haystack_min_lines"):
            make_niah_dataset(
                n_samples=5,
                haystack_min_lines=80, haystack_max_lines=60,
            )

    def test_too_few_lines_raises(self):
        with pytest.raises(ValueError, match="haystack_min_lines must be >= 10"):
            make_niah_dataset(
                n_samples=5,
                haystack_min_lines=5, haystack_max_lines=20,
            )

    def test_empty_prefixes_raises(self):
        with pytest.raises(ValueError, match="needle_prefixes must be non-empty"):
            make_niah_dataset(n_samples=5, needle_prefixes=())

    def test_min_code_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="needle_code_min"):
            make_niah_dataset(
                n_samples=5,
                needle_code_min=9999, needle_code_max=1000,
            )

    def test_custom_prefix_set(self):
        samples = make_niah_dataset(
            n_samples=20, seed=0,
            needle_prefixes=("FOO", "BAR"),
        )
        for s in samples:
            assert s.answer_text.startswith(("FOO-", "BAR-"))


# ---------------------------------------------------------------------------
# recall_predicate
# ---------------------------------------------------------------------------


class TestRecallPredicate:
    def _sample(self, code: str) -> NIAHSample:
        return NIAHSample(
            prompt_text="Q?",
            answer_text=code,
            needle_line_index=10,
            needle_text=f"\nIMPORTANT: code {code}.\n",
        )

    def test_exact_match(self):
        s = self._sample("ALPHA-1234")
        assert recall_predicate("The answer is ALPHA-1234.", s)

    def test_no_match(self):
        s = self._sample("ALPHA-1234")
        assert not recall_predicate("I don't know.", s)

    def test_partial_match_fails(self):
        s = self._sample("ALPHA-1234")
        assert not recall_predicate("ALPHA-1235", s)
        assert not recall_predicate("ALPHA-12", s)

    def test_case_sensitive(self):
        s = self._sample("ALPHA-1234")
        assert not recall_predicate("alpha-1234", s)

    def test_substring_anywhere(self):
        s = self._sample("BETA-5678")
        assert recall_predicate("BETA-5678 was the code", s)
        assert recall_predicate("the code was BETA-5678", s)
        assert recall_predicate("xxxBETA-5678xxx", s)

    def test_empty_decoded(self):
        s = self._sample("ALPHA-1234")
        assert not recall_predicate("", s)


# ---------------------------------------------------------------------------
# aggregate_recall
# ---------------------------------------------------------------------------


class TestAggregateRecall:
    def _samples(self, codes):
        return [
            NIAHSample(
                prompt_text="Q?", answer_text=c,
                needle_line_index=10,
                needle_text=f"code {c}",
            )
            for c in codes
        ]

    def test_all_correct(self):
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [0.1, 0.2, 0.3]
        result = aggregate_recall("test", samples, decoded, latencies)
        assert result.recall == 1.0
        assert result.samples_correct == 3
        assert result.samples_total == 3

    def test_none_correct(self):
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["X-9", "Y-8", "Z-7"]
        latencies = [0.1, 0.2, 0.3]
        result = aggregate_recall("test", samples, decoded, latencies)
        assert result.recall == 0.0
        assert result.samples_correct == 0

    def test_partial_correct(self):
        samples = self._samples(["A-1", "B-2", "C-3", "D-4"])
        decoded = ["A-1", "wrong", "C-3", "wrong"]
        latencies = [0.1, 0.2, 0.3, 0.4]
        result = aggregate_recall("test", samples, decoded, latencies)
        assert result.recall == 0.5
        assert result.samples_correct == 2
        assert result.per_sample_correct == [True, False, True, False]

    def test_mean_and_median_latency(self):
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [1.0, 2.0, 3.0]
        result = aggregate_recall("test", samples, decoded, latencies)
        assert result.mean_latency_s == 2.0
        assert result.median_latency_s == 2.0

    def test_median_latency_even_count(self):
        samples = self._samples(["A-1", "B-2", "C-3", "D-4"])
        decoded = ["A-1", "B-2", "C-3", "D-4"]
        latencies = [1.0, 2.0, 3.0, 4.0]
        result = aggregate_recall("test", samples, decoded, latencies)
        # median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5
        assert result.median_latency_s == 2.5

    def test_decoded_count_mismatch_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1"]
        latencies = [0.1, 0.2]
        with pytest.raises(ValueError, match="decoded_texts"):
            aggregate_recall("test", samples, decoded, latencies)

    def test_latency_count_mismatch_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1", "B-2"]
        latencies = [0.1]
        with pytest.raises(ValueError, match="latencies_s"):
            aggregate_recall("test", samples, decoded, latencies)

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="samples must be non-empty"):
            aggregate_recall("test", [], [], [])

    def test_per_sample_decoded_preserved(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["yes A-1 here", "no answer"]
        result = aggregate_recall("test", samples, decoded, [0.1, 0.1])
        assert result.per_sample_decoded == ["yes A-1 here", "no answer"]


# ---------------------------------------------------------------------------
# make_sink_window_4d_mask
# ---------------------------------------------------------------------------


def _allowed_positions(q: int, sink: int, window: int):
    return set(range(min(sink, q + 1))) | set(
        range(max(sink, q - window + 1), q + 1)
    )


class TestSinkWindow4DMask:
    def test_shape(self):
        m = make_sink_window_4d_mask(
            seq_len=10, sink=4, window=3,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert m.shape == (1, 1, 10, 10)

    def test_allowed_positions_zero(self):
        m = make_sink_window_4d_mask(
            seq_len=12, sink=2, window=3,
            device=torch.device("cpu"), dtype=torch.float32,
        )[0, 0]
        for q in range(12):
            for k in range(12):
                if k in _allowed_positions(q, 2, 3):
                    assert m[q, k] == 0.0, f"q={q} k={k} should be 0.0"

    def test_masked_positions_finfo_min(self):
        m = make_sink_window_4d_mask(
            seq_len=10, sink=2, window=2,
            device=torch.device("cpu"), dtype=torch.float32,
        )[0, 0]
        # Position q=5, sink=2, window=2 → allowed = {0, 1, 4, 5}
        # Masked: {2, 3, 6, 7, 8, 9}. We only check k <= q (causal).
        # In our function we do mask out k > q implicitly because the
        # window range never exceeds q.
        masked_val = m[5, 3]
        assert masked_val == torch.finfo(torch.float32).min

    def test_bf16_uses_finite_min(self):
        m = make_sink_window_4d_mask(
            seq_len=8, sink=2, window=2,
            device=torch.device("cpu"), dtype=torch.bfloat16,
        )[0, 0]
        assert m.dtype == torch.bfloat16
        masked_val = m[7, 3]
        # bf16 finfo.min is finite (not -inf), important for MPS
        # softmax which NaN-propagates from -inf in some kernels.
        assert torch.isfinite(masked_val)
        assert float(masked_val) < -1e30

    def test_softmax_zeros_forbidden_keys(self):
        seq_len, sink, window = 16, 2, 4
        torch.manual_seed(0)
        scores = torch.randn(1, 1, seq_len, seq_len)
        mask = make_sink_window_4d_mask(
            seq_len, sink, window,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        weights = torch.softmax(scores + mask, dim=-1)
        for q in range(seq_len):
            allowed = _allowed_positions(q, sink, window)
            for k in range(seq_len):
                if k not in allowed:
                    assert weights[0, 0, q, k].item() < 1e-6, (
                        f"forbidden (q={q}, k={k}) got weight "
                        f"{weights[0, 0, q, k].item()}"
                    )
            allowed_sum = sum(
                weights[0, 0, q, k].item() for k in allowed
            )
            assert abs(allowed_sum - 1.0) < 1e-4

    def test_negative_seq_len_raises(self):
        with pytest.raises(ValueError, match="must all be non-negative"):
            make_sink_window_4d_mask(
                seq_len=-1, sink=4, window=8,
                device=torch.device("cpu"), dtype=torch.float32,
            )

    def test_negative_sink_raises(self):
        with pytest.raises(ValueError, match="must all be non-negative"):
            make_sink_window_4d_mask(
                seq_len=10, sink=-1, window=8,
                device=torch.device("cpu"), dtype=torch.float32,
            )


# ---------------------------------------------------------------------------
# evaluate orchestration
# ---------------------------------------------------------------------------


class TestEvaluateOrchestration:
    def test_basic_loop_invokes_decode_per_sample(self):
        samples = make_niah_dataset(n_samples=5, seed=0)
        call_count = [0]

        def fake_decode(sample):
            call_count[0] += 1
            return sample.answer_text, 0.1  # always correct

        result = evaluate("fake", samples, fake_decode)
        assert call_count[0] == 5
        assert result.samples_total == 5
        assert result.samples_correct == 5
        assert result.recall == 1.0

    def test_partial_correctness_propagates(self):
        samples = make_niah_dataset(n_samples=10, seed=0)

        def fake_decode(sample):
            # Only odd indices get the answer right
            idx = samples.index(sample)
            if idx % 2 == 1:
                return sample.answer_text, 0.5
            return "wrong", 0.5

        result = evaluate("fake", samples, fake_decode)
        # 5 odd indices in [0..9]
        assert result.samples_correct == 5

    def test_latency_stats_computed(self):
        samples = make_niah_dataset(n_samples=3, seed=0)
        latencies = [0.1, 0.5, 1.5]
        i = [0]

        def fake_decode(sample):
            t = latencies[i[0]]
            i[0] += 1
            return sample.answer_text, t

        result = evaluate("fake", samples, fake_decode)
        assert result.mean_latency_s == sum(latencies) / 3
        assert result.median_latency_s == 0.5


# ---------------------------------------------------------------------------
# K1.G — memory tracking helpers
# ---------------------------------------------------------------------------


from inference_engine.v04.niah_eval import (
    format_memory_summary,
    record_memory,
    reset_memory_peak,
)


class TestRecordMemoryCPU:
    """CPU path is what Linux CI exercises (no CUDA on the agent VM,
    no MPS). The CPU branch optionally reads psutil RSS; if psutil
    is missing the call returns None for that field but doesn't
    raise."""

    def test_returns_dict_with_device_kind(self):
        snapshot = record_memory(torch.device("cpu"))
        assert snapshot["device_kind"] == "cpu"

    def test_peak_fields_are_none_on_cpu(self):
        snapshot = record_memory(torch.device("cpu"))
        assert snapshot["peak_allocated_bytes"] is None
        assert snapshot["peak_reserved_bytes"] is None

    def test_current_allocated_is_int_or_none(self):
        snapshot = record_memory(torch.device("cpu"))
        # Either psutil is present and we get an int RSS, or absent
        # and we get None — both shapes are valid.
        cur = snapshot["current_allocated_bytes"]
        assert cur is None or isinstance(cur, int)

    def test_snapshot_is_json_serializable(self):
        import json
        snapshot = record_memory(torch.device("cpu"))
        # Should round-trip through JSON without raising
        s = json.dumps(snapshot)
        loaded = json.loads(s)
        assert loaded["device_kind"] == "cpu"


class TestResetMemoryPeak:
    def test_cpu_reset_is_noop(self):
        # Should not raise on CPU device
        reset_memory_peak(torch.device("cpu"))


class TestFormatMemorySummary:
    def test_cuda_format_with_peak(self):
        snapshot = {
            "device_kind": "cuda",
            "peak_allocated_bytes": 4_000_000_000,
            "current_allocated_bytes": 2_000_000_000,
            "device_total_bytes": 80_000_000_000,
        }
        s = format_memory_summary(snapshot)
        assert "cuda" in s
        assert "peak=4.00GB" in s
        assert "current=2.00GB" in s
        # Percentage: 4/80 = 5%
        assert "5%" in s

    def test_cuda_format_without_total(self):
        snapshot = {
            "device_kind": "cuda",
            "peak_allocated_bytes": 4_000_000_000,
            "current_allocated_bytes": 2_000_000_000,
        }
        s = format_memory_summary(snapshot)
        assert "cuda" in s

    def test_mps_format(self):
        snapshot = {
            "device_kind": "mps",
            "current_allocated_bytes": 3_000_000_000,
            "driver_allocated_bytes": 5_000_000_000,
        }
        s = format_memory_summary(snapshot)
        assert "mps" in s
        assert "current=3.00GB" in s
        assert "driver=5.00GB" in s
        assert "no peak counter" in s

    def test_mps_format_with_none_values(self):
        snapshot = {
            "device_kind": "mps",
            "current_allocated_bytes": None,
            "driver_allocated_bytes": None,
        }
        s = format_memory_summary(snapshot)
        assert "n/a" in s

    def test_cpu_format_with_rss(self):
        snapshot = {
            "device_kind": "cpu",
            "current_allocated_bytes": 1_500_000_000,
        }
        s = format_memory_summary(snapshot)
        assert "cpu" in s
        assert "rss=1.50GB" in s

    def test_cpu_format_without_rss(self):
        snapshot = {
            "device_kind": "cpu",
            "current_allocated_bytes": None,
        }
        s = format_memory_summary(snapshot)
        # Falls through to the catch-all branch
        assert "cpu" in s

    def test_unknown_device_kind(self):
        snapshot = {
            "device_kind": "tpu",
            "current_allocated_bytes": None,
        }
        s = format_memory_summary(snapshot)
        assert "tpu" in s

    def test_summary_is_single_line(self):
        snapshot = {
            "device_kind": "cuda",
            "peak_allocated_bytes": 1_000_000_000,
            "current_allocated_bytes": 500_000_000,
            "device_total_bytes": 80_000_000_000,
        }
        s = format_memory_summary(snapshot)
        assert "\n" not in s
