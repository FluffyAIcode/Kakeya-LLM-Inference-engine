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
    aggregate_attention_window_metrics,
    aggregate_recall,
    compute_effective_attention_window,
    evaluate,
    format_attention_window_summary,
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

    @staticmethod
    def _toks(n: int, value: int = 24) -> list:
        """Helper: dummy decode_token_counts of length n."""
        return [value] * n

    def test_all_correct(self):
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [0.1, 0.2, 0.3]
        result = aggregate_recall(
            "test", samples, decoded, latencies, self._toks(3),
        )
        assert result.recall == 1.0
        assert result.samples_correct == 3
        assert result.samples_total == 3

    def test_none_correct(self):
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["X-9", "Y-8", "Z-7"]
        latencies = [0.1, 0.2, 0.3]
        result = aggregate_recall(
            "test", samples, decoded, latencies, self._toks(3),
        )
        assert result.recall == 0.0
        assert result.samples_correct == 0

    def test_partial_correct(self):
        samples = self._samples(["A-1", "B-2", "C-3", "D-4"])
        decoded = ["A-1", "wrong", "C-3", "wrong"]
        latencies = [0.1, 0.2, 0.3, 0.4]
        result = aggregate_recall(
            "test", samples, decoded, latencies, self._toks(4),
        )
        assert result.recall == 0.5
        assert result.samples_correct == 2
        assert result.per_sample_correct == [True, False, True, False]

    def test_mean_and_median_latency(self):
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [1.0, 2.0, 3.0]
        result = aggregate_recall(
            "test", samples, decoded, latencies, self._toks(3),
        )
        assert result.mean_latency_s == 2.0
        assert result.median_latency_s == 2.0

    def test_median_latency_even_count(self):
        samples = self._samples(["A-1", "B-2", "C-3", "D-4"])
        decoded = ["A-1", "B-2", "C-3", "D-4"]
        latencies = [1.0, 2.0, 3.0, 4.0]
        result = aggregate_recall(
            "test", samples, decoded, latencies, self._toks(4),
        )
        # median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5
        assert result.median_latency_s == 2.5

    def test_decoded_count_mismatch_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1"]
        latencies = [0.1, 0.2]
        with pytest.raises(ValueError, match="decoded_texts"):
            aggregate_recall(
                "test", samples, decoded, latencies, self._toks(2),
            )

    def test_latency_count_mismatch_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1", "B-2"]
        latencies = [0.1]
        with pytest.raises(ValueError, match="latencies_s"):
            aggregate_recall(
                "test", samples, decoded, latencies, self._toks(2),
            )

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="samples must be non-empty"):
            aggregate_recall("test", [], [], [], [])

    def test_per_sample_decoded_preserved(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["yes A-1 here", "no answer"]
        result = aggregate_recall(
            "test", samples, decoded, [0.1, 0.1], self._toks(2),
        )
        assert result.per_sample_decoded == ["yes A-1 here", "no answer"]


# ---------------------------------------------------------------------------
# K1.I — token-throughput accounting
# ---------------------------------------------------------------------------


class TestThroughputAccounting:
    """K1.I: aggregate_recall computes per-sample throughput as
    decode_tokens / latency_s, and aggregates mean / median / min /
    max across samples. Used by ADR 0008 §11.8 v0.4 GA acceptance
    criterion 7 (throughput floor; see ADR §11.11 K2 KakeyaLattice
    composition for the cross-config target ratios)."""

    @staticmethod
    def _samples(codes):
        return [
            NIAHSample(
                prompt_text="Q?", answer_text=c,
                needle_line_index=10, needle_text=f"code {c}",
            )
            for c in codes
        ]

    def test_basic_throughput_arithmetic(self):
        # 24 tokens in 1s → 24 tok/s for every sample
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [1.0, 1.0, 1.0]
        decode_tokens = [24, 24, 24]
        result = aggregate_recall(
            "test", samples, decoded, latencies, decode_tokens,
        )
        assert result.per_sample_throughput_tokens_per_sec == [24.0, 24.0, 24.0]
        assert result.mean_throughput_tokens_per_sec == 24.0
        assert result.median_throughput_tokens_per_sec == 24.0
        assert result.min_throughput_tokens_per_sec == 24.0
        assert result.max_throughput_tokens_per_sec == 24.0

    def test_throughput_varies_with_latency(self):
        # All 24 tokens, but different latencies → different tok/s
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [1.0, 2.0, 4.0]
        decode_tokens = [24, 24, 24]
        result = aggregate_recall(
            "test", samples, decoded, latencies, decode_tokens,
        )
        # Throughputs: 24, 12, 6
        assert result.per_sample_throughput_tokens_per_sec == [24.0, 12.0, 6.0]
        assert result.mean_throughput_tokens_per_sec == pytest.approx(14.0)
        # Median of [24, 12, 6] (sorted: [6, 12, 24]) = 12
        assert result.median_throughput_tokens_per_sec == 12.0
        assert result.min_throughput_tokens_per_sec == 6.0
        assert result.max_throughput_tokens_per_sec == 24.0

    def test_throughput_varies_with_decode_count(self):
        # EOS-terminated samples: same latency, fewer tokens → lower tok/s
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["A-1", "B-2", "C-3"]
        latencies = [1.0, 1.0, 1.0]
        # Sample 0 hit EOS at 12 tokens; sample 1 at 18; sample 2 ran full 24.
        decode_tokens = [12, 18, 24]
        result = aggregate_recall(
            "test", samples, decoded, latencies, decode_tokens,
        )
        assert result.per_sample_throughput_tokens_per_sec == [12.0, 18.0, 24.0]
        assert result.mean_throughput_tokens_per_sec == 18.0
        assert result.median_throughput_tokens_per_sec == 18.0

    def test_zero_latency_yields_zero_throughput_no_div_by_zero(self):
        # Synthetic / test: latency 0 must not raise; throughput→0
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1", "B-2"]
        latencies = [0.0, 1.0]
        decode_tokens = [24, 24]
        result = aggregate_recall(
            "test", samples, decoded, latencies, decode_tokens,
        )
        assert result.per_sample_throughput_tokens_per_sec == [0.0, 24.0]

    def test_decode_token_count_mismatch_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1", "B-2"]
        with pytest.raises(ValueError, match="decode_token_counts"):
            aggregate_recall(
                "test", samples, decoded, [0.1, 0.1], [24],
            )

    def test_negative_decode_tokens_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1", "B-2"]
        with pytest.raises(ValueError, match="non-negative"):
            aggregate_recall(
                "test", samples, decoded, [0.1, 0.1], [24, -1],
            )

    def test_negative_latency_raises(self):
        samples = self._samples(["A-1", "B-2"])
        decoded = ["A-1", "B-2"]
        with pytest.raises(ValueError, match="non-negative"):
            aggregate_recall(
                "test", samples, decoded, [0.1, -0.1], [24, 24],
            )

    def test_per_sample_decode_tokens_preserved(self):
        # Order and values must reach the result unchanged
        samples = self._samples(["A-1", "B-2", "C-3", "D-4", "E-5"])
        decoded = ["A-1", "B-2", "C-3", "D-4", "E-5"]
        decode_tokens = [10, 12, 14, 16, 18]
        result = aggregate_recall(
            "test", samples, decoded, [1.0] * 5, decode_tokens,
        )
        assert result.per_sample_decode_tokens == decode_tokens

    def test_median_with_even_count(self):
        # 4 samples: throughputs [10, 20, 30, 40] → median = (20+30)/2 = 25
        samples = self._samples(["A", "B", "C", "D"])
        decoded = list(samples[i].answer_text for i in range(4))
        result = aggregate_recall(
            "test", samples, decoded, [1.0, 1.0, 1.0, 1.0], [10, 20, 30, 40],
        )
        assert result.median_throughput_tokens_per_sec == 25.0
        assert result.mean_throughput_tokens_per_sec == 25.0

    def test_throughput_independent_of_correctness(self):
        # A wrong decode still has a real wall time and decode count;
        # throughput accounting must not depend on recall outcome.
        samples = self._samples(["A-1", "B-2", "C-3"])
        decoded = ["wrong", "wrong", "wrong"]
        result = aggregate_recall(
            "test", samples, decoded, [1.0, 2.0, 4.0], [24, 24, 24],
        )
        assert result.recall == 0.0
        assert result.per_sample_throughput_tokens_per_sec == [24.0, 12.0, 6.0]
        assert result.mean_throughput_tokens_per_sec == pytest.approx(14.0)


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
            return sample.answer_text, 0.1, 24  # always correct

        result = evaluate("fake", samples, fake_decode)
        assert call_count[0] == 5
        assert result.samples_total == 5
        assert result.samples_correct == 5
        assert result.recall == 1.0

    def test_partial_correctness_propagates(self):
        samples = make_niah_dataset(n_samples=10, seed=0)

        def fake_decode(sample):
            idx = samples.index(sample)
            if idx % 2 == 1:
                return sample.answer_text, 0.5, 24
            return "wrong", 0.5, 24

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
            return sample.answer_text, t, 24

        result = evaluate("fake", samples, fake_decode)
        assert result.mean_latency_s == sum(latencies) / 3
        assert result.median_latency_s == 0.5

    def test_decode_token_count_threaded_to_throughput(self):
        # K1.I: decode_fn's third return element must reach the
        # aggregated throughput stats unchanged.
        samples = make_niah_dataset(n_samples=4, seed=0)
        # All samples take 1.0s, generate {12, 18, 24, 24} tokens.
        # Throughputs: 12, 18, 24, 24 tok/s. Mean 19.5, median 21.
        token_counts = [12, 18, 24, 24]
        i = [0]

        def fake_decode(sample):
            tc = token_counts[i[0]]
            i[0] += 1
            return sample.answer_text, 1.0, tc

        result = evaluate("fake", samples, fake_decode)
        assert result.per_sample_decode_tokens == token_counts
        assert result.per_sample_throughput_tokens_per_sec == [
            12.0, 18.0, 24.0, 24.0,
        ]
        assert result.mean_throughput_tokens_per_sec == 19.5
        # median of [12, 18, 24, 24] = (18 + 24) / 2 = 21
        assert result.median_throughput_tokens_per_sec == 21.0
        assert result.min_throughput_tokens_per_sec == 12.0
        assert result.max_throughput_tokens_per_sec == 24.0


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


# ---------------------------------------------------------------------------
# K1.H: effective attention-window metric
# ---------------------------------------------------------------------------


class TestComputeEffectiveAttentionWindow:
    def test_oracle_returns_full_seq_len(self):
        m = compute_effective_attention_window(
            "oracle_full_attention",
            seq_len=1428, sink_size=4, window_size=64,
        )
        assert m["effective_keys_at_last_query"] == 1428
        assert m["effective_attention_fraction"] == 1.0
        assert m["structural_constraint"] == "causal"

    def test_v04_returns_full_seq_len_independent_of_local_cache(self):
        m = compute_effective_attention_window(
            "v04_dlm_restored",
            seq_len=1428, sink_size=4, window_size=64,
        )
        # Even with a tiny local cache, the structural attention
        # range is the full preceding context because evicted
        # positions are reconstructed by the dLM proposer.
        assert m["effective_keys_at_last_query"] == 1428
        assert m["effective_attention_fraction"] == 1.0
        assert "causal_with_dlm_reconstruction" in m["structural_constraint"]
        assert "sink=4" in m["structural_constraint"]
        assert "window=64" in m["structural_constraint"]

    def test_v03_capped_at_sink_plus_window_when_long(self):
        m = compute_effective_attention_window(
            "v03_sink_window",
            seq_len=1428, sink_size=4, window_size=64,
        )
        assert m["effective_keys_at_last_query"] == 68
        assert m["effective_attention_fraction"] == pytest.approx(68 / 1428)
        assert m["structural_constraint"] == "sink=4+window=64"

    def test_v03_uncapped_when_seq_len_fits(self):
        # If seq_len < sink + window, the cap doesn't bind.
        m = compute_effective_attention_window(
            "v03_sink_window",
            seq_len=32, sink_size=4, window_size=64,
        )
        assert m["effective_keys_at_last_query"] == 32
        assert m["effective_attention_fraction"] == 1.0

    def test_v03_fraction_collapses_at_long_context(self):
        # The intelligence cap of v0.3 is the headline finding —
        # at 100k context, sink+window=68 is 0.07 % coverage.
        m = compute_effective_attention_window(
            "v03_sink_window",
            seq_len=100_000, sink_size=4, window_size=64,
        )
        assert m["effective_keys_at_last_query"] == 68
        assert m["effective_attention_fraction"] == pytest.approx(
            68 / 100_000
        )

    def test_zero_seq_len_returns_zero(self):
        m = compute_effective_attention_window(
            "oracle_full_attention",
            seq_len=0, sink_size=4, window_size=64,
        )
        assert m["effective_keys_at_last_query"] == 0
        assert m["effective_attention_fraction"] == 0.0

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="unknown config_name"):
            compute_effective_attention_window(
                "kakeya_lattice",
                seq_len=100, sink_size=4, window_size=64,
            )

    def test_negative_seq_len_raises(self):
        with pytest.raises(ValueError, match="seq_len"):
            compute_effective_attention_window(
                "oracle_full_attention",
                seq_len=-1, sink_size=4, window_size=64,
            )

    def test_negative_sink_or_window_raises(self):
        with pytest.raises(ValueError, match="sink_size"):
            compute_effective_attention_window(
                "v03_sink_window",
                seq_len=100, sink_size=-1, window_size=64,
            )
        with pytest.raises(ValueError, match="window_size"):
            compute_effective_attention_window(
                "v03_sink_window",
                seq_len=100, sink_size=4, window_size=-1,
            )

    def test_returned_dict_keys_are_self_describing(self):
        m = compute_effective_attention_window(
            "oracle_full_attention",
            seq_len=10, sink_size=4, window_size=64,
        )
        # JSON evidence consumers depend on these exact keys.
        assert set(m.keys()) == {
            "config",
            "seq_len",
            "effective_keys_at_last_query",
            "effective_attention_fraction",
            "structural_constraint",
        }


class TestAggregateAttentionWindowMetrics:
    def test_oracle_aggregate_full(self):
        agg = aggregate_attention_window_metrics(
            "oracle_full_attention",
            prompt_token_lens=[100, 200, 300],
            sink_size=4, window_size=64,
        )
        assert agg["samples_total"] == 3
        assert agg["effective_keys_at_last_query_mean"] == 200.0
        assert agg["effective_keys_at_last_query_min"] == 100
        assert agg["effective_keys_at_last_query_max"] == 300
        assert agg["effective_keys_at_last_query_median"] == 200.0
        assert agg["effective_attention_fraction_mean"] == 1.0
        assert agg["effective_attention_fraction_min"] == 1.0
        assert agg["effective_attention_fraction_max"] == 1.0

    def test_v03_aggregate_capped(self):
        agg = aggregate_attention_window_metrics(
            "v03_sink_window",
            prompt_token_lens=[1000, 2000, 4000],
            sink_size=4, window_size=64,
        )
        # All three are above sink+window, so all clamp to 68.
        assert agg["effective_keys_at_last_query_mean"] == 68.0
        assert agg["effective_keys_at_last_query_min"] == 68
        assert agg["effective_keys_at_last_query_max"] == 68
        # Fractions diverge: 68/1000, 68/2000, 68/4000.
        assert agg["effective_attention_fraction_max"] == pytest.approx(
            68 / 1000
        )
        assert agg["effective_attention_fraction_min"] == pytest.approx(
            68 / 4000
        )

    def test_v04_aggregate_matches_oracle_shape(self):
        # v0.4's contract: structural attention range = oracle.
        # Only the constraint label differs.
        oracle = aggregate_attention_window_metrics(
            "oracle_full_attention",
            prompt_token_lens=[1000, 2000, 4000],
            sink_size=4, window_size=64,
        )
        v04 = aggregate_attention_window_metrics(
            "v04_dlm_restored",
            prompt_token_lens=[1000, 2000, 4000],
            sink_size=4, window_size=64,
        )
        assert v04["effective_keys_at_last_query_mean"] == oracle[
            "effective_keys_at_last_query_mean"
        ]
        assert v04["effective_attention_fraction_mean"] == oracle[
            "effective_attention_fraction_mean"
        ]
        assert "dlm_reconstruction" in v04["structural_constraint"]
        assert "dlm_reconstruction" not in oracle["structural_constraint"]

    def test_per_sample_list_preserved(self):
        agg = aggregate_attention_window_metrics(
            "v03_sink_window",
            prompt_token_lens=[100, 200],
            sink_size=4, window_size=64,
        )
        assert len(agg["per_sample"]) == 2
        assert agg["per_sample"][0]["seq_len"] == 100
        assert agg["per_sample"][1]["seq_len"] == 200
        # All entries reuse the structural_constraint label.
        for s in agg["per_sample"]:
            assert s["structural_constraint"] == "sink=4+window=64"

    def test_median_with_even_count(self):
        agg = aggregate_attention_window_metrics(
            "oracle_full_attention",
            prompt_token_lens=[100, 200, 300, 400],
            sink_size=4, window_size=64,
        )
        # Median of [100, 200, 300, 400] = 250
        assert agg["effective_keys_at_last_query_median"] == 250.0

    def test_median_with_odd_count(self):
        agg = aggregate_attention_window_metrics(
            "oracle_full_attention",
            prompt_token_lens=[100, 200, 300],
            sink_size=4, window_size=64,
        )
        assert agg["effective_keys_at_last_query_median"] == 200.0

    def test_empty_prompt_lens_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_attention_window_metrics(
                "oracle_full_attention",
                prompt_token_lens=[],
                sink_size=4, window_size=64,
            )

    def test_single_sample_aggregate(self):
        agg = aggregate_attention_window_metrics(
            "v03_sink_window",
            prompt_token_lens=[1000],
            sink_size=4, window_size=64,
        )
        assert agg["samples_total"] == 1
        assert agg["effective_keys_at_last_query_min"] == 68
        assert agg["effective_keys_at_last_query_max"] == 68
        assert agg["effective_keys_at_last_query_median"] == 68.0


class TestFormatAttentionWindowSummary:
    def test_oracle_formats_with_full_coverage(self):
        agg = aggregate_attention_window_metrics(
            "oracle_full_attention",
            prompt_token_lens=[1428, 1500, 1600],
            sink_size=4, window_size=64,
        )
        s = format_attention_window_summary(agg)
        assert "100.00%" in s
        assert "causal" in s

    def test_v03_formats_with_low_coverage(self):
        agg = aggregate_attention_window_metrics(
            "v03_sink_window",
            prompt_token_lens=[1428, 1500, 1600],
            sink_size=4, window_size=64,
        )
        s = format_attention_window_summary(agg)
        # 68 / ~1500 ≈ 4.5% — far from 100%
        assert "sink=4+window=64" in s
        assert "%" in s

    def test_v04_formats_distinct_constraint(self):
        agg = aggregate_attention_window_metrics(
            "v04_dlm_restored",
            prompt_token_lens=[1428],
            sink_size=4, window_size=64,
        )
        s = format_attention_window_summary(agg)
        assert "100.00%" in s
        assert "dlm_reconstruction" in s

    def test_summary_is_single_line(self):
        agg = aggregate_attention_window_metrics(
            "oracle_full_attention",
            prompt_token_lens=[100],
            sink_size=4, window_size=64,
        )
        s = format_attention_window_summary(agg)
        assert "\n" not in s

    def test_missing_metrics_falls_through(self):
        # Defensive: if a caller passes an incomplete dict, format
        # should not crash.
        s = format_attention_window_summary(
            {"structural_constraint": "unknown"}
        )
        assert "n/a" in s
