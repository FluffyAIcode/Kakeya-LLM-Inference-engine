"""Unit tests for `kv_cache_proposer.metrics`."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from transformers.cache_utils import DynamicCache

from kv_cache_proposer.metrics import (
    NetBytesPerTokenReport,
    cache_kv_bytes,
    cache_token_count,
    measure_proposer_weight_bytes,
)
from kv_cache_proposer.speculative import SpeculativeRunResult
from kv_cache_proposer.baseline import BaselineRunResult


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def test_cache_kv_bytes_empty_cache(verifier_session) -> None:
    cache = DynamicCache(config=verifier_session.model.config)
    assert cache_kv_bytes(cache) == 0
    assert cache_token_count(cache) == 0


def test_cache_kv_bytes_after_prefill(fresh_verifier_factory) -> None:
    verifier = fresh_verifier_factory(sink=4, window=64)
    verifier.prefill([1, 2, 3, 4, 5])
    bytes_ = cache_kv_bytes(verifier.cache)
    tokens = cache_token_count(verifier.cache)
    assert bytes_ > 0
    assert tokens == 5


def test_cache_token_count_handles_no_layers() -> None:
    """A `DynamicCache` constructed with `config=None` and no `ddp_cache_data`
    creates with zero layers; `cache_token_count` should return 0 cleanly."""
    cache = DynamicCache()
    assert cache_token_count(cache) == 0


def test_measure_proposer_weight_bytes(proposer_session) -> None:
    weight_bytes = measure_proposer_weight_bytes(proposer_session)
    assert weight_bytes > 0
    assert weight_bytes == proposer_session.stats.weight_bytes


# ---------------------------------------------------------------------------
# Report computation — synthetic inputs
# ---------------------------------------------------------------------------

def _mk_spec_result(**overrides) -> SpeculativeRunResult:
    base = dict(
        output_token_ids=[10, 11, 12, 13],
        accepted_per_block=[2, 1],
        proposed_per_block=[4, 4],
        proposer_forward_calls=8,
        proposer_diffusion_steps=8,
        verifier_forward_calls=4,
        verifier_tokens_consumed=20,
        proposer_peak_activation_bytes=1024 * 1024,    # 1 MB
        proposer_weight_bytes=64 * 1024 * 1024,        # 64 MB
        verifier_peak_kv_bytes=2 * 1024 * 1024,        # 2 MB
        verifier_final_kv_bytes=2 * 1024 * 1024,
        verifier_peak_activation_bytes=512 * 1024,
        verifier_weight_bytes=128 * 1024 * 1024,
        verifier_final_kv_token_count=20,
        wall_time_seconds=1.0,
    )
    base.update(overrides)
    return SpeculativeRunResult(**base)


def _mk_base_result(**overrides) -> BaselineRunResult:
    base = dict(
        output_token_ids=[10, 11, 12, 13],
        forward_calls=5,
        tokens_consumed=20,
        peak_kv_bytes=10 * 1024 * 1024,      # 10 MB total
        final_kv_bytes=10 * 1024 * 1024,
        weight_bytes=128 * 1024 * 1024,
        final_kv_token_count=100,
    )
    base.update(overrides)
    return BaselineRunResult(**base)


def test_report_assembles_all_fields() -> None:
    spec = _mk_spec_result()
    base = _mk_base_result()
    report = NetBytesPerTokenReport.compute(
        speculative=spec,
        baseline=base,
        sink_size=4,
        window_size=64,
        block_size=4,
        batch_size=8,
        verifier_peak_activation_bytes=spec.verifier_peak_activation_bytes,
    )
    # KV per token = peak_kv_bytes / final_kv_token_count
    assert report.verifier_baseline_kv_bytes_per_token == pytest.approx(
        base.peak_kv_bytes / base.final_kv_token_count
    )
    assert report.cache_budget_slots == 68
    assert report.proposer_kv_bytes_per_token == 0.0
    assert report.proposer_kv_bytes_total == 0
    assert report.compression_ratio > 0
    assert report.peak_activation_bytes_per_gpu == max(
        spec.proposer_peak_activation_bytes, spec.verifier_peak_activation_bytes
    )
    # Output equivalence
    assert report.output_exact_match is True
    assert report.output_match_prefix_length == 4
    # Acceptance rate from the spec result
    assert report.acceptance_rate == pytest.approx(3 / 8)


def test_report_detects_output_divergence() -> None:
    spec = _mk_spec_result(output_token_ids=[10, 11, 99, 13])
    base = _mk_base_result(output_token_ids=[10, 11, 12, 13])
    report = NetBytesPerTokenReport.compute(
        speculative=spec, baseline=base,
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    assert report.output_exact_match is False
    assert report.output_match_prefix_length == 2


def test_report_handles_zero_output() -> None:
    """Edge: speculative produced no tokens (e.g. immediate EOS)."""
    spec = _mk_spec_result(
        output_token_ids=[], accepted_per_block=[], proposed_per_block=[]
    )
    base = _mk_base_result(output_token_ids=[])
    report = NetBytesPerTokenReport.compute(
        speculative=spec, baseline=base,
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    assert report.acceptance_rate == 0.0
    assert report.output_exact_match is True  # both empty
    assert report.output_match_prefix_length == 0


def test_report_projection_table_shape() -> None:
    report = NetBytesPerTokenReport.compute(
        speculative=_mk_spec_result(),
        baseline=_mk_base_result(),
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    assert len(report.projection_points) == 8
    for B, S, metric, ratio in report.projection_points:
        assert B > 0 and S > 0
        assert metric > 0 and ratio > 0


def test_report_render_includes_key_fields() -> None:
    report = NetBytesPerTokenReport.compute(
        speculative=_mk_spec_result(),
        baseline=_mk_base_result(),
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    txt = report.render()
    assert "Net Bytes per Token" in txt
    assert "Compression vs baseline" in txt
    assert "exact match" in txt
    assert "Projected Net Bytes per Token" in txt
    # All projection rows are printed
    for B, S, _, _ in report.projection_points:
        assert f"{B:>4}" in txt or str(B) in txt
        assert f"{S:>10,}" in txt or f"{S:,}" in txt


def test_report_zero_seq_len_baseline_does_not_divide_by_zero() -> None:
    """The compute method clamps seq_len with `max(..., 1)`."""
    spec = _mk_spec_result(verifier_final_kv_token_count=0, output_token_ids=[])
    base = _mk_base_result(final_kv_token_count=0, output_token_ids=[])
    report = NetBytesPerTokenReport.compute(
        speculative=spec, baseline=base,
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    # Should not crash; all per-token figures defined.
    assert report.sequence_length_tokens >= 1
    assert report.net_bytes_per_token_kv_only >= 0


def test_report_is_serializable() -> None:
    """The report dataclass must serialize cleanly to JSON via asdict."""
    import json
    report = NetBytesPerTokenReport.compute(
        speculative=_mk_spec_result(),
        baseline=_mk_base_result(),
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    d = asdict(report)
    s = json.dumps(d)
    assert "compression_ratio" in s
    assert "net_bytes_per_token_kv_only" in s


def test_report_default_verifier_activation_zero() -> None:
    """If the caller omits verifier_peak_activation_bytes, it defaults to 0."""
    report = NetBytesPerTokenReport.compute(
        speculative=_mk_spec_result(),
        baseline=_mk_base_result(),
        sink_size=4, window_size=64, block_size=4, batch_size=8,
    )
    assert report.verifier_peak_activation_bytes_total == 0
