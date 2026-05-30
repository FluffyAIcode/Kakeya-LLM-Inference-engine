"""Unit tests for :mod:`inference_engine.server.metrics`."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from inference_engine.server.metrics import Metrics


@pytest.fixture
def metrics() -> Metrics:
    return Metrics.build()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_build_returns_fresh_registry():
    a = Metrics.build()
    b = Metrics.build()
    assert isinstance(a.registry, CollectorRegistry)
    assert a.registry is not b.registry


def test_build_registers_all_documented_metrics(metrics):
    """Every metric named in the module docstring is reachable by name
    on the registry. Catches accidental rename or missing
    registration."""
    expected = {
        "http_requests_total",
        "http_request_duration_seconds",
        "inference_completions_total",
        "inference_completion_tokens",
        "inference_acceptance_rate",
        "scheduler_active_sessions",
        "scheduler_pool_in_use",
        "scheduler_pool_total",
        "scheduler_pending",
        "scheduler_kv_live_bytes",
        "scheduler_admission_total",
    }
    found = set()
    for collector in metrics.registry._collector_to_names.values():
        for name in collector:
            # Strip prometheus_client suffixes (_total, _bucket, etc.)
            for suffix in ("_total", "_bucket", "_count", "_sum", "_created"):
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    if base in expected:
                        found.add(base)
            if name in expected:
                found.add(name)
    assert expected == found


# ---------------------------------------------------------------------------
# record_http_request
# ---------------------------------------------------------------------------


def test_record_http_request_increments_counter(metrics):
    metrics.record_http_request(
        method="GET", path="/healthz", status=200, duration_s=0.001,
    )
    sample = _get_metric_value(
        metrics, "http_requests_total",
        labels={"method": "GET", "path": "/healthz", "status": "200"},
    )
    assert sample == 1.0


def test_record_http_request_observes_latency_histogram(metrics):
    for _ in range(3):
        metrics.record_http_request(
            method="POST", path="/v1/x", status=200, duration_s=0.05,
        )
    count = _get_metric_value(
        metrics, "http_request_duration_seconds_count",
        labels={"method": "POST", "path": "/v1/x"},
    )
    assert count == 3.0


# ---------------------------------------------------------------------------
# record_admission
# ---------------------------------------------------------------------------


def test_record_admission_admitted(metrics):
    metrics.record_admission(admitted=True)
    metrics.record_admission(admitted=True)
    val = _get_metric_value(
        metrics, "scheduler_admission_total",
        labels={"result": "admitted"},
    )
    assert val == 2.0


def test_record_admission_rejected(metrics):
    metrics.record_admission(admitted=False)
    val = _get_metric_value(
        metrics, "scheduler_admission_total",
        labels={"result": "rejected"},
    )
    assert val == 1.0


# ---------------------------------------------------------------------------
# record_completion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_record_completion_counter(metrics, reason):
    metrics.record_completion(
        finish_reason=reason, n_tokens=10, acceptance_rate=0.5,
    )
    val = _get_metric_value(
        metrics, "inference_completions_total",
        labels={"finish_reason": reason},
    )
    assert val == 1.0


def test_record_completion_observes_token_histogram(metrics):
    metrics.record_completion(
        finish_reason="stop", n_tokens=42, acceptance_rate=0.3,
    )
    count = _get_metric_value(metrics, "inference_completion_tokens_count")
    assert count == 1.0


def test_record_completion_observes_acceptance_histogram(metrics):
    metrics.record_completion(
        finish_reason="stop", n_tokens=8, acceptance_rate=0.42,
    )
    count = _get_metric_value(metrics, "inference_acceptance_rate_count")
    assert count == 1.0


def test_record_completion_skips_acceptance_when_none(metrics):
    """acceptance_rate=None must not record into the histogram."""
    metrics.record_completion(
        finish_reason="stop", n_tokens=8, acceptance_rate=None,
    )
    count = _get_metric_value(metrics, "inference_acceptance_rate_count")
    assert count == 0.0


def test_record_completion_clamps_acceptance(metrics):
    """Out-of-range acceptance values are clamped to [0, 1]."""
    metrics.record_completion(
        finish_reason="stop", n_tokens=1, acceptance_rate=2.5,
    )
    metrics.record_completion(
        finish_reason="stop", n_tokens=1, acceptance_rate=-0.1,
    )
    # Both observations counted; the histogram saw values 1.0 and 0.0.
    count = _get_metric_value(metrics, "inference_acceptance_rate_count")
    assert count == 2.0


# ---------------------------------------------------------------------------
# snapshot_scheduler
# ---------------------------------------------------------------------------


def test_snapshot_scheduler_sets_gauges(metrics):
    metrics.snapshot_scheduler(
        active=2, pool_in_use=2, pool_total=4, pending=3,
        kv_live_bytes=12345,
    )
    assert _get_metric_value(metrics, "scheduler_active_sessions") == 2.0
    assert _get_metric_value(metrics, "scheduler_pool_in_use") == 2.0
    assert _get_metric_value(metrics, "scheduler_pool_total") == 4.0
    assert _get_metric_value(metrics, "scheduler_pending") == 3.0
    assert _get_metric_value(metrics, "scheduler_kv_live_bytes") == 12345.0


def test_snapshot_scheduler_overwrites_previous(metrics):
    metrics.snapshot_scheduler(
        active=5, pool_in_use=5, pool_total=8, pending=10,
        kv_live_bytes=99999,
    )
    metrics.snapshot_scheduler(
        active=0, pool_in_use=0, pool_total=8, pending=0,
        kv_live_bytes=0,
    )
    assert _get_metric_value(metrics, "scheduler_active_sessions") == 0.0
    assert _get_metric_value(metrics, "scheduler_pending") == 0.0
    assert _get_metric_value(metrics, "scheduler_kv_live_bytes") == 0.0


def test_snapshot_scheduler_kv_live_bytes_default_zero(metrics):
    """Calling snapshot_scheduler without kv_live_bytes still sets the
    gauge — to 0 — so a /metrics scrape never sees a stale prior
    value."""
    metrics.scheduler_kv_live_bytes.set(123456)
    metrics.snapshot_scheduler(
        active=1, pool_in_use=1, pool_total=2, pending=0,
    )
    assert _get_metric_value(metrics, "scheduler_kv_live_bytes") == 0.0


# ---------------------------------------------------------------------------
# render() / content_type
# ---------------------------------------------------------------------------


def test_render_emits_prometheus_text(metrics):
    metrics.record_http_request(
        method="GET", path="/healthz", status=200, duration_s=0.001,
    )
    body = metrics.render()
    assert isinstance(body, bytes)
    text = body.decode("utf-8")
    assert "# HELP http_requests_total" in text
    assert "# TYPE http_requests_total counter" in text
    assert 'method="GET"' in text


def test_content_type_is_prometheus_format(metrics):
    assert "text/plain" in metrics.content_type
    # Prometheus exposition format markers — version is part of the
    # content type per the spec.
    assert "version=" in metrics.content_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_metric_value(
    metrics: Metrics, name: str, *, labels: dict | None = None,
) -> float:
    """Read a metric sample directly off the registry by name + labels.

    Used by tests instead of parsing Prometheus text every time. We
    walk the collectors and match exact sample names + label sets.
    """
    target_labels = labels or {}
    for metric in metrics.registry.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(k) == v for k, v in target_labels.items()):
                return float(sample.value)
    return 0.0
