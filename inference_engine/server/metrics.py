"""Prometheus metrics for the HTTP serving stack.

Collects per-process counters / gauges / histograms describing
admission, generation, and pool occupancy. Exposed as plain text via
``GET /metrics`` (Prometheus exposition format 0.0.4 — the same shape
``prometheus_client.exposition.generate_latest`` emits).

Why per-app registry, not the global one
----------------------------------------

``prometheus_client`` ships a process-wide :data:`REGISTRY` singleton.
Sharing it across tests is awful: counters from test A leak into the
visible state of test B, and metric registration ("metric already
exists") fights every fixture. We therefore construct a fresh
:class:`CollectorRegistry` per :func:`Metrics` instance, store it on
``app.state.metrics``, and the ``/metrics`` route reads only that
registry. Production gets one app + one registry per process; tests
get one registry per test, with no globals to clean up.

What we collect (and why)
-------------------------

The metric set is deliberately small. Every metric here corresponds
to a real diagnosis question on the engine:

  * ``http_requests_total{method, path, status}`` — basic traffic
    accounting, plus 4xx/5xx ratios.
  * ``http_request_duration_seconds`` — request-level latency
    histogram. Buckets cover 5ms..30s, the realistic range for
    generation requests.
  * ``inference_completions_total{finish_reason}`` —
    completion vs length finish reason ratio is the
    primary alignment-quality indicator we surface.
  * ``inference_completion_tokens`` — distribution of completion
    sizes; long-tail informs max_tokens defaults.
  * ``inference_acceptance_rate`` — speculative-decoding acceptance
    histogram. The single most important "is this engine actually
    doing speculative decoding?" diagnostic.
  * ``scheduler_active_sessions`` (gauge) — how many sessions are
    currently in ADMITTED state.
  * ``scheduler_pool_in_use`` (gauge) — slabs currently held.
  * ``scheduler_pool_total`` (gauge) — total slab capacity.
  * ``scheduler_pending`` (gauge) — queued admissions under QUEUE
    policy.
  * ``scheduler_admission_total{result}`` —
    ``result=admitted|rejected``.

Histograms not included (yet) but reserved as good targets:
``time_to_first_token`` (would require streaming-side instrumentation
in the route handler), ``proposer_forward_calls``,
``verifier_forward_calls`` per request. We leave those for a
follow-up PR once we have empirical data on what we actually want
to alert on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST


# Latency histogram buckets, in seconds. Chosen for the realistic
# range of inference requests: a fast 1-token completion under 50 ms,
# a long-form Chinese answer up to ~30 s. Beyond 30 s the request is
# typically pathological and we don't need fine-grained buckets.
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0, 30.0, float("inf"),
)

# Token-count buckets — completion lengths in our setting tend to be
# < 256 for short answers, < 1024 for medium, < 4096 for long.
_TOKEN_BUCKETS = (
    1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, float("inf"),
)

# Acceptance-rate buckets. Sub-1 because acceptance is in [0, 1].
# Granular near the typical 0.05–0.3 range we observe today, less
# granular near 1.0 (rarely hit pre-alignment training).
_ACCEPTANCE_BUCKETS = (
    0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95, 1.0,
)


@dataclass
class Metrics:
    """Container for all per-app Prometheus metrics + their registry."""

    registry: CollectorRegistry
    http_requests_total: Counter
    http_request_duration_seconds: Histogram
    inference_completions_total: Counter
    inference_completion_tokens: Histogram
    inference_acceptance_rate: Histogram
    scheduler_active_sessions: Gauge
    scheduler_pool_in_use: Gauge
    scheduler_pool_total: Gauge
    scheduler_pending: Gauge
    scheduler_kv_live_bytes: Gauge
    scheduler_admission_total: Counter
    # ADR 0007 §2.10 — cross-request KV reuse observability.
    # Both ``path`` labels are first-class outcomes; neither is an
    # "error" or "fallback" (per ADR 0007 §2.4.c).
    path_selection_total: Counter
    continuation_tokens_skipped_total: Counter
    verifier_prefill_duration_seconds: Histogram
    cache_invariant_violations_total: Counter

    @classmethod
    def build(cls) -> "Metrics":
        """Construct a fresh registry + register all metrics on it.

        Each call returns a brand-new instance with no shared state.
        Tests construct one per test; production constructs one per
        :class:`FastAPI` app.
        """
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            http_requests_total=Counter(
                "http_requests_total",
                "Total HTTP requests, by method and status.",
                labelnames=["method", "path", "status"],
                registry=registry,
            ),
            http_request_duration_seconds=Histogram(
                "http_request_duration_seconds",
                "HTTP request duration in seconds.",
                labelnames=["method", "path"],
                buckets=_LATENCY_BUCKETS,
                registry=registry,
            ),
            inference_completions_total=Counter(
                "inference_completions_total",
                "Total chat completions, by finish_reason.",
                labelnames=["finish_reason"],
                registry=registry,
            ),
            inference_completion_tokens=Histogram(
                "inference_completion_tokens",
                "Tokens emitted per completion.",
                buckets=_TOKEN_BUCKETS,
                registry=registry,
            ),
            inference_acceptance_rate=Histogram(
                "inference_acceptance_rate",
                "Speculative-decoding acceptance rate per completion.",
                buckets=_ACCEPTANCE_BUCKETS,
                registry=registry,
            ),
            scheduler_active_sessions=Gauge(
                "scheduler_active_sessions",
                "Sessions in ADMITTED state.",
                registry=registry,
            ),
            scheduler_pool_in_use=Gauge(
                "scheduler_pool_in_use",
                "Slabs currently held by active sessions.",
                registry=registry,
            ),
            scheduler_pool_total=Gauge(
                "scheduler_pool_total",
                "Total slab pool capacity.",
                registry=registry,
            ),
            scheduler_pending=Gauge(
                "scheduler_pending",
                "Submissions queued for admission under QUEUE policy.",
                registry=registry,
            ),
            scheduler_kv_live_bytes=Gauge(
                "scheduler_kv_live_bytes",
                "Bytes of KV cache attributable to in-flight sessions. "
                "Reads 0 when no session is active (the verifier may "
                "still hold residual cache between turns, but that "
                "carry-over is reset on the next prefill, so it does "
                "not count as 'live' usage). Verifies the ADR 0006 §2.3 "
                "long-session memory-stability claim: bounded by the "
                "per-session sink+window configuration.",
                registry=registry,
            ),
            scheduler_admission_total=Counter(
                "scheduler_admission_total",
                "Total admission attempts, by result.",
                labelnames=["result"],
                registry=registry,
            ),
            path_selection_total=Counter(
                "path_selection_total",
                "Total path-selection decisions made by the verifier "
                "for cross-request KV cache reuse (ADR 0007 §2.4). "
                "Both 'continuation' and 'new_session' are first-class "
                "first-class outcomes; neither is an 'error' or "
                "'fallback' (§2.4.c). Healthy long-session agent "
                "workloads see continuation rate >= 95%.",
                labelnames=["path"],
                registry=registry,
            ),
            continuation_tokens_skipped_total=Counter(
                "continuation_tokens_skipped_total",
                "Cumulative prompt tokens that the continuation path "
                "did not need to re-prefill (ADR 0007 §2.10). Sums "
                "ContinuationPlan.skip_n across every continuation-"
                "path request the server has handled. The win.",
                registry=registry,
            ),
            verifier_prefill_duration_seconds=Histogram(
                "verifier_prefill_duration_seconds",
                "Wall time of the prefill phase of a single request, "
                "partitioned by path. Continuation-path histogram "
                "centers around per-incremental-token cost; "
                "new-session-path histogram tracks full-prefill cost "
                "(O(history_length)).",
                labelnames=["path"],
                buckets=(
                    0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                    1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
                ),
                registry=registry,
            ),
            cache_invariant_violations_total=Counter(
                "cache_invariant_violations_total",
                "Count of ADR 0007 §2.9 INV-1 / INV-2 detections at "
                "runtime. Should always read 0; any non-zero value is "
                "a critical operational alert (page on it).",
                labelnames=["kind"],
                registry=registry,
            ),
        )

    # ------------------------------------------------------------------
    # Convenience setters used by app.py / route handlers.
    # ------------------------------------------------------------------

    def record_http_request(self, *, method: str, path: str, status: int,
                            duration_s: float) -> None:
        self.http_requests_total.labels(
            method=method, path=path, status=str(status),
        ).inc()
        self.http_request_duration_seconds.labels(
            method=method, path=path,
        ).observe(duration_s)

    def record_admission(self, *, admitted: bool) -> None:
        self.scheduler_admission_total.labels(
            result="admitted" if admitted else "rejected"
        ).inc()

    def record_path_selection(self, *, path: str, tokens_skipped: int,
                              prefill_duration_s: float) -> None:
        """Record one path-selection decision (ADR 0007 §2.10).

        ``path`` must be ``"continuation"`` or ``"new_session"``. The
        method does not validate the label set explicitly because
        prometheus-client's ``labels()`` already raises for unknown
        labels; we want such a violation to surface loudly per the
        no-silent-failure principle.
        """
        self.path_selection_total.labels(path=path).inc()
        if tokens_skipped > 0:
            self.continuation_tokens_skipped_total.inc(tokens_skipped)
        self.verifier_prefill_duration_seconds.labels(path=path).observe(
            float(prefill_duration_s)
        )

    def record_cache_invariant_violation(self, *, kind: str) -> None:
        """Record an INV-1 or INV-2 detection (ADR 0007 §2.9).

        ``kind`` must be ``"inv1"`` or ``"inv2"``. Should never be
        called in healthy operation; any increment of this counter
        is a critical alert.
        """
        self.cache_invariant_violations_total.labels(kind=kind).inc()

    def record_completion(self, *, finish_reason: str, n_tokens: int,
                          acceptance_rate: Optional[float]) -> None:
        self.inference_completions_total.labels(
            finish_reason=finish_reason,
        ).inc()
        self.inference_completion_tokens.observe(n_tokens)
        if acceptance_rate is not None:
            self.inference_acceptance_rate.observe(
                max(0.0, min(1.0, float(acceptance_rate)))
            )

    def snapshot_scheduler(self, *, active: int, pool_in_use: int,
                           pool_total: int, pending: int,
                           kv_live_bytes: int = 0) -> None:
        self.scheduler_active_sessions.set(active)
        self.scheduler_pool_in_use.set(pool_in_use)
        self.scheduler_pool_total.set(pool_total)
        self.scheduler_pending.set(pending)
        self.scheduler_kv_live_bytes.set(kv_live_bytes)

    # ------------------------------------------------------------------
    # Exposition
    # ------------------------------------------------------------------

    def render(self) -> bytes:
        """Return the Prometheus text-format encoding of this registry."""
        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST
