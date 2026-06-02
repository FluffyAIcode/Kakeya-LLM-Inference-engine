"""Pure aggregation helpers for the gRPC long-session bench.

The bench script under ``scripts/bench_agentic/bench_session_long_run.py``
walks one gRPC session through many turns, recording per-turn
metrics: latency, KV bytes, history length, error / success. After
the run it calls :func:`aggregate_run` here to compute the headline
KPIs:

  * ``kv_bounded`` — does ``kv_live_bytes`` stay under a tight band
    across all turns? (ADR 0006 §2.3.a, ADR 0008 §7 G2.)
  * ``prefill_bounded`` — does per-turn latency stay flat as the
    history grows? (ADR 0008 §7 G2 prefill claim, the v0.3 GA gate
    that was a non-claim on the deprecated HTTP shim.)
  * Latency p50/p95, KV min/mean/max, n_turns, n_errors.

Splitting this out of the CLI script means the aggregation logic is
fully unit-testable and the script itself stays focused on IO. The
script also computes a 10-minute bucket breakdown for visual sanity-
check on long runs (4h+); that bucketing logic lives here too.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile, ``None`` if input is empty.

    Implemented locally instead of pulling in ``numpy`` so the bench
    has no scientific-stack dependency.
    """
    if not values:
        return None
    if not 0.0 <= pct <= 1.0:
        raise ValueError(f"pct must be in [0, 1], got {pct}")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = pct * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _kv_bounded(kv_values: List[int], *, tolerance: float = 0.10) -> Optional[bool]:
    """Returns ``True`` iff the KV-bytes series stays within
    ``tolerance`` (default 10%) of its minimum across every turn.

    Returns ``None`` when there are not enough successful turns to
    answer (≤1 sample). The tolerance is a relative band — if the
    minimum is 0 we treat that as a pathologically small denominator
    and use ``max(min, 1)`` to avoid div-by-zero, the same convention
    ``bench_long_session.py`` uses.
    """
    if len(kv_values) <= 1:
        return None
    lo = min(kv_values)
    hi = max(kv_values)
    return (hi - lo) / max(lo, 1) < tolerance


def _prefill_bounded(
    latencies: List[float],
    *,
    head_window: int = 5,
    tail_window: int = 5,
    drift_threshold_s: float = 5.0,
) -> Optional[bool]:
    """Returns ``True`` iff median per-turn latency on the LAST
    ``tail_window`` turns is within ``drift_threshold_s`` seconds of
    the median on the FIRST ``head_window`` turns.

    This is the prefill-bounded contract: a healthy session-bound
    runtime processes only the new user message per turn, so latency
    should not grow with conversation length. On the deprecated HTTP
    shim, by contrast, every turn re-prefills the full history and
    latency grows linearly — that's the failure mode this metric
    catches.

    ``None`` when the run is too short to bracket head and tail
    windows without overlap.
    """
    if len(latencies) < head_window + tail_window:
        return None
    head = latencies[:head_window]
    tail = latencies[-tail_window:]
    head_p50 = statistics.median(head)
    tail_p50 = statistics.median(tail)
    return (tail_p50 - head_p50) <= drift_threshold_s


def _latency_drift_p50_s(
    latencies: List[float],
    *,
    head_window: int = 5,
    tail_window: int = 5,
) -> Optional[float]:
    """Drift in seconds between head-window p50 and tail-window p50.

    Positive = latency grew over the run. Returns ``None`` for
    runs too short to bracket head and tail without overlap.
    """
    if len(latencies) < head_window + tail_window:
        return None
    head = latencies[:head_window]
    tail = latencies[-tail_window:]
    return float(statistics.median(tail) - statistics.median(head))


def _bucketize_10min(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Partition successful turns by their wall-clock bucket
    (10-minute granularity, indexed from 0). Each bucket reports
    ``n_turns``, p50/p95 latency, and mean kv_live_bytes — gives a
    visual sanity check of latency / memory drift across a long run.

    Empty input or all-error input returns an empty list.
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for t in turns:
        if not t.get("ok"):
            continue
        bucket_idx = int(t["t_relative_s"] // 600)
        buckets.setdefault(bucket_idx, []).append(t)

    out: List[Dict[str, Any]] = []
    for idx in sorted(buckets):
        items = buckets[idx]
        latencies = [float(t["latency_s"]) for t in items]
        kv_values = [
            int(t["kv_live_bytes"]) for t in items
            if t.get("kv_live_bytes") is not None
        ]
        out.append(
            {
                "bucket_index": idx,
                "n_turns": len(items),
                "p50_latency_s": _percentile(latencies, 0.50),
                "p95_latency_s": _percentile(latencies, 0.95),
                "mean_kv_live_bytes": (
                    statistics.mean(kv_values) if kv_values else None
                ),
            }
        )
    return out


def aggregate_run(
    turns: List[Dict[str, Any]],
    *,
    duration_s: float,
    kv_tolerance: float = 0.10,
    drift_head_window: int = 5,
    drift_tail_window: int = 5,
    drift_threshold_s: float = 5.0,
) -> Dict[str, Any]:
    """Build the aggregate report from a list of per-turn records.

    Each turn dict must carry at least:
      * ``ok`` — bool
      * ``t_relative_s`` — float, seconds since run start
      * ``latency_s`` — float (only if ``ok``)
      * ``kv_live_bytes`` — int or ``None``  (only if ``ok``)

    Returns a dict with the headline KPIs ADR 0006 §2.3.a / ADR 0008
    §7 G2 speak to: ``kv_bounded``, ``prefill_bounded``, latency
    p50/p95, kv min/mean/max, error count, 10-minute bucket break-
    down.
    """
    successes = [t for t in turns if t.get("ok")]
    errors = [t for t in turns if not t.get("ok")]

    latencies = [float(t["latency_s"]) for t in successes]
    kv_values = [
        int(t["kv_live_bytes"]) for t in successes
        if t.get("kv_live_bytes") is not None
    ]

    return {
        "n_turns": len(successes),
        "n_errors": len(errors),
        "duration_s": float(duration_s),
        "p50_latency_s": _percentile(latencies, 0.50),
        "p95_latency_s": _percentile(latencies, 0.95),
        "min_kv_live_bytes": min(kv_values) if kv_values else None,
        "mean_kv_live_bytes": (
            statistics.mean(kv_values) if kv_values else None
        ),
        "max_kv_live_bytes": max(kv_values) if kv_values else None,
        "kv_bounded": _kv_bounded(kv_values, tolerance=kv_tolerance),
        "prefill_bounded": _prefill_bounded(
            latencies,
            head_window=drift_head_window,
            tail_window=drift_tail_window,
            drift_threshold_s=drift_threshold_s,
        ),
        "latency_drift_p50_s": _latency_drift_p50_s(
            latencies,
            head_window=drift_head_window,
            tail_window=drift_tail_window,
        ),
        "buckets_10min": _bucketize_10min(turns),
    }
