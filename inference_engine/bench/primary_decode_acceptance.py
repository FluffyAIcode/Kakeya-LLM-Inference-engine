"""Pure result and gate logic for Primary decode Mac acceptance.

The hardware runner lives in ``scripts/bench_agentic``.  This module has no
MLX, gRPC, or macOS imports so report decisions remain unit-testable in CI.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from inference_engine.bench.session_long_run import aggregate_run


SCHEMA_ID = "https://kakeya.dev/schemas/primary-decode-acceptance-v1.json"
GIB = 1 << 30


@dataclass(frozen=True)
class GateResult:
    """One independently reportable acceptance decision."""

    gate: str
    status: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "error", "skipped"}:
            raise ValueError(f"invalid gate status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gate(
    name: str,
    passed: bool,
    summary: str,
    *,
    metrics: Optional[dict[str, Any]] = None,
    thresholds: Optional[dict[str, Any]] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> GateResult:
    return GateResult(
        gate=name,
        status="passed" if passed else "failed",
        summary=summary,
        metrics=metrics or {},
        thresholds=thresholds or {},
        evidence=evidence or {},
    )


def footprint_gate(
    *,
    fresh_bytes: int,
    final_bytes: int,
    sessions_completed: int,
    required_sessions: int = 100,
    max_growth_bytes: int = 2 * GIB,
    active_sessions: int = 0,
) -> GateResult:
    """Evaluate the 100 sequential-session process-footprint gate."""
    growth = final_bytes - fresh_bytes
    passed = (
        sessions_completed == required_sessions
        and growth <= max_growth_bytes
        and active_sessions == 0
    )
    return _gate(
        "sequential_session_footprint",
        passed,
        (
            f"{sessions_completed}/{required_sessions} sessions; "
            f"footprint growth {growth} bytes; active sessions {active_sessions}"
        ),
        metrics={
            "fresh_process_footprint_bytes": fresh_bytes,
            "final_process_footprint_bytes": final_bytes,
            "growth_bytes": growth,
            "sessions_completed": sessions_completed,
            "active_sessions": active_sessions,
        },
        thresholds={
            "required_sessions": required_sessions,
            "max_growth_bytes": max_growth_bytes,
            "required_active_sessions": 0,
        },
    )


def _bucket_medians(values: list[int], bucket_count: int = 8) -> list[float]:
    if not values:
        return []
    size = max(1, math.ceil(len(values) / bucket_count))
    return [
        float(statistics.median(values[start : start + size]))
        for start in range(0, len(values), size)
    ]


def monotonic_memory_growth(
    values: Iterable[int],
    *,
    minimum_growth_bytes: int = 64 << 20,
) -> tuple[bool, list[float]]:
    """Detect sustained bucket-to-bucket footprint growth, ignoring tiny noise."""
    buckets = _bucket_medians([int(value) for value in values])
    monotonic = (
        len(buckets) >= 4
        and all(right >= left for left, right in zip(buckets, buckets[1:]))
        and buckets[-1] - buckets[0] >= minimum_growth_bytes
    )
    return monotonic, buckets


def endurance_gate(
    records: list[dict[str, Any]],
    *,
    duration_s: float,
    target_duration_s: float = 4 * 60 * 60,
    max_p50_drift_s: float = 5.0,
    minimum_memory_growth_bytes: int = 64 << 20,
) -> GateResult:
    """Evaluate mixed long/short prompt endurance records.

    ``aggregate_run`` is intentionally reused from the existing gRPC
    long-session benchmark for latency/KV aggregation.
    """
    aggregate = aggregate_run(
        records,
        duration_s=duration_s,
        drift_threshold_s=max_p50_drift_s,
    )
    successes = [record for record in records if record.get("ok")]
    prompt_kinds = {str(record.get("prompt_kind")) for record in successes}
    footprints = [
        int(record["process_footprint_bytes"])
        for record in successes
        if record.get("process_footprint_bytes") is not None
    ]
    monotonic, memory_buckets = monotonic_memory_growth(
        footprints,
        minimum_growth_bytes=minimum_memory_growth_bytes,
    )
    drift = aggregate["latency_drift_p50_s"]
    passed = (
        duration_s >= target_duration_s
        and aggregate["n_errors"] == 0
        and {"short", "long"}.issubset(prompt_kinds)
        and not monotonic
        and drift is not None
        and drift <= max_p50_drift_s
    )
    return _gate(
        "mixed_prompt_endurance",
        passed,
        (
            f"{duration_s:.1f}/{target_duration_s:.1f}s; "
            f"errors={aggregate['n_errors']}; monotonic_memory={monotonic}; "
            f"p50_drift_s={drift}"
        ),
        metrics={
            "duration_s": duration_s,
            "prompt_kinds": sorted(prompt_kinds),
            "monotonic_memory_growth": monotonic,
            "memory_bucket_medians_bytes": memory_buckets,
            "aggregate": aggregate,
        },
        thresholds={
            "target_duration_s": target_duration_s,
            "max_errors": 0,
            "required_prompt_kinds": ["short", "long"],
            "max_p50_drift_s": max_p50_drift_s,
            "minimum_monotonic_growth_bytes": minimum_memory_growth_bytes,
        },
        evidence={"records": records},
    )


def disconnect_gate(
    *,
    cancellation_elapsed_s: float,
    active_generations: int,
    active_sessions: int,
    limit_s: float = 5.0,
) -> GateResult:
    passed = (
        cancellation_elapsed_s <= limit_s
        and active_generations == 0
        and active_sessions == 0
    )
    return _gate(
        "disconnect_cancellation",
        passed,
        (
            f"settled in {cancellation_elapsed_s:.3f}s; "
            f"active_generations={active_generations}; "
            f"active_sessions={active_sessions}"
        ),
        metrics={
            "cancellation_elapsed_s": cancellation_elapsed_s,
            "active_generations": active_generations,
            "active_sessions": active_sessions,
        },
        thresholds={
            "max_cancellation_elapsed_s": limit_s,
            "required_active_generations": 0,
            "required_active_sessions": 0,
        },
    )


def hang_recycle_gate(
    *,
    injection_accepted: bool,
    recycle_elapsed_s: float,
    worker_pid_before: int,
    worker_pid_after: int,
    restart_count_before: int,
    restart_count_after: int,
    limit_s: float = 120.0,
) -> GateResult:
    recycled = (
        worker_pid_before != worker_pid_after
        and restart_count_after > restart_count_before
    )
    passed = injection_accepted and recycled and recycle_elapsed_s <= limit_s
    return _gate(
        "injected_hang_recycle",
        passed,
        (
            f"accepted={injection_accepted}; recycled={recycled}; "
            f"elapsed={recycle_elapsed_s:.3f}s"
        ),
        metrics={
            "injection_accepted": injection_accepted,
            "recycle_elapsed_s": recycle_elapsed_s,
            "worker_pid_before": worker_pid_before,
            "worker_pid_after": worker_pid_after,
            "restart_count_before": restart_count_before,
            "restart_count_after": restart_count_after,
        },
        thresholds={"max_recycle_elapsed_s": limit_s},
    )


def kv_restore_gate(
    *,
    baseline_first_token_id: int,
    restored_first_token_id: int,
    baseline_logits_sha256: str,
    restored_logits_sha256: str,
    restore_source: str,
) -> GateResult:
    token_match = baseline_first_token_id == restored_first_token_id
    logits_match = (
        bool(baseline_logits_sha256)
        and baseline_logits_sha256 == restored_logits_sha256
    )
    source_ok = restore_source == "allens_kv+proof_checkpoint"
    return _gate(
        "kv_restore_parity",
        token_match and logits_match and source_ok,
        (
            f"token_match={token_match}; logits_match={logits_match}; "
            f"restore_source={restore_source}"
        ),
        metrics={
            "baseline_first_token_id": baseline_first_token_id,
            "restored_first_token_id": restored_first_token_id,
            "token_match": token_match,
            "baseline_logits_sha256": baseline_logits_sha256,
            "restored_logits_sha256": restored_logits_sha256,
            "logits_match": logits_match,
            "restore_source": restore_source,
        },
        thresholds={
            "require_token_match": True,
            "require_logits_sha256_match": True,
            "required_restore_source": "allens_kv+proof_checkpoint",
        },
    )


def _percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = quantile * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_summary(samples_s: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in samples_s]
    return {
        "sample_count": len(values),
        "p50_decode_latency_s": _percentile(values, 0.50),
        "p95_decode_latency_s": _percentile(values, 0.95),
    }


def latency_baseline_from_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read either this harness schema or the existing MLX verifier report."""
    summary = payload.get("latency")
    if isinstance(summary, Mapping):
        return {
            "p50_decode_latency_s": float(summary["p50_decode_latency_s"]),
            "p95_decode_latency_s": float(summary["p95_decode_latency_s"]),
            "source": "primary_decode_acceptance",
        }
    mlx = payload.get("mlx")
    if isinstance(mlx, Mapping):
        count = int(mlx["n_tokens"])
        if count <= 0:
            raise ValueError("MLX benchmark baseline has no generated tokens")
        per_token = float(mlx["generation_time_s"]) / count
        return {
            "p50_decode_latency_s": per_token,
            "p95_decode_latency_s": per_token,
            "source": "bench_mlx_verifier_mean_per_token",
        }
    raise ValueError("unsupported latency baseline report")


def latency_gate(
    samples_s: Iterable[float],
    *,
    baseline: Mapping[str, Any],
    long_run_p50_drift_s: float,
    max_regression_fraction: float = 0.15,
    max_long_run_p50_drift_s: float = 5.0,
) -> GateResult:
    current = latency_summary(samples_s)
    current_p50 = current["p50_decode_latency_s"]
    current_p95 = current["p95_decode_latency_s"]
    baseline_p50 = float(baseline["p50_decode_latency_s"])
    baseline_p95 = float(baseline["p95_decode_latency_s"])
    if current_p50 is None or current_p95 is None:
        return GateResult(
            gate="decode_latency_regression",
            status="failed",
            summary="no decode latency samples",
            metrics=current,
        )
    if baseline_p50 <= 0 or baseline_p95 <= 0:
        raise ValueError("latency baseline values must be > 0")
    p50_regression = current_p50 / baseline_p50 - 1.0
    p95_regression = current_p95 / baseline_p95 - 1.0
    passed = (
        p50_regression <= max_regression_fraction
        and p95_regression <= max_regression_fraction
        and long_run_p50_drift_s <= max_long_run_p50_drift_s
    )
    return _gate(
        "decode_latency_regression",
        passed,
        (
            f"p50 regression={p50_regression:.2%}; "
            f"p95 regression={p95_regression:.2%}; "
            f"long-run drift={long_run_p50_drift_s:.3f}s"
        ),
        metrics={
            **current,
            "baseline_p50_decode_latency_s": baseline_p50,
            "baseline_p95_decode_latency_s": baseline_p95,
            "p50_regression_fraction": p50_regression,
            "p95_regression_fraction": p95_regression,
            "long_run_p50_drift_s": long_run_p50_drift_s,
            "baseline_source": baseline.get("source", "unknown"),
        },
        thresholds={
            "max_regression_fraction": max_regression_fraction,
            "max_long_run_p50_drift_s": max_long_run_p50_drift_s,
        },
    )


def error_gate(name: str, exc: BaseException) -> GateResult:
    return GateResult(
        gate=name,
        status="error",
        summary=f"{type(exc).__name__}: {exc}",
        evidence={"error_class": type(exc).__name__, "error": str(exc)},
    )


def build_report(
    *,
    gates: Iterable[GateResult],
    started_at: str,
    finished_at: str,
    duration_s: float,
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    gate_dicts = [gate.to_dict() for gate in gates]
    statuses = [gate["status"] for gate in gate_dicts]
    if any(status in {"failed", "error"} for status in statuses):
        status = "failed"
    elif gate_dicts and all(item == "passed" for item in statuses):
        status = "passed"
    else:
        status = "incomplete"
    report = {
        "$schema": SCHEMA_ID,
        "schema_version": 1,
        "kind": "primary_decode_mac_acceptance",
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": float(duration_s),
        "config": dict(config),
        "environment": dict(environment),
        "git": dict(git),
        "gates": gate_dicts,
    }
    latency = next(
        (
            gate["metrics"]
            for gate in gate_dicts
            if gate["gate"] == "decode_latency_regression"
            and gate["metrics"].get("p50_decode_latency_s") is not None
        ),
        None,
    )
    if latency is not None:
        report["latency"] = {
            "p50_decode_latency_s": latency["p50_decode_latency_s"],
            "p95_decode_latency_s": latency["p95_decode_latency_s"],
            "sample_count": latency["sample_count"],
        }
    return report


def render_junit(report: Mapping[str, Any]) -> str:
    gates = list(report["gates"])
    failures = sum(gate["status"] == "failed" for gate in gates)
    errors = sum(gate["status"] == "error" for gate in gates)
    skipped = sum(gate["status"] == "skipped" for gate in gates)
    suite = ET.Element(
        "testsuite",
        {
            "name": "primary-decode-mac-acceptance",
            "tests": str(len(gates)),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
            "time": str(report["duration_s"]),
        },
    )
    for gate in gates:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "primary_decode.acceptance",
                "name": str(gate["gate"]),
            },
        )
        if gate["status"] == "failed":
            ET.SubElement(case, "failure", {"message": gate["summary"]}).text = (
                json.dumps(gate, sort_keys=True)
            )
        elif gate["status"] == "error":
            ET.SubElement(case, "error", {"message": gate["summary"]}).text = (
                json.dumps(gate, sort_keys=True)
            )
        elif gate["status"] == "skipped":
            ET.SubElement(case, "skipped", {"message": gate["summary"]})
        ET.SubElement(case, "system-out").text = json.dumps(gate, sort_keys=True)
    return ET.tostring(suite, encoding="unicode")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
