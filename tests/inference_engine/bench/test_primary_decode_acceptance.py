from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from inference_engine.bench.primary_decode_acceptance import (
    GIB,
    GateResult,
    atomic_write_json,
    build_report,
    disconnect_gate,
    endurance_gate,
    error_gate,
    footprint_gate,
    hang_recycle_gate,
    kv_restore_gate,
    latency_baseline_from_report,
    latency_gate,
    latency_summary,
    monotonic_memory_growth,
    render_junit,
)


def _records(count: int = 12) -> list[dict]:
    return [
        {
            "ok": True,
            "t_relative_s": float(index),
            "latency_s": 1.0,
            "kv_live_bytes": 100,
            "process_footprint_bytes": 1000 + (index % 2),
            "prompt_kind": "short" if index % 2 == 0 else "long",
        }
        for index in range(count)
    ]


def _report(gates: list[GateResult]) -> dict:
    return build_report(
        gates=gates,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_s=1.0,
        config={"mode": "test"},
        environment={"machine": "arm64"},
        git={"commit": "abc"},
    )


def test_gate_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="invalid gate status"):
        GateResult("x", "unknown", "bad")


def test_footprint_gate_passes_and_reports_failures():
    passed = footprint_gate(
        fresh_bytes=10,
        final_bytes=20,
        sessions_completed=100,
    )
    assert passed.status == "passed"
    assert passed.metrics["growth_bytes"] == 10

    failed = footprint_gate(
        fresh_bytes=10,
        final_bytes=10 + 2 * GIB + 1,
        sessions_completed=99,
        active_sessions=1,
    )
    assert failed.status == "failed"
    assert failed.thresholds["required_sessions"] == 100


def test_monotonic_memory_growth_buckets_and_noise():
    growing, buckets = monotonic_memory_growth(
        [0, 20 << 20, 40 << 20, 80 << 20],
        minimum_growth_bytes=64 << 20,
    )
    assert growing is True
    assert len(buckets) == 4
    noisy, _ = monotonic_memory_growth([100, 99, 101, 100])
    assert noisy is False
    assert monotonic_memory_growth([]) == (False, [])


def test_endurance_gate_reuses_long_run_aggregation():
    result = endurance_gate(
        _records(),
        duration_s=10,
        target_duration_s=10,
    )
    assert result.status == "passed"
    assert result.metrics["aggregate"]["n_turns"] == 12
    assert result.metrics["prompt_kinds"] == ["long", "short"]


def test_endurance_gate_fails_incomplete_or_growing_run():
    records = _records()
    for index, record in enumerate(records):
        record["process_footprint_bytes"] = index * (64 << 20)
    records[0]["ok"] = False
    records[0]["error"] = "boom"
    result = endurance_gate(
        records,
        duration_s=9,
        target_duration_s=10,
        minimum_memory_growth_bytes=1,
    )
    assert result.status == "failed"
    assert result.metrics["monotonic_memory_growth"] is True
    assert result.metrics["aggregate"]["n_errors"] == 1


def test_disconnect_gate_requires_time_and_zero_counters():
    assert disconnect_gate(
        cancellation_elapsed_s=4.9,
        active_generations=0,
        active_sessions=0,
    ).status == "passed"
    assert disconnect_gate(
        cancellation_elapsed_s=5.1,
        active_generations=1,
        active_sessions=1,
    ).status == "failed"


def test_hang_gate_requires_pid_and_restart_counter_change():
    passed = hang_recycle_gate(
        injection_accepted=True,
        recycle_elapsed_s=119,
        worker_pid_before=1,
        worker_pid_after=2,
        restart_count_before=3,
        restart_count_after=4,
    )
    assert passed.status == "passed"
    failed = hang_recycle_gate(
        injection_accepted=False,
        recycle_elapsed_s=121,
        worker_pid_before=1,
        worker_pid_after=1,
        restart_count_before=3,
        restart_count_after=3,
    )
    assert failed.status == "failed"


def test_kv_restore_gate_requires_token_logits_and_source():
    passed = kv_restore_gate(
        baseline_first_token_id=7,
        restored_first_token_id=7,
        baseline_logits_sha256="abc",
        restored_logits_sha256="abc",
        restore_source="allens_kv+proof_checkpoint",
    )
    assert passed.status == "passed"
    failed = kv_restore_gate(
        baseline_first_token_id=7,
        restored_first_token_id=8,
        baseline_logits_sha256="",
        restored_logits_sha256="def",
        restore_source="other",
    )
    assert failed.status == "failed"
    assert failed.metrics["logits_match"] is False


def test_latency_summary_and_baseline_formats():
    assert latency_summary([])["p50_decode_latency_s"] is None
    summary = latency_summary([1, 2, 3, 4])
    assert summary == {
        "sample_count": 4,
        "p50_decode_latency_s": 2.5,
        "p95_decode_latency_s": pytest.approx(3.85),
    }
    acceptance = latency_baseline_from_report(
        {"latency": {"p50_decode_latency_s": 1, "p95_decode_latency_s": 2}}
    )
    assert acceptance["source"] == "primary_decode_acceptance"
    existing = latency_baseline_from_report(
        {"mlx": {"n_tokens": 4, "generation_time_s": 2}}
    )
    assert existing["p50_decode_latency_s"] == 0.5
    with pytest.raises(ValueError, match="no generated tokens"):
        latency_baseline_from_report(
            {"mlx": {"n_tokens": 0, "generation_time_s": 2}}
        )
    with pytest.raises(ValueError, match="unsupported"):
        latency_baseline_from_report({})


def test_latency_gate_pass_fail_and_validation():
    baseline = {
        "p50_decode_latency_s": 1.0,
        "p95_decode_latency_s": 1.0,
        "source": "test",
    }
    passed = latency_gate(
        [1.0] * 10,
        baseline=baseline,
        long_run_p50_drift_s=4.9,
    )
    assert passed.status == "passed"
    assert passed.metrics["baseline_source"] == "test"
    assert latency_gate(
        [2.0] * 10,
        baseline=baseline,
        long_run_p50_drift_s=6,
    ).status == "failed"
    assert latency_gate(
        [],
        baseline=baseline,
        long_run_p50_drift_s=0,
    ).summary == "no decode latency samples"
    with pytest.raises(ValueError, match="must be > 0"):
        latency_gate(
            [1],
            baseline={"p50_decode_latency_s": 0, "p95_decode_latency_s": 1},
            long_run_p50_drift_s=0,
        )


def test_build_report_status_and_latency_summary():
    latency = latency_gate(
        [1.0, 1.0],
        baseline={"p50_decode_latency_s": 1, "p95_decode_latency_s": 1},
        long_run_p50_drift_s=0,
    )
    passed = _report([latency])
    assert passed["status"] == "passed"
    assert passed["latency"]["sample_count"] == 2
    assert _report([GateResult("x", "failed", "no")])["status"] == "failed"
    assert _report([GateResult("x", "error", "boom")])["status"] == "failed"
    assert _report([GateResult("x", "skipped", "later")])["status"] == "incomplete"
    assert _report([])["status"] == "incomplete"


def test_error_gate_and_junit_rendering():
    gates = [
        GateResult("pass", "passed", "ok"),
        GateResult("fail", "failed", "no"),
        error_gate("error", RuntimeError("boom")),
        GateResult("skip", "skipped", "later"),
    ]
    report = _report(gates)
    xml = ET.fromstring(render_junit(report))
    assert xml.attrib["tests"] == "4"
    assert xml.attrib["failures"] == "1"
    assert xml.attrib["errors"] == "1"
    assert xml.attrib["skipped"] == "1"
    assert xml.find("./testcase[@name='fail']/failure") is not None
    assert xml.find("./testcase[@name='error']/error") is not None
    assert xml.find("./testcase[@name='skip']/skipped") is not None


def test_atomic_write_json_replaces_destination(tmp_path):
    path = tmp_path / "report.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 2}
    assert list(tmp_path.iterdir()) == [path]
