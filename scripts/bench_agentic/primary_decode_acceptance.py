#!/usr/bin/env python3
"""Standalone Mac acceptance runner for the Primary decode worker.

This runner intentionally drives public gRPC sessions for workload traffic.
Worker-only observations and fault injection are delegated to
``--worker-control-command`` using the JSON protocol documented in
``docs/primary_decode_acceptance.md``.

The full endurance mode defaults to four hours.  It is never started
implicitly: ``--mode`` is required.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from inference_engine.bench.primary_decode_acceptance import (
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
    render_junit,
)


SHORT_PROMPTS = [
    "Define a KV cache in one sentence.",
    "Name one benefit of speculative decoding.",
    "What does p95 latency mean?",
]
LONG_PROMPTS = [
    (
        "Analyze a local inference service that uses a bounded sink-window KV "
        "cache. Explain cancellation, worker recycling, checkpoint recovery, "
        "and memory-pressure failure modes. Give concrete operator checks and "
        "keep the answer below six paragraphs. "
    )
    * 8,
    (
        "Compare process isolation and thread isolation for an Apple Silicon "
        "decode runtime, including hangs, unified memory, IPC overhead, state "
        "restoration, and observability tradeoffs. "
    )
    * 10,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkerControl:
    """Stateless JSON-over-stdin adapter to decode-worker-only controls."""

    def __init__(self, command: str, timeout_s: float = 10.0) -> None:
        self._argv = shlex.split(command)
        if not self._argv:
            raise ValueError("worker control command must not be empty")
        self._timeout_s = timeout_s

    def call(self, operation: str, **payload: Any) -> dict[str, Any]:
        request = {
            "schema_version": 1,
            "operation": operation,
            "payload": payload,
        }
        completed = subprocess.run(
            self._argv,
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=self._timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"worker control {operation!r} exited {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"worker control {operation!r} returned invalid JSON"
            ) from exc
        if (
            response.get("schema_version") != 1
            or response.get("operation") != operation
        ):
            raise RuntimeError(f"worker control {operation!r} response mismatch")
        if not response.get("ok"):
            raise RuntimeError(
                f"worker control {operation!r} failed: {response.get('error')}"
            )
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"worker control {operation!r} data must be an object")
        return data

    def snapshot(self) -> dict[str, Any]:
        data = self.call("snapshot")
        required = {
            "runtime_pid",
            "worker_pid",
            "worker_restart_count",
            "process_footprint_bytes",
            "active_sessions",
            "active_generations",
        }
        missing = required.difference(data)
        if missing:
            raise RuntimeError(f"worker snapshot missing fields: {sorted(missing)}")
        return data


class Harness:
    def __init__(self, args: argparse.Namespace) -> None:
        from kakeya import Client
        from transformers import AutoTokenizer

        self.args = args
        self.Client = Client
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
        eos = self.tokenizer.eos_token_id
        self.eos_ids = [] if eos is None else [int(eos)]
        self.control = WorkerControl(
            args.worker_control_command,
            timeout_s=args.control_timeout_s,
        )

    def _tokens(self, prompt: str) -> list[int]:
        return list(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )

    def _one_session(
        self,
        prompt: str,
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        with self.Client(self.args.grpc_address) as client:
            with client.create_session(eos_token_ids=self.eos_ids) as session:
                session.append(self._tokens(prompt))
                started = time.perf_counter()
                emitted = list(session.generate(max_tokens=max_tokens))
                wall_s = time.perf_counter() - started
                total_s = float(session.last_total_duration_seconds or wall_s)
                info = session.info()
                return {
                    "emitted": emitted,
                    "wall_s": wall_s,
                    "total_s": total_s,
                    "decode_latency_s": total_s / max(len(emitted), 1),
                    "kv_live_bytes": info.kv_live_bytes,
                }

    def footprint(self) -> GateResult:
        fresh = self._wait_for(
            lambda snapshot: (
                int(snapshot["active_sessions"]) == 0
                and int(snapshot["active_generations"]) == 0
            ),
            timeout_s=self.args.settle_timeout_s,
        )
        completed = 0
        for index in range(self.args.session_count):
            self._one_session(
                SHORT_PROMPTS[index % len(SHORT_PROMPTS)],
                max_tokens=self.args.session_max_tokens,
            )
            completed += 1
        final = self._wait_for(
            lambda snapshot: int(snapshot["active_sessions"]) == 0,
            timeout_s=self.args.settle_timeout_s,
        )
        if self.args.footprint_settle_s > 0:
            time.sleep(self.args.footprint_settle_s)
            final = self.control.snapshot()
        return footprint_gate(
            fresh_bytes=int(fresh["process_footprint_bytes"]),
            final_bytes=int(final["process_footprint_bytes"]),
            sessions_completed=completed,
            required_sessions=100,
            max_growth_bytes=2 << 30,
            active_sessions=int(final["active_sessions"]),
        )

    def endurance(self) -> GateResult:
        records: list[dict[str, Any]] = []
        started = time.perf_counter()
        with self.Client(self.args.grpc_address) as client:
            with client.create_session(eos_token_ids=self.eos_ids) as session:
                turn = 0
                while time.perf_counter() - started < self.args.endurance_duration_s:
                    prompt_kind = "short" if turn % 2 == 0 else "long"
                    prompts = SHORT_PROMPTS if prompt_kind == "short" else LONG_PROMPTS
                    relative_s = time.perf_counter() - started
                    try:
                        session.append(self._tokens(prompts[turn % len(prompts)]))
                        call_started = time.perf_counter()
                        emitted = list(
                            session.generate(max_tokens=self.args.endurance_max_tokens)
                        )
                        latency_s = time.perf_counter() - call_started
                        snapshot = self.control.snapshot()
                        records.append(
                            {
                                "ok": True,
                                "t_relative_s": relative_s,
                                "latency_s": latency_s,
                                "kv_live_bytes": session.info().kv_live_bytes,
                                "process_footprint_bytes": int(
                                    snapshot["process_footprint_bytes"]
                                ),
                                "prompt_kind": prompt_kind,
                                "n_emitted": len(emitted),
                                "decode_latency_s": float(
                                    session.last_total_duration_seconds or latency_s
                                )
                                / max(len(emitted), 1),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - report every failure
                        records.append(
                            {
                                "ok": False,
                                "t_relative_s": relative_s,
                                "prompt_kind": prompt_kind,
                                "error_class": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    turn += 1
                    sleep_s = self.args.endurance_turn_spacing_s - (
                        time.perf_counter() - started - relative_s
                    )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        duration_s = time.perf_counter() - started
        return endurance_gate(
            records,
            duration_s=duration_s,
            target_duration_s=14400.0,
            max_p50_drift_s=5.0,
        )

    def disconnect(self) -> GateResult:
        client = self.Client(self.args.grpc_address)
        session = client.create_session(eos_token_ids=self.eos_ids)
        session.append(self._tokens(LONG_PROMPTS[0]))
        finished = threading.Event()

        def consume() -> None:
            try:
                for _ in session.generate(max_tokens=self.args.disconnect_max_tokens):
                    pass
            except Exception:  # noqa: BLE001 - channel closure is expected
                pass
            finally:
                finished.set()

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        self._wait_for(
            lambda snapshot: int(snapshot["active_generations"]) > 0,
            timeout_s=self.args.probe_start_timeout_s,
        )
        started = time.perf_counter()
        client.close()
        settled = self._wait_for(
            lambda snapshot: (
                int(snapshot["active_generations"]) == 0
                and int(snapshot["active_sessions"]) == 0
            ),
            timeout_s=self.args.disconnect_limit_s,
        )
        elapsed = time.perf_counter() - started
        finished.wait(timeout=1.0)
        return disconnect_gate(
            cancellation_elapsed_s=elapsed,
            active_generations=int(settled["active_generations"]),
            active_sessions=int(settled["active_sessions"]),
            limit_s=5.0,
        )

    def hang(self) -> GateResult:
        with self.Client(self.args.grpc_address) as client:
            session = client.create_session(eos_token_ids=self.eos_ids)
            session.append(self._tokens(SHORT_PROMPTS[0]))
            before = self.control.snapshot()
            injected = self.control.call(
                "inject_hang",
                phase="next_forward",
                expected_worker_pid=int(before["worker_pid"]),
            )
            thread = threading.Thread(
                target=lambda: list(
                    session.generate(
                        max_tokens=2,
                        inter_token_timeout_s=self.args.hang_limit_s + 15,
                    )
                ),
                daemon=True,
            )
            started = time.perf_counter()
            thread.start()
            after = self._wait_for(
                lambda snapshot: (
                    int(snapshot["worker_pid"]) != int(before["worker_pid"])
                    and int(snapshot["worker_restart_count"])
                    > int(before["worker_restart_count"])
                ),
                timeout_s=self.args.hang_limit_s,
            )
            elapsed = time.perf_counter() - started
        return hang_recycle_gate(
            injection_accepted=bool(injected.get("accepted")),
            recycle_elapsed_s=elapsed,
            worker_pid_before=int(before["worker_pid"]),
            worker_pid_after=int(after["worker_pid"]),
            restart_count_before=int(before["worker_restart_count"]),
            restart_count_after=int(after["worker_restart_count"]),
            limit_s=120.0,
        )

    def kv_restore(self) -> GateResult:
        data = self.control.call(
            "kv_restore_parity",
            prompt_token_ids=self._tokens(LONG_PROMPTS[1]),
        )
        return kv_restore_gate(
            baseline_first_token_id=int(data["baseline_first_token_id"]),
            restored_first_token_id=int(data["restored_first_token_id"]),
            baseline_logits_sha256=str(data["baseline_logits_sha256"]),
            restored_logits_sha256=str(data["restored_logits_sha256"]),
            restore_source=str(data["restore_source"]),
        )

    def latency(self, long_run_p50_drift_s: float) -> GateResult:
        baseline_payload = json.loads(Path(self.args.latency_baseline).read_text())
        baseline = latency_baseline_from_report(baseline_payload)
        samples = [
            self._one_session(
                SHORT_PROMPTS[index % len(SHORT_PROMPTS)],
                max_tokens=self.args.latency_max_tokens,
            )["decode_latency_s"]
            for index in range(self.args.latency_samples)
        ]
        return latency_gate(
            samples,
            baseline=baseline,
            long_run_p50_drift_s=long_run_p50_drift_s,
            max_regression_fraction=0.15,
            max_long_run_p50_drift_s=5.0,
        )

    def _wait_for(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout_s
        last: dict[str, Any] = {}
        while time.perf_counter() < deadline:
            last = self.control.snapshot()
            if predicate(last):
                return last
            time.sleep(self.args.poll_interval_s)
        raise TimeoutError(f"worker state did not settle in {timeout_s}s; last={last}")


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["all", "footprint", "endurance", "disconnect", "hang", "kv-restore", "latency"],
    )
    parser.add_argument("--grpc-address", default="127.0.0.1:50051")
    parser.add_argument("--tokenizer-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--worker-control-command", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--junit-output", required=True)
    parser.add_argument("--latency-baseline")
    parser.add_argument("--long-run-report")
    parser.add_argument("--session-count", type=int, default=100)
    parser.add_argument("--session-max-tokens", type=int, default=8)
    parser.add_argument("--endurance-duration-s", type=float, default=14400.0)
    parser.add_argument("--endurance-turn-spacing-s", type=float, default=5.0)
    parser.add_argument("--endurance-max-tokens", type=int, default=32)
    parser.add_argument("--disconnect-max-tokens", type=int, default=4096)
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--latency-max-tokens", type=int, default=32)
    parser.add_argument("--disconnect-limit-s", type=float, default=5.0)
    parser.add_argument("--hang-limit-s", type=float, default=120.0)
    parser.add_argument("--settle-timeout-s", type=float, default=30.0)
    parser.add_argument("--footprint-settle-s", type=float, default=5.0)
    parser.add_argument("--probe-start-timeout-s", type=float, default=15.0)
    parser.add_argument("--control-timeout-s", type=float, default=240.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    return parser


def _drift_from_report(path: str) -> float:
    payload = json.loads(Path(path).read_text())
    for gate in payload["gates"]:
        if gate["gate"] == "mixed_prompt_endurance":
            drift = gate["metrics"]["aggregate"]["latency_drift_p50_s"]
            if drift is None:
                raise ValueError("endurance report has no p50 drift")
            return float(drift)
    raise ValueError("endurance report is missing mixed_prompt_endurance gate")


def main() -> int:
    args = _parser().parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("Primary decode acceptance requires Apple Silicon macOS", file=sys.stderr)
        return 2
    if args.mode in {"all", "latency"} and not args.latency_baseline:
        print("--latency-baseline is required for latency mode", file=sys.stderr)
        return 2
    if args.mode == "latency" and not args.long_run_report:
        print("--long-run-report is required for standalone latency mode", file=sys.stderr)
        return 2

    started_at = _utc_now()
    started = time.perf_counter()
    gates: list[GateResult] = []
    harness = Harness(args)
    operations = {
        "footprint": harness.footprint,
        "endurance": harness.endurance,
        "disconnect": harness.disconnect,
        "hang": harness.hang,
        "kv-restore": harness.kv_restore,
    }
    selected = (
        ["footprint", "endurance", "disconnect", "hang", "kv-restore"]
        if args.mode == "all"
        else [args.mode]
    )
    endurance_result: GateResult | None = None
    for name in selected:
        if name == "latency":
            continue
        try:
            result = operations[name]()
        except Exception as exc:  # noqa: BLE001 - preserve report on hardware failure
            result = error_gate(name, exc)
        gates.append(result)
        if name == "endurance":
            endurance_result = result

    if args.mode in {"all", "latency"}:
        try:
            if endurance_result is not None:
                drift = endurance_result.metrics["aggregate"]["latency_drift_p50_s"]
                if drift is None:
                    raise ValueError("endurance run has no p50 drift")
                long_run_drift = float(drift)
            else:
                long_run_drift = _drift_from_report(args.long_run_report)
            gates.append(harness.latency(long_run_drift))
        except Exception as exc:  # noqa: BLE001
            gates.append(error_gate("decode_latency_regression", exc))

    duration_s = time.perf_counter() - started
    report = build_report(
        gates=gates,
        started_at=started_at,
        finished_at=_utc_now(),
        duration_s=duration_s,
        config={
            key: value
            for key, value in vars(args).items()
            if key != "worker_control_command"
        },
        environment={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        git=_git_metadata(),
    )
    atomic_write_json(Path(args.output), report)
    junit_path = Path(args.junit_output)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.write_text(render_junit(report) + "\n")
    print(
        f"[acceptance] status={report['status']} gates={len(gates)} "
        f"json={args.output} junit={args.junit_output}",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
