"""Agentic workload benchmarks.

This package houses benchmark scripts that measure Kakeya under the
workload shapes that ADR 0006 identifies as our primary positioning:
multi-agent concurrent execution, long-running sessions, tool-call
reliability, mid-stream cancellation, cross-session memory recall.

Each script is a standalone CLI that:
  * targets a running Kakeya server (and optionally a parallel
    mlx_lm.server for comparison)
  * emits a structured JSON report under
    results/platform-tests/bench_agentic_<name>_<ts>.json
  * prints a tabular summary suitable for v0.3.0 release notes

Scripts in this package are NOT part of the unit-test suite. They are
end-to-end benchmarks that run on real models against real HTTP
endpoints. Run manually on the target hardware (Mac M-series for
Apple Silicon claims, CUDA box for CUDA claims).
"""
