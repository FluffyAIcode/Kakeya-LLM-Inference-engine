"""Integration test for the `kv_cache_proposer.run_demo` CLI entry point.

The CLI body is excluded from unit-test coverage (see .coveragerc); this
test invokes it as a subprocess to ensure imports, argparse, and the
end-to-end happy path remain intact. It uses the smallest viable run
(max_new_tokens=4) so total wall-time stays under ~30 s on CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(120)
def test_run_demo_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "demo.json"
    cmd = [
        sys.executable, "-m", "kv_cache_proposer.run_demo",
        "--max-new-tokens", "4",
        "--block-size", "2",
        "--num-diffusion-steps", "2",
        "--sink-size", "4",
        "--window-size", "32",
        "--batch-size-for-amortization", "8",
        "--prompt", "Reply with exactly: OK.",
        "--results-json", str(report),
    ]
    env = {**os.environ, "PYTHONPATH": str(repo_root), "OMP_NUM_THREADS": "4"}
    proc = subprocess.run(
        cmd, env=env, cwd=str(repo_root),
        capture_output=True, text=True, check=False,
    )
    # Demo exits 0 if equivalence regime passes; this small config has
    # sink+window=36 covering the full short sequence, so it should.
    assert proc.returncode == 0, (
        f"run_demo failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-1000:]}"
    )
    assert report.exists(), "results JSON not produced"
    payload = json.loads(report.read_text())
    assert "config" in payload
    assert "speculative" in payload
    assert "baseline" in payload
    assert "net_bytes_per_token_report" in payload
    # Equivalence holds in this regime
    assert payload["net_bytes_per_token_report"]["output_exact_match"] is True
