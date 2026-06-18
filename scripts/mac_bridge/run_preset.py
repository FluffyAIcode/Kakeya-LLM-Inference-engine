"""Mac-bridge executor — runs ONE validated preset on the Mac runner.

Invoked by .github/workflows/mac-bridge.yaml with the manifest that the
requesting agent committed at .mac-bridge/request.json. All allowlist
and parameter validation lives in inference_engine.bridge.manifest
(unit-tested on the Linux gate); this CLI only sequences subprocesses
and tees logs.

No shell is ever involved: commands are argv lists from
``build_commands`` passed straight to ``subprocess.run``.

Usage:
    python3 scripts/mac_bridge/run_preset.py --manifest .mac-bridge/request.json
    python3 scripts/mac_bridge/run_preset.py --manifest ... --dry-run

CLI plumbing around the unit-tested manifest library; exempt from
unit-test coverage by the scripts/serve.py convention.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from inference_engine.bridge.manifest import (
    ManifestError,
    build_commands,
    parse_manifest_text,
)
from inference_engine.bridge.runner_python import (
    GATE_MODULE,
    gate_error_message,
    preset_requires_gate,
    resolve_workload_python,
    substitute_python,
    workload_python_candidates,
)

LOG_DIR = Path(".mac-bridge/logs")


def _can_import_gate_module(pybin: str) -> bool:
    """True iff interpreter ``pybin`` can import the gate module (mlx_lm)."""
    try:
        return subprocess.run(
            [pybin, "-c", f"import {GATE_MODULE}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=".mac-bridge/request.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate + print the resolved argv lists without "
                         "executing anything (used by Linux-side checks).")
    args = ap.parse_args()

    try:
        request = parse_manifest_text(Path(args.manifest).read_text())
        commands = build_commands(request, dict(os.environ))
    except (OSError, ManifestError) as exc:
        print(f"[mac-bridge] REJECTED: {exc}", file=sys.stderr)
        return 2

    print(f"[mac-bridge] preset={request.preset.name} "
          f"params={dict(request.params)} ref={request.ref} "
          f"requested_by={request.requested_by}", file=sys.stderr)
    if args.dry_run:
        for argv in commands:
            print(json.dumps(argv))
        return 0

    # Layer B — resolve a PINNED workload interpreter instead of trusting the
    # bare ``python3`` on PATH (which a reboot can repoint to a python without
    # mlx_lm). Layer C — gate: mlx-/k3- engine presets fail fast with a clear
    # message when no mlx_lm-capable interpreter exists.
    pinned = os.environ.get("KAKEYA_MAC_PYTHON")
    candidates = workload_python_candidates(os.environ)
    resolved = resolve_workload_python(
        candidates, _can_import_gate_module, pinned=pinned)
    pybin = resolved.path if resolved else "python3"
    gate_ok = bool(resolved and resolved.gate_module_ok)
    print(f"[mac-bridge] workload python={pybin} {GATE_MODULE}_ok={gate_ok} "
          f"pinned={pinned!r} candidates={candidates}", file=sys.stderr)
    if preset_requires_gate(request.preset.name) and not gate_ok:
        print(f"::error::{gate_error_message(request.preset.name, pybin)}",
              file=sys.stderr)
        return 90

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "preset": request.preset.name,
        "params": dict(request.params),
        "nonce": request.nonce,
        "commands": [],
    }
    # Make the resolved interpreter authoritative for BOTH bare-``python3``
    # commands (rewritten here) and the launcher (which reads KAKEYA_MAC_PYTHON).
    sub_env = dict(os.environ)
    sub_env["KAKEYA_MAC_PYTHON"] = pybin
    rc = 0
    for idx, argv in enumerate(commands):
        argv = substitute_python(argv, pybin)
        log_path = LOG_DIR / f"{request.preset.name}-{idx}.log"
        print(f"[mac-bridge] exec[{idx}]: {argv}", file=sys.stderr)
        t0 = time.perf_counter()
        with log_path.open("wb") as log:
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                  env=sub_env)
        elapsed = time.perf_counter() - t0
        summary["commands"].append({
            "argv": argv,
            "exit_code": proc.returncode,
            "seconds": round(elapsed, 1),
            "log": str(log_path),
        })
        print(f"[mac-bridge] exec[{idx}] exit={proc.returncode} "
              f"({elapsed:.1f}s) log={log_path}", file=sys.stderr)
        if proc.returncode != 0:
            rc = proc.returncode
            break

    # Evidence discipline (design doc C4): K3 acceptance reports produced
    # by this run must satisfy the evidence gate ON THE MAC, so a
    # non-conforming report fails the bridge run itself.
    if rc == 0 and request.preset.validate_reports:
        gate = subprocess.run(
            [sys.executable, "scripts/validate_k3_reports.py",
             "results/research"],
        )
        summary["evidence_gate_exit_code"] = gate.returncode
        if gate.returncode != 0:
            rc = gate.returncode

    summary["exit_code"] = rc
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
