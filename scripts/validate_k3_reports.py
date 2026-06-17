"""CI walker: validate committed K3 Mac evidence reports.

Runs in the Linux gate (see ``.github/workflows/ci.yaml``). Every
committed ``results/research/*.json`` whose ``kind`` is the K3 Mac
acceptance schema is validated against the evidence rules in
:mod:`inference_engine.bench.k3_report_gate`, so a report that claims
an inadmissible speedup / recall / memory number cannot land silently.

Reports with ``schema_version < 2`` predate the gate: they are printed
as grandfathered legacy (NON-EVIDENCE) warnings and do not fail the
build — re-run the hardened harness to produce citable evidence.

Usage::

    PYTHONPATH=. python3 scripts/validate_k3_reports.py [results/research]

Exit codes: 0 = all gated reports admissible (or legacy); 1 = at least
one schema-2 report violates the evidence rules.

CLI plumbing around the unit-tested ``k3_report_gate`` library;
exempt from unit-test coverage by the same convention as
``scripts/serve.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inference_engine.bench.k3_report_gate import (
    is_gated_report,
    is_legacy_report,
    is_liveness_report,
    summarize_violations,
    validate_report,
)


def main(argv: list) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("results/research")
    if not root.exists():
        print(f"[k3-evidence-gate] {root} does not exist; nothing to check")
        return 0
    checked = legacy = failures = 0
    for path in sorted(root.rglob("*.json")):
        try:
            report = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        gated = is_gated_report(report)
        live = is_liveness_report(report)
        if not (gated or live):
            continue
        # The schema-2 legacy grandfather applies only to the NIAH acceptance
        # report; liveness reports (§4 contract) are always asserted.
        if gated and is_legacy_report(report):
            legacy += 1
            print(f"[legacy] {path}: schema<2 — grandfathered, NON-EVIDENCE "
                  "(rerun with the hardened harness to make claims)")
            continue
        checked += 1
        violations = validate_report(report)
        if violations:
            failures += 1
            print(f"[FAIL] {path}")
            print(summarize_violations(violations))
        else:
            print(f"[ok]   {path}")
    print(f"[k3-evidence-gate] checked={checked} legacy={legacy} "
          f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
