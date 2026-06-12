"""Mac-bridge client — request a Mac run by pushing a request branch.

Implements the git-bus protocol from
docs/design/mac-bridge-cloud-agent-access.md §2.1: branch
``mac-bridge/<preset>-<nonce>`` from the workload ref, overlay the
bridge files if the ref predates them, commit the manifest at
``.mac-bridge/request.json``, push. The push triggers
.github/workflows/mac-bridge.yaml on the kakeya-mac-m4 runner; results
come back as commits on the same branch (plus workflow artifacts).

Designed for Cursor cloud agents: needs only git push permission —
no workflow-dispatch token, no VPN, no SSH key.

Usage:
    python3 scripts/mac_bridge/request_run.py --preset mlx-env-probe
    python3 scripts/mac_bridge/request_run.py --preset k3-step2-fused \
        --ref origin/some-branch --param n_samples=5 --param block_size=4
    # Inspect without pushing:
    python3 scripts/mac_bridge/request_run.py --preset mlx-env-probe --no-push

CLI plumbing around the unit-tested manifest library; exempt from
unit-test coverage by the scripts/serve.py convention.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

from inference_engine.bridge.manifest import (
    MANIFEST_PATH,
    ManifestError,
    PRESETS,
    parse_manifest,
)

# Files that must exist on the pushed branch for the bridge to work
# (`on: push` workflows execute the pushed commit's definition). When
# the workload ref predates the bridge, these are overlaid from the
# client's own checkout.
BRIDGE_FILES = (
    ".github/workflows/mac-bridge.yaml",
    "scripts/mac_bridge/run_preset.py",
    "scripts/validate_k3_reports.py",
    "inference_engine/bridge/__init__.py",
    "inference_engine/bridge/manifest.py",
    "inference_engine/bench/k3_report_gate.py",
)


def _git(*argv: str, capture: bool = False) -> str:
    proc = subprocess.run(
        ["git", *argv],
        check=True,
        stdout=subprocess.PIPE if capture else None,
        text=True,
    )
    return proc.stdout.strip() if capture else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--param", action="append", default=[],
                    metavar="K=V", help="Preset parameter; repeatable.")
    ap.add_argument("--ref", default="HEAD",
                    help="Workload ref to run against (default: HEAD).")
    ap.add_argument("--requested-by", default="cloud-agent")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--no-push", action="store_true",
                    help="Build the request branch locally but do not push "
                         "(inspection / dry runs).")
    args = ap.parse_args()

    params = {}
    for kv in args.param:
        if "=" not in kv:
            print(f"--param must be K=V, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        params[k] = v

    nonce = f"{int(time.time())}-{secrets.token_hex(3)}"
    manifest = {
        "schema_version": 1,
        "preset": args.preset,
        "params": params,
        "ref": args.ref,
        "requested_by": args.requested_by,
        "nonce": nonce,
    }
    try:
        request = parse_manifest(manifest)
    except ManifestError as exc:
        print(f"[mac-bridge] invalid request: {exc}", file=sys.stderr)
        return 2
    branch = request.branch_name

    start_point = args.ref if args.ref != "HEAD" else "HEAD"
    repo_root = Path(_git("rev-parse", "--show-toplevel", capture=True))

    # Snapshot bridge files from the CLIENT checkout before switching:
    # the workload ref may predate the bridge.
    overlay = {
        rel: (repo_root / rel).read_bytes()
        for rel in BRIDGE_FILES
        if (repo_root / rel).exists()
    }

    print(f"[mac-bridge] creating {branch} from {start_point}", file=sys.stderr)
    _git("checkout", "-b", branch, start_point)
    try:
        changed = False
        for rel, blob in overlay.items():
            dst = repo_root / rel
            if not dst.exists() or dst.read_bytes() != blob:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(blob)
                changed = True
        manifest_path = repo_root / MANIFEST_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        _git("add", "-A")
        _git("commit", "-q", "-m",
             f"mac-bridge request: {args.preset} (nonce {nonce})"
             + ("\n\n(bridge files overlaid onto pre-bridge ref)" if changed else ""))
        if args.no_push:
            print(f"[mac-bridge] built {branch} (NOT pushed; --no-push)",
                  file=sys.stderr)
        else:
            _git("push", "-u", args.remote, branch)
            print(f"[mac-bridge] pushed {branch}; the kakeya-mac-m4 runner "
                  "will pick it up. Poll with:\n"
                  f"  python3 scripts/mac_bridge/fetch_results.py --branch {branch}",
                  file=sys.stderr)
    except Exception:
        # Leave the workspace on the request branch for inspection, but
        # surface the failure loudly.
        raise
    print(branch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
