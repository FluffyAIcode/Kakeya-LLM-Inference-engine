"""Mac-bridge poller — wait for and fetch a request branch's results.

Read-only on the GitHub side (uses ``gh run list`` / ``gh run view``,
both view operations) plus plain ``git fetch`` for the result commit
the runner pushes back. Suitable for Cursor cloud agents, whose ``gh``
is restricted to read-only operations.

Usage:
    python3 scripts/mac_bridge/fetch_results.py --branch mac-bridge/<name>
    python3 scripts/mac_bridge/fetch_results.py --branch ... --wait 1800

CLI plumbing; exempt from unit-test coverage by the scripts/serve.py
convention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

WORKFLOW = "mac-bridge.yaml"


def _run(argv, capture=True):
    return subprocess.run(argv, check=False, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def _latest_run(branch: str):
    proc = _run(["gh", "run", "list", "--workflow", WORKFLOW,
                 "--branch", branch, "--limit", "1",
                 "--json", "databaseId,status,conclusion,url"])
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    runs = json.loads(proc.stdout)
    return runs[0] if runs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--wait", type=int, default=0,
                    help="Seconds to keep polling (0 = single check).")
    ap.add_argument("--poll-interval", type=float, default=30.0)
    args = ap.parse_args()

    deadline = time.time() + args.wait
    while True:
        run = _latest_run(args.branch)
        if run is None:
            print(f"[mac-bridge] no {WORKFLOW} run for {args.branch} yet",
                  file=sys.stderr)
        else:
            print(f"[mac-bridge] run {run['databaseId']}: "
                  f"status={run['status']} conclusion={run['conclusion'] or '-'} "
                  f"{run['url']}", file=sys.stderr)
            if run["status"] == "completed":
                # Pull the result commit the runner pushed back.
                subprocess.run(["git", "fetch", args.remote, args.branch],
                               check=False)
                print(f"[mac-bridge] results (if any) are on "
                      f"{args.remote}/{args.branch} under .mac-bridge/logs/ "
                      f"and results/research/; inspect with:\n"
                      f"  git show {args.remote}/{args.branch} --stat",
                      file=sys.stderr)
                return 0 if run["conclusion"] == "success" else 1
        if time.time() >= deadline:
            return 3 if run is None else 2
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
