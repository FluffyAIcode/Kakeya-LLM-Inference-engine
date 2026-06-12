"""kakeya_mac — one-command cloud-agent front door to the Mac bridge.

Wraps the request/poll/fetch scripts into the three commands an agent
(or human) actually types:

    # 0. One-time sanity check of THIS environment (nothing to install;
    #    the bridge client is stdlib-only):
    python3 scripts/mac_bridge/kakeya_mac.py doctor

    # 1. Run a preset on the Mac and wait for the result:
    python3 scripts/mac_bridge/kakeya_mac.py run --preset mlx-env-probe --wait 600

    # 2. Check a request later:
    python3 scripts/mac_bridge/kakeya_mac.py status --branch <request-branch>

`run` auto-detects Cursor cloud-agent branch policy: if the current
branch looks like `AgentMemory/<name>-<suffix>`, the request branch is
created as `AgentMemory/mac-bridge-<preset>-<nonce>-<suffix>` so the
push stays inside the agent's allowed namespace (the workflow accepts
both namespaces).

CLI plumbing around request_run.py / fetch_results.py (themselves thin
wrappers over the unit-tested manifest library); exempt from unit-test
coverage by the scripts/serve.py convention.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

_AGENT_BRANCH = re.compile(r"^AgentMemory/.*?(-[a-z0-9]{4,8})$")


def _run(argv, *, capture=False, check=False):
    return subprocess.run(argv, text=True, check=check,
                          stdout=subprocess.PIPE if capture else None)


def _current_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture=True).stdout.strip()


def _branch_policy_args() -> list:
    """Stay inside a cloud agent's AgentMemory/<...>-<suffix> namespace."""
    match = _AGENT_BRANCH.match(_current_branch())
    if not match:
        return []
    suffix = match.group(1)
    return ["--branch-prefix", "AgentMemory/mac-bridge-",
            "--branch-suffix", suffix]


def cmd_doctor(_args) -> int:
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            detail = fn()
            print(f"  OK   {name}{': ' + detail if detail else ''}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")

    def _python():
        if sys.version_info < (3, 10):
            raise RuntimeError(f"python {sys.version.split()[0]} too old")
        return sys.version.split()[0]

    def _repo():
        root = _run(["git", "rev-parse", "--show-toplevel"],
                    capture=True, check=True).stdout.strip()
        if not (Path(root) / "scripts/mac_bridge/run_preset.py").exists():
            raise RuntimeError("bridge files missing on this ref")
        return root

    def _push():
        proc = subprocess.run(
            ["git", "push", "--dry-run", "origin",
             "HEAD:refs/heads/mac-bridge/doctor-probe"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError("git push --dry-run failed (no push rights?)")
        return "push permission verified (dry-run; no ref created)"

    def _gh():
        proc = _run(["gh", "auth", "status"], capture=True)
        if proc.returncode != 0:
            raise RuntimeError("gh not authenticated (status polling will "
                               "fall back to plain git fetch)")
        return "authenticated (read-only polling available)"

    def _manifest():
        sys.path.insert(0, str(SCRIPTS.parent.parent))
        from inference_engine.bridge.manifest import PRESETS
        return f"{len(PRESETS)} presets allowlisted"

    print("[kakeya-mac] doctor:")
    check("python", _python)
    check("repo + bridge files", _repo)
    check("git push permission", _push)
    check("gh (optional)", _gh)
    check("manifest allowlist import", _manifest)
    policy = _branch_policy_args()
    print(f"  OK   branch namespace: "
          f"{'AgentMemory/mac-bridge-*' if policy else 'mac-bridge/**'}")
    print(f"[kakeya-mac] {'READY' if failures == 0 else 'NOT READY'}")
    return 1 if failures else 0


def cmd_run(args) -> int:
    req = [sys.executable, str(SCRIPTS / "request_run.py"),
           "--preset", args.preset, "--requested-by", args.requested_by]
    for kv in args.param:
        req += ["--param", kv]
    if args.ref:
        req += ["--ref", args.ref]
    req += _branch_policy_args()
    if args.no_push:
        req += ["--no-push"]
    proc = _run(req, capture=True)
    sys.stderr.flush()
    branch = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    if proc.returncode != 0 or not branch:
        print("[kakeya-mac] request failed", file=sys.stderr)
        return proc.returncode or 1
    print(branch)
    if args.no_push or args.wait <= 0:
        return 0
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "fetch_results.py"),
         "--branch", branch, "--wait", str(args.wait)],
    ).returncode


def cmd_status(args) -> int:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "fetch_results.py"),
         "--branch", args.branch, "--wait", str(args.wait)],
    ).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="verify this environment can use the bridge")

    run_p = sub.add_parser("run", help="request a Mac run (optionally wait)")
    run_p.add_argument("--preset", required=True)
    run_p.add_argument("--param", action="append", default=[], metavar="K=V")
    run_p.add_argument("--ref", default="",
                       help="workload ref (default: current HEAD)")
    run_p.add_argument("--requested-by", default="kakeya-mac-cli")
    run_p.add_argument("--wait", type=int, default=0,
                       help="seconds to wait for completion (0 = fire and "
                            "forget)")
    run_p.add_argument("--no-push", action="store_true")

    st_p = sub.add_parser("status", help="poll an existing request branch")
    st_p.add_argument("--branch", required=True)
    st_p.add_argument("--wait", type=int, default=0)

    args = ap.parse_args()
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "run":
        return cmd_run(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
