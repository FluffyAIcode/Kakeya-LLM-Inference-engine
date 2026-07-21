#!/usr/bin/env python3
"""One-shot external watchdog for the Primary decode LaunchAgent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def evaluate(
    liveness: dict,
    previous: dict,
    *,
    now: float,
    stall_seconds: float,
) -> tuple[dict, bool]:
    """Require two observations of the same stale decode token."""
    stale = (
        liveness.get("phase") == "decode"
        and now - float(liveness.get("updated_at_unix", now)) >= stall_seconds
    )
    identity = [
        liveness.get("pid"),
        liveness.get("session_id"),
        liveness.get("token_index"),
    ]
    count = (
        int(previous.get("consecutive", 0)) + 1
        if stale and previous.get("identity") == identity
        else (1 if stale else 0)
    )
    state = {"identity": identity, "consecutive": count, "checked_at_unix": now}
    return state, count >= 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liveness-file", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--unhealthy-file", required=True)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--stall-seconds", type=float, default=120.0)
    args = parser.parse_args()

    liveness_path = Path(args.liveness_file).expanduser()
    state_path = Path(args.state_file).expanduser()
    unhealthy_path = Path(args.unhealthy_file).expanduser()
    liveness = _read(liveness_path)
    previous = _read(state_path)
    state, stalled = evaluate(
        liveness,
        previous,
        now=time.time(),
        stall_seconds=args.stall_seconds,
    )
    memory_unhealthy = bool(_read(unhealthy_path))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True))
    if not stalled and not memory_unhealthy:
        return 0

    reason = "decode_stall" if stalled else "memory"
    unhealthy_path.parent.mkdir(parents=True, exist_ok=True)
    unhealthy_path.write_text(json.dumps({
        "reason": reason,
        "observed": liveness,
        "updated_at_unix": time.time(),
    }, sort_keys=True))
    subprocess.run(
        [
            "launchctl",
            "kickstart",
            "-k",
            f"gui/{os.getuid()}/{args.runtime_label}",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
