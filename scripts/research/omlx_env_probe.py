"""READ-ONLY probe: is oMLX (jundot/omlx) installed & headlessly launchable here?

oMLX is a macOS, menu-bar-managed LLM inference server with continuous batching
(BatchGenerator/BatchKVCache) and an OpenAI-compatible HTTP API — a candidate to
do the Gemma-4 *parallel* inference that vllm-mlx could not (it crashed with a
``shared_kv`` TypeError on batched Gemma-4). Before we can benchmark parallel
inference we must know: (a) is it installed, (b) is there a CLI to launch the
server headlessly (no GUI), and (c) what is the exact launch/serve syntax.

This script ONLY inspects the machine and captures ``--help`` output. It starts
no server, loads no model, and changes nothing. Output is JSON on stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional


def _run(argv: List[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        return {"argv": argv, "rc": p.returncode, "out": out[:4000]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": argv, "rc": None, "err": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    report: Dict[str, Any] = {"kind": "omlx_env_probe"}

    # 1) CLI binaries on PATH.
    bins = {name: shutil.which(name)
            for name in ("omlx", "omlx-server", "omlx", "omlx-cli")}
    report["which"] = bins

    # 2) App bundles (menu-bar app install).
    app_paths = [
        "/Applications/oMLX.app",
        "/Applications/omlx.app",
        os.path.expanduser("~/Applications/oMLX.app"),
    ]
    report["app_bundles"] = {p: os.path.isdir(p) for p in app_paths}
    # Server binary commonly shipped inside the app bundle.
    for p in app_paths:
        cand = os.path.join(p, "Contents", "Resources")
        if os.path.isdir(cand):
            try:
                report.setdefault("app_resources", {})[p] = sorted(
                    os.listdir(cand))[:40]
            except OSError:
                pass

    # 3) Homebrew / pip provenance (best-effort; ignore failures).
    report["brew_omlx"] = _run(["brew", "list", "--versions", "omlx"])
    report["pip_omlx"] = _run(["python3", "-m", "pip", "show", "omlx"])

    # 4) Capture the launch/serve CLI for whichever entrypoint exists — this is
    #    what we need to script a headless server for the parallel bench.
    entry: Optional[str] = bins.get("omlx") or bins.get("omlx-server")
    report["entrypoint"] = entry
    if entry:
        report["help"] = _run([entry, "--help"])
        report["version"] = _run([entry, "--version"])
        for sub in ("serve", "launch", "server"):
            report[f"help_{sub}"] = _run([entry, sub, "--help"])

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
