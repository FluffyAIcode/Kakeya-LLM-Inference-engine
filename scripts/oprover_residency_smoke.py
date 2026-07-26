#!/usr/bin/env python3
"""One controlled, journaled OProver residency and Lean verification smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from autoresearch.prefill.model_residency import ModelResidencyScheduler
from autoresearch.prefill.oprover_advisor import OFFICIAL_REVISION


RUNTIME_LABEL = "ai.kakeya.grpc-runtime-prefill"
WATCHDOG_LABEL = "ai.kakeya.decode-watchdog"


def _launch_domain() -> str:
    return f"gui/{os.getuid()}"


def _launch_pid(label: str) -> int:
    result = subprocess.run(
        ["launchctl", "print", f"{_launch_domain()}/{label}"],
        text=True, capture_output=True, check=False,
    )
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid ="):
            try:
                return int(stripped.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _json_get(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _wait(predicate, *, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise RuntimeError(message)


class LaunchctlProcessManager:
    def __init__(self, *, model_dir: Path, port: int, quant: str) -> None:
        self.model_dir = model_dir
        self.port = port
        self.quant = quant
        self.oprover: subprocess.Popen | None = None
        self.runtime_pid = _launch_pid(RUNTIME_LABEL)
        self.runtime_plist = (
            Path.home() / "Library/LaunchAgents"
            / f"{RUNTIME_LABEL}.plist"
        )
        self.watchdog_plist = (
            Path.home() / "Library/LaunchAgents"
            / f"{WATCHDOG_LABEL}.plist"
        )

    def quiesce_and_snapshot(self) -> None:
        live = Path.home() / ".kakeya/proof_live_status.json"
        if live.is_file():
            state = json.loads(live.read_text(encoding="utf-8"))
            if state.get("execution_state") not in {"idle", "completed", "failed"}:
                raise RuntimeError("PROOF_SUPERVISOR_NOT_QUIESCENT")
            pid = int(state.get("supervisor_pid", 0) or 0)
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("PROOF_SUPERVISOR_STILL_RUNNING")

    def stop_gemma(self) -> int:
        if self.runtime_pid <= 0:
            raise RuntimeError("PRIMARY_RUNTIME_OWNERSHIP_UNKNOWN")
        subprocess.run(
            ["launchctl", "bootout", f"{_launch_domain()}/{WATCHDOG_LABEL}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        result = subprocess.run(
            ["launchctl", "bootout", f"{_launch_domain()}/{RUNTIME_LABEL}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("PRIMARY_RUNTIME_BOOTOUT_FAILED")
        _wait(
            lambda: not _port_open(51051) and _launch_pid(RUNTIME_LABEL) == 0,
            timeout=60, message="PRIMARY_RUNTIME_UNLOAD_TIMEOUT",
        )
        return self.runtime_pid

    def gemma_is_resident(self) -> bool:
        return _port_open(51051) or _launch_pid(RUNTIME_LABEL) > 0

    def load_oprover(self) -> int:
        if self.gemma_is_resident():
            raise RuntimeError("PRIMARY_RUNTIME_STILL_RESIDENT")
        env = {
            **os.environ,
            "HF_HOME": str(
                Path.home() / f".cache/kakeya/oprover-cd9ffd-{self.quant}"
            ),
            "KAKEYA_CACHE_NAMESPACE": f"oprover-cd9ffd-{self.quant}",
        }
        self.oprover = subprocess.Popen(
            [
                sys.executable, "-m", "mlx_lm", "server",
                "--model", str(self.model_dir),
                "--host", "127.0.0.1", "--port", str(self.port),
                "--temp", "1", "--top-p", "0.999",
                "--max-tokens", "4096",
                "--prompt-cache-size", "1", "--log-level", "ERROR",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        _wait(
            lambda: self.oprover is not None
            and self.oprover.poll() is None
            and _port_open(self.port),
            timeout=120, message="OPROVER_LOAD_TIMEOUT",
        )
        return int(self.oprover.pid)

    def oprover_is_resident(self) -> bool:
        return self.oprover is not None and self.oprover.poll() is None

    def unload_oprover(self, pid: int) -> None:
        if self.oprover is None or self.oprover.pid != pid:
            raise RuntimeError("OPROVER_PID_OWNERSHIP_MISMATCH")
        self.oprover.terminate()
        try:
            self.oprover.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.oprover.kill()
            self.oprover.wait(timeout=10)

    def restore_gemma(self) -> int:
        if self.oprover_is_resident():
            raise RuntimeError("OPROVER_STILL_RESIDENT")
        for plist in (self.runtime_plist, self.watchdog_plist):
            result = subprocess.run(
                ["launchctl", "bootstrap", _launch_domain(), str(plist)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError("LAUNCH_SERVICE_RESTORE_FAILED")
        _wait(
            lambda: self.gemma_healthy() and self.allens_healthy(),
            timeout=240, message="GEMMA_NETWORK_RESTORE_TIMEOUT",
        )
        pid = _launch_pid(RUNTIME_LABEL)
        if pid <= 0:
            raise RuntimeError("RESTORED_RUNTIME_PID_MISSING")
        return pid

    def gemma_healthy(self) -> bool:
        try:
            runtime = _json_get("http://127.0.0.1:8091/healthz")
            summary = _json_get("http://127.0.0.1:8090/v1/network/summary")
            nodes = _json_get("http://127.0.0.1:8090/v1/network/nodes")
        except (OSError, ValueError, urllib.error.URLError):
            return False
        primary = next(
            (node for node in nodes if node.get("id") == "head-runtime"), {}
        )
        model_ids = {
            str(model.get("model_id", "")) for model in primary.get("models", ())
        }
        return (
            runtime.get("status") == "ok"
            and int(summary.get("online_nodes", 0)) == 2
            and any("gemma-4-26B-A4B-it-mlx-4bit" in item for item in model_ids)
        )

    def allens_healthy(self) -> bool:
        try:
            nodes = _json_get("http://127.0.0.1:8090/v1/network/nodes")
        except (OSError, ValueError, urllib.error.URLError):
            return False
        allens = next(
            (node for node in nodes if node.get("id") == "allens-mini"), {}
        )
        models = json.dumps(allens.get("models", ())).casefold()
        cache = json.dumps(allens.get("cache", {})).casefold()
        return (
            allens.get("status") == "online"
            and "gemma" in models
            and "oprover" not in models
            and "oprover" not in cache
        )


def _candidate_options(text: str) -> tuple[str, ...]:
    clean = re.sub(r"(?s)<think>.*?</think>", "", text)
    clean = re.sub(r"```(?:lean4?|Lean4?)?", "", clean)
    clean = clean.replace("```", "").strip()
    options = [clean]
    if ":=" in clean:
        options.append(clean.rsplit(":=", 1)[1].strip())
    markers = tuple(match.start() for match in re.finditer(r"\bby\b", clean))
    if markers:
        options.extend(clean[index:] for index in reversed(markers))
    if clean and not markers:
        options.append("by " + clean)
    return tuple(dict.fromkeys(item.strip() for item in options if item.strip()))


def _verify_with_lean(candidate: str, project_root: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="kakeya-oprover-smoke-") as raw:
        source = Path(raw) / "Smoke.lean"
        source.write_text(
            "import Mathlib\n"
            "theorem oprover_smoke (P : Prop) (h : P) : P :=\n"
            f"{candidate}\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["lake", "env", "lean", str(source)],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        return result.returncode == 0


def _advise_and_verify(
    *,
    model_dir: Path,
    port: int,
    project_root: Path,
    server_pid: int,
    quant: str,
) -> dict:
    prompt = (
        "**Current Task:**\n"
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n"
        "theorem oprover_smoke (P : Prop) (h : P) : P := by\n"
        "```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and "
        "strategies. The plan should highlight key ideas, intermediate lemmas, "
        "and proof structures that will guide the construction of the final "
        "formal proof. Please learn from the previous failed attempt and error "
        "messages to avoid similar mistakes.\n\n"
        "**Previous Failed Attempt:**\n\n```lean4\n\n```\n\n"
        "**Error Messages:**\n\n"
    )
    body = json.dumps({
        "model": str(model_dir),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 1,
        "top_p": 0.999,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())
    latency_ms = int((time.monotonic() - started) * 1000)
    choice = payload["choices"][0]
    message = choice["message"]
    text = str(
        message.get("content")
        or message.get("reasoning_content")
        or ""
    )
    selected = next(
        (
            candidate for candidate in _candidate_options(text)
            if _verify_with_lean(candidate, project_root)
        ),
        "",
    )
    if not selected:
        raise RuntimeError(
            "OPROVER_CANDIDATE_FAILED_ISOLATED_LEAN:"
            f"length={len(text)}:"
            f"contains_by={bool(re.search(r'\\bby\\b', text))}:"
            f"finish={choice.get('finish_reason', 'unknown')}"
        )
    rss = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(server_pid)],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    return {
        "lean_verified": True,
        "candidate_sha256": hashlib.sha256(selected.encode()).hexdigest(),
        "latency_ms": latency_ms,
        "server_rss_bytes": int(rss or 0) * 1024,
        "cache_namespace": f"oprover-cd9ffd-{quant}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--quant", choices=("q4", "q5"), required=True)
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    pm = LaunchctlProcessManager(
        model_dir=model_dir, port=args.port, quant=args.quant,
    )
    scheduler = ModelResidencyScheduler(
        state_path=args.state_path,
        process_manager=pm,
        headroom_check=lambda: not pm.gemma_is_resident(),
        model_id="m-a-p/OProver-8B",
        model_revision=OFFICIAL_REVISION,
        tokenizer_id="m-a-p/OProver-8B",
        gemma_cache_namespace="gemma-local4-v1",
        oprover_cache_namespace=f"oprover-cd9ffd-{args.quant}",
    )
    if args.recover_only:
        state = scheduler.recover_only()
        print(json.dumps({
            "active_model": state.active_model,
            "phase": state.phase,
            "recovered": True,
        }, sort_keys=True))
        return 0
    result = scheduler.run_exclusive(
        lambda: _advise_and_verify(
            model_dir=model_dir,
            port=args.port,
            project_root=project_root,
            server_pid=int(pm.oprover.pid if pm.oprover else 0),
            quant=args.quant,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
