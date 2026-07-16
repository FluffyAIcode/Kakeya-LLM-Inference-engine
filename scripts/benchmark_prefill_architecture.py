#!/usr/bin/env python3
"""One-command three-phase benchmark for Primary decode + allens prefill."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

PHASE_KEYS = (
    "local_hits", "remote_hits", "remote_jobs", "tokens_reused",
    "tokens_computed", "bytes_received", "hot_promotions",
    "hot_promotion_bytes", "fallbacks", "remote_job_failures",
)


def _json_request(url: str, *, api_key: str = "", method: str = "GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _delta(before: dict, after: dict) -> dict:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in PHASE_KEYS
    }


def _port_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _wait_ready(host: str, port: int, *, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_ready(host, port):
            return
        time.sleep(2)
    raise RuntimeError(f"service did not become ready on {host}:{port}")


def _ensure_services(worker_ssh: str) -> None:
    uid = os.getuid()
    primary = f"gui/{uid}/ai.kakeya.grpc-runtime-prefill"
    if subprocess.run(
        ["launchctl", "print", primary],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        plist = Path.home() / "Library/LaunchAgents/ai.kakeya.grpc-runtime-prefill.plist"
        subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)], check=True)
    remote = (
        'DOMAIN="gui/$(id -u)"; '
        'launchctl print "$DOMAIN/ai.kakeya.prefill-worker" >/dev/null 2>&1 || '
        'launchctl bootstrap "$DOMAIN" '
        '"$HOME/Library/LaunchAgents/ai.kakeya.prefill-worker.plist"'
    )
    subprocess.run(["ssh", worker_ssh, remote], check=True)
    _wait_ready("127.0.0.1", 51051)
    _wait_ready("169.254.27.104", 53051)


def _restart_primary() -> None:
    label = f"gui/{os.getuid()}/ai.kakeya.grpc-runtime-prefill"
    subprocess.run(["launchctl", "kickstart", "-k", label], check=True)
    _wait_ready("127.0.0.1", 51051, timeout=180)
    _wait_ready("127.0.0.1", 8090, timeout=180)
    time.sleep(35)


def _run_stage(client, eos_ids, token_ids, output_tokens: int, get_stats) -> dict:
    before = get_stats()
    e2e_start = time.perf_counter()
    with client.create_session(eos_token_ids=eos_ids, client_label="fleet-benchmark") as s:
        append_start = time.perf_counter()
        s.append(token_ids)
        append_end = time.perf_counter()
        first_token_at = None
        count = 0
        for _token in s.generate(max_tokens=output_tokens):
            count += 1
            if first_token_at is None:
                first_token_at = time.perf_counter()
        done_at = time.perf_counter()
    after = get_stats()
    first_token_at = first_token_at or done_at
    return {
        "prefix_tokens": len(token_ids),
        "output_tokens": count,
        "append_s": append_end - append_start,
        "ttft_s": first_token_at - e2e_start,
        "decode_s": max(0.0, done_at - append_end),
        "e2e_s": done_at - e2e_start,
        "delta": _delta(before, after),
    }


def _gate(name: str, stage: dict) -> tuple[bool, str]:
    delta = stage["delta"]
    if delta["fallbacks"] or delta["remote_job_failures"]:
        return False, "fallback_or_remote_failure"
    if delta["tokens_computed"] != 0:
        return False, "primary_computed_prefill"
    if name == "remote_compute":
        ok = (
            delta["remote_jobs"] >= 1
            and delta["remote_hits"] >= 1
            and delta["hot_promotions"] >= 1
        )
        return ok, "remote_worker"
    if name == "primary_hot_hit":
        ok = (
            delta["local_hits"] >= 1
            and delta["remote_hits"] == 0
            and delta["remote_jobs"] == 0
        )
        return ok, "primary_hot"
    ok = (
        delta["remote_hits"] >= 1
        and delta["remote_jobs"] == 0
        and delta["hot_promotions"] >= 1
    )
    return ok, "allens_offload"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ssh", default="allens")
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key-file", default="~/.kakeya/network_api_key")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--minimum-prefix-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--report", default="/tmp/kakeya-prefill-benchmark.json")
    parser.add_argument("--skip-ensure", action="store_true")
    args = parser.parse_args()

    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    if not args.skip_ensure:
        _ensure_services(args.worker_ssh)
    api_key = Path(args.api_key_file).expanduser().read_text().strip()
    nodes = _json_request(f"{args.dashboard}/v1/network/nodes")
    worker = next(
        (node for node in nodes if node["id"] == "allens-mini"),
        None,
    )
    if not worker or not worker.get("prefill_worker"):
        raise SystemExit("allens prefill worker capability is not online")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    nonce = uuid.uuid4().hex
    sentence = f"Distributed prefill benchmark run {nonce}. "
    text = sentence
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    while len(token_ids) < args.minimum_prefix_tokens:
        text += sentence
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    prefix_id = hashlib.sha256(
        os.urandom(32)
        + b"".join(int(t).to_bytes(4, "little") for t in token_ids)
    ).hexdigest()

    run = _json_request(
        f"{args.dashboard}/v1/network/benchmarks",
        api_key=api_key,
        method="POST",
        body={
            "kind": "distributed_prefill_fleet_benchmark",
            "config": {
                "model_id": "gemma-4-26B-A4B-it-mlx-4bit",
                "topology": "primary-decode-allens-prefill",
                "prefill_policy": "remote-required",
                "prefix_tokens": len(token_ids),
                "output_tokens": args.output_tokens,
                "prefix_id": prefix_id,
            },
        },
    )
    run_id = run["id"]
    stages = []

    def get_stats():
        return _json_request(f"{args.dashboard}/v1/network/prefill")

    try:
        with Client(args.address) as client:
            for name in ("remote_compute", "primary_hot_hit"):
                stage = _run_stage(
                    client,
                    _resolve_eos_token_ids(tokenizer),
                    token_ids,
                    args.output_tokens,
                    get_stats,
                )
                stage["name"] = name
                stage["ok"], stage["hit_source"] = _gate(name, stage)
                stages.append(stage)
                _json_request(
                    f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                    api_key=api_key,
                    method="PATCH",
                    body={"stages": [stage]},
                )
                if not stage["ok"]:
                    raise RuntimeError(f"phase failed: {name}")
        _restart_primary()
        with Client(args.address) as client:
            stage = _run_stage(
                client,
                _resolve_eos_token_ids(tokenizer),
                token_ids,
                args.output_tokens,
                get_stats,
            )
        stage["name"] = "allens_cold_restore"
        stage["ok"], stage["hit_source"] = _gate(stage["name"], stage)
        stages.append(stage)
        if not stage["ok"]:
            raise RuntimeError("phase failed: allens_cold_restore")
        completed = _json_request(
            f"{args.dashboard}/v1/network/benchmarks/{run_id}",
            api_key=api_key,
            method="PATCH",
            body={
                "stages": [stage],
                "status": "completed",
                "finished_at": time.time(),
            },
        )
    except Exception:
        _json_request(
            f"{args.dashboard}/v1/network/benchmarks/{run_id}",
            api_key=api_key,
            method="PATCH",
            body={"status": "failed", "finished_at": time.time()},
        )
        raise

    Path(args.report).write_text(json.dumps(completed, indent=2))
    print(json.dumps({
        "ok": True,
        "run_id": run_id,
        "report": args.report,
        "summary": completed["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
