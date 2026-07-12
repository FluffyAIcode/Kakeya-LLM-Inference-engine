#!/usr/bin/env python3
"""Fill distributed prefill caches from an in-memory live gRPC capture queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path


def _request_json(url: str, *, api_key: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _node_cache(nodes: list[dict], node_id: str) -> dict:
    return next(node["cache"] for node in nodes if node["id"] == node_id)


def _ratio(cache: dict) -> float:
    total = int(cache["bytes_used"]) + int(cache["bytes_free"])
    return int(cache["bytes_used"]) / total if total else 0.0


def _memory_free_percent(ssh_target: str = "") -> int:
    command = ["memory_pressure"]
    if ssh_target:
        command = ["ssh", "-o", "ConnectTimeout=5", ssh_target, "memory_pressure"]
    output = subprocess.check_output(command, text=True, timeout=15)
    marker = "System-wide memory free percentage:"
    line = next(line for line in output.splitlines() if marker in line)
    return int(line.split(marker, 1)[1].strip().rstrip("%"))


def _safe_report_item(item: dict, *, replay_tokens: int, wall_seconds: float) -> dict:
    return {
        "capture_id": item["capture_id"],
        "captured_tokens": int(item["token_count"]),
        "replay_tokens": int(replay_tokens),
        "wall_seconds": wall_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key-file", default="~/.kakeya/network_api_key")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--head-node-id", default="head-runtime")
    parser.add_argument("--cache-node-id", default="allens-mini")
    parser.add_argument("--target-one", type=float, default=0.90)
    parser.add_argument("--target-two", type=float, default=0.95)
    parser.add_argument("--churn-gb", type=float, default=0.9)
    parser.add_argument("--pause-seconds", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--min-memory-free-percent", type=int, default=5)
    parser.add_argument("--peer-ssh", default="allen@169.254.27.104")
    parser.add_argument("--stop-file", default="/tmp/kakeya-cache-fill.stop")
    parser.add_argument("--report", default="/tmp/kakeya-cache-fill-report.json")
    args = parser.parse_args()
    if not (0 < args.target_one <= args.target_two < 1):
        raise SystemExit("targets must satisfy 0 < target-one <= target-two < 1")

    from kakeya import Client
    from kakeya.errors import ResourceExhaustedError
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    api_key = Path(args.api_key_file).expanduser().read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_ids = _resolve_eos_token_ids(tokenizer)
    stop_file = Path(args.stop_file)
    report = {
        "schema_version": 1,
        "started_at": time.time(),
        "items": [],
        "stages": [],
    }
    captured: list[dict] = []
    replay_index = 0
    baseline_fallbacks = 0
    baseline_publish_failures = 0

    def snapshot():
        nodes = _request_json(f"{args.dashboard}/v1/network/nodes", api_key=api_key)
        prefill = _request_json(f"{args.dashboard}/v1/network/prefill", api_key=api_key)
        head = _node_cache(nodes, args.head_node_id)
        peer = _node_cache(nodes, args.cache_node_id)
        return {
            "head": head,
            "peer": peer,
            "head_ratio": _ratio(head),
            "peer_ratio": _ratio(peer),
            "prefill": prefill,
        }

    def safety(current):
        if stop_file.exists():
            raise RuntimeError(f"stop file present: {stop_file}")
        if (
            int(current["prefill"].get("publish_failures", 0))
            > baseline_publish_failures
        ):
            raise RuntimeError(current["prefill"].get("last_publish_error", "publish failure"))
        if int(current["prefill"].get("fallbacks", 0)) > baseline_fallbacks:
            raise RuntimeError(current["prefill"].get("last_fallback_reason", "prefill fallback"))
        local_free = _memory_free_percent()
        peer_free = _memory_free_percent(args.peer_ssh)
        if min(local_free, peer_free) < args.min_memory_free_percent:
            raise RuntimeError(
                f"memory pressure: head={local_free}% peer={peer_free}% free",
            )

    def get_captures():
        response = _request_json(
            f"{args.dashboard}/v1/network/maintenance/capture/drain",
            api_key=api_key,
            body={"max_items": args.capture_batch},
        )
        captured.extend(response["items"])

    def replay_one():
        nonlocal replay_index
        if not captured:
            get_captures()
        if not captured:
            time.sleep(args.poll_seconds)
            return
        item = captured[replay_index % len(captured)]
        replay_index += 1
        nonce = tokenizer.encode(
            f"cache-fill-{uuid.uuid4().hex} ",
            add_special_tokens=False,
        )
        token_ids = [*nonce, *item["token_ids"]]
        started = time.perf_counter()
        try:
            with Client(args.address) as client:
                with client.create_session(
                    eos_token_ids=eos_ids,
                    client_label=f"cache-fill-{replay_index}",
                ) as session:
                    session.append(token_ids)
                    list(session.generate(max_tokens=1))
        except ResourceExhaustedError:
            time.sleep(args.poll_seconds)
            return
        report["items"].append(_safe_report_item(
            item,
            replay_tokens=len(token_ids),
            wall_seconds=time.perf_counter() - started,
        ))

    def fill_to(name: str, target: float):
        while True:
            current = snapshot()
            safety(current)
            if (
                current["head_ratio"] >= target
                and current["peer_ratio"] >= target
            ):
                report["stages"].append({
                    "name": name,
                    "completed_at": time.time(),
                    "snapshot": current,
                })
                Path(args.report).write_text(json.dumps(report, indent=2))
                return current
            replay_one()

    initial = snapshot()
    baseline_fallbacks = int(initial["prefill"].get("fallbacks", 0))
    baseline_publish_failures = int(
        initial["prefill"].get("publish_failures", 0),
    )
    report["baseline"] = initial
    fill_to("target_one", args.target_one)
    time.sleep(args.pause_seconds)
    at_target_two = fill_to("target_two", args.target_two)
    time.sleep(args.pause_seconds)
    churn_target = int(at_target_two["peer"].get("bytes_evicted", 0)) + int(
        args.churn_gb * (1 << 30),
    )
    while True:
        current = snapshot()
        safety(current)
        if int(current["peer"].get("bytes_evicted", 0)) >= churn_target:
            report["stages"].append({
                "name": "lru_churn",
                "completed_at": time.time(),
                "snapshot": current,
            })
            break
        replay_one()
    report["finished_at"] = time.time()
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "ok": True,
        "report": args.report,
        "replays": len(report["items"]),
        "final": report["stages"][-1],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
