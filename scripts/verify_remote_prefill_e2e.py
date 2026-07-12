#!/usr/bin/env python3
"""Prove a remote cache hit, or explicitly require remote compute plus import."""
from __future__ import annotations

import argparse
import json
import secrets
import time
import urllib.request


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def _delta(before: dict, after: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _acceptance(
    before: dict,
    after: dict,
    *,
    minimum_prefix_tokens: int,
    require_worker: bool,
) -> tuple[bool, str]:
    remote_hit = (
        _delta(before, after, "remote_hits") >= 1
        and _delta(before, after, "tokens_reused") >= minimum_prefix_tokens
    )
    remote_compute = remote_hit and _delta(before, after, "remote_jobs") >= 1
    path = "remote_compute" if remote_compute else "remote_cache" if remote_hit else "none"
    return (remote_compute if require_worker else remote_hit), path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--minimum-prefix-tokens", type=int, default=128)
    parser.add_argument("--head-node-id", default="head-runtime")
    parser.add_argument("--require-worker", action="store_true")
    args = parser.parse_args()

    from kakeya import Client
    from transformers import AutoTokenizer

    from scripts.chat_grpc import _resolve_eos_token_ids

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    nonce = secrets.token_hex(8)
    sentence = (
        "Kakeya remote prefill verification context. "
        f"Unique run {nonce}. "
    )
    text = sentence
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    while len(token_ids) < args.minimum_prefix_tokens:
        text += sentence
        token_ids = tokenizer.encode(text, add_special_tokens=True)

    nodes = _get_json(f"{args.dashboard}/v1/network/nodes")
    workers = [
        node for node in nodes
        if node.get("prefill_worker") and node.get("status") == "online"
    ]
    cache_nodes = [
        node for node in nodes
        if (
            node["id"] != args.head_node_id
            and node.get("cache")
            and node.get("status") == "online"
        )
    ]
    if not cache_nodes or (args.require_worker and not workers):
        print(json.dumps({
            "ok": False,
            "reason": (
                "no online prefill worker capability"
                if args.require_worker and not workers
                else "no online remote cache capability"
            ),
            "nodes": nodes,
        }, indent=2))
        return 2

    before = _get_json(f"{args.dashboard}/v1/network/prefill")
    started = time.perf_counter()
    with Client(args.address) as client:
        with client.create_session(
            eos_token_ids=_resolve_eos_token_ids(tokenizer),
            client_label="remote-prefill-e2e",
        ) as session:
            session.append(token_ids)
            list(session.generate(max_tokens=1))
    elapsed = time.perf_counter() - started
    after = _get_json(f"{args.dashboard}/v1/network/prefill")

    accepted, path = _acceptance(
        before,
        after,
        minimum_prefix_tokens=args.minimum_prefix_tokens,
        require_worker=args.require_worker,
    )
    result = {
        "ok": accepted,
        "accepted_path": path,
        "cache_nodes": [node["id"] for node in cache_nodes],
        "worker_nodes": [node["id"] for node in workers],
        "prefix_tokens": len(token_ids),
        "wall_seconds": elapsed,
        "delta": {
            key: _delta(before, after, key)
            for key in (
                "remote_jobs",
                "remote_hits",
                "tokens_reused",
                "tokens_computed",
                "bytes_received",
                "remote_job_failures",
                "fallbacks",
            )
        },
        "before": before,
        "after": after,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
