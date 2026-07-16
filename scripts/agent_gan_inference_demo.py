#!/usr/bin/env python3
"""Real Generator/Critic multi-agent inference over the two-Mac KV architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from scripts.benchmark_prefill_architecture import (
    _delta,
    _ensure_services,
    _json_request,
)


def _agent_cache_gate(warm_delta: dict, actual_delta: dict) -> bool:
    return (
        warm_delta["remote_jobs"] >= 1
        and warm_delta["remote_hits"] >= 1
        and actual_delta["local_hits"] >= 1
        and actual_delta["remote_jobs"] == 0
        and actual_delta["tokens_computed"] == 0
        and actual_delta["fallbacks"] == 0
    )


def _output_metadata(text: str) -> dict:
    return {
        "output_chars": len(text),
        "output_hash": hashlib.sha256(text.encode()).hexdigest(),
    }


def _infer(
    client,
    eos_ids,
    token_ids,
    output_tokens: int,
    get_stats,
    on_token=None,
):
    before = get_stats()
    started = time.perf_counter()
    with client.create_session(eos_token_ids=eos_ids, client_label="agent-gan") as s:
        append_started = time.perf_counter()
        s.append(token_ids)
        append_done = time.perf_counter()
        first_at = None
        generated = []
        for token in s.generate(max_tokens=output_tokens):
            generated.append(int(token))
            if on_token is not None:
                on_token(generated)
            if first_at is None:
                first_at = time.perf_counter()
        done = time.perf_counter()
    after = get_stats()
    first_at = first_at or done
    return generated, {
        "prefix_tokens": len(token_ids),
        "output_tokens": len(generated),
        "append_s": append_done - append_started,
        "ttft_s": first_at - started,
        "decode_s": done - append_done,
        "e2e_s": done - started,
        "delta": _delta(before, after),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ssh", default="allens")
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key-file", default="~/.kakeya/network_api_key")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Generator/Critic cycles. The 16GB Gemma worker supports one "
             "reliably; larger values require more worker memory or timeout.",
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--report", default="/tmp/kakeya-agent-gan-demo.json")
    parser.add_argument("--skip-ensure", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0 or args.output_tokens <= 0:
        raise SystemExit("rounds and output-tokens must be > 0")

    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    if not args.skip_ensure:
        _ensure_services(args.worker_ssh)
    api_key = Path(args.api_key_file).expanduser().read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_ids = _resolve_eos_token_ids(tokenizer)
    run_nonce = uuid.uuid4().hex
    task = (
        "Evaluate and improve the current two-Mac architecture where Primary "
        "is decode-only, allens performs prefill, Primary keeps hot KV, and "
        "allens provides cold KV offload. Produce concrete correctness, "
        "throughput, memory, and failure-mode recommendations. "
        f"Evaluation run {run_nonce}."
    )
    generator_history = [
        {
            "role": "system",
            "content": (
                "You are the Generator agent. Propose a technically precise "
                "architecture improvement. Respond with actionable reasoning."
            ),
        },
        {"role": "user", "content": task},
    ]
    critic_history = [{
        "role": "system",
        "content": (
            "You are the Critic/Discriminator agent. Attack the proposal, "
            "identify false assumptions and bottlenecks, score it from 0 to "
            "10, and demand specific corrections."
        ),
    }]

    run = _json_request(
        f"{args.dashboard}/v1/network/benchmarks",
        api_key=api_key,
        method="POST",
        body={
            "kind": "agent_gan_inference_demo",
            "config": {
                "model_id": "gemma-4-26B-A4B-it-mlx-4bit",
                "topology": "primary-decode-allens-prefill",
                "agents": ["generator", "critic"],
                "rounds": args.rounds,
                "output_tokens": args.output_tokens,
            },
        },
    )
    run_id = run["id"]
    all_stages = []

    def get_stats():
        return _json_request(f"{args.dashboard}/v1/network/prefill")

    def execute_agent(client, name, round_index, history):
        token_ids = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )
        warm_tokens, warm = _infer(client, eos_ids, token_ids, 1, get_stats)
        del warm_tokens
        generated, actual = _infer(
            client,
            eos_ids,
            token_ids,
            args.output_tokens,
            get_stats,
        )
        text = tokenizer.decode(generated, skip_special_tokens=True)
        delta = actual["delta"]
        ok = _agent_cache_gate(warm["delta"], delta)
        stage = {
            **actual,
            "name": f"agent_{name}",
            "agent": name,
            "round": round_index,
            "hit_source": "primary_hot" if delta["local_hits"] else "unknown",
            "ok": ok,
            "warmup_prefix_tokens": warm["prefix_tokens"],
            "warmup_tokens_reused": (
                warm["delta"]["tokens_reused"]
                if warm["delta"]["remote_jobs"] == 0 else 0
            ),
            "warmup_wall_s": warm["e2e_s"],
            "warmup_remote_jobs": warm["delta"]["remote_jobs"],
            **_output_metadata(text),
        }
        if not ok:
            raise RuntimeError(f"{name} round {round_index} cache gate failed")
        _json_request(
            f"{args.dashboard}/v1/network/benchmarks/{run_id}",
            api_key=api_key,
            method="PATCH",
            body={"stages": [stage]},
        )
        all_stages.append(stage)
        print(f"\n[{name.upper()} round {round_index}]\n{text}\n", flush=True)
        return text

    try:
        with Client(args.address) as client:
            critic_feedback = ""
            for round_index in range(1, args.rounds + 1):
                if critic_feedback:
                    generator_history.append({
                        "role": "user",
                        "content": (
                            "Revise the architecture using this critic feedback:\n"
                            + critic_feedback
                        ),
                    })
                proposal = execute_agent(
                    client, "generator", round_index, generator_history,
                )
                generator_history.append({"role": "assistant", "content": proposal})
                critic_history.append({
                    "role": "user",
                    "content": (
                        f"Architecture task:\n{task}\n\nGenerator proposal:\n{proposal}"
                    ),
                })
                critic_feedback = execute_agent(
                    client, "critic", round_index, critic_history,
                )
                critic_history.append({
                    "role": "assistant",
                    "content": critic_feedback,
                })
        completed = _json_request(
            f"{args.dashboard}/v1/network/benchmarks/{run_id}",
            api_key=api_key,
            method="PATCH",
            body={
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
