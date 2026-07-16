#!/usr/bin/env python3
"""Interactive Generator/Critic REPL with real-time token streaming."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
import uuid
from pathlib import Path

from scripts.agent_gan_inference_demo import (
    _agent_cache_gate,
    _infer,
)
from scripts.benchmark_prefill_architecture import (
    _ensure_services,
    _json_request,
)


def install_signal_protection() -> None:
    def ignore_sigterm(signum, _frame):
        print(
            f"\n[protected] ignored external signal {signum}. "
            "Type /quit to approve shutdown.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, ignore_sigterm)


class TokenPrinter:
    def __init__(self, tokenizer, label: str) -> None:
        self.tokenizer = tokenizer
        self.last = ""
        print(f"{label}> ", end="", flush=True)

    def __call__(self, token_ids) -> None:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(text[len(self.last):], end="", flush=True)
        self.last = text

    def finish(self) -> None:
        print(flush=True)


def _stage(name: str, warm: dict, actual: dict, text: str) -> dict:
    delta = actual["delta"]
    return {
        **actual,
        "name": f"agent_{name}",
        "agent": name,
        "round": 1,
        "hit_source": "primary_hot" if delta["local_hits"] else "unknown",
        "ok": _agent_cache_gate(warm["delta"], delta) and actual["complete"],
        "warmup_prefix_tokens": warm["prefix_tokens"],
        "warmup_tokens_reused": (
            warm["delta"]["tokens_reused"]
            if warm["delta"]["remote_jobs"] == 0 else 0
        ),
        "warmup_wall_s": warm["e2e_s"],
        "warmup_remote_jobs": warm["delta"]["remote_jobs"],
        "output_chars": len(text),
        "output_hash": hashlib.sha256(text.encode()).hexdigest(),
    }


def main() -> int:
    install_signal_protection()
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ssh", default="allens")
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument("--api-key-file", default="~/.kakeya/network_api_key")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=0,
        help="Optional client response cap; 0 means generate until model EOS.",
    )
    parser.add_argument("--skip-ensure", action="store_true")
    args = parser.parse_args()

    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    if not args.skip_ensure:
        print("[startup] ensuring Primary and allens services...", flush=True)
        _ensure_services(args.worker_ssh)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_ids = _resolve_eos_token_ids(tokenizer)
    api_key = Path(args.api_key_file).expanduser().read_text().strip()

    def get_stats():
        return _json_request(f"{args.dashboard}/v1/network/prefill")

    print(
        "Kakeya Agent GAN REPL ready. Type a prompt; /quit exits.\n"
        "Each turn runs allens Prefill → Primary hot Generator → "
        "allens Prefill → Primary hot Critic.",
        flush=True,
    )
    with Client(args.address) as client:
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except EOFError:
                print("\n[bye]")
                break
            if not prompt:
                continue
            if prompt.lower() in {"/quit", "/exit"}:
                print("[bye]")
                break
            run_nonce = uuid.uuid4().hex
            run = _json_request(
                f"{args.dashboard}/v1/network/benchmarks",
                api_key=api_key,
                method="POST",
                body={
                    "kind": "agent_gan_interactive",
                    "config": {
                        "model_id": "gemma-4-26B-A4B-it-mlx-4bit",
                        "topology": "primary-decode-allens-prefill",
                        "agents": ["generator", "critic"],
                        "rounds": 1,
                        "output_tokens": args.output_tokens,
                    },
                },
            )
            run_id = run["id"]
            try:
                generator_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are the Generator agent. Produce a concrete, "
                            "technically rigorous answer. For open or unsolved "
                            "problems, state the accepted boundary honestly, "
                            "provide rigorous context, and never fabricate a "
                            "proof. Internal run "
                            f"{run_nonce}."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                generator_ids = tokenizer.apply_chat_template(
                    generator_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )
                print(
                    f"[allens] Generator Prefill: {len(generator_ids)} tokens...",
                    flush=True,
                )
                _, generator_warm = _infer(
                    client, eos_ids, generator_ids, 1, get_stats,
                )
                generator_printer = TokenPrinter(tokenizer, "generator")
                generator_tokens, generator_actual = _infer(
                    client,
                    eos_ids,
                    generator_ids,
                    args.output_tokens,
                    get_stats,
                    on_token=generator_printer,
                    max_response_tokens=args.max_response_tokens,
                )
                generator_printer.finish()
                generator_text = tokenizer.decode(
                    generator_tokens,
                    skip_special_tokens=True,
                )
                generator_stage = _stage(
                    "generator",
                    generator_warm,
                    generator_actual,
                    generator_text,
                )
                if not generator_stage["ok"]:
                    raise RuntimeError("Generator KV gate failed")
                _json_request(
                    f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                    api_key=api_key,
                    method="PATCH",
                    body={"stages": [generator_stage]},
                )

                critic_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are the Critic/Discriminator. Score the answer "
                            "0-10, identify false assumptions, and propose "
                            "specific corrections. Do not penalize a correct "
                            "statement that an open problem has no accepted "
                            "proof. Call an answer incomplete only when its "
                            "completion status is not EOS or its syntax is "
                            "visibly cut off. Internal run "
                            f"{run_nonce}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Original task:\n{prompt}\n\n"
                            f"Generator answer:\n{generator_text}\n\n"
                            "Generator completion status: "
                            f"{generator_actual['stop_reason']}; "
                            f"complete={generator_actual['complete']}"
                        ),
                    },
                ]
                critic_ids = tokenizer.apply_chat_template(
                    critic_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )
                print(
                    f"[allens] Critic Prefill: {len(critic_ids)} tokens...",
                    flush=True,
                )
                _, critic_warm = _infer(
                    client, eos_ids, critic_ids, 1, get_stats,
                )
                critic_printer = TokenPrinter(tokenizer, "critic")
                critic_tokens, critic_actual = _infer(
                    client,
                    eos_ids,
                    critic_ids,
                    args.output_tokens,
                    get_stats,
                    on_token=critic_printer,
                    max_response_tokens=args.max_response_tokens,
                )
                critic_printer.finish()
                critic_text = tokenizer.decode(
                    critic_tokens,
                    skip_special_tokens=True,
                )
                critic_stage = _stage(
                    "critic",
                    critic_warm,
                    critic_actual,
                    critic_text,
                )
                if not critic_stage["ok"]:
                    raise RuntimeError("Critic KV gate failed")
                completed = _json_request(
                    f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                    api_key=api_key,
                    method="PATCH",
                    body={
                        "stages": [critic_stage],
                        "status": "completed",
                        "finished_at": time.time(),
                    },
                )
                summary = completed["summary"]
                print(
                    "[metrics] "
                    f"KV hit={summary['workload_kv_token_hit_rate']:.1%} "
                    f"decode={summary['aggregate_decode_tok_s']:.2f} tok/s "
                    f"latency={summary['generation_latency_ms_p50']:.2f} ms/token "
                    f"e2e={summary['aggregate_e2e_tok_s']:.2f} tok/s "
                    f"run={run_id}",
                    flush=True,
                )
            except Exception as exc:
                _json_request(
                    f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                    api_key=api_key,
                    method="PATCH",
                    body={"status": "failed", "finished_at": time.time()},
                )
                print(f"[error] {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
