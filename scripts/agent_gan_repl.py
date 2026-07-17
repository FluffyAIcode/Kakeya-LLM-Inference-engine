#!/usr/bin/env python3
"""Interactive Generator/Critic REPL with real-time token streaming."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
import time
import uuid
from pathlib import Path

from scripts.agent_gan_inference_demo import (
    _agent_cache_gate,
    _infer,
    build_critic_context,
)
from scripts.benchmark_prefill_architecture import (
    _ensure_services,
    _json_request,
)
from inference_engine.bench.prefill_fleet_report import summarize_stages


def install_signal_protection() -> None:
    def ignore_sigterm(signum, _frame):
        print(
            f"\n[protected] ignored external signal {signum}. "
            "Type /quit to approve shutdown.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, ignore_sigterm)


def _telemetry_request(url: str, **kwargs):
    try:
        return _json_request(url, timeout=2, **kwargs)
    except Exception as exc:
        print(
            f"[telemetry-warning] {type(exc).__name__}: {exc}; "
            "inference will continue",
            flush=True,
        )
        return None


def build_generator_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer rigorously. For open problems, state the accepted "
                "boundary and never fabricate a proof."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def build_critic_messages(
    prompt: str,
    generator_response: str,
    *,
    stop_reason: str,
    complete: bool,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Score the answer 0-10, identify false claims, and give "
                "specific corrections. Do not penalize an honest statement "
                "that an open problem is unsolved. Review the complete response "
                "as one semantic argument; do not sample or summarize it."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{prompt}\n\nComplete response:\n{generator_response}\n\n"
                f"Completion: {stop_reason}; complete={complete}"
            ),
        },
    ]


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


class PrefillHeartbeat:
    def __init__(self, label: str, interval_s: float = 30.0) -> None:
        self.label = label
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.started = 0.0
        self.thread = None

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=1)

    def _run(self):
        while not self.stop.wait(self.interval_s):
            elapsed = time.perf_counter() - self.started
            print(
                f"[allens] {self.label} Prefill still running: {elapsed:.0f}s",
                flush=True,
            )


def _stage(
    name: str,
    warm: dict,
    actual: dict,
    text: str,
    extra_metrics=None,
) -> dict:
    delta = actual["delta"]
    stage = {
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
    stage.update(extra_metrics or {})
    return stage


def _gate_failure(name: str, warm: dict, actual: dict) -> RuntimeError:
    keys = (
        "local_hits",
        "remote_hits",
        "remote_jobs",
        "tokens_reused",
        "tokens_computed",
        "fallbacks",
        "remote_job_failures",
    )
    compact = lambda delta: {key: delta.get(key, 0) for key in keys}
    return RuntimeError(
        f"{name} KV gate failed: "
        f"warm={compact(warm['delta'])} actual={compact(actual['delta'])}",
    )


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
    if args.output_tokens <= 0:
        raise SystemExit("output-tokens must be > 0")

    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    if not args.skip_ensure:
        print("[startup] ensuring Primary and allens services...", flush=True)
        _ensure_services(args.worker_ssh)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_ids = _resolve_eos_token_ids(tokenizer)
    api_key = Path(args.api_key_file).expanduser().read_text().strip()

    telemetry_state = {"degraded": False, "last_stats": {}}

    def get_stats():
        stats = _telemetry_request(f"{args.dashboard}/v1/network/prefill")
        if stats is None:
            telemetry_state["degraded"] = True
            return dict(telemetry_state["last_stats"])
        telemetry_state["last_stats"] = stats
        return stats

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
            telemetry_state["degraded"] = False
            run = _telemetry_request(
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
            remote_run = run is not None
            run_id = run["id"] if remote_run else f"local_{run_nonce[:16]}"
            try:
                generator_messages = build_generator_messages(prompt)
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
                with PrefillHeartbeat("Generator"):
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
                if not generator_stage["ok"] and not telemetry_state["degraded"]:
                    raise _gate_failure(
                        "Generator",
                        generator_warm,
                        generator_actual,
                    )
                if remote_run:
                    _telemetry_request(
                        f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                        api_key=api_key,
                        method="PATCH",
                        body={"stages": [generator_stage]},
                    )

                critic_context, context_metrics = build_critic_context(
                    tokenizer,
                    generator_text,
                )
                if (
                    critic_context != generator_text
                    or context_metrics["critic_omitted_tokens"] != 0
                    or context_metrics["review_scope"] != "full"
                ):
                    raise RuntimeError("Critic full-context invariant violated")
                critic_messages = build_critic_messages(
                    prompt,
                    critic_context,
                    stop_reason=generator_actual["stop_reason"],
                    complete=generator_actual["complete"],
                )
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
                with PrefillHeartbeat("Critic"):
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
                    extra_metrics=context_metrics,
                )
                if not critic_stage["ok"] and not telemetry_state["degraded"]:
                    raise _gate_failure("Critic", critic_warm, critic_actual)
                completed = None
                if remote_run:
                    completed = _telemetry_request(
                        f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                        api_key=api_key,
                        method="PATCH",
                        body={
                            "stages": [critic_stage],
                            "status": "completed",
                            "finished_at": time.time(),
                        },
                    )
                summary = (
                    completed["summary"]
                    if completed is not None
                    else summarize_stages([generator_stage, critic_stage])
                )
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
                if remote_run:
                    _telemetry_request(
                        f"{args.dashboard}/v1/network/benchmarks/{run_id}",
                        api_key=api_key,
                        method="PATCH",
                        body={"status": "failed", "finished_at": time.time()},
                    )
                print(f"[error] {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
