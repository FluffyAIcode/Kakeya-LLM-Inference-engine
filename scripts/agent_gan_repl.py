#!/usr/bin/env python3
"""Interactive Generator/Critic REPL with real-time token streaming."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
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


class TimestampedTee:
    """Mirror Terminal output to a line-timestamped, immediately flushed log."""

    def __init__(self, terminal, log_path: Path, timestamp_fn=None) -> None:
        self.terminal = terminal
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        self.timestamp_fn = timestamp_fn or (
            lambda: datetime.now().astimezone().isoformat(timespec="milliseconds")
        )
        self._line_start = True
        self._lock = threading.RLock()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def fileno(self):
        return self.terminal.fileno()

    def isatty(self):
        return self.terminal.isatty()

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self.terminal.write(text)
            for part in text.splitlines(keepends=True):
                if self._line_start:
                    self.log.write(f"[{self.timestamp_fn()}] ")
                self.log.write(part)
                self._line_start = part.endswith(("\n", "\r"))
            self.log.flush()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self.terminal.flush()
            self.log.flush()

    def log_only(self, event: str) -> None:
        with self._lock:
            if not self._line_start:
                self.log.write("\n")
            self.log.write(f"[{self.timestamp_fn()}] {event.rstrip()}\n")
            self.log.flush()
            self._line_start = True

    def close_log(self) -> None:
        with self._lock:
            if not self.log.closed:
                self.log.flush()
                self.log.close()


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


_RUNTIME_ARTIFACT = re.compile(
    r"^\s*(?:generator>|critic>|prompt>|\[(?:metrics|allens|error|"
    r"telemetry-warning|protected|supervisor)\]|Traceback\b)",
    re.IGNORECASE,
)


def is_runtime_artifact_prompt(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and bool(_RUNTIME_ARTIFACT.match(lines[0]))


def build_generator_messages(
    goal: str,
    *,
    steering: str = "",
    previous_generator: str = "",
    previous_critic: str = "",
) -> list[dict[str, str]]:
    feedback = ""
    if previous_generator or previous_critic:
        feedback = (
            "\n\nComplete previous Generator response:\n"
            f"{previous_generator}\n\nComplete previous Critic correction:\n"
            f"{previous_critic}\n\nApply the Critic's Next Adversarial Step "
            "while remaining anchored to the immutable goal."
        )
    steering_text = (
        f"\n\nCurrent human steering (subordinate to the goal):\n{steering}"
        if steering else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "Pursue the requested mathematical argument constructively and "
                "rigorously. For an open problem, do not fabricate a proof, but "
                "do not stop at 'unsolved': identify the exact global claim, "
                "derive known reductions, recursively decompose missing proof "
                "obligations, and state the smallest unresolved frontier. "
                "Distinguish unknown from impossible."
            ),
        },
        {
            "role": "user",
            "content": (
                f"IMMUTABLE RESEARCH GOAL:\n{goal}"
                f"{feedback}{steering_text}"
            ),
        },
    ]


def build_critic_messages(
    goal: str,
    generator_response: str,
    *,
    steering: str = "",
    stop_reason: str,
    complete: bool,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act as a recursive adversarial proof analyst. Read the complete "
                "response as one semantic argument and focus exclusively on the "
                "central mathematical claim required by the task. Ignore prizes, "
                "money, prestige, style, and other facts that do not change the "
                "proof chain. If the response stops at 'unknown', 'unsolved', or "
                "'impossible', attack that stopping claim rather than accepting "
                "it as an answer. Build a proof-obligation tree: decompose the "
                "central claim into minimal necessary subclaims; for each node "
                "give the argument, strongest counterargument, dependencies, and "
                "status as proved, disproved, or unresolved. Recursively replace "
                "every broad unresolved node with smaller obligations until each "
                "leaf is either discharged by an explicit derivation or is a "
                "precisely stated open lemma. Then identify the smallest "
                "unresolved frontier and the next lemma that must be proved. "
                "Never output a numeric score or blanket approval. Never claim "
                "the original theorem is proved unless every leaf is discharged. "
                "Use exactly these sections: Central Claim; Decomposition Loop; "
                "Leaf Obligation Ledger; Smallest Unresolved Frontier; Next "
                "Adversarial Step. Do not sample, summarize, simplify, or use a "
                "fallback review."
                " Begin with `Goal Alignment: ALIGNED` or `Goal Alignment: "
                "DRIFTED`. If drifted, discard the off-topic branch and restore "
                "the proof-obligation frontier for the immutable goal."
            ),
        },
        {
            "role": "user",
            "content": (
                f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
                f"Current steering:\n{steering or '(none)'}\n\n"
                f"Complete response:\n{generator_response}\n\n"
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
    def __init__(
        self,
        label: str,
        interval_s: float = 30.0,
        stats_provider=None,
    ) -> None:
        self.label = label
        self.interval_s = interval_s
        self.stats_provider = stats_provider
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
            progress = ""
            if self.stats_provider is not None:
                stats = self.stats_provider()
                total = int(stats.get("remote_job_tokens_total", 0))
                computed = int(stats.get("remote_job_tokens_computed", 0))
                if total > 0:
                    percent = min(100.0, computed / total * 100.0)
                    eta = (
                        elapsed * (total - computed) / computed
                        if computed > 0 else 0.0
                    )
                    progress = (
                        f" · {computed}/{total} tokens ({percent:.1f}%)"
                        + (f" · ETA {eta:.0f}s" if computed > 0 else "")
                    )
            print(
                f"[allens] {self.label} Prefill: {elapsed:.0f}s{progress}",
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
    parser.add_argument(
        "--log-file",
        default="~/.kakeya/logs/agent_gan_repl.log",
        help="Timestamped local transcript log.",
    )
    args = parser.parse_args()
    if args.output_tokens <= 0:
        raise SystemExit("output-tokens must be > 0")
    transcript = TimestampedTee(
        sys.stdout,
        Path(args.log_file).expanduser(),
    )
    sys.stdout = transcript
    sys.stderr = transcript
    atexit.register(transcript.close_log)
    install_signal_protection()
    transcript.log_only(
        f"[session-start] pid={os.getpid()} "
        f"log={transcript.log_path}",
    )

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
        "Kakeya Agent GAN REPL ready. First prompt sets the immutable goal.\n"
        "Use /continue to apply Critic feedback, /new <goal> to reset, "
        "and /quit to exit.\n"
        "Each turn runs allens Prefill → Primary hot Generator → "
        "allens Prefill → Primary hot Critic.",
        flush=True,
    )
    print(f"[log] {transcript.log_path}", flush=True)
    research_goal = ""
    previous_generator = ""
    previous_critic = ""
    with Client(args.address) as client:
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except EOFError:
                print("\n[bye]")
                break
            transcript.log_only(f"[input] {prompt or '(empty)'}")
            if not prompt:
                continue
            if prompt.lower() in {"/quit", "/exit"}:
                print("[bye]")
                break
            if prompt.lower().startswith("/new"):
                new_goal = prompt[4:].strip()
                research_goal = new_goal
                previous_generator = ""
                previous_critic = ""
                if not research_goal:
                    print("[goal] cleared; enter a new research goal", flush=True)
                    continue
                prompt = research_goal
                print(f"[goal] reset: {research_goal}", flush=True)
            elif prompt.lower() == "/continue":
                if not research_goal:
                    print("[goal-error] no active research goal", flush=True)
                    continue
                prompt = ""
            elif is_runtime_artifact_prompt(prompt):
                print(
                    "[input-rejected] runtime output cannot become a research "
                    "prompt; use /continue or /new <goal>",
                    flush=True,
                )
                continue
            elif not research_goal:
                research_goal = prompt
                print(f"[goal] anchored: {research_goal}", flush=True)
                prompt = ""
            steering = prompt
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
                        "goal_anchor": hashlib.sha256(
                            research_goal.encode(),
                        ).hexdigest(),
                        "feedback_applied": bool(previous_critic),
                    },
                },
            )
            remote_run = run is not None
            run_id = run["id"] if remote_run else f"local_{run_nonce[:16]}"
            started_at = datetime.now().astimezone().isoformat(
                timespec="milliseconds",
            )
            print(
                f"[inference-start] time={started_at} run={run_id} "
                f"goal={hashlib.sha256(research_goal.encode()).hexdigest()}",
                flush=True,
            )
            try:
                generator_messages = build_generator_messages(
                    research_goal,
                    steering=steering,
                    previous_generator=previous_generator,
                    previous_critic=previous_critic,
                )
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
                with PrefillHeartbeat("Generator", stats_provider=get_stats):
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
                    protocol="goal_anchored_recursive_gan_v3",
                )
                if (
                    critic_context != generator_text
                    or context_metrics["critic_omitted_tokens"] != 0
                    or context_metrics["review_scope"] != "full"
                ):
                    raise RuntimeError("Critic full-context invariant violated")
                critic_messages = build_critic_messages(
                    research_goal,
                    critic_context,
                    steering=steering,
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
                with PrefillHeartbeat("Critic", stats_provider=get_stats):
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
                previous_generator = generator_text
                previous_critic = critic_text
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
                print(
                    f"[inference-complete] time="
                    f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} "
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
                print(
                    f"[inference-failed] time="
                    f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} "
                    f"run={run_id} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
    transcript.log_only("[session-end]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
