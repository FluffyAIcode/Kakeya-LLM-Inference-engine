#!/usr/bin/env python3
"""Interactive Generator/Critic REPL with real-time token streaming."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import select
import signal
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
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
            if self.log.closed:
                return len(text)
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
            if not self.log.closed:
                self.log.flush()

    def log_only(self, event: str) -> None:
        with self._lock:
            if self.log.closed:
                return
            if not self._line_start:
                self.log.write("\n")
            self.log.write(f"[{self.timestamp_fn()}] {event.rstrip()}\n")
            self.log.flush()
            self._line_start = True

    def close_log(self) -> None:
        with self._lock:
            if sys.stdout is self:
                sys.stdout = self.terminal
            if sys.stderr is self:
                sys.stderr = self.terminal
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


class ReplPhase(str, Enum):
    WAITING_FOR_GOAL = "waiting_for_goal"
    READY = "ready"
    RUNNING = "running"


@dataclass(frozen=True)
class ReplCommand:
    action: str
    payload: str = ""


@dataclass
class ReplCheckpoint:
    research_goal: str
    previous_generator: str = ""
    previous_critic: str = ""
    last_run_id: str = ""
    schema_version: int = 1


@dataclass
class CriticIssueBatch:
    issue_id: str
    issues: list[str]
    status: str = "pending"
    consumed_by_run: str = ""
    schema_version: int = 1


@dataclass
class ProofObligation:
    obligation_id: str
    statement: str
    status: str = "UNRESOLVED"
    parent_id: str = ""
    last_run_id: str = ""
    last_evidence: str = ""


@dataclass
class ProofObligationLedger:
    ledger_id: str
    obligations: list[ProofObligation]
    version: int = 1
    schema_version: int = 1


def save_proof_ledger(path: Path, ledger: ProofObligationLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = asdict(ledger)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_proof_ledger(path: Path) -> ProofObligationLedger | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    obligations = [
        ProofObligation(**item) for item in raw.pop("obligations", [])
    ]
    ledger = ProofObligationLedger(obligations=obligations, **raw)
    if (
        ledger.schema_version != 1
        or not ledger.ledger_id
        or not ledger.obligations
        or len({item.obligation_id for item in ledger.obligations})
        != len(ledger.obligations)
    ):
        raise ValueError("invalid proof obligation ledger")
    return ledger


def pending_obligations(
    ledger: ProofObligationLedger | None,
) -> list[ProofObligation]:
    if ledger is None:
        return []
    return [
        item for item in ledger.obligations
        if item.status == "UNRESOLVED"
    ]


def format_proof_ledger(ledger: ProofObligationLedger) -> str:
    items = "\n".join(
        f"- {item.obligation_id}: {item.statement}"
        for item in pending_obligations(ledger)
    )
    return (
        f"PROOF OBLIGATION LEDGER id={ledger.ledger_id} "
        f"version={ledger.version}\n{items}\n"
        "Generator requirement: emit `### ISSUE_RESPONSE <ID>` for every "
        "pending ID, with `Correction:`, `Derivation:`, and `Remaining gap:`. "
        "Critic requirement: emit `### ISSUE_VERDICT <ID>` for every pending "
        "ID, with `Status: PROVED|DISPROVED|UNRESOLVED`, `Evidence:`, and "
        "`Missing lemma:`."
    )


_ISSUE_RESPONSE = re.compile(
    r"^### ISSUE_RESPONSE\s+(\S+)",
    re.MULTILINE,
)
_ISSUE_VERDICT = re.compile(
    r"^### ISSUE_VERDICT\s+(\S+)\s*$"
    r"(?P<body>.*?)(?=^### ISSUE_VERDICT\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)


def generator_issue_coverage(
    text: str,
    obligations: list[ProofObligation],
) -> tuple[set[str], set[str]]:
    required = {item.obligation_id for item in obligations}
    covered = set(_ISSUE_RESPONSE.findall(text)) & required
    return covered, required - covered


def apply_critic_verdicts(
    ledger: ProofObligationLedger,
    critic_text: str,
    run_id: str,
) -> dict[str, str]:
    pending_ids = {
        item.obligation_id for item in pending_obligations(ledger)
    }
    verdicts: dict[str, tuple[str, str]] = {}
    for match in _ISSUE_VERDICT.finditer(critic_text):
        obligation_id = match.group(1)
        if obligation_id not in pending_ids:
            continue
        body = match.group("body")
        status_match = re.search(
            r"^Status:\s*(PROVED|DISPROVED|UNRESOLVED)\s*$",
            body,
            re.MULTILINE,
        )
        evidence_match = re.search(
            r"^Evidence:\s*(.+)$",
            body,
            re.MULTILINE,
        )
        missing_match = re.search(
            r"^Missing lemma:\s*(.*)$",
            body,
            re.MULTILINE,
        )
        if not status_match or not evidence_match or missing_match is None:
            continue
        status = status_match.group(1)
        evidence = evidence_match.group(1).strip()
        missing = missing_match.group(1).strip().lower()
        if status in {"PROVED", "DISPROVED"} and (
            len(evidence) < 40
            or missing not in {"", "none", "(none)"}
        ):
            status = "UNRESOLVED"
            evidence = (
                "Closure rejected: proof/counterexample evidence was too "
                "short or a missing lemma remained."
            )
        verdicts[obligation_id] = (status, evidence)
    applied: dict[str, str] = {}
    for item in ledger.obligations:
        if item.obligation_id not in pending_ids:
            continue
        status, evidence = verdicts.get(
            item.obligation_id,
            ("UNRESOLVED", "Critic supplied no structurally valid verdict."),
        )
        item.status = status
        item.last_evidence = evidence
        item.last_run_id = run_id
        applied[item.obligation_id] = status
    ledger.version += 1
    return applied


def save_critic_issue_batch(path: Path, batch: CriticIssueBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(batch), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_pending_critic_issues(path: Path) -> CriticIssueBatch | None:
    if not path.exists():
        return None
    batch = CriticIssueBatch(
        **json.loads(path.read_text(encoding="utf-8")),
    )
    if (
        batch.schema_version != 1
        or not batch.issue_id
        or not batch.issues
        or any(not str(issue).strip() for issue in batch.issues)
    ):
        raise ValueError("invalid Critic issue inbox")
    return batch if batch.status == "pending" else None


def format_critic_issue_injection(batch: CriticIssueBatch) -> str:
    items = "\n".join(
        f"{index}. {issue.strip()}"
        for index, issue in enumerate(batch.issues, start=1)
    )
    return (
        "\n\nEXTERNAL RIGOROUS MATHEMATICAL ISSUES "
        f"(id={batch.issue_id}):\n{items}\n"
        "The Generator must address every issue explicitly. The Critic must "
        "verify every correction and keep unresolved items in the frontier."
    )


def consume_critic_issue_batch(
    path: Path,
    batch: CriticIssueBatch,
    run_id: str,
) -> None:
    batch.status = "consumed"
    batch.consumed_by_run = run_id
    save_critic_issue_batch(path, batch)


def parse_repl_command(raw: str, phase: ReplPhase) -> ReplCommand:
    text = raw.strip()
    lower = text.lower()
    if lower in {"/quit", "/exit"}:
        return ReplCommand("quit")
    if lower.startswith("/new"):
        goal = text[4:].strip()
        if not goal:
            raise ValueError("usage: /new <goal>")
        return ReplCommand("new", goal)
    if phase == ReplPhase.WAITING_FOR_GOAL:
        if text.startswith("/"):
            raise ValueError("set a goal with /new <goal>")
        if not text:
            raise ValueError("research goal must be non-empty")
        return ReplCommand("new", text)
    if phase == ReplPhase.RUNNING:
        raise ValueError("inference is running; input is disabled")
    if lower == "/continue":
        return ReplCommand("continue")
    if lower.startswith("/steer"):
        steering = text[6:].strip()
        if not steering:
            raise ValueError("usage: /steer <text>")
        return ReplCommand("steer", steering)
    raise ValueError(
        "command rejected; use /continue, /steer <text>, "
        "/new <goal>, or /quit",
    )


def save_checkpoint(path: Path, checkpoint: ReplCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_checkpoint(path: Path) -> ReplCheckpoint | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = ReplCheckpoint(**raw)
    if checkpoint.schema_version != 1 or not checkpoint.research_goal:
        raise ValueError("invalid Agent GAN checkpoint")
    return checkpoint


_TIMESTAMP_PREFIX = re.compile(r"^\[[^\]]+\]\s?")


def recover_checkpoint_from_log(
    log_path: Path,
    run_id: str,
) -> ReplCheckpoint:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    goal = ""
    active = False
    section = ""
    generator: list[str] = []
    critic: list[str] = []
    for raw_line in lines:
        line = _TIMESTAMP_PREFIX.sub("", raw_line, count=1)
        if line.startswith("[goal] anchored:"):
            goal = line.split(":", 1)[1].strip()
        if line.startswith("[goal] reset:"):
            goal = line.split(":", 1)[1].strip()
        if f"[inference-start]" in line and f"run={run_id}" in line:
            active = True
            section = ""
            generator = []
            critic = []
            continue
        if not active:
            continue
        if line.startswith("generator>"):
            section = "generator"
            generator.append(line.removeprefix("generator>").lstrip())
            continue
        if line.startswith("[allens] Critic Prefill:"):
            section = ""
            continue
        if line.startswith("critic>"):
            section = "critic"
            critic.append(line.removeprefix("critic>").lstrip())
            continue
        if line.startswith("[metrics]"):
            break
        if section == "generator":
            generator.append(line)
        elif section == "critic":
            critic.append(line)
    generator_text = "\n".join(generator).strip()
    critic_text = "\n".join(critic).strip()
    if not goal or not generator_text or not critic_text:
        raise ValueError(f"complete run {run_id!r} not found in transcript")
    return ReplCheckpoint(
        research_goal=goal,
        previous_generator=generator_text,
        previous_critic=critic_text,
        last_run_id=run_id,
    )


def build_generator_messages(
    goal: str,
    *,
    steering: str = "",
    previous_generator: str = "",
    previous_critic: str = "",
    proof_ledger: str = "",
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
    ledger_text = (
        f"\n\n{proof_ledger}" if proof_ledger else ""
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
                f"{feedback}{steering_text}{ledger_text}"
            ),
        },
    ]


def build_critic_messages(
    goal: str,
    generator_response: str,
    *,
    steering: str = "",
    proof_ledger: str = "",
    stop_reason: str,
    complete: bool,
) -> list[dict[str, str]]:
    ledger_text = (
        f"\n\n{proof_ledger}" if proof_ledger else ""
    )
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
                "the proof-obligation frontier for the immutable goal. When a "
                "PROOF OBLIGATION LEDGER is present, adjudicate every pending "
                "ID with the exact ISSUE_VERDICT format before any new frontier."
            ),
        },
        {
            "role": "user",
            "content": (
                f"IMMUTABLE RESEARCH GOAL:\n{goal}\n\n"
                f"Current steering:\n{steering or '(none)'}\n\n"
                f"Complete response:\n{generator_response}\n\n"
                f"Completion: {stop_reason}; complete={complete}"
                f"{ledger_text}"
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
    parser.add_argument(
        "--state-file",
        default="~/.kakeya/agent_gan_state.json",
        help="Private resumable Generator/Critic checkpoint.",
    )
    parser.add_argument(
        "--critic-inbox",
        default="~/.kakeya/agent_gan_critic_inbox.json",
        help="Private pending rigorous-math issues for the next turn.",
    )
    parser.add_argument(
        "--proof-ledger",
        default="~/.kakeya/agent_gan_proof_ledger.json",
        help="Private persistent mathematical proof obligations.",
    )
    parser.add_argument(
        "--candidate-file",
        default="",
        help="AutoResearch candidate strategy applied to this experiment.",
    )
    parser.add_argument("--recover-run", default="")
    parser.add_argument(
        "--recover-log",
        default="~/.kakeya/logs/agent_gan_repl.log",
    )
    parser.add_argument("--auto-continue", action="store_true")
    parser.add_argument(
        "--no-auto-loop",
        action="store_false",
        dest="auto_loop",
        help="Pause after each successful turn instead of continuing.",
    )
    parser.set_defaults(auto_loop=True)
    parser.add_argument(
        "--auto-loop-boundary-wait-s",
        type=float,
        default=0.5,
        help="Boundary window for an explicit command before auto-continue.",
    )
    args = parser.parse_args()
    if args.output_tokens <= 0:
        raise SystemExit("output-tokens must be > 0")
    if args.auto_loop_boundary_wait_s < 0:
        raise SystemExit("auto-loop-boundary-wait-s must be >= 0")
    research_candidate = None
    if args.candidate_file:
        from autoresearch.prefill.prepare import _load_candidate
        research_candidate = _load_candidate(
            Path(args.candidate_file).expanduser(),
        )
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

    state_path = Path(args.state_file).expanduser()
    critic_inbox_path = Path(args.critic_inbox).expanduser()
    proof_ledger_path = Path(args.proof_ledger).expanduser()
    if args.recover_run:
        recovered = recover_checkpoint_from_log(
            Path(args.recover_log).expanduser(),
            args.recover_run,
        )
        save_checkpoint(state_path, recovered)
        print(
            f"[state-recovered] run={recovered.last_run_id} "
            f"generator_chars={len(recovered.previous_generator)} "
            f"critic_chars={len(recovered.previous_critic)}",
            flush=True,
        )
    checkpoint = load_checkpoint(state_path)
    research_goal = checkpoint.research_goal if checkpoint else ""
    previous_generator = checkpoint.previous_generator if checkpoint else ""
    previous_critic = checkpoint.previous_critic if checkpoint else ""
    phase = ReplPhase.READY if checkpoint else ReplPhase.WAITING_FOR_GOAL
    auto_loop_active = bool(args.auto_loop)
    pending_input = (
        "/continue"
        if checkpoint and (args.auto_continue or auto_loop_active)
        else ""
    )
    print(
        "Kakeya Agent GAN REPL ready.\n"
        "Commands: /new <goal>, /continue, /steer <text>, /quit.\n"
        "Raw text is accepted only as the initial goal; inference output can "
        "never become implicit steering.\n"
        f"Auto-loop is {'enabled' if auto_loop_active else 'disabled'}; "
        "successful turns continue automatically and exceptions pause it.\n"
        "Each turn runs allens Prefill → Primary hot Generator → "
        "allens Prefill → Primary hot Critic.",
        flush=True,
    )
    print(f"[log] {transcript.log_path}", flush=True)
    print(f"[state] {state_path} phase={phase.value}", flush=True)
    print(f"[critic-inbox] {critic_inbox_path}", flush=True)
    print(f"[proof-ledger] {proof_ledger_path}", flush=True)
    if checkpoint:
        print(
            f"[state-restored] run={checkpoint.last_run_id or '(none)'} "
            f"goal={hashlib.sha256(research_goal.encode()).hexdigest()}",
            flush=True,
        )
    with Client(args.address) as client:
        while True:
            if pending_input:
                raw_input = pending_input
                pending_input = ""
                print(f"\nprompt> {raw_input}", flush=True)
            elif auto_loop_active and phase == ReplPhase.READY:
                print("\nprompt> ", end="", flush=True)
                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    args.auto_loop_boundary_wait_s,
                )
                if readable:
                    raw_input = input()
                else:
                    raw_input = "/continue"
                    print(raw_input, flush=True)
            else:
                try:
                    raw_input = input("\nprompt> ")
                except EOFError:
                    print("\n[bye]")
                    break
            transcript.log_only(f"[input] {raw_input or '(empty)'}")
            try:
                command = parse_repl_command(raw_input, phase)
            except ValueError as exc:
                print(f"[input-rejected] {exc}", flush=True)
                continue
            if command.action == "quit":
                print("[bye]")
                break
            if command.action == "new":
                research_goal = command.payload
                previous_generator = ""
                previous_critic = ""
                save_checkpoint(
                    state_path,
                    ReplCheckpoint(research_goal=research_goal),
                )
                phase = ReplPhase.READY
                auto_loop_active = bool(args.auto_loop)
                print(f"[goal] reset: {research_goal}", flush=True)
            steering = command.payload if command.action == "steer" else ""
            generator_steering = steering
            critic_strategy = ""
            if research_candidate is not None:
                generator_steering = "\n\n".join(filter(None, (
                    steering,
                    str(research_candidate.GENERATOR_DIRECTIVE),
                )))
                critic_strategy = str(
                    research_candidate.CRITIC_DIRECTIVE,
                ) + (
                    "\n\nAt the end, emit exactly:\n"
                    "### AUTORESEARCH_VERDICT\n"
                    f"Candidate ID: {research_candidate.CANDIDATE_ID}\n"
                    "Outcome: SUPPORTED|FALSIFIED|INCONCLUSIVE\n"
                    "Evidence: <specific derivation, counterexample, or failed lemma>\n"
                    "New frontier: <one concrete smaller proof obligation that "
                    "is not a restatement of the current obligation>"
                )
            if command.action in {"continue", "steer"} and args.auto_loop:
                auto_loop_active = True
            phase = ReplPhase.RUNNING
            critic_issue_batch = load_pending_critic_issues(
                critic_inbox_path,
            )
            critic_issue_injection = (
                format_critic_issue_injection(critic_issue_batch)
                if critic_issue_batch is not None else ""
            )
            proof_ledger = load_proof_ledger(proof_ledger_path)
            turn_obligations = pending_obligations(proof_ledger)
            proof_ledger_text = (
                format_proof_ledger(proof_ledger)
                if turn_obligations else ""
            )
            if proof_ledger is not None:
                print(
                    f"[proof-ledger-loaded] id={proof_ledger.ledger_id} "
                    f"version={proof_ledger.version} "
                    f"pending={len(turn_obligations)}",
                    flush=True,
                )
                for item in turn_obligations:
                    print(
                        f"[proof-obligation-pending] "
                        f"id={item.obligation_id} {item.statement}",
                        flush=True,
                    )
            if critic_issue_batch is not None:
                print(
                    f"[critic-issue-injection] id="
                    f"{critic_issue_batch.issue_id} "
                    f"count={len(critic_issue_batch.issues)}",
                    flush=True,
                )
                for index, issue in enumerate(
                    critic_issue_batch.issues,
                    start=1,
                ):
                    print(
                        f"[critic-issue-{index}] {issue}",
                        flush=True,
                    )
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
                        "critic_issue_id": (
                            critic_issue_batch.issue_id
                            if critic_issue_batch is not None else ""
                        ),
                        "critic_issue_count": (
                            len(critic_issue_batch.issues)
                            if critic_issue_batch is not None else 0
                        ),
                        "proof_ledger_id": (
                            proof_ledger.ledger_id
                            if proof_ledger is not None else ""
                        ),
                        "proof_ledger_version": (
                            proof_ledger.version
                            if proof_ledger is not None else 0
                        ),
                        "proof_obligations_pending": len(turn_obligations),
                        "autoresearch_candidate_id": (
                            str(research_candidate.CANDIDATE_ID)
                            if research_candidate is not None else ""
                        ),
                        "autoresearch_target_obligation": (
                            str(research_candidate.TARGET_OBLIGATION_ID)
                            if research_candidate is not None else ""
                        ),
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
                    steering=generator_steering,
                    previous_generator=previous_generator,
                    previous_critic=(
                        previous_critic + critic_issue_injection
                    ),
                    proof_ledger=proof_ledger_text,
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
                covered_issues, missing_issues = generator_issue_coverage(
                    generator_text,
                    turn_obligations,
                )
                if turn_obligations:
                    print(
                        f"[generator-issue-coverage] "
                        f"covered={len(covered_issues)}/"
                        f"{len(turn_obligations)} "
                        f"missing={','.join(sorted(missing_issues)) or '(none)'}",
                        flush=True,
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
                    steering="\n\n".join(filter(None, (
                        steering,
                        critic_strategy,
                        critic_issue_injection,
                    ))),
                    proof_ledger=proof_ledger_text + (
                        "\nGENERATOR COVERAGE FAILURE: missing "
                        + ", ".join(sorted(missing_issues))
                        if missing_issues else ""
                    ),
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
                applied_verdicts = {}
                if proof_ledger is not None and turn_obligations:
                    applied_verdicts = apply_critic_verdicts(
                        proof_ledger,
                        critic_text,
                        run_id,
                    )
                    for obligation_id, status in applied_verdicts.items():
                        print(
                            f"[critic-verdict] id={obligation_id} "
                            f"status={status}",
                            flush=True,
                        )
                    print(
                        f"[proof-ledger-result] id={proof_ledger.ledger_id} "
                        f"version={proof_ledger.version} "
                        f"unresolved={len(pending_obligations(proof_ledger))}",
                        flush=True,
                    )
                critic_stage = _stage(
                    "critic",
                    critic_warm,
                    critic_actual,
                    critic_text,
                    extra_metrics={
                        **context_metrics,
                        "proof_ledger_id": (
                            proof_ledger.ledger_id
                            if proof_ledger is not None else ""
                        ),
                        "proof_obligations_total": len(turn_obligations),
                        "proof_obligations_covered": len(covered_issues),
                        "proof_obligations_unresolved": (
                            len(pending_obligations(proof_ledger))
                            if proof_ledger is not None else 0
                        ),
                    },
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
                save_checkpoint(
                    state_path,
                    ReplCheckpoint(
                        research_goal=research_goal,
                        previous_generator=previous_generator,
                        previous_critic=previous_critic,
                        last_run_id=run_id,
                    ),
                )
                if proof_ledger is not None and turn_obligations:
                    save_proof_ledger(proof_ledger_path, proof_ledger)
                    for item in proof_ledger.obligations:
                        if item.obligation_id not in applied_verdicts:
                            continue
                        event = (
                            "proof-obligation-carried"
                            if item.status == "UNRESOLVED"
                            else "proof-obligation-closed"
                        )
                        print(
                            f"[{event}] id={item.obligation_id} "
                            f"status={item.status} run={run_id}",
                            flush=True,
                        )
                    print(
                        f"[proof-ledger-checkpoint] "
                        f"id={proof_ledger.ledger_id} "
                        f"version={proof_ledger.version}",
                        flush=True,
                    )
                if critic_issue_batch is not None:
                    consume_critic_issue_batch(
                        critic_inbox_path,
                        critic_issue_batch,
                        run_id,
                    )
                    print(
                        f"[critic-issues-consumed] id="
                        f"{critic_issue_batch.issue_id} run={run_id}",
                        flush=True,
                    )
                phase = ReplPhase.READY
                if auto_loop_active:
                    print(
                        "[auto-loop] successful turn complete; "
                        "/continue queued",
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
                phase = ReplPhase.READY
                auto_loop_active = False
                print(
                    "[auto-loop-paused] inference exception; checkpoint "
                    "preserved. Use /continue after remediation.",
                    flush=True,
                )
    transcript.log_only("[session-end]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
