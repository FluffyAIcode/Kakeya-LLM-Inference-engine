#!/usr/bin/env python3
"""Real Karpathy-style AutoResearch supervisor around one-shot GAN experiments."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from autoresearch.prefill.prepare import _load_candidate, evaluate


REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "target_obligation_id",
    "hypothesis",
    "generator_directive",
    "critic_directive",
    "prefill_compute_chunk_tokens",
)


def _json_request(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def _wait_port(host: str, port: int, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"service did not become ready: {host}:{port}")


def _candidate_snapshot(module) -> dict:
    return {
        "candidate_id": str(module.CANDIDATE_ID),
        "target_obligation_id": str(module.TARGET_OBLIGATION_ID),
        "hypothesis": str(module.HYPOTHESIS),
        "generator_directive": str(module.GENERATOR_DIRECTIVE),
        "critic_directive": str(module.CRITIC_DIRECTIVE),
        "prefill_compute_chunk_tokens": int(
            module.PREFILL_COMPUTE_CHUNK_TOKENS,
        ),
        "snapshot_mode": str(module.SNAPSHOT_MODE),
        "max_segment_seconds": float(module.MAX_SEGMENT_SECONDS),
        "require_full_context": bool(module.REQUIRE_FULL_CONTEXT),
        "allow_fallback": bool(module.ALLOW_FALLBACK),
    }


def validate_candidate(candidate: dict) -> None:
    text_fields = {
        "candidate_id",
        "target_obligation_id",
        "hypothesis",
        "generator_directive",
        "critic_directive",
    }
    missing = [
        field
        for field in REQUIRED_CANDIDATE_FIELDS
        if (
            not candidate.get(field)
            or (
                field in text_fields
                and not isinstance(candidate.get(field), str)
            )
        )
    ]
    if missing:
        raise ValueError(f"candidate missing fields: {missing}")
    if candidate["prefill_compute_chunk_tokens"] not in (64, 128, 256):
        raise ValueError("chunk tokens must be one of 64, 128, 256")
    if candidate.get("snapshot_mode", "final_only") != "final_only":
        raise ValueError("snapshot mode must remain final_only")
    if candidate.get("require_full_context", True) is not True:
        raise ValueError("candidate must require full context")
    if candidate.get("allow_fallback", False) is not False:
        raise ValueError("candidate must forbid fallback")


def _select_repair_target(current: dict, ledger: dict) -> str:
    leaves = _pending_leaf_ids(ledger)
    if not leaves:
        raise ValueError("proof ledger has no unresolved leaf")
    current_target = str(current.get("target_obligation_id", ""))
    if current_target in leaves:
        return current_target
    parents = {
        str(item.get("obligation_id", "")): str(item.get("parent_id", ""))
        for item in ledger.get("obligations", [])
    }

    def distance_from_current(obligation_id: str) -> int:
        distance = 0
        cursor = obligation_id
        visited = set()
        while cursor and cursor not in visited:
            if cursor == current_target:
                return distance
            visited.add(cursor)
            cursor = parents.get(cursor, "")
            distance += 1
        return -1

    descendants = [
        (distance_from_current(obligation_id), obligation_id)
        for obligation_id in leaves
        if distance_from_current(obligation_id) >= 0
    ]
    if descendants:
        return max(descendants)[1]
    return leaves[0]


def repair_candidate_schema(
    candidate: dict,
    *,
    current: dict,
    ledger: dict,
) -> tuple[dict, list[str]]:
    aliases = {
        "candidate_id": ("id", "strategy_id"),
        "target_obligation_id": (
            "target", "target_id", "obligation_id",
        ),
        "hypothesis": ("strategy_hypothesis",),
        "generator_directive": (
            "generator", "generator_prompt", "generator_strategy",
        ),
        "critic_directive": (
            "critic", "critic_prompt", "critic_strategy",
        ),
        "prefill_compute_chunk_tokens": (
            "chunk_tokens", "compute_chunk_tokens",
        ),
    }
    normalized = {
        str(key).strip().lower().replace("-", "_"): value
        for key, value in candidate.items()
    }
    repaired = dict(candidate)
    changed: list[str] = []
    for field, field_aliases in aliases.items():
        if repaired.get(field):
            continue
        for alias in (field, *field_aliases):
            value = normalized.get(alias)
            if value not in (None, ""):
                repaired[field] = value
                changed.append(field)
                break
    nested_hypothesis = repaired.get("hypothesis")
    if isinstance(nested_hypothesis, dict):
        nested_target = (
            nested_hypothesis.get("target_obligation")
            or nested_hypothesis.get("target_obligation_id")
            or nested_hypothesis.get("target")
        )
        if nested_target and not repaired.get("target_obligation_id"):
            repaired["target_obligation_id"] = str(nested_target).strip()
            changed.append("target_obligation_id")
        repaired["hypothesis"] = str(
            nested_hypothesis.get("statement")
            or nested_hypothesis.get("text")
            or nested_hypothesis.get("claim")
            or ""
        ).strip()
        changed.append("hypothesis")
    for field in (
        "candidate_id",
        "target_obligation_id",
        "generator_directive",
        "critic_directive",
    ):
        value = repaired.get(field)
        if isinstance(value, dict):
            repaired[field] = str(
                value.get("statement")
                or value.get("text")
                or value.get("content")
                or ""
            ).strip()
            changed.append(field)
        elif value is not None and not isinstance(value, str):
            repaired[field] = str(value).strip()
            changed.append(field)
    target = str(repaired.get("target_obligation_id", ""))
    leaves = _pending_leaf_ids(ledger)
    if target not in leaves:
        repaired["target_obligation_id"] = _select_repair_target(
            current,
            ledger,
        )
        if "target_obligation_id" not in changed:
            changed.append("target_obligation_id")
    target = repaired["target_obligation_id"]
    statement = next(
        str(item.get("statement", ""))
        for item in ledger.get("obligations", [])
        if item.get("obligation_id") == target
    )
    hypothesis = str(repaired.get("hypothesis", "")).strip()
    repaired["hypothesis"] = hypothesis
    plan = repaired.get("plan", {})
    plan_steps = plan.get("steps", []) if isinstance(plan, dict) else []
    plan_text = " ".join(
        f"Step {index}: {str(step).strip()}"
        for index, step in enumerate(plan_steps, start=1)
        if str(step).strip()
    )
    if not repaired.get("generator_directive") and hypothesis:
        repaired["generator_directive"] = (
            f"Focus exclusively on {target}: {statement} "
            f"Construct and test this hypothesis: {hypothesis} "
            f"{plan_text}"
        )
        changed.append("generator_directive")
    if not repaired.get("critic_directive") and hypothesis:
        repaired["critic_directive"] = (
            f"Attempt to falsify the {target} hypothesis: {hypothesis} "
            "Identify the first invalid inference and one strictly smaller "
            "missing lemma."
        )
        changed.append("critic_directive")
    fixed_chunk_tokens = int(current["prefill_compute_chunk_tokens"])
    proposed_chunk_tokens = repaired.get("prefill_compute_chunk_tokens")
    try:
        proposed_chunk_tokens = int(proposed_chunk_tokens)
    except (TypeError, ValueError):
        proposed_chunk_tokens = None
    if (
        proposed_chunk_tokens is None
        or int(proposed_chunk_tokens) != fixed_chunk_tokens
    ):
        repaired["prefill_compute_chunk_tokens"] = fixed_chunk_tokens
        changed.append("prefill_compute_chunk_tokens")
    else:
        repaired["prefill_compute_chunk_tokens"] = fixed_chunk_tokens
    return repaired, changed


def render_candidate(candidate: dict) -> str:
    validate_candidate(candidate)
    return (
        '"""AutoResearch agent-editable strategy. Generated by supervisor."""\n\n'
        f"CANDIDATE_ID = {candidate['candidate_id']!r}\n"
        f"TARGET_OBLIGATION_ID = {candidate['target_obligation_id']!r}\n"
        f"HYPOTHESIS = {candidate['hypothesis']!r}\n"
        f"GENERATOR_DIRECTIVE = {candidate['generator_directive']!r}\n"
        f"CRITIC_DIRECTIVE = {candidate['critic_directive']!r}\n"
        f"PREFILL_COMPUTE_CHUNK_TOKENS = "
        f"{candidate['prefill_compute_chunk_tokens']}\n"
        'SNAPSHOT_MODE = "final_only"\n'
        f"MAX_SEGMENT_SECONDS = {candidate.get('max_segment_seconds', 300.0)!r}\n"
        "REQUIRE_FULL_CONTEXT = True\n"
        "ALLOW_FALLBACK = False\n"
    )


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            value["strategy_parse_mode"] = "json"
            return value
    for block in re.findall(
        r"```(?:json|python)?\s*(.*?)```",
        stripped,
        re.DOTALL | re.IGNORECASE,
    ):
        try:
            value = ast.literal_eval(block.strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            value["strategy_parse_mode"] = "python-literal"
            return value
    target_matches = re.findall(
        r"(?:Targeting\s+Leaf|pending\s+leaf)\*{0,2}\s*:?\s*"
        r"(?:\*\*)?`?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)`?(?:\*\*)?",
        stripped,
        re.IGNORECASE,
    )
    objective_match = re.search(
        r"Objective\*{0,2}\s*:\s*(.+)$",
        stripped,
        re.MULTILINE | re.IGNORECASE,
    )
    steps = [
        re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
        for line in stripped.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+\S", line)
        and "Objective" not in line
        and "Constraint" not in line
    ]
    objective = (
        objective_match.group(1).strip()
        if objective_match is not None
        else ""
    )
    if not objective:
        prose = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip()
            and not line.strip().startswith(("#", "```"))
        ]
        objective = " ".join(prose)
    if not objective:
        raise ValueError("strategy agent returned no usable candidate")
    digest = hashlib.sha256(stripped.encode()).hexdigest()[:12]
    return {
        "candidate_id": f"candidate-prose-{digest}",
        "target_obligation_id": (
            target_matches[-1] if target_matches else ""
        ),
        "hypothesis": objective,
        "plan": {"steps": steps},
        "strategy_parse_mode": "prose",
    }


def parse_research_verdict(output: str, candidate_id: str) -> dict:
    event_matches = re.findall(
        r"^\[autoresearch-verdict\]\s+(\{.*\})\s*$",
        output,
        re.MULTILINE,
    )
    if event_matches:
        fields = json.loads(event_matches[-1])
        if fields.get("candidate_id") != candidate_id:
            raise ValueError("research verdict candidate ID mismatch")
        if fields.get("outcome") not in {
            "SUPPORTED", "FALSIFIED", "DECOMPOSED", "INCONCLUSIVE",
        }:
            raise ValueError("invalid research verdict outcome")
        if (
            len(str(fields.get("evidence", ""))) < 40
            or len(str(fields.get("new_frontier", ""))) < 30
        ):
            raise ValueError("research verdict lacks substantive evidence/frontier")
        return {
            "outcome": fields["outcome"],
            "evidence": fields["evidence"],
            "new_frontier": fields["new_frontier"],
        }
    matches = list(re.finditer(
        r"^(?:critic>\s*)?### AUTORESEARCH_VERDICT\s*$"
        r"(?P<body>.*?)(?=^### |\Z)",
        output,
        re.MULTILINE | re.DOTALL,
    ))
    if not matches:
        raise ValueError("Critic emitted no AUTORESEARCH_VERDICT")
    body = matches[-1].group("body")
    fields = {}
    for name in ("Candidate ID", "Outcome", "Evidence", "New frontier"):
        match = re.search(
            rf"^{re.escape(name)}:\s*(.+)$",
            body,
            re.MULTILINE,
        )
        if not match:
            raise ValueError(f"research verdict missing {name}")
        fields[name] = match.group(1).strip()
    if fields["Candidate ID"] != candidate_id:
        raise ValueError("research verdict candidate ID mismatch")
    if fields["Outcome"] not in {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}:
        raise ValueError("invalid research verdict outcome")
    if len(fields["Evidence"]) < 40 or len(fields["New frontier"]) < 30:
        raise ValueError("research verdict lacks substantive evidence/frontier")
    return {
        "outcome": fields["Outcome"],
        "evidence": fields["Evidence"],
        "new_frontier": fields["New frontier"],
    }


def _pending_leaf_ids(ledger: dict) -> list[str]:
    obligations = ledger.get("obligations", [])
    unresolved = {
        str(item.get("obligation_id", ""))
        for item in obligations
        if item.get("status") == "UNRESOLVED"
    }
    unresolved_parents = {
        str(item.get("parent_id", ""))
        for item in obligations
        if (
            item.get("status") == "UNRESOLVED"
            and item.get("parent_id")
        )
    }
    return sorted(unresolved - unresolved_parents)


def build_strategy_research_state(
    *,
    current: dict,
    ledger: dict,
    results_text: str,
) -> dict:
    target_id = _select_repair_target(current, ledger)
    obligations = {
        str(item.get("obligation_id", "")): item
        for item in ledger.get("obligations", [])
    }
    ancestry = []
    cursor = target_id
    visited = set()
    while cursor and cursor not in visited:
        visited.add(cursor)
        item = obligations[cursor]
        ancestry.append({
            "obligation_id": cursor,
            "statement": item.get("statement", ""),
            "status": item.get("status", ""),
            "parent_id": item.get("parent_id", ""),
            "last_run_id": item.get("last_run_id", ""),
            "last_evidence": item.get("last_evidence", ""),
        })
        cursor = str(item.get("parent_id", ""))
    ancestry.reverse()
    ancestry_ids = {
        item["obligation_id"] for item in ancestry
    }
    relevant_results = []
    if results_text.strip():
        for row in csv.DictReader(
            io.StringIO(results_text),
            delimiter="\t",
        ):
            if row.get("target_obligation_id") not in ancestry_ids:
                continue
            relevant_results.append({
                "candidate_id": row.get("candidate_id", ""),
                "target_obligation_id": row.get(
                    "target_obligation_id",
                    "",
                ),
                "hypothesis_sha256": row.get("hypothesis_sha256", ""),
                "research_outcome": row.get("research_outcome", ""),
                "research_evidence": row.get("research_evidence", ""),
                "new_frontier": row.get("new_frontier", ""),
                "kept": row.get("kept", ""),
                "error": row.get("error", ""),
            })
    return {
        "target_leaf_id": target_id,
        "target_ancestry": ancestry,
        "relevant_experiments": relevant_results,
        "current_candidate": {
            "candidate_id": current.get("candidate_id", ""),
            "target_obligation_id": current.get(
                "target_obligation_id",
                "",
            ),
            "hypothesis": current.get("hypothesis", ""),
            "prefill_compute_chunk_tokens": current.get(
                "prefill_compute_chunk_tokens",
            ),
        },
    }


def build_strategy_prompt(
    *,
    program: str,
    current: dict,
    results_text: str,
    ledger: dict,
) -> str:
    research_state = build_strategy_research_state(
        current=current,
        ledger=ledger,
        results_text=results_text,
    )
    return (
        "You are the AutoResearch strategy agent. Follow the human-owned "
        "program exactly. Attack TARGET_LEAF_ID and propose "
        "one falsifiable GAN strategy experiment. Return JSON only with keys: "
        + ", ".join(REQUIRED_CANDIDATE_FIELDS)
        + ". prefill_compute_chunk_tokens is immutable and must equal "
        f"{current['prefill_compute_chunk_tokens']}. "
        "Do not weaken full context, final-only snapshots, or no-fallback rules."
        " The hypothesis must not repeat any hypothesis hash in RESEARCH_STATE. "
        "It must either construct a concrete object or attempt a concrete "
        "counterexample for the target leaf. target_obligation_id must equal "
        "TARGET_LEAF_ID. Every statement and evidence item below is complete; "
        "do not infer omitted text from unrelated branches."
        f"\n\nPROGRAM:\n{program}"
        "\n\nRESEARCH_STATE:\n"
        f"{json.dumps(research_state, ensure_ascii=False)}"
    )


class StrategyPrefillHeartbeat:
    def __init__(
        self,
        dashboard: str = "http://127.0.0.1:8090",
        interval_s: float = 10.0,
    ) -> None:
        self.dashboard = dashboard.rstrip("/")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: dict = {}
        self._last: tuple | None = None

    def __enter__(self):
        try:
            self._baseline = _json_request(
                f"{self.dashboard}/v1/network/summary",
            ).get("prefill", {})
        except Exception as exc:
            print(
                "[autoresearch] Strategy Prefill telemetry warning: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)
        self._emit()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._emit()

    def _delta(self, current: dict, name: str) -> int:
        return max(
            0,
            int(current.get(name, 0)) - int(self._baseline.get(name, 0)),
        )

    def _emit(self) -> None:
        try:
            current = _json_request(
                f"{self.dashboard}/v1/network/summary",
            ).get("prefill", {})
        except Exception:
            return
        total = self._delta(current, "remote_job_tokens_total")
        computed = self._delta(current, "remote_job_tokens_computed")
        state = (
            computed,
            total,
            self._delta(current, "remote_hits"),
            self._delta(current, "tokens_reused"),
        )
        if not total or state == self._last:
            return
        self._last = state
        percent = min(100.0, 100.0 * computed / total)
        print(
            f"[autoresearch] Strategy Prefill: {computed}/{total} tokens "
            f"({percent:.1f}%) · remote_hits={state[2]} reused={state[3]}",
            flush=True,
        )


def propose_candidate(
    *,
    address: str,
    tokenizer_id: str,
    program: str,
    current: dict,
    results_text: str,
    ledger: dict,
    max_prefill_tokens: int = 4096,
) -> dict:
    from kakeya import Client
    from transformers import AutoTokenizer
    from scripts.chat_grpc import _resolve_eos_token_ids

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    prompt = build_strategy_prompt(
        program=program,
        current=current,
        results_text=results_text,
        ledger=ledger,
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    if len(ids) > max_prefill_tokens:
        raise ValueError(
            "Strategy Prefill token budget exceeded without truncation: "
            f"{len(ids)} > {max_prefill_tokens}",
        )
    generated: list[int] = []
    print(
        f"[autoresearch] Strategy Prefill: 0/{len(ids)} tokens (0.0%)",
        flush=True,
    )
    with Client(address) as client:
        with client.create_session(
            eos_token_ids=_resolve_eos_token_ids(tokenizer),
            client_label="autoresearch-strategy",
        ) as session:
            with StrategyPrefillHeartbeat():
                session.append(ids)
            print(
                f"[autoresearch] Strategy Prefill complete: {len(ids)} tokens",
                flush=True,
            )
            while len(generated) < 2048:
                before = len(generated)
                generated.extend(int(token) for token in session.generate(max_tokens=64))
                print(
                    f"[autoresearch] Strategy Decode: {len(generated)} tokens "
                    f"stop_reason={session.last_stop_reason}",
                    flush=True,
                )
                if session.last_stop_reason != 1:
                    break
                if len(generated) == before:
                    raise RuntimeError("strategy agent made no progress")
            if session.last_stop_reason != 2:
                raise RuntimeError(
                    f"strategy agent did not reach EOS: {session.last_stop_reason}",
                )
    strategy_output = tokenizer.decode(generated, skip_special_tokens=True)
    print(
        f"[autoresearch] Strategy Output: {strategy_output.strip()}",
        flush=True,
    )
    candidate = _extract_json(strategy_output)
    parse_mode = candidate.pop("strategy_parse_mode", "unknown")
    print(
        f"[autoresearch] phase=strategy-parse mode={parse_mode}",
        flush=True,
    )
    candidate, repaired_fields = repair_candidate_schema(
        candidate,
        current=current,
        ledger=ledger,
    )
    if repaired_fields:
        print(
            "[autoresearch] phase=strategy-schema-repair "
            f"fields={','.join(sorted(set(repaired_fields)))} "
            f"target={candidate.get('target_obligation_id', '')}",
            flush=True,
        )
    candidate.update({
        "snapshot_mode": "final_only",
        "max_segment_seconds": 300.0,
        "require_full_context": True,
        "allow_fallback": False,
    })
    validate_candidate(candidate)
    return candidate


def check_runtime_health(
    worker_address: str,
    dashboard: str = "http://127.0.0.1:8090",
) -> dict:
    worker_host, worker_port_text = worker_address.rsplit(":", 1)
    _wait_port(worker_host, int(worker_port_text))
    _wait_port("127.0.0.1", 51051)
    _wait_port("127.0.0.1", 8090)
    summary = _json_request(
        f"{dashboard.rstrip('/')}/v1/network/summary",
    )
    if int(summary.get("online_nodes", 0)) < 1:
        raise RuntimeError("prefill fleet has no online worker")
    return summary


def _backup(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
    else:
        temporary = path.with_suffix(path.suffix + ".restore")
        temporary.write_bytes(content)
        os.chmod(temporary, 0o600)
        temporary.replace(path)


def run_gan_experiment(
    *,
    repo: Path,
    candidate_path: Path,
    state_path: Path,
    timeout_s: float,
) -> tuple[str, str]:
    command = [
        "bash", str(repo / "scripts/run_agent_gan_repl.sh"),
        "--skip-ensure", "--no-auto-loop",
        "--candidate-file", str(candidate_path),
        "--state-file", str(state_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=repo,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write("/continue\n/quit\n")
    process.stdin.flush()
    process.stdin.close()
    timed_out = threading.Event()

    def terminate_on_timeout() -> None:
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout_s, terminate_on_timeout)
    timer.daemon = True
    timer.start()
    lines: list[str] = []
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = process.wait()
    finally:
        timer.cancel()
    output = "".join(lines)
    if timed_out.is_set():
        raise TimeoutError(
            f"GAN experiment exceeded {timeout_s}s: {output[-4000:]}",
        )
    if returncode != 0:
        raise RuntimeError(
            f"GAN experiment failed ({returncode}): {output[-4000:]}",
        )
    matches = re.findall(r"run=(br_[0-9a-f]+)", output)
    if not matches:
        raise RuntimeError("GAN experiment produced no benchmark run id")
    run_id = matches[-1]
    return run_id, output


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def best_kept(results: list[dict]) -> dict | None:
    kept = [row for row in results if row.get("kept") == "True"]
    if not kept:
        return None
    return min(
        kept,
        key=lambda row: (
            int(row["proof_obligations_unresolved"]),
            float(row["metric_cold_critic_prefill_s"]),
        ),
    )


def should_keep(result: dict, baseline: dict | None) -> bool:
    if not result["accepted"]:
        return False
    if result.get("research_outcome") not in {
        "SUPPORTED", "FALSIFIED", "DECOMPOSED",
    }:
        return False
    if not result.get("hypothesis_novel", False):
        return False
    if baseline is None:
        return True
    return int(result["proof_obligations_unresolved"]) <= int(
        baseline["proof_obligations_unresolved"],
    )


RESULT_FIELDS = (
    "timestamp", "experiment_id", "run_id", "candidate_id",
    "target_obligation_id", "constraints_pass", "accepted", "kept",
    "metric_cold_critic_prefill_s", "baseline_metric_s",
    "proof_obligations_total", "proof_obligations_covered",
    "proof_obligations_unresolved", "compute_chunk_tokens",
    "candidate_sha256", "report_path",
    "hypothesis_sha256", "research_outcome", "research_evidence",
    "new_frontier", "transcript_path", "error",
)


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            old_fields = tuple(reader.fieldnames or ())
            old_rows = list(reader)
        if old_fields != RESULT_FIELDS:
            temporary = path.with_suffix(path.suffix + ".migrating")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=RESULT_FIELDS,
                    delimiter="\t",
                )
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({
                        field: old_row.get(field, "")
                        for field in RESULT_FIELDS
                    })
            temporary.replace(path)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def run_iteration(args, iteration: int) -> dict:
    root = Path(__file__).resolve().parents[2]
    ar = Path(__file__).resolve().parent
    candidate_path = ar / "candidate.py"
    results_path = Path(args.results).expanduser()
    reports_dir = Path(args.reports_dir).expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file).expanduser()
    ledger_path = Path(args.proof_ledger).expanduser()
    program = (ar / "program.md").read_text()
    results = read_results(results_path)
    baseline = best_kept(results)
    current_module = _load_candidate(candidate_path)
    current = _candidate_snapshot(current_module)
    previous_candidate = candidate_path.read_bytes()
    previous_state = _backup(state_path)
    previous_ledger = _backup(ledger_path)

    proposed = current
    gan_completed = False
    run_id = ""
    experiment_id = ""
    report_path = reports_dir / "not-started.json"
    transcript_path = reports_dir / "not-started.log"
    hypothesis_sha256 = ""
    candidate_sha256 = hashlib.sha256(previous_candidate).hexdigest()
    try:
        print(
            f"[autoresearch] iteration={iteration} "
            f"phase=runtime-health-check candidate={current['candidate_id']}",
            flush=True,
        )
        health = check_runtime_health(
            args.worker_address,
            args.dashboard,
        )
        print(
            "[autoresearch] phase=runtime-healthy "
            f"online_nodes={health.get('online_nodes', 0)} "
            f"kv_hit_rate={health.get('kv_hit_rate', 0):.1%}",
            flush=True,
        )
        if baseline is None and iteration == 0:
            print(
                "[autoresearch] phase=baseline using current candidate",
                flush=True,
            )
        else:
            print(
                "[autoresearch] phase=strategy-proposal real-gemma",
                flush=True,
            )
            ledger_data = json.loads(ledger_path.read_text())
            proposed = propose_candidate(
                address=args.address,
                tokenizer_id=args.tokenizer_id,
                program=program,
                current=current,
                results_text=(
                    results_path.read_text() if results_path.exists() else ""
                ),
                ledger=ledger_data,
                max_prefill_tokens=args.strategy_max_prefill_tokens,
            )
            if proposed["target_obligation_id"] not in _pending_leaf_ids(
                ledger_data,
            ):
                raise ValueError(
                    "strategy agent targeted a non-leaf proof obligation",
                )
            candidate_path.write_text(render_candidate(proposed))
            print(
                f"[autoresearch] phase=candidate-written "
                f"candidate={proposed['candidate_id']} "
                f"target={proposed['target_obligation_id']}",
                flush=True,
            )
        validate_candidate(proposed)
        hypothesis_sha256 = hashlib.sha256(
            proposed["hypothesis"].strip().lower().encode(),
        ).hexdigest()
        seen_hypotheses = {
            row.get("hypothesis_sha256", "")
            for row in results
            if row.get("hypothesis_sha256")
        }
        if hypothesis_sha256 in seen_hypotheses:
            raise ValueError("strategy agent repeated a previous hypothesis")
        candidate_sha256 = hashlib.sha256(
            candidate_path.read_bytes(),
        ).hexdigest()
        experiment_id = (
            f"ar_{int(time.time())}_{iteration}_"
            f"{hashlib.sha256(candidate_path.read_bytes()).hexdigest()[:8]}"
        )
        report_path = reports_dir / f"{experiment_id}.json"
        transcript_path = reports_dir / f"{experiment_id}.log"
        print(
            f"[autoresearch] phase=gan-experiment id={experiment_id}",
            flush=True,
        )
        run_id, gan_output = run_gan_experiment(
            repo=root,
            candidate_path=candidate_path,
            state_path=state_path,
            timeout_s=args.experiment_timeout_s,
        )
        gan_completed = True
        transcript_path.write_text(gan_output)
        report = _json_request(
            f"http://127.0.0.1:8090/v1/network/benchmarks/{run_id}",
        )
        if report.get("status") != "completed":
            raise RuntimeError(
                f"GAN benchmark is not completed: {report.get('status')}",
            )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        candidate_module = _load_candidate(candidate_path)
        result = evaluate(report, candidate_module)
        verdict = parse_research_verdict(
            gan_output,
            proposed["candidate_id"],
        )
        result.update({
            "research_outcome": verdict["outcome"],
            "research_evidence": verdict["evidence"],
            "new_frontier": verdict["new_frontier"],
            "transcript_path": str(transcript_path),
            "hypothesis_novel": True,
        })
        keep = should_keep(result, baseline)
        print(
            f"[autoresearch] phase=evaluate accepted={result['accepted']} "
            f"unresolved={result['proof_obligations_unresolved']} "
            f"outcome={verdict['outcome']} "
            f"prefill_s={result['metric_cold_critic_prefill_s']:.3f} "
            f"decision={'keep' if keep else 'revert'}",
            flush=True,
        )
        row = {
            "timestamp": time.time(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "candidate_id": proposed["candidate_id"],
            "target_obligation_id": proposed["target_obligation_id"],
            "constraints_pass": result["accepted"],
            "accepted": result["accepted"],
            "kept": keep,
            "metric_cold_critic_prefill_s": result[
                "metric_cold_critic_prefill_s"
            ],
            "baseline_metric_s": (
                baseline["metric_cold_critic_prefill_s"] if baseline else ""
            ),
            "proof_obligations_total": result["proof_obligations_total"],
            "proof_obligations_covered": result["proof_obligations_covered"],
            "proof_obligations_unresolved": result[
                "proof_obligations_unresolved"
            ],
            "compute_chunk_tokens": result["compute_chunk_tokens"],
            "candidate_sha256": candidate_sha256,
            "report_path": str(report_path),
            "hypothesis_sha256": hypothesis_sha256,
            "research_outcome": verdict["outcome"],
            "research_evidence": verdict["evidence"],
            "new_frontier": verdict["new_frontier"],
            "transcript_path": str(transcript_path),
        }
        append_result(results_path, row)
        if not keep:
            candidate_path.write_bytes(previous_candidate)
            print(
                "[autoresearch] phase=candidate-reverted "
                "completed-run-preserved",
                flush=True,
            )
        else:
            print("[autoresearch] phase=kept", flush=True)
        return row
    except Exception as exc:
        print(
            f"[autoresearch] phase=failed error={type(exc).__name__}: {exc}",
            flush=True,
        )
        candidate_path.write_bytes(previous_candidate)
        if not gan_completed:
            _restore(state_path, previous_state)
            _restore(ledger_path, previous_ledger)
        if not gan_completed:
            raise
        row = {
            "timestamp": time.time(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "candidate_id": proposed.get("candidate_id", ""),
            "target_obligation_id": proposed.get("target_obligation_id", ""),
            "constraints_pass": False,
            "accepted": False,
            "kept": False,
            "baseline_metric_s": (
                baseline["metric_cold_critic_prefill_s"] if baseline else ""
            ),
            "compute_chunk_tokens": proposed.get(
                "prefill_compute_chunk_tokens",
                "",
            ),
            "candidate_sha256": candidate_sha256,
            "report_path": str(report_path),
            "hypothesis_sha256": hypothesis_sha256,
            "research_outcome": "EVALUATION_FAILED",
            "transcript_path": str(transcript_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        append_result(results_path, row)
        print(
            f"[autoresearch] phase=completed-run-preserved run={run_id}",
            flush=True,
        )
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--worker-address",
        default="169.254.27.104:53051",
    )
    parser.add_argument("--address", default="127.0.0.1:51051")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--strategy-max-prefill-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--tokenizer-id",
        default=str(
            Path.home()
            / "kakeya-models/gemma-4-26B-A4B-it-mlx-4bit"
        ),
    )
    parser.add_argument(
        "--results",
        default=str(Path.home() / ".kakeya/autoresearch/prefill/results.tsv"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(Path.home() / ".kakeya/autoresearch/prefill/reports"),
    )
    parser.add_argument(
        "--state-file",
        default=str(Path.home() / ".kakeya/agent_gan_state.json"),
    )
    parser.add_argument(
        "--proof-ledger",
        default=str(Path.home() / ".kakeya/agent_gan_proof_ledger.json"),
    )
    parser.add_argument("--experiment-timeout-s", type=float, default=7200)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("iterations must be > 0")
    if args.strategy_max_prefill_tokens <= 0:
        raise SystemExit("strategy-max-prefill-tokens must be > 0")
    for iteration in range(args.iterations):
        row = run_iteration(args, iteration)
        print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
