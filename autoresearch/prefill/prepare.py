#!/usr/bin/env python3
"""Fixed evaluation harness for Karpathy-style Prefill autoresearch."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("prefill_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(report: dict, candidate) -> dict:
    stages = report.get("stages", [])
    critic = next(
        (stage for stage in stages if stage.get("name") == "agent_critic"),
        None,
    )
    if critic is None:
        raise ValueError("report has no Critic stage")
    prefix_tokens = int(critic.get("prefix_tokens", 0))
    warmup_s = float(critic.get("warmup_wall_s", 0))
    measured_tps = prefix_tokens / warmup_s if warmup_s > 0 else 0.0
    estimated_max_segment_s = (
        candidate.PREFILL_COMPUTE_CHUNK_TOKENS / measured_tps
        if measured_tps > 0 else float("inf")
    )
    delta = critic.get("delta", {})
    constraints = {
        "stage_ok": bool(critic.get("ok")),
        "complete": bool(critic.get("complete")),
        "full_context": (
            critic.get("review_scope") == "full"
            and int(critic.get("critic_omitted_tokens", -1)) == 0
            and int(critic.get("critic_context_tokens", -1))
            == int(critic.get("generator_full_tokens", -2))
        ),
        "recursive_protocol": (
            critic.get("critic_protocol")
            == "goal_anchored_recursive_gan_v3"
        ),
        "no_fallback": int(delta.get("fallbacks", 0)) == 0,
        "no_job_failure": int(delta.get("remote_job_failures", 0)) == 0,
        "segment_under_budget": (
            estimated_max_segment_s <= candidate.MAX_SEGMENT_SECONDS
        ),
        "final_only_snapshot": candidate.SNAPSHOT_MODE == "final_only",
    }
    return {
        "accepted": all(constraints.values()),
        "metric_cold_critic_prefill_s": warmup_s,
        "measured_prefill_tps": measured_tps,
        "estimated_max_segment_s": estimated_max_segment_s,
        "compute_chunk_tokens": candidate.PREFILL_COMPUTE_CHUNK_TOKENS,
        "constraints": constraints,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).with_name("candidate.py"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).with_name("results.tsv"),
    )
    args = parser.parse_args()
    candidate = _load_candidate(args.candidate)
    result = evaluate(json.loads(args.report.read_text()), candidate)
    write_header = not args.results.exists()
    with args.results.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "timestamp",
                "accepted",
                "metric_cold_critic_prefill_s",
                "measured_prefill_tps",
                "estimated_max_segment_s",
                "compute_chunk_tokens",
            ),
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({"timestamp": time.time(), **{
            key: result[key] for key in writer.fieldnames if key != "timestamp"
        }})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
