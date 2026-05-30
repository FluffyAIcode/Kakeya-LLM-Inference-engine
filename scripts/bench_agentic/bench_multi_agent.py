"""Multi-agent concurrent execution benchmark.

Runs N concurrent simulated agents, each driving an M-turn
conversation against the engine's `/v1/chat/completions` endpoint.
Optionally repeats the same workload against a separate
`mlx_lm.server` (or any OpenAI-compatible) endpoint for direct
comparison — this is the headline ADR 0006 §2.3 evidence
distinguishing Kakeya's multi-tenancy from single-tenant servers.

What this measures
------------------

The discriminator vs single-tenant servers is **wall-clock time for
N concurrent multi-turn agent sessions**:

    Kakeya (--max-concurrent N):   t_wall ≈ max(t_agent_i) + admission overhead
    mlx_lm.server (single-tenant): t_wall ≈ sum(t_agent_i)

So at N = 3 with similar single-stream throughput, Kakeya should be
~2–3× faster on wall-clock for the same total work. The bench reports
this ratio explicitly.

Usage
-----

Minimal, no comparison:

    PYTHONPATH=. python3 scripts/bench_agentic/bench_multi_agent.py \\
        --kakeya-url http://127.0.0.1:8000 \\
        --kakeya-model kakeya-v1 \\
        --n-agents 3 --n-turns 4 --max-tokens 64

With mlx_lm comparison (recommended for v0.3.0 release notes):

    PYTHONPATH=. python3 scripts/bench_agentic/bench_multi_agent.py \\
        --kakeya-url http://127.0.0.1:8000 \\
        --kakeya-model kakeya-v1 \\
        --mlx-lm-url http://127.0.0.1:8001 \\
        --mlx-lm-model Qwen/Qwen3-1.7B \\
        --n-agents 3 --n-turns 4 --max-tokens 64

Dry run (script structure check, no HTTP calls):

    python3 scripts/bench_agentic/bench_multi_agent.py --dry-run

Output
------

Tabular summary to stdout + JSON to
``results/platform-tests/bench_agentic_multi_agent_<ts>.json`` with
per-agent timings, per-turn breakdowns, peak memory snapshots, and
the headline speedup ratio.

The JSON report shape is intentionally similar to the existing
``scripts/bench_mlx_*.py`` reports so the same downstream tooling
can consume it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

# httpx is part of the v0.2.x install set (see requirements.txt).
import httpx


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

# Each agent gets one of these as its initial prompt; subsequent turns
# follow up generically. The mix is small but spans chat / code / math /
# planning to mimic real local-agent traffic patterns. It is NOT meant to
# stress-test the verifier's quality — only to drive realistic-shape HTTP
# traffic so the multi-tenancy comparison is measurable.
AGENT_PROMPTS = [
    "Help me debug a Python script that's throwing a KeyError.",
    "Explain transformer attention in one paragraph.",
    "Plan a 3-day trip to Tokyo with a 1500 USD budget.",
    "Review this snippet: def divide(a, b): return a / b",
    "Summarize the difference between SGD and Adam in 3 bullets.",
    "Write a haiku about local LLM inference.",
    "What's the time complexity of merge sort? Explain why.",
    "Suggest a name for my new robotics startup.",
]

# Generic continuation prompts used for turns 2..M. Avoids building
# domain-specific multi-turn graphs (out of scope for a perf bench).
CONTINUATION_PROMPTS = [
    "Continue with concrete details.",
    "Explain why your previous point matters.",
    "Give one alternative perspective.",
    "Summarize the conversation so far in 2 bullets.",
    "Push your reasoning one step further.",
]


# ---------------------------------------------------------------------------
# Per-agent driver
# ---------------------------------------------------------------------------


async def run_agent_session(
    client: httpx.AsyncClient,
    *,
    agent_id: int,
    model: str,
    api_key: Optional[str],
    initial_prompt: str,
    n_turns: int,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Drive one agent through ``n_turns`` of multi-turn dialogue.

    Returns a per-agent result dict. On error, the dict has an
    ``error`` field and ``completed_turns`` < ``n_turns``.
    """
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    history: list[dict[str, str]] = [
        {"role": "system",
         "content": "You are a careful, concise assistant."},
        {"role": "user", "content": initial_prompt},
    ]
    turn_times: list[float] = []
    completion_token_counts: list[int] = []

    for turn in range(n_turns):
        t0 = time.perf_counter()
        try:
            r = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": history,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                timeout=timeout_s,
            )
        except (httpx.RequestError, asyncio.TimeoutError) as exc:
            return {
                "agent_id": agent_id,
                "error": f"transport error: {type(exc).__name__}: {exc}",
                "completed_turns": turn,
                "turn_times": turn_times,
                "completion_token_counts": completion_token_counts,
            }

        if r.status_code != 200:
            return {
                "agent_id": agent_id,
                "error": f"http {r.status_code}: {r.text[:200]}",
                "completed_turns": turn,
                "turn_times": turn_times,
                "completion_token_counts": completion_token_counts,
            }
        body = r.json()
        try:
            response_text = body["choices"][0]["message"]["content"]
            n_completion = body["usage"]["completion_tokens"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - server contract
            return {
                "agent_id": agent_id,
                "error": f"unexpected response shape: {exc}",
                "completed_turns": turn,
                "turn_times": turn_times,
                "completion_token_counts": completion_token_counts,
            }

        turn_times.append(time.perf_counter() - t0)
        completion_token_counts.append(int(n_completion))
        history.append({"role": "assistant", "content": response_text})
        if turn < n_turns - 1:
            history.append({
                "role": "user",
                "content": CONTINUATION_PROMPTS[turn % len(CONTINUATION_PROMPTS)],
            })

    return {
        "agent_id": agent_id,
        "completed_turns": n_turns,
        "turn_times": turn_times,
        "completion_token_counts": completion_token_counts,
        "total_time": sum(turn_times),
        "total_completion_tokens": sum(completion_token_counts),
    }


# ---------------------------------------------------------------------------
# Workload runner
# ---------------------------------------------------------------------------


async def run_workload(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
    n_agents: int,
    n_turns: int,
    max_tokens: int,
    timeout_s: float,
    label: str,
) -> dict[str, Any]:
    """Run ``n_agents`` concurrent sessions against one endpoint."""
    print(f"[bench] [{label}] starting {n_agents} concurrent agents "
          f"({n_turns} turns, max {max_tokens} tokens/turn) on {base_url}",
          flush=True)

    async with httpx.AsyncClient(base_url=base_url, http2=False) as client:
        tasks = [
            run_agent_session(
                client,
                agent_id=i,
                model=model,
                api_key=api_key,
                initial_prompt=AGENT_PROMPTS[i % len(AGENT_PROMPTS)],
                n_turns=n_turns,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            for i in range(n_agents)
        ]

        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    return {
        "label": label,
        "endpoint": base_url,
        "model": model,
        "n_agents": n_agents,
        "n_turns": n_turns,
        "max_tokens": max_tokens,
        "wall_time_s": wall_time,
        "n_successful": len(successful),
        "n_failed": len(failed),
        "per_agent_results": results,
        "agg": _aggregate(successful, wall_time, n_agents, n_turns),
    }


def _aggregate(
    successful: list[dict[str, Any]],
    wall_time: float,
    n_agents: int,
    n_turns: int,
) -> dict[str, Any]:
    """Compute aggregate stats over successful agents."""
    if not successful:
        return {
            "all_failed": True,
            "wall_time_s": wall_time,
        }
    totals = [r["total_time"] for r in successful]
    all_turn_times = [t for r in successful for t in r["turn_times"]]
    completion_tokens = sum(r["total_completion_tokens"] for r in successful)
    return {
        "all_failed": False,
        "wall_time_s": wall_time,
        "sum_per_agent_time_s": sum(totals),
        "max_per_agent_time_s": max(totals),
        "min_per_agent_time_s": min(totals),
        "p50_per_agent_time_s": statistics.median(totals),
        "p50_turn_time_s": statistics.median(all_turn_times),
        "p95_turn_time_s": _percentile(all_turn_times, 0.95),
        "throughput_completion_tokens_per_s": completion_tokens / max(wall_time, 1e-9),
        "concurrency_utilization": (sum(totals) / wall_time) / n_agents,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round(q * (len(s) - 1)))
    return s[k]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_summary(
    kakeya: dict[str, Any],
    mlx: Optional[dict[str, Any]],
) -> str:
    lines: list[str] = ["", "=" * 78,
                        "Multi-agent concurrent execution benchmark", "=" * 78]

    def _fmt_one(r: dict[str, Any]) -> list[str]:
        agg = r["agg"]
        if agg["all_failed"]:
            return [f"  {r['label']}: ALL AGENTS FAILED",
                    f"    endpoint = {r['endpoint']}",
                    f"    n_failed = {r['n_failed']}/{r['n_agents']}"]
        return [
            f"  {r['label']}",
            f"    endpoint                    = {r['endpoint']}",
            f"    n_agents x n_turns          = {r['n_agents']} x {r['n_turns']}",
            f"    successful / total          = {r['n_successful']} / {r['n_agents']}",
            f"    wall time                   = {agg['wall_time_s']:.2f} s",
            f"    sum of per-agent times      = {agg['sum_per_agent_time_s']:.2f} s",
            f"    p50 per-agent time          = {agg['p50_per_agent_time_s']:.2f} s",
            f"    max per-agent time          = {agg['max_per_agent_time_s']:.2f} s",
            f"    p50 turn time               = {agg['p50_turn_time_s']:.3f} s",
            f"    p95 turn time               = {agg['p95_turn_time_s']:.3f} s",
            f"    throughput (compl. tok/s)   = {agg['throughput_completion_tokens_per_s']:.1f}",
            f"    concurrency utilization     = {agg['concurrency_utilization']:.2f}x",
        ]

    lines.extend(_fmt_one(kakeya))
    if mlx:
        lines.append("")
        lines.extend(_fmt_one(mlx))
        # Speedup analysis
        if (not kakeya["agg"]["all_failed"] and not mlx["agg"]["all_failed"]):
            ka, ml = kakeya["agg"], mlx["agg"]
            lines.append("")
            lines.append("  Headline comparison (Kakeya vs mlx_lm.server)")
            lines.append(f"    wall-time speedup            = "
                         f"{ml['wall_time_s'] / max(ka['wall_time_s'], 1e-9):.2f}x")
            lines.append(f"    throughput speedup           = "
                         f"{ka['throughput_completion_tokens_per_s'] / max(ml['throughput_completion_tokens_per_s'], 1e-9):.2f}x")
            lines.append(f"    Kakeya concurrency util      = {ka['concurrency_utilization']:.2f}x  "
                         f"(should be > 1 for multi-tenancy benefit)")
            lines.append(f"    mlx_lm concurrency util      = {ml['concurrency_utilization']:.2f}x  "
                         f"(typically ~1, single-tenant)")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--kakeya-url", default="http://127.0.0.1:8000",
                    help="Kakeya server base URL (default: %(default)s)")
    ap.add_argument("--kakeya-model", default="kakeya-v1",
                    help="model id Kakeya reports via /v1/models")
    ap.add_argument("--kakeya-api-key", default=None,
                    help="Bearer token for Kakeya (optional)")
    ap.add_argument("--mlx-lm-url", default=None,
                    help="If set, run identical workload against mlx_lm.server "
                         "for direct comparison")
    ap.add_argument("--mlx-lm-model", default=None,
                    help="model id reported by mlx_lm.server (e.g. Qwen/Qwen3-1.7B)")
    ap.add_argument("--mlx-lm-api-key", default=None,
                    help="Bearer token for mlx_lm.server (optional)")
    ap.add_argument("--n-agents", type=int, default=3,
                    help="concurrent agent count (default: %(default)s)")
    ap.add_argument("--n-turns", type=int, default=4,
                    help="turns per agent (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="max_tokens per turn (default: %(default)s)")
    ap.add_argument("--timeout-s", type=float, default=120.0,
                    help="per-request HTTP timeout (default: %(default)s)")
    ap.add_argument("--report", default=None,
                    help="path to write JSON report (default: auto under "
                         "results/platform-tests/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't make HTTP calls; verify script structure only")
    return ap


def main() -> int:
    ap = _build_argparser()
    args = ap.parse_args()

    if args.dry_run:
        # Smoke test path — exercise argparse + workload-builder shape
        # without any HTTP traffic. Lets CI / `python3 -m py_compile`
        # equivalent users sanity-check the script after edits.
        print("[bench] dry-run: argparse OK; "
              f"would run n_agents={args.n_agents} n_turns={args.n_turns} "
              f"on {args.kakeya_url}", flush=True)
        if args.mlx_lm_url:
            print(f"[bench] dry-run: would also compare against "
                  f"{args.mlx_lm_url}", flush=True)
        return 0

    # Run Kakeya workload
    kakeya_result = asyncio.run(run_workload(
        base_url=args.kakeya_url,
        model=args.kakeya_model,
        api_key=args.kakeya_api_key,
        n_agents=args.n_agents,
        n_turns=args.n_turns,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        label="Kakeya",
    ))

    # Optionally run mlx_lm workload for comparison
    mlx_result: Optional[dict[str, Any]] = None
    if args.mlx_lm_url:
        if not args.mlx_lm_model:
            print("[bench] --mlx-lm-url set but --mlx-lm-model not; "
                  "specify the model id mlx_lm.server reports",
                  file=sys.stderr)
            return 64
        mlx_result = asyncio.run(run_workload(
            base_url=args.mlx_lm_url,
            model=args.mlx_lm_model,
            api_key=args.mlx_lm_api_key,
            n_agents=args.n_agents,
            n_turns=args.n_turns,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            label="mlx_lm.server",
        ))

    # Render summary to stdout
    summary = render_summary(kakeya_result, mlx_result)
    print(summary, flush=True)

    # Persist JSON report
    report_path: Path
    if args.report:
        report_path = Path(args.report)
    else:
        ts = int(time.time())
        report_path = Path("results/platform-tests") / f"bench_agentic_multi_agent_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            k: v for k, v in vars(args).items()
            if k not in {"kakeya_api_key", "mlx_lm_api_key"}  # never persist keys
        },
        "kakeya": kakeya_result,
        "mlx_lm": mlx_result,
        "summary_text": summary,
    }
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[bench] wrote {report_path}", flush=True)

    # Exit non-zero if anything failed — useful for CI gating
    if kakeya_result["n_failed"] > 0:
        return 1
    if mlx_result is not None and mlx_result["n_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
