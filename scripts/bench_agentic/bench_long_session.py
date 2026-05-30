"""Long-session memory-stability benchmark.

Drives a single long-running multi-turn session against an
OpenAI-compatible chat endpoint (Kakeya by default) for
``--duration-s`` seconds and verifies the headline ADR 0006 §2.3
claim: **with sink+window KV cache the per-session memory footprint
stays flat regardless of how long the agent has been talking**.

What this measures
------------------

For each completed turn we record:

  * wall-clock turn latency (s)
  * completion tokens (from ``usage.completion_tokens``)
  * tokens/s for that turn
  * server-side KV pool state, scraped from ``/metrics``::
        scheduler_active_sessions
        scheduler_pool_in_use
        scheduler_pool_total
        scheduler_pending
        scheduler_kv_live_bytes
  * client-side RSS (best-effort; uses /proc/self/status on Linux,
    psutil if installed, otherwise ``None``)

After the run we compute:

  * memory drift  — KV bytes at hour H minus KV bytes at hour 0,
                    bucketed every 10 minutes
  * latency drift — p50/p95 turn latency, bucketed every 10 minutes
  * pool sanity   — does ``pool_in_use`` go back to 0 between turns?

The single-session model is intentional: the discriminator vs
unbounded-context servers (``mlx_lm.server`` and friends) is that
**Kakeya's KV bytes per session are bounded by sink+window**, so a
session can run for hours without OOM. A single agent driven for
4 hours is therefore the cleanest evidence.

A note on what this bench measures and what it doesn't (per the
analysis of the 2026-05-30 Mac M4 run, ``bench_long_session_mac_
1780130542.aborted.json``):

  * **KV memory** stays bounded across hours (the §2.3 claim). The
    ``scheduler_kv_live_bytes`` gauge is what proves it.
  * **Per-turn latency** does NOT stay bounded. The OpenAI
    chat-completions protocol is stateless: every turn the client
    sends the full history, the server tokenizes it from scratch
    and the verifier prefills the entire prompt, so prefill cost
    grows linearly with history length. Sink+window only bounds
    *generation-phase* memory, not prefill cost. A 30-min run on
    Mac M4 showed p50 turn latency growing from ~15 s to ~55 s as
    history grew from ~50 to ~3700 tokens. This is a **protocol-
    level limitation**, not a memory-stability failure. Cross-
    request KV reuse (a v0.4 feature) is the eventual fix; until
    then, agent applications should manage prompt length via
    summarization or sliding windows.

The bench reports both metrics independently — KV bounded check is
a hard claim, latency drift is a measurement, not a gate.

Usage
-----

Smoke test (no HTTP traffic, structure check only):

    python3 scripts/bench_agentic/bench_long_session.py --dry-run

Short Kakeya run (5 minutes, useful for CI / gating):

    PYTHONPATH=. python3 scripts/bench_agentic/bench_long_session.py \\
        --kakeya-url http://127.0.0.1:8000 \\
        --kakeya-model kakeya-v1 \\
        --duration-s 300 --turn-spacing-s 6 --max-tokens 64

Full 4-hour Mac run (the headline ADR 0006 §2.3 claim):

    PYTHONPATH=. python3 scripts/bench_agentic/bench_long_session.py \\
        --kakeya-url http://127.0.0.1:8000 \\
        --kakeya-model kakeya-v1 \\
        --duration-s 14400 --turn-spacing-s 5 --max-tokens 64 \\
        --report results/platform-tests/bench_long_session_mac_$(date +%s).json

Output
------

Tabular summary to stdout + JSON report to ``--report`` (or an
auto-generated path under ``results/platform-tests/``). The JSON
contains the full per-turn series so the same data can be replotted
later without rerunning.

A live progress line is printed every ``--progress-every-s`` seconds
so a 4-hour run is observable from a terminal without external
monitoring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx


_FOLLOWUPS = [
    "Continue with one more concrete detail.",
    "Push your reasoning one step further.",
    "Re-state the key idea in one sentence.",
    "Suggest a possible failure mode.",
    "What would change if the input doubled in size?",
    "Compare your answer to a textbook treatment.",
    "List one concrete next step.",
    "Summarize the conversation so far in 2 bullets.",
]

_INITIAL_PROMPT = (
    "You are a careful local agent helping me debug a long-running "
    "Python service. Start by asking me one targeted question."
)


# ---------------------------------------------------------------------------
# Memory & metrics scraping
# ---------------------------------------------------------------------------


def _client_rss_bytes() -> Optional[int]:
    """Best-effort client-side RSS in bytes. Returns None if unavailable.

    On Linux we read ``/proc/self/status``; on macOS we fall back to
    ``psutil`` if installed; otherwise None. The benchmark does not
    require this to function — it just enriches the report.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[-1].lower() == "kb":
                        return int(parts[1]) * 1024
    except OSError:
        pass
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # pragma: no cover - psutil not available
        return None


# Names match the ones registered by inference_engine.server.metrics
# (Prometheus-client does NOT add a service prefix). Changing any of
# these breaks the bench's KV-bounded check; if you rename a metric on
# the server, update both ends in the same commit.
_METRIC_NAMES = (
    "scheduler_active_sessions",
    "scheduler_pool_in_use",
    "scheduler_pool_total",
    "scheduler_pending",
    "scheduler_kv_live_bytes",
)


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(?P<value>[-+0-9eE.\.NaNinf]+)\s*$"
)


def _parse_prom_text(body: str) -> dict[str, float]:
    """Tiny Prometheus text parser.

    Only the four metrics we care about are extracted; everything
    else is skipped. Values are coerced to float; ``NaN`` and ``inf``
    are preserved as floats. We do not try to handle multi-label
    series — these are gauges with no labels in our exporter.
    """
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if m is None:
            continue
        name = m.group("name")
        if name not in _METRIC_NAMES:
            continue
        try:
            out[name] = float(m.group("value"))
        except ValueError:  # pragma: no cover - malformed exporter
            continue
    return out


async def _scrape_metrics(
    client: httpx.AsyncClient,
    *,
    metrics_path: str,
    timeout_s: float,
) -> Optional[dict[str, float]]:
    try:
        r = await client.get(metrics_path, timeout=timeout_s)
    except (httpx.RequestError, asyncio.TimeoutError):
        return None
    if r.status_code != 200:
        return None
    return _parse_prom_text(r.text)


# ---------------------------------------------------------------------------
# Long-session driver
# ---------------------------------------------------------------------------


async def _do_one_turn(
    client: httpx.AsyncClient,
    *,
    headers: dict[str, str],
    model: str,
    history: list[dict[str, str]],
    max_tokens: int,
    timeout_s: float,
) -> tuple[Optional[str], int, Optional[str]]:
    """Send one chat-completions request. Returns (text, n_compl, err)."""
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
        return None, 0, f"transport: {type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return None, 0, f"http {r.status_code}: {r.text[:200]}"
    body = r.json()
    try:
        text = body["choices"][0]["message"]["content"]
        n_compl = int(body["usage"]["completion_tokens"])
    except (KeyError, IndexError) as exc:  # pragma: no cover - server contract
        return None, 0, f"unexpected response shape: {exc}"
    return text, n_compl, None


async def run_long_session(
    *,
    base_url: str,
    metrics_path: str,
    model: str,
    api_key: Optional[str],
    duration_s: float,
    turn_spacing_s: float,
    max_tokens: int,
    timeout_s: float,
    progress_every_s: float,
    checkpoint_path: Optional[Path],
) -> dict[str, Any]:
    """Drive a single agent session for ``duration_s`` seconds."""
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    history: list[dict[str, str]] = [
        {"role": "system",
         "content": "You are a careful, concise local agent."},
        {"role": "user", "content": _INITIAL_PROMPT},
    ]

    turns: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    last_progress = t_start
    last_checkpoint = t_start
    turn_idx = 0

    async with httpx.AsyncClient(base_url=base_url, http2=False) as client:
        # Snapshot metrics once before any traffic — establishes
        # baseline for memory drift.
        baseline = await _scrape_metrics(
            client, metrics_path=metrics_path, timeout_s=timeout_s,
        )

        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= duration_s:
                break

            t_turn = time.perf_counter()
            text, n_compl, err = await _do_one_turn(
                client, headers=headers, model=model,
                history=history, max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            latency = time.perf_counter() - t_turn
            metrics = await _scrape_metrics(
                client, metrics_path=metrics_path, timeout_s=timeout_s,
            )
            rss = _client_rss_bytes()

            if err is not None:
                errors.append({
                    "turn": turn_idx, "elapsed_s": elapsed,
                    "error": err, "latency_s": latency,
                })
                # Backoff briefly on transport errors so we don't pin
                # the CPU spamming requests if the server is down.
                await asyncio.sleep(min(turn_spacing_s, 1.0))
                turn_idx += 1
                continue

            assert text is not None  # mypy; err is None on success path
            turns.append({
                "turn": turn_idx,
                "elapsed_s": elapsed,
                "latency_s": latency,
                "completion_tokens": n_compl,
                "tokens_per_s": (n_compl / latency) if latency > 0 else 0.0,
                "metrics": metrics,
                "client_rss_bytes": rss,
            })
            history.append({"role": "assistant", "content": text})
            history.append({
                "role": "user",
                "content": _FOLLOWUPS[turn_idx % len(_FOLLOWUPS)],
            })
            turn_idx += 1

            now = time.perf_counter()
            if now - last_progress >= progress_every_s:
                _print_progress(elapsed, turn_idx, turns, errors, metrics)
                last_progress = now
            if checkpoint_path is not None and now - last_checkpoint >= 60:
                _atomic_write_json(
                    checkpoint_path,
                    _build_payload(
                        turns=turns, errors=errors, baseline=baseline,
                        duration_s=duration_s, turn_spacing_s=turn_spacing_s,
                        partial=True,
                    ),
                )
                last_checkpoint = now

            # Pace the loop. ``turn_spacing_s`` is the *minimum* gap
            # between turn-starts, not between turn-ends — this keeps
            # the per-hour turn count stable even as latency grows.
            sleep_s = max(0.0, turn_spacing_s - latency)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    return _build_payload(
        turns=turns, errors=errors, baseline=baseline,
        duration_s=duration_s, turn_spacing_s=turn_spacing_s,
        partial=False,
    )


def _print_progress(
    elapsed: float,
    turn_idx: int,
    turns: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    metrics: Optional[dict[str, float]],
) -> None:
    last = turns[-1] if turns else None
    last_lat = f"{last['latency_s']:.2f}s" if last else "-"
    kv = (metrics or {}).get("scheduler_kv_live_bytes")
    kv_str = f"{kv / (1024 * 1024):.1f} MiB" if kv is not None else "?"
    print(
        f"[bench] t={elapsed/60:6.1f} min | turns={turn_idx:5d} "
        f"| ok={len(turns):5d} | err={len(errors):3d} "
        f"| last_lat={last_lat} | kv_live={kv_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round(q * (len(s) - 1)))
    return s[k]


def _bucketize(
    turns: list[dict[str, Any]],
    bucket_s: float,
) -> list[dict[str, Any]]:
    """Group turns into ``bucket_s``-second windows and aggregate."""
    if not turns:
        return []
    buckets: dict[int, list[dict[str, Any]]] = {}
    for t in turns:
        idx = int(t["elapsed_s"] // bucket_s)
        buckets.setdefault(idx, []).append(t)
    out: list[dict[str, Any]] = []
    for idx in sorted(buckets):
        bucket_turns = buckets[idx]
        latencies = [b["latency_s"] for b in bucket_turns]
        kv_vals = [
            (b["metrics"] or {}).get("scheduler_kv_live_bytes")
            for b in bucket_turns
        ]
        kv_vals_clean = [v for v in kv_vals if v is not None]
        out.append({
            "bucket_index": idx,
            "bucket_start_s": idx * bucket_s,
            "n_turns": len(bucket_turns),
            "p50_latency_s": statistics.median(latencies),
            "p95_latency_s": _percentile(latencies, 0.95),
            "mean_kv_live_bytes":
                statistics.mean(kv_vals_clean) if kv_vals_clean else None,
            "max_kv_live_bytes":
                max(kv_vals_clean) if kv_vals_clean else None,
        })
    return out


def _aggregate(
    turns: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    if not turns:
        return {
            "all_failed": True,
            "n_turns": 0,
            "n_errors": len(errors),
            "duration_s": duration_s,
        }
    latencies = [t["latency_s"] for t in turns]
    kv_series = [
        (t["metrics"] or {}).get("scheduler_kv_live_bytes")
        for t in turns
    ]
    kv_clean = [v for v in kv_series if v is not None]
    pool_in_use_series = [
        (t["metrics"] or {}).get("scheduler_pool_in_use")
        for t in turns
    ]
    pool_in_use_clean = [v for v in pool_in_use_series if v is not None]

    # Per-bucket aggregates for drift analysis. Default 600s (10 min).
    buckets = _bucketize(turns, 600.0)
    if len(buckets) >= 2:
        first = buckets[0]
        last = buckets[-1]
        latency_drift_s = (last["p50_latency_s"] - first["p50_latency_s"])
        kv_drift_bytes = None
        if (first["mean_kv_live_bytes"] is not None
                and last["mean_kv_live_bytes"] is not None):
            kv_drift_bytes = (
                last["mean_kv_live_bytes"] - first["mean_kv_live_bytes"]
            )
    else:
        latency_drift_s = 0.0
        kv_drift_bytes = 0.0

    return {
        "all_failed": False,
        "n_turns": len(turns),
        "n_errors": len(errors),
        "duration_s": duration_s,
        "p50_latency_s": statistics.median(latencies),
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_latency_s": statistics.mean(latencies),
        "min_kv_live_bytes": min(kv_clean) if kv_clean else None,
        "max_kv_live_bytes": max(kv_clean) if kv_clean else None,
        "mean_kv_live_bytes":
            statistics.mean(kv_clean) if kv_clean else None,
        "kv_bounded": (
            None if not kv_clean
            else (max(kv_clean) - min(kv_clean)) / max(min(kv_clean), 1.0) < 0.10
        ),
        "pool_in_use_max":
            max(pool_in_use_clean) if pool_in_use_clean else None,
        "pool_in_use_settles_to_zero": (
            None if not pool_in_use_clean
            else min(pool_in_use_clean) == 0
        ),
        "latency_drift_p50_s": latency_drift_s,
        "kv_drift_bytes": kv_drift_bytes,
        "buckets_10min": buckets,
    }


def _build_payload(
    *,
    turns: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    baseline: Optional[dict[str, float]],
    duration_s: float,
    turn_spacing_s: float,
    partial: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "partial": partial,
        "duration_s": duration_s,
        "turn_spacing_s": turn_spacing_s,
        "baseline_metrics": baseline,
        "turns": turns,
        "errors": errors,
        "agg": _aggregate(turns, errors, duration_s),
    }


def render_summary(payload: dict[str, Any]) -> str:
    agg = payload["agg"]
    lines = ["", "=" * 78,
             "Long-session memory-stability benchmark",
             "=" * 78]
    if agg.get("all_failed"):
        lines.append("  ALL TURNS FAILED — see 'errors' in JSON for details")
        lines.append(f"  duration_s = {agg['duration_s']:.1f}")
        lines.append(f"  n_errors   = {agg['n_errors']}")
        lines.append("=" * 78)
        return "\n".join(lines)

    def _fmt_bytes(b: Optional[float]) -> str:
        if b is None:
            return "?"
        return f"{b / (1024 * 1024):.1f} MiB"

    lines.extend([
        f"  duration                       = {agg['duration_s']:.1f} s "
        f"({agg['duration_s']/3600:.2f} h)",
        f"  successful turns               = {agg['n_turns']}",
        f"  errored turns                  = {agg['n_errors']}",
        f"  p50 turn latency               = {agg['p50_latency_s']:.3f} s",
        f"  p95 turn latency               = {agg['p95_latency_s']:.3f} s",
        f"  mean turn latency              = {agg['mean_latency_s']:.3f} s",
        f"  KV live bytes  min / mean / max= "
        f"{_fmt_bytes(agg['min_kv_live_bytes'])} / "
        f"{_fmt_bytes(agg['mean_kv_live_bytes'])} / "
        f"{_fmt_bytes(agg['max_kv_live_bytes'])}",
        f"  KV bounded (<10% spread)       = {agg['kv_bounded']}",
        f"  pool_in_use settles to zero    = {agg['pool_in_use_settles_to_zero']}",
        f"  pool_in_use max                = {agg['pool_in_use_max']}",
        f"  latency drift (last-first p50) = {agg['latency_drift_p50_s']:+.3f} s",
        f"  KV drift (last-first 10-min)   = "
        f"{(agg['kv_drift_bytes'] or 0) / (1024 * 1024):+.2f} MiB",
    ])
    if agg["buckets_10min"]:
        lines.append("")
        lines.append("  Per-10min buckets:")
        lines.append("    bucket  n_turns  p50_lat   p95_lat   mean_kv")
        for b in agg["buckets_10min"]:
            mkb = b["mean_kv_live_bytes"]
            lines.append(
                f"    {b['bucket_index']:>5}  "
                f"{b['n_turns']:>7}  "
                f"{b['p50_latency_s']:>7.3f}s  "
                f"{b['p95_latency_s']:>7.3f}s  "
                f"{(mkb / (1024 * 1024)) if mkb is not None else float('nan'):>7.1f} MiB"
            )
    lines.append("=" * 78)
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--kakeya-url", default="http://127.0.0.1:8000")
    ap.add_argument("--kakeya-model", default="kakeya-v1")
    ap.add_argument("--kakeya-api-key", default=None)
    ap.add_argument("--metrics-path", default="/metrics")
    ap.add_argument("--duration-s", type=float, default=14400.0,
                    help="total bench duration in seconds (default: 4h)")
    ap.add_argument("--turn-spacing-s", type=float, default=5.0,
                    help="minimum gap between turn starts")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--progress-every-s", type=float, default=60.0)
    ap.add_argument("--report", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="path for partial-result checkpoints written every "
                         "60s; defaults to alongside --report")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't make HTTP calls; verify script structure only")
    return ap


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Optional[Path]]:
    if args.report:
        report_path = Path(args.report)
    else:
        ts = int(time.time())
        report_path = (
            Path("results/platform-tests")
            / f"bench_long_session_{ts}.json"
        )
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif args.report or True:
        checkpoint = report_path.with_suffix(".partial.json")
    else:  # pragma: no cover - unreachable
        checkpoint = None
    return report_path, checkpoint


def main() -> int:
    args = _build_argparser().parse_args()

    if args.dry_run:
        print(
            f"[bench] dry-run: argparse OK; would drive single agent for "
            f"{args.duration_s:.0f}s @ turn_spacing={args.turn_spacing_s}s "
            f"on {args.kakeya_url}",
            flush=True,
        )
        return 0

    report_path, checkpoint = _resolve_paths(args)
    print(f"[bench] writing checkpoints to {checkpoint}", flush=True)
    print(f"[bench] final report -> {report_path}", flush=True)

    payload = asyncio.run(run_long_session(
        base_url=args.kakeya_url,
        metrics_path=args.metrics_path,
        model=args.kakeya_model,
        api_key=args.kakeya_api_key,
        duration_s=args.duration_s,
        turn_spacing_s=args.turn_spacing_s,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        progress_every_s=args.progress_every_s,
        checkpoint_path=checkpoint,
    ))

    summary = render_summary(payload)
    print(summary, flush=True)
    payload["summary_text"] = summary
    payload["config"] = {
        k: v for k, v in vars(args).items() if k != "kakeya_api_key"
    }
    _atomic_write_json(report_path, payload)
    print(f"\n[bench] wrote {report_path}", flush=True)

    if payload["agg"].get("all_failed"):
        return 1
    if payload["agg"].get("n_errors", 0) > 0:
        # Non-fatal — long sessions can hit transient transport errors.
        # We print but still return success so a 4-hour run doesn't
        # CI-fail on a single blip. CI gating should look at the JSON
        # report's ``agg.kv_bounded`` and drift fields instead.
        print(f"[bench] note: {payload['agg']['n_errors']} transient errors "
              f"(report still considered valid)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
