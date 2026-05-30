"""Tool-call JSON-reliability benchmark.

Verifies the ADR 0006 §2.3 headline claim — *"the engine emits valid
JSON for tool-call payloads at a parse rate that is reliable enough
for production agent loops"* — by sending a realistic corpus of
tool-call prompts to an OpenAI-compatible chat endpoint and grading
the responses on three axes:

  1. **Parse rate**   — does the response decode as JSON at all?
  2. **Schema rate**  — does the JSON satisfy the required-field
                        and type contract for the requested tool?
  3. **Value rate**   — for tools with explicit constraints (e.g.
                        the city in get_weather must be one of the
                        cities mentioned in the prompt) does the
                        decoded value also satisfy them?

Each grade is reported as a fraction over ``--trials`` × |corpus|.
Failures are persisted verbatim so the JSON report can be replayed
later as a regression corpus.

The bench is **engine-agnostic** — it talks to any
``/v1/chat/completions`` endpoint, so the same script is used to
compare Kakeya against ``mlx_lm.server`` or any vendor API. There is
no Kakeya-specific code path.

Usage
-----

Smoke test (no HTTP traffic):

    python3 scripts/bench_agentic/bench_tool_call_reliability.py --dry-run

Default Kakeya run:

    PYTHONPATH=. python3 scripts/bench_agentic/bench_tool_call_reliability.py \\
        --kakeya-url http://127.0.0.1:8000 \\
        --kakeya-model kakeya-v1 \\
        --trials 5 --max-tokens 128

With mlx_lm comparison:

    PYTHONPATH=. python3 scripts/bench_agentic/bench_tool_call_reliability.py \\
        --kakeya-url http://127.0.0.1:8000 \\
        --kakeya-model kakeya-v1 \\
        --mlx-lm-url http://127.0.0.1:8001 \\
        --mlx-lm-model Qwen/Qwen3-1.7B \\
        --trials 5

Output
------

Stdout summary + JSON report under ``results/platform-tests/``. The
JSON has the full per-trial response, decode result, and grade — the
exact data needed to write release-note tables and to feed alignment
training as a regression set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx


# ---------------------------------------------------------------------------
# Tool corpus
#
# Each tool defines:
#   * a name and an OpenAI-style JSON schema (function-calling shape)
#   * a list of (prompt, expected_args) pairs
#   * a value-validator callable that returns True iff the decoded
#     args match what the prompt asked for
#
# The corpus is intentionally small (5 tools × 4 prompts) to keep a
# default --trials=5 run under a minute on a laptop while still
# spanning the common tool-call shapes (single-string arg, enum arg,
# list arg, nested object arg, numeric range arg).
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a tool-using assistant. When the user asks for an action, "
    "respond with EXACTLY ONE JSON object — no prose, no markdown fence, "
    "no explanation. The object has two top-level fields: "
    '"tool" (string) and "arguments" (object). Match the tool name and '
    "argument keys exactly as specified by the user."
)


@dataclass(frozen=True)
class ToolCase:
    tool_name: str
    prompt: str
    required_fields: tuple[str, ...]
    field_types: dict[str, type]
    validator: Callable[[dict[str, Any]], bool]


def _validate_get_weather_sf(args: dict[str, Any]) -> bool:
    city = args.get("city")
    return isinstance(city, str) and "san francisco" in city.lower()


def _validate_get_weather_tokyo(args: dict[str, Any]) -> bool:
    city = args.get("city")
    return isinstance(city, str) and "tokyo" in city.lower()


def _validate_set_alarm(args: dict[str, Any]) -> bool:
    t = args.get("time")
    if not isinstance(t, str):
        return False
    return bool(re.match(r"^\d{1,2}:\d{2}(\s?(am|pm|AM|PM))?$", t.strip()))


def _validate_search(args: dict[str, Any]) -> bool:
    q = args.get("query")
    if not isinstance(q, str) or not q:
        return False
    return "transformer" in q.lower() or "attention" in q.lower()


def _validate_send_email(args: dict[str, Any]) -> bool:
    to = args.get("to")
    subj = args.get("subject")
    body = args.get("body")
    if not (isinstance(to, str) and isinstance(subj, str)
            and isinstance(body, str)):
        return False
    if "@" not in to:
        return False
    return len(subj) > 0 and len(body) > 0


def _validate_calculate(args: dict[str, Any]) -> bool:
    expr = args.get("expression")
    if not isinstance(expr, str):
        return False
    # The prompt explicitly asks for 17 * 23. Accept any expression
    # that contains both numbers and a multiplication operator.
    return ("17" in expr and "23" in expr and ("*" in expr or "x" in expr.lower()))


CORPUS: tuple[ToolCase, ...] = (
    ToolCase(
        tool_name="get_weather",
        prompt=(
            "I'm in San Francisco and want to know the current "
            'weather. Call the get_weather tool with the "city" arg.'
        ),
        required_fields=("city",),
        field_types={"city": str},
        validator=_validate_get_weather_sf,
    ),
    ToolCase(
        tool_name="get_weather",
        prompt=(
            "Check the weather in Tokyo right now. Use the "
            'get_weather tool; argument key is "city".'
        ),
        required_fields=("city",),
        field_types={"city": str},
        validator=_validate_get_weather_tokyo,
    ),
    ToolCase(
        tool_name="set_alarm",
        prompt=(
            'Set an alarm for 7:30 am. Call set_alarm with arg "time" '
            'as a string like "7:30 am".'
        ),
        required_fields=("time",),
        field_types={"time": str},
        validator=_validate_set_alarm,
    ),
    ToolCase(
        tool_name="search",
        prompt=(
            'Find recent papers about transformer attention. Use the '
            '"search" tool; the only arg is "query" (string).'
        ),
        required_fields=("query",),
        field_types={"query": str},
        validator=_validate_search,
    ),
    ToolCase(
        tool_name="send_email",
        prompt=(
            "Send an email to alice@example.com with subject 'Lunch?' "
            "and body 'Want to grab lunch tomorrow at 1pm?'. Use the "
            'send_email tool; args are "to", "subject", "body".'
        ),
        required_fields=("to", "subject", "body"),
        field_types={"to": str, "subject": str, "body": str},
        validator=_validate_send_email,
    ),
    ToolCase(
        tool_name="calculate",
        prompt=(
            'Compute 17 * 23 and return it via the calculate tool. '
            'Argument is "expression" (string).'
        ),
        required_fields=("expression",),
        field_types={"expression": str},
        validator=_validate_calculate,
    ),
)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_first_json_object(text: str) -> Optional[str]:
    """Pull the first JSON object out of ``text``.

    A well-behaved model returns a single JSON object. A
    less-well-behaved model wraps it in ``\\`\\`\\`json`` fences or
    prefixes prose. We accept both shapes by:

      1. Trying ``text`` directly as JSON.
      2. Looking for the first fenced ``{ ... }`` block.
      3. Falling back to a brace-balanced scan from the first ``{``.

    Returns the raw JSON substring (still a string) or ``None`` if
    no balanced object can be found.
    """
    s = text.strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    case_index: int
    tool_name: str
    trial_index: int
    raw_response: str
    extracted_json: Optional[str]
    parsed: Optional[dict[str, Any]]
    parse_ok: bool
    schema_ok: bool
    value_ok: bool
    latency_s: float
    error: Optional[str] = None


def _grade(
    case: ToolCase, raw: str
) -> tuple[Optional[str], Optional[dict[str, Any]], bool, bool, bool]:
    """Return ``(extracted, parsed, parse_ok, schema_ok, value_ok)``.

    Each subsequent flag implies the previous: a False parse_ok
    forces schema_ok and value_ok to False.
    """
    extracted = extract_first_json_object(raw)
    if extracted is None:
        return None, None, False, False, False
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        return extracted, None, False, False, False
    parse_ok = isinstance(parsed, dict)
    if not parse_ok:
        return extracted, None, False, False, False

    # Schema check: tool name must match, all required fields must be
    # present in arguments with the declared type.
    args = parsed.get("arguments")
    tool = parsed.get("tool")
    schema_ok = (
        isinstance(tool, str)
        and tool == case.tool_name
        and isinstance(args, dict)
        and all(f in args for f in case.required_fields)
        and all(
            isinstance(args[f], case.field_types[f])
            for f in case.required_fields
        )
    )
    value_ok = False
    if schema_ok:
        try:
            value_ok = bool(case.validator(args))
        except Exception:  # pragma: no cover - defensive for stray args
            value_ok = False
    return extracted, parsed, True, schema_ok, value_ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _do_one_trial(
    client: httpx.AsyncClient,
    *,
    headers: dict[str, str],
    model: str,
    case: ToolCase,
    case_idx: int,
    trial_idx: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> TrialResult:
    body_json = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case.prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(
            "/v1/chat/completions", headers=headers,
            json=body_json, timeout=timeout_s,
        )
    except (httpx.RequestError, asyncio.TimeoutError) as exc:
        return TrialResult(
            case_index=case_idx, tool_name=case.tool_name,
            trial_index=trial_idx, raw_response="",
            extracted_json=None, parsed=None,
            parse_ok=False, schema_ok=False, value_ok=False,
            latency_s=time.perf_counter() - t0,
            error=f"transport: {type(exc).__name__}: {exc}",
        )
    latency = time.perf_counter() - t0
    if r.status_code != 200:
        return TrialResult(
            case_index=case_idx, tool_name=case.tool_name,
            trial_index=trial_idx, raw_response="",
            extracted_json=None, parsed=None,
            parse_ok=False, schema_ok=False, value_ok=False,
            latency_s=latency,
            error=f"http {r.status_code}: {r.text[:200]}",
        )
    try:
        raw = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:  # pragma: no cover
        return TrialResult(
            case_index=case_idx, tool_name=case.tool_name,
            trial_index=trial_idx, raw_response="",
            extracted_json=None, parsed=None,
            parse_ok=False, schema_ok=False, value_ok=False,
            latency_s=latency,
            error=f"unexpected shape: {exc}",
        )

    extracted, parsed, parse_ok, schema_ok, value_ok = _grade(case, raw)
    return TrialResult(
        case_index=case_idx, tool_name=case.tool_name,
        trial_index=trial_idx, raw_response=raw,
        extracted_json=extracted, parsed=parsed,
        parse_ok=parse_ok, schema_ok=schema_ok, value_ok=value_ok,
        latency_s=latency,
    )


async def run_workload(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
    trials: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    label: str,
) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    print(f"[bench] [{label}] starting tool-call grading "
          f"({len(CORPUS)} cases x {trials} trials = "
          f"{len(CORPUS) * trials} requests) on {base_url}",
          flush=True)

    results: list[TrialResult] = []
    async with httpx.AsyncClient(base_url=base_url, http2=False) as client:
        t0 = time.perf_counter()
        # Sequential by design: agent reliability is per-call, not
        # concurrency-dependent. Keeps the bench independent of how
        # many slabs the engine has.
        for case_idx, case in enumerate(CORPUS):
            for trial_idx in range(trials):
                result = await _do_one_trial(
                    client, headers=headers, model=model, case=case,
                    case_idx=case_idx, trial_idx=trial_idx,
                    max_tokens=max_tokens, temperature=temperature,
                    timeout_s=timeout_s,
                )
                results.append(result)
        wall_time = time.perf_counter() - t0

    return _aggregate(results, label, base_url, model, wall_time)


def _aggregate(
    results: list[TrialResult],
    label: str,
    base_url: str,
    model: str,
    wall_time: float,
) -> dict[str, Any]:
    n = len(results)
    parse_ok = sum(1 for r in results if r.parse_ok)
    schema_ok = sum(1 for r in results if r.schema_ok)
    value_ok = sum(1 for r in results if r.value_ok)
    transport_errors = [r for r in results if r.error]
    latencies = [r.latency_s for r in results if r.error is None]

    # Per-tool breakdown
    by_tool: dict[str, dict[str, int]] = {}
    for r in results:
        b = by_tool.setdefault(r.tool_name, {
            "n": 0, "parse_ok": 0, "schema_ok": 0, "value_ok": 0,
            "errors": 0,
        })
        b["n"] += 1
        if r.error:
            b["errors"] += 1
        if r.parse_ok:
            b["parse_ok"] += 1
        if r.schema_ok:
            b["schema_ok"] += 1
        if r.value_ok:
            b["value_ok"] += 1

    return {
        "label": label,
        "endpoint": base_url,
        "model": model,
        "n_trials": n,
        "wall_time_s": wall_time,
        "parse_rate": parse_ok / n if n else 0.0,
        "schema_rate": schema_ok / n if n else 0.0,
        "value_rate": value_ok / n if n else 0.0,
        "transport_error_count": len(transport_errors),
        "p50_latency_s": statistics.median(latencies) if latencies else 0.0,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "by_tool": by_tool,
        "trials": [r.__dict__ for r in results],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_summary(
    kakeya: dict[str, Any], mlx: Optional[dict[str, Any]],
) -> str:
    lines = ["", "=" * 78,
             "Tool-call JSON-reliability benchmark", "=" * 78]

    def _fmt_one(r: dict[str, Any]) -> list[str]:
        out = [
            f"  {r['label']}",
            f"    endpoint        = {r['endpoint']}",
            f"    model           = {r['model']}",
            f"    trials          = {r['n_trials']}",
            f"    wall time       = {r['wall_time_s']:.2f} s",
            f"    parse rate      = {r['parse_rate']*100:6.2f} %  "
            "(decodes as JSON object)",
            f"    schema rate     = {r['schema_rate']*100:6.2f} %  "
            "(tool name + required fields + types)",
            f"    value rate      = {r['value_rate']*100:6.2f} %  "
            "(decoded values match prompt)",
            f"    transport errs  = {r['transport_error_count']}",
            f"    p50 latency     = {r['p50_latency_s']:.3f} s",
            "    by tool:",
        ]
        for tool, b in sorted(r["by_tool"].items()):
            out.append(
                f"      {tool:>14s}  n={b['n']:>3d}  "
                f"parse={b['parse_ok']:>3d}  "
                f"schema={b['schema_ok']:>3d}  "
                f"value={b['value_ok']:>3d}  "
                f"errors={b['errors']:>2d}"
            )
        return out

    lines.extend(_fmt_one(kakeya))
    if mlx:
        lines.append("")
        lines.extend(_fmt_one(mlx))
        lines.append("")
        lines.append("  Headline comparison:")
        for k in ("parse_rate", "schema_rate", "value_rate"):
            lines.append(
                f"    {k:>14s}  Kakeya={kakeya[k]*100:6.2f}%  "
                f"vs  mlx_lm={mlx[k]*100:6.2f}%  "
                f"(diff {(kakeya[k] - mlx[k])*100:+.2f} pp)"
            )
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
    ap.add_argument("--kakeya-url", default="http://127.0.0.1:8000")
    ap.add_argument("--kakeya-model", default="kakeya-v1")
    ap.add_argument("--kakeya-api-key", default=None)
    ap.add_argument("--mlx-lm-url", default=None)
    ap.add_argument("--mlx-lm-model", default=None)
    ap.add_argument("--mlx-lm-api-key", default=None)
    ap.add_argument("--trials", type=int, default=5,
                    help="trials per case (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; 0.0 (default) = "
                         "greedy, the most-charitable setting for "
                         "reliability claims")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="don't make HTTP calls; verify script structure only")
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    if args.dry_run:
        print(
            f"[bench] dry-run: argparse OK; would run "
            f"{len(CORPUS) * args.trials} trials on {args.kakeya_url}",
            flush=True,
        )
        if args.mlx_lm_url:
            print(f"[bench] dry-run: would also compare against "
                  f"{args.mlx_lm_url}", flush=True)
        # Sanity check the corpus extractor offline so a regression
        # in extract_first_json_object surfaces immediately.
        sample = '{"tool": "x", "arguments": {"y": 1}}'
        ex = extract_first_json_object(sample)
        if ex != sample:
            print(f"[bench] FAIL: extractor returned {ex!r}", file=sys.stderr)
            return 1
        return 0

    kakeya_result = asyncio.run(run_workload(
        base_url=args.kakeya_url,
        model=args.kakeya_model,
        api_key=args.kakeya_api_key,
        trials=args.trials,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        label="Kakeya",
    ))

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
            trials=args.trials,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            label="mlx_lm.server",
        ))

    summary = render_summary(kakeya_result, mlx_result)
    print(summary, flush=True)

    if args.report:
        report_path = Path(args.report)
    else:
        ts = int(time.time())
        report_path = (
            Path("results/platform-tests")
            / f"bench_tool_call_reliability_{ts}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            k: v for k, v in vars(args).items()
            if k not in {"kakeya_api_key", "mlx_lm_api_key"}
        },
        "kakeya": kakeya_result,
        "mlx_lm": mlx_result,
        "summary_text": summary,
    }
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[bench] wrote {report_path}", flush=True)

    # Non-zero exit if Kakeya transport-failed every trial. Don't
    # gate on parse_rate < 1.0: that is the *measurement*, not a
    # contract, and a 99.6% rate at sample size 30 is still a useful
    # report.
    if kakeya_result["n_trials"] == kakeya_result["transport_error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
