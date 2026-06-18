"""Parallel-inference benchmark for an oMLX (jundot/omlx) OpenAI server.

Tests the exact capability vllm-mlx could NOT deliver on Gemma-4: serving many
concurrent requests via continuous batching, *correctly* (no cross-request
contamination) and *faster than serial* (real batching, not a queue), without
crashing (vllm-mlx died with a ``shared_kv`` TypeError on batched Gemma-4).

It drives an ALREADY-RUNNING oMLX server (start the menu-bar app / ``omlx``
server and load a Gemma-4 model first) over its OpenAI-compatible HTTP API —
stdlib only (``urllib`` + threads), no SDK needed.

Method (needle-in-a-haystack, one UNIQUE needle per request):
  * Phase SERIAL     — N requests one-at-a-time   → baseline latency/throughput.
  * Phase CONCURRENT — N requests fired together  → wall time, throughput, and
    whether each answer still contains ITS OWN needle (batching correctness).
Verdict: concurrent must (a) not error, (b) preserve per-request correctness,
(c) beat serial wall time (speedup > 1) — i.e. genuine parallel decoding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple


def _post(url: str, payload: Dict[str, Any], *, timeout: float,
          api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        return e.code, {"_error": f"HTTP {e.code}: {body}"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"_error": f"{type(e).__name__}: {e}"}


def _extract_text(obj: Dict[str, Any]) -> str:
    try:
        ch = obj["choices"][0]
        return ch.get("message", {}).get("content") or ch.get("text") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _build_items(n: int, haystack_lines: int) -> List[Dict[str, Any]]:
    items = []
    for i in range(n):
        secret = 10000 + i * 7 + 3            # unique per request
        filler = "\n".join(
            f"Line {j}: the quick brown fox jumps over the lazy dog."
            for j in range(haystack_lines))
        prompt = (
            f"{filler}\n\nIMPORTANT: The secret access code for session "
            f"#{i} is {secret}.\n{filler}\n\n"
            f"Question: What is the secret access code for session #{i}? "
            f"Reply with ONLY the number.")
        items.append({"idx": i, "secret": str(secret), "prompt": prompt})
    return items


def _request(base_url: str, model: str, item: Dict[str, Any], *,
             max_tokens: int, timeout: float, api_key: Optional[str]) -> Dict[str, Any]:
    chat = {
        "model": model, "max_tokens": max_tokens, "temperature": 0.0,
        "messages": [{"role": "user", "content": item["prompt"]}],
    }
    t0 = time.perf_counter()
    status, obj = _post(base_url.rstrip("/") + "/chat/completions", chat,
                        timeout=timeout, api_key=api_key)
    if status != 200 or "_error" in obj:
        # Fallback to /v1/completions (raw prompt) for servers without a chat
        # template on the loaded model.
        comp = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
                "prompt": item["prompt"]}
        status, obj = _post(base_url.rstrip("/") + "/completions", comp,
                            timeout=timeout, api_key=api_key)
    dt = time.perf_counter() - t0
    text = _extract_text(obj)
    usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
    return {
        "idx": item["idx"], "secret": item["secret"], "status": status,
        "latency_s": round(dt, 3), "ok": status == 200 and "_error" not in obj,
        "hit": item["secret"] in text, "completion_tokens":
            usage.get("completion_tokens"), "error": obj.get("_error"),
        "text_head": text[:120],
    }


def _summarize(label: str, results: List[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    ok = [r for r in results if r["ok"]]
    hits = [r for r in ok if r["hit"]]
    toks = sum(r["completion_tokens"] or 0 for r in ok)
    return {
        "phase": label, "n": len(results), "ok": len(ok), "errors": len(results) - len(ok),
        "needle_hits": len(hits), "wall_s": round(wall_s, 3),
        "completion_tokens": toks,
        "throughput_tok_s": round(toks / wall_s, 2) if wall_s > 0 else 0.0,
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 3)
            if results else 0.0,
        "first_errors": [r["error"] for r in results if r["error"]][:3],
    }


def _wait_ready(base_url: str, timeout: float, api_key: Optional[str]) -> bool:
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("OMLX_BASE_URL", ""),
                    help="oMLX OpenAI base, e.g. http://127.0.0.1:10240/v1")
    ap.add_argument("--model", default=os.environ.get("OMLX_MODEL", ""),
                    help="Model id as oMLX exposes it (see GET /v1/models).")
    ap.add_argument("--api-key", default=os.environ.get("OMLX_API_KEY") or None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--haystack-lines", type=int, default=60)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--output", default="results/research/omlx_parallel_bench.json")
    args = ap.parse_args()

    if not args.base_url or not args.model:
        print("ERROR: set --base-url and --model (or OMLX_BASE_URL/OMLX_MODEL). "
              "Start the oMLX server and load a Gemma-4 model first.",
              file=sys.stderr)
        return 2

    print(f"[omlx-bench] base_url={args.base_url} model={args.model} "
          f"concurrency={args.concurrency}", file=sys.stderr)
    if not _wait_ready(args.base_url, 120, args.api_key):
        print("ERROR: oMLX server not reachable at "
              f"{args.base_url}/models within 120s.", file=sys.stderr)
        return 3

    items = _build_items(args.concurrency, args.haystack_lines)
    kw = dict(max_tokens=args.max_tokens, timeout=args.request_timeout,
              api_key=args.api_key)

    # Phase SERIAL.
    t0 = time.perf_counter()
    serial = [_request(args.base_url, args.model, it, **kw) for it in items]
    serial_sum = _summarize("serial", serial, time.perf_counter() - t0)

    # Phase CONCURRENT — fire all at once.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        conc = list(ex.map(
            lambda it: _request(args.base_url, args.model, it, **kw), items))
    conc_sum = _summarize("concurrent", conc, time.perf_counter() - t0)

    speedup = (round(serial_sum["wall_s"] / conc_sum["wall_s"], 2)
               if conc_sum["wall_s"] > 0 else 0.0)
    verdict = {
        "parallel_works": conc_sum["errors"] == 0,
        "correctness_preserved": conc_sum["needle_hits"] == conc_sum["ok"]
            and conc_sum["ok"] == args.concurrency,
        "wall_speedup_vs_serial": speedup,
        "is_real_parallelism": (conc_sum["errors"] == 0
                                and conc_sum["needle_hits"] == args.concurrency
                                and speedup > 1.2),
    }
    report = {
        "kind": "omlx_parallel_bench", "base_url": args.base_url,
        "model": args.model, "concurrency": args.concurrency,
        "serial": serial_sum, "concurrent": conc_sum, "verdict": verdict,
        "results_concurrent": conc,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"serial": serial_sum, "concurrent": conc_sum,
                      "verdict": verdict}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
