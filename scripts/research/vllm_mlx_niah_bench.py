#!/usr/bin/env python3
"""vLLM-MLX parallel-NIAH probe (Mac bridge preset ``vllm-mlx-niah``).

Answers one question on the SAME local MLX gemma verifier we used to reproduce
the MLX ``B>1, L=1`` batched-decode recall bug: **is vLLM-MLX both parallel AND
recall-preserving on our config?**

It launches ``vllm-mlx serve --continuous-batching`` on the given model, fires
``--sessions`` concurrent needle-in-a-haystack requests (each with its OWN unique
needle, so a batching/cross-talk bug shows up as a recall drop), and reports:

  * per-session recall (needle found in that session's answer),
  * aggregate decode tok/s at N concurrent vs an N=1 baseline (parallel speedup).

Stdlib only (urllib + threads + subprocess); vLLM-MLX is the served process. The
script ALWAYS writes a verdict JSON (``status`` field) — even on install/load/
server failure — so the bridge round-trip returns a usable answer, not a crash.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    print(f"[vllm-mlx-niah] {msg}", file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _vllm_mlx_version() -> Optional[str]:
    try:
        import importlib.metadata as m

        return m.version("vllm-mlx")
    except Exception:
        return None


def _build_niah_items(sessions: int, haystack_lines: int) -> List[Dict[str, str]]:
    """One independent NIAH item per session, each with a UNIQUE access code."""
    rng = random.Random(1234)
    cities = [
        "Lima", "Oslo", "Cairo", "Tokyo", "Quito", "Accra", "Riga", "Bern",
        "Doha", "Suva", "Male", "Kyiv", "Vienna", "Hanoi", "Sofia", "Dakar",
    ]
    items: List[Dict[str, str]] = []
    for i in range(sessions):
        code = f"{rng.randrange(16**6):06X}"  # unique 6-hex code per session
        filler = [
            f"On day {j}, the courier from {rng.choice(cities)} logged a "
            f"routine delivery of crate {rng.randrange(1000)}."
            for j in range(max(1, haystack_lines))
        ]
        needle = f"IMPORTANT: the access code for vault {i} is {code}."
        pos = rng.randrange(len(filler) + 1)
        filler.insert(pos, needle)
        prompt = (
            "Read the following log carefully.\n\n"
            + "\n".join(filler)
            + f"\n\nQuestion: What is the access code for vault {i}? "
            "Answer with ONLY the code."
        )
        items.append({"prompt": prompt, "code": code, "session": i})
    return items


def _post_chat(
    base_url: str, prompt: str, max_new_tokens: int, timeout: float,
) -> Tuple[bool, str, int, float, str]:
    """POST /v1/chat/completions. Returns (ok, text, completion_tokens, latency, err)."""
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return False, "", 0, time.time() - t0, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - report any transport error
        return False, "", 0, time.time() - t0, f"{type(exc).__name__}: {exc}"
    latency = time.time() - t0
    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except Exception:
        text = ""
    ctoks = 0
    usage = payload.get("usage") or {}
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        ctoks = usage["completion_tokens"]
    if ctoks <= 0:  # fallback estimate if server omits usage
        ctoks = max(1, len(text.split()))
    return True, text, ctoks, latency, ""


def _wait_for_server(
    base_url: str, proc: subprocess.Popen, timeout: float,
) -> Tuple[bool, str]:
    """Poll until the server answers, or it exits, or we time out."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server process exited early (rc={proc.returncode})"
        for path in ("/version", "/v1/models", "/health"):
            try:
                with urllib.request.urlopen(base_url + path, timeout=5) as r:
                    if r.status == 200:
                        return True, path
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    return False, f"timeout after {timeout:.0f}s (last: {last})"


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text("utf-8", "replace").splitlines()[-n:])
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM-MLX parallel NIAH probe")
    ap.add_argument("--model-path", required=True,
                    help="Local MLX model dir (the same gemma verifier as the bug repro).")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--haystack-lines", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--server-timeout", type=float, default=900.0,
                    help="Seconds to wait for model load + server readiness.")
    ap.add_argument("--req-timeout", type=float, default=300.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "kind": "vllm_mlx_niah_parallel",
        "schema_version": 1,
        "status": "init",
        "config": {
            "model_path": args.model_path,
            "sessions": args.sessions,
            "haystack_lines": args.haystack_lines,
            "max_new_tokens": args.max_new_tokens,
            "engine": "vllm-mlx serve --continuous-batching --use-paged-cache",
        },
        "vllm_mlx_version": _vllm_mlx_version(),
    }

    def _flush(status: str, **extra: Any) -> None:
        report["status"] = status
        report.update(extra)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _log(f"status={status}; wrote {out}")

    if report["vllm_mlx_version"] is None:
        _flush("vllm_mlx_not_installed",
               error="`import importlib.metadata; version('vllm-mlx')` failed — "
                     "the pip-install step must run before this script.")
        return 0
    _log(f"vllm-mlx version: {report['vllm_mlx_version']}")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / f"vllm_mlx_serve_{port}.log"
    serve_argv = [
        "vllm-mlx", "serve", args.model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "--continuous-batching", "--use-paged-cache",
        "--max-request-tokens", "32768",
    ]
    _log("launching: " + " ".join(serve_argv))
    proc: Optional[subprocess.Popen] = None
    try:
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(serve_argv, stdout=logf, stderr=subprocess.STDOUT)

        ready, detail = _wait_for_server(base_url, proc, args.server_timeout)
        if not ready:
            _flush("server_failed",
                   error=f"server not ready: {detail}",
                   server_log_tail=_tail(log_path))
            return 0
        _log(f"server ready ({detail})")

        items = _build_niah_items(args.sessions, args.haystack_lines)

        # Warmup (lazy MLX graph compile) — not measured.
        _post_chat(base_url, "Reply with the word ready.", 8, args.req_timeout)

        # N=1 baseline (single request decode tok/s).
        ok0, text0, ct0, lat0, err0 = _post_chat(
            base_url, items[0]["prompt"], args.max_new_tokens, args.req_timeout)
        n1_tps = (ct0 / lat0) if (ok0 and lat0 > 0) else 0.0

        # N concurrent (the parallel path — continuous batching).
        results: List[Optional[Tuple[bool, str, int, float, str]]] = [None] * len(items)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
            futs = {
                ex.submit(_post_chat, base_url, it["prompt"],
                          args.max_new_tokens, args.req_timeout): k
                for k, it in enumerate(items)
            }
            for fut in futs:
                k = futs[fut]
                try:
                    results[k] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[k] = (False, "", 0, 0.0, f"{type(exc).__name__}: {exc}")
        wall = max(time.time() - t0, 1e-6)

        per_session: List[Dict[str, Any]] = []
        hits = 0
        total_ctoks = 0
        n_ok = 0
        for k, (it, res) in enumerate(zip(items, results)):
            ok, text, ctoks, lat, err = res  # type: ignore[misc]
            found = ok and (it["code"] in (text or ""))
            hits += 1 if found else 0
            total_ctoks += ctoks if ok else 0
            n_ok += 1 if ok else 0
            per_session.append({
                "session": it["session"], "ok": ok, "needle_found": found,
                "expected_code": it["code"],
                "answer_excerpt": (text or "")[:80], "completion_tokens": ctoks,
                "latency_s": round(lat, 3), "error": err,
            })

        recall = hits / len(items) if items else 0.0
        agg_tps = total_ctoks / wall
        _flush(
            "ok",
            recall=round(recall, 4),
            sessions_ok=n_ok,
            n1_decode_tps=round(n1_tps, 2),
            aggregate_decode_tps=round(agg_tps, 2),
            parallel_speedup_vs_n1=round(agg_tps / n1_tps, 3) if n1_tps > 0 else None,
            concurrent_wall_s=round(wall, 3),
            total_completion_tokens=total_ctoks,
            per_session=per_session,
            server_log_tail=_tail(log_path, 20),
        )
        verdict = (
            f"recall={recall:.3f} ({hits}/{len(items)}), "
            f"agg_decode={agg_tps:.1f} tok/s, N=1={n1_tps:.1f} tok/s, "
            f"parallel={'YES' if agg_tps > n1_tps else 'no-gain'}"
        )
        _log("VERDICT: " + verdict)
        return 0
    except Exception as exc:  # noqa: BLE001
        _flush("error", error=f"{type(exc).__name__}: {exc}",
               server_log_tail=_tail(log_path))
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
