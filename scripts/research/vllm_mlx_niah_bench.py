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


def _get_model_id(base_url: str) -> Optional[str]:
    """Resolve the served model id from /v1/models (the MLX verifier path)."""
    try:
        with urllib.request.urlopen(base_url + "/v1/models", timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = data.get("data") or []
        if models and isinstance(models[0], dict) and models[0].get("id"):
            return str(models[0]["id"])
    except Exception:
        pass
    return None


def _extract(payload: Dict[str, Any]) -> Tuple[str, int]:
    """Pull text + completion_tokens from a completions OR chat-completions body."""
    text = ""
    try:
        ch = payload["choices"][0]
        text = ch.get("text") or (ch.get("message") or {}).get("content") or ""
    except Exception:
        text = ""
    ctoks = 0
    usage = payload.get("usage") or {}
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        ctoks = usage["completion_tokens"]
    if ctoks <= 0:
        ctoks = max(1, len((text or "").split()))
    return text or "", ctoks


def _post(base_url: str, path: str, body: Dict[str, Any], timeout: float):
    req = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_generate(
    base_url: str, model_id: str, prompt: str, max_new_tokens: int, timeout: float,
) -> Tuple[bool, str, int, float, str, str]:
    """Generate via /v1/completions (raw prompt), falling back to chat.

    Returns (ok, text, completion_tokens, latency, err, endpoint). The MLX
    verifier is a raw checkpoint (no chat template) so /v1/completions is the
    primary path; chat is a fallback for instruct builds.
    """
    t0 = time.time()
    # gemma-4-IT needs its chat template; a raw /v1/completions prompt makes the
    # instruct model emit <end_of_turn> immediately (empty answer). So try chat
    # first (server applies the template), then fall back to /v1/completions with
    # the gemma turn markers wrapped manually.
    gemma = (
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    )
    attempts = (
        ("/v1/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens), "temperature": 0.0,
        }),
        ("/v1/completions", {
            "model": model_id, "prompt": gemma,
            "max_tokens": int(max_new_tokens), "temperature": 0.0,
            "stop": ["<end_of_turn>"],
        }),
    )
    last_err = ""
    for path, body in attempts:
        try:
            payload = _post(base_url, path, body, timeout)
        except urllib.error.HTTPError as exc:
            last_err = f"{path} HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:200]}"
            continue  # try the next endpoint shape
        except Exception as exc:  # noqa: BLE001
            return False, "", 0, time.time() - t0, f"{path}: {type(exc).__name__}: {exc}", path
        text, ctoks = _extract(payload)
        return True, text, ctoks, time.time() - t0, "", path
    return False, "", 0, time.time() - t0, last_err, "none"


def _wait_for_server(
    base_url: str, proc: subprocess.Popen, timeout: float,
) -> Tuple[bool, str]:
    """Poll until the server answers, or it exits, or we time out."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server process exited early (rc={proc.returncode})"
        for path in ("/v1/models", "/version", "/health"):
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

        model_id = _get_model_id(base_url) or "default"
        report["served_model_id"] = model_id
        _log(f"served model id: {model_id}")

        items = _build_niah_items(args.sessions, args.haystack_lines)

        # Warmup (lazy MLX graph compile) — not measured.
        _post_generate(base_url, model_id, "Reply with the word ready.",
                       8, args.req_timeout)

        # N=1 baseline (single request decode tok/s).
        ok0, text0, ct0, lat0, err0, ep0 = _post_generate(
            base_url, model_id, items[0]["prompt"], args.max_new_tokens,
            args.req_timeout)
        n1_tps = (ct0 / lat0) if (ok0 and lat0 > 0) else 0.0
        report["endpoint_used"] = ep0
        if not ok0:
            report["n1_error"] = err0

        # N concurrent (the parallel path — continuous batching).
        results: List[Optional[Tuple[bool, str, int, float, str, str]]] = [None] * len(items)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
            futs = {
                ex.submit(_post_generate, base_url, model_id, it["prompt"],
                          args.max_new_tokens, args.req_timeout): k
                for k, it in enumerate(items)
            }
            for fut in futs:
                k = futs[fut]
                try:
                    results[k] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[k] = (False, "", 0, 0.0, f"{type(exc).__name__}: {exc}", "none")
        wall = max(time.time() - t0, 1e-6)

        per_session: List[Dict[str, Any]] = []
        hits = 0
        total_ctoks = 0
        n_ok = 0
        for k, (it, res) in enumerate(zip(items, results)):
            ok, text, ctoks, lat, err, _ep = res  # type: ignore[misc]
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
