"""Agent-connection capacity load test for the Kakeya gRPC RuntimeService.

Test case 1 (cloud-agent ⇄ Mac mini via the Mac bridge): simulate N
concurrent "agents" — each an independent gRPC channel + session — against a
single running ``RuntimeService`` and find the maximum number of concurrent
agent connections the node sustains, plus the bounded per-session KV residency
and the resulting whole-node KV upper bound.

What this measures (and what it does NOT):

* Measures: concurrent **session/connection admission** scaling — how many
  independent gRPC channels + open sessions the node holds at once, the
  create/generate latency curve vs concurrency, server RSS growth, the
  per-session bounded KV (``GetSessionInfo.kv_live_bytes``), and the admission
  semantics at ``--capacity`` (LRU eviction / ``RESOURCE_EXHAUSTED``).
* Does NOT measure parallel-inference throughput: v0.3 is single-tenant — one
  shared verifier, RPC handlers serialized on one asyncio loop (per-session
  verifier binding is deferred to v0.4 / PR-A3c). Concurrent ``Generate`` calls
  therefore serialize; the latency curve reflects that and is reported as such.

The server is launched as a subprocess (mirrors real deployment); clients are
threads using the Python SDK. Self-contained: one process, one JSON report.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _raise_fd_limit(target: int = 100_000) -> Dict[str, Any]:
    """Best-effort raise of RLIMIT_NOFILE so the parent (and the server
    subprocess it spawns, which inherits the soft limit) can hold many
    concurrent gRPC channels. Returns the before/after for the report."""
    info: Dict[str, Any] = {"requested": target}
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        info["before"] = [soft, hard]
        new_soft = hard if hard not in (-1, resource.RLIM_INFINITY) else target
        new_soft = min(new_soft, target) if new_soft > 0 else target
        new_hard = hard
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, new_hard))
        except (ValueError, OSError):
            # Some platforms forbid raising; fall back to the hard cap.
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, hard), hard))
        info["after"] = list(resource.getrlimit(resource.RLIMIT_NOFILE))
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _rss_mb(pid: int) -> Optional[float]:
    """Resident set size in MB for a pid, via ``ps`` (Linux + macOS)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        )
        kb = int(out.stdout.strip().split()[0])
        return round(kb / 1024.0, 1)
    except Exception:
        return None


def _pctl(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 4)


def _wait_ready(address: str, timeout_s: float) -> bool:
    """Poll until a CreateSession+close round-trips (server is serving)."""
    from kakeya import Client
    from kakeya.errors import KakeyaError

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            c = Client(address)
            s = c.create_session(client_label="readyprobe")
            s.close()
            c.close()
            return True
        except (KakeyaError, Exception):  # noqa: BLE001 - readiness poll
            time.sleep(2.0)
    return False


class _AgentResult:
    __slots__ = ("created", "error", "create_s", "gen_s", "kv_bytes", "gen_tokens")

    def __init__(self) -> None:
        self.created = False
        self.error: Optional[str] = None
        self.create_s: Optional[float] = None
        self.gen_s: Optional[float] = None
        self.kv_bytes: Optional[int] = None
        self.gen_tokens: int = 0


def _run_level(
    address: str, n: int, prompt_ids: List[int], gen_tokens: int,
    seed: int,
) -> List[_AgentResult]:
    """Open ``n`` concurrent agents; hold all sessions open simultaneously,
    then generate concurrently. Returns per-agent results."""
    from kakeya import Client
    from kakeya.errors import KakeyaError

    results = [_AgentResult() for _ in range(n)]
    created_barrier = threading.Barrier(n, timeout=max(60.0, n * 1.0))
    clients: List[Any] = [None] * n
    sessions: List[Any] = [None] * n

    def worker(i: int) -> None:
        r = results[i]
        try:
            t0 = time.time()
            c = Client(address)
            sess = c.create_session(client_label=f"agent-{i}")
            # Prefill a per-agent context (chunked) so each session carries a
            # realistic KV footprint up to the verifier's bounded window.
            for off in range(0, len(prompt_ids), 256):
                sess.append(prompt_ids[off:off + 256])
            r.create_s = time.time() - t0
            r.created = True
            clients[i] = c
            sessions[i] = sess
        except KakeyaError as exc:
            r.error = type(exc).__name__
            return
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}:{exc}"[:120]
            return
        # Hold until every agent in this level has its session open, so the
        # node is genuinely holding N concurrent connections at the peak.
        try:
            created_barrier.wait()
        except threading.BrokenBarrierError:
            pass
        try:
            t1 = time.time()
            toks = list(sess.generate(max_tokens=gen_tokens, seed=seed))
            r.gen_s = time.time() - t1
            r.gen_tokens = len(toks)
            r.kv_bytes = sess.info().kv_live_bytes
        except Exception as exc:  # noqa: BLE001
            r.error = (r.error or "") + f"|gen:{type(exc).__name__}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(n):
        try:
            if sessions[i] is not None:
                sessions[i].close()
            if clients[i] is not None:
                clients[i].close()
        except Exception:  # noqa: BLE001
            pass
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "mlx"])
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--capacity", type=int, default=512,
                    help="server SessionStore + SlabPool size (the admission cap)")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128,256",
                    help="comma-separated concurrent-agent counts to ramp")
    ap.add_argument("--gen-tokens", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--context-len", type=int, default=0,
                    help="per-agent prefill length (tokens); overrides "
                         "--prompt-len when >0. Fills each session's KV up to "
                         "the bounded window to probe the memory ceiling.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-ready-timeout", type=float, default=600.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    fd_info = _raise_fd_limit()
    print(f"[loadtest] RLIMIT_NOFILE: {fd_info}", flush=True)

    port = _free_port()
    address = f"127.0.0.1:{port}"
    server_cmd = [
        sys.executable, "scripts/start_grpc_runtime_server.py",
        "--backend", args.backend,
        "--verifier-id", args.verifier_id,
        "--bind", address,
        "--capacity", str(args.capacity),
        "--sink", str(args.sink),
        "--window", str(args.window),
        "--skip-cache-check",
        "--log-level", "WARNING",
    ]
    print(f"[loadtest] launching server: {' '.join(server_cmd)}", flush=True)
    server = subprocess.Popen(server_cmd)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    prefill_len = args.context_len if args.context_len > 0 else args.prompt_len
    # Cycle small valid token ids to reach the requested prefill length.
    prompt_ids = [1 + (j % 64) for j in range(prefill_len)]
    rows: List[Dict[str, Any]] = []
    peak_rss = 0.0
    try:
        if not _wait_ready(address, args.server_ready_timeout):
            print("[loadtest] ERROR: server never became ready", flush=True)
            return 2
        print(f"[loadtest] server ready on {address} "
              f"(backend={args.backend}, capacity={args.capacity})", flush=True)

        for n in levels:
            t0 = time.time()
            res = _run_level(address, n, prompt_ids, args.gen_tokens, args.seed)
            wall = time.time() - t0
            rss = _rss_mb(server.pid)
            if rss:
                peak_rss = max(peak_rss, rss)
            created = sum(1 for r in res if r.created)
            gen_ok = sum(1 for r in res if r.gen_tokens > 0)
            errs: Dict[str, int] = {}
            for r in res:
                if r.error:
                    key = r.error.split("|")[0].split(":")[0]
                    errs[key] = errs.get(key, 0) + 1
            create_lat = [r.create_s for r in res if r.create_s is not None]
            gen_lat = [r.gen_s for r in res if r.gen_s is not None]
            kvs = [r.kv_bytes for r in res if r.kv_bytes is not None]
            row = {
                "agents": n,
                "created_ok": created,
                "generate_ok": gen_ok,
                "errors": errs,
                "create_latency_s": {"p50": _pctl(create_lat, 0.5),
                                     "p95": _pctl(create_lat, 0.95)},
                "generate_latency_s": {"p50": _pctl(gen_lat, 0.5),
                                       "p95": _pctl(gen_lat, 0.95)},
                "per_session_kv_bytes": (max(kvs) if kvs else None),
                "server_rss_mb": rss,
                "wall_s": round(wall, 2),
            }
            rows.append(row)
            print(f"[loadtest] agents={n:5d} created={created}/{n} "
                  f"gen_ok={gen_ok} errs={errs} "
                  f"create_p95={row['create_latency_s']['p95']}s "
                  f"gen_p95={row['generate_latency_s']['p95']}s "
                  f"kv/sess={row['per_session_kv_bytes']}B rss={rss}MB", flush=True)
            if created < n:
                print(f"[loadtest] admission/resource ceiling hit at n={n} "
                      f"(created {created}); stopping ramp.", flush=True)
                break
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:  # noqa: BLE001
            server.kill()

    full = [r for r in rows if r["created_ok"] == r["agents"] and not r["errors"]]
    max_conc = max((r["agents"] for r in full), default=0)
    per_sess_kv = next((r["per_session_kv_bytes"] for r in reversed(rows)
                        if r["per_session_kv_bytes"]), None)
    report = {
        "kind": "grpc_agent_capacity_loadtest",
        "schema_version": 1,
        "config": {
            "backend": args.backend,
            "verifier_id": args.verifier_id,
            "capacity": args.capacity,
            "sink": args.sink,
            "window": args.window,
            "gen_tokens": args.gen_tokens,
            "prompt_len": args.prompt_len,
            "context_len": args.context_len,
            "prefill_len": prefill_len,
            "fd_limit": fd_info,
            "levels": levels,
            "single_tenant_note": (
                "v0.3 single-tenant: shared verifier, RPCs serialized on one "
                "asyncio loop; this measures connection/session admission "
                "scaling, not parallel inference."),
        },
        "results": rows,
        "summary": {
            "max_concurrent_agents_clean": max_conc,
            "per_session_kv_bytes": per_sess_kv,
            "per_session_kv_mb": (round(per_sess_kv / 1e6, 4) if per_sess_kv else None),
            "node_kv_upper_bound_mb": (
                round(args.capacity * per_sess_kv / 1e6, 2) if per_sess_kv else None),
            "node_kv_upper_bound_note": (
                "capacity * per-session bounded KV — the whole-node resident-KV "
                "ceiling, independent of context length or agent churn."),
            "server_peak_rss_mb": peak_rss or None,
        },
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[loadtest] wrote {args.output}", flush=True)
    s = report["summary"]
    print(f"[loadtest] DONE max_concurrent_agents_clean={s['max_concurrent_agents_clean']} "
          f"per_session_kv={s['per_session_kv_mb']}MB "
          f"node_kv_upper_bound={s['node_kv_upper_bound_mb']}MB "
          f"peak_rss={s['server_peak_rss_mb']}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
