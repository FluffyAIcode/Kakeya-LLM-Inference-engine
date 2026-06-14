"""PR-A3c end-to-end: true multi-tenant serving through the gRPC RuntimeService.

Launches the gRPC runtime (``--backend restored --multi-tenant``: per-session
verifier binding) and drives **N concurrent SDK clients**, each its own session
with a distinct NIAH needle. Verifies **per-session recall** (the bottom line)
and **isolation** — concurrent/interleaved sessions must each recall their OWN
needle, which only holds if each session has its own KV cache (PR-A3c).

Recall-preserving restored S5 path only (CUDA). This is the served-path
counterpart to the batched engine test (k3_cuda_multitenant_parallel_bench.py):
here the parallelism is N real gRPC clients hitting one server.
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
from typing import Any, Dict, List


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _wait_ready(addr: str, timeout_s: float) -> bool:
    from kakeya import Client
    from kakeya.errors import KakeyaError
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            c = Client(addr); s = c.create_session(); s.close(); c.close()
            return True
        except (KakeyaError, Exception):  # noqa: BLE001
            time.sleep(3.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-id", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--f-theta-dir", default="results/research/f_theta_v5_s5_sliding")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--haystack-lines", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--multi-tenant", action="store_true", default=True)
    ap.add_argument("--single-tenant", dest="multi_tenant", action="store_false",
                    help="control: shared verifier (expect cross-session corruption)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from inference_engine.v04 import make_niah_dataset

    tok = AutoTokenizer.from_pretrained(args.verifier_id)
    N = args.sessions
    samples = make_niah_dataset(n_samples=N, haystack_min_lines=args.haystack_lines,
                                haystack_max_lines=args.haystack_lines, seed=0)

    def encode(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids[0].tolist()

    prompts = [encode(s.prompt_text) for s in samples]
    answers = [s.answer_text for s in samples]
    eos = tok.eos_token_id

    port = _free_port()
    addr = f"127.0.0.1:{port}"
    cmd = [sys.executable, "scripts/start_grpc_runtime_server.py",
           "--backend", "restored", "--verifier-id", args.verifier_id,
           "--drafter-id", args.drafter_id, "--f-theta-dir", args.f_theta_dir,
           "--device", "cuda", "--bind", addr, "--capacity", str(max(N, 4)),
           "--sink", str(args.sink), "--window", str(args.window),
           "--skip-cache-check", "--log-level", "WARNING"]
    if args.multi_tenant:
        cmd.append("--multi-tenant")
    print(f"[e2e] launching server: {' '.join(cmd)}", flush=True)
    server = subprocess.Popen(cmd)
    results: List[Dict[str, Any]] = [None] * N  # type: ignore
    try:
        if not _wait_ready(addr, 900):
            print("[e2e] ERROR server not ready", flush=True)
            return 2
        print(f"[e2e] server ready on {addr} (multi_tenant={args.multi_tenant})",
              flush=True)

        from kakeya import Client
        barrier = threading.Barrier(N, timeout=600)

        def worker(i: int) -> None:
            c = Client(addr)
            sess = c.create_session(eos_token_ids=[eos] if eos is not None else [])
            sess.append(prompts[i])
            barrier.wait()                      # all sessions primed → interleave decode
            toks = list(sess.generate(max_tokens=args.max_new_tokens))
            text = tok.decode(toks, skip_special_tokens=True)
            results[i] = {
                "session": i, "answer": answers[i],
                "recall": answers[i] in text,
                "decoded": text[:60], "gen_tokens": len(toks),
                "prompt_tokens": len(prompts[i]),
            }
            sess.close(); c.close()

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:  # noqa: BLE001
            server.kill()

    recall = sum(1 for r in results if r and r["recall"]) / N
    report = {
        "kind": "k3_grpc_multitenant_e2e",
        "config": {"sessions": N, "multi_tenant": args.multi_tenant,
                   "verifier_id": args.verifier_id, "max_new_tokens": args.max_new_tokens,
                   "sink": args.sink, "window": args.window},
        "per_session_recall": round(recall, 3),
        "wall_s": round(wall, 2),
        "results": results,
    }
    for r in results:
        if r:
            print(f"[e2e] session {r['session']}: recall={r['recall']} "
                  f"answer={r['answer']} -> '{r['decoded']}'", flush=True)
    print(f"[e2e] per-session recall = {recall} ({N} concurrent sessions, "
          f"multi_tenant={args.multi_tenant})", flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[e2e] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
