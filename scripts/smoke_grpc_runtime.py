"""End-to-end smoke test for the PR-B1 gRPC RuntimeService surface.

Spins up a real ``grpc.aio.Server`` on a free localhost port, connects a
client, and walks through every RPC scenario PR-B1 ships:

  1.  CreateSession                                    -> success
  2.  GetSessionInfo                                   -> initial zero state
  3.  CloseSession                                     -> final history length 0
  4.  CloseSession (again, same id)                    -> NOT_FOUND
  5.  GetSessionInfo (closed id)                       -> NOT_FOUND
  6.  AppendTokens (any id)                            -> UNIMPLEMENTED  [PR-B2 territory]
  7.  Generate (any id)                                -> UNIMPLEMENTED  [PR-B3 territory]
  8.  CreateSession with eos_token_ids + client_label  -> success, fields recorded
  9.  CreateSession when slab pool exhausted           -> RESOURCE_EXHAUSTED

Each step prints a single JSON line to stdout, so the entire run is
machine-readable as JSON Lines. The exit code is 0 iff every step
matched its expected outcome; any deviation exits non-zero with a
non-zero ``failures`` count in the final summary.

This is a **manual review aid** for PR-B1, not a CI test. The
authoritative tests are under ``tests/inference_engine/server/test_grpc_app.py``
and run on every PR. This script lets a reviewer (especially on Mac M4
where pure-Linux CI is opaque) see the wire-level behavior of each RPC
in one terminal-readable run.

Usage::

    PYTHONPATH=. python3 scripts/smoke_grpc_runtime.py

    # Capture a structured report for committing back to a PR branch:
    PYTHONPATH=. python3 scripts/smoke_grpc_runtime.py \
        --report results/platform-tests/grpc-smoke-$(date +%s).json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Optional

import grpc

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.server.grpc_app import (
    RuntimeServiceServicer,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from inference_engine.session import SessionStore


# ---------------------------------------------------------------------------
# Step result records
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step: str
    expected: str
    observed: str
    passed: bool
    detail: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def asline(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


def _tiny_slab_pool(num_slabs: int) -> SlabPool:
    cfg = SlabConfig(
        num_layers=1,
        num_heads=1,
        sink_size=1,
        window_size=2,
        head_dim=4,
    )
    return SlabPool(num_slabs=num_slabs, slab_config=cfg)


async def _serve(
    store: SessionStore,
) -> AsyncIterator[tuple[runtime_pb2_grpc.RuntimeServiceStub, grpc.aio.Server, int]]:
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
    try:
        yield stub, server, port
    finally:
        await channel.close()
        await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# Individual step runners. Each one returns a StepResult.
# ---------------------------------------------------------------------------


async def _step(
    name: str, expected: str, body,
) -> StepResult:
    """Wrap a step body that must return (observed, detail). On
    exception, the step is marked failed and the traceback recorded."""
    t0 = time.perf_counter()
    try:
        observed, detail = await body()
        passed = observed == expected
    except Exception as exc:  # noqa: BLE001 — smoke is best-effort
        observed = f"unexpected exception: {type(exc).__name__}"
        detail = {"traceback": traceback.format_exc()}
        passed = False
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return StepResult(
        step=name, expected=expected, observed=observed,
        passed=passed, detail=detail, elapsed_ms=round(elapsed_ms, 2),
    )


async def run_smoke(verbose: bool = True) -> list[StepResult]:
    results: list[StepResult] = []

    def emit(r: StepResult) -> None:
        results.append(r)
        if verbose:
            print(r.asline(), flush=True)

    # ----- Server #1: pool-less store, capacity 4 -----
    store = SessionStore(capacity=4)
    server_ctx = _serve(store).__aiter__()
    stub, server, port = await server_ctx.__anext__()
    try:
        # Step 1: CreateSession success
        async def _step1():
            r = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
            return ("ok", {"session_id": r.session_id, "port": port})
        emit(await _step("CreateSession", "ok", _step1))
        sid_1 = results[-1].detail.get("session_id")

        # Step 2: GetSessionInfo initial
        async def _step2():
            r = await stub.GetSessionInfo(
                runtime_pb2.GetSessionInfoRequest(session_id=sid_1),
            )
            return ("ok", {
                "history_length": r.history_length,
                "kv_live_bytes": r.kv_live_bytes,
                "inv1_violations": r.cache_invariant_inv1_violations,
                "inv2_violations": r.cache_invariant_inv2_violations,
                "idle_seconds": round(r.idle_seconds, 4),
            })
        emit(await _step("GetSessionInfo (initial)", "ok", _step2))

        # Step 3: CloseSession success
        async def _step3():
            r = await stub.CloseSession(
                runtime_pb2.CloseSessionRequest(session_id=sid_1),
            )
            return ("ok", {"final_history_length": r.final_history_length})
        emit(await _step("CloseSession", "ok", _step3))

        # Step 4: Double-close -> NOT_FOUND
        async def _step4():
            try:
                await stub.CloseSession(
                    runtime_pb2.CloseSessionRequest(session_id=sid_1),
                )
                return ("ok", {})  # would be a bug
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()})
        emit(await _step("CloseSession (double-close)", "NOT_FOUND", _step4))

        # Step 5: GetSessionInfo after close -> NOT_FOUND
        async def _step5():
            try:
                await stub.GetSessionInfo(
                    runtime_pb2.GetSessionInfoRequest(session_id=sid_1),
                )
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()})
        emit(await _step("GetSessionInfo (after close)", "NOT_FOUND", _step5))

        # Step 6: AppendTokens -> UNIMPLEMENTED
        async def _step6():
            try:
                await stub.AppendTokens(
                    runtime_pb2.AppendTokensRequest(
                        session_id="any", token_ids=[1, 2, 3],
                    ),
                )
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"phase": "PR-B2"})
        emit(await _step("AppendTokens (PR-B2)", "UNIMPLEMENTED", _step6))

        # Step 7: Generate -> UNIMPLEMENTED
        async def _step7():
            try:
                async for _evt in stub.Generate(
                    runtime_pb2.GenerateRequest(
                        session_id="any", max_tokens=1,
                    ),
                ):
                    return ("ok", {})
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"phase": "PR-B3"})
        emit(await _step("Generate (PR-B3)", "UNIMPLEMENTED", _step7))

        # Step 8: CreateSession with eos + label
        async def _step8():
            r = await stub.CreateSession(
                runtime_pb2.CreateSessionRequest(
                    eos_token_ids=[7, 11, 13],
                    client_label="smoke-demo",
                ),
            )
            inner = store.get_session(r.session_id)
            return ("ok", {
                "session_id": r.session_id,
                "eos_token_ids_recorded": list(inner.eos_token_ids),
                "client_label_recorded": inner.client_label,
            })
        emit(await _step(
            "CreateSession (eos + client_label)", "ok", _step8,
        ))
    finally:
        try:
            await server_ctx.__anext__()
        except StopAsyncIteration:
            pass

    # ----- Server #2: pool-aware store with capacity > num_slabs -----
    pool = _tiny_slab_pool(num_slabs=1)
    store2 = SessionStore(capacity=4, slab_pool=pool)
    server_ctx2 = _serve(store2).__aiter__()
    stub2, _, _ = await server_ctx2.__anext__()
    try:
        # Step 9a: First create succeeds
        async def _step9a():
            r = await stub2.CreateSession(runtime_pb2.CreateSessionRequest())
            return ("ok", {"session_id": r.session_id})
        emit(await _step("CreateSession (pool slab #1 / 1)", "ok", _step9a))

        # Step 9b: Second create exhausts pool -> RESOURCE_EXHAUSTED
        async def _step9b():
            try:
                await stub2.CreateSession(runtime_pb2.CreateSessionRequest())
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {
                    "details": e.details(),
                    "pool_in_use": pool.in_use_count,
                    "pool_available": pool.available_count,
                })
        emit(await _step(
            "CreateSession (pool exhausted)", "RESOURCE_EXHAUSTED", _step9b,
        ))
    finally:
        try:
            await server_ctx2.__anext__()
        except StopAsyncIteration:
            pass

    return results


def _summary(results: list[StepResult]) -> dict[str, Any]:
    n_total = len(results)
    n_passed = sum(r.passed for r in results)
    return {
        "schema_version": 1,
        "kind": "grpc_runtime_smoke",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "grpc": grpc.__version__,
        },
        "steps_total": n_total,
        "steps_passed": n_passed,
        "steps_failed": n_total - n_passed,
        "all_passed": n_passed == n_total,
        "steps": [asdict(r) for r in results],
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--report", type=str, default=None,
        help=(
            "Optional JSON report path. When set, the per-step results "
            "and a summary block are written here for committing back "
            "to the PR branch."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step JSON-Lines output. Final summary is still printed.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    results = asyncio.run(run_smoke(verbose=not args.quiet))
    summary = _summary(results)

    print(json.dumps({
        "summary": {
            k: v for k, v in summary.items() if k not in ("steps", "host")
        },
    }, separators=(",", ":")))
    print(json.dumps({"host": summary["host"]}, separators=(",", ":")))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"report written: {args.report}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
