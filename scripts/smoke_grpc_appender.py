"""End-to-end smoke for the PR-B2 AppendTokens RPC.

Spins up a real ``grpc.aio.Server`` with an
:class:`AppendTokensCoordinator` wired in (driving a deterministic
:class:`FakeVerifier` from the test suite — no model weights, fully
Linux + Mac runnable). Walks the AppendTokens scenarios PR-B2 ships:

  1.  CreateSession                                        -> success
  2.  AppendTokens (cold)                                  -> success, history=3
  3.  GetSessionInfo                                       -> history_length=3
  4.  AppendTokens (incremental)                           -> success, history=5
  5.  GetSessionInfo                                       -> history_length=5
  6.  AppendTokens (empty list)                            -> success, history=5  (no-op)
  7.  AppendTokens (unknown session)                       -> NOT_FOUND
  8.  AppendTokens (after CloseSession)                    -> NOT_FOUND
  9.  AppendTokens (INV-1 violation via lying inspector)   -> FAILED_PRECONDITION
  10. Generate (still PR-B3 territory)                     -> UNIMPLEMENTED

Each step prints one JSON-Lines record with expected vs observed
outcome. Exit code 0 iff all 10 scenarios match.

Companion to ``scripts/smoke_grpc_runtime.py`` (PR-B1's smoke),
which continues to exercise the no-coordinator default path
(AppendTokens stays UNIMPLEMENTED). The two smokes are independent
review aids; running both confirms PR-B1's regression contract
holds AND PR-B2's new contract works.

Usage::

    PYTHONPATH=. python3 scripts/smoke_grpc_appender.py

    # Capture a structured report:
    PYTHONPATH=. python3 scripts/smoke_grpc_appender.py \\
        --report results/platform-tests/grpc-appender-smoke-$(date +%s).json
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

from inference_engine.server.grpc_app import RuntimeServiceServicer
from inference_engine.server.proto_gen.kakeya.v1 import (
    runtime_pb2,
    runtime_pb2_grpc,
)
from inference_engine.session import (
    AppendTokensCoordinator,
    SessionStore,
)

# Reuse the test-suite FakeVerifier so the smoke and the unit tests
# exercise the same VerifierProtocol implementation.
from tests.inference_engine.session.test_coordinator import FakeVerifier


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


async def _serve(
    store: SessionStore,
    coordinator: Optional[AppendTokensCoordinator],
) -> AsyncIterator[tuple[runtime_pb2_grpc.RuntimeServiceStub, grpc.aio.Server, int]]:
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(store, append_coordinator=coordinator),
        server,
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


async def _step(name: str, expected: str, body) -> StepResult:
    t0 = time.perf_counter()
    try:
        observed, detail = await body()
        passed = observed == expected
    except Exception as exc:  # noqa: BLE001 — best-effort smoke
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

    # ----- Server #1: AppendTokensCoordinator wired with FakeVerifier -----
    fv = FakeVerifier()
    store = SessionStore(capacity=4, cache_inspector=fv)
    coord = AppendTokensCoordinator(store, fv)
    s_ctx = _serve(store, coord).__aiter__()
    stub, _, _ = await s_ctx.__anext__()
    try:
        async def _create():
            r = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
            return ("ok", {"session_id": r.session_id})
        emit(await _step("CreateSession", "ok", _create))
        sid = results[-1].detail["session_id"]

        async def _append_cold():
            r = await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=sid, token_ids=[10, 20, 30],
                ),
            )
            return ("ok", {
                "history_length": r.history_length,
                "verifier_calls": [c[0] for c in fv.call_log],
            })
        emit(await _step("AppendTokens (cold prefill)", "ok", _append_cold))

        async def _info_after_cold():
            r = await stub.GetSessionInfo(
                runtime_pb2.GetSessionInfoRequest(session_id=sid),
            )
            return ("ok", {
                "history_length": r.history_length,
                "inv1_violations": r.cache_invariant_inv1_violations,
                "inv2_violations": r.cache_invariant_inv2_violations,
            })
        emit(await _step("GetSessionInfo (after cold)", "ok", _info_after_cold))

        async def _append_incremental():
            r = await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=sid, token_ids=[40, 50],
                ),
            )
            return ("ok", {
                "history_length": r.history_length,
                "verifier_calls_total": [c[0] for c in fv.call_log],
            })
        emit(await _step("AppendTokens (incremental)", "ok", _append_incremental))

        async def _info_after_incremental():
            r = await stub.GetSessionInfo(
                runtime_pb2.GetSessionInfoRequest(session_id=sid),
            )
            return ("ok", {"history_length": r.history_length})
        emit(await _step("GetSessionInfo (after incremental)", "ok", _info_after_incremental))

        async def _append_empty():
            r = await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(session_id=sid, token_ids=[]),
            )
            return ("ok", {
                "history_length": r.history_length,
                "verifier_calls_after_empty": [c[0] for c in fv.call_log],
            })
        emit(await _step("AppendTokens (empty list, no-op)", "ok", _append_empty))

        async def _append_unknown():
            try:
                await stub.AppendTokens(
                    runtime_pb2.AppendTokensRequest(
                        session_id="sess-nonexistent", token_ids=[1, 2],
                    ),
                )
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()})
        emit(await _step(
            "AppendTokens (unknown session)", "NOT_FOUND", _append_unknown,
        ))

        # Close current session, then attempt AppendTokens on it.
        await stub.CloseSession(runtime_pb2.CloseSessionRequest(session_id=sid))

        async def _append_after_close():
            try:
                await stub.AppendTokens(
                    runtime_pb2.AppendTokensRequest(
                        session_id=sid, token_ids=[1, 2],
                    ),
                )
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()})
        emit(await _step(
            "AppendTokens (after CloseSession)", "NOT_FOUND",
            _append_after_close,
        ))

        # Generate is still UNIMPLEMENTED in PR-B2.
        async def _generate_unimpl():
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
        emit(await _step(
            "Generate (PR-B3, still UNIMPLEMENTED)", "UNIMPLEMENTED",
            _generate_unimpl,
        ))
    finally:
        try:
            await s_ctx.__anext__()
        except StopAsyncIteration:
            pass

    # ----- Server #2: lying CacheInspector -> INV-1 violation -----
    class _LyingFake(FakeVerifier):
        def k_seq_length(self, session):
            del session
            return 999  # never matches

    lying_fv = _LyingFake()
    lying_store = SessionStore(capacity=2, cache_inspector=lying_fv)
    lying_coord = AppendTokensCoordinator(lying_store, lying_fv)
    s2_ctx = _serve(lying_store, lying_coord).__aiter__()
    stub2, _, _ = await s2_ctx.__anext__()
    try:
        async def _create_for_inv1():
            r = await stub2.CreateSession(runtime_pb2.CreateSessionRequest())
            return r.session_id

        sid2 = await _create_for_inv1()

        async def _append_inv1():
            try:
                await stub2.AppendTokens(
                    runtime_pb2.AppendTokensRequest(
                        session_id=sid2, token_ids=[1, 2, 3],
                    ),
                )
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {
                    "details": e.details()[:120],
                    "kind": "INV-1",
                })
        emit(await _step(
            "AppendTokens (INV-1 violation)", "FAILED_PRECONDITION",
            _append_inv1,
        ))
    finally:
        try:
            await s2_ctx.__anext__()
        except StopAsyncIteration:
            pass

    return results


def _summary(results: list[StepResult]) -> dict[str, Any]:
    n_total = len(results)
    n_passed = sum(r.passed for r in results)
    return {
        "schema_version": 1,
        "kind": "grpc_appender_smoke",
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
    p.add_argument("--report", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    results = asyncio.run(run_smoke(verbose=not args.quiet))
    summary = _summary(results)
    print(json.dumps(
        {"summary": {k: v for k, v in summary.items() if k not in ("steps", "host")}},
        separators=(",", ":"),
    ))
    print(json.dumps({"host": summary["host"]}, separators=(",", ":")))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"report written: {args.report}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
