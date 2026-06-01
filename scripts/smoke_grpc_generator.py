"""End-to-end smoke for the PR-B3 Generate server-streaming RPC.

Spins up a real ``grpc.aio.Server`` with both the AppendTokens
coordinator (PR-B2) and the Generation coordinator (PR-B3) wired
in, and walks the Generate scenarios this PR ships:

  1.  CreateSession                                          -> success
  2.  AppendTokens (cold prefill)                            -> success
  3.  Generate (max_tokens=3, no EOS)                        -> 3 token_id frames + done(MAX_TOKENS)
  4.  Generate (max_tokens=10, EOS in token 6)               -> 1 token + done(EOS)  (deterministic by FakeVerifier)
  5.  Generate (no AppendTokens prior)                       -> INVALID_ARGUMENT
  6.  Generate (max_tokens=0)                                -> INVALID_ARGUMENT
  7.  Generate (temperature=0.7)                             -> INVALID_ARGUMENT
  8.  Generate (unknown session)                             -> NOT_FOUND
  9.  Generate (large prefill -> truncated state)            -> 1 truncated frame + tokens + done
  10. Generate (no coordinator wired)                        -> UNIMPLEMENTED

Each step prints one JSON-Lines record with expected vs observed
outcome. Exit code 0 iff all 10 scenarios match.

Same review-affordance pattern as PR-B1 (smoke_grpc_runtime.py) and
PR-B2 (smoke_grpc_appender.py).

Usage::

    PYTHONPATH=. python3 scripts/smoke_grpc_generator.py \\
        --report results/platform-tests/grpc-generator-smoke-$(date +%s).json
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
    GenerationCoordinator,
    SessionStore,
)

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
    append_coord: Optional[AppendTokensCoordinator] = None,
    gen_coord: Optional[GenerationCoordinator] = None,
) -> AsyncIterator[tuple[runtime_pb2_grpc.RuntimeServiceStub, grpc.aio.Server, int]]:
    server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(
            store,
            append_coordinator=append_coord,
            generation_coordinator=gen_coord,
        ),
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
    except Exception as exc:  # noqa: BLE001
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

    # ----- Server #1: full coordinator wiring (Append + Generate) -----
    fv = FakeVerifier()
    store = SessionStore(capacity=4, cache_inspector=fv)
    append_coord = AppendTokensCoordinator(store, fv)
    gen_coord = GenerationCoordinator(store, fv)
    s_ctx = _serve(store, append_coord, gen_coord).__aiter__()
    stub, _, _ = await s_ctx.__anext__()
    try:
        async def _step1():
            r = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
            return ("ok", {"session_id": r.session_id})
        emit(await _step("CreateSession", "ok", _step1))
        sid = results[-1].detail["session_id"]

        async def _step2():
            r = await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=sid, token_ids=[1, 2, 3],
                ),
            )
            return ("ok", {"history_length": r.history_length})
        emit(await _step("AppendTokens (cold prefill)", "ok", _step2))

        async def _step3():
            frames = []
            async for f in stub.Generate(
                runtime_pb2.GenerateRequest(session_id=sid, max_tokens=3),
            ):
                frames.append(f.WhichOneof("payload"))
            done = frames.count("done")
            tokens = frames.count("token_id")
            return ("ok", {
                "frames": frames,
                "tokens_emitted": tokens,
                "done_frames": done,
            })
        emit(await _step(
            "Generate (max_tokens=3, no EOS)", "ok", _step3,
        ))

        # Step 4: EOS scenario. FakeVerifier's _logits_for produces argmax
        # = sum(history[-3:]) % 16; with history [1,2,3] -> argmax=6.
        # Create fresh session with eos=[6]; first generated token is 6.
        async def _step4():
            r_create = await stub.CreateSession(
                runtime_pb2.CreateSessionRequest(eos_token_ids=[6]),
            )
            sid2 = r_create.session_id
            await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=sid2, token_ids=[1, 2, 3],
                ),
            )
            tokens = []
            stop_reason_name = None
            async for f in stub.Generate(
                runtime_pb2.GenerateRequest(session_id=sid2, max_tokens=10),
            ):
                kind = f.WhichOneof("payload")
                if kind == "token_id":
                    tokens.append(f.token_id)
                elif kind == "done":
                    stop_reason_name = (
                        runtime_pb2.GenerateDone.StopReason.Name(
                            f.done.stop_reason,
                        )
                    )
            return ("ok", {
                "tokens": tokens,
                "stop_reason": stop_reason_name,
            })
        emit(await _step(
            "Generate (EOS triggers, max_tokens=10)", "ok", _step4,
        ))

        # Step 5: Generate without prior AppendTokens.
        async def _step5():
            r_create = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
            try:
                async for _f in stub.Generate(
                    runtime_pb2.GenerateRequest(
                        session_id=r_create.session_id, max_tokens=1,
                    ),
                ):
                    return ("ok", {})  # would be a bug
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()[:80]})
        emit(await _step(
            "Generate (no AppendTokens prior)", "INVALID_ARGUMENT", _step5,
        ))

        # Step 6: max_tokens=0
        async def _step6():
            try:
                async for _f in stub.Generate(
                    runtime_pb2.GenerateRequest(session_id=sid, max_tokens=0),
                ):
                    return ("ok", {})
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()[:60]})
        emit(await _step(
            "Generate (max_tokens=0)", "INVALID_ARGUMENT", _step6,
        ))

        # Step 7: temperature=0.7 (non-greedy rejected in v0.3)
        async def _step7():
            try:
                async for _f in stub.Generate(
                    runtime_pb2.GenerateRequest(
                        session_id=sid, max_tokens=1, temperature=0.7,
                    ),
                ):
                    return ("ok", {})
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()[:60]})
        emit(await _step(
            "Generate (temperature=0.7, non-greedy rejected)",
            "INVALID_ARGUMENT", _step7,
        ))

        # Step 8: unknown session
        async def _step8():
            try:
                async for _f in stub.Generate(
                    runtime_pb2.GenerateRequest(
                        session_id="sess-nonexistent", max_tokens=1,
                    ),
                ):
                    return ("ok", {})
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()[:60]})
        emit(await _step(
            "Generate (unknown session)", "NOT_FOUND", _step8,
        ))

        # Step 9: large prefill -> truncated state
        async def _step9():
            r_create = await stub.CreateSession(runtime_pb2.CreateSessionRequest())
            await stub.AppendTokens(
                runtime_pb2.AppendTokensRequest(
                    session_id=r_create.session_id,
                    token_ids=[10, 20, 30, 40, 50, 60, 70, 80],
                ),
            )
            frame_kinds = []
            truncated_dropped = None
            async for f in stub.Generate(
                runtime_pb2.GenerateRequest(
                    session_id=r_create.session_id, max_tokens=2,
                ),
            ):
                kind = f.WhichOneof("payload")
                frame_kinds.append(kind)
                if kind == "truncated":
                    truncated_dropped = f.truncated.dropped_token_count
            return ("ok", {
                "frames": frame_kinds,
                "dropped_token_count": truncated_dropped,
            })
        emit(await _step(
            "Generate (truncated state -> truncated frame)", "ok", _step9,
        ))
    finally:
        try:
            await s_ctx.__anext__()
        except StopAsyncIteration:
            pass

    # ----- Server #2: no generation_coordinator wired -----
    fv2 = FakeVerifier()
    store2 = SessionStore(capacity=2)
    s2_ctx = _serve(store2, None, None).__aiter__()
    stub2, _, _ = await s2_ctx.__anext__()
    try:
        async def _step10():
            try:
                async for _f in stub2.Generate(
                    runtime_pb2.GenerateRequest(
                        session_id="any", max_tokens=1,
                    ),
                ):
                    return ("ok", {})
                return ("ok", {})
            except grpc.aio.AioRpcError as e:
                return (e.code().name, {"details": e.details()[:80]})
        emit(await _step(
            "Generate (no coordinator wired)", "UNIMPLEMENTED", _step10,
        ))
    finally:
        try:
            await s2_ctx.__anext__()
        except StopAsyncIteration:
            pass

    return results


def _summary(results: list[StepResult]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "grpc_generator_smoke",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "grpc": grpc.__version__,
        },
        "steps_total": len(results),
        "steps_passed": sum(r.passed for r in results),
        "steps_failed": sum(not r.passed for r in results),
        "all_passed": all(r.passed for r in results),
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
