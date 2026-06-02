"""gRPC long-session bench (PR-E1b of ADR 0008 Phase E).

Walks ONE gRPC session through many short turns, recording per-turn
latency and ``session.info().kv_live_bytes``. Validates the two
ADR 0008 §7 GA gates the deprecated HTTP shim's ``bench_long_session.py``
cannot answer:

  * **memory bounded**:    ``agg.kv_bounded`` is True (KV stays within
                           a tight band across the whole run).
  * **prefill bounded**:   ``agg.prefill_bounded`` is True (per-turn
                           latency is flat across the run — no drift
                           with history length).

The HTTP shim's bench fails on prefill-bounded by architecture: every
``/v1/chat/completions`` request re-prefills the full conversation
history. The session-bound gRPC contract makes prefill cost depend
only on the size of each new user message, regardless of how long
the conversation is. This bench measures that empirically.

Usage::

    # Terminal 1 — start the runtime
    PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
        --backend cpu --verifier-id Qwen/Qwen3-0.6B \
        --bind 127.0.0.1:50051 \
        --capacity 1 --sink 4 --window 64

    # Terminal 2 — run the bench
    PYTHONPATH=.:sdks/python python3 \
        scripts/bench_agentic/bench_session_long_run.py \
        --grpc-address 127.0.0.1:50051 \
        --tokenizer-id Qwen/Qwen3-0.6B \
        --duration-s 14400 --turn-spacing-s 30 \
        --output results/platform-tests/bench_session_4h_$(date +%s).json

CLI plumbing only — pure aggregation lives in
:mod:`inference_engine.bench.session_long_run` and is unit-tested
under the Linux 100% coverage gate. This script itself is exempt by
the same convention as ``serve.py`` / ``run_demo.py`` / ``chat.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference_engine.bench.session_long_run import aggregate_run


# Workload — a fixed rotating set of short user messages so per-turn
# token counts stay small and the bench's prefill-bounded claim is a
# clean signal about session-bound prefill, not about variability in
# message sizes. Six messages chosen so the deepest history-length
# cycle is ~6 turns; long enough to exercise multiple sink+window
# trims.
_USER_MESSAGES: List[str] = [
    "What is a sliding window KV cache?",
    "Explain the role of the sink tokens.",
    "How does this differ from prefix caching?",
    "What are the typical sink and window sizes?",
    "Walk me through one inference step.",
    "Summarize what we discussed in two sentences.",
]


def _build_argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--grpc-address", default="127.0.0.1:50051",
        help="host:port of a running kakeya gRPC RuntimeService. "
             "Default points at the local server scripts/"
             "start_grpc_runtime_server.py defaults to.",
    )
    ap.add_argument(
        "--tokenizer-id", default="Qwen/Qwen3-0.6B",
        help="HF model id for the tokenizer. MUST match the verifier "
             "the gRPC server is running, otherwise token ids will "
             "be misinterpreted server-side.",
    )
    ap.add_argument(
        "--duration-s", type=float, default=1800.0,
        help="Total wall-clock duration of the run, in seconds. "
             "Default 1800 (30 min smoke). Use 14400 for the full 4h.",
    )
    ap.add_argument(
        "--turn-spacing-s", type=float, default=30.0,
        help="Wall-clock spacing between turn STARTS. If a turn "
             "takes longer than this, the next turn starts "
             "immediately — turn 0's start time is t=0, not "
             "t=spacing.",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=64,
        help="max_tokens for each Generate call.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path to write the JSON report. Atomic-replace via "
             "tmp + os.replace so a SIGTERM mid-write doesn't leave "
             "a half-written file.",
    )
    ap.add_argument(
        "--partial-checkpoint-every-s", type=float, default=600.0,
        help="Every N seconds, write a snapshot to "
             "<output>.partial.json so a long-running bench has "
             "evidence on disk even if the host reboots before "
             "completion.",
    )
    return ap


def _now() -> float:
    return time.monotonic()


def _wallclock() -> float:
    return time.time()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _build_payload(
    *,
    turns: List[Dict[str, Any]],
    args: argparse.Namespace,
    started_at: float,
    finished_at: Optional[float],
    duration_s: float,
    partial: bool,
    abort_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "bench_session_long_run",
        "partial": partial,
        "abort_reason": abort_reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "config": {
            "grpc_address": args.grpc_address,
            "tokenizer_id": args.tokenizer_id,
            "duration_s_target": args.duration_s,
            "turn_spacing_s": args.turn_spacing_s,
            "max_tokens": args.max_tokens,
        },
        "turns": turns,
        "agg": aggregate_run(turns, duration_s=duration_s),
    }


def _run_one_turn(
    *,
    session,
    tokenizer,
    user_message: str,
    max_tokens: int,
    t_relative_s: float,
) -> Dict[str, Any]:
    """One bench iteration. On error, returns ``ok=False`` with the
    error class + str instead of raising, so the run continues and
    the error surfaces in the aggregate report."""
    try:
        # Tokenize the NEW user message only — this is the whole
        # point of session-bound runtime. Compare to bench_long_session.py
        # where every turn sends the full conversation history.
        new_tokens = tokenizer.encode(user_message, add_special_tokens=False)
        t0 = _now()
        session.append(new_tokens)
        emitted: List[int] = []
        for token_id in session.generate(max_tokens=max_tokens):
            emitted.append(token_id)
        latency_s = _now() - t0
        info = session.info()
        return {
            "ok": True,
            "t_relative_s": t_relative_s,
            "latency_s": latency_s,
            "kv_live_bytes": info.kv_live_bytes,
            "history_length": info.history_length,
            "n_emitted": len(emitted),
            "user_message_tokens": len(new_tokens),
        }
    except Exception as exc:  # noqa: BLE001 - we want to log every error class
        return {
            "ok": False,
            "t_relative_s": t_relative_s,
            "error_class": type(exc).__name__,
            "error_str": str(exc),
        }


def main() -> int:
    ap = _build_argument_parser()
    args = ap.parse_args()

    # Lazy imports — these pull the HF stack, only do it when actually
    # running, so --help stays fast and the unit tests on aggregate_run
    # don't need to install HF on Linux.
    from transformers import AutoTokenizer
    from kakeya import Client

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[bench] loading tokenizer {args.tokenizer_id!r}",
        file=sys.stderr, flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos = tokenizer.eos_token_id
    eos_ids: List[int] = [int(eos)] if eos is not None else []

    print(
        f"[bench] connecting to {args.grpc_address}",
        file=sys.stderr, flush=True,
    )
    started_at = _wallclock()
    t_origin = _now()
    last_checkpoint_at = t_origin
    turns: List[Dict[str, Any]] = []
    abort_reason: Optional[str] = None

    stop_requested = False

    def _on_signal(signum, _frame):  # pragma: no cover - signal-driven
        nonlocal stop_requested, abort_reason
        stop_requested = True
        abort_reason = f"signal {signum}"

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    try:
        with Client(args.grpc_address) as client:
            with client.create_session(eos_token_ids=eos_ids) as session:
                turn_idx = 0
                while not stop_requested:
                    t_relative = _now() - t_origin
                    if t_relative >= args.duration_s:
                        break
                    msg = _USER_MESSAGES[turn_idx % len(_USER_MESSAGES)]
                    record = _run_one_turn(
                        session=session,
                        tokenizer=tokenizer,
                        user_message=msg,
                        max_tokens=args.max_tokens,
                        t_relative_s=t_relative,
                    )
                    turns.append(record)
                    turn_idx += 1

                    # Partial checkpoint — write snapshot every N seconds
                    # so a host reboot doesn't lose hours of evidence.
                    if (
                        args.partial_checkpoint_every_s > 0
                        and (_now() - last_checkpoint_at)
                        >= args.partial_checkpoint_every_s
                    ):
                        _atomic_write_json(
                            out_path.with_suffix(out_path.suffix + ".partial"),
                            _build_payload(
                                turns=turns, args=args,
                                started_at=started_at, finished_at=None,
                                duration_s=_now() - t_origin,
                                partial=True,
                            ),
                        )
                        last_checkpoint_at = _now()

                    # Pace turn STARTS at args.turn_spacing_s.
                    next_start = (
                        t_origin + (turn_idx * args.turn_spacing_s)
                    )
                    sleep_s = next_start - _now()
                    if sleep_s > 0:
                        # Sleep in small chunks so SIGTERM is responsive.
                        deadline = _now() + sleep_s
                        while not stop_requested and _now() < deadline:
                            time.sleep(min(0.5, deadline - _now()))
    except Exception as exc:  # noqa: BLE001 - the bench's job is to summarize, not crash
        abort_reason = f"{type(exc).__name__}: {exc}"
        print(
            f"[bench] aborting due to: {abort_reason}",
            file=sys.stderr, flush=True,
        )

    duration_s = _now() - t_origin
    finished_at = _wallclock()
    payload = _build_payload(
        turns=turns, args=args,
        started_at=started_at, finished_at=finished_at,
        duration_s=duration_s,
        partial=False,
        abort_reason=abort_reason,
    )
    _atomic_write_json(out_path, payload)
    print(
        f"[bench] wrote {out_path}: "
        f"n_turns={payload['agg']['n_turns']} "
        f"n_errors={payload['agg']['n_errors']} "
        f"duration_s={duration_s:.1f} "
        f"kv_bounded={payload['agg']['kv_bounded']} "
        f"prefill_bounded={payload['agg']['prefill_bounded']}",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
