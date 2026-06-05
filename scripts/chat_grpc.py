"""Streaming chat REPL over the Kakeya gRPC runtime (v0.3).

A multi-turn chat client that uses the Python SDK to talk to a
running ``RuntimeService``. Demonstrates the session-bound
architecture's killer feature: the server keeps the running KV
cache, so every turn after the first appends only the new user
message — independent of conversation length.

Compare to ``scripts/chat.py`` (v0.2): that REPL re-prefilled the
full conversation on every turn against an in-process
``SpeculativeEngine``. This REPL holds one ``Session`` open across
turns and the server keeps O(history) cache; per-turn prefill is
O(new_user_message).

Usage::

    # 1. In one terminal, start the runtime
    PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \\
        --backend cpu --verifier-id Qwen/Qwen3-0.6B \\
        --bind 127.0.0.1:50051

    # 2. In another terminal, chat
    PYTHONPATH=.:sdks/python python3 scripts/chat_grpc.py
    # Or, with options:
    PYTHONPATH=.:sdks/python python3 scripts/chat_grpc.py \\
        --address 127.0.0.1:50051 \\
        --tokenizer-id Qwen/Qwen3-0.6B \\
        --max-tokens 64

REPL controls
-------------

  Type your message + Enter to send.
  Ctrl-D or empty line to exit.
  ``/reset`` on its own line: close current session, open new one
                              (clear context).
  ``/info`` on its own line: print server-side session state
                              (history length, KV bytes, idle time).
  ``/help``: this list.

Per the project's CLI-plumbing convention this script is exempt
from the unit-test coverage gate. End-to-end behavior is exercised
by the SDK integration tests at
``tests/integration/test_sdk_real.py`` which drive the same SDK
methods this REPL drives.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


_HELP = """
Commands:
  /help              show this help
  /reset             close current session, start a fresh one
  /info              show server-side session state
  /exit              quit (or Ctrl-D / empty line)
""".strip()


def _print_banner(address: str, tokenizer_id: str) -> None:
    print(
        f"Kakeya v0.3 chat — {address}  ({tokenizer_id})\n"
        f"Session-bound runtime: server keeps history, you only send "
        f"new tokens per turn.\n"
        f"Type /help for commands; Ctrl-D or empty line to quit.\n",
        file=sys.stderr, flush=True,
    )


def _read_user_input(prompt: str = "you> ") -> Optional[str]:
    """Read a single user line from stdin.

    Returns ``None`` on EOF (Ctrl-D) or empty input. Empty input is
    a terminate signal — the user can use ``/reset`` to clear context
    without exiting the REPL.
    """
    try:
        line = input(prompt)
    except EOFError:
        return None
    if not line.strip():
        return None
    return line


def _generate_and_print(
    session,
    tokenizer,
    new_tokens: List[int],
    max_tokens: int,
) -> int:
    """Drive one append + generate cycle. Streams tokens to stdout
    as they arrive, returns the count emitted. The generator's
    metadata (stop reason, durations) is read after iteration via
    ``session.last_*`` properties.
    """
    session.append(new_tokens)

    print("kakeya> ", end="", flush=True)
    n = 0
    accumulated = []
    try:
        for token_id in session.generate(max_tokens=max_tokens):
            n += 1
            accumulated.append(token_id)
            # Decode incrementally — tokenizer.decode on the running
            # buffer gives the right text including BPE merges that
            # span multiple tokens. We re-decode the full buffer
            # each time (Qwen3-family tokenizers re-decode in <1ms
            # for a 64-token buffer; per-token decoding loses some
            # whitespace correctness on the tokenizer level).
            text_so_far = tokenizer.decode(
                accumulated, skip_special_tokens=True,
            )
            # Print only the suffix that's new since last frame.
            if hasattr(_generate_and_print, "_last_text"):
                last = _generate_and_print._last_text
            else:
                last = ""
            new_text = text_so_far[len(last):]
            print(new_text, end="", flush=True)
            _generate_and_print._last_text = text_so_far
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
    finally:
        # Reset the per-call decoder state so the next turn starts
        # fresh.
        if hasattr(_generate_and_print, "_last_text"):
            del _generate_and_print._last_text

    print()  # final newline
    return n


def _print_session_info(session) -> None:
    info = session.info()
    print(
        f"  history_length = {info.history_length}\n"
        f"  kv_live_bytes  = {info.kv_live_bytes:,}\n"
        f"  idle_seconds   = {info.idle_seconds:.3f}\n"
        f"  inv1_violations= {info.cache_invariant_inv1_violations}\n"
        f"  inv2_violations= {info.cache_invariant_inv2_violations}",
        file=sys.stderr, flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--address", default="127.0.0.1:50051",
        help="host:port of a running kakeya gRPC RuntimeService",
    )
    ap.add_argument(
        "--tokenizer-id", default="Qwen/Qwen3-0.6B",
        help="HF model id for the tokenizer. MUST match the verifier "
             "the server is running.",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=64,
        help="max_tokens per turn",
    )
    ap.add_argument(
        "--system-prompt", default="You are a helpful assistant.",
        help="System prompt prepended on the first turn (Qwen3 chat "
             "template). Pass empty string to skip.",
    )
    args = ap.parse_args()

    # Lazy imports keep --help fast.
    from kakeya import Client
    from kakeya.errors import KakeyaError
    from transformers import AutoTokenizer

    print(f"[chat] loading tokenizer {args.tokenizer_id} ...",
          file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos = tokenizer.eos_token_id
    eos_ids: List[int] = [int(eos)] if eos is not None else []

    _print_banner(args.address, args.tokenizer_id)

    def _make_session(client):
        s = client.create_session(eos_token_ids=eos_ids)
        # Seed with the system prompt on turn 0 (no generation yet).
        if args.system_prompt:
            seed_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": args.system_prompt}],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
                enable_thinking=False,
            )
            if seed_ids:
                s.append(seed_ids)
        return s

    with Client(args.address) as client:
        session = _make_session(client)
        try:
            while True:
                user_line = _read_user_input()
                if user_line is None:
                    print("[bye]", file=sys.stderr)
                    break

                # Slash commands
                if user_line.startswith("/"):
                    cmd = user_line.strip().lower()
                    if cmd in ("/exit", "/quit"):
                        print("[bye]", file=sys.stderr)
                        break
                    if cmd == "/help":
                        print(_HELP, file=sys.stderr)
                        continue
                    if cmd == "/reset":
                        try:
                            session.close()
                        except KakeyaError:
                            pass
                        session = _make_session(client)
                        print("[session reset]", file=sys.stderr)
                        continue
                    if cmd == "/info":
                        try:
                            _print_session_info(session)
                        except KakeyaError as exc:
                            print(f"[info error: {exc}]", file=sys.stderr)
                        continue
                    print(f"[unknown command: {cmd}; try /help]",
                          file=sys.stderr)
                    continue

                # Tokenize the user message via the chat template — this
                # gives Qwen3 the role marker tokens, not raw text.
                new_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_line}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )

                try:
                    _generate_and_print(
                        session=session,
                        tokenizer=tokenizer,
                        new_tokens=new_tokens,
                        max_tokens=args.max_tokens,
                    )
                except KakeyaError as exc:
                    print(f"[runtime error: {exc}]", file=sys.stderr)
                    # Try to recover by resetting the session — the
                    # server may have evicted it.
                    try:
                        session.close()
                    except KakeyaError:
                        pass
                    session = _make_session(client)
                    print("[session re-created after error]",
                          file=sys.stderr)
        finally:
            try:
                session.close()
            except KakeyaError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
