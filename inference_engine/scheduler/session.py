"""Per-session state for the scheduler.

A :class:`Session` carries everything the scheduler needs about one
request from admission through completion: input prompt, output
budget, EOS set, the slab assigned to it, the async queue tokens
flow through, and lifecycle bookkeeping.

State machine:

    PENDING  → ADMITTED  → COMPLETED
                       \\→ CANCELLED
                       \\→ FAILED

PENDING:    queued, no slab yet (only reachable under QUEUE policy).
ADMITTED:   slab acquired, generation in progress.
COMPLETED:  generation finished cleanly (EOS or max_new_tokens).
CANCELLED:  client cancelled (disconnect, explicit cancel, or queue timeout).
FAILED:     engine raised; error stored on the session.

State transitions are one-way (PENDING → ADMITTED → terminal). The
scheduler enforces this; sessions in a terminal state cannot be
re-driven.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


class SessionState(str, enum.Enum):
    PENDING = "pending"
    ADMITTED = "admitted"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Session:
    """One scheduled inference request."""

    prompt_ids: List[int]
    max_new_tokens: int
    eos_token_ids: List[int]

    id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex}")
    submitted_at: float = field(default_factory=time.monotonic)
    state: SessionState = SessionState.PENDING

    # Populated when state -> ADMITTED.
    admitted_at: Optional[float] = None
    # Populated when state moves to a terminal state.
    finished_at: Optional[float] = None
    # Tokens emitted so far, in commit order.
    output_token_ids: List[int] = field(default_factory=list)
    # Set when state == FAILED.
    error: Optional[BaseException] = None
    # Per-session async queue carrying committed tokens. Consumers of
    # the scheduler.iter_tokens() async iterator drain this; the
    # scheduler's worker pushes into it.
    token_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if self.max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}"
            )
        if not self.eos_token_ids:
            raise ValueError("eos_token_ids must be non-empty")

    # State transitions ------------------------------------------------

    def mark_admitted(self) -> None:
        if self.state != SessionState.PENDING:
            raise RuntimeError(
                f"cannot admit session in state {self.state.value}; "
                "only PENDING sessions may be admitted"
            )
        self.state = SessionState.ADMITTED
        self.admitted_at = time.monotonic()

    def mark_completed(self) -> None:
        self._finalize(SessionState.COMPLETED)

    def mark_cancelled(self) -> None:
        self._finalize(SessionState.CANCELLED)

    def mark_failed(self, error: BaseException) -> None:
        self.error = error
        self._finalize(SessionState.FAILED)

    def _finalize(self, terminal: SessionState) -> None:
        if self.state in {
            SessionState.COMPLETED, SessionState.CANCELLED, SessionState.FAILED
        }:
            raise RuntimeError(
                f"session {self.id} already finalized as {self.state.value}; "
                f"cannot transition to {terminal.value}"
            )
        self.state = terminal
        self.finished_at = time.monotonic()

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            SessionState.COMPLETED, SessionState.CANCELLED, SessionState.FAILED,
        }
