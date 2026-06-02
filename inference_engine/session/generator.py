"""GenerationCoordinator — ADR 0008 PR-B3 (Phase B).

Session-aware token generation against a verifier. v0.3 ships
**greedy decoding only**; speculative-decoding integration (the
DLM proposer + AR verifier rejection sampling that is Kakeya's
distinguishing feature) is reserved for a later PR. The wire
contract — :class:`runtime_pb2.GenerateResponse` with its
``token_id`` / ``done`` / ``truncated`` ``oneof`` payload — is
algorithm-agnostic, so the upgrade path lands without breaking
clients.

The coordinator yields a stream of typed events:

  * :class:`TokenEvent` — one per committed token, in order
  * :class:`HistoryTruncatedEvent` — emitted at most once at the
    start of a Generate call when the session is already operating
    in sink+window-truncated mode (per `runtime.proto` contract:
    "Emitted at most once per Generate call, before any token_id
    event in that call.")
  * :class:`DoneEvent` — terminal; emitted exactly once at the end

Layering note: this coordinator depends on the same
:class:`VerifierProtocol` PR-B2 introduced. It does not call
``verifier.prefill`` — that is the AppendTokens path's
responsibility (PR-B2). Generate operates on whatever cache state
:meth:`AppendTokensCoordinator.append_tokens` left behind, which
is precisely the byte-exact prefill-incremental contract from
ADR 0008 §2.3 in action.

Anomaly invariants:

  * INV-1 (parallel-sequence consistency): enforced after every
    generated token via :meth:`SessionStore.append_tokens`'s
    INV-1 check (the same check PR-B2's coordinator triggers on
    user-submitted tokens).
  * INV-2 (position monotonicity): enforced after every token
    via :meth:`SessionStore.record_position_advance`.
  * INV-3 (continuation-path determinism): for the same
    ``(session_id, history_token_ids)`` pair under greedy
    decoding, repeated Generate calls produce bit-identical token
    sequences. Tested with a deterministic ``FakeVerifier`` in
    the unit suite and against the real Qwen3 verifier under
    ``tests/core/`` on Mac M4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch

from inference_engine.session.coordinator import (
    VerifierProtocol,
    _sync_slab_bytes,
)
from inference_engine.session.store import SessionStore


# Stop-reason string constants. Mirror the protobuf enum but defined
# here so this module has no protobuf dependency (the gRPC servicer
# does the string -> enum translation).
STOP_REASON_MAX_TOKENS = "max_tokens"
STOP_REASON_EOS = "eos"
STOP_REASON_CANCELLED = "cancelled"
STOP_REASON_TRUNCATED = "truncated"


@dataclass(frozen=True)
class TokenEvent:
    """One committed token, yielded in generation order."""

    token_id: int


@dataclass(frozen=True)
class HistoryTruncatedEvent:
    """Cache no longer holds the full session history.

    ``dropped_token_count`` is the difference between the session's
    full history length and what the verifier's sink+window cache
    currently holds. Per the runtime contract this event is
    emitted at most once per Generate call, before any TokenEvent.
    """

    dropped_token_count: int


@dataclass(frozen=True)
class DoneEvent:
    """Terminal event for a Generate call.

    ``prefill_seconds`` is 0.0 in PR-B3 because Generate has no
    separate prefill phase — the prefill ran inside the preceding
    AppendTokens call. The field is preserved on the wire for
    forward-compatibility with future PRs that re-introduce a
    prefill step (e.g., for speculative-decoding warmup).
    """

    stop_reason: str
    generated_token_count: int
    prefill_seconds: float
    total_seconds: float


GenerateEvent = Union[TokenEvent, HistoryTruncatedEvent, DoneEvent]


class GenerationCoordinator:
    """Greedy session-aware token generation against a verifier."""

    def __init__(
        self,
        store: SessionStore,
        verifier: VerifierProtocol,
    ) -> None:
        self._store = store
        self._verifier = verifier

    def generate(
        self,
        session_id: str,
        *,
        max_tokens: int,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Iterator[GenerateEvent]:
        """Yield a stream of GenerateEvents for ``session_id``.

        Raises:
          * :class:`SessionNotFoundError` — unknown / closed / evicted
            session id.
          * :class:`ValueError` — invalid argument (e.g.,
            ``max_tokens < 1``, sampling param set in v0.3 greedy
            mode, no AppendTokens preceded this call).
          * :class:`InvariantViolation` — INV-1 / INV-2 violation
            during a generation step.

        v0.3 greedy contract:
          * ``temperature`` / ``top_p`` / ``top_k`` MUST be unset
            (or in their no-op default of 0 / unset / 1
            respectively). Setting any of them raises ValueError —
            the runtime refuses to silently downgrade a non-greedy
            request to greedy (per ADR 0008 §2.10 "no graceful
            degradation").
          * ``seed`` is accepted (per OQ-4 default) but ignored:
            greedy decoding has no RNG to seed, and the
            byte-exact contract over a fixed seed reduces to the
            byte-exact contract under any seed because there is no
            seed-dependent randomness.
        """
        if max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1, got {max_tokens}"
            )
        if temperature is not None and float(temperature) != 0.0:
            raise ValueError(
                f"v0.3 supports only greedy decoding; temperature "
                f"must be 0 or unset, got {temperature}"
            )
        if top_p is not None:
            raise ValueError(
                "v0.3 supports only greedy decoding; top_p must be "
                "unset (greedy ignores it)"
            )
        if top_k is not None and int(top_k) != 1:
            raise ValueError(
                f"v0.3 supports only greedy decoding; top_k must be "
                f"1 or unset, got {top_k}"
            )
        # seed is accepted but not used in greedy; explicitly ignore.
        del seed

        session = self._store.get_session(session_id)
        if session.next_global_position == 0:
            raise ValueError(
                "session has no history yet; AppendTokens must "
                "precede Generate (the first token's logits are "
                "the prefill's last position)"
            )

        # Emit HistoryTruncated at start if the cache is already in
        # truncated mode. Per the proto contract, this event is
        # emitted at most once per Generate call and BEFORE any
        # token_id event — we honor both by checking once at the
        # start and never emitting again during this call.
        history_len = len(session.history_token_ids)
        cached_len = len(session.cached_token_sequence)
        if history_len > cached_len:
            yield HistoryTruncatedEvent(
                dropped_token_count=history_len - cached_len,
            )

        eos_set = set(session.eos_token_ids)
        t0 = time.perf_counter()
        # Generate has no separate prefill phase in PR-B3; report 0.
        prefill_seconds = 0.0
        generated_count = 0

        for _step in range(max_tokens):
            # Greedy: argmax of the verifier's last next_token_logits.
            next_token = int(
                torch.argmax(self._verifier.next_token_logits).item()
            )

            # Forward + commit (forwarded == accepted for prompt-mode
            # appends; same contract used by AppendTokens, just one
            # token at a time).
            block_logits = self._verifier.forward_block([next_token])
            self._verifier.commit_or_truncate(forwarded=1, accepted=1)
            self._verifier.next_token_logits = block_logits[-1].clone()

            # Mirror state from verifier onto session BEFORE the
            # store's INV-1 check runs (it compares
            # session.cached_token_sequence length against
            # verifier.k_seq_length).
            session.cached_token_sequence = list(
                self._verifier.cached_token_sequence,
            )
            self._store.append_tokens(session_id, [next_token])
            self._store.record_position_advance(
                session_id, self._verifier.next_global_position,
            )
            generated_count += 1

            yield TokenEvent(token_id=next_token)

            if next_token in eos_set:
                # Mirror final KV bytes onto the slab so the next
                # GetSessionInfo reads the correct live count
                # (PR-E1c). Once the cache is at sink+window
                # capacity, this value plateaus and the caller can
                # observe the architectural KV bound empirically.
                _sync_slab_bytes(session, self._verifier)
                yield DoneEvent(
                    stop_reason=STOP_REASON_EOS,
                    generated_token_count=generated_count,
                    prefill_seconds=prefill_seconds,
                    total_seconds=time.perf_counter() - t0,
                )
                return

        _sync_slab_bytes(session, self._verifier)
        yield DoneEvent(
            stop_reason=STOP_REASON_MAX_TOKENS,
            generated_token_count=generated_count,
            prefill_seconds=prefill_seconds,
            total_seconds=time.perf_counter() - t0,
        )
