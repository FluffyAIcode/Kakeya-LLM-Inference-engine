"""Request scheduler with admission control and fair queuing.

The scheduler is the single entry point for concurrent inference
requests. Construction binds it to one engine and one slab pool;
``submit(...)`` returns an async iterator yielding committed tokens.

Lifecycle of a single submission:

    1. submit(prompt, max_new_tokens, eos_set)
       └── if pool full and policy=REJECT: raise RequestRejected
       └── if pool full and policy=QUEUE:  await pool capacity (or timeout)
    2. Acquire slab from pool, mark session ADMITTED.
    3. Spawn worker that:
         - calls engine.generate(prompt, max_new_tokens, eos_set, on_token)
         - on_token: push token id into session.token_queue
         - on completion: push terminal sentinel into queue, release slab.
    4. iter_tokens(session) async iterator drains token_queue until
       the terminal sentinel arrives.
    5. The session's terminal state (COMPLETED / CANCELLED / FAILED)
       is observable via session.state once iter_tokens exhausts.

Concurrency:

  * Multiple submit() calls produce multiple worker tasks.
  * The underlying engine.generate() is **serialized** through an
    internal asyncio.Lock — without true batched-tensor verification
    we cannot run two generations concurrently against one verifier
    KV cache. This is honest: the scheduler buys you concurrent
    *requests* and admission control today; concurrent *forwards*
    will come when batched verification lands.
  * The slab pool is the right place for that future change to slot
    into — when batched-tensor verification lands, the scheduler
    will hand N slabs to the verifier per step instead of one,
    without changing any other code.

Cancellation:

  * Client cancellation (HTTP disconnect or explicit ``cancel``)
    propagates by setting session.state to CANCELLED via
    ``cancel_session``. The on_token callback returns True on the
    next commit, the engine stops, the worker releases the slab.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable, List, Optional, Protocol, runtime_checkable

from inference_engine.memory.pool import PoolExhausted, SlabPool
from inference_engine.memory.slab import KVSlab

from .config import AdmissionPolicy, SchedulerConfig
from .session import Session, SessionState


@runtime_checkable
class Engine(Protocol):
    """Minimal generation contract the scheduler depends on.

    Structurally identical to
    ``inference_engine.server.engine.Engine`` but redeclared here so
    the scheduler module does not import the server (a deliberate
    layering choice — the scheduler is consumed by both HTTP and
    non-HTTP entry points). A future refactor can lift this protocol
    into a top-level ``inference_engine.engine`` module that both
    server and scheduler import; for now the duck-typed protocol
    pattern is enough.
    """

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: List[int],
        on_token: Optional[Callable[[int], bool]] = None,
    ): ...  # pragma: no cover - Protocol body


# Sentinel pushed to a session's token_queue when generation finishes.
_SESSION_DONE = object()


class RequestRejected(RuntimeError):
    """Raised by :meth:`Scheduler.submit` when admission fails.

    Two failure modes:
      * Pool full under REJECT policy
      * Queue wait exceeded ``queue_max_wait_s`` under QUEUE policy
    """


@dataclass
class _SchedulerStats:
    """Lightweight running counters for diagnostics."""
    total_submitted: int = 0
    total_admitted: int = 0
    total_rejected: int = 0
    total_completed: int = 0
    total_cancelled: int = 0
    total_failed: int = 0


class Scheduler:
    """Request scheduler over a single engine and slab pool."""

    def __init__(
        self,
        engine: Engine,
        pool: SlabPool,
        config: SchedulerConfig,
    ) -> None:
        if config.max_concurrent != pool.total_count:
            raise ValueError(
                f"SchedulerConfig.max_concurrent={config.max_concurrent} "
                f"does not match pool.total_count={pool.total_count}"
            )
        self._engine = engine
        self._pool = pool
        self._config = config
        self._engine_lock = asyncio.Lock()
        # Active session bookkeeping: session_id -> (session, slab,
        # worker_task). The worker_task is needed for shutdown so we
        # can cancel + await mid-flight generations.
        self._active: dict[str, tuple[Session, KVSlab, asyncio.Task]] = {}
        self._active_lock = asyncio.Lock()
        self._stats = _SchedulerStats()
        # FIFO queue of (Session, future) pending admission under
        # QUEUE policy. Future is resolved with the slab once one
        # frees up.
        self._wait_queue: List[tuple[Session, asyncio.Future]] = []
        self._wait_queue_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: List[int],
    ) -> Session:
        """Submit a request; returns a Session whose tokens you iterate.

        Caller usage::

            session = await scheduler.submit(prompt_ids, 256, eos_ids)
            async for token_id in scheduler.iter_tokens(session):
                ...
            assert session.state == SessionState.COMPLETED
        """
        session = Session(
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=list(eos_token_ids),
        )
        self._stats.total_submitted += 1

        slab = await self._admit(session)
        session.mark_admitted()
        async with self._active_lock:
            worker = asyncio.create_task(self._run_session(session, slab))
            self._active[session.id] = (session, slab, worker)
        self._stats.total_admitted += 1
        return session

    async def iter_tokens(self, session: Session) -> AsyncIterator[int]:
        """Drain a session's token stream.

        Yields committed token ids in order. The async iterator
        terminates when the worker pushes the terminal sentinel,
        which corresponds to a state transition into COMPLETED,
        CANCELLED, or FAILED. After exhaustion, ``session.state``
        is observable; on FAILED, ``session.error`` is the cause.

        If the session was CANCELLED before any token arrived (e.g.
        the caller cancelled almost immediately after submit), the
        iterator yields nothing and exits.
        """
        while True:
            item = await session.token_queue.get()
            if item is _SESSION_DONE:
                break
            yield int(item)
        # If the worker stored an exception on the session, surface
        # it now — the caller's iter_tokens loop should error out.
        if session.state == SessionState.FAILED and session.error is not None:
            raise session.error

    async def cancel_session(self, session: Session) -> None:
        """Request cancellation of an active session.

        The next on_token callback inside the worker will return True,
        the engine stops, the worker releases the slab. Idempotent: a
        second cancel of an already-finalized session is a no-op.
        """
        if session.is_terminal:
            return
        # Marking CANCELLED here rather than in the worker because
        # the on_token callback consults session.state to decide
        # whether to stop, and we want the cancel signal to be
        # visible immediately.
        session.mark_cancelled()

    async def shutdown(self) -> None:
        """Cancel all active sessions and await worker teardown.

        Used by the HTTP server during graceful shutdown. After this
        returns, the slab pool has all slabs released and no worker
        tasks are running.

        Order matters: we reject pending admissions FIRST, so that
        when active workers release their slabs there is no waiter
        to inherit them — the slabs simply return to the pool.
        Otherwise the freed slab would be handed to a queued
        submission that we are about to cancel anyway.
        """
        # 1. Reject all queued (PENDING) submissions.
        async with self._wait_queue_lock:
            for _session, fut in self._wait_queue:
                if not fut.done():
                    fut.set_exception(
                        RequestRejected("scheduler shutting down")
                    )
            self._wait_queue.clear()

        # 2. Cancel all active sessions.
        async with self._active_lock:
            sessions = list(self._active.values())
        for session, _slab, _worker in sessions:
            if not session.is_terminal:
                session.mark_cancelled()

        # 3. Await all workers; they each release their own slab back
        # to the pool (and there are no waiters to wake — see step 1).
        for _, _, worker in sessions:
            try:
                await worker
            except BaseException:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return len(self._wait_queue)

    @property
    def stats(self) -> _SchedulerStats:
        return self._stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _admit(self, session: Session) -> KVSlab:
        """Acquire a slab for ``session``, applying admission policy.

        On success returns the slab. On failure raises
        :class:`RequestRejected`.
        """
        slab = self._pool.acquire_optional()
        if slab is not None:
            return slab

        # Pool exhausted.
        if self._config.admission_policy == AdmissionPolicy.REJECT:
            self._stats.total_rejected += 1
            raise RequestRejected(
                f"slab pool exhausted ({self._pool.total_count} in use); "
                "admission policy is REJECT"
            )

        # QUEUE policy: park on the wait queue.
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        async with self._wait_queue_lock:
            self._wait_queue.append((session, future))

        try:
            if self._config.queue_max_wait_s > 0:
                slab = await asyncio.wait_for(
                    future, timeout=self._config.queue_max_wait_s
                )
            else:
                slab = await future
        except asyncio.TimeoutError as exc:
            # Remove ourselves from the queue.
            async with self._wait_queue_lock:
                self._wait_queue = [
                    (s, f) for s, f in self._wait_queue if s.id != session.id
                ]
            self._stats.total_rejected += 1
            raise RequestRejected(
                f"queue wait exceeded queue_max_wait_s="
                f"{self._config.queue_max_wait_s}"
            ) from exc
        return slab

    async def _release_slab_and_wake_waiter(self, slab: KVSlab) -> None:
        """Release a slab back to the pool. If anyone is waiting on
        the admission queue, hand the slab directly to them and
        re-acquire from the pool to keep the inflight counts
        consistent. Otherwise simply release.
        """
        async with self._wait_queue_lock:
            if self._wait_queue:
                _waiting_session, future = self._wait_queue.pop(0)
                if not future.done():
                    # The slab transfer is direct: we don't release
                    # to the pool and re-acquire (which would race
                    # against another submit). We pass our slab to
                    # the waiter; they own the lifecycle from here.
                    future.set_result(slab)
                    return
        # No waiters — return to pool.
        self._pool.release(slab)

    async def _run_session(self, session: Session, slab: KVSlab) -> None:
        """Worker coroutine for one session.

        Calls engine.generate under the engine lock with an on_token
        callback that pushes tokens into the session's queue. When
        generation finishes, pushes the terminal sentinel and releases
        the slab.
        """
        try:
            async with self._engine_lock:
                # Re-check state after acquiring the lock — caller may
                # have cancelled while we were waiting.
                if session.state == SessionState.CANCELLED:
                    return
                loop = asyncio.get_running_loop()

                def on_token(tok_id: int) -> bool:
                    if session.state == SessionState.CANCELLED:
                        return True
                    session.output_token_ids.append(int(tok_id))
                    enqueue = asyncio.run_coroutine_threadsafe(
                        session.token_queue.put(int(tok_id)), loop
                    )
                    # Preserve token-before-sentinel ordering. The worker
                    # runs in a thread, while the terminal sentinel is pushed
                    # back on the event loop after generate() returns.
                    enqueue.result()
                    return False

                result = await asyncio.to_thread(
                    self._engine.generate,
                    session.prompt_ids, session.max_new_tokens,
                    session.eos_token_ids, on_token,
                )

            # Out of engine lock — finalize state.
            _ = result  # tokens were already streamed via on_token
            if session.state == SessionState.CANCELLED:
                # Already counted by cancel_session caller; we just
                # observe the terminal state here.
                self._stats.total_cancelled += 1
            elif session.state == SessionState.ADMITTED:
                session.mark_completed()
                self._stats.total_completed += 1
        except BaseException as exc:
            session.mark_failed(exc)
            self._stats.total_failed += 1
        finally:
            # Always: signal end-of-stream to consumers and free the slab.
            await session.token_queue.put(_SESSION_DONE)
            await self._release_slab_and_wake_waiter(slab)
            async with self._active_lock:
                self._active.pop(session.id, None)
