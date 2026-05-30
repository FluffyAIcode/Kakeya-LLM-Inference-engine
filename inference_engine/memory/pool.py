"""Pool of pre-allocated KV slabs.

A :class:`SlabPool` owns ``N`` :class:`KVSlab` instances and hands them
out per session. ``acquire`` returns a free slab in O(1); ``release``
returns it to the pool, ``reset()``-ing its logical size so the next
session sees an empty slab.

The pool is thread-safe (a ``threading.Lock`` guards the free list)
because the future continuous-batching scheduler runs decoder threads
that may release slabs concurrently with new admissions.

When all slabs are in use, ``acquire`` raises :class:`PoolExhausted`.
We do **not** queue / wait — admission control is the scheduler's job
(E4), which decides whether to reject a new request, queue it, or
spawn additional capacity. The pool only reports "full" honestly.

Allocation cost: O(num_slabs × num_layers × num_heads × capacity ×
head_dim × 2). For a typical config (32 layers × 32 heads × 68 ×
128 × bf16 × 2 = 17 MB per slab), 64 slabs occupy ~1 GB. This is the
single largest persistent allocation in the engine and is intentional
— sized once at startup, never re-sized at runtime.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from .slab import KVSlab, SlabConfig


class PoolExhausted(RuntimeError):
    """Raised by :meth:`SlabPool.acquire` when no slabs are free."""


class SlabPool:
    """Fixed-size pool of pre-allocated KV slabs."""

    def __init__(self, num_slabs: int, slab_config: SlabConfig) -> None:
        if num_slabs <= 0:
            raise ValueError(f"num_slabs must be positive, got {num_slabs}")
        self._slab_config = slab_config
        self._all_slabs: List[KVSlab] = [KVSlab(slab_config) for _ in range(num_slabs)]
        self._free: List[int] = list(range(num_slabs))
        self._in_use: set[int] = set()
        self._lock = threading.Lock()
        # Map slab object id() -> index, used by release() to find the
        # right slot without scanning. Set up at construction; never
        # mutated.
        self._index_by_id = {id(s): i for i, s in enumerate(self._all_slabs)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> KVSlab:
        """Return a free slab. Raises :class:`PoolExhausted` if none."""
        with self._lock:
            if not self._free:
                raise PoolExhausted(
                    f"all {self.total_count} slabs in use; admission control "
                    "must reject or queue this session"
                )
            idx = self._free.pop()
            self._in_use.add(idx)
            slab = self._all_slabs[idx]
        return slab

    def release(self, slab: KVSlab) -> None:
        """Return a slab to the pool. Resets its logical size to 0.

        Raises :class:`ValueError` if the slab does not belong to
        this pool, or if it has already been released (double-free).
        """
        idx = self._index_by_id.get(id(slab))
        if idx is None:
            raise ValueError("slab does not belong to this pool")
        with self._lock:
            if idx not in self._in_use:
                raise ValueError(
                    f"slab at index {idx} is not currently in use "
                    "(double release?)"
                )
            self._in_use.discard(idx)
            self._free.append(idx)
        slab.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_count(self) -> int:
        return len(self._all_slabs)

    @property
    def available_count(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def in_use_count(self) -> int:
        with self._lock:
            return len(self._in_use)

    @property
    def slab_config(self) -> SlabConfig:
        return self._slab_config

    @property
    def total_kv_bytes(self) -> int:
        """Sum of physical KV bytes across all slabs (capacity, not live)."""
        return sum(s.kv_bytes for s in self._all_slabs)

    @property
    def live_kv_bytes(self) -> int:
        """Sum of *live* KV bytes across slabs currently in use.

        This is what we want to expose as a Prometheus gauge for the
        long-session memory-stability claim (ADR 0006 §2.3): it's the
        actual KV memory consumed by active sessions right now, not
        the pool's pre-allocated capacity. Free slabs contribute 0
        (their ``logical_size`` is reset on release).

        :class:`~inference_engine.scheduler.pooled_verifier.PooledVerifier`
        keeps each slab's ``live_kv_bytes_override`` synced with the
        real verifier KV size, so this aggregate matches the verifier
        backend's actual memory footprint.
        """
        with self._lock:
            in_use_set = set(self._in_use)
        return sum(
            self._all_slabs[i].live_kv_bytes for i in sorted(in_use_set)
        )

    def acquire_optional(self) -> Optional[KVSlab]:
        """Acquire a slab if available, else return ``None`` instead of raising.

        For schedulers that prefer a None check over a try/except in
        the hot path. Functionally equivalent to ``acquire`` wrapped
        in a try, but avoids the exception construction cost when
        the pool is full.
        """
        with self._lock:
            if not self._free:
                return None
            idx = self._free.pop()
            self._in_use.add(idx)
        return self._all_slabs[idx]
