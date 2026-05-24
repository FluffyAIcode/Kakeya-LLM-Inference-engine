"""PooledVerifier wrapper: ties a verifier's lifecycle to a slab pool.

This is the **intermediate step** described in ADR 0003. It does not
make slab tensors hold the real KV — that is the deferred full
refactor. What it does:

  * On ``prefill()``: acquires a slab from the pool (releasing any
    previously held one).
  * On ``reset()``: releases the held slab.
  * After every forward (``prefill``, ``forward_block``,
    ``append_token``, ``commit_or_truncate``): writes the verifier's
    real ``stats.peak_kv_bytes`` snapshot into the slab's
    ``live_kv_bytes_override`` so ``slab.live_kv_bytes`` reports
    real numbers, not placeholder tensor bytes.

The wrapper is a structural pass-through: every public method on
the underlying verifier is delegated. We only intercept the calls
that change cache state.

Why a wrapper rather than modifying SinkWindowVerifier directly:

  * Avoids a circular dependency between ``kv_cache_proposer``
    (where the verifier lives) and ``inference_engine.memory``
    (where the pool lives). Today layering goes
    ``inference_engine -> kv_cache_proposer``; reversing that for
    the verifier would invert the import graph.
  * Keeps the verifier's ``DynamicCache`` path bit-identical to
    v0.1.0. The wrapper does not touch the model forward.
  * Makes the integration optional: callers that don't care about
    pool-backed memory accounting use the bare verifier; callers
    that do (multi-session HTTP) wrap with ``PooledVerifier``.

Verifier protocol assumed:

    verifier.prefill(prompt_ids: list[int]) -> None
    verifier.forward_block(tokens: list[int]) -> torch.Tensor
    verifier.commit_or_truncate(forwarded: int, accepted: int) -> None
    verifier.append_token(token_id: int) -> torch.Tensor
    verifier.reset() -> None
    verifier.stats.peak_kv_bytes  (int, updated by verifier)
    verifier.tokenizer            (passthrough)
    verifier.next_token_logits    (passthrough)
    verifier.cache_logical_size   (passthrough)
    verifier.next_global_position (passthrough)

Both PyTorch ``SinkWindowVerifier`` and MLX
``MLXSinkWindowVerifier`` satisfy this; future verifiers must too.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import KVSlab


class PooledVerifier:
    """Wraps a verifier; manages slab-pool acquire/release per session."""

    def __init__(self, verifier: Any, pool: SlabPool) -> None:
        if pool is None:
            raise ValueError("pool must not be None")
        self._verifier = verifier
        self._pool = pool
        self._slab: Optional[KVSlab] = None

    # ------------------------------------------------------------------
    # Verifier-protocol methods we intercept
    # ------------------------------------------------------------------

    def prefill(self, prompt_ids: List[int]) -> None:
        # Acquire a slab for this session; release any prior one
        # (defensive — same verifier instance reused across sessions).
        self._release_slab_if_held()
        self._slab = self._pool.acquire()
        try:
            self._verifier.prefill(prompt_ids)
            self._sync_slab_bytes()
        except BaseException:
            # Release the slab on failure so the pool is not stuck.
            self._release_slab_if_held()
            raise

    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        out = self._verifier.forward_block(tokens)
        self._sync_slab_bytes()
        return out

    def commit_or_truncate(self, forwarded: int, accepted: int) -> None:
        self._verifier.commit_or_truncate(forwarded=forwarded, accepted=accepted)
        self._sync_slab_bytes()

    def append_token(self, token_id: int) -> torch.Tensor:
        out = self._verifier.append_token(token_id)
        self._sync_slab_bytes()
        return out

    def reset(self) -> None:
        self._verifier.reset()
        self._release_slab_if_held()

    # ------------------------------------------------------------------
    # Pass-through attributes the speculative decoder reads directly
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self._verifier.tokenizer

    @property
    def stats(self):
        return self._verifier.stats

    @property
    def next_token_logits(self):
        return self._verifier.next_token_logits

    @next_token_logits.setter
    def next_token_logits(self, value):
        self._verifier.next_token_logits = value

    @property
    def cache_logical_size(self) -> int:
        return self._verifier.cache_logical_size

    @property
    def next_global_position(self) -> int:
        return self._verifier.next_global_position

    @property
    def config(self):
        return self._verifier.config

    @property
    def slab(self) -> Optional[KVSlab]:
        """The currently-held slab, if any. Public for tests."""
        return self._slab

    @property
    def pool(self) -> SlabPool:
        return self._pool

    @property
    def inner(self):
        """The wrapped verifier — escape hatch for callers that need
        access to verifier-specific extras (e.g. ``quantization``
        on MLX). Use sparingly; depending on this defeats the
        wrapper's abstraction."""
        return self._verifier

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_slab_bytes(self) -> None:
        """Copy the verifier's real KV byte count onto the slab.

        The verifier updates ``stats.peak_kv_bytes`` on every forward.
        We use the *current* size (the verifier also publishes
        ``stats.peak_kv_bytes`` as a running max — which is fine
        for our purposes since we want pool gauges to reflect the
        worst case during the session).
        """
        if self._slab is None:  # pragma: no cover - defensive; all callers acquire first
            return
        bytes_ = int(getattr(self._verifier.stats, "peak_kv_bytes", 0))
        self._slab.live_kv_bytes_override = bytes_

    def _release_slab_if_held(self) -> None:
        if self._slab is not None:
            self._pool.release(self._slab)
            self._slab = None
