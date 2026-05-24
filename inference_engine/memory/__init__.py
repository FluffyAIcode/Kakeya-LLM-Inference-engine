"""Memory primitives for the engine (E3).

The engine's per-session KV cache is a fixed-size object thanks to the
sink+window invariant (see ADR 0001 §4 and ``docs/local-inference-engine.md``
section "Why we are *not* using PagedAttention"). With known maximum
size, we can pre-allocate the cache as a contiguous slab and skip the
fragmentation / page-table machinery PagedAttention exists to solve.

Submodules:
    slab    KVSlab — single-session fixed-capacity sink+window KV cache.
    pool    SlabPool — pool of N pre-allocated slabs handed out per
            session and returned on session end.

Both are platform-neutral (torch CPU/CUDA only; MLX integration is a
follow-up). They are consumed by E4's continuous batching scheduler
and by the future production verifier wiring.
"""

from .pool import PoolExhausted, SlabPool
from .slab import KVSlab, SlabConfig

__all__ = ["KVSlab", "SlabConfig", "SlabPool", "PoolExhausted"]
