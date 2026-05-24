"""Request scheduler with admission control + fair queuing (E4).

Drives N concurrent inference requests against a single underlying
:class:`~inference_engine.server.engine.Engine`. Each admitted request
gets a :class:`~inference_engine.memory.KVSlab` from a
:class:`~inference_engine.memory.SlabPool`; when all slabs are in use,
new requests either wait in a fair FIFO queue or are rejected
(rejection by default; queueing opt-in via ``admission_policy``).

What this is and is not:

  * It IS a request-level scheduler with admission control,
    per-session slab management, fair FIFO queuing, and clean
    cancellation. Concurrent ``submit`` calls produce concurrent
    async iterators that yield tokens as they're committed.
  * It is NOT (yet) batched-tensor verification. The underlying
    engine still runs one session's forward at a time, serialized by
    an internal lock. The slab pool is the bookkeeping that makes
    future batched-tensor verification a small additional change
    — the verifier just gets handed N slabs instead of one.

Submodules:
    config       SchedulerConfig dataclass.
    session      Session record + state machine.
    scheduler    The Scheduler class itself.
"""

from .config import AdmissionPolicy, SchedulerConfig
from .pooled_verifier import PooledVerifier
from .scheduler import RequestRejected, Scheduler
from .session import Session, SessionState

__all__ = [
    "AdmissionPolicy",
    "PooledVerifier",
    "RequestRejected",
    "Scheduler",
    "SchedulerConfig",
    "Session",
    "SessionState",
]
