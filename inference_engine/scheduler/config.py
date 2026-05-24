"""Scheduler configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AdmissionPolicy(str, Enum):
    """How the scheduler handles admission when all slabs are in use.

    ``REJECT`` — submit() raises :class:`RequestRejected` immediately.
                Caller surfaces this as HTTP 429 (or equivalent).
    ``QUEUE``  — submit() blocks until a slab frees up. Fair FIFO
                ordering by submission time.
    """

    REJECT = "reject"
    QUEUE = "queue"


@dataclass(frozen=True)
class SchedulerConfig:
    """Process-wide scheduler tunables.

    ``max_concurrent`` must equal the underlying SlabPool's
    ``num_slabs``; the scheduler validates this on construction so a
    misconfiguration fails at startup, not at the first overflow.

    ``admission_policy`` defaults to ``REJECT`` because the HTTP
    layer prefers immediate 429 over indefinite client wait. Batch
    workloads that explicitly want queueing flip to ``QUEUE``.

    ``queue_max_wait_s`` only takes effect under ``QUEUE`` policy and
    bounds how long a submission can wait before timing out (and
    raising). 0 disables the timeout (wait forever).
    """

    max_concurrent: int
    admission_policy: AdmissionPolicy = AdmissionPolicy.REJECT
    queue_max_wait_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be positive, got {self.max_concurrent}"
            )
        if self.queue_max_wait_s < 0:
            raise ValueError(
                f"queue_max_wait_s must be >= 0, got {self.queue_max_wait_s}"
            )
