"""Async pipeline primitives (E5).

Provides reusable building blocks for cancellable, buffered, async
producer/consumer pipelines. Used today by E2's streaming layer (which
has an inline implementation that this module subsumes) and tomorrow
by the proposer/verifier compute-overlap work that will benchmark
``T_proposer`` and ``T_verifier`` separately and overlap them when the
hardware is GPU-bound on one but not the other.

Why a separate module rather than reusing E2's streaming bridge:

  * E2's bridge is bound to the engine.generate() shape: a single
    sync function with an on_token callback. The pipeline primitive
    here is generic over any sync producer (decoder, scheduler,
    multiplexer). When the proposer/verifier overlap work lands it
    will need a producer that emits *blocks* of tokens with
    accept/reject metadata, not single ids — the streaming bridge
    cannot serve that shape, but this primitive can.
  * Sharing concurrency primitives across multiple async producers
    lets us factor the disconnect / cancellation / exception
    propagation logic into one tested place.

Submodules:
    coordinator   PipelineCoordinator and helpers.
"""

from .coordinator import (
    PipelineClosed,
    PipelineCoordinator,
    PipelineError,
    StreamSentinel,
)

__all__ = [
    "PipelineClosed",
    "PipelineCoordinator",
    "PipelineError",
    "StreamSentinel",
]
