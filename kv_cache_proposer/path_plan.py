"""Path-selection result types for cross-request KV cache reuse.

Implements the result types ``ContinuationPlan`` and ``NewSession``
described in ADR 0007 §2.4. Both CPU and MLX verifiers expose a
``path_select(prompt)`` method returning one of these.

ADR 0007 §2.4 contract recap:

  Continuation precondition (both must hold):
    1. ``len(prompt) >= cache_logical_end`` (the new prompt extends
       at or past the position the cache already covers).
    2. The new prompt's tokens at every cached logical position
       equal ``cached_token_sequence`` at the corresponding slot.

  When the precondition holds → ``ContinuationPlan(skip_n,
  new_tokens)``: the verifier should run ``prefill_incremental``
  on ``new_tokens`` (the suffix of ``prompt`` after the cached
  prefix), reusing the existing K/V cache state.

  When the precondition fails (cold start, shorter history,
  diverging history) → ``NewSession(prompt)``: the verifier should
  run a full ``prefill(prompt)`` (which calls ``reset()`` first and
  rebuilds the cache from scratch).

The two paths are first-class deterministic actions per ADR 0007
§2.4.c. Selecting NewSession is **not** a fallback from
ContinuationPlan — both produce bit-identical output for their
input class (per §2.7); the only difference is computational cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union


@dataclass(frozen=True)
class ContinuationPlan:
    """Continuation path: reuse the cached prefix.

    Attributes
    ----------
    skip_n
        Number of tokens at the start of the new prompt that are
        already covered by the cache. The verifier should NOT
        re-prefill these. By construction
        ``skip_n == verifier.next_global_position`` at the moment
        ``path_select`` ran (see ADR 0007 §2.9 INV-2).
    new_tokens
        The suffix of the new prompt that is NOT yet in the cache.
        Length = ``len(prompt) - skip_n``. Always non-empty when the
        plan is returned (a continuation that adds zero new tokens
        is encoded as ``ContinuationPlan(skip_n=len(prompt),
        new_tokens=[])`` only in the unusual case where the new
        prompt exactly matches the cache state — most callers
        handle that as a no-op forward).
    """

    skip_n: int
    new_tokens: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.skip_n < 0:
            raise ValueError(f"skip_n must be >= 0, got {self.skip_n}")
        # Note: new_tokens may be empty (the rare exact-match case).


@dataclass(frozen=True)
class NewSession:
    """New-session path: reset the cache and run full prefill.

    Triggered when any §2.4.b sub-case applies: cold start, shorter
    history, or diverging history.

    Attributes
    ----------
    prompt
        The full prompt to prefill. Always non-empty (the verifier
        rejects empty prompts upstream).
    """

    prompt: List[int]

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("NewSession.prompt must be non-empty")


PathPlan = Union[ContinuationPlan, NewSession]


__all__ = ["ContinuationPlan", "NewSession", "PathPlan"]
