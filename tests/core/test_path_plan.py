"""Unit tests for kv_cache_proposer.path_plan.

Covers ADR 0007 §2.4 result types: ContinuationPlan and NewSession.
"""

from __future__ import annotations

import pytest

from kv_cache_proposer.path_plan import (
    ContinuationPlan,
    NewSession,
    PathPlan,
)


# ---------------------------------------------------------------------------
# ContinuationPlan
# ---------------------------------------------------------------------------


def test_continuation_plan_default_new_tokens_is_empty_list():
    plan = ContinuationPlan(skip_n=5)
    assert plan.skip_n == 5
    assert plan.new_tokens == []


def test_continuation_plan_with_new_tokens():
    plan = ContinuationPlan(skip_n=3, new_tokens=[10, 20, 30])
    assert plan.skip_n == 3
    assert plan.new_tokens == [10, 20, 30]


def test_continuation_plan_negative_skip_n_raises():
    with pytest.raises(ValueError, match="skip_n must be >= 0"):
        ContinuationPlan(skip_n=-1, new_tokens=[1])


def test_continuation_plan_zero_skip_n_is_valid():
    """skip_n=0 is the edge case where the cached state happens to
    cover the empty prefix. Valid per the design."""
    plan = ContinuationPlan(skip_n=0, new_tokens=[1, 2, 3])
    assert plan.skip_n == 0


def test_continuation_plan_is_frozen():
    plan = ContinuationPlan(skip_n=5, new_tokens=[1, 2])
    with pytest.raises((AttributeError, TypeError)):
        plan.skip_n = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NewSession
# ---------------------------------------------------------------------------


def test_new_session_with_prompt():
    session = NewSession(prompt=[1, 2, 3])
    assert session.prompt == [1, 2, 3]


def test_new_session_empty_prompt_raises():
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        NewSession(prompt=[])


def test_new_session_is_frozen():
    session = NewSession(prompt=[1])
    with pytest.raises((AttributeError, TypeError)):
        session.prompt = [99]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PathPlan union typing
# ---------------------------------------------------------------------------


def test_path_plan_union_accepts_either():
    """PathPlan is a Union; both concrete types satisfy it."""
    plans: list[PathPlan] = [
        ContinuationPlan(skip_n=5, new_tokens=[1, 2]),
        NewSession(prompt=[3, 4]),
    ]
    assert isinstance(plans[0], ContinuationPlan)
    assert isinstance(plans[1], NewSession)
