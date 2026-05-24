"""Unit tests for SchedulerConfig + AdmissionPolicy."""

from __future__ import annotations

import pytest

from inference_engine.scheduler.config import AdmissionPolicy, SchedulerConfig


def test_construction_defaults():
    cfg = SchedulerConfig(max_concurrent=4)
    assert cfg.max_concurrent == 4
    assert cfg.admission_policy is AdmissionPolicy.REJECT
    assert cfg.queue_max_wait_s == 0.0


def test_explicit_construction():
    cfg = SchedulerConfig(
        max_concurrent=8,
        admission_policy=AdmissionPolicy.QUEUE,
        queue_max_wait_s=1.5,
    )
    assert cfg.max_concurrent == 8
    assert cfg.admission_policy is AdmissionPolicy.QUEUE
    assert cfg.queue_max_wait_s == 1.5


@pytest.mark.parametrize("n", [0, -1, -100])
def test_non_positive_max_concurrent_raises(n):
    with pytest.raises(ValueError, match="max_concurrent must be positive"):
        SchedulerConfig(max_concurrent=n)


@pytest.mark.parametrize("w", [-0.1, -1.0])
def test_negative_queue_wait_raises(w):
    with pytest.raises(ValueError, match="queue_max_wait_s must be >= 0"):
        SchedulerConfig(max_concurrent=1, queue_max_wait_s=w)


def test_admission_policy_enum_values():
    assert AdmissionPolicy.REJECT.value == "reject"
    assert AdmissionPolicy.QUEUE.value == "queue"


def test_frozen_dataclass():
    cfg = SchedulerConfig(max_concurrent=2)
    with pytest.raises(Exception):
        cfg.max_concurrent = 8  # type: ignore[misc]
