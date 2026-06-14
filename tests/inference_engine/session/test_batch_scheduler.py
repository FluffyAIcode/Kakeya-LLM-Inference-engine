"""Guard-logic tests for BatchedDecodeScheduler (Linux, no model forward).

The batched-forward correctness is validated end-to-end on H200 (per-session
recall 1.0 in k3_served_batched_scheduler_bench.py). Here we lock the cheap
invariants: empty cohort, and the same-length cohort requirement.
"""

from __future__ import annotations

import pytest

from inference_engine.session.batch_scheduler import BatchedDecodeScheduler


class _FakeLayer:
    def __init__(self, t):
        import torch
        self.keys = torch.zeros(1, 2, t, 4)
        self.values = torch.zeros(1, 2, t, 4)


class _FakePast:
    def __init__(self, t):
        self.layers = [_FakeLayer(t), _FakeLayer(t)]


class _FakeAdapter:
    def __init__(self, t):
        self._past = _FakePast(t)
        self._past_len = t
        import torch
        self.next_token_logits = torch.zeros(10)


def test_empty_cohort_is_noop():
    sched = BatchedDecodeScheduler(verifier_model=None, device="cpu")
    out = sched.run_cohort([], max_tokens=4)
    assert out["tokens"] == [] and out["decode_tokens_per_s"] == 0.0


def test_cohort_must_share_one_cache_length():
    sched = BatchedDecodeScheduler(verifier_model=None, device="cpu")
    with pytest.raises(ValueError, match="one cache length"):
        sched._stack_caches([_FakeAdapter(5), _FakeAdapter(7)])


def test_stack_caches_batches_equal_length():
    sched = BatchedDecodeScheduler(verifier_model=None, device="cpu")
    batched = sched._stack_caches([_FakeAdapter(5), _FakeAdapter(5), _FakeAdapter(5)])
    # 2 layers, each stacked to batch dim 3
    assert len(batched.layers) == 2
    assert batched.layers[0].keys.shape[0] == 3
    assert batched.layers[0].keys.shape[2] == 5


def test_prefill_required():
    sched = BatchedDecodeScheduler(verifier_model=None, device="cpu")
    a = _FakeAdapter(5)
    a._past = None
    with pytest.raises(ValueError, match="prefilled"):
        sched._stack_caches([a])
