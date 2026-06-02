"""Shared fixtures for the verifier-independent scheduler tests.

PR-N2 retired the ``DeterministicEngine`` + ``DeterministicTokenizer``
test doubles that previously lived here. The scheduler's runtime
behavior — admission control, lifecycle, cancellation, concurrency,
shutdown — moved to ``tests/integration/test_scheduler_real.py``
where it runs against a real ``SpeculativeEngine`` over Qwen3-0.6B.

What stays on Linux: the slab-pool fixtures (verifier-independent;
they describe storage shape, not model behavior). They're consumed by
``test_scheduler_validation.py`` (argument validation paths that
reject before the engine is touched).

The previously co-located ``test_pooled_verifier.py`` is intentionally
left in place with its own ``_FakeVerifier`` because PR-D2 retires
the ``PooledVerifier`` module entirely (HTTP shim refactor onto
``SessionStore``); cleaning up the test file before the module
disappears would be throwaway work.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig


@pytest.fixture
def slab_config() -> SlabConfig:
    return SlabConfig(
        num_layers=2, num_heads=2, sink_size=1,
        window_size=2, head_dim=4, dtype=torch.float32,
    )


@pytest.fixture
def small_pool(slab_config: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=3, slab_config=slab_config)


@pytest.fixture
def single_pool(slab_config: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=1, slab_config=slab_config)
