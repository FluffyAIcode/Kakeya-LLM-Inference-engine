"""Unit tests for :class:`PooledVerifier`.

Uses a real concrete ``_FakeVerifier`` class — not ``unittest.mock`` —
that mimics the verifier protocol with deterministic, in-memory state.
This lets us verify the wrapper's lifecycle without loading real
Qwen3 weights (which would make CI slow and HF-cache-bound).
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.pooled_verifier import PooledVerifier


class _FakeStats:
    def __init__(self) -> None:
        self.peak_kv_bytes = 0


class _FakeConfig:
    def __init__(self) -> None:
        self.sink_size = 1
        self.window_size = 4
        self.dtype = torch.float32
        self.device = "cpu"


class _FakeVerifier:
    """Deterministic verifier-protocol implementation for tests."""

    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.stats = _FakeStats()
        self.tokenizer = "tokenizer-marker"
        self.next_token_logits: torch.Tensor | None = None
        self.cache_logical_size = 0
        self.next_global_position = 0
        # Recording for assertions.
        self.calls: list[tuple] = []

    def prefill(self, prompt_ids):
        self.calls.append(("prefill", tuple(prompt_ids)))
        self.cache_logical_size = len(prompt_ids)
        self.next_global_position = len(prompt_ids)
        self.next_token_logits = torch.zeros(8)
        self.stats.peak_kv_bytes = len(prompt_ids) * 100  # fake "100 bytes/token"

    def forward_block(self, tokens):
        self.calls.append(("forward_block", tuple(tokens)))
        self.cache_logical_size += len(tokens)
        self.stats.peak_kv_bytes = self.cache_logical_size * 100
        return torch.zeros(len(tokens), 8)

    def commit_or_truncate(self, *, forwarded, accepted):
        self.calls.append(("commit_or_truncate", forwarded, accepted))
        self.cache_logical_size -= (forwarded - accepted)
        self.next_global_position += accepted
        self.stats.peak_kv_bytes = self.cache_logical_size * 100

    def append_token(self, token_id):
        self.calls.append(("append_token", token_id))
        self.cache_logical_size += 1
        self.next_global_position += 1
        self.stats.peak_kv_bytes = self.cache_logical_size * 100
        out = torch.zeros(8)
        self.next_token_logits = out
        return out

    def reset(self):
        self.calls.append(("reset",))
        self.cache_logical_size = 0
        self.next_global_position = 0
        self.next_token_logits = None


@pytest.fixture
def slab_config() -> SlabConfig:
    return SlabConfig(
        num_layers=1, num_heads=1, sink_size=0, window_size=1,
        head_dim=1, dtype=torch.float32,
    )


@pytest.fixture
def pool(slab_config: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=2, slab_config=slab_config)


@pytest.fixture
def fake_verifier() -> _FakeVerifier:
    return _FakeVerifier()


@pytest.fixture
def pooled(fake_verifier: _FakeVerifier, pool: SlabPool) -> PooledVerifier:
    return PooledVerifier(verifier=fake_verifier, pool=pool)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_none_pool(fake_verifier):
    with pytest.raises(ValueError, match="pool must not be None"):
        PooledVerifier(verifier=fake_verifier, pool=None)  # type: ignore[arg-type]


def test_construction_no_slab_held_initially(pooled, pool):
    assert pooled.slab is None
    assert pool.in_use_count == 0


# ---------------------------------------------------------------------------
# prefill: acquires slab + syncs bytes
# ---------------------------------------------------------------------------


def test_prefill_acquires_slab(pooled, pool, fake_verifier):
    pooled.prefill([1, 2, 3])
    assert pooled.slab is not None
    assert pool.in_use_count == 1
    assert ("prefill", (1, 2, 3)) in fake_verifier.calls


def test_prefill_writes_real_bytes_to_slab_override(pooled):
    pooled.prefill([1, 2, 3])
    # _FakeVerifier reports 3 * 100 = 300 bytes after prefill.
    assert pooled.slab.live_kv_bytes_override == 300
    # And live_kv_bytes returns the override.
    assert pooled.slab.live_kv_bytes == 300


def test_repeated_prefill_releases_old_slab(pooled, pool):
    pooled.prefill([1])
    first_slab = pooled.slab
    pooled.prefill([2])
    second_slab = pooled.slab
    # Different slab instance (or same after release+re-acquire); the
    # invariant is that pool only holds one active slab for this
    # verifier.
    assert pool.in_use_count == 1
    # Old slab was released.
    assert first_slab is not None
    if first_slab is not second_slab:
        # The pool's free list contains first_slab now (it was released).
        # We don't probe pool internals; just verify in_use_count.
        pass


def test_prefill_failure_releases_slab(pool):
    """If the wrapped verifier raises during prefill, the wrapper
    must release the slab so the pool isn't leaked."""

    class _RaisingVerifier(_FakeVerifier):
        def prefill(self, prompt_ids):
            raise RuntimeError("synthetic prefill failure")

    pooled = PooledVerifier(verifier=_RaisingVerifier(), pool=pool)
    with pytest.raises(RuntimeError, match="synthetic prefill failure"):
        pooled.prefill([1, 2, 3])
    assert pool.in_use_count == 0
    assert pooled.slab is None


# ---------------------------------------------------------------------------
# forward_block / commit_or_truncate / append_token sync slab bytes
# ---------------------------------------------------------------------------


def test_forward_block_updates_slab_bytes(pooled):
    pooled.prefill([1, 2, 3])
    out = pooled.forward_block([4, 5])
    assert out.shape == (2, 8)
    # cache_logical_size is now 5; 5*100 = 500.
    assert pooled.slab.live_kv_bytes_override == 500


def test_commit_or_truncate_updates_slab_bytes(pooled):
    pooled.prefill([1, 2, 3])
    pooled.forward_block([4, 5])
    pooled.commit_or_truncate(forwarded=2, accepted=1)
    # cache shrinks by 1 (drop=2-1=1) -> logical_size=4 -> 400 bytes.
    assert pooled.slab.live_kv_bytes_override == 400


def test_append_token_updates_slab_bytes(pooled):
    pooled.prefill([1, 2, 3])
    pooled.append_token(99)
    # logical_size went 3 -> 4; 400 bytes.
    assert pooled.slab.live_kv_bytes_override == 400


def test_methods_passthrough_to_underlying_verifier(pooled, fake_verifier):
    pooled.prefill([7, 8])
    pooled.forward_block([9])
    pooled.commit_or_truncate(forwarded=1, accepted=1)
    pooled.append_token(42)
    # Inspect the recorded call list on the underlying verifier.
    assert fake_verifier.calls == [
        ("prefill", (7, 8)),
        ("forward_block", (9,)),
        ("commit_or_truncate", 1, 1),
        ("append_token", 42),
    ]


# ---------------------------------------------------------------------------
# reset releases slab
# ---------------------------------------------------------------------------


def test_reset_releases_slab(pooled, pool):
    pooled.prefill([1, 2])
    assert pool.in_use_count == 1
    pooled.reset()
    assert pool.in_use_count == 0
    assert pooled.slab is None


def test_reset_when_no_slab_held_is_noop(pooled, fake_verifier):
    pooled.reset()
    assert pooled.slab is None
    assert ("reset",) in fake_verifier.calls


# ---------------------------------------------------------------------------
# Pass-through properties
# ---------------------------------------------------------------------------


def test_tokenizer_passthrough(pooled, fake_verifier):
    assert pooled.tokenizer == fake_verifier.tokenizer


def test_stats_passthrough(pooled, fake_verifier):
    pooled.prefill([1])
    assert pooled.stats is fake_verifier.stats


def test_config_passthrough(pooled, fake_verifier):
    assert pooled.config is fake_verifier.config


def test_cache_logical_size_passthrough(pooled):
    pooled.prefill([1, 2, 3])
    assert pooled.cache_logical_size == 3


def test_next_global_position_passthrough(pooled):
    pooled.prefill([1, 2])
    assert pooled.next_global_position == 2


def test_next_token_logits_get_and_set(pooled):
    pooled.prefill([1])
    assert pooled.next_token_logits is not None
    new_logits = torch.ones(8)
    pooled.next_token_logits = new_logits
    assert torch.equal(pooled.next_token_logits, new_logits)


def test_inner_returns_wrapped_verifier(pooled, fake_verifier):
    assert pooled.inner is fake_verifier


def test_pool_property_returns_pool(pooled, pool):
    assert pooled.pool is pool
