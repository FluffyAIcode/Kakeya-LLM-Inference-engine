import pytest

from scripts.start_prefill_worker_node import adaptive_cache_budget


def test_adaptive_budget_uses_only_model_headroom():
    gib = 1 << 30
    assert adaptive_cache_budget(
        total_bytes=16 * gib,
        active_model_bytes=10 * gib,
        ceiling_bytes=8 * gib,
        minimum_bytes=1 * gib,
        reserve_bytes=2 * gib,
    ) == 4 * gib
    assert adaptive_cache_budget(
        total_bytes=16 * gib,
        active_model_bytes=15 * gib,
        ceiling_bytes=8 * gib,
        minimum_bytes=1 * gib,
        reserve_bytes=2 * gib,
    ) == 1 * gib


def test_adaptive_budget_caps_spare_memory_and_validates():
    gib = 1 << 30
    assert adaptive_cache_budget(
        total_bytes=32 * gib,
        active_model_bytes=1 * gib,
        ceiling_bytes=8 * gib,
        minimum_bytes=1 * gib,
        reserve_bytes=2 * gib,
    ) == 8 * gib
    with pytest.raises(ValueError):
        adaptive_cache_budget(
            total_bytes=0,
            active_model_bytes=0,
            ceiling_bytes=1,
            minimum_bytes=1,
            reserve_bytes=0,
        )
