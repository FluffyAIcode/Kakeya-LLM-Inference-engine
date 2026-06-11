"""Unit tests for inference_engine.distributed.mlx_ring (ADR 0009).

Same split as tests/backends/mlx/test_env.py: the Linux gate
exercises the "mlx absent → structured unavailable" branch end-to-end
on its own machine (no mocking of the import system); the
mlx-runtime-present branches carry ``pragma: no cover`` and are
exercised on Apple Silicon hosts.

Coverage target: 100% on ``inference_engine/distributed/mlx_ring.py``.
"""

from __future__ import annotations

import platform

import pytest

from inference_engine.distributed.capability import NodeCapability
from inference_engine.distributed.mlx_ring import (
    RingEnvironment,
    probe_ring_environment,
    ring_path_available,
)


def _card(node_id: str, ring_address: str) -> NodeCapability:
    return NodeCapability(
        node_id=node_id, grpc_address=f"{node_id}:1", ring_address=ring_address,
    )


def test_probe_never_raises_and_is_structured():
    env = probe_ring_environment()
    assert isinstance(env, RingEnvironment)
    if not env.is_available:
        assert env.failure_reason
        assert env.world_size == 0


@pytest.mark.skipif(
    platform.machine() == "arm64",
    reason="Linux-gate branch: asserts the mlx-absent probe result",
)
def test_probe_reports_unavailable_without_mlx():
    env = probe_ring_environment()
    assert not env.is_available
    assert "mlx.core.distributed import failed" in env.failure_reason
    assert env.backend == ""
    assert env.rank == 0


def test_render_unavailable_mentions_reason():
    env = RingEnvironment(
        is_available=False, backend="", rank=0, world_size=0,
        failure_reason="mlx missing",
    )
    assert env.render() == "mlx ring UNAVAILABLE (mlx missing)"


def test_render_available_mentions_rank_and_backend():
    env = RingEnvironment(
        is_available=True, backend="ring", rank=1, world_size=2,
        failure_reason="",
    )
    assert env.render() == "mlx ring OK: backend=ring rank=1/2"


def test_ring_address_empty_when_unavailable():
    env = RingEnvironment(
        is_available=False, backend="", rank=0, world_size=0,
        failure_reason="x",
    )
    assert env.ring_address("mini") == ""


def test_ring_address_is_host_colon_rank_when_available():
    env = RingEnvironment(
        is_available=True, backend="ring", rank=1, world_size=2,
        failure_reason="",
    )
    assert env.ring_address("mini-attic") == "mini-attic:1"


def test_ring_path_requires_both_endpoints():
    assert ring_path_available(_card("a", "a:0"), _card("b", "b:1"))
    assert not ring_path_available(_card("a", "a:0"), _card("b", ""))
    assert not ring_path_available(_card("a", ""), _card("b", "b:1"))
    assert not ring_path_available(_card("a", ""), _card("b", ""))
