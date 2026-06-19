"""Unit tests for inference_engine.distributed.placement (ADR 0009).

Pins the determinism and remote-preference properties the design doc
§3 promises: same snapshot ⇒ same plan, proposer evicted from the
verifier host whenever possible, no-fallback errors otherwise.

Coverage target: 100% on ``inference_engine/distributed/placement.py``.
"""

from __future__ import annotations

import pytest

from inference_engine.distributed.capability import (
    NGRAM_MODEL_ID,
    CapabilityRole,
    ModelCapability,
    NodeCapability,
)
from inference_engine.distributed.placement import (
    PlacementError,
    plan_spec_decode_placement,
)

T0 = 1_000_000.0


def _node(
    node_id: str,
    *,
    models: tuple,
    memory_gb: int = 16,
) -> NodeCapability:
    return NodeCapability(
        node_id=node_id,
        grpc_address=f"{node_id}:50051",
        unified_memory_bytes=memory_gb << 30,
        models=models,
        announced_at_unix=T0,
    )


VERIFIER = ModelCapability("Qwen/Qwen3-0.6B", CapabilityRole.VERIFIER, "bf16")
PROPOSER = ModelCapability(NGRAM_MODEL_ID, CapabilityRole.PROPOSER, "none")


def test_two_node_split_places_roles_on_distinct_hosts():
    a = _node("a", models=(VERIFIER,))
    b = _node("b", models=(PROPOSER,))
    plan = plan_spec_decode_placement([a, b])
    assert plan.verifier_node.node_id == "a"
    assert plan.proposer_node.node_id == "b"
    assert not plan.colocated


def test_proposer_prefers_non_verifier_host():
    # Both nodes can propose; the verifier lands on "a" (higher
    # memory), so the proposer must land on "b" even though "a" also
    # advertises the proposer role.
    a = _node("a", models=(VERIFIER, PROPOSER), memory_gb=24)
    b = _node("b", models=(PROPOSER,), memory_gb=8)
    plan = plan_spec_decode_placement([a, b])
    assert plan.verifier_node.node_id == "a"
    assert plan.proposer_node.node_id == "b"
    assert not plan.colocated


def test_colocation_is_last_resort():
    a = _node("a", models=(VERIFIER, PROPOSER))
    plan = plan_spec_decode_placement([a])
    assert plan.verifier_node.node_id == "a"
    assert plan.proposer_node.node_id == "a"
    assert plan.colocated


def test_prefer_remote_can_be_disabled():
    fast_local = ModelCapability(
        NGRAM_MODEL_ID, CapabilityRole.PROPOSER, tokens_per_second=100.0,
    )
    a = _node("a", models=(VERIFIER, fast_local), memory_gb=24)
    b = _node("b", models=(PROPOSER,), memory_gb=8)
    plan = plan_spec_decode_placement([a, b], prefer_remote_proposer=False)
    assert plan.proposer_node.node_id == "a"
    assert plan.colocated


def test_throughput_dominates_then_memory_then_node_id():
    slow = ModelCapability("v", CapabilityRole.VERIFIER, tokens_per_second=5.0)
    fast = ModelCapability("v", CapabilityRole.VERIFIER, tokens_per_second=9.0)
    p = _node("p", models=(PROPOSER,))

    # Throughput wins over memory.
    a = _node("a", models=(slow,), memory_gb=64)
    b = _node("b", models=(fast,), memory_gb=8)
    assert plan_spec_decode_placement([a, b, p]).verifier_node.node_id == "b"

    # Equal throughput: memory wins.
    c = _node("c", models=(fast,), memory_gb=64)
    assert plan_spec_decode_placement([b, c, p]).verifier_node.node_id == "c"

    # Equal everything: lexicographically smaller node_id, for
    # determinism across nodes computing the same plan.
    d = _node("d", models=(fast,), memory_gb=64)
    assert plan_spec_decode_placement([d, c, p]).verifier_node.node_id == "c"


def test_model_id_pins_are_honored():
    v06 = ModelCapability("Qwen/Qwen3-0.6B", CapabilityRole.VERIFIER)
    v17 = ModelCapability("Qwen/Qwen3-1.7B", CapabilityRole.VERIFIER)
    a = _node("a", models=(v06,))
    b = _node("b", models=(v17,), memory_gb=8)
    c = _node("c", models=(PROPOSER,))
    plan = plan_spec_decode_placement(
        [a, b, c], verifier_model_id="Qwen/Qwen3-1.7B",
        proposer_model_id=NGRAM_MODEL_ID,
    )
    assert plan.verifier_node.node_id == "b"
    assert plan.verifier_model == v17
    assert plan.proposer_model == PROPOSER


def test_no_verifier_candidate_raises():
    with pytest.raises(PlacementError, match="verifier"):
        plan_spec_decode_placement([_node("a", models=(PROPOSER,))])


def test_no_verifier_for_pinned_model_raises_with_model_in_message():
    a = _node("a", models=(VERIFIER, PROPOSER))
    with pytest.raises(PlacementError, match="Qwen/Qwen3-32B"):
        plan_spec_decode_placement([a], verifier_model_id="Qwen/Qwen3-32B")


def test_no_proposer_candidate_raises():
    with pytest.raises(PlacementError, match="proposer"):
        plan_spec_decode_placement([_node("a", models=(VERIFIER,))])


def test_no_proposer_for_pinned_model_raises_with_model_in_message():
    a = _node("a", models=(VERIFIER, PROPOSER))
    with pytest.raises(PlacementError, match="dflash"):
        plan_spec_decode_placement([a], proposer_model_id="dflash")


def test_render_is_stable_and_complete():
    a = _node("a", models=(VERIFIER,))
    b = _node("b", models=(PROPOSER,))
    rendered = plan_spec_decode_placement([a, b]).render()
    assert rendered == (
        "verifier=Qwen/Qwen3-0.6B@a(a:50051) "
        "proposer=ngram@b(b:50051) colocated=False"
    )
