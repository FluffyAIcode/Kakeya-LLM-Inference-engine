"""Linux CI unit tests for inference_engine/v04/kv_merge.py.

These tests exercise the K/V merge primitive that combines verifier's
locally-computed K/V with captured proposer K/V at evicted positions.
The merge function is pure tensor manipulation with no HF / model
dependency, so all tests run in <0.1 s on Linux CI without model
downloads.

Test classes:

* TestComputeEvictedPositions — sink+window range arithmetic.
* TestMergeKVHappyPath — basic correctness on small fixtures.
* TestMergeKVPositionValidation — sortedness / dedup / range raises.
* TestMergeKVShapeValidation — batch / heads / dim / dtype / device
  consistency raises.
* TestMergeKVDifferentiability — gradient flow through captured
  branch and severance at evicted positions on local branch.
* TestMergeKVEdgeCases — empty evicted list, all-evicted, single-
  position, boundary positions.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.v04.kv_merge import (
    compute_evicted_positions,
    merge_kv_at_evicted_positions,
)


# ---------------------------------------------------------------------------
# compute_evicted_positions
# ---------------------------------------------------------------------------


class TestComputeEvictedPositions:
    def test_typical_case(self):
        # seq_len=20, sink=4, window=8 → kept = {0..3} ∪ {12..19}
        # evicted = {4, 5, 6, 7, 8, 9, 10, 11}
        evicted = compute_evicted_positions(seq_len=20, sink_size=4, window_size=8)
        assert evicted == [4, 5, 6, 7, 8, 9, 10, 11]

    def test_no_eviction_when_sink_plus_window_covers_seq(self):
        evicted = compute_evicted_positions(seq_len=10, sink_size=4, window_size=6)
        assert evicted == []

    def test_no_eviction_when_sink_plus_window_exceeds_seq(self):
        evicted = compute_evicted_positions(seq_len=5, sink_size=4, window_size=6)
        assert evicted == []

    def test_zero_sink(self):
        # All trailing tokens are window; everything before is evicted
        evicted = compute_evicted_positions(seq_len=10, sink_size=0, window_size=4)
        assert evicted == [0, 1, 2, 3, 4, 5]

    def test_zero_window(self):
        # All initial tokens are sink; everything after is evicted
        evicted = compute_evicted_positions(seq_len=10, sink_size=3, window_size=0)
        assert evicted == [3, 4, 5, 6, 7, 8, 9]

    def test_zero_sink_zero_window_evicts_everything(self):
        evicted = compute_evicted_positions(seq_len=5, sink_size=0, window_size=0)
        assert evicted == [0, 1, 2, 3, 4]

    def test_seq_len_zero_returns_empty(self):
        evicted = compute_evicted_positions(seq_len=0, sink_size=4, window_size=8)
        assert evicted == []

    def test_negative_seq_len_raises(self):
        with pytest.raises(ValueError, match="must all be non-negative"):
            compute_evicted_positions(seq_len=-1, sink_size=4, window_size=8)

    def test_negative_sink_raises(self):
        with pytest.raises(ValueError, match="must all be non-negative"):
            compute_evicted_positions(seq_len=10, sink_size=-1, window_size=8)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="must all be non-negative"):
            compute_evicted_positions(seq_len=10, sink_size=4, window_size=-1)


# ---------------------------------------------------------------------------
# merge_kv_at_evicted_positions — happy path
# ---------------------------------------------------------------------------


class TestMergeKVHappyPath:
    def _make_inputs(
        self, B=1, T=10, H=2, D=4, n_evicted=3, dtype=torch.float32, seed=0,
    ):
        torch.manual_seed(seed)
        K_local = torch.randn(B, T, H, D, dtype=dtype)
        V_local = torch.randn(B, T, H, D, dtype=dtype)
        K_captured = torch.randn(B, n_evicted, H, D, dtype=dtype)
        V_captured = torch.randn(B, n_evicted, H, D, dtype=dtype)
        return K_local, V_local, K_captured, V_captured

    def test_evicted_positions_use_captured_values(self):
        K_local, V_local, K_captured, V_captured = self._make_inputs(T=10, n_evicted=3)
        positions = [4, 5, 6]
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_captured, V_captured, positions,
        )
        # At evicted positions, merged equals captured
        for slot, p in enumerate(positions):
            assert torch.equal(K_merged[:, p], K_captured[:, slot])
            assert torch.equal(V_merged[:, p], V_captured[:, slot])

    def test_non_evicted_positions_preserve_local_values(self):
        K_local, V_local, K_captured, V_captured = self._make_inputs(T=10, n_evicted=3)
        positions = [4, 5, 6]
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_captured, V_captured, positions,
        )
        kept_positions = [p for p in range(10) if p not in positions]
        for p in kept_positions:
            assert torch.equal(K_merged[:, p], K_local[:, p])
            assert torch.equal(V_merged[:, p], V_local[:, p])

    def test_output_shape_matches_local(self):
        K_local, V_local, K_captured, V_captured = self._make_inputs(
            B=2, T=15, H=3, D=8, n_evicted=4,
        )
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_captured, V_captured,
            [3, 5, 7, 11],
        )
        assert K_merged.shape == K_local.shape
        assert V_merged.shape == V_local.shape

    def test_output_is_clone_not_view(self):
        """Mutating output must not affect inputs."""
        K_local, V_local, K_captured, V_captured = self._make_inputs(T=8, n_evicted=2)
        K_local_snapshot = K_local.clone()
        V_local_snapshot = V_local.clone()

        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_captured, V_captured, [2, 5],
        )
        K_merged.fill_(999.0)
        V_merged.fill_(-999.0)
        assert torch.equal(K_local, K_local_snapshot)
        assert torch.equal(V_local, V_local_snapshot)


# ---------------------------------------------------------------------------
# Position validation
# ---------------------------------------------------------------------------


class TestMergeKVPositionValidation:
    def _inputs(self, T=10, n_evicted=3):
        K_local = torch.randn(1, T, 2, 4)
        V_local = torch.randn(1, T, 2, 4)
        K_cap = torch.randn(1, n_evicted, 2, 4)
        V_cap = torch.randn(1, n_evicted, 2, 4)
        return K_local, V_local, K_cap, V_cap

    def test_unsorted_positions_raises(self):
        K_local, V_local, K_cap, V_cap = self._inputs(n_evicted=3)
        with pytest.raises(ValueError, match="sorted ascending"):
            merge_kv_at_evicted_positions(
                K_local, V_local, K_cap, V_cap, [5, 2, 7],
            )

    def test_duplicate_positions_raises(self):
        K_local, V_local, K_cap, V_cap = self._inputs(n_evicted=3)
        with pytest.raises(ValueError, match="sorted ascending"):
            merge_kv_at_evicted_positions(
                K_local, V_local, K_cap, V_cap, [2, 2, 5],
            )

    def test_negative_position_raises(self):
        K_local, V_local, K_cap, V_cap = self._inputs(n_evicted=2)
        with pytest.raises(ValueError, match="must lie in"):
            merge_kv_at_evicted_positions(
                K_local, V_local, K_cap, V_cap, [-1, 5],
            )

    def test_position_at_seqlen_raises(self):
        K_local, V_local, K_cap, V_cap = self._inputs(T=10, n_evicted=2)
        with pytest.raises(ValueError, match="must lie in"):
            merge_kv_at_evicted_positions(
                K_local, V_local, K_cap, V_cap, [5, 10],
            )

    def test_position_beyond_seqlen_raises(self):
        K_local, V_local, K_cap, V_cap = self._inputs(T=10, n_evicted=2)
        with pytest.raises(ValueError, match="must lie in"):
            merge_kv_at_evicted_positions(
                K_local, V_local, K_cap, V_cap, [5, 100],
            )


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestMergeKVShapeValidation:
    def test_local_kv_shape_mismatch_raises(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 3, 4)  # different num_heads
        K_cap = torch.randn(1, 2, 2, 4)
        V_cap = torch.randn(1, 2, 2, 4)
        with pytest.raises(ValueError, match="K_local shape"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_local_3d_rejected(self):
        K_local = torch.randn(10, 2, 4)  # rank 3
        V_local = torch.randn(10, 2, 4)
        K_cap = torch.randn(1, 2, 2, 4)
        V_cap = torch.randn(1, 2, 2, 4)
        with pytest.raises(ValueError, match="K_local must be 4-D"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_captured_3d_rejected(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(2, 2, 4)  # rank 3
        V_cap = torch.randn(2, 2, 4)
        with pytest.raises(ValueError, match="K_captured must be 4-D"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_batch_mismatch_raises(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(2, 2, 2, 4)  # batch=2, mismatch
        V_cap = torch.randn(2, 2, 2, 4)
        with pytest.raises(ValueError, match="batch mismatch"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_num_kv_heads_mismatch_raises(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 2, 4, 4)  # H=4, mismatch
        V_cap = torch.randn(1, 2, 4, 4)
        with pytest.raises(ValueError, match="num_kv_heads mismatch"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_head_dim_mismatch_raises(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 2, 2, 8)  # D=8, mismatch
        V_cap = torch.randn(1, 2, 2, 8)
        with pytest.raises(ValueError, match="head_dim mismatch"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_captured_T_dim_mismatch_raises(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 5, 2, 4)  # T=5 but only 2 positions listed
        V_cap = torch.randn(1, 5, 2, 4)
        with pytest.raises(ValueError, match="T-dim 5 != len"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_dtype_mismatch_raises(self):
        K_local = torch.randn(1, 10, 2, 4, dtype=torch.float32)
        V_local = torch.randn(1, 10, 2, 4, dtype=torch.float32)
        K_cap = torch.randn(1, 2, 2, 4, dtype=torch.float64)
        V_cap = torch.randn(1, 2, 2, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="dtype mismatch"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])

    def test_captured_KV_shape_mismatch_internal_raises(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 2, 2, 4)
        V_cap = torch.randn(1, 2, 3, 4)  # different from K_cap
        with pytest.raises(ValueError, match="K_captured shape"):
            merge_kv_at_evicted_positions(K_local, V_local, K_cap, V_cap, [3, 7])


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestMergeKVDifferentiability:
    def test_gradient_flows_through_captured(self):
        """A trainable cross-model projection in K2/K3 must receive
        gradient through the merge — the captured branch needs to
        carry gradient information."""
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 2, 2, 4, requires_grad=True)
        V_cap = torch.randn(1, 2, 2, 4, requires_grad=True)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [3, 7],
        )
        loss = K_merged.sum() + V_merged.sum()
        loss.backward()
        assert K_cap.grad is not None
        assert V_cap.grad is not None
        # All evicted positions contributed to loss → grad is non-zero
        assert (K_cap.grad != 0).any()
        assert (V_cap.grad != 0).any()

    def test_gradient_flows_through_local_at_kept_positions(self):
        """Local K/V at non-evicted positions must still carry gradient
        — only the evicted slots are overridden."""
        K_local = torch.randn(1, 10, 2, 4, requires_grad=True)
        V_local = torch.randn(1, 10, 2, 4, requires_grad=True)
        K_cap = torch.randn(1, 2, 2, 4)
        V_cap = torch.randn(1, 2, 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [3, 7],
        )
        loss = K_merged.sum() + V_merged.sum()
        loss.backward()
        assert K_local.grad is not None
        assert V_local.grad is not None
        # Kept positions have gradient (they appear in the loss)
        kept = [p for p in range(10) if p not in (3, 7)]
        for p in kept:
            assert (K_local.grad[:, p] != 0).any(), f"position {p} has zero K grad"
            assert (V_local.grad[:, p] != 0).any(), f"position {p} has zero V grad"

    def test_gradient_severed_at_evicted_positions_on_local(self):
        """At evicted positions, the local K/V are overridden, so no
        gradient should flow back to local K_local at those positions."""
        K_local = torch.randn(1, 10, 2, 4, requires_grad=True)
        V_local = torch.randn(1, 10, 2, 4, requires_grad=True)
        K_cap = torch.randn(1, 2, 2, 4)
        V_cap = torch.randn(1, 2, 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [3, 7],
        )
        loss = K_merged.sum() + V_merged.sum()
        loss.backward()
        # Evicted positions on local branch have zero gradient
        for p in (3, 7):
            assert torch.all(K_local.grad[:, p] == 0), (
                f"position {p} should have zero K_local grad (overridden by capture)"
            )
            assert torch.all(V_local.grad[:, p] == 0), (
                f"position {p} should have zero V_local grad"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMergeKVEdgeCases:
    def test_empty_evicted_list_returns_clone_of_local(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        # Empty evicted: K_cap / V_cap should be unused but we still
        # need to provide them — pass a 0-length T dim
        K_cap = torch.empty(1, 0, 2, 4)
        V_cap = torch.empty(1, 0, 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [],
        )
        assert torch.equal(K_merged, K_local)
        assert torch.equal(V_merged, V_local)
        # And it's a clone not a view
        K_merged.fill_(999.0)
        assert (K_local != 999.0).any()

    def test_all_positions_evicted(self):
        T = 5
        K_local = torch.randn(1, T, 2, 4)
        V_local = torch.randn(1, T, 2, 4)
        K_cap = torch.randn(1, T, 2, 4)
        V_cap = torch.randn(1, T, 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, list(range(T)),
        )
        # All positions overridden → merged equals captured
        assert torch.equal(K_merged, K_cap)
        assert torch.equal(V_merged, V_cap)

    def test_single_position_evicted(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 1, 2, 4)
        V_cap = torch.randn(1, 1, 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [4],
        )
        assert torch.equal(K_merged[:, 4], K_cap[:, 0])
        assert torch.equal(V_merged[:, 4], V_cap[:, 0])
        # All other positions unchanged
        for p in range(10):
            if p != 4:
                assert torch.equal(K_merged[:, p], K_local[:, p])

    def test_boundary_position_zero(self):
        K_local = torch.randn(1, 10, 2, 4)
        V_local = torch.randn(1, 10, 2, 4)
        K_cap = torch.randn(1, 1, 2, 4)
        V_cap = torch.randn(1, 1, 2, 4)
        K_merged, _ = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [0],
        )
        assert torch.equal(K_merged[:, 0], K_cap[:, 0])

    def test_boundary_position_seqlen_minus_one(self):
        T = 10
        K_local = torch.randn(1, T, 2, 4)
        V_local = torch.randn(1, T, 2, 4)
        K_cap = torch.randn(1, 1, 2, 4)
        V_cap = torch.randn(1, 1, 2, 4)
        K_merged, _ = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [T - 1],
        )
        assert torch.equal(K_merged[:, T - 1], K_cap[:, 0])

    def test_consecutive_positions_block(self):
        """A contiguous block of evicted positions (the common case
        coming from compute_evicted_positions output)."""
        T = 20
        K_local = torch.randn(1, T, 2, 4)
        V_local = torch.randn(1, T, 2, 4)
        positions = list(range(4, 12))  # sink=4, window=8 → middle 4..11 evicted
        K_cap = torch.randn(1, len(positions), 2, 4)
        V_cap = torch.randn(1, len(positions), 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, positions,
        )
        # Verify the contiguous block was correctly overridden
        for slot, p in enumerate(positions):
            assert torch.equal(K_merged[:, p], K_cap[:, slot])
            assert torch.equal(V_merged[:, p], V_cap[:, slot])

    def test_batch_size_greater_than_one(self):
        """Although v0.4 inference is single-batch, the merge function
        should handle B > 1 correctly so the same primitive can be
        reused for batched offline processing."""
        B = 3
        K_local = torch.randn(B, 10, 2, 4)
        V_local = torch.randn(B, 10, 2, 4)
        K_cap = torch.randn(B, 2, 2, 4)
        V_cap = torch.randn(B, 2, 2, 4)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [3, 7],
        )
        for b in range(B):
            assert torch.equal(K_merged[b, 3], K_cap[b, 0])
            assert torch.equal(K_merged[b, 7], K_cap[b, 1])

    def test_dtype_preservation_bf16(self):
        K_local = torch.randn(1, 10, 2, 4, dtype=torch.bfloat16)
        V_local = torch.randn(1, 10, 2, 4, dtype=torch.bfloat16)
        K_cap = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
        V_cap = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
        K_merged, V_merged = merge_kv_at_evicted_positions(
            K_local, V_local, K_cap, V_cap, [3, 7],
        )
        assert K_merged.dtype == torch.bfloat16
        assert V_merged.dtype == torch.bfloat16
