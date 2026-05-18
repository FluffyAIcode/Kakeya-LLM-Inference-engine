"""KV cache and NBT (Net Bytes per Token) accounting.

NBT is the project's main figure of merit:

    NBT = verifier_kv_bytes + proposer_kv_bytes + proposer_weight_bytes/(B*S)
                                                + proposer_activation_bytes/(B*L)

A proper end-to-end NBT requires fixing a (B, S, L) operating point — see
:meth:`NBTReport.compute`. Per-token bytes for the verifier are reported as
``peak_kv_bytes / final_token_count`` so the number reflects the *bound* the
sink+window cache imposed during the run, not just its terminal value.

The :class:`NBTReport` also includes a projection table: given the empirically
measured per-slot KV bytes, peak proposer activation and proposer weight
bytes, what is NBT at canonical operating points (B,S) such as B=64, S=128k or
B=8, S=1M?  These numbers are *not* extrapolated through new model runs —
they reuse the measured constants and only re-amortize them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .speculative import SpeculativeRunResult
from .baseline import BaselineRunResult


def cache_kv_bytes(cache) -> int:
    total = 0
    for layer in cache.layers:
        if layer.keys is not None:
            total += layer.keys.numel() * layer.keys.element_size()
        if layer.values is not None:
            total += layer.values.numel() * layer.values.element_size()
    return total


def cache_token_count(cache) -> int:
    if not cache.layers:
        return 0
    layer0 = cache.layers[0]
    if layer0.keys is None:
        return 0
    return int(layer0.keys.shape[2])


def measure_proposer_weight_bytes(proposer) -> int:
    return sum(p.numel() * p.element_size() for p in proposer.model.parameters())


@dataclass
class NBTReport:
    # Inputs / scenario
    sequence_length_tokens: int
    batch_size: int
    block_size: int
    sink_size: int
    window_size: int

    # Verifier side (per token, amortized over sequence_length_tokens)
    verifier_baseline_kv_bytes_total: int
    verifier_baseline_kv_bytes_per_token: float
    verifier_residual_kv_bytes_total: int
    verifier_residual_kv_bytes_per_token: float

    # Proposer side
    proposer_weight_bytes_total: int
    proposer_weight_bytes_per_token: float  # = weight / (B * S)
    proposer_peak_activation_bytes_total: int
    proposer_activation_bytes_per_token: float  # = peak / (B * block_size)

    # Aggregated NBT
    nbt_bytes_per_token: float
    baseline_bytes_per_token: float
    compression_ratio: float

    # Output equivalence
    speculative_output_tokens: int
    baseline_output_tokens: int
    output_match_prefix_length: int
    output_exact_match: bool
    acceptance_rate: float

    # Per-slot KV bytes for the verifier — used by the projection table.
    verifier_kv_bytes_per_slot: float = 0.0
    cache_budget_slots: int = 0
    projection_points: List[Tuple[int, int, float, float]] = field(default_factory=list)
    """Each entry is (B, S, projected_nbt_bytes_per_token, projected_compression_ratio)."""

    @classmethod
    def compute(
        cls,
        speculative: SpeculativeRunResult,
        baseline: BaselineRunResult,
        sink_size: int,
        window_size: int,
        block_size: int,
        batch_size: int,
    ) -> "NBTReport":
        seq_len = max(baseline.final_kv_token_count, 1)
        baseline_total = baseline.peak_kv_bytes
        baseline_per_token = baseline_total / seq_len

        residual_total = speculative.verifier_peak_kv_bytes
        # The sink+window cache is bounded; the per-token figure is the
        # bound divided by *full* sequence length so it reflects the
        # marginal cost as the sequence grows.
        residual_per_token = residual_total / seq_len

        weight_per_token = (
            speculative.proposer_weight_bytes / (batch_size * seq_len)
        )
        activation_per_token = (
            speculative.proposer_peak_activation_bytes / (batch_size * block_size)
        )

        nbt_per_token = (
            residual_per_token + weight_per_token + activation_per_token
        )
        # Output equivalence diagnostic.
        a = speculative.output_token_ids
        b = baseline.output_token_ids
        prefix_match = 0
        for x, y in zip(a, b):
            if x == y:
                prefix_match += 1
            else:
                break
        exact = (a == b)

        # Projection table -- the per-slot KV byte cost is a model constant
        # (depends only on layers x kv_heads x head_dim x dtype) so we can
        # project NBT to any (B, S) without re-running. The "per-slot" cost
        # was directly measured: total residual / cache slots actually used.
        cache_budget_slots = sink_size + window_size
        slots_used_speculative = max(speculative.verifier_final_kv_token_count, 1)
        kv_per_slot = residual_total / slots_used_speculative
        # Verifier KV at projected S: peak is bounded by min(S, cache_budget) slots
        # (during prefill the cache fills up to S, then trims to budget).
        projections: List[Tuple[int, int, float, float]] = []
        operating_points = [
            (1, 8 * 1024),
            (8, 8 * 1024),
            (8, 32 * 1024),
            (8, 128 * 1024),
            (8, 1024 * 1024),
            (32, 128 * 1024),
            (64, 128 * 1024),
            (64, 1024 * 1024),
        ]
        for B, S in operating_points:
            v_kv_total = kv_per_slot * min(S, cache_budget_slots)
            v_kv_per_token = v_kv_total / S
            w_per_token = speculative.proposer_weight_bytes / (B * S)
            a_per_token = speculative.proposer_peak_activation_bytes / (B * block_size)
            nbt = v_kv_per_token + w_per_token + a_per_token
            base_per_token = kv_per_slot  # full DynamicCache: every token costs one slot
            ratio = base_per_token / max(nbt, 1e-9)
            projections.append((B, S, nbt, ratio))

        return cls(
            sequence_length_tokens=seq_len,
            batch_size=batch_size,
            block_size=block_size,
            sink_size=sink_size,
            window_size=window_size,
            verifier_baseline_kv_bytes_total=baseline_total,
            verifier_baseline_kv_bytes_per_token=baseline_per_token,
            verifier_residual_kv_bytes_total=residual_total,
            verifier_residual_kv_bytes_per_token=residual_per_token,
            proposer_weight_bytes_total=speculative.proposer_weight_bytes,
            proposer_weight_bytes_per_token=weight_per_token,
            proposer_peak_activation_bytes_total=speculative.proposer_peak_activation_bytes,
            proposer_activation_bytes_per_token=activation_per_token,
            nbt_bytes_per_token=nbt_per_token,
            baseline_bytes_per_token=baseline_per_token,
            compression_ratio=baseline_per_token / max(nbt_per_token, 1e-9),
            speculative_output_tokens=len(a),
            baseline_output_tokens=len(b),
            output_match_prefix_length=prefix_match,
            output_exact_match=exact,
            acceptance_rate=speculative.acceptance_rate,
            verifier_kv_bytes_per_slot=kv_per_slot,
            cache_budget_slots=cache_budget_slots,
            projection_points=projections,
        )

    def render(self) -> str:
        kb = lambda b: f"{b / 1024:8.2f} KB"
        mb = lambda b: f"{b / (1024 * 1024):8.2f} MB"
        rows = [
            "=" * 72,
            "NBT Report (Net Bytes per Token, sink+window verifier KV)",
            "=" * 72,
            f"  scenario: B={self.batch_size}, S={self.sequence_length_tokens} tokens, "
            f"L_block={self.block_size}, sink={self.sink_size}, window={self.window_size}",
            "",
            "Verifier KV cache:",
            f"  baseline (full)      total={mb(self.verifier_baseline_kv_bytes_total)}  "
            f"per-token={self.verifier_baseline_kv_bytes_per_token:8.1f} B",
            f"  sink+window residual total={mb(self.verifier_residual_kv_bytes_total)}  "
            f"per-token={self.verifier_residual_kv_bytes_per_token:8.1f} B",
            "",
            "Proposer overhead (per token, amortized):",
            f"  weights  {mb(self.proposer_weight_bytes_total):>14}  -> {self.proposer_weight_bytes_per_token:8.1f} B/token  (weights / (B*S))",
            f"  peak act {mb(self.proposer_peak_activation_bytes_total):>14}  -> {self.proposer_activation_bytes_per_token:8.1f} B/token  (peak / (B*L_block))",
            "",
            f"NBT = {self.nbt_bytes_per_token:8.1f} B/token   (sink+window + proposer overhead)",
            f"Baseline KV = {self.baseline_bytes_per_token:8.1f} B/token   (full DynamicCache)",
            f"Compression vs baseline: {self.compression_ratio:6.2f}x",
            "",
            "Output equivalence (greedy):",
            f"  speculative tokens generated: {self.speculative_output_tokens}",
            f"  baseline    tokens generated: {self.baseline_output_tokens}",
            f"  shared greedy prefix length:  {self.output_match_prefix_length}",
            f"  exact match: {self.output_exact_match}",
            f"  proposer acceptance rate: {self.acceptance_rate:6.3f}",
            "",
            "Projected NBT at canonical operating points:",
            f"  (per-slot verifier KV measured = {self.verifier_kv_bytes_per_slot:8.0f} B; "
            f"cache_budget = {self.cache_budget_slots} slots)",
            "  " + "-" * 64,
            f"  {'B':>4}  {'S':>10}  {'NBT B/token':>14}  {'compression':>12}",
            "  " + "-" * 64,
        ]
        for (B, S, nbt, ratio) in self.projection_points:
            rows.append(
                f"  {B:>4}  {S:>10,}  {nbt:>14,.1f}  {ratio:>11.2f}x"
            )
        rows.append("  " + "-" * 64)
        rows.append("=" * 72)
        return "\n".join(rows)
