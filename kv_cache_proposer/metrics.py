"""KV cache and NBT (Net Bytes per Token) accounting.

Memory taxonomy (production GPU inference, single GPU):

  Persistent (lives across forwards):
    * model weights         - constant per GPU, shared across all sessions
    * KV cache (verifier)   - grows with each session's emitted tokens
    * KV cache (proposer)   - same, if the proposer maintains one
                              (this implementation recomputes the proposer
                              per block, so its persistent KV is zero)

  Transient (lives only during one model(...) call):
    * activations           - allocated when the forward starts, released
                              when the forward returns; never accumulates
                              across forwards.

Per the discussion in the project notes, only persistent memory should be
amortized per token. Activation peak is reported separately as a *GPU
capacity constraint* (the forward must fit in HBM), not a per-token cost.

Definitions used by this module:

    NBT_kv_only =  verifier_KV_per_token
                 + proposer_KV_per_token
                 + proposer_weight_bytes / (B * S)

    peak_activation_per_gpu = max activation bytes observed during any
                              single forward call (proposer + verifier
                              forwards independently; we report the max
                              of both).

Compression ratio is reported against the verifier's full DynamicCache
KV-per-token baseline. Activation is *not* in numerator or denominator of
the ratio; it is a separate fitness gate.
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

    # Verifier KV (persistent)
    verifier_baseline_kv_bytes_total: int
    verifier_baseline_kv_bytes_per_token: float
    verifier_residual_kv_bytes_total: int
    verifier_residual_kv_bytes_per_token: float
    verifier_kv_bytes_per_slot: float
    cache_budget_slots: int

    # Proposer KV (persistent) — zero in this implementation
    proposer_kv_bytes_total: int
    proposer_kv_bytes_per_token: float

    # Proposer weights (persistent, GPU-level constant; per-token amortized)
    proposer_weight_bytes_total: int
    proposer_weight_bytes_per_token: float  # = weight / (B * S)

    # Transient capacity constraint (NOT in NBT)
    proposer_peak_activation_bytes_total: int
    verifier_peak_activation_bytes_total: int
    peak_activation_bytes_per_gpu: int  # max(proposer, verifier)

    # Aggregated NBT (kv_only) and compression
    nbt_kv_only_bytes_per_token: float
    baseline_bytes_per_token: float
    compression_ratio: float

    # Output equivalence
    speculative_output_tokens: int
    baseline_output_tokens: int
    output_match_prefix_length: int
    output_exact_match: bool
    acceptance_rate: float

    projection_points: List[Tuple[int, int, float, float]] = field(default_factory=list)
    """Each entry: (B, S, projected_nbt_kv_only_bytes_per_token, compression_ratio)."""

    @classmethod
    def compute(
        cls,
        speculative: SpeculativeRunResult,
        baseline: BaselineRunResult,
        sink_size: int,
        window_size: int,
        block_size: int,
        batch_size: int,
        verifier_peak_activation_bytes: int = 0,
    ) -> "NBTReport":
        seq_len = max(baseline.final_kv_token_count, 1)

        # ---- Verifier KV ----
        baseline_total = baseline.peak_kv_bytes
        baseline_per_token = baseline_total / seq_len

        residual_total = speculative.verifier_peak_kv_bytes
        residual_per_token = residual_total / seq_len

        cache_budget_slots = sink_size + window_size
        slots_used_speculative = max(speculative.verifier_final_kv_token_count, 1)
        kv_per_slot = residual_total / slots_used_speculative

        # ---- Proposer KV (zero by design in this build) ----
        proposer_kv_total = 0
        proposer_kv_per_token = 0.0

        # ---- Proposer weights ----
        weight_per_token = (
            speculative.proposer_weight_bytes / max(batch_size * seq_len, 1)
        )

        # ---- NBT_kv_only ----
        nbt_kv_only = (
            residual_per_token + proposer_kv_per_token + weight_per_token
        )

        # ---- Activation peaks (capacity, NOT in NBT) ----
        peak_act_gpu = max(
            speculative.proposer_peak_activation_bytes,
            verifier_peak_activation_bytes,
        )

        # ---- Output equivalence ----
        a = speculative.output_token_ids
        b = baseline.output_token_ids
        prefix_match = 0
        for x, y in zip(a, b):
            if x == y:
                prefix_match += 1
            else:
                break
        exact = (a == b)

        # ---- Projection table (kv_only NBT, holding kv_per_slot constant) ----
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
            nbt = v_kv_per_token + 0.0 + w_per_token  # proposer KV = 0 here
            base_per_token = kv_per_slot
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
            verifier_kv_bytes_per_slot=kv_per_slot,
            cache_budget_slots=cache_budget_slots,
            proposer_kv_bytes_total=proposer_kv_total,
            proposer_kv_bytes_per_token=proposer_kv_per_token,
            proposer_weight_bytes_total=speculative.proposer_weight_bytes,
            proposer_weight_bytes_per_token=weight_per_token,
            proposer_peak_activation_bytes_total=speculative.proposer_peak_activation_bytes,
            verifier_peak_activation_bytes_total=verifier_peak_activation_bytes,
            peak_activation_bytes_per_gpu=peak_act_gpu,
            nbt_kv_only_bytes_per_token=nbt_kv_only,
            baseline_bytes_per_token=baseline_per_token,
            compression_ratio=baseline_per_token / max(nbt_kv_only, 1e-9),
            speculative_output_tokens=len(a),
            baseline_output_tokens=len(b),
            output_match_prefix_length=prefix_match,
            output_exact_match=exact,
            acceptance_rate=speculative.acceptance_rate,
            projection_points=projections,
        )

    def render(self) -> str:
        mb = lambda b: f"{b / (1024 * 1024):8.2f} MB"
        rows = [
            "=" * 76,
            "NBT Report — KV-only definition (activation reported separately)",
            "=" * 76,
            f"  scenario: B={self.batch_size}, S={self.sequence_length_tokens} tokens, "
            f"L_block={self.block_size}, sink={self.sink_size}, window={self.window_size}",
            "",
            "Persistent memory (per token, amortized):",
            "  Verifier KV cache:",
            f"    baseline (full DynamicCache)   total={mb(self.verifier_baseline_kv_bytes_total)}  "
            f"per-token={self.verifier_baseline_kv_bytes_per_token:9.1f} B",
            f"    sink+window residual (peak)    total={mb(self.verifier_residual_kv_bytes_total)}  "
            f"per-token={self.verifier_residual_kv_bytes_per_token:9.1f} B",
            f"    per-slot KV constant           {self.verifier_kv_bytes_per_slot:>9.0f} B/slot   "
            f"(cache_budget = {self.cache_budget_slots} slots)",
            "",
            "  Proposer KV cache:",
            f"    {self.proposer_kv_bytes_total} B total  -> {self.proposer_kv_bytes_per_token:.1f} B/token  "
            "(this build recomputes the proposer per block, so persistent proposer KV = 0)",
            "",
            "  Proposer weights (model parameters, GPU-shared, amortized over B*S):",
            f"    total={mb(self.proposer_weight_bytes_total)}   "
            f"-> {self.proposer_weight_bytes_per_token:9.1f} B/token  "
            f"(weights / (B={self.batch_size} * S={self.sequence_length_tokens}))",
            "",
            f"  NBT_kv_only = {self.nbt_kv_only_bytes_per_token:9.1f} B/token   "
            f"(verifier_KV + proposer_KV + weights/(B*S))",
            f"  Baseline KV = {self.baseline_bytes_per_token:9.1f} B/token   "
            f"(full DynamicCache)",
            f"  Compression vs baseline: {self.compression_ratio:7.2f}x",
            "",
            "Transient memory (capacity constraint, NOT in NBT):",
            f"  Proposer peak activation (single forward): {mb(self.proposer_peak_activation_bytes_total)}",
            f"  Verifier peak activation (single forward): {mb(self.verifier_peak_activation_bytes_total)}",
            f"  Peak activation per GPU = max(both)      : {mb(self.peak_activation_bytes_per_gpu)}",
            "  (this is the maximum HBM occupancy seen during any single model(...) call;",
            "   it is freed when that call returns and does not accumulate across forwards.)",
            "",
            "Output equivalence (greedy):",
            f"  speculative tokens generated:  {self.speculative_output_tokens}",
            f"  baseline    tokens generated:  {self.baseline_output_tokens}",
            f"  shared greedy prefix length:   {self.output_match_prefix_length}",
            f"  exact match:                   {self.output_exact_match}",
            f"  proposer acceptance rate:      {self.acceptance_rate:6.3f}",
            "",
            "Projected NBT_kv_only at canonical operating points:",
            f"  (per-slot verifier KV measured = {self.verifier_kv_bytes_per_slot:8.0f} B; "
            f"cache_budget = {self.cache_budget_slots} slots; proposer KV = 0)",
            "  " + "-" * 64,
            f"  {'B':>4}  {'S':>10}  {'NBT B/token':>14}  {'compression':>12}",
            "  " + "-" * 64,
        ]
        for (B, S, nbt, ratio) in self.projection_points:
            rows.append(
                f"  {B:>4}  {S:>10,}  {nbt:>14,.1f}  {ratio:>11.2f}x"
            )
        rows.append("  " + "-" * 64)
        rows.append("=" * 76)
        return "\n".join(rows)
