"""Representation-alignment training (ADR 0001 §4).

This package implements EAGLE-3-style alignment between the dllm-hub
MDLM proposer and a Qwen3 AR verifier. The recipe has four stages:

    Stage 1 (this commit) — proposer_surgery.ReprAlignedSurgery
        Frozen verifier embed/lm_head + learnable bridge projections.

    Stage 2 (planned)     — data_collection.OnPolicyHiddenStateCache
        Verifier-generated trajectories with per-token hidden states.

    Stage 3 (planned)     — trainer.ReprAlignTrainer
        Smooth-L1 hidden-state alignment + temperature-scaled KL +
        mask-recovery auxiliary loss; LoRA on proposer backbone.

    Stage 4 (planned)     — eval.AcceptanceEvaluator
        Held-out alpha measurement at K in {1, 2, 4}; gates the v1 ship.

See docs/adr/0001-proposer-sizing-and-alignment.md for the full
decision context.
"""

from .proposer_surgery import ReprAlignedSurgery, SurgeryConfig

__all__ = ["ReprAlignedSurgery", "SurgeryConfig"]
