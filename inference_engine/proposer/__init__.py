"""Proposer-side optimizations.

Currently exports:
  * SparseLogitsProposer — drop-in replacement for the dense
    `kv_cache_proposer.proposer.DLMProposer` that computes lm_head
    logits only at masked positions, dramatically reducing per-step
    activation memory and per-step compute.
"""

from .sparse_logits import SparseLogitsProposer

__all__ = ["SparseLogitsProposer"]
