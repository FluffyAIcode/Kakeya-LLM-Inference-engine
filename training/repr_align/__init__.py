"""Representation-alignment training (ADR 0001 §4).

This package implements EAGLE-3-style alignment between the dllm-hub
MDLM proposer and a Qwen3 AR verifier. The recipe has four stages:

    Stage 1 (shipped, v0.1.x)   — proposer_surgery.ReprAlignedSurgery
        Frozen verifier embed/lm_head + learnable bridge projections.

    Stage 2 (in progress, v0.3) — data_collection.{schema, prompt_pool,
                                                  parquet_writer, ...}
        Verifier-generated trajectories with per-token hidden states.

    Stage 3 (planned)           — trainer.ReprAlignTrainer
        Smooth-L1 hidden-state alignment + temperature-scaled KL +
        mask-recovery auxiliary loss; LoRA on proposer backbone.

    Stage 4 (planned)           — eval.AcceptanceEvaluator
        Held-out alpha measurement at K in {1, 2, 4}; gates the v1 ship.

See docs/adr/0001-proposer-sizing-and-alignment.md and ADR 0004 for
full decision context.

Lazy-import policy
------------------
``proposer_surgery`` pulls in ``torch`` and ``transformers``, which
are heavy. ``data_collection`` is intentionally torch-free so the
data preparation pipeline can run on CPU-only / smaller VMs and so
unit-test collectors are not gated on torch's C-extension import.

Therefore the public symbols ``ReprAlignedSurgery`` and
``SurgeryConfig`` are exposed via :pep:`562` ``__getattr__`` rather
than eager imports — ``training.repr_align.ReprAlignedSurgery``
still works, but ``import training.repr_align.data_collection.schema``
does not pay the torch import cost.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ReprAlignedSurgery", "SurgeryConfig"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import proposer_surgery  # noqa: WPS433 - lazy by design
        return getattr(proposer_surgery, name)
    raise AttributeError(f"module 'training.repr_align' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
