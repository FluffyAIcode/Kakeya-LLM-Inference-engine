"""On-policy verifier rollout data collection (ADR 0004 §2.2 + §2.5).

This subpackage is the **contract layer** for representation-
alignment training. It defines the shard schema, builds the multi-
domain prompt pool, and provides the atomic versioned-parquet
writer that downstream rollout workers (PR 2/3 of this work line)
flush captured tokens into.

Layering
--------

    schema.py          — single source of truth for the data
                         contract; defines RolloutMeta, RolloutRow
                         and the pyarrow Schema.
    prompt_pool.py     — multi-domain composition with quotas,
                         length filter, language tagging, dedup.
    parquet_writer.py  — atomic shard writer (data + meta) at the
                         versioned path layout from ADR 0004 §2.5.

This package intentionally does **not** import torch, transformers
or mlx. The verifier rollout worker (a separate module that DOES
load real models) sits on top and feeds rows here.

Stage roadmap (from training/repr_align/__init__.py)
----------------------------------------------------

  Stage 1 — proposer_surgery.ReprAlignedSurgery       (shipped, v0.1.x)
  Stage 2 — data_collection.{schema, prompt_pool,     (THIS PR — core)
            parquet_writer}
            data_collection.rollout_worker            (next PR — does
                                                        real model
                                                        inference)
            data_collection.configs/                  (next PR — 7
                                                        per-domain
                                                        YAML configs)
  Stage 3 — trainer.ReprAlignTrainer                  (planned)
  Stage 4 — eval.AcceptanceEvaluator                  (planned)
"""

from .schema import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TOPK_LOGITS,
    SCHEMA_VERSION,
    RolloutMeta,
    RolloutRow,
    build_pyarrow_schema,
    row_to_pydict,
    system_prompt_hash,
)
from .prompt_pool import (
    CharRatioLanguageDetector,
    Deduper,
    DomainQuota,
    LanguageDetector,
    LengthFilter,
    PoolConfig,
    Prompt,
    PromptPool,
    ShingleJaccardDeduper,
)
from .parquet_writer import (
    RolloutShardWriter,
    list_shards,
    next_shard_id,
    read_meta,
    shard_dir,
)

__all__ = [
    # schema
    "SCHEMA_VERSION",
    "DEFAULT_TOPK_LOGITS",
    "DEFAULT_BLOCK_SIZE",
    "RolloutMeta",
    "RolloutRow",
    "build_pyarrow_schema",
    "row_to_pydict",
    "system_prompt_hash",
    # prompt_pool
    "Prompt",
    "DomainQuota",
    "PoolConfig",
    "PromptPool",
    "LengthFilter",
    "LanguageDetector",
    "CharRatioLanguageDetector",
    "Deduper",
    "ShingleJaccardDeduper",
    # parquet_writer
    "RolloutShardWriter",
    "list_shards",
    "next_shard_id",
    "read_meta",
    "shard_dir",
]
