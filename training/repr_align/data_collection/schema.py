"""Data-collection schema (ADR 0004 §2.2 + §2.5).

Single source of truth for the on-policy verifier rollout schema.
Downstream rollout workers, parquet writers and trainers all import
from here. Schema changes bump :data:`SCHEMA_VERSION` and write to a
new versioned path; readers refuse mismatched data.

The schema captures **per generated token**, the data needed to
supervise the proposer's representation alignment loss:

  * last-layer hidden state of the verifier      (bf16, dim=H)
  * top-K logits + probabilities                 (K=20 by default)
  * committed token id                            (the token actually
                                                   emitted at this step)
  * position id                                   (post-sink-window
                                                   trim, global)
  * cache_logical_size                            (the verifier KV
                                                   cache slot count
                                                   at emission time)
  * block index + position-in-block               (deployment-aligned;
                                                   block_size is per-
                                                   shard metadata)

A *shard* is a single Parquet file plus a JSON metadata sidecar.
Every row in a shard shares the same verifier_id / verifier_dtype /
sink+window config, so the trainer can reject mismatched shards
before reading a single row.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import pyarrow as pa


SCHEMA_VERSION: str = "1"

# Reasonable bounds — real config is enforced at validate-time, not
# at type-check time. These are the hard contract values.
DEFAULT_TOPK_LOGITS: int = 20
DEFAULT_BLOCK_SIZE: int = 4

_VERIFIER_ID_RE = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+(?:-[A-Za-z0-9.]+)*$")
_DTYPE_LITERALS = frozenset({"bf16", "fp16", "fp32", "int4", "int8"})


def system_prompt_hash(text: str) -> str:
    """Stable 16-char prefix of the SHA-256 of a system prompt.

    Stored on every row so the trainer can stratify acceptance rate
    by system-prompt rotation. Truncated to 16 hex chars (= 64 bits)
    which is plenty for ADR 0004 §2.2's 5–10-prompt rotation.
    """
    if not isinstance(text, str):
        raise TypeError(f"system prompt must be str, got {type(text).__name__}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class RolloutMeta:
    """Per-shard metadata (JSON sidecar contents).

    Every shard's reader must validate that the loaded shard's meta
    matches what the trainer / evaluator expects. Mismatches raise
    rather than silently mix data.
    """

    verifier_id: str
    verifier_dtype: str
    sink_size: int
    window_size: int
    block_size: int
    schema_version: str
    captured_at: str
    n_rows: int
    topk_logits: int = DEFAULT_TOPK_LOGITS

    def __post_init__(self) -> None:
        if not _VERIFIER_ID_RE.match(self.verifier_id):
            raise ValueError(
                f"verifier_id must be 'org/name' shape, got {self.verifier_id!r}"
            )
        if self.verifier_dtype not in _DTYPE_LITERALS:
            raise ValueError(
                f"verifier_dtype must be one of {sorted(_DTYPE_LITERALS)}, "
                f"got {self.verifier_dtype!r}"
            )
        if self.sink_size < 0:
            raise ValueError(f"sink_size must be >= 0, got {self.sink_size}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.topk_logits <= 0:
            raise ValueError(f"topk_logits must be > 0, got {self.topk_logits}")
        if self.n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {self.n_rows}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        # Timestamp: must parse as ISO-8601 with timezone.
        try:
            datetime.fromisoformat(self.captured_at)
        except ValueError as exc:
            raise ValueError(
                f"captured_at must be ISO-8601 with timezone, "
                f"got {self.captured_at!r}"
            ) from exc

    @classmethod
    def now(
        cls,
        *,
        verifier_id: str,
        verifier_dtype: str,
        sink_size: int,
        window_size: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        n_rows: int = 0,
        topk_logits: int = DEFAULT_TOPK_LOGITS,
    ) -> "RolloutMeta":
        """Build a meta record stamped with the current UTC timestamp."""
        return cls(
            verifier_id=verifier_id,
            verifier_dtype=verifier_dtype,
            sink_size=sink_size,
            window_size=window_size,
            block_size=block_size,
            schema_version=SCHEMA_VERSION,
            captured_at=datetime.now(timezone.utc).isoformat(),
            n_rows=n_rows,
            topk_logits=topk_logits,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutRow:
    """One captured token (ADR 0004 §2.2).

    ``hidden_state`` is the verifier's last-layer hidden state at
    this token position (length = verifier hidden_size). ``top_logits``
    and ``top_probs`` are aligned arrays of length ``topk_logits``
    (per-shard meta).
    """

    prompt_id: str
    domain: str
    language: str
    system_prompt_hash: str
    sequence_index: int
    position_in_sequence: int
    position_in_block: int
    block_index: int
    cache_logical_size: int
    token_id: int
    top_token_ids: Sequence[int]
    top_probs: Sequence[float]
    hidden_state: Sequence[float]
    verifier_top1_prob: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")
        if not self.language:
            raise ValueError("language must be non-empty")
        if len(self.system_prompt_hash) != 16:
            raise ValueError(
                f"system_prompt_hash must be 16 hex chars, "
                f"got {self.system_prompt_hash!r} (len={len(self.system_prompt_hash)})"
            )
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be >= 0")
        if self.position_in_sequence < 0:
            raise ValueError("position_in_sequence must be >= 0")
        if self.position_in_block < 0:
            raise ValueError("position_in_block must be >= 0")
        if self.block_index < 0:
            raise ValueError("block_index must be >= 0")
        if self.cache_logical_size <= 0:
            raise ValueError("cache_logical_size must be > 0")
        if self.token_id < 0:
            raise ValueError("token_id must be >= 0")
        if len(self.top_token_ids) != len(self.top_probs):
            raise ValueError(
                "top_token_ids and top_probs must have equal length, got "
                f"{len(self.top_token_ids)} vs {len(self.top_probs)}"
            )
        if not self.top_probs:
            raise ValueError("top_probs must be non-empty")
        if not self.hidden_state:
            raise ValueError("hidden_state must be non-empty")
        if any(p < 0.0 or p > 1.0 for p in self.top_probs):
            raise ValueError("each top_probs entry must be in [0, 1]")
        # top1 prob is derived; expose it on the row for filter rules
        # (ADR 0004 §2.2 drops rows with top1_prob < 0.30).
        object.__setattr__(self, "verifier_top1_prob", float(self.top_probs[0]))


def build_pyarrow_schema(
    *,
    hidden_size: int,
    topk_logits: int = DEFAULT_TOPK_LOGITS,
) -> pa.Schema:
    """Build the pyarrow Schema for a rollout shard.

    Hidden size is per-verifier; pyarrow needs the schema to be
    closed (fixed-size lists for hidden_state, top_logits, top_probs)
    so the trainer can vector-load batches without per-row shape
    checks.
    """
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    if topk_logits <= 0:
        raise ValueError(f"topk_logits must be > 0, got {topk_logits}")
    return pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("domain", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("system_prompt_hash", pa.string(), nullable=False),
            pa.field("sequence_index", pa.int64(), nullable=False),
            pa.field("position_in_sequence", pa.int64(), nullable=False),
            pa.field("position_in_block", pa.int32(), nullable=False),
            pa.field("block_index", pa.int64(), nullable=False),
            pa.field("cache_logical_size", pa.int64(), nullable=False),
            pa.field("token_id", pa.int64(), nullable=False),
            pa.field(
                "top_token_ids",
                pa.list_(pa.int64(), list_size=topk_logits),
                nullable=False,
            ),
            pa.field(
                "top_probs",
                pa.list_(pa.float32(), list_size=topk_logits),
                nullable=False,
            ),
            pa.field(
                "hidden_state",
                pa.list_(pa.float32(), list_size=hidden_size),
                nullable=False,
            ),
            pa.field("verifier_top1_prob", pa.float32(), nullable=False),
        ]
    )


def row_to_pydict(row: RolloutRow, *, expected_topk: int, expected_hidden: int) -> dict[str, Any]:
    """Convert a :class:`RolloutRow` to the dict shape pyarrow expects.

    Defensive: enforces the per-shard sizes so a shard never accepts
    a row with the wrong list length, which would corrupt the
    fixed-size-list schema and silently produce undefined data on
    read.
    """
    if len(row.top_token_ids) != expected_topk:
        raise ValueError(
            f"top_token_ids length {len(row.top_token_ids)} != "
            f"expected_topk {expected_topk}"
        )
    if len(row.top_probs) != expected_topk:
        raise ValueError(
            f"top_probs length {len(row.top_probs)} != "
            f"expected_topk {expected_topk}"
        )
    if len(row.hidden_state) != expected_hidden:
        raise ValueError(
            f"hidden_state length {len(row.hidden_state)} != "
            f"expected_hidden {expected_hidden}"
        )
    return {
        "prompt_id": row.prompt_id,
        "domain": row.domain,
        "language": row.language,
        "system_prompt_hash": row.system_prompt_hash,
        "sequence_index": row.sequence_index,
        "position_in_sequence": row.position_in_sequence,
        "position_in_block": row.position_in_block,
        "block_index": row.block_index,
        "cache_logical_size": row.cache_logical_size,
        "token_id": row.token_id,
        "top_token_ids": list(row.top_token_ids),
        "top_probs": [float(p) for p in row.top_probs],
        "hidden_state": [float(h) for h in row.hidden_state],
        "verifier_top1_prob": row.verifier_top1_prob,
    }


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_TOPK_LOGITS",
    "DEFAULT_BLOCK_SIZE",
    "RolloutMeta",
    "RolloutRow",
    "build_pyarrow_schema",
    "row_to_pydict",
    "system_prompt_hash",
]
