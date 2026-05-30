"""Versioned Parquet shard writer (ADR 0004 §2.5).

Path layout (mandated by ADR 0004 §2.5)::

    data/alignment/<verifier_id>/<verifier_dtype>/<schema_version>/
        shard_<NNNNN>.parquet
        shard_<NNNNN>.meta.json

Writes are **atomic at the shard level**: rows accumulate in memory
(or in a temp file on the same filesystem) and the rename to the
final shard path happens after the JSON sidecar has been flushed.
A consumer that opens ``shard_NNNNN.parquet`` is therefore
guaranteed to find a matching ``shard_NNNNN.meta.json`` and a
schema-conforming row group.

Each shard owns one ``RolloutMeta``. The writer enforces:

  * every row's ``system_prompt_hash`` is present (the trainer
    needs it for stratified eval per ADR 0004 §2.7);
  * every row's list lengths match the per-shard ``topk_logits``
    and ``hidden_size`` (otherwise pyarrow's fixed-size-list schema
    would silently truncate or error mid-batch);
  * the meta sidecar's ``n_rows`` is updated to the actual row
    count on close.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    DEFAULT_TOPK_LOGITS,
    RolloutMeta,
    RolloutRow,
    build_pyarrow_schema,
    row_to_pydict,
)


_SHARD_FNAME_RE = re.compile(r"^shard_(\d{5})\.parquet$")


def shard_dir(
    root: str | os.PathLike[str],
    *,
    verifier_id: str,
    verifier_dtype: str,
    schema_version: str,
) -> Path:
    """Compute the canonical shard directory for a verifier."""
    if not verifier_id or "/" not in verifier_id:
        raise ValueError(
            f"verifier_id must be 'org/name' shape, got {verifier_id!r}"
        )
    if not verifier_dtype:
        raise ValueError("verifier_dtype must be non-empty")
    if not schema_version:
        raise ValueError("schema_version must be non-empty")
    # Path components are sanitized by the dataclass validators upstream;
    # we still belt-and-braces against `..` traversal here.
    for part in (verifier_id, verifier_dtype, schema_version):
        if ".." in part.split(os.sep):
            raise ValueError(f"path traversal in {part!r}")
    return Path(root) / "alignment" / verifier_id / verifier_dtype / schema_version


def next_shard_id(directory: Path) -> int:
    """Return the next unused shard id in ``directory`` (0 if empty)."""
    if not directory.exists():
        return 0
    used: list[int] = []
    for fname in os.listdir(directory):
        m = _SHARD_FNAME_RE.match(fname)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1 if used else 0


class RolloutShardWriter:
    """Buffered Parquet writer for one shard.

    Use as a context manager so the close path (which flushes the
    pyarrow writer, atomically renames the temp file, and writes
    the meta sidecar) always runs::

        with RolloutShardWriter(
            root_dir="data",
            meta=RolloutMeta.now(...),
            hidden_size=2048,
        ) as w:
            for row in rows:
                w.write(row)
        # After block exit: shard_00000.parquet + shard_00000.meta.json
        # exist and are consistent with each other.

    The writer never overwrites an existing shard. If a shard with
    the same id already exists at the target path, ``__init__``
    refuses; pick the next id via :func:`next_shard_id`.
    """

    def __init__(
        self,
        *,
        root_dir: str | os.PathLike[str],
        meta: RolloutMeta,
        hidden_size: int,
        shard_id: int | None = None,
        batch_size: int = 256,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._meta = meta
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._dir = shard_dir(
            root_dir,
            verifier_id=meta.verifier_id,
            verifier_dtype=meta.verifier_dtype,
            schema_version=meta.schema_version,
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._shard_id = (
            shard_id if shard_id is not None else next_shard_id(self._dir)
        )
        if self._shard_id < 0:
            raise ValueError(f"shard_id must be >= 0, got {self._shard_id}")
        self._final_path = self._dir / f"shard_{self._shard_id:05d}.parquet"
        self._meta_path = self._dir / f"shard_{self._shard_id:05d}.meta.json"
        self._tmp_path = self._dir / f".shard_{self._shard_id:05d}.parquet.tmp"
        if self._final_path.exists():
            raise FileExistsError(
                f"shard already exists: {self._final_path}; pick a fresh shard_id"
            )

        self._schema = build_pyarrow_schema(
            hidden_size=hidden_size,
            topk_logits=meta.topk_logits or DEFAULT_TOPK_LOGITS,
        )
        self._writer = pq.ParquetWriter(self._tmp_path, self._schema)
        self._buffer: list[dict] = []
        self._n_rows = 0
        self._closed = False

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "RolloutShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._abort()
            return
        self.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, row: RolloutRow) -> None:
        """Buffer one row; flushes when ``batch_size`` is hit."""
        if self._closed:
            raise RuntimeError("writer is closed")
        record = row_to_pydict(
            row,
            expected_topk=self._meta.topk_logits,
            expected_hidden=self._hidden_size,
        )
        self._buffer.append(record)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def write_many(self, rows: Iterable[RolloutRow]) -> int:
        """Write an iterable, returning the number of rows written."""
        n = 0
        for row in rows:
            self.write(row)
            n += 1
        return n

    @property
    def n_rows(self) -> int:
        """Rows already flushed to the parquet temp file."""
        return self._n_rows

    @property
    def final_path(self) -> Path:
        return self._final_path

    @property
    def meta_path(self) -> Path:
        return self._meta_path

    # ------------------------------------------------------------------
    # Close path
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._n_rows += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._writer.close()
        # Update the meta with the actual flushed row count and write
        # the JSON sidecar BEFORE renaming the parquet temp file.
        # Readers see meta-and-data appearing together; if the rename
        # fails after the meta write we leave a stale meta which the
        # next writer (with the same shard_id) will overwrite.
        meta_with_count = RolloutMeta(
            verifier_id=self._meta.verifier_id,
            verifier_dtype=self._meta.verifier_dtype,
            sink_size=self._meta.sink_size,
            window_size=self._meta.window_size,
            block_size=self._meta.block_size,
            schema_version=self._meta.schema_version,
            captured_at=self._meta.captured_at,
            n_rows=self._n_rows,
            topk_logits=self._meta.topk_logits,
        )
        meta_tmp = self._meta_path.with_suffix(self._meta_path.suffix + ".tmp")
        with meta_tmp.open("w") as f:
            json.dump(meta_with_count.to_json_dict(), f, indent=2, sort_keys=True)
        meta_tmp.replace(self._meta_path)
        self._tmp_path.replace(self._final_path)
        self._closed = True

    def _abort(self) -> None:
        """Discard the in-progress shard. Used on exception in the
        context manager so a half-written file doesn't pretend to be
        a complete shard."""
        if self._closed:
            return
        try:
            self._writer.close()
        finally:
            for p in (self._tmp_path, self._meta_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            self._closed = True


def read_meta(meta_path: str | os.PathLike[str]) -> RolloutMeta:
    """Load and validate a shard's meta sidecar."""
    with Path(meta_path).open("r") as f:
        payload = json.load(f)
    return RolloutMeta(**payload)


def list_shards(directory: str | os.PathLike[str]) -> list[Path]:
    """Return all completed shard parquet paths in ``directory``,
    sorted by shard id."""
    d = Path(directory)
    if not d.exists():
        return []
    found: list[tuple[int, Path]] = []
    for entry in os.listdir(d):
        m = _SHARD_FNAME_RE.match(entry)
        if m:
            found.append((int(m.group(1)), d / entry))
    found.sort()
    return [p for _, p in found]


__all__ = [
    "RolloutShardWriter",
    "list_shards",
    "next_shard_id",
    "read_meta",
    "shard_dir",
]
