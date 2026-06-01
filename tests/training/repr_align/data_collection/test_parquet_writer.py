"""Unit tests for ``training.repr_align.data_collection.parquet_writer``."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from training.repr_align.data_collection.parquet_writer import (
    RolloutShardWriter,
    list_shards,
    next_shard_id,
    read_meta,
    shard_dir,
)
from training.repr_align.data_collection.schema import (
    DEFAULT_TOPK_LOGITS,
    SCHEMA_VERSION,
    RolloutMeta,
    RolloutRow,
    system_prompt_hash,
)


# ---------------------------------------------------------------------------
# shard_dir + next_shard_id + list_shards
# ---------------------------------------------------------------------------


def test_shard_dir_layout(tmp_path: Path):
    d = shard_dir(
        tmp_path,
        verifier_id="Qwen/Qwen3-1.7B",
        verifier_dtype="bf16",
        schema_version=SCHEMA_VERSION,
    )
    assert d == tmp_path / "alignment" / "Qwen/Qwen3-1.7B" / "bf16" / SCHEMA_VERSION


def test_shard_dir_rejects_bad_verifier_id(tmp_path: Path):
    with pytest.raises(ValueError, match="verifier_id"):
        shard_dir(tmp_path, verifier_id="no_slash", verifier_dtype="bf16",
                  schema_version=SCHEMA_VERSION)


def test_shard_dir_rejects_empty_dtype(tmp_path: Path):
    with pytest.raises(ValueError, match="verifier_dtype"):
        shard_dir(tmp_path, verifier_id="org/name", verifier_dtype="",
                  schema_version=SCHEMA_VERSION)


def test_shard_dir_rejects_empty_schema_version(tmp_path: Path):
    with pytest.raises(ValueError, match="schema_version"):
        shard_dir(tmp_path, verifier_id="org/name", verifier_dtype="bf16",
                  schema_version="")


def test_shard_dir_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="path traversal"):
        shard_dir(tmp_path, verifier_id="org/name",
                  verifier_dtype=f"..{__import__('os').sep}etc",
                  schema_version=SCHEMA_VERSION)


def test_next_shard_id_empty_returns_zero(tmp_path: Path):
    assert next_shard_id(tmp_path) == 0


def test_next_shard_id_missing_dir_returns_zero(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    assert next_shard_id(nonexistent) == 0


def test_next_shard_id_skips_unrelated_files(tmp_path: Path):
    (tmp_path / "shard_00003.parquet").write_bytes(b"")
    (tmp_path / "random.txt").write_text("x")
    assert next_shard_id(tmp_path) == 4


def test_list_shards_missing_dir_returns_empty(tmp_path: Path):
    assert list_shards(tmp_path / "missing") == []


def test_list_shards_orders_by_id(tmp_path: Path):
    (tmp_path / "shard_00010.parquet").write_bytes(b"")
    (tmp_path / "shard_00002.parquet").write_bytes(b"")
    (tmp_path / "junk.parquet").write_bytes(b"")
    paths = list_shards(tmp_path)
    assert [p.name for p in paths] == ["shard_00002.parquet", "shard_00010.parquet"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


HIDDEN = 4
TOPK = 3


def _make_meta(n_rows: int = 0) -> RolloutMeta:
    return RolloutMeta.now(
        verifier_id="Qwen/Qwen3-1.7B",
        verifier_dtype="bf16",
        sink_size=4,
        window_size=64,
        block_size=4,
        n_rows=n_rows,
        topk_logits=TOPK,
    )


def _make_row(i: int = 0) -> RolloutRow:
    return RolloutRow(
        prompt_id=f"p{i}",
        domain="chat_en",
        language="en",
        system_prompt_hash=system_prompt_hash("you are helpful"),
        sequence_index=0,
        position_in_sequence=i,
        position_in_block=i % 4,
        block_index=i // 4,
        cache_logical_size=1 + i,
        token_id=100 + i,
        top_token_ids=[100 + i, 7, 99],
        top_probs=[0.7, 0.2, 0.1],
        hidden_state=[0.1 * i, 0.2, 0.3, 0.4],
    )


# ---------------------------------------------------------------------------
# RolloutShardWriter — happy path
# ---------------------------------------------------------------------------


def test_writer_writes_then_reads_back(tmp_path: Path):
    meta = _make_meta()
    with RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN, batch_size=2,
    ) as w:
        n = w.write_many(_make_row(i) for i in range(5))
        assert n == 5

    # Files exist at the expected versioned path
    expected_dir = (
        tmp_path / "alignment" / meta.verifier_id / meta.verifier_dtype
        / SCHEMA_VERSION
    )
    parquet = expected_dir / "shard_00000.parquet"
    meta_json = expected_dir / "shard_00000.meta.json"
    assert parquet.exists()
    assert meta_json.exists()

    # Parquet round-trip
    table = pq.read_table(parquet)
    assert table.num_rows == 5
    assert table.column("token_id").to_pylist() == [100, 101, 102, 103, 104]

    # Meta sidecar reflects the actual row count
    loaded = read_meta(meta_json)
    assert loaded.n_rows == 5
    assert loaded.verifier_id == meta.verifier_id
    assert loaded.schema_version == SCHEMA_VERSION


def test_writer_n_rows_property_tracks_flushed_count(tmp_path: Path):
    meta = _make_meta()
    with RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN, batch_size=2,
    ) as w:
        w.write(_make_row(0))
        # Below batch_size: not yet flushed
        assert w.n_rows == 0
        w.write(_make_row(1))
        # Now batch fired: 2 rows in parquet temp
        assert w.n_rows == 2


def test_writer_picks_next_shard_id_when_existing_present(tmp_path: Path):
    meta = _make_meta()
    # First shard
    with RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
    ) as w1:
        w1.write(_make_row(0))
    # Second shard auto-picks id 1
    with RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
    ) as w2:
        w2.write(_make_row(0))
        assert w2.final_path.name == "shard_00001.parquet"


def test_writer_explicit_shard_id(tmp_path: Path):
    meta = _make_meta()
    with RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN, shard_id=42,
    ) as w:
        w.write(_make_row(0))
        assert w.meta_path.name == "shard_00042.meta.json"
    out = list_shards(w.final_path.parent)
    assert out[0].name == "shard_00042.parquet"


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_writer_rejects_zero_hidden(tmp_path: Path):
    with pytest.raises(ValueError, match="hidden_size"):
        RolloutShardWriter(
            root_dir=tmp_path, meta=_make_meta(), hidden_size=0,
        )


def test_writer_rejects_zero_batch(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_size"):
        RolloutShardWriter(
            root_dir=tmp_path, meta=_make_meta(),
            hidden_size=HIDDEN, batch_size=0,
        )


def test_writer_rejects_negative_shard_id(tmp_path: Path):
    with pytest.raises(ValueError, match="shard_id"):
        RolloutShardWriter(
            root_dir=tmp_path, meta=_make_meta(),
            hidden_size=HIDDEN, shard_id=-1,
        )


def test_writer_rejects_existing_shard_path(tmp_path: Path):
    meta = _make_meta()
    d = shard_dir(
        tmp_path,
        verifier_id=meta.verifier_id,
        verifier_dtype=meta.verifier_dtype,
        schema_version=meta.schema_version,
    )
    d.mkdir(parents=True, exist_ok=True)
    (d / "shard_00007.parquet").write_bytes(b"placeholder")
    with pytest.raises(FileExistsError):
        RolloutShardWriter(
            root_dir=tmp_path, meta=meta,
            hidden_size=HIDDEN, shard_id=7,
        )


def test_writer_write_after_close_raises(tmp_path: Path):
    meta = _make_meta()
    w = RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
    )
    w.write(_make_row(0))
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        w.write(_make_row(1))


def test_writer_close_is_idempotent(tmp_path: Path):
    meta = _make_meta()
    w = RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
    )
    w.write(_make_row(0))
    w.close()
    # Second close is a no-op
    w.close()
    assert w.final_path.exists()


def test_writer_aborts_on_exception(tmp_path: Path):
    meta = _make_meta()
    with pytest.raises(RuntimeError, match="boom"):
        with RolloutShardWriter(
            root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
        ) as w:
            w.write(_make_row(0))
            raise RuntimeError("boom")
    # Final shard never appeared
    assert not w.final_path.exists()
    # Temp file cleaned up
    tmp = w.final_path.with_name(f".{w.final_path.name}.tmp")
    assert not tmp.exists()


def test_writer_abort_idempotent(tmp_path: Path):
    """Calling _abort after a clean close is a no-op."""
    meta = _make_meta()
    w = RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
    )
    w.close()
    # _abort on already-closed writer must not raise
    w._abort()  # type: ignore[attr-defined]


def test_writer_abort_handles_missing_temp_files(tmp_path: Path):
    """Cover the FileNotFoundError branch in _abort."""
    meta = _make_meta()
    w = RolloutShardWriter(
        root_dir=tmp_path, meta=meta, hidden_size=HIDDEN,
    )
    # Manually delete the temp parquet so _abort hits FileNotFoundError
    w._tmp_path.unlink()  # type: ignore[attr-defined]
    # Now abort should still complete cleanly
    w._abort()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# read_meta
# ---------------------------------------------------------------------------


def test_read_meta_round_trip(tmp_path: Path):
    meta = _make_meta(n_rows=5)
    p = tmp_path / "meta.json"
    with p.open("w") as f:
        json.dump(meta.to_json_dict(), f)
    loaded = read_meta(p)
    assert loaded == meta


def test_read_meta_rejects_bad_payload(tmp_path: Path):
    p = tmp_path / "meta.json"
    bad = dict(
        verifier_id="bad", verifier_dtype="bf16", sink_size=0,
        window_size=64, block_size=4, schema_version=SCHEMA_VERSION,
        captured_at="2026-05-30T00:00:00+00:00", n_rows=0,
        topk_logits=DEFAULT_TOPK_LOGITS,
    )
    with p.open("w") as f:
        json.dump(bad, f)
    with pytest.raises(ValueError, match="verifier_id"):
        read_meta(p)
