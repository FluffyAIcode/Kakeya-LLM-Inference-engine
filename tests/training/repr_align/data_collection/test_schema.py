"""Unit tests for ``training.repr_align.data_collection.schema``.

Real concrete inputs only — no mocks per project rule.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pytest

from training.repr_align.data_collection.schema import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TOPK_LOGITS,
    SCHEMA_VERSION,
    RolloutMeta,
    RolloutRow,
    build_pyarrow_schema,
    row_to_pydict,
    system_prompt_hash,
)


# ---------------------------------------------------------------------------
# system_prompt_hash
# ---------------------------------------------------------------------------


def test_system_prompt_hash_is_deterministic():
    a = system_prompt_hash("hello")
    b = system_prompt_hash("hello")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_system_prompt_hash_differs_on_different_inputs():
    assert system_prompt_hash("a") != system_prompt_hash("b")


def test_system_prompt_hash_rejects_non_str():
    with pytest.raises(TypeError):
        system_prompt_hash(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RolloutMeta
# ---------------------------------------------------------------------------


def _good_meta(**overrides):
    base = dict(
        verifier_id="Qwen/Qwen3-1.7B",
        verifier_dtype="bf16",
        sink_size=4,
        window_size=64,
        block_size=4,
        schema_version=SCHEMA_VERSION,
        captured_at=datetime.now(timezone.utc).isoformat(),
        n_rows=0,
        topk_logits=DEFAULT_TOPK_LOGITS,
    )
    base.update(overrides)
    return RolloutMeta(**base)


def test_meta_happy_path_now_factory():
    m = RolloutMeta.now(
        verifier_id="Qwen/Qwen3-1.7B",
        verifier_dtype="bf16",
        sink_size=4,
        window_size=64,
    )
    assert m.schema_version == SCHEMA_VERSION
    assert m.block_size == DEFAULT_BLOCK_SIZE
    assert m.topk_logits == DEFAULT_TOPK_LOGITS
    # Round-trip through to_json_dict
    d = m.to_json_dict()
    assert d["verifier_id"] == "Qwen/Qwen3-1.7B"
    assert d["schema_version"] == SCHEMA_VERSION


def test_meta_rejects_bad_verifier_id():
    with pytest.raises(ValueError, match="verifier_id"):
        _good_meta(verifier_id="no_slash_here")


def test_meta_rejects_bad_dtype():
    with pytest.raises(ValueError, match="verifier_dtype"):
        _good_meta(verifier_dtype="float7")


def test_meta_rejects_negative_sink_size():
    with pytest.raises(ValueError, match="sink_size"):
        _good_meta(sink_size=-1)


def test_meta_rejects_zero_window_size():
    with pytest.raises(ValueError, match="window_size"):
        _good_meta(window_size=0)


def test_meta_rejects_zero_block_size():
    with pytest.raises(ValueError, match="block_size"):
        _good_meta(block_size=0)


def test_meta_rejects_zero_topk():
    with pytest.raises(ValueError, match="topk_logits"):
        _good_meta(topk_logits=0)


def test_meta_rejects_negative_n_rows():
    with pytest.raises(ValueError, match="n_rows"):
        _good_meta(n_rows=-1)


def test_meta_rejects_wrong_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        _good_meta(schema_version="999")


def test_meta_rejects_bad_timestamp():
    with pytest.raises(ValueError, match="captured_at"):
        _good_meta(captured_at="yesterday")


# ---------------------------------------------------------------------------
# RolloutRow
# ---------------------------------------------------------------------------


def _good_row(**overrides):
    base = dict(
        prompt_id="p0",
        domain="chat_en",
        language="en",
        system_prompt_hash=system_prompt_hash("you are helpful"),
        sequence_index=0,
        position_in_sequence=0,
        position_in_block=0,
        block_index=0,
        cache_logical_size=1,
        token_id=42,
        top_token_ids=[42, 7, 99],
        top_probs=[0.7, 0.2, 0.1],
        hidden_state=[0.1, 0.2, 0.3, 0.4],
    )
    base.update(overrides)
    return RolloutRow(**base)


def test_row_happy_path_derives_top1_prob():
    r = _good_row()
    assert r.verifier_top1_prob == pytest.approx(0.7)


def test_row_rejects_empty_prompt_id():
    with pytest.raises(ValueError, match="prompt_id"):
        _good_row(prompt_id="")


def test_row_rejects_empty_domain():
    with pytest.raises(ValueError, match="domain"):
        _good_row(domain="")


def test_row_rejects_empty_language():
    with pytest.raises(ValueError, match="language"):
        _good_row(language="")


def test_row_rejects_bad_system_prompt_hash_length():
    with pytest.raises(ValueError, match="system_prompt_hash"):
        _good_row(system_prompt_hash="abc")


def test_row_rejects_negative_sequence_index():
    with pytest.raises(ValueError, match="sequence_index"):
        _good_row(sequence_index=-1)


def test_row_rejects_negative_position_in_sequence():
    with pytest.raises(ValueError, match="position_in_sequence"):
        _good_row(position_in_sequence=-1)


def test_row_rejects_negative_position_in_block():
    with pytest.raises(ValueError, match="position_in_block"):
        _good_row(position_in_block=-1)


def test_row_rejects_negative_block_index():
    with pytest.raises(ValueError, match="block_index"):
        _good_row(block_index=-1)


def test_row_rejects_zero_cache_logical_size():
    with pytest.raises(ValueError, match="cache_logical_size"):
        _good_row(cache_logical_size=0)


def test_row_rejects_negative_token_id():
    with pytest.raises(ValueError, match="token_id"):
        _good_row(token_id=-1)


def test_row_rejects_mismatched_top_arrays():
    with pytest.raises(ValueError, match="top_token_ids and top_probs"):
        _good_row(top_token_ids=[1, 2], top_probs=[0.5, 0.3, 0.2])


def test_row_rejects_empty_top_probs():
    with pytest.raises(ValueError, match="top_probs"):
        _good_row(top_token_ids=[], top_probs=[])


def test_row_rejects_empty_hidden_state():
    with pytest.raises(ValueError, match="hidden_state"):
        _good_row(hidden_state=[])


def test_row_rejects_out_of_range_prob():
    with pytest.raises(ValueError, match=r"top_probs entry"):
        _good_row(top_probs=[1.2, 0.0, 0.0])


# ---------------------------------------------------------------------------
# build_pyarrow_schema
# ---------------------------------------------------------------------------


def test_schema_has_expected_fields_and_sizes():
    schema = build_pyarrow_schema(hidden_size=8, topk_logits=5)
    fields = {f.name: f for f in schema}
    assert set(fields) >= {
        "prompt_id", "domain", "language", "system_prompt_hash",
        "token_id", "top_token_ids", "top_probs", "hidden_state",
        "verifier_top1_prob",
    }
    h = fields["hidden_state"].type
    assert isinstance(h, pa.FixedSizeListType)
    assert h.list_size == 8
    t = fields["top_probs"].type
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 5


def test_schema_rejects_zero_hidden_size():
    with pytest.raises(ValueError, match="hidden_size"):
        build_pyarrow_schema(hidden_size=0)


def test_schema_rejects_zero_topk():
    with pytest.raises(ValueError, match="topk_logits"):
        build_pyarrow_schema(hidden_size=8, topk_logits=0)


# ---------------------------------------------------------------------------
# row_to_pydict
# ---------------------------------------------------------------------------


def test_row_to_pydict_produces_pyarrow_compatible_dict():
    schema = build_pyarrow_schema(hidden_size=4, topk_logits=3)
    row = _good_row()
    record = row_to_pydict(row, expected_topk=3, expected_hidden=4)
    table = pa.Table.from_pylist([record], schema=schema)
    assert table.num_rows == 1
    assert table["token_id"][0].as_py() == 42
    # top_probs round-trip preserves values within float32 precision
    assert pytest.approx(table["top_probs"][0].as_py(), abs=1e-6) == [0.7, 0.2, 0.1]


def test_row_to_pydict_rejects_topk_mismatch():
    row = _good_row()
    with pytest.raises(ValueError, match="top_token_ids length"):
        row_to_pydict(row, expected_topk=5, expected_hidden=4)


def test_row_to_pydict_rejects_topprobs_mismatch():
    # Build a row where top_token_ids has the right length for one topk
    # but we ask for a different topk that matches token ids only.
    row = RolloutRow(
        prompt_id="p", domain="d", language="en",
        system_prompt_hash=system_prompt_hash("x"),
        sequence_index=0, position_in_sequence=0, position_in_block=0,
        block_index=0, cache_logical_size=1, token_id=1,
        top_token_ids=[1, 2, 3], top_probs=[0.5, 0.3, 0.2],
        hidden_state=[0.1, 0.2, 0.3, 0.4],
    )
    # expected_topk matches top_token_ids (3) but the function still
    # checks both arrays equal expected_topk; we trip the second check
    # by constructing a row with mismatched lengths via a separate
    # helper that bypasses post-init. Here we instead test the
    # second branch with hidden mismatch:
    with pytest.raises(ValueError, match="hidden_state length"):
        row_to_pydict(row, expected_topk=3, expected_hidden=8)


def test_row_to_pydict_top_probs_branch_independent():
    """Cover the top_probs length branch by passing a row whose
    top_probs length differs from top_token_ids length when going
    through row_to_pydict (constructed via __dict__ patching to
    bypass __post_init__'s guard, which is exactly what the
    defensive check inside row_to_pydict is for)."""
    row = _good_row()
    object.__setattr__(row, "top_probs", [0.5, 0.5])
    with pytest.raises(ValueError, match="top_probs length"):
        row_to_pydict(row, expected_topk=3, expected_hidden=4)
