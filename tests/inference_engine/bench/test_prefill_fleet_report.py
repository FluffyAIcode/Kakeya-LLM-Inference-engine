import pytest

from inference_engine.bench.prefill_fleet_report import (
    assert_public_safe,
    normalize_stage,
    summarize_stages,
)


def _stage(name="remote_compute", source="remote_worker"):
    return {
        "name": name,
        "hit_source": source,
        "ok": True,
        "prefix_tokens": 100,
        "output_tokens": 10,
        "append_s": 5.0,
        "ttft_s": 5.2,
        "decode_s": 2.0,
        "e2e_s": 7.0,
        "delta": {"bytes_received": 1000},
    }


def test_normalize_stage_derives_throughput_and_latency():
    stage = normalize_stage(_stage())
    assert stage["prefill_or_restore_tok_s"] == 20.0
    assert stage["decode_tok_s"] == 5.0
    assert stage["generation_latency_ms_per_token"] == 200.0
    assert stage["e2e_tok_s"] == 10 / 7


def test_summary_aggregates_sources_and_medians():
    summary = summarize_stages([
        _stage(),
        _stage("primary_hot_hit", "primary_hot"),
        {**_stage("allens_cold_restore", "allens_offload"), "ok": False},
    ])
    assert summary["stages_total"] == 3
    assert summary["stages_failed"] == 1
    assert summary["hit_source_counts"]["remote_worker"] == 1
    assert summary["hit_source_counts"]["primary_hot"] == 1
    assert summary["bytes_received"] == 3000
    assert summarize_stages([])["decode_tok_s_p50"] == 0


def test_schema_rejects_unknown_and_private_fields():
    with pytest.raises(ValueError, match="unknown benchmark phase"):
        normalize_stage(_stage("bad"))
    with pytest.raises(ValueError, match="unknown hit_source"):
        normalize_stage(_stage(source="peer:1"))
    with pytest.raises(ValueError, match="non-negative"):
        normalize_stage({**_stage(), "append_s": -1})
    with pytest.raises(ValueError, match="private benchmark field"):
        assert_public_safe({"prompt": "secret"})
    with pytest.raises(ValueError, match="private path"):
        assert_public_safe({"value": "/Users/private/model"})
    with pytest.raises(ValueError, match="private path"):
        assert_public_safe(["169.254.27.104"])
    assert normalize_stage({
        **_stage(),
        "output_tokens": 0,
        "prefix_tokens": 0,
        "append_s": 0,
        "decode_s": 0,
        "e2e_s": 0,
    })["decode_tok_s"] == 0
