from scripts.fill_prefill_cache_from_live_grpc import (
    _node_cache,
    _ratio,
    _safe_report_item,
)


def test_capacity_helpers_are_node_specific():
    nodes = [
        {"id": "head", "cache": {"bytes_used": 9, "bytes_free": 1}},
        {"id": "peer", "cache": {"bytes_used": 8, "bytes_free": 2}},
    ]
    assert _ratio(_node_cache(nodes, "head")) == 0.9
    assert _ratio(_node_cache(nodes, "peer")) == 0.8
    assert _ratio({"bytes_used": 0, "bytes_free": 0}) == 0.0


def test_safe_report_never_contains_token_ids():
    item = {
        "capture_id": "salted-id",
        "token_count": 512,
        "token_ids": [1, 2, 3],
    }
    report = _safe_report_item(item, replay_tokens=520, wall_seconds=1.25)
    assert report == {
        "capture_id": "salted-id",
        "captured_tokens": 512,
        "replay_tokens": 520,
        "wall_seconds": 1.25,
    }
    assert "token_ids" not in report
