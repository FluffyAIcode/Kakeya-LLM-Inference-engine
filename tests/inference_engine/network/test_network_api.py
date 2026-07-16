from __future__ import annotations

from fastapi.testclient import TestClient

from inference_engine.distributed.capability import (
    CacheCompatibility,
    CapabilityRegistry,
    NodeCapability,
)
from inference_engine.distributed.cache_fill import CacheFillCapture
from inference_engine.distributed.prefill_cache import PrefixCacheStore
from inference_engine.network.api import create_network_app
from inference_engine.network.state import NetworkState


def _client(tmp_path):
    compatibility = CacheCompatibility(model_id="m")
    state = NetworkState(
        CapabilityRegistry(NodeCapability(node_id="head", grpc_address="head:1")),
        PrefixCacheStore(compatibility, max_bytes=100, node_id="head"),
        state_path=tmp_path / "state.json",
        prefill_stats_provider=lambda: {
            "remote_jobs": 3,
            "remote_hits": 2,
            "tokens_reused": 192,
        },
    )
    capture = CacheFillCapture(max_items=4)
    client = TestClient(create_network_app(
        state,
        api_key="secret",
        cache_fill_capture=capture,
    ))
    client.network_state = state
    client.cache_fill_capture = capture
    return client


def test_dashboard_health_and_read_apis(tmp_path):
    client = _client(tmp_path)
    assert client.get("/").status_code == 200
    dashboard = client.get("/network").text
    assert "Kakeya Inference Network" in dashboard
    assert "Remote prefill jobs" in dashboard
    assert "LRU evictions" in dashboard
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/v1/network/summary").json()["online_nodes"] == 1
    assert len(client.get("/v1/network/nodes").json()) == 1
    assert client.get("/v1/network/groups").json() == []
    assert "nodes" in client.get("/v1/network/topology").json()
    assert client.get("/v1/network/kvfs").json()["uri"].startswith("kv://")
    assert client.get("/v1/network/tokens").json()["completed"] == 0
    assert client.get("/v1/network/prefill").json()["remote_jobs"] == 3
    assert client.get("/v1/network/maintenance/capture").status_code == 401
    events = client.get("/v1/network/events?once=true")
    assert events.status_code == 200
    assert "event: summary" in events.text


def test_write_apis_require_key_and_update_state(tmp_path):
    client = _client(tmp_path)
    body = {"alias": "peer", "address": "peer:2", "region": "HK"}
    assert client.post("/v1/network/nodes/register", json=body).status_code == 401
    registered = client.post(
        "/v1/network/nodes/register",
        json=body,
        headers={"X-API-Key": "secret"},
    )
    assert registered.status_code == 200
    assert registered.json()["pairing_token"].startswith("kn_pair_")
    group = client.post(
        "/v1/network/groups",
        json={"name": "g", "node_ids": ["head", "peer"]},
        headers={"X-API-Key": "secret"},
    )
    assert group.status_code == 200
    telemetry = client.post(
        "/v1/network/telemetry/tokens",
        json={"node_id": "head", "completed": 10, "kv_assisted": 7},
        headers={"X-API-Key": "secret"},
    )
    assert telemetry.json()["status"] == "accepted"
    assert client.get("/v1/network/tokens").json()["kv_assisted"] == 7
    client.cache_fill_capture.observe(client_label="live", token_ids=[1, 2])
    status = client.get(
        "/v1/network/maintenance/capture",
        headers={"X-API-Key": "secret"},
    )
    assert status.json()["queued"] == 1
    drained = client.post(
        "/v1/network/maintenance/capture/drain",
        json={"max_items": 1},
        headers={"X-API-Key": "secret"},
    )
    assert drained.json()["items"][0]["token_ids"] == [1, 2]


def test_write_error_mapping_and_event_stream(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(
        client.network_state,
        "register_node",
        lambda **_: (_ for _ in ()).throw(ValueError("bad registration")),
    )
    response = client.post(
        "/v1/network/nodes/register",
        json={"alias": "a", "address": "a:1"},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 400
    monkeypatch.setattr(
        client.network_state,
        "create_group",
        lambda **_: (_ for _ in ()).throw(ValueError("bad group")),
    )
    assert client.post(
        "/v1/network/groups",
        json={"name": "g", "node_ids": ["a"]},
        headers={"X-API-Key": "secret"},
    ).status_code == 400
    monkeypatch.setattr(
        client.network_state,
        "record_tokens",
        lambda **_: (_ for _ in ()).throw(ValueError("bad tokens")),
    )
    assert client.post(
        "/v1/network/telemetry/tokens",
        json={"node_id": "a", "completed": 1},
        headers={"X-API-Key": "secret"},
    ).status_code == 400


def test_disabled_capture_and_missing_maintenance_key(tmp_path):
    compatibility = CacheCompatibility(model_id="m")
    state = NetworkState(
        CapabilityRegistry(NodeCapability(node_id="head", grpc_address="head:1")),
        PrefixCacheStore(compatibility, max_bytes=100, node_id="head"),
        state_path=tmp_path / "disabled.json",
    )
    without_capture = TestClient(create_network_app(state, api_key="secret"))
    assert without_capture.get(
        "/v1/network/maintenance/capture",
        headers={"X-API-Key": "secret"},
    ).status_code == 404
    assert without_capture.post(
        "/v1/network/maintenance/capture/drain",
        json={"max_items": 1},
        headers={"X-API-Key": "secret"},
    ).status_code == 404
    without_key = TestClient(create_network_app(
        state,
        api_key="",
        cache_fill_capture=CacheFillCapture(),
    ))
    assert without_key.get("/v1/network/maintenance/capture").status_code == 401
