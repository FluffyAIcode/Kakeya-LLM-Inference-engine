import json

import psutil
from fastapi.testclient import TestClient

from inference_engine.server import runtime_health
from inference_engine.server.runtime_health import (
    DecodeLiveness,
    PrimaryMemoryGovernor,
    create_runtime_health_app,
)
from inference_engine.session import SessionStore


class Verifier:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def test_process_footprint_uses_fork_free_psutil(monkeypatch):
    seen = []

    class Process:
        def __init__(self, pid):
            seen.append(pid)

        def memory_info(self):
            return type("MemoryInfo", (), {"rss": 12345})()

    monkeypatch.setattr(runtime_health.psutil, "Process", Process)
    assert runtime_health.process_footprint_bytes(42) == 12345
    assert seen == [42]


def test_process_footprint_returns_zero_when_process_disappears(monkeypatch):
    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(runtime_health.psutil, "Process", missing)
    assert runtime_health.process_footprint_bytes(42) == 0


def test_liveness_is_atomically_published(tmp_path):
    path = tmp_path / "live.json"
    live = DecodeLiveness(path, clock=lambda: 123.0, pid=42)
    live.update("decode", "session-1", 7)
    assert json.loads(path.read_text()) == {
        "phase": "decode",
        "session_id": "session-1",
        "token_index": 7,
        "updated_at_unix": 123.0,
        "pid": 42,
    }


def test_memory_governor_levels_and_idle_cleanup():
    store = SessionStore(capacity=1)
    verifier = Verifier()
    cleared = []
    footprint = [18]
    governor = PrimaryMemoryGovernor(
        store,
        verifier,
        warning_bytes=10,
        drain_bytes=20,
        unhealthy_bytes=30,
        memory_provider=lambda: (3, 4, 5),
        footprint_provider=lambda: footprint[0],
        clear_cache=lambda: cleared.append(True),
    )
    assert governor.sample().level == "warning"
    session = store.create_session()
    governor.on_session_removed(session.session_id, "test")
    assert verifier.resets == 0
    store.close_session(session.session_id)
    governor.on_session_removed(session.session_id, "test")
    assert verifier.resets == 1
    assert cleared == [True]
    footprint[0] = 20
    governor.sample()
    assert governor.draining
    footprint[0] = 30
    governor.sample()
    assert governor.unhealthy


def test_runtime_health_endpoint_reports_unhealthy_marker(tmp_path):
    unhealthy = tmp_path / "unhealthy.json"
    live = DecodeLiveness(tmp_path / "live.json", unhealthy_path=unhealthy)
    governor = PrimaryMemoryGovernor(
        SessionStore(capacity=1),
        Verifier(),
        warning_bytes=10,
        drain_bytes=20,
        unhealthy_bytes=30,
        memory_provider=lambda: (0, 0, 0),
        footprint_provider=lambda: 0,
    )
    client = TestClient(create_runtime_health_app(live, governor))
    assert client.get("/healthz").status_code == 200
    unhealthy.write_text("{}")
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
