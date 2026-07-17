from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "scripts" / "start_grpc_runtime_server.py"


def test_network_http_runs_outside_blocking_grpc_event_loop():
    source = RUNTIME.read_text()
    assert 'name="kakeya-network-http"' in source
    assert "target=http_server.run" in source
    assert "http_thread.start()" in source
    assert "asyncio.create_task(http_server.serve())" not in source
    assert "asyncio.to_thread(http_thread.join" in source
