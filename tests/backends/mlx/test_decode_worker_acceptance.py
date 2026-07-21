from __future__ import annotations

import socket

from inference_engine.backends.mlx.decode_worker import _recv_frame, _send_frame
from inference_engine.backends.mlx.decode_worker_acceptance import (
    DecodeWorkerAcceptanceServer,
    normalize_control_socket_path,
)


class FakeWorker:
    pid = 222
    restart_count = 3

    def acceptance_snapshot(self, **values):
        return {
            **values,
            "worker_pid": self.pid,
            "worker_restart_count": self.restart_count,
        }

    def inject_hang(self, expected_worker_pid):
        return {
            "accepted": expected_worker_pid == self.pid,
            "worker_pid": self.pid,
        }

    def kv_restore_parity(self, prompt_token_ids):
        return {
            "baseline_first_token_id": prompt_token_ids[-1] + 1,
            "restored_first_token_id": prompt_token_ids[-1] + 1,
            "baseline_logits_sha256": "abc",
            "restored_logits_sha256": "abc",
            "restore_source": "allens_kv+proof_checkpoint",
        }


def _call(path: str, operation: str, payload: dict):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(path)
        _send_frame(
            connection,
            {"schema_version": 1, "operation": operation, "payload": payload},
        )
        response, _ = _recv_frame(connection)
        return response
    finally:
        connection.close()


def test_acceptance_control_server_routes_local_operations(tmp_path):
    worker = FakeWorker()
    server = DecodeWorkerAcceptanceServer(
        socket_path=str(tmp_path / "acceptance.sock"),
        worker=worker,
        runtime_pid=111,
        active_sessions=lambda: 2,
        active_generations=lambda: 1,
        process_footprint=lambda: 4096,
    )
    server.start()
    try:
        snapshot = _call(server.socket_path, "snapshot", {})
        assert snapshot["ok"] is True
        assert snapshot["data"] == {
            "runtime_pid": 111,
            "worker_pid": 222,
            "worker_restart_count": 3,
            "process_footprint_bytes": 4096,
            "active_sessions": 2,
            "active_generations": 1,
        }
        assert _call(
            server.socket_path,
            "inject_hang",
            {"phase": "next_forward", "expected_worker_pid": 222},
        )["data"]["accepted"] is True
        parity = _call(
            server.socket_path,
            "kv_restore_parity",
            {"prompt_token_ids": [7, 8]},
        )
        assert parity["data"]["restored_first_token_id"] == 9
        error = _call(server.socket_path, "unknown", {})
        assert error["ok"] is False
        assert "unknown acceptance operation" in error["error"]
    finally:
        server.close()
    assert not (tmp_path / "acceptance.sock").exists()


def test_acceptance_control_validation_and_long_path(tmp_path):
    worker = FakeWorker()
    server = DecodeWorkerAcceptanceServer(
        socket_path=str(tmp_path / "unused.sock"),
        worker=worker,
        runtime_pid=1,
        active_sessions=lambda: 0,
        active_generations=lambda: 0,
        process_footprint=lambda: 0,
    )
    try:
        server.dispatch(
            "inject_hang",
            {"phase": "wrong", "expected_worker_pid": 222},
        )
    except ValueError as exc:
        assert "next_forward" in str(exc)
    else:
        raise AssertionError("invalid phase was accepted")
    long_path = "/" + "x" * 200
    assert normalize_control_socket_path(long_path).endswith(".sock")
