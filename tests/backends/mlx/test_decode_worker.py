"""Linux-runnable protocol/recovery tests for the isolated MLX worker."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_engine.backends.mlx.decode_worker import (
    PROTOCOL_VERSION,
    DecodeWorkerClient,
    DecodeWorkerConfig,
    DecodeWorkerSessionClosed,
    _recv_frame,
    _send_frame,
)
from inference_engine.distributed.capability import CacheCompatibility
from inference_engine.session.coordinator import OperationCancelledError


class FakeDecodeVerifier:
    dimensions = (3, 2, 8)

    def __init__(self, model_id: str = "fake") -> None:
        self.model_id = model_id
        self.model = SimpleNamespace()
        self.stats = SimpleNamespace(weight_bytes=1234)
        self.cached_token_sequence: list[int] = []
        self.next_global_position = 0

    def spawn(self):
        return type(self)(self.model_id)

    def prefill(self, tokens):
        self.cached_token_sequence = list(tokens)[-8:]
        self.next_global_position = len(tokens)

    def append_accepted_tokens(self, tokens):
        if -999 in tokens:
            time.sleep(5)
        self.cached_token_sequence = (self.cached_token_sequence + list(tokens))[-8:]
        self.next_global_position += len(tokens)
        marker = self.model_id.removeprefix("crash-after-append:")
        if marker != self.model_id and not Path(marker).exists():
            Path(marker).write_text("crashed")
            os._exit(17)

    def greedy_next_token_id(self):
        return ((self.cached_token_sequence or [0])[-1] + 1) % 1000

    def live_kv_bytes(self):
        return len(self.cached_token_sequence) * 32

    def import_snapshot(self, payload, _compatibility):
        tokens = json.loads(payload)["tokens"]
        self.prefill(tokens)

    def export_snapshot(self, _compatibility):
        return json.dumps({"tokens": self.cached_token_sequence}).encode()

    def logits_sha256(self):
        next_token = ((self.cached_token_sequence or [0])[-1] + 1) % 1000
        return hashlib.sha256(f"fake-int64:[1]:{next_token}".encode()).hexdigest()


def build_fake_verifier(config):
    return FakeDecodeVerifier(str(config["model_id"]))


def _client(
    tmp_path,
    *,
    model_id="fake",
    timeout=2.0,
    sink_size=2,
    window_size=6,
):
    return DecodeWorkerClient(DecodeWorkerConfig(
        model_id=model_id,
        sink_size=sink_size,
        window_size=window_size,
        request_timeout_s=timeout,
        startup_timeout_s=10.0,
        socket_path=str(tmp_path / "decode.sock"),
        verifier_factory=(
            "tests.backends.mlx.test_decode_worker:build_fake_verifier"
        ),
    ))


def test_protocol_frame_preserves_binary_snapshot_without_pickle():
    left, right = socket.socketpair()
    try:
        payload = b"\x00KPKV1\xffsnapshot"
        _send_frame(left, {
            "version": PROTOCOL_VERSION,
            "request_id": "request-1",
            "operation": "ImportSnapshot",
        }, payload)
        header, received = _recv_frame(right)
        assert header == {
            "operation": "ImportSnapshot",
            "request_id": "request-1",
            "version": PROTOCOL_VERSION,
        }
        assert received == payload
    finally:
        left.close()
        right.close()


def test_protocol_health_and_all_operations(tmp_path):
    client = _client(tmp_path)
    try:
        health = client.health()
        assert health["protocol_version"] == PROTOCOL_VERSION
        assert client.dimensions == (3, 2, 8)

        session = client.get("s1")
        session.prefill([1, 2, 3])
        session.append_accepted_tokens([4])
        assert session.generate_step() == 5
        assert session.cached_token_sequence == [1, 2, 3, 4, 5]

        compatibility = CacheCompatibility(model_id="fake")
        session.import_snapshot(
            json.dumps({"tokens": [10, 20]}).encode(),
            compatibility,
        )
        assert session.next_global_position == 2
        assert session.generate_step() == 21
        client.close_session("s1")
        assert client.health()["session_count"] == 0
    finally:
        client.close()


def test_worker_proxy_sink_window_slice_matches_child_layout(tmp_path):
    client = _client(tmp_path)
    try:
        session = client.get("slice")
        short = [1, 2, 3]
        assert session._sink_window_slice(short) == short
        assert session._sink_window_slice(short) is not short

        sequence = list(range(20))
        assert session._sink_window_slice(sequence) == (
            sequence[:2] + sequence[-6:]
        )
    finally:
        client.close()


def test_worker_proxy_sink_window_slice_supports_zero_window(tmp_path):
    client = _client(tmp_path, window_size=0)
    try:
        session = client.get("sink-only")
        assert session._sink_window_slice(list(range(10))) == [0, 1]
    finally:
        client.close()


def test_worker_matches_in_process_greedy_parity(tmp_path):
    expected = FakeDecodeVerifier()
    expected.prefill([7, 8, 9])
    baseline = []
    for _ in range(6):
        token = expected.greedy_next_token_id()
        expected.append_accepted_tokens([token])
        baseline.append(token)

    client = _client(tmp_path)
    try:
        remote = client.get("parity")
        remote.prefill([7, 8, 9])
        assert [remote.generate_step() for _ in range(6)] == baseline
    finally:
        client.close()


def test_cancellation_hard_kills_worker_and_preserves_checkpoint(tmp_path):
    client = _client(tmp_path, timeout=10.0)
    try:
        session = client.get("cancel")
        session.prefill([1, 2])
        old_pid = client.pid
        cancel = threading.Event()
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        with pytest.raises(OperationCancelledError):
            session.append_accepted_tokens([-999], cancel_event=cancel)
        timer.cancel()

        # A later request restarts and restores only acknowledged tokens.
        assert session.generate_step() == 3
        assert client.pid != old_pid
        assert session.cached_token_sequence == [1, 2, 3]
    finally:
        client.close()


def test_crash_recovers_checkpoint_and_retries_current_step(tmp_path):
    marker = tmp_path / "crashed"
    client = _client(tmp_path, model_id=f"crash-after-append:{marker}")
    try:
        session = client.get("recover")
        session.prefill([40, 41])
        old_pid = client.pid
        assert session.generate_step() == 42
        assert marker.exists()
        assert client.restart_count == 1
        assert client.pid != old_pid
        assert session.cached_token_sequence == [40, 41, 42]
    finally:
        client.close()


def test_snapshot_plus_post_checkpoint_replay_survives_hard_kill(tmp_path):
    client = _client(tmp_path)
    try:
        session = client.get("snapshot")
        session.import_snapshot(
            json.dumps({"tokens": [10, 11, 12]}).encode(),
            CacheCompatibility(model_id="fake"),
        )
        session.append_accepted_tokens([13, 14])
        old_pid = client.pid
        client.hard_kill()
        assert session.generate_step() == 15
        assert client.pid != old_pid
        assert session.cached_token_sequence == [10, 11, 12, 13, 14, 15]
    finally:
        client.close()


def test_close_during_snapshot_import_waits_for_checkpoint_publication(tmp_path):
    client = _client(tmp_path)
    imported = threading.Event()
    release_import = threading.Event()
    close_done = threading.Event()
    failures = []
    try:
        session = client.get("import-close-race")
        original_request = client._session_request

        def paused_request(session_id, operation, arguments, payload=b"", **kwargs):
            state = original_request(
                session_id, operation, arguments, payload, **kwargs,
            )
            if operation == "ImportSnapshot":
                imported.set()
                assert release_import.wait(timeout=2.0)
            return state

        client._session_request = paused_request

        def run_import():
            try:
                session.import_snapshot(
                    json.dumps({"tokens": [10, 11, 12]}).encode(),
                    CacheCompatibility(model_id="fake"),
                )
            except BaseException as exc:
                failures.append(exc)

        def run_close():
            try:
                client.close_session(session.session_id)
            except BaseException as exc:
                failures.append(exc)
            finally:
                close_done.set()

        import_thread = threading.Thread(target=run_import)
        close_thread = threading.Thread(target=run_close)
        import_thread.start()
        assert imported.wait(timeout=2.0)
        close_thread.start()

        # The child has acknowledged ImportSnapshot, but router checkpoint
        # publication is deliberately paused. Cleanup must remain blocked.
        assert not close_done.wait(timeout=0.1)
        assert session.session_id in client._checkpoints

        release_import.set()
        import_thread.join(timeout=2.0)
        close_thread.join(timeout=2.0)
        assert not import_thread.is_alive()
        assert not close_thread.is_alive()
        assert failures == []
        assert session.session_id not in client._checkpoints
        assert session.session_id not in client._proxies
        assert client.health()["session_count"] == 0
    finally:
        release_import.set()
        client.close()


def test_closed_proxy_fails_with_explicit_session_error(tmp_path):
    client = _client(tmp_path)
    try:
        session = client.get("closed")
        session.prefill([1, 2])
        client.close_session(session.session_id)
        with pytest.raises(DecodeWorkerSessionClosed, match="is closed"):
            session.reset()
    finally:
        client.close()


def test_recycle_lazily_restores_other_sessions(tmp_path):
    client = _client(tmp_path)
    try:
        first = client.get("first")
        second = client.get("second")
        first.prefill([1, 2])
        second.prefill([90, 91])

        client.hard_kill()
        assert first.generate_step() == 3
        # First's recovery restarted the process, which also lost second's KV.
        # Second must notice the worker generation and restore independently.
        assert second.generate_step() == 92
        assert second.cached_token_sequence == [90, 91, 92]
    finally:
        client.close()


def test_acceptance_snapshot_hang_and_kv_restore_interfaces(tmp_path):
    client = _client(tmp_path, timeout=0.2)
    try:
        snapshot = client.acceptance_snapshot(
            runtime_pid=100,
            process_footprint_bytes=200,
            active_sessions=3,
            active_generations=1,
        )
        assert snapshot == {
            "runtime_pid": 100,
            "worker_pid": client.pid,
            "worker_restart_count": 0,
            "process_footprint_bytes": 200,
            "active_sessions": 3,
            "active_generations": 1,
        }

        old_pid = client.pid
        session = client.get("hang")
        session.prefill([1, 2])
        assert client.inject_hang(old_pid)["accepted"] is True
        session.append_accepted_tokens([3])
        assert session.generate_step() == 4
        assert client.pid != old_pid
        assert client.restart_count == 1

        parity = client.kv_restore_parity([10, 11, 12])
        assert parity["baseline_first_token_id"] == 13
        assert parity["restored_first_token_id"] == 13
        assert parity["baseline_logits_sha256"] == parity["restored_logits_sha256"]
        assert parity["restore_source"] == "allens_kv+proof_checkpoint"
    finally:
        client.close()
