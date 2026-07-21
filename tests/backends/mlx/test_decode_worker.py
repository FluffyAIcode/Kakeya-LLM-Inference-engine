"""Linux-runnable protocol/recovery tests for the isolated MLX worker."""

from __future__ import annotations

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


def build_fake_verifier(config):
    return FakeDecodeVerifier(str(config["model_id"]))


def _client(tmp_path, *, model_id="fake", timeout=2.0):
    return DecodeWorkerClient(DecodeWorkerConfig(
        model_id=model_id,
        sink_size=2,
        window_size=6,
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

