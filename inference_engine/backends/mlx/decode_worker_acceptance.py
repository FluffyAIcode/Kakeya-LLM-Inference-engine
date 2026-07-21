"""Local-only acceptance controls for the isolated MLX decode worker."""

from __future__ import annotations

import hashlib
import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from inference_engine.backends.mlx.decode_worker import (
    DecodeWorkerClient,
    _recv_frame,
    _send_frame,
)


def normalize_control_socket_path(path: str) -> str:
    if not path:
        return ""
    if len(os.fsencode(path)) < 100:
        return path
    return str(
        Path(tempfile.gettempdir())
        / f"kdw-accept-{hashlib.sha256(os.fsencode(path)).hexdigest()[:20]}.sock"
    )


class DecodeWorkerAcceptanceServer:
    """Serve fault injection and observations on a mode-0600 UDS."""

    def __init__(
        self,
        *,
        socket_path: str,
        worker: DecodeWorkerClient,
        runtime_pid: int,
        active_sessions: Callable[[], int],
        active_generations: Callable[[], int],
        process_footprint: Callable[[], int],
    ) -> None:
        normalized = normalize_control_socket_path(socket_path)
        if not normalized:
            raise ValueError("acceptance control socket path must not be empty")
        self.socket_path = normalized
        self._worker = worker
        self._runtime_pid = int(runtime_pid)
        self._active_sessions = active_sessions
        self._active_generations = active_generations
        self._process_footprint = process_footprint
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        server.listen(8)
        server.settimeout(0.2)
        self._server = server
        self._thread = threading.Thread(
            target=self._serve,
            name="kakeya-decode-acceptance",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        server = self._server
        self._server = None
        if server is not None:
            server.close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        Path(self.socket_path).unlink(missing_ok=True)

    def _serve(self) -> None:
        while not self._stop.is_set():
            server = self._server
            if server is None:
                return
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                operation = ""
                try:
                    request, _ = _recv_frame(connection)
                    if int(request.get("schema_version", -1)) != 1:
                        raise ValueError("acceptance protocol version mismatch")
                    operation = str(request.get("operation", ""))
                    data = self.dispatch(
                        operation,
                        dict(request.get("payload", {})),
                    )
                    _send_frame(
                        connection,
                        {
                            "schema_version": 1,
                            "operation": operation,
                            "ok": True,
                            "data": data,
                        },
                    )
                except BaseException as exc:
                    try:
                        _send_frame(
                            connection,
                            {
                                "schema_version": 1,
                                "operation": operation,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "data": {},
                            },
                        )
                    except OSError:
                        pass

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation == "snapshot":
            return self._worker.acceptance_snapshot(
                runtime_pid=self._runtime_pid,
                process_footprint_bytes=self._process_footprint(),
                active_sessions=self._active_sessions(),
                active_generations=self._active_generations(),
            )
        if operation == "inject_hang":
            if payload.get("phase") != "next_forward":
                raise ValueError("inject_hang phase must be 'next_forward'")
            return self._worker.inject_hang(int(payload["expected_worker_pid"]))
        if operation == "kv_restore_parity":
            return self._worker.kv_restore_parity(
                [int(token) for token in payload["prompt_token_ids"]]
            )
        raise ValueError(f"unknown acceptance operation {operation!r}")
