"""Feature-flagged MLX decode process with versioned local UDS IPC.

The router imports this module without importing MLX.  The child process loads
the model once, owns all session K/V, and executes one request at a time.  A
request is a length-prefixed JSON header plus an optional opaque binary payload
(used for allens-compatible K/V snapshots); pickle is never used on the wire.

Protocol v1 operations:

``Init`` creates/replaces a session and optionally prefills tokens;
``ImportSnapshot`` restores a portable prefill checkpoint; ``Append`` commits
accepted tokens; ``GenerateStep`` atomically chooses and commits one greedy
token; ``Close`` drops session K/V; and ``Health`` reports process/model state.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import multiprocessing
import os
import select
import signal
import socket
import struct
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from inference_engine.session.coordinator import OperationCancelledError

PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 2 << 30


class DecodeWorkerError(RuntimeError):
    """Base class for decode-worker failures."""


class DecodeWorkerUnavailable(DecodeWorkerError):
    """The worker exited, hung, or broke its local transport."""


class DecodeWorkerRemoteError(DecodeWorkerError):
    """The worker completed a request with an application error."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


@dataclass(frozen=True)
class DecodeWorkerConfig:
    model_id: str
    sink_size: int = 4
    window_size: int = 64
    request_timeout_s: float = 120.0
    startup_timeout_s: float = 180.0
    socket_path: str = ""
    verifier_factory: str = (
        "inference_engine.backends.mlx.decode_worker:_build_mlx_verifier"
    )


@dataclass
class ProofCheckpoint:
    """Router-owned restart source.

    ``snapshot`` is an immutable allens-compatible prefill snapshot.  Tokens
    committed after that boundary are retained in ``replay_token_ids``.  When
    no snapshot exists, the replay list is the complete committed history.
    The current request is added only after its response arrives, making a
    retry after an ambiguous hard kill idempotent.
    """

    snapshot: bytes | None = None
    compatibility: dict[str, Any] | None = None
    replay_token_ids: list[int] = field(default_factory=list)
    initialized: bool = False


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("decode worker closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(
    sock: socket.socket,
    header: dict[str, Any],
    payload: bytes = b"",
) -> None:
    body = bytes(payload)
    encoded = json.dumps(
        {**header, "payload_length": len(body)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > MAX_HEADER_BYTES or len(body) > MAX_PAYLOAD_BYTES:
        raise ValueError("decode worker frame exceeds protocol limits")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded + body)


def _recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if header_size <= 0 or header_size > MAX_HEADER_BYTES:
        raise ValueError("invalid decode worker header length")
    header = json.loads(_recv_exact(sock, header_size))
    payload_size = int(header.pop("payload_length", 0))
    if payload_size < 0 or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError("invalid decode worker payload length")
    return header, _recv_exact(sock, payload_size)


def _resolve(path: str) -> Callable[..., Any]:
    module_name, symbol = path.split(":", 1)
    return getattr(importlib.import_module(module_name), symbol)


def _build_mlx_verifier(config: dict[str, Any]):
    """Child-only default factory; importing it is what imports MLX."""
    import torch
    from kv_cache_proposer.verifier import VerifierConfig
    from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier

    return MLXSinkWindowVerifier(VerifierConfig(
        model_id=str(config["model_id"]),
        dtype=torch.bfloat16,
        device="cpu",
        sink_size=int(config["sink_size"]),
        window_size=int(config["window_size"]),
    ))


def _model_dimensions(verifier: Any) -> tuple[int, int, int]:
    declared = getattr(verifier, "dimensions", None)
    if declared is not None:
        return tuple(int(value) for value in declared)
    try:
        cfg = getattr(verifier.model, "config", None) or getattr(
            verifier.model, "args", None
        )
        cfg = getattr(cfg, "text_config", None) or cfg
        return (
            int(getattr(cfg, "num_hidden_layers")),
            int(
                getattr(cfg, "num_key_value_heads", None)
                or getattr(cfg, "num_attention_heads")
            ),
            int(
                getattr(cfg, "head_dim", None)
                or (cfg.hidden_size // cfg.num_attention_heads)
            ),
        )
    except (AttributeError, TypeError):
        from inference_engine.backends.mlx.cross_model_dlm_verifier import (
            per_layer_kv_geometry,
            resolve_mlx_text_model,
        )
        geometry = per_layer_kv_geometry(resolve_mlx_text_model(verifier.model))
        if not geometry:
            raise
        return (
            len(geometry),
            max(item[0] for item in geometry),
            max(item[1] for item in geometry),
        )


def _state(verifier: Any) -> dict[str, Any]:
    cached = [int(token) for token in verifier.cached_token_sequence]
    kv_bytes = getattr(verifier, "live_kv_bytes", lambda: 0)()
    return {
        "cached_token_ids": cached,
        "k_seq_length": len(cached),
        "next_global_position": int(verifier.next_global_position),
        "kv_live_bytes": int(kv_bytes),
    }


def _canonical_logits_sha256(verifier: Any) -> str:
    """Hash the current last-token logits without losing dtype metadata."""
    custom = getattr(verifier, "logits_sha256", None)
    if custom is not None:
        return str(custom())
    logits = getattr(verifier, "_next_token_logits_mx", None)
    if logits is None:
        logits = verifier.next_token_logits
    if logits is None:
        raise RuntimeError("decode worker session has no next-token logits")
    from inference_engine.distributed.tensor_codec import (
        mlx_to_wire,
        to_proto_fields,
        torch_to_wire,
    )

    wire = (
        torch_to_wire(logits)
        if hasattr(logits, "detach")
        else mlx_to_wire(logits)
    )
    dtype, shape, data = to_proto_fields(wire)
    metadata = json.dumps(
        {"dtype": dtype, "shape": [int(value) for value in shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(metadata + b"\0" + data).hexdigest()


class _WorkerRuntime:
    def __init__(self, config: dict[str, Any], factory_path: str) -> None:
        self.template = _resolve(factory_path)(config)
        self.sessions: dict[str, Any] = {}
        self.started_at = time.time()
        self._hang_next_forward = False
        try:
            self.dimensions = _model_dimensions(self.template)
        except (AttributeError, TypeError):
            # Test factories and future adapters may expose dimensions directly.
            self.dimensions = tuple(
                int(value) for value in getattr(
                    self.template, "dimensions", (1, 1, 1)
                )
            )

    def _new_session(self):
        spawn = getattr(self.template, "spawn", None)
        if spawn is None:
            raise RuntimeError("decode worker verifier must implement spawn()")
        return spawn()

    def _before_forward(self) -> None:
        if not self._hang_next_forward:
            return
        self._hang_next_forward = False
        while True:
            time.sleep(60.0)

    def dispatch(
        self,
        operation: str,
        session_id: str,
        arguments: dict[str, Any],
        payload: bytes,
    ) -> dict[str, Any] | tuple[dict[str, Any], bytes]:
        if operation == "Health":
            memory = (0, 0, 0)
            try:
                import mlx.core as mx
                memory = (
                    int(mx.get_active_memory()),
                    int(mx.get_cache_memory()),
                    int(mx.get_peak_memory()),
                )
            except (ImportError, AttributeError, RuntimeError):
                pass
            return {
                "pid": os.getpid(),
                "protocol_version": PROTOCOL_VERSION,
                "session_count": len(self.sessions),
                "uptime_seconds": time.time() - self.started_at,
                "num_layers": self.dimensions[0],
                "num_kv_heads": self.dimensions[1],
                "head_dim": self.dimensions[2],
                "weight_bytes": int(
                    getattr(getattr(self.template, "stats", None), "weight_bytes", 0)
                ),
                "mlx_active_bytes": memory[0],
                "mlx_cache_bytes": memory[1],
                "mlx_peak_bytes": memory[2],
            }
        if operation == "InjectHang":
            expected_pid = int(arguments["expected_worker_pid"])
            if expected_pid != os.getpid():
                return {"accepted": False, "worker_pid": os.getpid()}
            if self._hang_next_forward:
                return {"accepted": False, "worker_pid": os.getpid()}
            self._hang_next_forward = True
            return {"accepted": True, "worker_pid": os.getpid()}
        if not session_id:
            raise ValueError(f"{operation} requires session_id")
        if operation == "Init":
            verifier = self._new_session()
            self.sessions[session_id] = verifier
            tokens = [int(token) for token in arguments.get("token_ids", ())]
            if tokens:
                self._before_forward()
                verifier.prefill(tokens)
            return _state(verifier)
        if operation == "Close":
            existed = self.sessions.pop(session_id, None) is not None
            return {"closed": existed}
        try:
            verifier = self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown decode session {session_id!r}") from exc
        if operation == "ImportSnapshot":
            custom_import = getattr(verifier, "import_snapshot", None)
            if custom_import is not None:
                custom_import(payload, arguments["compatibility"])
                return _state(verifier)
            from inference_engine.backends.mlx.prefill_snapshot import (
                import_mlx_prefill_snapshot,
            )
            from inference_engine.distributed.capability import CacheCompatibility

            verifier.reset()
            imported = import_mlx_prefill_snapshot(
                payload,
                verifier.cache,
                compatibility=CacheCompatibility(**arguments["compatibility"]),
            )
            if imported.next_token_logits is None:
                raise ValueError("decode snapshot is missing continuation logits")
            verifier.cached_token_sequence = list(imported.cached_token_ids)
            verifier.next_global_position = int(imported.token_count)
            verifier.cache_logical_size = len(imported.cached_token_ids)
            if hasattr(verifier, "_set_next_token_logits_mx") and not hasattr(
                imported.next_token_logits, "detach"
            ):
                verifier._set_next_token_logits_mx(imported.next_token_logits)
            else:
                verifier.next_token_logits = imported.next_token_logits
            return {
                **_state(verifier),
                "token_count": imported.token_count,
                "block_hash": imported.block_hash.hex(),
            }
        if operation == "ExportSnapshot":
            compatibility = dict(arguments["compatibility"])
            custom_export = getattr(verifier, "export_snapshot", None)
            if custom_export is not None:
                snapshot = bytes(custom_export(compatibility))
            else:
                from inference_engine.backends.mlx.prefill_snapshot import (
                    export_mlx_prefill_snapshot,
                )
                from inference_engine.distributed.capability import CacheCompatibility

                next_token_logits = getattr(
                    verifier, "_next_token_logits_mx", None
                )
                if next_token_logits is None:
                    next_token_logits = verifier.next_token_logits
                snapshot = export_mlx_prefill_snapshot(
                    verifier.cache,
                    token_count=int(verifier.next_global_position),
                    cached_token_ids=verifier.cached_token_sequence,
                    compatibility=CacheCompatibility(**compatibility),
                    next_token_logits=next_token_logits,
                )
            return (
                {
                    **_state(verifier),
                    "first_token_id": int(verifier.greedy_next_token_id()),
                    "logits_sha256": _canonical_logits_sha256(verifier),
                },
                snapshot,
            )
        if operation == "Append":
            tokens = [int(token) for token in arguments["token_ids"]]
            if not tokens:
                raise ValueError("Append token_ids must be non-empty")
            self._before_forward()
            verifier.append_accepted_tokens(tokens)
            return _state(verifier)
        if operation == "GenerateStep":
            token_id = int(verifier.greedy_next_token_id())
            input_logits_sha256 = _canonical_logits_sha256(verifier)
            self._before_forward()
            verifier.append_accepted_tokens([token_id])
            return {
                **_state(verifier),
                "token_id": token_id,
                "input_logits_sha256": input_logits_sha256,
            }
        raise ValueError(f"unknown decode worker operation {operation!r}")


def _serve_worker(
    config: dict[str, Any],
    socket_path: str,
    factory_path: str,
) -> None:
    runtime = _WorkerRuntime(config, factory_path)
    path = Path(socket_path)
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen(1)
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        request, payload = _recv_frame(connection)
                    except EOFError:
                        break
                    request_id = str(request.get("request_id", ""))
                    try:
                        if int(request.get("version", -1)) != PROTOCOL_VERSION:
                            raise ValueError("decode worker protocol version mismatch")
                        result = runtime.dispatch(
                            str(request["operation"]),
                            str(request.get("session_id", "")),
                            dict(request.get("arguments", {})),
                            payload,
                        )
                        response_payload = b""
                        if isinstance(result, tuple):
                            result, response_payload = result
                        _send_frame(connection, {
                            "version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "ok": True,
                            "result": result,
                        }, response_payload)
                    except BaseException as exc:
                        _send_frame(connection, {
                            "version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
    finally:
        server.close()
        path.unlink(missing_ok=True)


class DecodeWorkerClient:
    """Serialized router-side process owner and recovery coordinator."""

    def __init__(self, config: DecodeWorkerConfig) -> None:
        if config.request_timeout_s <= 0 or config.startup_timeout_s <= 0:
            raise ValueError("decode worker timeouts must be > 0")
        self.config = config
        requested_socket = config.socket_path or str(
            Path(tempfile.gettempdir()) / f"kakeya-decode-{uuid.uuid4().hex}.sock"
        )
        # Darwin's sockaddr_un path is only 104 bytes. Pytest temp roots and
        # launchd working directories can exceed it, so deterministically
        # shorten an explicit path while preserving per-path uniqueness.
        self.socket_path = (
            requested_socket
            if len(os.fsencode(requested_socket)) < 100
            else str(
                Path(tempfile.gettempdir())
                / (
                    "kdw-"
                    + hashlib.sha256(os.fsencode(requested_socket)).hexdigest()[:24]
                    + ".sock"
                )
            )
        )
        self._process: multiprocessing.Process | None = None
        self._socket: socket.socket | None = None
        self._lock = threading.RLock()
        self._checkpoints: dict[str, ProofCheckpoint] = {}
        self._proxies: dict[str, DecodeWorkerSession] = {}
        self._generation = 0
        self._restored_generation: dict[str, int] = {}
        self.restart_count = 0
        self._health: dict[str, Any] = {}
        self.start()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return (
            int(self._health["num_layers"]),
            int(self._health["num_kv_heads"]),
            int(self._health["head_dim"]),
        )

    def start(self) -> None:
        with self._lock:
            self._stop_locked()
            Path(self.socket_path).unlink(missing_ok=True)
            ctx = multiprocessing.get_context("spawn")
            payload = {
                "model_id": self.config.model_id,
                "sink_size": self.config.sink_size,
                "window_size": self.config.window_size,
            }
            self._process = ctx.Process(
                target=_serve_worker,
                args=(payload, self.socket_path, self.config.verifier_factory),
                name="kakeya-mlx-decode-worker",
                daemon=True,
            )
            self._process.start()
            deadline = time.monotonic() + self.config.startup_timeout_s
            last_error: BaseException | None = None
            while time.monotonic() < deadline:
                if not self._process.is_alive():
                    raise DecodeWorkerUnavailable(
                        f"decode worker exited during startup "
                        f"(exitcode={self._process.exitcode})"
                    )
                try:
                    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    connection.connect(self.socket_path)
                    self._socket = connection
                    self._health = self._request_locked(
                        "Health", "", {}, b"", timeout_s=5.0
                    )
                    self._generation += 1
                    return
                except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                    last_error = exc
                    time.sleep(0.02)
            self._stop_locked()
            raise DecodeWorkerUnavailable(
                f"decode worker startup timed out: {last_error}"
            )

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._health = self._request_locked("Health", "", {}, b"")
            return dict(self._health)

    def inject_hang(self, expected_worker_pid: int) -> dict[str, Any]:
        with self._lock:
            return self._request_locked(
                "InjectHang",
                "",
                {"expected_worker_pid": int(expected_worker_pid)},
                b"",
            )

    def acceptance_snapshot(
        self,
        *,
        runtime_pid: int,
        process_footprint_bytes: int,
        active_sessions: int,
        active_generations: int,
    ) -> dict[str, Any]:
        return {
            "runtime_pid": int(runtime_pid),
            "worker_pid": int(self.pid or 0),
            "worker_restart_count": int(self.restart_count),
            "process_footprint_bytes": int(process_footprint_bytes),
            "active_sessions": int(active_sessions),
            "active_generations": int(active_generations),
        }

    def kv_restore_parity(self, prompt_token_ids: list[int]) -> dict[str, Any]:
        """Exercise an Allens snapshot plus router proof-checkpoint restore."""
        tokens = [int(token) for token in prompt_token_ids]
        if not tokens:
            raise ValueError("kv_restore_parity requires prompt_token_ids")
        from inference_engine.distributed.capability import CacheCompatibility

        compatibility = CacheCompatibility(
            model_id=self.config.model_id,
            sink_size=self.config.sink_size,
            window_size=self.config.window_size,
        )
        compat = asdict(compatibility)
        session_id = f"acceptance-{uuid.uuid4().hex}"
        session = self.get(session_id)
        try:
            session.prefill(tokens)
            with self._lock:
                baseline, snapshot = self._request_with_payload_locked(
                    "ExportSnapshot",
                    session_id,
                    {"compatibility": compat},
                    b"",
                )
                checkpoint = self._checkpoints[session_id]
                checkpoint.snapshot = snapshot
                checkpoint.compatibility = compat
                checkpoint.replay_token_ids = []
                checkpoint.initialized = True
                self._restored_generation[session_id] = self._generation
            self.hard_kill()
            restored = self._session_request(
                session_id,
                "GenerateStep",
                {},
            )
            return {
                "baseline_first_token_id": int(baseline["first_token_id"]),
                "restored_first_token_id": int(restored["token_id"]),
                "baseline_logits_sha256": str(baseline["logits_sha256"]),
                "restored_logits_sha256": str(
                    restored["input_logits_sha256"]
                ),
                "restore_source": "allens_kv+proof_checkpoint",
            }
        finally:
            self.close_session(session_id)

    def get(self, session_id: str) -> "DecodeWorkerSession":
        with self._lock:
            proxy = self._proxies.get(session_id)
            if proxy is None:
                proxy = DecodeWorkerSession(self, session_id)
                self._proxies[session_id] = proxy
                self._checkpoints.setdefault(session_id, ProofCheckpoint())
            return proxy

    session = get

    def close_session(self, session_id: str) -> None:
        with self._lock:
            try:
                self._request_locked("Close", session_id, {}, b"", timeout_s=2.0)
            except DecodeWorkerError:
                pass
            self._checkpoints.pop(session_id, None)
            self._proxies.pop(session_id, None)
            self._restored_generation.pop(session_id, None)

    def k_seq_length(self, session: Any) -> int:
        return self.get(session.session_id).k_seq_length(session)

    def kv_live_bytes(self, session: Any) -> int:
        return self.get(session.session_id).kv_live_bytes(session)

    def memory_bytes(self) -> tuple[int, int, int]:
        health = self.health()
        return (
            int(health.get("mlx_active_bytes", 0)),
            int(health.get("mlx_cache_bytes", 0)),
            int(health.get("mlx_peak_bytes", 0)),
        )

    def hard_kill(self) -> None:
        with self._lock:
            self._stop_locked(hard=True)

    def close(self) -> None:
        with self._lock:
            self._stop_locked()
            Path(self.socket_path).unlink(missing_ok=True)

    def _stop_locked(self, *, hard: bool = False) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        process = self._process
        self._process = None
        if process is not None and process.is_alive():
            if hard and process.pid:
                os.kill(process.pid, signal.SIGKILL)
            else:
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive() and process.pid:
                os.kill(process.pid, signal.SIGKILL)
                process.join(timeout=1.0)

    def _request_locked(
        self,
        operation: str,
        session_id: str,
        arguments: dict[str, Any],
        payload: bytes,
        *,
        timeout_s: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        result, _ = self._request_with_payload_locked(
            operation,
            session_id,
            arguments,
            payload,
            timeout_s=timeout_s,
            cancel_event=cancel_event,
        )
        return result

    def _request_with_payload_locked(
        self,
        operation: str,
        session_id: str,
        arguments: dict[str, Any],
        payload: bytes,
        *,
        timeout_s: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        if self._socket is None:
            raise DecodeWorkerUnavailable("decode worker is not connected")
        request_id = uuid.uuid4().hex
        try:
            _send_frame(self._socket, {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "operation": operation,
                "session_id": session_id,
                "arguments": arguments,
            }, payload)
            deadline = time.monotonic() + (
                self.config.request_timeout_s if timeout_s is None else timeout_s
            )
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._stop_locked(hard=True)
                    raise OperationCancelledError(
                        f"decode worker {operation} cancelled"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_locked(hard=True)
                    raise DecodeWorkerUnavailable(
                        f"decode worker {operation} timed out"
                    )
                ready, _, _ = select.select(
                    [self._socket], [], [], min(0.05, remaining)
                )
                if ready:
                    break
            response, response_payload = _recv_frame(self._socket)
        except OperationCancelledError:
            raise
        except (OSError, EOFError, ValueError, json.JSONDecodeError) as exc:
            self._stop_locked(hard=True)
            raise DecodeWorkerUnavailable(str(exc)) from exc
        if (
            response.get("version") != PROTOCOL_VERSION
            or response.get("request_id") != request_id
        ):
            self._stop_locked(hard=True)
            raise DecodeWorkerUnavailable("decode worker response correlation failed")
        if not response.get("ok"):
            error_type = str(response.get("error_type", "RemoteError"))
            message = str(response.get("error", ""))
            if error_type == "ValueError":
                raise ValueError(message)
            raise DecodeWorkerRemoteError(error_type, message)
        return dict(response.get("result", {})), response_payload

    def _session_request(
        self,
        session_id: str,
        operation: str,
        arguments: dict[str, Any],
        payload: bytes = b"",
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if (
                operation not in {"Init", "ImportSnapshot", "Close"}
                and checkpoint is not None
                and checkpoint.initialized
                and self._restored_generation.get(session_id) != self._generation
            ):
                self._restore_locked(session_id)
            for attempt in range(2):
                try:
                    return self._request_locked(
                        operation,
                        session_id,
                        arguments,
                        payload,
                        cancel_event=cancel_event,
                    )
                except OperationCancelledError:
                    raise
                except DecodeWorkerUnavailable:
                    if attempt:
                        raise
                    self.restart_count += 1
                    self.start()
                    self._restore_locked(session_id)
            raise AssertionError("unreachable")

    def _restore_locked(self, session_id: str) -> None:
        checkpoint = self._checkpoints.get(session_id)
        if checkpoint is None or not checkpoint.initialized:
            return
        initial_tokens = (
            checkpoint.replay_token_ids if checkpoint.snapshot is None else []
        )
        self._request_locked(
            "Init", session_id, {"token_ids": initial_tokens}, b""
        )
        if checkpoint.snapshot is not None:
            self._request_locked(
                "ImportSnapshot",
                session_id,
                {"compatibility": checkpoint.compatibility or {}},
                checkpoint.snapshot,
            )
            if checkpoint.replay_token_ids:
                self._request_locked(
                    "Append",
                    session_id,
                    {"token_ids": checkpoint.replay_token_ids},
                    b"",
                )
        self._restored_generation[session_id] = self._generation

    def _mark_current(self, session_id: str) -> None:
        with self._lock:
            self._restored_generation[session_id] = self._generation


class DecodeWorkerSession:
    """Verifier-shaped proxy used by existing append/generate coordinators."""

    is_decode_worker_proxy = True

    def __init__(self, client: DecodeWorkerClient, session_id: str) -> None:
        self.client = client
        self.session_id = session_id
        self.cached_token_sequence: list[int] = []
        self.next_global_position = 0
        self._kv_live_bytes = 0

    def _apply(self, state: dict[str, Any]) -> dict[str, Any]:
        self.cached_token_sequence = [
            int(token) for token in state.get("cached_token_ids", ())
        ]
        self.next_global_position = int(
            state.get("next_global_position", self.next_global_position)
        )
        self._kv_live_bytes = int(state.get("kv_live_bytes", 0))
        return state

    def prefill(
        self,
        prompt_ids: list[int],
        cancel_event: threading.Event | None = None,
    ) -> None:
        tokens = [int(token) for token in prompt_ids]
        state = self.client._session_request(
            self.session_id,
            "Init",
            {"token_ids": tokens},
            cancel_event=cancel_event,
        )
        checkpoint = self.client._checkpoints[self.session_id]
        checkpoint.snapshot = None
        checkpoint.compatibility = None
        checkpoint.replay_token_ids = list(tokens)
        checkpoint.initialized = True
        self.client._mark_current(self.session_id)
        self._apply(state)

    def append_accepted_tokens(
        self,
        tokens: list[int],
        cancel_event: threading.Event | None = None,
    ) -> None:
        committed = [int(token) for token in tokens]
        state = self.client._session_request(
            self.session_id,
            "Append",
            {"token_ids": committed},
            cancel_event=cancel_event,
        )
        self.client._checkpoints[self.session_id].replay_token_ids.extend(committed)
        self._apply(state)

    def generate_step(
        self,
        cancel_event: threading.Event | None = None,
    ) -> int:
        state = self.client._session_request(
            self.session_id,
            "GenerateStep",
            {},
            cancel_event=cancel_event,
        )
        token_id = int(state["token_id"])
        self.client._checkpoints[self.session_id].replay_token_ids.append(token_id)
        self._apply(state)
        return token_id

    def import_snapshot(
        self,
        payload: bytes,
        compatibility: Any,
    ) -> dict[str, Any]:
        compat = asdict(compatibility)
        # Ensure a worker-side session exists before importing its cache.
        self.client._session_request(
            self.session_id, "Init", {"token_ids": []}
        )
        state = self.client._session_request(
            self.session_id,
            "ImportSnapshot",
            {"compatibility": compat},
            bytes(payload),
        )
        checkpoint = self.client._checkpoints[self.session_id]
        checkpoint.snapshot = bytes(payload)
        checkpoint.compatibility = compat
        checkpoint.replay_token_ids = []
        checkpoint.initialized = True
        self.client._mark_current(self.session_id)
        return self._apply(state)

    def reset(self) -> None:
        state = self.client._session_request(
            self.session_id, "Init", {"token_ids": []}
        )
        checkpoint = self.client._checkpoints[self.session_id]
        checkpoint.snapshot = None
        checkpoint.compatibility = None
        checkpoint.replay_token_ids = []
        checkpoint.initialized = True
        self.client._mark_current(self.session_id)
        self._apply(state)

    def k_seq_length(self, _session: Any) -> int:
        return len(self.cached_token_sequence)

    def kv_live_bytes(self, _session: Any) -> int:
        return self._kv_live_bytes
