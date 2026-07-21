"""Primary decode liveness and unified-memory governance."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse


GIB = 1 << 30


@dataclass(frozen=True)
class DecodeSnapshot:
    phase: str
    session_id: str
    token_index: int
    updated_at_unix: float
    pid: int


class DecodeLiveness:
    """Thread-safe liveness state mirrored to an atomically replaced file."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        pid: Optional[int] = None,
        unhealthy_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.unhealthy_path = (
            Path(unhealthy_path).expanduser() if unhealthy_path else None
        )
        if self.unhealthy_path is not None:
            self.unhealthy_path.unlink(missing_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot = DecodeSnapshot(
            phase="idle",
            session_id="",
            token_index=0,
            updated_at_unix=clock(),
            pid=os.getpid() if pid is None else pid,
        )
        self._publish()

    def update(self, phase: str, session_id: str = "", token_index: int = 0) -> None:
        with self._lock:
            self._snapshot = DecodeSnapshot(
                phase=phase,
                session_id=session_id,
                token_index=int(token_index),
                updated_at_unix=self._clock(),
                pid=self._snapshot.pid,
            )
            self._publish()

    def snapshot(self) -> dict:
        with self._lock:
            result = asdict(self._snapshot)
        result["unhealthy"] = bool(
            self.unhealthy_path and self.unhealthy_path.exists()
        )
        return result

    def _publish(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(asdict(self._snapshot), sort_keys=True))
        os.replace(temporary, self.path)


@dataclass(frozen=True)
class MemorySnapshot:
    mlx_active_bytes: int
    mlx_cache_bytes: int
    mlx_peak_bytes: int
    process_footprint_bytes: int
    active_sessions: int
    kv_bytes: int
    level: str
    updated_at_unix: float


def process_footprint_bytes(pid: Optional[int] = None) -> int:
    """Return resident process bytes using the macOS/Linux ``ps`` contract."""
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
            text=True,
            timeout=1,
        )
        return int(output.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def mlx_memory_bytes() -> tuple[int, int, int]:
    """Best-effort MLX active/cache/peak counters without importing MLX in CI."""
    try:
        import mlx.core as mx
    except ImportError:
        return (0, 0, 0)

    def read(name: str) -> int:
        getter = getattr(mx, name, None)
        if getter is None:
            getter = getattr(getattr(mx, "metal", None), name, None)
        try:
            return int(getter()) if getter is not None else 0
        except (RuntimeError, TypeError, ValueError):
            return 0

    return (
        read("get_active_memory"),
        read("get_cache_memory"),
        read("get_peak_memory"),
    )


class PrimaryMemoryGovernor:
    """Classify memory pressure and clean idle shared-verifier state."""

    def __init__(
        self,
        store,
        verifier,
        *,
        warning_bytes: int = 18 * GIB,
        drain_bytes: int = 20 * GIB,
        unhealthy_bytes: int = int(21.5 * GIB),
        memory_provider: Callable[[], tuple[int, int, int]] = mlx_memory_bytes,
        footprint_provider: Callable[[], int] = process_footprint_bytes,
        clear_cache: Optional[Callable[[], None]] = None,
    ) -> None:
        if not warning_bytes < drain_bytes < unhealthy_bytes:
            raise ValueError("memory thresholds must satisfy warning < drain < unhealthy")
        self.store = store
        self.verifier = verifier
        self.warning_bytes = warning_bytes
        self.drain_bytes = drain_bytes
        self.unhealthy_bytes = unhealthy_bytes
        self._memory_provider = memory_provider
        self._footprint_provider = footprint_provider
        self._clear_cache = clear_cache or self._default_clear_cache
        self._lock = threading.Lock()
        self._last = self.sample()

    @property
    def draining(self) -> bool:
        with self._lock:
            return self._last.level in {"drain", "unhealthy"}

    @property
    def unhealthy(self) -> bool:
        with self._lock:
            return self._last.level == "unhealthy"

    def sample(self) -> MemorySnapshot:
        with self._lock:
            active, cache, peak = self._memory_provider()
            footprint = self._footprint_provider()
            pressure = max(active + cache, footprint)
            if pressure >= self.unhealthy_bytes:
                level = "unhealthy"
            elif pressure >= self.drain_bytes:
                level = "drain"
            elif pressure >= self.warning_bytes:
                level = "warning"
            else:
                level = "ok"
            self._last = MemorySnapshot(
                mlx_active_bytes=active,
                mlx_cache_bytes=cache,
                mlx_peak_bytes=peak,
                process_footprint_bytes=footprint,
                active_sessions=self.store.active_count,
                kv_bytes=self.store.total_kv_live_bytes,
                level=level,
                updated_at_unix=time.time(),
            )
            return self._last

    def on_session_removed(self, _session_id: str, _reason: str) -> None:
        """Reset and reclaim MLX cache when the final session leaves under pressure."""
        snapshot = self.sample()
        if snapshot.active_sessions or snapshot.level == "ok":
            return
        reset = getattr(self.verifier, "reset", None)
        if reset is not None:
            reset()
        gc.collect()
        self._clear_cache()
        self.sample()

    @staticmethod
    def _default_clear_cache() -> None:
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.clear_cache()


def create_runtime_health_app(
    liveness: DecodeLiveness,
    memory: PrimaryMemoryGovernor,
) -> FastAPI:
    """Create a read-only diagnostics app for operators and watchdogs."""
    app = FastAPI(title="Kakeya Primary Runtime Health", version="1")

    @app.get("/healthz")
    def healthz():
        live = liveness.snapshot()
        mem = asdict(memory.sample())
        unhealthy = bool(live["unhealthy"] or mem["level"] == "unhealthy")
        return JSONResponse(
            status_code=503 if unhealthy else 200,
            content={"status": "unhealthy" if unhealthy else "ok", "liveness": live, "memory": mem},
        )

    @app.get("/v1/runtime/liveness")
    def runtime_liveness():
        return liveness.snapshot()

    @app.get("/v1/runtime/memory")
    def runtime_memory():
        return asdict(memory.sample())

    return app
