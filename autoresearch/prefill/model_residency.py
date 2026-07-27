"""Journaled, mutually-exclusive Gemma/OProver residency scheduler."""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol


class ResidencyPhase(str, Enum):
    GEMMA_SERVING = "GEMMA_SERVING"
    QUIESCE_SNAPSHOT = "QUIESCE_SNAPSHOT"
    GEMMA_UNLOAD = "GEMMA_UNLOAD"
    HEADROOM_CHECK = "HEADROOM_CHECK"
    OPROVER_LOAD = "OPROVER_LOAD"
    OPROVER_ADVISE = "OPROVER_ADVISE"
    OPROVER_UNLOAD = "OPROVER_UNLOAD"
    GEMMA_RESTORE = "GEMMA_RESTORE"
    GEMMA_HEALTH_VERIFY = "GEMMA_HEALTH_VERIFY"
    FAILED_CLOSED = "FAILED_CLOSED"


@dataclass
class ResidencyState:
    phase: str = ResidencyPhase.GEMMA_SERVING.value
    active_model: str = "gemma"
    owner_pid: int = 0
    gemma_pid: int = 0
    oprover_pid: int = 0
    model_id: str = ""
    model_revision: str = ""
    tokenizer_id: str = ""
    cache_namespace: str = ""
    updated_at: float = 0.0
    error_code: str = ""
    journal_sequence: int = 0
    owner_generation: int = 0
    gemma_executable: str = ""
    gemma_model_id: str = ""
    gemma_model_revision: str = ""
    gemma_start_token: str = ""


@dataclass(frozen=True)
class ProcessIdentity:
    """Non-secret process identity returned by a trusted host inspector."""

    pid: int
    executable: str
    model_id: str
    model_revision: str
    start_token: str
    runtime_healthy: bool


class ProcessManager(Protocol):
    def quiesce_and_snapshot(self) -> None: ...
    def stop_gemma(self) -> int: ...
    def gemma_is_resident(self) -> bool: ...
    def load_oprover(self) -> int: ...
    def oprover_is_resident(self) -> bool: ...
    def unload_oprover(self, pid: int) -> None: ...
    def restore_gemma(self) -> int: ...
    def gemma_healthy(self) -> bool: ...
    def allens_healthy(self) -> bool: ...
    def gemma_identity(self) -> ProcessIdentity | None: ...


class ResidencyError(RuntimeError):
    pass


class ModelResidencyScheduler:
    def __init__(
        self,
        *,
        state_path: Path,
        process_manager: ProcessManager,
        headroom_check: Callable[[], bool],
        model_id: str,
        model_revision: str,
        tokenizer_id: str,
        gemma_cache_namespace: str,
        oprover_cache_namespace: str,
        gemma_executable: str = "",
        gemma_model_id: str = "",
        gemma_model_revision: str = "",
    ) -> None:
        if not model_revision or not tokenizer_id:
            raise ValueError("OProver revision and tokenizer ID must be pinned")
        if gemma_cache_namespace == oprover_cache_namespace:
            raise ValueError("OPROVER_CACHE_NAMESPACE_MUST_BE_SEPARATE")
        self.state_path = Path(state_path).expanduser()
        self.lock_path = self.state_path.with_suffix(".lock")
        self.journal_path = self.state_path.with_suffix(".journal.jsonl")
        self.pm = process_manager
        self.headroom_check = headroom_check
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_id = tokenizer_id
        self.gemma_cache_namespace = gemma_cache_namespace
        self.oprover_cache_namespace = oprover_cache_namespace
        self.gemma_executable = gemma_executable
        self.gemma_model_id = gemma_model_id
        self.gemma_model_revision = gemma_model_revision

    def run_exclusive(self, advise: Callable[[], object]) -> object:
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(self.lock_path, 0o600)
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResidencyError("RESIDENCY_LOCK_HELD") from exc
            state = self._load()
            self._recover(state)
            try:
                self._transition(state, ResidencyPhase.QUIESCE_SNAPSHOT)
                self.pm.quiesce_and_snapshot()
                self._transition(state, ResidencyPhase.GEMMA_UNLOAD)
                state.gemma_pid = int(self.pm.stop_gemma())
                if state.gemma_pid <= 0:
                    raise ResidencyError("GEMMA_OWNED_PID_REQUIRED")
                if self.pm.gemma_is_resident():
                    raise ResidencyError("MUTUAL_EXCLUSION_GEMMA_STILL_RESIDENT")
                self._transition(state, ResidencyPhase.HEADROOM_CHECK)
                if not self.headroom_check():
                    raise ResidencyError("OPROVER_MEMORY_HEADROOM_INSUFFICIENT")
                self._transition(
                    state, ResidencyPhase.OPROVER_LOAD, active_model="oprover",
                )
                state.oprover_pid = int(self.pm.load_oprover())
                if state.oprover_pid <= 0:
                    raise ResidencyError("OPROVER_OWNED_PID_REQUIRED")
                self._save(state)
                if not self.pm.oprover_is_resident() or self.pm.gemma_is_resident():
                    raise ResidencyError("OPROVER_PROCESS_OWNERSHIP_AMBIGUOUS")
                self._transition(state, ResidencyPhase.OPROVER_ADVISE)
                return advise()
            except BaseException as exc:
                state.error_code = _safe_error(exc)
                raise
            finally:
                self._restore(state)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def recover_only(self) -> ResidencyState:
        """Reconcile a crash journal without starting another model swap."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(self.lock_path, 0o600)
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResidencyError("RESIDENCY_LOCK_HELD") from exc
            state = self._load()
            self._recover(state)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return state

    def reconcile_gemma_owner(self) -> ResidencyState:
        """Atomically repair stale metadata only after full identity validation."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(self.lock_path, 0o600)
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResidencyError("RESIDENCY_LOCK_HELD") from exc
            state = self._load()
            self._reconcile_serving_owner(state)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return state

    def _restore(self, state: ResidencyState) -> None:
        try:
            if self.pm.oprover_is_resident():
                self._transition(state, ResidencyPhase.OPROVER_UNLOAD)
                if state.oprover_pid <= 0:
                    raise ResidencyError("OPROVER_OWNED_PID_REQUIRED")
                self.pm.unload_oprover(state.oprover_pid)
            if self.pm.oprover_is_resident():
                raise ResidencyError("OPROVER_UNLOAD_NOT_CONFIRMED")
            state.oprover_pid = 0
            self._transition(
                state, ResidencyPhase.GEMMA_RESTORE, active_model="none",
            )
            if not self.pm.gemma_is_resident():
                state.gemma_pid = int(self.pm.restore_gemma())
                if state.gemma_pid <= 0:
                    raise ResidencyError("GEMMA_RESTORE_OWNED_PID_REQUIRED")
            self._transition(
                state, ResidencyPhase.GEMMA_HEALTH_VERIFY, active_model="gemma",
            )
            if not self.pm.gemma_healthy() or not self.pm.allens_healthy():
                raise ResidencyError("GEMMA_OR_ALLENS_HEALTH_RESTORE_FAILED")
            if self.pm.oprover_is_resident():
                raise ResidencyError("CONCURRENT_MODEL_RESIDENCY_DETECTED")
            state.error_code = ""
            state.owner_pid = 0
            self._transition(state, ResidencyPhase.GEMMA_SERVING)
        except BaseException as exc:
            state.error_code = _safe_error(exc)
            self._transition(
                state, ResidencyPhase.FAILED_CLOSED, active_model="unknown",
            )
            raise

    def _recover(self, state: ResidencyState) -> None:
        try:
            phase = ResidencyPhase(state.phase)
        except ValueError as exc:
            raise ResidencyError("UNKNOWN_RESIDENCY_PHASE") from exc
        if phase is ResidencyPhase.GEMMA_SERVING:
            if self.pm.oprover_is_resident():
                raise ResidencyError("STALE_OPROVER_PROCESS_REQUIRES_OWNED_PID")
            self._reconcile_serving_owner(state)
            state.owner_pid = 0
            state.oprover_pid = 0
            self._save(state)
            return
        self._restore(state)

    def _reconcile_serving_owner(self, state: ResidencyState) -> None:
        inspector = getattr(self.pm, "gemma_identity", None)
        if inspector is None:
            if state.gemma_pid > 0 and not self.pm.gemma_is_resident():
                raise ResidencyError("GEMMA_OWNER_IDENTITY_INSPECTOR_REQUIRED")
            return
        identity = inspector()
        if identity is None:
            if state.gemma_pid > 0 or self.pm.gemma_is_resident():
                raise ResidencyError("GEMMA_OWNER_LIVE_IDENTITY_REQUIRED")
            return
        if identity.pid <= 0 or not identity.runtime_healthy:
            raise ResidencyError("GEMMA_OWNER_RUNTIME_UNHEALTHY")
        expected = {
            "executable": self.gemma_executable or state.gemma_executable,
            "model_id": self.gemma_model_id or state.gemma_model_id,
            "model_revision": (
                self.gemma_model_revision or state.gemma_model_revision
            ),
        }
        actual = {
            "executable": identity.executable,
            "model_id": identity.model_id,
            "model_revision": identity.model_revision,
        }
        if any(not value for value in (*actual.values(), identity.start_token)):
            raise ResidencyError("GEMMA_OWNER_IDENTITY_INCOMPLETE")
        if any(expected[name] and expected[name] != actual[name] for name in expected):
            raise ResidencyError("GEMMA_OWNER_IDENTITY_MISMATCH")
        if (
            state.gemma_pid == identity.pid
            and state.gemma_start_token
            and state.gemma_start_token != identity.start_token
        ):
            raise ResidencyError("GEMMA_OWNER_PID_REUSE_REJECTED")
        changed = (
            state.gemma_pid != identity.pid
            or state.gemma_start_token != identity.start_token
            or state.gemma_executable != identity.executable
            or state.gemma_model_id != identity.model_id
            or state.gemma_model_revision != identity.model_revision
        )
        if changed:
            state.gemma_pid = identity.pid
            state.gemma_executable = identity.executable
            state.gemma_model_id = identity.model_id
            state.gemma_model_revision = identity.model_revision
            state.gemma_start_token = identity.start_token
            state.owner_generation += 1
            state.journal_sequence += 1
            state.updated_at = time.time()
            self._save(state)

    def _transition(
        self,
        state: ResidencyState,
        phase: ResidencyPhase,
        *,
        active_model: str | None = None,
    ) -> None:
        state.phase = phase.value
        state.active_model = active_model or state.active_model
        state.owner_pid = (
            0 if phase is ResidencyPhase.GEMMA_SERVING else os.getpid()
        )
        state.model_id = self.model_id
        state.model_revision = self.model_revision
        state.tokenizer_id = self.tokenizer_id
        state.cache_namespace = (
            self.oprover_cache_namespace
            if state.active_model == "oprover"
            else self.gemma_cache_namespace
            if state.active_model == "gemma"
            else ""
        )
        state.updated_at = time.time()
        state.journal_sequence += 1
        self._save(state)

    def _load(self) -> ResidencyState:
        try:
            return ResidencyState(**json.loads(
                self.state_path.read_text(encoding="utf-8")
            ))
        except FileNotFoundError:
            return ResidencyState(updated_at=time.time())

    def _save(self, state: ResidencyState) -> None:
        payload = asdict(state)
        encoded = json.dumps(payload, sort_keys=True)
        with self.journal_path.open("a", encoding="utf-8") as journal:
            os.chmod(self.journal_path, 0o600)
            journal.write(encoded + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o600)
        with temporary.open("r+", encoding="utf-8") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_path)
        os.chmod(self.state_path, 0o600)
        directory = os.open(self.state_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


_SECRET_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|secret|access[_ -]?token|authorization)"
)


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}:{exc}"
    if "/" in text or _SECRET_PATTERN.search(text):
        return f"{type(exc).__name__}:redacted"
    return re.sub(r"[^A-Za-z0-9_.:+\- ]", "_", text)[:240]
