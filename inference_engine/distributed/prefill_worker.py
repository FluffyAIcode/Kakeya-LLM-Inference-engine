"""Queued prefill-only worker jobs and gRPC service (ADR 0017)."""
from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Protocol, Sequence

import grpc

from inference_engine.distributed.capability import (
    CacheCompatibility,
    CompressionCodec,
)
from inference_engine.distributed.prefill_auth import (
    FleetAuthConfig,
    PrefillAuthError,
    signed_metadata,
    verify_metadata,
)
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    compatibility_fingerprint,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)


class PrefillJobState(IntEnum):
    UNSPECIFIED = 0
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5


class PrefillComputeEngine(Protocol):
    """Model-specific worker that produces the final restorable snapshot."""

    def compute_prefill(
        self,
        token_ids: Sequence[int],
        block_hashes: Sequence[bytes],
        *,
        compression: CompressionCodec,
        cancelled: threading.Event,
    ) -> Sequence[CacheBlock]: ...


@dataclass
class PrefillJob:
    job_id: str
    request_id: str
    tenant_id: str
    token_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    compression: CompressionCodec
    state: PrefillJobState = PrefillJobState.QUEUED
    tokens_computed: int = 0
    lease_id: str = ""
    block_hash: bytes = b""
    payload_sha256: bytes = b""
    transfer_bytes: int = 0
    failure_reason: str = ""
    compute_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    cancelled: threading.Event = field(default_factory=threading.Event)
    future: Future | None = field(default=None, repr=False)
    request_digest: bytes = b""
    deadline_at: float = 0.0


class PrefillJobStore:
    """Bounded, idempotent job queue around one or more prefill engines."""

    def __init__(
        self,
        engine: PrefillComputeEngine | None,
        cache_store: PrefixCacheStore,
        *,
        engine_factory: Callable[[], PrefillComputeEngine] | None = None,
        max_concurrent_jobs: int = 1,
        max_jobs: int = 128,
        completed_ttl_s: float = 600.0,
        max_prompt_tokens: int = 131_072,
    ) -> None:
        if min(
            max_concurrent_jobs,
            max_jobs,
            completed_ttl_s,
            max_prompt_tokens,
        ) <= 0:
            raise ValueError("worker limits must be > 0")
        if (engine is None) == (engine_factory is None):
            raise ValueError("provide exactly one of engine or engine_factory")
        self.engine = engine
        self.engine_factory = engine_factory
        self.cache_store = cache_store
        self.max_concurrent_jobs = int(max_concurrent_jobs)
        self.max_jobs = int(max_jobs)
        self.completed_ttl_s = float(completed_ttl_s)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self._jobs: dict[str, PrefillJob] = {}
        self._requests: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="kakeya-prefill-worker",
        )

    def warmup(self) -> None:
        """Construct a factory-backed engine on its eventual compute thread."""
        self._executor.submit(self._engine_for_current_thread).result()

    def submit(
        self,
        *,
        request_id: str,
        tenant_id: str,
        token_ids: Sequence[int],
        block_hashes: Sequence[bytes],
        compatibility: CacheCompatibility,
        compression: CompressionCodec,
        deadline_ms: int = 0,
    ) -> PrefillJob:
        if not request_id or not tenant_id:
            raise ValueError("request_id and tenant_id must be non-empty")
        if deadline_ms < 0:
            raise ValueError("deadline_ms must be >= 0")
        if compatibility != self.cache_store.compatibility:
            raise ValueError("prefill worker compatibility mismatch")
        if not token_ids or not block_hashes:
            raise ValueError("token_ids and block_hashes must be non-empty")
        if len(token_ids) > self.max_prompt_tokens:
            raise ValueError("prefill prompt exceeds worker token limit")
        expected_blocks = (
            len(token_ids) + compatibility.block_size_tokens - 1
        ) // compatibility.block_size_tokens
        if len(block_hashes) != expected_blocks:
            raise ValueError(
                f"expected {expected_blocks} block hashes, got {len(block_hashes)}",
            )
        if any(len(bytes(block_hash)) != 32 for block_hash in block_hashes):
            raise ValueError("every prefill block hash must be SHA-256 (32 bytes)")
        digest = hashlib.sha256(
            compatibility_fingerprint(compatibility)
            + b"".join(int(token).to_bytes(4, "little") for token in token_ids)
            + b"".join(bytes(h) for h in block_hashes)
            + int(compression).to_bytes(2, "little")
            + int(deadline_ms).to_bytes(8, "little", signed=False)
        ).digest()
        with self._lock:
            self._gc_locked()
            request_key = (tenant_id, request_id)
            existing_id = self._requests.get(request_key)
            if existing_id is not None:
                existing = self._jobs[existing_id]
                if existing.request_digest != digest:
                    raise ValueError(
                        "idempotency key reused with different prefill request",
                    )
                return existing
            active = sum(
                job.state in (PrefillJobState.QUEUED, PrefillJobState.RUNNING)
                for job in self._jobs.values()
            )
            if active >= self.max_jobs:
                raise ValueError("prefill worker job queue is full")
            job = PrefillJob(
                job_id=uuid.uuid4().hex,
                request_id=request_id,
                tenant_id=tenant_id,
                token_ids=tuple(int(token) for token in token_ids),
                block_hashes=tuple(bytes(h) for h in block_hashes),
                compression=compression,
                request_digest=digest,
                deadline_at=(
                    time.time() + deadline_ms / 1000.0
                    if deadline_ms > 0 else 0.0
                ),
            )
            self._jobs[job.job_id] = job
            self._requests[request_key] = job.job_id
            job.future = self._executor.submit(self._run, job.job_id)
            return job

    def get(self, job_id: str, tenant_id: str) -> PrefillJob:
        with self._lock:
            self._gc_locked()
            job = self._jobs.get(job_id)
            if job is None or job.tenant_id != tenant_id:
                raise KeyError(job_id)
            return job

    def cancel(self, job_id: str, tenant_id: str) -> bool:
        with self._lock:
            job = self.get(job_id, tenant_id)
            if job.state in (
                PrefillJobState.COMPLETED,
                PrefillJobState.FAILED,
                PrefillJobState.CANCELLED,
            ):
                return False
            job.cancelled.set()
            if job.future is not None and job.future.cancel():
                job.state = PrefillJobState.CANCELLED
                job.finished_at = time.time()
            return True

    def stats(self) -> tuple[int, int, float, int]:
        with self._lock:
            running = sum(j.state == PrefillJobState.RUNNING for j in self._jobs.values())
            queued = sum(j.state == PrefillJobState.QUEUED for j in self._jobs.values())
            queued_tokens = sum(
                len(j.token_ids)
                for j in self._jobs.values()
                if j.state in (PrefillJobState.QUEUED, PrefillJobState.RUNNING)
            )
            load = min(1.0, (running + queued) / self.max_concurrent_jobs)
            return running, queued, load, queued_tokens

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancelled.is_set():
                job.state = PrefillJobState.CANCELLED
                job.finished_at = time.time()
                return
            job.state = PrefillJobState.RUNNING
        started = time.perf_counter()
        timer = None
        if job.deadline_at:
            remaining = job.deadline_at - time.time()
            if remaining <= 0:
                job.cancelled.set()
            else:
                timer = threading.Timer(remaining, job.cancelled.set)
                timer.daemon = True
                timer.start()
        try:
            blocks = tuple(self._engine_for_current_thread().compute_prefill(
                job.token_ids,
                job.block_hashes,
                compression=job.compression,
                cancelled=job.cancelled,
            ))
            if job.cancelled.is_set():
                raise InterruptedError("prefill job cancelled")
            if len(blocks) != len(job.block_hashes):
                raise RuntimeError(
                    "prefill engine must return one snapshot per block hash",
                )
            for block in blocks:
                if job.cancelled.is_set():
                    raise InterruptedError("prefill job cancelled")
                self.cache_store.put(block)
            lease = self.cache_store.lookup(job.block_hashes)
            if not lease.lease_id:
                raise RuntimeError("computed snapshot was not discoverable")
            if job.cancelled.is_set():
                raise InterruptedError("prefill job cancelled")
            with self._lock:
                if job.cancelled.is_set():
                    raise InterruptedError("prefill job cancelled")
                job.state = PrefillJobState.COMPLETED
                job.tokens_computed = len(job.token_ids)
                job.lease_id = lease.lease_id
                final = blocks[-1]
                job.block_hash = final.block_hash
                job.payload_sha256 = final.payload_sha256
                job.transfer_bytes = final.nbytes
        except InterruptedError as exc:
            with self._lock:
                job.state = PrefillJobState.CANCELLED
                job.failure_reason = str(exc)
        except Exception as exc:
            with self._lock:
                job.state = PrefillJobState.FAILED
                job.failure_reason = f"{type(exc).__name__}: {exc}"
        finally:
            if timer is not None:
                timer.cancel()
            with self._lock:
                job.compute_ms = (time.perf_counter() - started) * 1000.0
                job.finished_at = time.time()

    def _engine_for_current_thread(self) -> PrefillComputeEngine:
        if self.engine is not None:
            return self.engine
        engine = getattr(self._thread_local, "engine", None)
        if engine is None:
            assert self.engine_factory is not None
            engine = self.engine_factory()
            self._thread_local.engine = engine
        return engine

    def _gc_locked(self) -> None:
        cutoff = time.time() - self.completed_ttl_s
        for job_id, job in list(self._jobs.items()):
            if job.finished_at and job.finished_at < cutoff:
                self._jobs.pop(job_id, None)
                self._requests.pop((job.tenant_id, job.request_id), None)
        overflow = len(self._jobs) - self.max_jobs * 2
        if overflow > 0:
            finished = sorted(
                (
                    job for job in self._jobs.values()
                    if job.finished_at
                ),
                key=lambda job: job.finished_at,
            )
            for job in finished[:overflow]:
                self._jobs.pop(job.job_id, None)
                self._requests.pop((job.tenant_id, job.request_id), None)


class PrefillWorkerServiceServicer(
    distributed_pb2_grpc.PrefillWorkerServiceServicer,
):
    def __init__(
        self,
        jobs: PrefillJobStore,
        *,
        node_id: str,
        cache_address: str,
        auth: FleetAuthConfig | None = None,
        tokens_per_second_prefill: float = 0.0,
    ) -> None:
        self.jobs = jobs
        self.node_id = node_id
        self.cache_address = cache_address
        self.auth = auth
        self.tokens_per_second_prefill = float(tokens_per_second_prefill)

    async def _authenticate(self, request, context) -> None:
        if self.auth is None:
            return
        try:
            verify_metadata(
                context.invocation_metadata(),
                request,
                self.auth,
            )
        except PrefillAuthError as exc:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))

    async def SubmitPrefillJob(self, request, context):  # noqa: N802
        await self._authenticate(request, context)
        if self.auth is not None and request.tenant_id != self.auth.tenant_id:
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "prefill job tenant mismatch",
            )
        try:
            job = await asyncio.to_thread(
                self.jobs.submit,
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                token_ids=list(request.token_ids),
                block_hashes=list(request.block_hashes),
                compatibility=CacheCompatibility.from_proto(request.compatibility),
                compression=CompressionCodec(request.preferred_compression),
                deadline_ms=request.deadline_ms,
            )
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return distributed_pb2.SubmitPrefillJobResponse(
            job_id=job.job_id,
            status=int(job.state),
            worker_node_id=self.node_id,
            queue_eta_ms=(
                self.jobs.stats()[3] / self.tokens_per_second_prefill * 1000.0
                if self.tokens_per_second_prefill > 0 else 0.0
            ),
        )

    async def GetPrefillJobStatus(self, request, context):  # noqa: N802
        await self._authenticate(request, context)
        if self.auth is not None and request.tenant_id != self.auth.tenant_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "tenant mismatch")
        try:
            job = self.jobs.get(request.job_id, request.tenant_id)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return _job_status_proto(job, self.cache_address)

    async def CancelPrefillJob(self, request, context):  # noqa: N802
        await self._authenticate(request, context)
        if self.auth is not None and request.tenant_id != self.auth.tenant_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "tenant mismatch")
        try:
            cancelled = self.jobs.cancel(request.job_id, request.tenant_id)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return distributed_pb2.CancelPrefillJobResponse(cancelled=cancelled)


def add_prefill_worker_service(
    server: grpc.aio.Server,
    jobs: PrefillJobStore,
    *,
    node_id: str,
    cache_address: str,
    auth: FleetAuthConfig | None = None,
    tokens_per_second_prefill: float = 0.0,
) -> PrefillWorkerServiceServicer:
    servicer = PrefillWorkerServiceServicer(
        jobs,
        node_id=node_id,
        cache_address=cache_address,
        auth=auth,
        tokens_per_second_prefill=tokens_per_second_prefill,
    )
    distributed_pb2_grpc.add_PrefillWorkerServiceServicer_to_server(
        servicer,
        server,
    )
    return servicer


def submit_prefill_job_sync(
    address: str,
    request: distributed_pb2.SubmitPrefillJobRequest,
    *,
    timeout_s: float,
    auth: FleetAuthConfig | None = None,
) -> distributed_pb2.SubmitPrefillJobResponse:
    metadata = signed_metadata(request, auth) if auth is not None else None
    with grpc.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.PrefillWorkerServiceStub(channel)
        return stub.SubmitPrefillJob(request, timeout=timeout_s, metadata=metadata)


def get_prefill_job_sync(
    address: str,
    request: distributed_pb2.GetPrefillJobStatusRequest,
    *,
    timeout_s: float,
    auth: FleetAuthConfig | None = None,
) -> distributed_pb2.GetPrefillJobStatusResponse:
    metadata = signed_metadata(request, auth) if auth is not None else None
    with grpc.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.PrefillWorkerServiceStub(channel)
        return stub.GetPrefillJobStatus(request, timeout=timeout_s, metadata=metadata)


def cancel_prefill_job_sync(
    address: str,
    request: distributed_pb2.CancelPrefillJobRequest,
    *,
    timeout_s: float,
    auth: FleetAuthConfig | None = None,
) -> distributed_pb2.CancelPrefillJobResponse:
    metadata = signed_metadata(request, auth) if auth is not None else None
    with grpc.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.PrefillWorkerServiceStub(channel)
        return stub.CancelPrefillJob(
            request,
            timeout=timeout_s,
            metadata=metadata,
        )


def _job_status_proto(
    job: PrefillJob,
    cache_address: str,
) -> distributed_pb2.GetPrefillJobStatusResponse:
    return distributed_pb2.GetPrefillJobStatusResponse(
        job_id=job.job_id,
        status=int(job.state),
        tokens_computed=job.tokens_computed,
        lease_id=job.lease_id,
        block_hash=job.block_hash,
        payload_sha256=job.payload_sha256,
        transfer_bytes=job.transfer_bytes,
        failure_reason=job.failure_reason,
        compute_ms=job.compute_ms,
        cache_address=cache_address,
    )

