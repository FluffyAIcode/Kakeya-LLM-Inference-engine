from __future__ import annotations

import asyncio
import threading
import time

import grpc
import pytest
import pytest_asyncio

from inference_engine.backends.mlx.prefill_worker import MLXPrefillComputeEngine
from inference_engine.distributed.capability import (
    CacheCompatibility,
    CompressionCodec,
)
from inference_engine.distributed.prefill_auth import (
    FleetAuthConfig,
    signed_metadata,
)
from inference_engine.distributed.prefill_cache import CacheBlock, PrefixCacheStore
from inference_engine.distributed.prefill_worker import (
    PrefillJob,
    PrefillJobState,
    PrefillJobStore,
    add_prefill_worker_service,
    estimate_final_snapshot_bytes,
    get_prefill_job_sync,
    submit_prefill_job_sync,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)

COMPAT = CacheCompatibility(
    model_id="m",
    tenant_namespace="tenant",
    block_size_tokens=2,
)
AUTH = FleetAuthConfig(b"k" * 32, "tenant", "client")


def test_snapshot_estimate_caps_at_sink_plus_sliding_window():
    compatibility = CacheCompatibility(
        model_id="m",
        block_size_tokens=64,
        sink_size=4,
        window_size=2048,
    )
    assert estimate_final_snapshot_bytes(2731, compatibility, 400_000) == (
        820_800_000
    )
    assert estimate_final_snapshot_bytes(1000, compatibility, 400_000) == (
        400_000_000
    )
    no_window = CacheCompatibility(model_id="m", window_size=0)
    assert estimate_final_snapshot_bytes(2731, no_window, 10) == 27_310
    assert estimate_final_snapshot_bytes(
        2731,
        no_window,
        400_000,
        max_retained_tokens=2052,
    ) == 820_800_000
    with pytest.raises(ValueError, match="must be > 0"):
        estimate_final_snapshot_bytes(0, compatibility, 400_000)


class _Engine:
    def __init__(self):
        self.calls = 0
        self.block = threading.Event()

    def compute_prefill(self, token_ids, block_hashes, *, compression, cancelled):
        self.calls += 1
        if self.block.is_set():
            while not cancelled.is_set():
                time.sleep(0.005)
            raise InterruptedError("prefill job cancelled")
        return [
            CacheBlock.create(h, min((i + 1) * 2, len(token_ids)), b"kv" + bytes([i]))
            for i, h in enumerate(block_hashes)
        ]


def test_mlx_worker_exports_only_final_snapshot(monkeypatch):
    class Logit:
        def clone(self):
            return self

    class Verifier:
        def __init__(self):
            self.next_token_logits = None
            self.forwarded = []

        def prefill(self, tokens):
            self.forwarded.extend(tokens)

        def forward_block(self, tokens):
            self.forwarded.extend(tokens)
            return [Logit() for _ in tokens]

        def commit_or_truncate(self, *, forwarded, accepted):
            assert forwarded == accepted

    verifier = Verifier()
    engine = MLXPrefillComputeEngine(
        verifier,
        COMPAT,
        compute_chunk_tokens=2,
    )
    snapshots = []
    progress = []
    engine.set_progress_callback(progress.append)

    def snapshot(**kwargs):
        snapshots.append(kwargs)
        return CacheBlock.create(
            kwargs["block_hash"],
            kwargs["token_count"],
            b"final",
        )

    monkeypatch.setattr(engine, "_snapshot", snapshot)
    hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    blocks = engine.compute_prefill(
        [1, 2, 3, 4, 5],
        hashes,
        compression=CompressionCodec.NONE,
        cancelled=threading.Event(),
    )
    engine.set_progress_callback(None)
    assert verifier.forwarded == [1, 2, 3, 4, 5]
    assert progress == [2, 4, 5]
    assert len(blocks) == 1
    assert snapshots == [{
        "token_count": 5,
        "block_hash": hashes[-1],
        "compression": CompressionCodec.NONE,
    }]
    with pytest.raises(ValueError, match="compute_chunk_tokens"):
        MLXPrefillComputeEngine(verifier, COMPAT, compute_chunk_tokens=0)


@pytest_asyncio.fixture
async def worker():
    engine = _Engine()
    cache = PrefixCacheStore(COMPAT, max_bytes=1024, node_id="worker")
    jobs = PrefillJobStore(engine, cache)
    server = grpc.aio.server()
    add_prefill_worker_service(
        server,
        jobs,
        node_id="worker",
        cache_address="cache:1",
        auth=AUTH,
        tokens_per_second_prefill=100,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"127.0.0.1:{port}", engine, cache, jobs
    finally:
        jobs.close()
        await server.stop(0)


def _submit(request_id="r"):
    return distributed_pb2.SubmitPrefillJobRequest(
        request_id=request_id,
        tenant_id="tenant",
        compatibility=COMPAT.to_proto(),
        token_ids=[1, 2, 3, 4],
        block_hashes=[b"a" * 32, b"b" * 32],
        preferred_compression=int(CompressionCodec.NONE),
    )


async def _rpc(address, method, request):
    async with grpc.aio.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.PrefillWorkerServiceStub(channel)
        return await getattr(stub, method)(
            request,
            metadata=signed_metadata(request, AUTH),
        )


@pytest.mark.asyncio
async def test_submit_is_idempotent_and_completes(worker):
    address, engine, cache, jobs = worker
    assert jobs.measured_tokens_per_second(100.0) == 100.0
    first = await _rpc(address, "SubmitPrefillJob", _submit())
    second = await _rpc(address, "SubmitPrefillJob", _submit())
    assert first.job_id == second.job_id
    for _ in range(100):
        request = distributed_pb2.GetPrefillJobStatusRequest(
            job_id=first.job_id,
            tenant_id="tenant",
        )
        status = await _rpc(address, "GetPrefillJobStatus", request)
        if status.status == int(PrefillJobState.COMPLETED):
            break
        await asyncio.sleep(0.01)
    assert status.status == int(PrefillJobState.COMPLETED)
    assert status.tokens_computed == 4
    assert status.lease_id and status.cache_address == "cache:1"
    assert engine.calls == 1
    assert len(cache.block_hashes()) == 2
    assert jobs.measured_tokens_per_second(100.0) > 0

    third = await _rpc(address, "SubmitPrefillJob", _submit("r2"))
    for _ in range(100):
        status = await _rpc(
            address,
            "GetPrefillJobStatus",
            distributed_pb2.GetPrefillJobStatusRequest(
                job_id=third.job_id,
                tenant_id="tenant",
            ),
        )
        if status.status == int(PrefillJobState.COMPLETED):
            break
        await asyncio.sleep(0.01)
    assert status.status == int(PrefillJobState.COMPLETED)
    assert engine.calls == 2


@pytest.mark.asyncio
async def test_service_rejects_unauthenticated_and_incompatible(worker):
    address, _, _, _ = worker
    async with grpc.aio.insecure_channel(address) as channel:
        stub = distributed_pb2_grpc.PrefillWorkerServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await stub.SubmitPrefillJob(_submit())
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
    bad = _submit("bad")
    bad.compatibility.CopyFrom(CacheCompatibility(model_id="x").to_proto())
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await _rpc(address, "SubmitPrefillJob", bad)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_cancel_running_job(worker):
    address, engine, _, _ = worker
    engine.block.set()
    submitted = await _rpc(address, "SubmitPrefillJob", _submit("cancel"))
    await asyncio.sleep(0.02)
    request = distributed_pb2.CancelPrefillJobRequest(
        job_id=submitted.job_id,
        tenant_id="tenant",
    )
    assert (await _rpc(address, "CancelPrefillJob", request)).cancelled
    for _ in range(100):
        status_request = distributed_pb2.GetPrefillJobStatusRequest(
            job_id=submitted.job_id,
            tenant_id="tenant",
        )
        status = await _rpc(address, "GetPrefillJobStatus", status_request)
        if status.status == int(PrefillJobState.CANCELLED):
            break
        await asyncio.sleep(0.01)
    assert status.status == int(PrefillJobState.CANCELLED)


def test_job_store_validation_queue_stats_and_gc():
    cache = PrefixCacheStore(COMPAT, max_bytes=1024, node_id="w")
    with pytest.raises(ValueError):
        PrefillJobStore(_Engine(), cache, max_jobs=0)
    with pytest.raises(ValueError, match="exactly one"):
        PrefillJobStore(None, cache)
    with pytest.raises(ValueError, match="exactly one"):
        PrefillJobStore(_Engine(), cache, engine_factory=_Engine)
    with pytest.raises(ValueError, match="max_retained_tokens"):
        PrefillJobStore(_Engine(), cache, max_retained_tokens=-1)
    blocking = _Engine()
    blocking.block.set()
    jobs = PrefillJobStore(blocking, cache, max_jobs=1, max_prompt_tokens=4)
    try:
        common = dict(
            request_id="r",
            tenant_id="tenant",
            token_ids=[1, 2, 3, 4],
            block_hashes=[b"a" * 32, b"b" * 32],
            compatibility=COMPAT,
            compression=CompressionCodec.NONE,
        )
        with pytest.raises(ValueError, match="request_id"):
            jobs.submit(**{**common, "request_id": ""})
        with pytest.raises(ValueError, match="non-empty"):
            jobs.submit(**{**common, "token_ids": []})
        with pytest.raises(ValueError, match="token limit"):
            jobs.submit(**{**common, "token_ids": [1] * 5,
                           "block_hashes": [b"a" * 32] * 3})
        with pytest.raises(ValueError, match="block hashes"):
            jobs.submit(**{**common, "block_hashes": [b"a" * 32]})
        with pytest.raises(ValueError, match="SHA-256"):
            jobs.submit(**{**common, "block_hashes": [b"a", b"b"]})
        with pytest.raises(ValueError, match="deadline_ms"):
            jobs.submit(**{**common, "deadline_ms": -1})
        job = jobs.submit(**common)
        with pytest.raises(ValueError, match="idempotency key"):
            jobs.submit(**{**common, "token_ids": [9, 2, 3, 4]})
        with pytest.raises(ValueError, match="idempotency key"):
            jobs.submit(**{**common, "deadline_ms": 100})
        with pytest.raises(ValueError, match="queue is full"):
            jobs.submit(**{**common, "request_id": "other"})
        with pytest.raises(KeyError):
            jobs.get(job.job_id, "other-tenant")
        assert jobs.cancel(job.job_id, "tenant")
    finally:
        jobs.close()

    completed_jobs = PrefillJobStore(
        _Engine(), cache, max_jobs=1, max_prompt_tokens=4,
    )
    try:
        job = completed_jobs.submit(**common)
        for _ in range(100):
            if job.state == PrefillJobState.COMPLETED:
                break
            time.sleep(0.005)
        assert completed_jobs.cancel(job.job_id, "tenant") is False
        assert completed_jobs.stats()[2] == 0
        job.finished_at = time.time() - 1000
        completed_jobs.completed_ttl_s = 1
        with pytest.raises(KeyError):
            completed_jobs.get(job.job_id, "tenant")
        for index in range(3):
            old = PrefillJob(
                f"old-{index}", f"old-{index}", "tenant",
                (1, 2), (b"a" * 32,), CompressionCodec.NONE,
                state=PrefillJobState.COMPLETED,
                finished_at=time.time(),
            )
            completed_jobs._jobs[old.job_id] = old
            completed_jobs._requests[(old.tenant_id, old.request_id)] = old.job_id
        completed_jobs._gc_locked()
        assert len(completed_jobs._jobs) <= completed_jobs.max_jobs * 2
    finally:
        completed_jobs.close()


def test_job_rejects_snapshot_capacity_before_model_compute():
    engine = _Engine()
    jobs = PrefillJobStore(
        engine,
        PrefixCacheStore(COMPAT, max_bytes=1024, node_id="small"),
        estimated_snapshot_bytes_per_token=600,
    )
    try:
        with pytest.raises(ValueError, match="estimated final snapshot"):
            jobs.submit(
                request_id="too-large",
                tenant_id="tenant",
                token_ids=[1, 2],
                block_hashes=[b"a" * 32],
                compatibility=COMPAT,
                compression=CompressionCodec.NONE,
            )
        assert engine.calls == 0
    finally:
        jobs.close()


def test_job_reservation_preserves_unrelated_restore_snapshot():
    cache = PrefixCacheStore(COMPAT, max_bytes=100, node_id="shared")
    old = CacheBlock.create(b"z" * 32, 2, b"old-snapshot")
    cache.put(old)
    jobs = PrefillJobStore(
        _Engine(),
        cache,
        estimated_snapshot_bytes_per_token=10,
    )
    try:
        job = jobs.submit(
            request_id="new-prefix",
            tenant_id="tenant",
            token_ids=[1, 2],
            block_hashes=[b"a" * 32],
            compatibility=COMPAT,
            compression=CompressionCodec.NONE,
        )
        job.future.result(timeout=1)
        assert job.state == PrefillJobState.COMPLETED
        assert old.block_hash in cache.block_hashes()
        assert job.block_hash in cache.block_hashes()
    finally:
        jobs.close()


def test_job_accepts_final_snapshot_only():
    class FinalOnlyEngine:
        def compute_prefill(self, token_ids, block_hashes, **_kwargs):
            return [
                CacheBlock.create(
                    block_hashes[-1],
                    len(token_ids),
                    b"final-only",
                ),
            ]

    cache = PrefixCacheStore(COMPAT, max_bytes=1024, node_id="final-only")
    jobs = PrefillJobStore(FinalOnlyEngine(), cache)
    try:
        job = jobs.submit(
            request_id="final-only",
            tenant_id="tenant",
            token_ids=[1, 2, 3, 4],
            block_hashes=[b"a" * 32, b"b" * 32],
            compatibility=COMPAT,
            compression=CompressionCodec.NONE,
        )
        job.future.result(timeout=1)
        assert job.state == PrefillJobState.COMPLETED
        assert cache.block_hashes() == (b"b" * 32,)
        assert cache.fetch(job.lease_id)[0].payload == b"final-only"
    finally:
        jobs.close()


def test_job_store_wires_and_clears_segment_progress_callback():
    class ProgressEngine:
        def __init__(self):
            self.callback = None
            self.callbacks = []

        def set_progress_callback(self, callback):
            self.callback = callback
            self.callbacks.append(callback)

        def compute_prefill(self, token_ids, block_hashes, **_kwargs):
            self.callback(2)
            self.callback(4)
            return [
                CacheBlock.create(block_hashes[-1], len(token_ids), b"final"),
            ]

    engine = ProgressEngine()
    jobs = PrefillJobStore(
        engine,
        PrefixCacheStore(COMPAT, max_bytes=1024, node_id="progress"),
    )
    try:
        job = jobs.submit(
            request_id="progress",
            tenant_id="tenant",
            token_ids=[1, 2, 3, 4],
            block_hashes=[b"a" * 32, b"b" * 32],
            compatibility=COMPAT,
            compression=CompressionCodec.NONE,
        )
        job.future.result(timeout=1)
        assert job.state == PrefillJobState.COMPLETED
        assert job.tokens_computed == 4
        assert callable(engine.callbacks[0])
        assert engine.callbacks[-1] is None
    finally:
        jobs.close()


def test_factory_engine_is_warmed_and_used_on_same_compute_thread():
    cache = PrefixCacheStore(COMPAT, max_bytes=1024, node_id="w")
    created_on = []
    computed_on = []

    class ThreadBoundEngine(_Engine):
        def compute_prefill(self, *args, **kwargs):
            computed_on.append(threading.get_ident())
            return super().compute_prefill(*args, **kwargs)

    def factory():
        created_on.append(threading.get_ident())
        return ThreadBoundEngine()

    jobs = PrefillJobStore(None, cache, engine_factory=factory)
    try:
        jobs.warmup()
        job = jobs.submit(
            request_id="thread-affinity",
            tenant_id="tenant",
            token_ids=[1, 2],
            block_hashes=[b"a" * 32],
            compatibility=COMPAT,
            compression=CompressionCodec.NONE,
        )
        for _ in range(100):
            if job.state == PrefillJobState.COMPLETED:
                break
            time.sleep(0.005)
        assert job.state == PrefillJobState.COMPLETED
        assert created_on == computed_on
        assert created_on[0] != threading.get_ident()
    finally:
        jobs.close()


def test_job_store_failure_modes_and_precancelled_run():
    cache = PrefixCacheStore(COMPAT, max_bytes=1024, node_id="w")

    class BadCount:
        def compute_prefill(self, *args, **kwargs):
            return []

    jobs = PrefillJobStore(BadCount(), cache)
    job = jobs.submit(
        request_id="bad",
        tenant_id="tenant",
        token_ids=[1, 2],
        block_hashes=[b"a" * 32],
        compatibility=COMPAT,
        compression=CompressionCodec.NONE,
    )
    for _ in range(100):
        if job.state == PrefillJobState.FAILED:
            break
        time.sleep(0.005)
    assert job.state == PrefillJobState.FAILED
    assert "one snapshot" in job.failure_reason

    manual = PrefillJob(
        "manual", "manual", "tenant", (1, 2), (b"a" * 32,),
        CompressionCodec.NONE,
    )
    manual.cancelled.set()
    jobs._jobs[manual.job_id] = manual
    jobs._run(manual.job_id)
    assert manual.state == PrefillJobState.CANCELLED
    expired = PrefillJob(
        "expired", "expired", "tenant", (1, 2), (b"a" * 32,),
        CompressionCodec.NONE,
        deadline_at=time.time() - 1,
    )
    jobs._jobs[expired.job_id] = expired
    jobs._run(expired.job_id)
    assert expired.state == PrefillJobState.CANCELLED
    jobs.close()


def test_queued_cancel_post_compute_cancel_and_undiscoverable():
    cache = PrefixCacheStore(COMPAT, max_bytes=4096, node_id="w")

    class Blocking:
        def __init__(self):
            self.release = threading.Event()

        def compute_prefill(self, token_ids, block_hashes, **kwargs):
            self.release.wait(1)
            return [CacheBlock.create(block_hashes[0], 2, b"ok")]

    blocking = Blocking()
    jobs = PrefillJobStore(blocking, cache, max_concurrent_jobs=1)
    common = dict(
        tenant_id="tenant",
        token_ids=[1, 2],
        block_hashes=[b"a" * 32],
        compatibility=COMPAT,
        compression=CompressionCodec.NONE,
    )
    first = jobs.submit(request_id="first", **common)
    second = jobs.submit(request_id="second", **common)
    assert jobs.cancel(second.job_id, "tenant")
    assert second.state == PrefillJobState.CANCELLED
    blocking.release.set()
    first.future.result(timeout=1)
    jobs.close()

    class CancelsAfterCompute:
        def compute_prefill(self, token_ids, block_hashes, *, cancelled, **kwargs):
            cancelled.set()
            return [CacheBlock.create(block_hashes[0], 2, b"ok")]

    cancelled_jobs = PrefillJobStore(
        CancelsAfterCompute(),
        PrefixCacheStore(COMPAT, max_bytes=4096, node_id="cancel"),
    )
    cancelled = cancelled_jobs.submit(request_id="after", **common)
    cancelled.future.result(timeout=1)
    assert cancelled.state == PrefillJobState.CANCELLED
    cancelled_jobs.close()

    class WrongHash:
        def compute_prefill(self, token_ids, block_hashes, **kwargs):
            return [CacheBlock.create(b"z" * 32, 2, b"wrong")]

    missing_jobs = PrefillJobStore(
        WrongHash(),
        PrefixCacheStore(COMPAT, max_bytes=4096, node_id="missing"),
    )
    missing = missing_jobs.submit(request_id="missing", **common)
    missing.future.result(timeout=1)
    assert missing.state == PrefillJobState.FAILED
    assert "final snapshot hash does not match request" in missing.failure_reason
    missing_jobs.close()

    deadline_engine = _Engine()
    deadline_engine.block.set()
    deadline_jobs = PrefillJobStore(deadline_engine, PrefixCacheStore(
        COMPAT, max_bytes=4096, node_id="deadline",
    ))
    deadline = deadline_jobs.submit(
        request_id="deadline",
        deadline_ms=10,
        **common,
    )
    deadline.future.result(timeout=1)
    assert deadline.state == PrefillJobState.CANCELLED
    deadline_jobs.close()


def test_cancel_during_publish_and_after_lease():
    class Engine:
        event = None

        def compute_prefill(self, token_ids, block_hashes, *, cancelled, **kwargs):
            self.event = cancelled
            return [
                CacheBlock.create(h, min((i + 1) * 2, 4), b"x" + bytes([i]))
                for i, h in enumerate(block_hashes)
            ]

    engine = Engine()

    class CancelBeforeAtomicPublishStore(PrefixCacheStore):
        def publish_and_lease(self, *args, **kwargs):
            engine.event.set()
            return super().publish_and_lease(*args, **kwargs)

    store = CancelBeforeAtomicPublishStore(
        COMPAT,
        max_bytes=4096,
        node_id="put",
    )
    jobs = PrefillJobStore(engine, store)
    job = jobs.submit(
        request_id="put",
        tenant_id="tenant",
        token_ids=[1, 2, 3, 4],
        block_hashes=[b"a" * 32, b"b" * 32],
        compatibility=COMPAT,
        compression=CompressionCodec.NONE,
    )
    job.future.result(timeout=1)
    assert job.state == PrefillJobState.CANCELLED
    jobs.close()

    engine2 = Engine()

    class CancelAfterAtomicPublishStore(PrefixCacheStore):
        def publish_and_lease(self, *args, **kwargs):
            lease = super().publish_and_lease(*args, **kwargs)
            engine2.event.set()
            return lease

    store2 = CancelAfterAtomicPublishStore(
        COMPAT,
        max_bytes=4096,
        node_id="lookup",
    )
    jobs2 = PrefillJobStore(engine2, store2)
    job2 = jobs2.submit(
        request_id="lookup",
        tenant_id="tenant",
        token_ids=[1, 2],
        block_hashes=[b"a" * 32],
        compatibility=COMPAT,
        compression=CompressionCodec.NONE,
    )
    job2.future.result(timeout=1)
    assert job2.state == PrefillJobState.CANCELLED
    jobs2.close()

    class SequenceEvent:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 4

        def set(self):
            self.calls = 4

    final_store = PrefixCacheStore(COMPAT, max_bytes=4096, node_id="final")
    final_jobs = PrefillJobStore(_Engine(), final_store)
    final_job = PrefillJob(
        "final", "final", "tenant", (1, 2), (b"a" * 32,),
        CompressionCodec.NONE,
    )
    final_job.cancelled = SequenceEvent()
    final_jobs._jobs[final_job.job_id] = final_job
    final_jobs._run(final_job.job_id)
    assert final_job.state == PrefillJobState.CANCELLED
    final_jobs.close()


@pytest.mark.asyncio
async def test_worker_service_tenant_and_not_found_errors(worker):
    address, _, _, _ = worker
    wrong_tenant = _submit("wrong-tenant")
    wrong_tenant.tenant_id = "other"
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await _rpc(address, "SubmitPrefillJob", wrong_tenant)
    assert exc.value.code() == grpc.StatusCode.PERMISSION_DENIED
    wrong_get = distributed_pb2.GetPrefillJobStatusRequest(
        job_id="anything", tenant_id="other",
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await _rpc(address, "GetPrefillJobStatus", wrong_get)
    assert exc.value.code() == grpc.StatusCode.PERMISSION_DENIED
    wrong_cancel = distributed_pb2.CancelPrefillJobRequest(
        job_id="anything", tenant_id="other",
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await _rpc(address, "CancelPrefillJob", wrong_cancel)
    assert exc.value.code() == grpc.StatusCode.PERMISSION_DENIED
    missing = distributed_pb2.GetPrefillJobStatusRequest(
        job_id="missing",
        tenant_id="tenant",
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await _rpc(address, "GetPrefillJobStatus", missing)
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND
    cancel = distributed_pb2.CancelPrefillJobRequest(
        job_id="missing",
        tenant_id="tenant",
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await _rpc(address, "CancelPrefillJob", cancel)
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_sync_clients_work_without_auth():
    cache = PrefixCacheStore(COMPAT, max_bytes=1024, node_id="worker")
    jobs = PrefillJobStore(_Engine(), cache)
    server = grpc.aio.server()
    add_prefill_worker_service(
        server,
        jobs,
        node_id="worker",
        cache_address="cache:1",
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    address = f"127.0.0.1:{port}"
    try:
        submitted = await asyncio.to_thread(
            submit_prefill_job_sync,
            address,
            _submit("no-auth"),
            timeout_s=2,
        )
        request = distributed_pb2.GetPrefillJobStatusRequest(
            job_id=submitted.job_id,
            tenant_id="tenant",
        )
        status = await asyncio.to_thread(
            get_prefill_job_sync,
            address,
            request,
            timeout_s=2,
        )
        assert status.job_id == submitted.job_id
    finally:
        jobs.close()
        await server.stop(0)

