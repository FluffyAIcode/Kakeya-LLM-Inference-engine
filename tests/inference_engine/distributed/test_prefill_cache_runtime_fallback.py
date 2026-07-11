from __future__ import annotations

from inference_engine.distributed.capability import (
    CacheCapability,
    CacheCompatibility,
    CompressionCodec,
    NodeCapability,
    NodeEndpoint,
    PrefillWorkerCapability,
)
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    chained_block_hashes,
)
from inference_engine.distributed.prefill_cache_runtime import (
    DistributedPrefillCacheHook,
    _Hit,
)
from inference_engine.distributed.prefill_scheduler import PrefillCostConfig
from inference_engine.server.proto_gen.kakeya.v1 import distributed_pb2


class _Verifier:
    def __init__(self):
        self.cache = []
        self.cached_token_sequence = []
        self.next_global_position = 0
        self.next_token_logits = None
        self.prefill_calls = 0

    def reset(self):
        self.cache = []
        self.cached_token_sequence = []
        self.next_global_position = 0

    def prefill(self, tokens):
        self.reset()
        self.prefill_calls += 1
        self.cached_token_sequence = list(tokens)
        self.next_global_position = len(tokens)
        self.next_token_logits = _Row(tokens[-1])

    def forward_block(self, tokens):
        self.cached_token_sequence.extend(tokens)
        self.next_global_position += len(tokens)
        return [_Row(token) for token in tokens]

    def commit_or_truncate(self, *, forwarded, accepted):
        assert forwarded == accepted


class _Row:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return _Row(self.value)


def test_import_failure_always_falls_back_to_full_local_prefill(monkeypatch):
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    store = PrefixCacheStore(compatibility, max_bytes=1024, node_id="head")
    hashes = chained_block_hashes([1, 2], compatibility)
    store.put(CacheBlock.create(hashes[0], 2, b"corrupt"))
    hook = DistributedPrefillCacheHook(
        store,
        compression=CompressionCodec.NONE,
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "import_mlx_prefill_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad snapshot")),
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "export_mlx_prefill_snapshot",
        lambda *args, **kwargs: b"fresh",
    )
    verifier = _Verifier()
    assert hook.prepare(verifier, [1, 2]) == 0
    assert verifier.prefill_calls == 1
    assert verifier.cached_token_sequence == [1, 2]
    assert hook.stats.fallbacks == 1
    assert "bad snapshot" in hook.stats.last_fallback_reason
    assert not store.invalidate(b"missing")
    hook.close()


def test_tenant_hmac_changes_hash_namespace(monkeypatch, tmp_path):
    from inference_engine.distributed.prefill_auth import FleetAuthConfig

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "export_mlx_prefill_snapshot",
        lambda *args, **kwargs: b"snapshot",
    )
    a = CacheCompatibility(
        model_id="m", block_size_tokens=2, tenant_namespace="a",
    )
    b = CacheCompatibility(
        model_id="m", block_size_tokens=2, tenant_namespace="b",
    )
    store_a = PrefixCacheStore(a, max_bytes=1024, node_id="a")
    store_b = PrefixCacheStore(b, max_bytes=1024, node_id="b")
    auth_a = FleetAuthConfig(b"x" * 32, "a", "node")
    auth_b = FleetAuthConfig(b"x" * 32, "b", "node")
    hook_a = DistributedPrefillCacheHook(
        store_a, auth=auth_a, compression=CompressionCodec.NONE,
    )
    hook_b = DistributedPrefillCacheHook(
        store_b, auth=auth_b, compression=CompressionCodec.NONE,
    )
    hook_a.prepare(_Verifier(), [1, 2])
    hook_b.prepare(_Verifier(), [1, 2])
    assert store_a.block_hashes() != store_b.block_hashes()
    hook_a.close()
    hook_b.close()


def test_runtime_validation_empty_and_provider_failure():
    store = PrefixCacheStore(
        CacheCompatibility(model_id="m"),
        max_bytes=1024,
        node_id="head",
    )
    with __import__("pytest").raises(ValueError):
        DistributedPrefillCacheHook(store, lookup_timeout_s=0)
    with __import__("pytest").raises(ValueError):
        DistributedPrefillCacheHook(store, replication_factor=-1)
    hook = DistributedPrefillCacheHook(
        store,
        registry_provider=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert hook.prepare(_Verifier(), []) == 0
    assert hook._cards() == ()
    hook.close()


def test_successful_local_import_suffix_and_on_reuse(monkeypatch):
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    store = PrefixCacheStore(compatibility, max_bytes=1024, node_id="head")
    hashes = chained_block_hashes([1, 2, 3, 4], compatibility)
    store.put(CacheBlock.create(hashes[0], 2, b"snapshot"))
    imported = type("Imported", (), {
        "token_count": 2,
        "cached_token_ids": (1, 2),
        "next_token_logits": _Row(2),
        "block_hash": hashes[0],
    })()
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "import_mlx_prefill_snapshot",
        lambda *args, **kwargs: imported,
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "export_mlx_prefill_snapshot",
        lambda *args, **kwargs: b"fresh",
    )
    reused = []
    hook = DistributedPrefillCacheHook(
        store,
        compression=CompressionCodec.NONE,
        on_reuse=reused.append,
    )
    verifier = _Verifier()
    assert hook.prepare(verifier, [1, 2, 3, 4]) == 2
    assert verifier.cached_token_sequence == [1, 2, 3, 4]
    assert hook.stats.local_hits == 1
    assert reused == [2]
    hook.close()


def test_import_budget_and_remote_worker_failures(monkeypatch):
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    store = PrefixCacheStore(compatibility, max_bytes=1024, node_id="head")
    hook = DistributedPrefillCacheHook(
        store,
        max_import_bytes=2,
        remote_compute_min_tokens=1,
        worker_timeout_s=0.001,
        worker_poll_interval_s=0.001,
    )
    verifier = _Verifier()
    assert hook._try_import(
        verifier, [1, 2], _Hit("local", "l", 1, 1, 1, b"x"),
    ) == 0
    assert hook._try_import(
        verifier, [1, 2], _Hit("local", "l", 1, 2, 3, b"big"),
    ) == 0
    assert hook._try_import(
        verifier, [1, 2], _Hit("peer", "l", 1, 2, 3),
    ) == 0

    invalid = [
        type("Imported", (), {
            "token_count": 0,
            "cached_token_ids": (),
            "next_token_logits": _Row(1),
            "block_hash": b"h" * 32,
        })(),
        type("Imported", (), {
            "token_count": 2,
            "cached_token_ids": (1, 2),
            "next_token_logits": _Row(1),
            "block_hash": b"x" * 32,
        })(),
        type("Imported", (), {
            "token_count": 2,
            "cached_token_ids": (1, 2),
            "next_token_logits": None,
            "block_hash": b"h" * 32,
        })(),
        type("Imported", (), {
            "token_count": 2,
            "cached_token_ids": (9, 9),
            "next_token_logits": _Row(1),
            "block_hash": b"h" * 32,
        })(),
    ]
    monkeypatch.setattr(
        hook,
        "_fetch_remote",
        lambda hit: b"ok",
    )
    for imported in invalid:
        monkeypatch.setattr(
            "inference_engine.distributed.prefill_cache_runtime."
            "import_mlx_prefill_snapshot",
            lambda *args, _imported=imported, **kwargs: _imported,
        )
        assert hook._try_import(
            verifier,
            [1, 2],
            _Hit("peer", "l", 1, 2, 2, block_hash=b"h" * 32),
        ) == 0

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "choose_prefill_worker",
        lambda *args, **kwargs: None,
    )
    assert hook._compute_remote([1, 2], [b"a" * 32]) is None

    target = type("Target", (), {
        "address": "worker:1",
        "rtt_ms": 1.0,
    })()
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "choose_prefill_worker",
        lambda *args, **kwargs: target,
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "submit_prefill_job_sync",
        lambda *args, **kwargs: type("Response", (), {"job_id": "j"})(),
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "get_prefill_job_sync",
        lambda *args, **kwargs: type("Status", (), {
            "status": 4,
            "failure_reason": "failed",
        })(),
    )
    assert hook._compute_remote([1, 2], [b"a" * 32]) is None
    assert hook.stats.remote_job_failures == 1

    # A perpetually queued job reaches the bounded deadline and falls back.
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "get_prefill_job_sync",
        lambda *args, **kwargs: type("Status", (), {
            "status": 1,
            "failure_reason": "",
        })(),
    )
    assert hook._compute_remote([1, 2], [b"a" * 32]) is None
    assert hook.stats.remote_job_failures == 2
    hook.close()


def test_dynamic_replica_selection_and_cost_reject(monkeypatch):
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    card = NodeCapability(
        "peer",
        "peer:1",
        caches=(CacheCapability(compatibility, "peer:2"),),
        endpoints=(NodeEndpoint("peer:2", "lan", 1, 5),),
    )
    store = PrefixCacheStore(compatibility, max_bytes=1024, node_id="head")
    hook = DistributedPrefillCacheHook(
        store,
        registry_provider=lambda: (card,),
        replication_factor=1,
        cost_config=PrefillCostConfig(
            local_prefill_tps=10000,
            default_worker_tps=1,
            link_mbps=1,
            default_rtt_ms=10,
            minimum_savings_ratio=0.5,
        ),
    )
    assert hook._publish_peers(b"h" * 32) == ("peer:2",)
    monkeypatch.setattr(
        hook,
        "_lookup_peer",
        lambda peer, hashes: _Hit(peer, "l", 1, 1, 10_000_000),
    )
    assert hook._best_hit([b"h" * 32]) is None
    assert hook._peer_rtt("peer:2") == 5
    assert hook._peer_rtt("unknown") == 0
    hook.close()


def test_publish_boundary_dispatches_selected_replica(monkeypatch):
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    store = PrefixCacheStore(compatibility, max_bytes=1024, node_id="head")
    hook = DistributedPrefillCacheHook(store, peers=("peer:1",))
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "export_mlx_prefill_snapshot",
        lambda *args, **kwargs: b"snapshot",
    )
    calls = []

    class Publisher:
        def submit(self, fn, *args, **kwargs):
            calls.append((fn, args, kwargs))

        def shutdown(self, **kwargs):
            pass

    hook._publisher = Publisher()
    verifier = _Verifier()
    verifier.prefill([1, 2])
    hashes = chained_block_hashes([1, 2], compatibility)
    hook._publish_boundary(verifier, [1, 2], hashes, 0, 2)
    assert calls and calls[0][1][0] == "peer:1"
    hook.close()


def test_unaligned_reused_prefix_computes_boundary_remainder(monkeypatch):
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    store = PrefixCacheStore(compatibility, max_bytes=1024, node_id="head")
    hook = DistributedPrefillCacheHook(store)
    monkeypatch.setattr(hook, "_publish_boundary", lambda *args, **kwargs: None)
    verifier = _Verifier()
    verifier.prefill([1, 2, 3])
    hashes = chained_block_hashes([1, 2, 3, 4, 5, 6], compatibility)
    hook._compute_and_publish(verifier, [1, 2, 3, 4, 5, 6], hashes, 3)
    # Snapshot ends at token 3; token 4 completes that block before [5,6].
    assert verifier.cached_token_sequence[-3:] == [4, 5, 6]
    hook.close()


class _Context:
    def __init__(self, stub):
        self.stub = stub

    def __enter__(self):
        return type("Channel", (), {
            "unary_unary": lambda *args, **kwargs: None,
        })()

    def __exit__(self, *args):
        return False


def test_fetch_remote_validates_stream(monkeypatch):
    import hashlib

    compatibility = CacheCompatibility(model_id="m")
    hook = DistributedPrefillCacheHook(
        PrefixCacheStore(compatibility, max_bytes=1024, node_id="head"),
    )
    payload = b"payload"

    class Stub:
        def __init__(self, channel):
            pass

        def FetchBlocks(self, request, **kwargs):
            return [distributed_pb2.FetchBlocksResponse(
                chunk_index=0,
                total_chunks=1,
                data=payload,
                block_sha256=hashlib.sha256(payload).digest(),
            )]

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime.grpc.insecure_channel",
        lambda address: _Context(Stub),
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        Stub,
    )
    assert hook._fetch_remote(_Hit("peer", "lease", 1, 1, 7)) == payload

    class EmptyStub(Stub):
        def FetchBlocks(self, request, **kwargs):
            return []

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        EmptyStub,
    )
    with __import__("pytest").raises(RuntimeError, match="incomplete"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 7))

    class ErrorStub(Stub):
        def FetchBlocks(self, request, **kwargs):
            import grpc
            raise grpc.RpcError("down")

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        ErrorStub,
    )
    with __import__("pytest").raises(RuntimeError, match="fetch failed"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 7))

    class CorruptStub(Stub):
        def FetchBlocks(self, request, **kwargs):
            return [distributed_pb2.FetchBlocksResponse(
                chunk_index=0,
                total_chunks=1,
                data=payload,
                block_sha256=b"x" * 32,
            )]

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        CorruptStub,
    )
    with __import__("pytest").raises(RuntimeError, match="checksum"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 7))

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        Stub,
    )
    with __import__("pytest").raises(RuntimeError, match="lease checksum"):
        hook._fetch_remote(_Hit(
            "peer",
            "lease",
            1,
            1,
            7,
            payload_sha256=b"x" * 32,
        ))

    with __import__("pytest").raises(RuntimeError, match="import budget"):
        hook._fetch_remote(
            _Hit(
                "peer", "lease", 1, 1, hook.max_import_bytes + 1,
            ),
        )

    class ChangedMetadataStub(Stub):
        def FetchBlocks(self, request, **kwargs):
            digest = hashlib.sha256(b"ab").digest()
            return [
                distributed_pb2.FetchBlocksResponse(
                    chunk_index=0, total_chunks=2, data=b"a",
                    block_hash=b"h", block_sha256=digest,
                ),
                distributed_pb2.FetchBlocksResponse(
                    chunk_index=1, total_chunks=3, data=b"b",
                    block_hash=b"h", block_sha256=digest,
                ),
            ]

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        ChangedMetadataStub,
    )
    with __import__("pytest").raises(RuntimeError, match="metadata changed"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 2))

    class TooManyChunksStub(Stub):
        def FetchBlocks(self, request, **kwargs):
            return [distributed_pb2.FetchBlocksResponse(
                chunk_index=0,
                total_chunks=65_537,
                data=b"",
                block_hash=b"h",
                block_sha256=hashlib.sha256(b"").digest(),
            )]

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        TooManyChunksStub,
    )
    with __import__("pytest").raises(RuntimeError, match="chunk count"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 0))

    class DuplicateStub(Stub):
        def FetchBlocks(self, request, **kwargs):
            digest = hashlib.sha256(b"aa").digest()
            return [
                distributed_pb2.FetchBlocksResponse(
                    chunk_index=0, total_chunks=2, data=b"a",
                    block_hash=b"h", block_sha256=digest,
                ),
                distributed_pb2.FetchBlocksResponse(
                    chunk_index=0, total_chunks=2, data=b"a",
                    block_hash=b"h", block_sha256=digest,
                ),
            ]

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        DuplicateStub,
    )
    with __import__("pytest").raises(RuntimeError, match="duplicate"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 2))

    hook.max_import_bytes = 1
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        Stub,
    )
    with __import__("pytest").raises(RuntimeError, match="import budget"):
        hook._fetch_remote(_Hit("peer", "lease", 1, 1, 1))
    hook.close()


def test_lookup_peer_success_miss_and_rpc_error(monkeypatch):
    import grpc

    compatibility = CacheCompatibility(model_id="m")
    hook = DistributedPrefillCacheHook(
        PrefixCacheStore(compatibility, max_bytes=1024, node_id="head"),
    )

    class LookupStub:
        response = distributed_pb2.LookupPrefixResponse(
            lease_id="lease",
            hit_block_count=1,
            hit_token_count=5,
            transfer_bytes=7,
        )

        def __init__(self, channel):
            pass

        def LookupPrefix(self, request, **kwargs):
            return self.response

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime.grpc.insecure_channel",
        lambda address: _Context(LookupStub),
    )
    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        LookupStub,
    )
    hit = hook._lookup_peer("peer", [b"h" * 32])
    assert hit is not None and hit.hit_tokens == 5
    LookupStub.response = distributed_pb2.LookupPrefixResponse()
    assert hook._lookup_peer("peer", [b"h" * 32]) is None
    LookupStub.response = distributed_pb2.LookupPrefixResponse(
        lease_id="lease",
        hit_block_count=2,
    )
    assert hook._lookup_peer("peer", [b"h" * 32]) is None

    class ErrorStub(LookupStub):
        def LookupPrefix(self, request, **kwargs):
            raise grpc.RpcError("down")

    monkeypatch.setattr(
        "inference_engine.distributed.prefill_cache_runtime."
        "distributed_pb2_grpc.PrefillCacheServiceStub",
        ErrorStub,
    )
    assert hook._lookup_peer("peer", [b"h" * 32]) is None
    hook.close()

