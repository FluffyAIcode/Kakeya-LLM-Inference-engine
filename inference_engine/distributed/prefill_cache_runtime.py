"""Synchronous runtime hook that applies distributed prefill-cache hits.

The gRPC RuntimeService uses a synchronous verifier underneath its asyncio
handlers. This hook keeps that contract: peer lookups run concurrently in a
small thread pool, a winning snapshot is imported once, and missing suffix
blocks are prefetched locally before decode begins.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import grpc

from inference_engine.backends.mlx.prefill_snapshot import (
    export_mlx_prefill_snapshot,
    import_mlx_prefill_snapshot,
)
from inference_engine.distributed.capability import (
    CacheCompatibility,
    CompressionCodec,
    NodeCapability,
)
from inference_engine.distributed.prefill_auth import (
    FleetAuthConfig,
    signed_metadata,
)
from inference_engine.distributed.prefill_cache import (
    CacheBlock,
    PrefixCacheStore,
    chained_block_hashes,
)
from inference_engine.distributed.prefill_compression import (
    compress_payload,
    decompress_payload,
)
from inference_engine.distributed.prefill_scheduler import (
    PrefillCostConfig,
    choose_prefill_worker,
    compatible_prefill_workers,
    remote_import_wins,
    select_cache_replicas,
)
from inference_engine.distributed.prefill_worker import (
    PrefillJobState,
    cancel_prefill_job_sync,
    get_prefill_job_sync,
    submit_prefill_job_sync,
)
from inference_engine.distributed.prefill_cache_service import (
    compatible_cache_peers,
)
from inference_engine.server.proto_gen.kakeya.v1 import (
    distributed_pb2,
    distributed_pb2_grpc,
)


@dataclass
class PrefillReuseStats:
    local_hits: int = 0
    remote_hits: int = 0
    misses: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0
    bytes_received: int = 0
    remote_jobs: int = 0
    remote_job_failures: int = 0
    remote_job_tokens_total: int = 0
    remote_job_tokens_computed: int = 0
    fallbacks: int = 0
    last_fallback_reason: str = ""
    publish_attempts: int = 0
    publish_successes: int = 0
    publish_failures: int = 0
    bytes_published: int = 0
    last_publish_error: str = ""
    hot_promotions: int = 0
    hot_promotion_bytes: int = 0
    hot_promotion_failures: int = 0


class RemotePrefillRequiredError(RuntimeError):
    """Raised when decode-only primary policy cannot obtain a complete KV."""


@dataclass(frozen=True)
class _Hit:
    source: str
    lease_id: str
    hit_blocks: int
    hit_tokens: int
    transfer_bytes: int
    payload: bytes | None = None
    rtt_ms: float = 0.0
    block_hash: bytes = b""
    payload_sha256: bytes = b""


class DistributedPrefillCacheHook:
    """Prepare a cold verifier using local/remote longest-prefix snapshots."""

    def __init__(
        self,
        local_store: PrefixCacheStore,
        *,
        peers: Sequence[str] = (),
        registry_provider: Callable[[], Sequence[NodeCapability]] | None = None,
        lookup_timeout_s: float = 2.0,
        fetch_timeout_s: float = 30.0,
        worker_timeout_s: float = 120.0,
        worker_poll_interval_s: float = 0.05,
        remote_compute_min_tokens: int = 128,
        max_import_bytes: int = 1 << 30,
        estimated_snapshot_bytes_per_token: int = 400_000,
        compression: CompressionCodec = CompressionCodec.ZLIB,
        replication_factor: int = 1,
        cost_config: PrefillCostConfig | None = None,
        auth: FleetAuthConfig | None = None,
        on_reuse=None,
        require_remote_compute: bool = False,
    ) -> None:
        if min(
            lookup_timeout_s,
            fetch_timeout_s,
            worker_timeout_s,
            worker_poll_interval_s,
            max_import_bytes,
            estimated_snapshot_bytes_per_token,
        ) <= 0:
            raise ValueError("prefill runtime limits must be > 0")
        if remote_compute_min_tokens < 0 or replication_factor < 0:
            raise ValueError("prefill thresholds must be >= 0")
        self.local_store = local_store
        self.compatibility = local_store.compatibility
        self.peers = tuple(dict.fromkeys(peer for peer in peers if peer))
        self.registry_provider = registry_provider
        self.lookup_timeout_s = float(lookup_timeout_s)
        self.fetch_timeout_s = float(fetch_timeout_s)
        self.worker_timeout_s = float(worker_timeout_s)
        self.worker_poll_interval_s = float(worker_poll_interval_s)
        self.remote_compute_min_tokens = int(remote_compute_min_tokens)
        self.max_import_bytes = int(max_import_bytes)
        self.estimated_snapshot_bytes_per_token = int(
            estimated_snapshot_bytes_per_token,
        )
        self.compression = CompressionCodec(compression)
        self.replication_factor = int(replication_factor)
        self.cost_config = cost_config or PrefillCostConfig()
        self.auth = auth
        self.require_remote_compute = bool(require_remote_compute)
        self._hash_key = auth.tenant_hash_key() if auth is not None else b""
        self.stats = PrefillReuseStats()
        self._stats_lock = threading.Lock()
        self._on_reuse = on_reuse
        self._publisher = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="prefill-kv-publish",
        )

    def prepare(self, verifier: Any, token_ids: Sequence[int]) -> int:
        """Restore the longest prefix, compute suffix, publish all new boundaries.

        Returns the number of tokens reused from cache.
        """
        tokens = [int(token) for token in token_ids]
        if not tokens:
            return 0
        hashes = chained_block_hashes(
            tokens,
            self.compatibility,
            hmac_key=self._hash_key,
        )
        hit = self._best_hit(hashes)
        reused = 0
        if (
            self.require_remote_compute
            and hit is not None
            and hit.hit_tokens != len(tokens)
        ):
            hit = None
        if hit is not None:
            reused = self._try_import(verifier, tokens, hit)
        if reused == 0 and (
            self.require_remote_compute
            or len(tokens) >= self.remote_compute_min_tokens
        ):
            remote_hit = self._compute_remote(tokens, hashes)
            if remote_hit is not None:
                reused = self._try_import(verifier, tokens, remote_hit)
        if reused == 0:
            self.stats.misses += 1
        if self.require_remote_compute and reused != len(tokens):
            verifier.reset()
            reason = self.stats.last_fallback_reason or (
                "no compatible remote prefill worker completed the request"
            )
            raise RemotePrefillRequiredError(reason)

        self._compute_and_publish(verifier, tokens, hashes, reused)
        return reused

    def _try_import(
        self,
        verifier: Any,
        tokens: list[int],
        hit: _Hit,
    ) -> int:
        try:
            expected_hit_tokens = min(
                hit.hit_blocks * self.compatibility.block_size_tokens,
                len(tokens),
            )
            if (
                hit.hit_blocks <= 0
                or hit.hit_tokens != expected_hit_tokens
            ):
                raise ValueError("prefill hit does not match a token boundary")
            if (
                hit.payload is None
                and hit.transfer_bytes > self.max_import_bytes
            ):
                raise ValueError("prefill snapshot exceeds wire import budget")
            payload = (
                hit.payload
                if hit.payload is not None
                else self._fetch_remote(hit)
            )
            if len(payload) > self.max_import_bytes:
                raise ValueError("prefill snapshot exceeds wire import budget")
            raw = decompress_payload(
                payload,
                max_uncompressed_bytes=self.max_import_bytes,
            )
            import_remote = getattr(verifier, "import_snapshot", None)
            if import_remote is not None:
                state = import_remote(raw, self.compatibility)
                imported_token_count = int(state["next_global_position"])
                imported_cached_ids = tuple(state["cached_token_ids"])
                imported_block_hash = bytes.fromhex(
                    state.get("block_hash", hit.block_hash.hex()),
                )
                imported_has_logits = True
            else:
                verifier.reset()
                imported = import_mlx_prefill_snapshot(
                    raw,
                    verifier.cache,
                    compatibility=self.compatibility,
                )
                imported_token_count = imported.token_count
                imported_cached_ids = imported.cached_token_ids
                imported_block_hash = imported.block_hash
                imported_has_logits = imported.next_token_logits is not None
            if (
                not (0 < imported_token_count <= len(tokens))
                or imported_token_count != hit.hit_tokens
            ):
                raise ValueError("prefill snapshot token_count is invalid")
            if hit.block_hash and imported_block_hash != hit.block_hash:
                raise ValueError("prefill snapshot block hash mismatch")
            if not imported_has_logits:
                raise ValueError("prefill snapshot is missing continuation logits")
            reused = min(imported_token_count, len(tokens))
            expected_prefix = tokens[:reused]
            sink_window = getattr(verifier, "_sink_window_slice", None)
            expected_cached = (
                list(sink_window(expected_prefix))
                if callable(sink_window)
                else expected_prefix
            )
            if list(imported_cached_ids) != expected_cached:
                raise ValueError("prefill snapshot cached token sequence mismatch")
            verifier.cached_token_sequence = list(imported_cached_ids)
            verifier.next_global_position = reused
            if import_remote is None and imported.next_token_logits is not None:
                verifier.next_token_logits = imported.next_token_logits
            self.stats.tokens_reused += reused
            if self._on_reuse is not None:
                self._on_reuse(reused)
            if hit.source == "local":
                self.stats.local_hits += 1
            else:
                self.stats.remote_hits += 1
                self._promote_remote_hit(hit, payload, reused)
            return reused
        except Exception as exc:
            # Cache is an optimization. A corrupt/expired/unreachable hit must
            # never determine request correctness.
            self.stats.fallbacks += 1
            self.stats.last_fallback_reason = f"{type(exc).__name__}: {exc}"
            if hit.source == "local" and hit.block_hash:
                self.local_store.invalidate(hit.block_hash)
            verifier.reset()
            return 0

    def _promote_remote_hit(
        self,
        hit: _Hit,
        payload: bytes,
        token_count: int,
    ) -> None:
        if not hit.block_hash:
            return
        try:
            stored = self.local_store.put(CacheBlock.create(
                hit.block_hash,
                token_count,
                payload,
            ))
        except ValueError:
            self.stats.hot_promotion_failures += 1
            return
        if stored:
            self.stats.hot_promotions += 1
            self.stats.hot_promotion_bytes += len(payload)

    def _compute_remote(
        self,
        tokens: list[int],
        hashes: list[bytes],
    ) -> _Hit | None:
        cards = tuple(
            card for card in self._cards()
            if card.node_id != self.local_store.node_id
        )
        if self.require_remote_compute:
            candidates = compatible_prefill_workers(cards, self.compatibility)
            target = min(
                candidates,
                key=lambda item: (
                    item.capability.load,
                    item.capability.queued_tokens,
                    -item.capability.tokens_per_second_prefill,
                    item.node_id,
                ),
                default=None,
            )
        else:
            target = choose_prefill_worker(
                cards,
                self.compatibility,
                prompt_tokens=len(tokens),
                estimated_snapshot_bytes=(
                    len(tokens) * self.estimated_snapshot_bytes_per_token
                ),
                config=self.cost_config,
            )
        if target is None:
            return None
        request = distributed_pb2.SubmitPrefillJobRequest(
            request_id=uuid.uuid4().hex,
            tenant_id=self.compatibility.tenant_namespace or "default",
            compatibility=self.compatibility.to_proto(),
            token_ids=tokens,
            block_hashes=hashes,
            deadline_ms=int(self.worker_timeout_s * 1000),
            preferred_compression=int(self.compression),
        )
        try:
            response = None
            response = submit_prefill_job_sync(
                target.address,
                request,
                timeout_s=self.lookup_timeout_s,
                auth=self.auth,
            )
            self.stats.remote_jobs += 1
            self.stats.remote_job_tokens_total = len(tokens)
            self.stats.remote_job_tokens_computed = 0
            deadline = time.monotonic() + self.worker_timeout_s
            while time.monotonic() < deadline:
                status_request = distributed_pb2.GetPrefillJobStatusRequest(
                    job_id=response.job_id,
                    tenant_id=request.tenant_id,
                )
                status = get_prefill_job_sync(
                    target.address,
                    status_request,
                    timeout_s=self.lookup_timeout_s,
                    auth=self.auth,
                )
                self.stats.remote_job_tokens_computed = min(
                    len(tokens),
                    int(status.tokens_computed),
                )
                if status.status == int(PrefillJobState.COMPLETED):
                    return _Hit(
                        source=status.cache_address or target.address,
                        lease_id=status.lease_id,
                        hit_blocks=len(hashes),
                        hit_tokens=status.tokens_computed,
                        transfer_bytes=status.transfer_bytes,
                        rtt_ms=target.rtt_ms,
                        block_hash=hashes[-1],
                        payload_sha256=bytes(status.payload_sha256),
                    )
                if status.status in (
                    int(PrefillJobState.FAILED),
                    int(PrefillJobState.CANCELLED),
                ):
                    raise RuntimeError(
                        status.failure_reason or "remote prefill job failed",
                    )
                time.sleep(self.worker_poll_interval_s)
            raise TimeoutError("remote prefill job timed out")
        except Exception as exc:
            if response is not None:
                try:
                    cancel_request = distributed_pb2.CancelPrefillJobRequest(
                        job_id=response.job_id,
                        tenant_id=request.tenant_id,
                    )
                    cancel_prefill_job_sync(
                        target.address,
                        cancel_request,
                        timeout_s=self.lookup_timeout_s,
                        auth=self.auth,
                    )
                except Exception:
                    pass
            self.stats.remote_job_failures += 1
            self.stats.fallbacks += 1
            self.stats.last_fallback_reason = f"{type(exc).__name__}: {exc}"
            return None

    def _compute_and_publish(
        self,
        verifier: Any,
        tokens: list[int],
        hashes: list[bytes],
        reused: int,
    ) -> None:
        size = self.compatibility.block_size_tokens
        if reused == 0:
            first_end = min(size, len(tokens))
            verifier.prefill(tokens[:first_end])
            self.stats.tokens_computed += first_end
            if not getattr(verifier, "is_decode_worker_proxy", False):
                self._publish_boundary(verifier, tokens, hashes, 0, first_end)
            reused = first_end
        cursor = reused
        while cursor < len(tokens):
            block_index = cursor // size
            end = min((block_index + 1) * size, len(tokens))
            block_tokens = tokens[cursor:end]
            append_accepted = getattr(verifier, "append_accepted_tokens", None)
            if append_accepted is not None:
                append_accepted(block_tokens)
            else:
                logits = verifier.forward_block(block_tokens)
                verifier.commit_or_truncate(
                    forwarded=len(block_tokens),
                    accepted=len(block_tokens),
                )
                verifier.next_token_logits = logits[-1].clone()
            self.stats.tokens_computed += len(block_tokens)
            if not getattr(verifier, "is_decode_worker_proxy", False):
                self._publish_boundary(verifier, tokens, hashes, block_index, end)
            cursor = end

    def _publish_boundary(
        self,
        verifier: Any,
        tokens: list[int],
        hashes: list[bytes],
        block_index: int,
        prefix_end: int,
    ) -> None:
        raw_payload = export_mlx_prefill_snapshot(
            verifier.cache,
            token_count=prefix_end,
            cached_token_ids=verifier.cached_token_sequence,
            compatibility=self.compatibility,
            next_token_logits=verifier.next_token_logits,
            block_hash=hashes[block_index],
        )
        payload = compress_payload(raw_payload, self.compression)
        block = CacheBlock.create(hashes[block_index], prefix_end, payload)
        self.local_store.put(block)
        peers = self._publish_peers(block.block_hash)
        if peers:
            from inference_engine.distributed.prefill_cache_service import (
                publish_block_sync,
            )
            for peer in peers:
                with self._stats_lock:
                    self.stats.publish_attempts += 1
                future = self._publisher.submit(
                    publish_block_sync,
                    peer,
                    self.compatibility,
                    block,
                    timeout_s=self.fetch_timeout_s,
                    auth=self.auth,
                )
                future.add_done_callback(
                    lambda completed, nbytes=block.nbytes: self._publish_done(
                        completed,
                        nbytes,
                    ),
                )

    def _publish_done(self, future, nbytes: int) -> None:
        try:
            stored = bool(future.result())
        except Exception as exc:
            with self._stats_lock:
                self.stats.publish_failures += 1
                self.stats.last_publish_error = f"{type(exc).__name__}: {exc}"
            return
        with self._stats_lock:
            self.stats.publish_successes += 1
            if stored:
                self.stats.bytes_published += int(nbytes)

    def close(self) -> None:
        self._publisher.shutdown(wait=False, cancel_futures=True)

    def _cards(self) -> tuple[NodeCapability, ...]:
        if self.registry_provider is None:
            return ()
        try:
            return tuple(self.registry_provider())
        except Exception:
            return ()

    def _cache_peers(self) -> tuple[str, ...]:
        dynamic = compatible_cache_peers(
            tuple(
                card for card in self._cards()
                if card.node_id != self.local_store.node_id
            ),
            self.compatibility,
        )
        return tuple(dict.fromkeys((*self.peers, *dynamic)))

    def _publish_peers(self, block_hash: bytes) -> tuple[str, ...]:
        cards = tuple(
            card for card in self._cards()
            if card.node_id != self.local_store.node_id
        )
        if cards:
            selected = select_cache_replicas(
                cards,
                self.compatibility,
                block_hash=block_hash,
                replication_factor=self.replication_factor,
            )
            if selected:
                return tuple(selected)
        return self._cache_peers()[:self.replication_factor]

    def _best_hit(self, hashes: Sequence[bytes]) -> _Hit | None:
        candidates: list[_Hit] = []
        local = self.local_store.lookup(hashes)
        if local.lease_id:
            blocks = self.local_store.fetch(local.lease_id)
            candidates.append(_Hit(
                source="local",
                lease_id=local.lease_id,
                hit_blocks=local.hit_block_count,
                hit_tokens=local.hit_token_count,
                transfer_bytes=local.transfer_bytes,
                payload=blocks[-1].payload,
                block_hash=blocks[-1].block_hash,
                payload_sha256=blocks[-1].payload_sha256,
            ))
        peers = self._cache_peers()
        if peers:
            with ThreadPoolExecutor(max_workers=min(8, len(peers))) as pool:
                futures = {
                    pool.submit(self._lookup_peer, peer, hashes): peer
                    for peer in peers
                }
                for future in as_completed(futures):
                    hit = future.result()
                    if hit is not None:
                        candidates.append(hit)
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda hit: (
                hit.hit_tokens,
                hit.source == "local",
                -hit.transfer_bytes,
            ),
        )
        if (
            best.source != "local"
            and not remote_import_wins(
                hit_tokens=best.hit_tokens,
                transfer_bytes=best.transfer_bytes,
                rtt_ms=best.rtt_ms,
                config=self.cost_config,
            )
        ):
            local = next(
                (candidate for candidate in candidates if candidate.source == "local"),
                None,
            )
            return local
        return best

    def _lookup_peer(self, peer: str, hashes: Sequence[bytes]) -> _Hit | None:
        try:
            with grpc.insecure_channel(peer) as channel:
                stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
                request = distributed_pb2.LookupPrefixRequest(
                    compatibility=self.compatibility.to_proto(),
                    block_hashes=hashes,
                )
                response = stub.LookupPrefix(
                    request,
                    timeout=self.lookup_timeout_s,
                    metadata=(
                        signed_metadata(request, self.auth)
                        if self.auth is not None else None
                    ),
                )
        except grpc.RpcError:
            return None
        if not response.lease_id or response.hit_block_count == 0:
            return None
        if response.hit_block_count > len(hashes):
            return None
        return _Hit(
            source=peer,
            lease_id=response.lease_id,
            hit_blocks=response.hit_block_count,
            hit_tokens=response.hit_token_count,
            transfer_bytes=response.transfer_bytes,
            rtt_ms=self._peer_rtt(peer),
            block_hash=bytes(hashes[response.hit_block_count - 1]),
            payload_sha256=bytes(response.payload_sha256),
        )

    def _peer_rtt(self, address: str) -> float:
        for card in self._cards():
            for endpoint in card.endpoints:
                if endpoint.address == address:
                    return endpoint.measured_rtt_ms
        return 0.0

    def _fetch_remote(self, hit: _Hit) -> bytes:
        if hit.transfer_bytes > self.max_import_bytes:
            raise RuntimeError("remote prefill payload exceeds import budget")
        parts: dict[int, bytes] = {}
        expected_chunks = 0
        expected_sha = b""
        expected_block_hash = b""
        received = 0
        try:
            with grpc.insecure_channel(hit.source) as channel:
                stub = distributed_pb2_grpc.PrefillCacheServiceStub(channel)
                request = distributed_pb2.FetchBlocksRequest(
                    lease_id=hit.lease_id,
                )
                for chunk in stub.FetchBlocks(
                    request,
                    timeout=self.fetch_timeout_s,
                    metadata=(
                        signed_metadata(request, self.auth)
                        if self.auth is not None else None
                    ),
                ):
                    if expected_chunks == 0:
                        expected_chunks = chunk.total_chunks
                        if expected_chunks > 65_536:
                            raise RuntimeError(
                                "remote prefill cache chunk count exceeds limit",
                            )
                        expected_sha = bytes(chunk.block_sha256)
                        expected_block_hash = bytes(chunk.block_hash)
                    elif (
                        chunk.total_chunks != expected_chunks
                        or bytes(chunk.block_sha256) != expected_sha
                        or bytes(chunk.block_hash) != expected_block_hash
                    ):
                        raise RuntimeError(
                            "remote prefill cache chunk metadata changed",
                        )
                    if (
                        chunk.chunk_index < 0
                        or chunk.chunk_index >= expected_chunks
                        or chunk.chunk_index in parts
                    ):
                        raise RuntimeError("invalid or duplicate prefill chunk")
                    data = bytes(chunk.data)
                    received += len(data)
                    if received > self.max_import_bytes:
                        raise RuntimeError(
                            "remote prefill payload exceeds import budget",
                        )
                    parts[chunk.chunk_index] = data
        except grpc.RpcError as exc:
            raise RuntimeError(f"remote prefill cache fetch failed: {exc}") from exc
        if expected_chunks <= 0 or len(parts) != expected_chunks:
            raise RuntimeError("remote prefill cache stream was incomplete")
        if hit.payload_sha256 and expected_sha != hit.payload_sha256:
            raise RuntimeError("remote prefill cache lease checksum changed")
        payload = b"".join(parts[index] for index in range(expected_chunks))
        if hashlib.sha256(payload).digest() != expected_sha:
            raise RuntimeError("remote prefill cache checksum mismatch")
        self.stats.bytes_received += len(payload)
        return payload
