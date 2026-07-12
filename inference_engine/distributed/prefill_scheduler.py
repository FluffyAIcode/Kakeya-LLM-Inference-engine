"""Pure placement/cost functions for distributed prefill compute and storage."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from inference_engine.distributed.capability import (
    CacheCompatibility,
    NodeCapability,
    PrefillWorkerCapability,
)


class PrefillAction(str, Enum):
    IMPORT = "import"
    REMOTE_COMPUTE = "remote_compute"
    LOCAL_COMPUTE = "local_compute"


@dataclass(frozen=True)
class PrefillCostConfig:
    local_prefill_tps: float = 20.0
    default_worker_tps: float = 20.0
    link_mbps: float = 1000.0
    default_rtt_ms: float = 2.0
    minimum_savings_ratio: float = 0.10
    primary_compute_penalty_ms: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.local_prefill_tps,
            self.default_worker_tps,
            self.link_mbps,
            self.default_rtt_ms,
        ) <= 0:
            raise ValueError("prefill cost metrics must be > 0")
        if not (0 <= self.minimum_savings_ratio < 1):
            raise ValueError("minimum_savings_ratio must be in [0, 1)")
        if self.primary_compute_penalty_ms < 0:
            raise ValueError("primary_compute_penalty_ms must be >= 0")


@dataclass(frozen=True)
class WorkerTarget:
    node_id: str
    address: str
    capability: PrefillWorkerCapability
    rtt_ms: float

    @property
    def queue_eta_ms(self) -> float:
        tps = self.capability.tokens_per_second_prefill
        if tps <= 0:
            return 0.0
        return self.capability.queued_tokens * 1000.0 / tps


def estimate_local_prefill_ms(tokens: int, config: PrefillCostConfig) -> float:
    return (
        max(0, tokens) / config.local_prefill_tps * 1000.0
        + config.primary_compute_penalty_ms
    )


def estimate_import_ms(
    transfer_bytes: int,
    *,
    rtt_ms: float,
    config: PrefillCostConfig,
) -> float:
    bytes_per_ms = config.link_mbps * 1_000_000.0 / 8.0 / 1000.0
    return max(rtt_ms, 0.0) + max(transfer_bytes, 0) / bytes_per_ms


def remote_import_wins(
    *,
    hit_tokens: int,
    transfer_bytes: int,
    rtt_ms: float,
    config: PrefillCostConfig,
) -> bool:
    local = estimate_local_prefill_ms(hit_tokens, config)
    remote = estimate_import_ms(
        transfer_bytes,
        rtt_ms=rtt_ms or config.default_rtt_ms,
        config=config,
    )
    return remote <= local * (1.0 - config.minimum_savings_ratio)


def compatible_prefill_workers(
    cards: Sequence[NodeCapability],
    compatibility: CacheCompatibility,
) -> list[WorkerTarget]:
    targets: list[WorkerTarget] = []
    for card in cards:
        rtt = min(
            (
                endpoint.measured_rtt_ms
                for endpoint in card.endpoints
                if endpoint.measured_rtt_ms > 0
            ),
            default=0.0,
        )
        for worker in card.prefill_workers:
            if (
                worker.accepts_compute_jobs
                and worker.compatibility == compatibility
                and (worker.worker_address or card.grpc_address)
            ):
                targets.append(WorkerTarget(
                    card.node_id,
                    worker.worker_address or card.grpc_address,
                    worker,
                    rtt,
                ))
    return targets


def choose_prefill_worker(
    cards: Sequence[NodeCapability],
    compatibility: CacheCompatibility,
    *,
    prompt_tokens: int,
    estimated_snapshot_bytes: int,
    config: PrefillCostConfig,
) -> WorkerTarget | None:
    candidates = compatible_prefill_workers(cards, compatibility)
    if not candidates:
        return None
    local_ms = estimate_local_prefill_ms(prompt_tokens, config)

    def cost(target: WorkerTarget) -> float:
        tps = (
            target.capability.tokens_per_second_prefill
            or config.default_worker_tps
        )
        compute_ms = prompt_tokens / tps * 1000.0
        import_ms = estimate_import_ms(
            estimated_snapshot_bytes,
            rtt_ms=target.rtt_ms or config.default_rtt_ms,
            config=config,
        )
        load_penalty = max(0.0, target.capability.load) * compute_ms
        return target.queue_eta_ms + compute_ms + import_ms + load_penalty

    best = min(candidates, key=lambda target: (
        cost(target),
        -target.capability.ram_bytes_free,
        target.node_id,
    ))
    if cost(best) > local_ms * (1.0 - config.minimum_savings_ratio):
        return None
    return best


def select_cache_replicas(
    cards: Sequence[NodeCapability],
    compatibility: CacheCompatibility,
    *,
    block_hash: bytes,
    replication_factor: int,
) -> list[str]:
    """Deterministic rendezvous placement avoids publishing to every peer."""
    if replication_factor <= 0:
        return []
    candidates: list[tuple[int, int, str]] = []
    for card in cards:
        for cache in card.caches:
            if cache.compatibility != compatibility or not cache.cache_address:
                continue
            score = int.from_bytes(hashlib.sha256(
                bytes(block_hash) + card.node_id.encode(),
            ).digest(), "big")
            candidates.append((
                score,
                cache.cache_bytes_free,
                cache.cache_address,
            ))
    candidates.sort(reverse=True)
    result: list[str] = []
    for _score, _free, address in candidates:
        if address not in result:
            result.append(address)
        if len(result) >= replication_factor:
            break
    return result

