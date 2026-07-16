"""Persistent product state projected from the P2P capability plane."""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from inference_engine.distributed.capability import CapabilityRegistry
from inference_engine.distributed.kv_namespace import VirtualKVNamespace
from inference_engine.distributed.prefill_cache import PrefixCacheStore


class NetworkState:
    def __init__(
        self,
        registry: CapabilityRegistry,
        cache_store: PrefixCacheStore,
        *,
        state_path: str | Path,
        prefill_stats_provider: Callable[[], Any] | None = None,
    ) -> None:
        self.registry = registry
        self.cache_store = cache_store
        self.kv_namespace = VirtualKVNamespace(cache_store.compatibility)
        self.state_path = Path(state_path).expanduser()
        self.prefill_stats_provider = prefill_stats_provider
        self._lock = threading.RLock()
        self._data = self._load()

    def register_node(
        self,
        *,
        alias: str,
        address: str,
        region: str,
        role: str = "hybrid",
    ) -> dict[str, Any]:
        if not alias or not address:
            raise ValueError("alias and address are required")
        now = time.time()
        item = {
            "id": secrets.token_hex(8),
            "alias": alias,
            "address": address,
            "region": region or "Private",
            "role": role,
            "status": "pending",
            "pairing_token": "kn_pair_" + secrets.token_urlsafe(18),
            "expires_at": now + 600,
            "created_at": now,
        }
        with self._lock:
            self._data["registrations"].append(item)
            self._save()
        return dict(item)

    def create_group(self, *, name: str, node_ids: list[str]) -> dict[str, Any]:
        if not name or not node_ids:
            raise ValueError("name and node_ids are required")
        group = {
            "id": secrets.token_hex(6),
            "name": name,
            "node_ids": list(dict.fromkeys(node_ids)),
            "created_at": time.time(),
        }
        with self._lock:
            self._data["groups"].append(group)
            self._save()
        return dict(group)

    def record_tokens(
        self,
        *,
        node_id: str,
        completed: int,
        kv_assisted: int = 0,
    ) -> None:
        if completed < 0 or kv_assisted < 0 or kv_assisted > completed:
            raise ValueError("invalid token counters")
        with self._lock:
            counters = self._data["tokens"].setdefault(
                node_id,
                {"completed": 0, "kv_assisted": 0},
            )
            counters["completed"] += int(completed)
            counters["kv_assisted"] += int(kv_assisted)
            self._save()

    def nodes(self) -> list[dict[str, Any]]:
        registrations = {
            item["alias"]: item
            for item in self._data["registrations"]
        }
        output: list[dict[str, Any]] = []
        for card in self.registry.snapshot():
            registration = registrations.get(card.node_id, {})
            cache = card.caches[0] if card.caches else None
            worker = card.prefill_workers[0] if card.prefill_workers else None
            endpoint = sorted(
                card.endpoints,
                key=lambda item: item.priority,
                reverse=True,
            )
            output.append({
                "id": card.node_id,
                "alias": card.node_id,
                "region": registration.get("region", "Private"),
                "role": registration.get(
                    "role",
                    "hybrid" if cache else "inference",
                ),
                "status": "online",
                "platform": card.platform,
                "memory_bytes": card.unified_memory_bytes,
                "models": [
                    {
                        "model_id": model.model_id,
                        "role": model.role.name.lower(),
                        "quantization": model.quantization,
                        "tokens_per_second": model.tokens_per_second,
                    }
                    for model in card.models
                ],
                "cache": (
                    {
                        "bytes_used": cache.cache_bytes_used,
                        "bytes_free": cache.cache_bytes_free,
                        "entry_count": cache.entry_count,
                        "epoch": cache.cache_epoch,
                        "tokens_served": cache.tokens_served,
                        "evictions": cache.evictions,
                        "bytes_evicted": cache.bytes_evicted,
                        "put_failures": cache.put_failures,
                        "format": cache.compatibility.cache_format_version,
                        "model_id": cache.compatibility.model_id,
                    }
                    if cache else None
                ),
                "prefill_worker": (
                    {
                        "address": worker.worker_address,
                        "max_concurrent_jobs": worker.max_concurrent_jobs,
                        "inflight_jobs": worker.inflight_jobs,
                        "queued_jobs": worker.queued_jobs,
                        "queued_tokens": worker.queued_tokens,
                        "load": worker.load,
                        "tokens_per_second": worker.tokens_per_second_prefill,
                        "ram_bytes_free": worker.ram_bytes_free,
                    }
                    if worker else None
                ),
                "endpoint": (
                    {
                        "address": endpoint[0].address,
                        "network": endpoint[0].network,
                        "priority": endpoint[0].priority,
                        "rtt_ms": endpoint[0].measured_rtt_ms,
                    }
                    if endpoint else {
                        "address": card.grpc_address,
                        "network": "default",
                        "priority": 0,
                        "rtt_ms": 0,
                    }
                ),
            })
        live_ids = {item["id"] for item in output}
        for registration in self._data["registrations"]:
            if registration["alias"] not in live_ids:
                output.append({
                    "id": registration["alias"],
                    "alias": registration["alias"],
                    "region": registration["region"],
                    "role": registration["role"],
                    "status": registration["status"],
                    "platform": "",
                    "memory_bytes": 0,
                    "models": [],
                    "cache": None,
                    "prefill_worker": None,
                    "endpoint": {
                        "address": registration["address"],
                        "network": "pending",
                        "priority": 0,
                        "rtt_ms": 0,
                    },
                })
        return output

    def groups(self) -> list[dict[str, Any]]:
        nodes = {node["id"]: node for node in self.nodes()}
        groups = []
        for group in self._data["groups"]:
            members = [nodes[node_id] for node_id in group["node_ids"] if node_id in nodes]
            groups.append({
                **group,
                "members": members,
                "online": sum(member["status"] == "online" for member in members),
            })
        return groups

    def summary(self) -> dict[str, Any]:
        nodes = self.nodes()
        counters = list(self._data["tokens"].values())
        cache_stats = self.cache_store.stats()
        completed = sum(item["completed"] for item in counters)
        assisted = sum(item["kv_assisted"] for item in counters)
        return {
            "online_nodes": sum(node["status"] == "online" for node in nodes),
            "registered_nodes": len(nodes),
            "groups": len(self._data["groups"]),
            "completed_tokens": completed,
            "kv_assisted_tokens": assisted,
            "kv_hit_rate": (assisted / completed if completed else 0.0),
            "cache_bytes_used": sum(
                (node["cache"] or {}).get("bytes_used", 0) for node in nodes
            ),
            "cache_bytes_free": sum(
                (node["cache"] or {}).get("bytes_free", 0) for node in nodes
            ),
            "local_lookup_hits": cache_stats.lookup_hits,
            "local_lookup_misses": cache_stats.lookup_misses,
            "local_tokens_served": cache_stats.tokens_served,
            "cache_evictions": sum(
                (node["cache"] or {}).get("evictions", 0) for node in nodes
            ),
            "cache_bytes_evicted": sum(
                (node["cache"] or {}).get("bytes_evicted", 0) for node in nodes
            ),
            "cache_put_failures": sum(
                (node["cache"] or {}).get("put_failures", 0) for node in nodes
            ),
            "prefill": self.prefill_stats(),
        }

    def prefill_stats(self) -> dict[str, Any]:
        if self.prefill_stats_provider is None:
            return {}
        stats = self.prefill_stats_provider()
        if is_dataclass(stats) and not isinstance(stats, type):
            return asdict(stats)
        return dict(stats)

    def topology(self) -> dict[str, Any]:
        nodes = self.nodes()
        edges = []
        for group in self.groups():
            ids = group["node_ids"]
            if len(ids) > 1:
                edges.extend({
                    "source": ids[0],
                    "target": target,
                    "group_id": group["id"],
                } for target in ids[1:])
        return {"nodes": nodes, "edges": edges}

    def virtual_kv_file(self) -> dict[str, Any]:
        return self.kv_namespace.describe(self.nodes())

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                data.setdefault("registrations", [])
                data.setdefault("groups", [])
                data.setdefault("tokens", {})
                return data
            except (OSError, ValueError):
                pass
        return {"registrations": [], "groups": [], "tokens": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        temp.replace(self.state_path)
