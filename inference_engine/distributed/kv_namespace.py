"""Logical content-addressed namespace over physically separate KV stores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from inference_engine.distributed.capability import CacheCompatibility
from inference_engine.distributed.prefill_cache import compatibility_fingerprint


@dataclass(frozen=True)
class VirtualKVMount:
    node_id: str
    address: str
    bytes_used: int
    bytes_free: int
    entry_count: int
    network: str
    tier: str


class VirtualKVNamespace:
    """Present cache-node RAM as one lookup namespace, never as coherent RAM."""

    def __init__(
        self,
        compatibility: CacheCompatibility,
        *,
        primary_node_id: str = "head-runtime",
    ) -> None:
        self.compatibility = compatibility
        self.primary_node_id = primary_node_id
        fingerprint = compatibility_fingerprint(compatibility).hex()
        tenant = compatibility.tenant_namespace or "default"
        self.uri = f"kv://{tenant}/{compatibility.model_id}/{fingerprint}"

    def describe(self, nodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
        mounts = []
        for node in nodes:
            cache = node.get("cache")
            if not cache or cache.get("model_id") != self.compatibility.model_id:
                continue
            endpoint = node.get("endpoint") or {}
            mounts.append(VirtualKVMount(
                node_id=node["id"],
                address=endpoint.get("address", ""),
                bytes_used=int(cache.get("bytes_used", 0)),
                bytes_free=int(cache.get("bytes_free", 0)),
                entry_count=int(cache.get("entry_count", 0)),
                network=endpoint.get("network", "default"),
                tier=(
                    "hot"
                    if node["id"] == self.primary_node_id
                    else "cold-offload"
                ),
            ))
        return {
            "uri": self.uri,
            "access": "content-addressed-lookup-fetch-import",
            "coherent_shared_memory": False,
            "mounts": [mount.__dict__ for mount in mounts],
            "bytes_used": sum(mount.bytes_used for mount in mounts),
            "bytes_free": sum(mount.bytes_free for mount in mounts),
            "entry_count": sum(mount.entry_count for mount in mounts),
        }
