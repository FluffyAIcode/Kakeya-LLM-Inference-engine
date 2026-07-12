"""Portable snapshot adapter for MLX prefill caches.

Snapshots are immutable checkpoints at token-block boundaries. They contain
the current bounded per-layer K/V tensors, logical position, cached token
sequence, and optional next-token logits. The wire container is JSON metadata
plus raw tensor buffers; no pickle is used.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from inference_engine.distributed.capability import CacheCompatibility
from inference_engine.distributed.prefill_cache import compatibility_fingerprint
from inference_engine.distributed.tensor_codec import (
    WireTensor,
    from_proto_fields,
    mlx_to_wire,
    to_proto_fields,
    torch_to_wire,
    wire_to_mlx,
    wire_to_torch,
)

_MAGIC = b"KPKV1"


@dataclass(frozen=True)
class ImportedPrefillSnapshot:
    token_count: int
    cached_token_ids: tuple[int, ...]
    next_token_logits: Any | None
    block_hash: bytes = b""


def export_mlx_prefill_snapshot(
    cache: Sequence[Any],
    *,
    token_count: int,
    cached_token_ids: Sequence[int],
    compatibility: CacheCompatibility,
    next_token_logits: Any | None = None,
    block_hash: bytes = b"",
) -> bytes:
    """Serialize current MLX cache state at one prefix boundary."""
    if token_count <= 0:
        raise ValueError("token_count must be > 0")
    tensors: list[tuple[str, WireTensor, str]] = []
    for index, layer in enumerate(cache):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise ValueError(f"cache layer {index} is empty")
        tensors.append((f"layer.{index}.k", mlx_to_wire(keys), "mlx"))
        tensors.append((f"layer.{index}.v", mlx_to_wire(values), "mlx"))
    if next_token_logits is not None:
        if hasattr(next_token_logits, "detach"):
            tensors.append(("next_token_logits", torch_to_wire(next_token_logits), "torch"))
        else:
            tensors.append(("next_token_logits", mlx_to_wire(next_token_logits), "mlx"))
    return _pack(
        tensors,
        metadata={
            "compatibility_sha256": compatibility_fingerprint(compatibility).hex(),
            "token_count": int(token_count),
            "cached_token_ids": [int(token) for token in cached_token_ids],
            "layer_count": len(cache),
            "block_hash": bytes(block_hash).hex(),
        },
    )


def import_mlx_prefill_snapshot(
    payload: bytes,
    cache: Sequence[Any],
    *,
    compatibility: CacheCompatibility,
) -> ImportedPrefillSnapshot:
    """Restore a snapshot into an allocated MLX cache list."""
    metadata, tensors = _unpack(payload)
    expected = compatibility_fingerprint(compatibility).hex()
    if metadata.get("compatibility_sha256") != expected:
        raise ValueError("prefill snapshot compatibility fingerprint mismatch")
    layer_count = int(metadata.get("layer_count", -1))
    if layer_count != len(cache):
        raise ValueError(
            f"snapshot layer_count {layer_count} != allocated cache {len(cache)}",
        )
    token_count = int(metadata["token_count"])
    for index, layer in enumerate(cache):
        key_wire, _ = tensors[f"layer.{index}.k"]
        value_wire, _ = tensors[f"layer.{index}.v"]
        keys = wire_to_mlx(key_wire)
        values = wire_to_mlx(value_wire)
        if hasattr(layer, "state"):
            layer.state = (keys, values)
        else:
            layer.keys, layer.values = keys, values
        if hasattr(layer, "offset"):
            layer.offset = token_count
    next_logits = None
    if "next_token_logits" in tensors:
        wire, framework = tensors["next_token_logits"]
        next_logits = wire_to_torch(wire) if framework == "torch" else wire_to_mlx(wire)
    return ImportedPrefillSnapshot(
        token_count=token_count,
        cached_token_ids=tuple(int(t) for t in metadata["cached_token_ids"]),
        next_token_logits=next_logits,
        block_hash=bytes.fromhex(metadata.get("block_hash", "")),
    )


def _pack(
    tensors: Sequence[tuple[str, WireTensor, str]],
    *,
    metadata: dict,
) -> bytes:
    raw_parts: list[bytes] = []
    tensor_meta: list[dict] = []
    offset = 0
    for name, wire, framework in tensors:
        dtype, shape, data = to_proto_fields(wire)
        tensor_meta.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "offset": offset,
            "length": len(data),
            "framework": framework,
        })
        raw_parts.append(data)
        offset += len(data)
    header = json.dumps(
        {**metadata, "tensors": tensor_meta},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _MAGIC + struct.pack("<I", len(header)) + header + b"".join(raw_parts)


def _unpack(payload: bytes) -> tuple[dict, dict[str, tuple[WireTensor, str]]]:
    if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + 4:
        raise ValueError("invalid prefill snapshot magic")
    header_len = struct.unpack("<I", payload[len(_MAGIC):len(_MAGIC) + 4])[0]
    header_start = len(_MAGIC) + 4
    header_end = header_start + header_len
    if header_end > len(payload):
        raise ValueError("truncated prefill snapshot header")
    metadata = json.loads(payload[header_start:header_end])
    raw = memoryview(payload)[header_end:]
    tensors: dict[str, tuple[WireTensor, str]] = {}
    for item in metadata.pop("tensors"):
        start = int(item["offset"])
        end = start + int(item["length"])
        if start < 0 or end > len(raw):
            raise ValueError("truncated prefill snapshot tensor")
        tensors[item["name"]] = (
            from_proto_fields(
                item["dtype"],
                item["shape"],
                bytes(raw[start:end]),
            ),
            item["framework"],
        )
    return metadata, tensors
