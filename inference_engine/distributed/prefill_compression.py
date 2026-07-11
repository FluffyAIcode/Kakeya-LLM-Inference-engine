"""Snapshot payload framing and compression for distributed prefill K/V."""
from __future__ import annotations

import hashlib
import struct
import zlib

from inference_engine.distributed.capability import CompressionCodec

_MAGIC = b"KPC1"
_HEADER = struct.Struct("<4sBQ32s")


def compress_payload(
    payload: bytes,
    codec: CompressionCodec,
    *,
    level: int = 3,
) -> bytes:
    raw = bytes(payload)
    if codec in (CompressionCodec.UNSPECIFIED, CompressionCodec.NONE):
        return raw
    if codec != CompressionCodec.ZLIB:
        raise ValueError(f"unsupported compression codec {codec!r}")
    if not (0 <= level <= 9):
        raise ValueError("zlib level must be in [0, 9]")
    compressed = zlib.compress(raw, level)
    return _HEADER.pack(
        _MAGIC,
        int(codec),
        len(raw),
        hashlib.sha256(raw).digest(),
    ) + compressed


def decompress_payload(
    payload: bytes,
    *,
    max_uncompressed_bytes: int,
) -> bytes:
    data = bytes(payload)
    if not data.startswith(_MAGIC):
        if len(data) > max_uncompressed_bytes:
            raise ValueError("uncompressed prefill payload exceeds import budget")
        return data
    if len(data) < _HEADER.size:
        raise ValueError("truncated compressed prefill payload header")
    _magic, raw_codec, expected_size, expected_sha = _HEADER.unpack(
        data[:_HEADER.size],
    )
    if expected_size > max_uncompressed_bytes:
        raise ValueError("prefill payload exceeds uncompressed import budget")
    try:
        codec = CompressionCodec(raw_codec)
    except ValueError as exc:
        raise ValueError(f"unsupported compression codec {raw_codec}") from exc
    if codec != CompressionCodec.ZLIB:
        raise ValueError(f"unsupported framed compression codec {codec.name}")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(
        data[_HEADER.size:],
        max_uncompressed_bytes + 1,
    )
    if decompressor.unconsumed_tail or len(raw) > max_uncompressed_bytes:
        raise ValueError("decompressed payload exceeds import budget")
    remaining = max_uncompressed_bytes - len(raw)
    raw += decompressor.flush(remaining + 1)
    if len(raw) > max_uncompressed_bytes:  # pragma: no cover - zlib flush guard
        raise ValueError("decompressed payload exceeds import budget")
    if decompressor.unused_data:
        raise ValueError("compressed prefill payload has trailing data")
    if len(raw) != expected_size:
        raise ValueError(
            f"decompressed payload size {len(raw)} != expected {expected_size}",
        )
    if hashlib.sha256(raw).digest() != expected_sha:
        raise ValueError("decompressed prefill payload checksum mismatch")
    return raw


def payload_sizes(payload: bytes) -> tuple[int, int]:
    """Return ``(wire_bytes, uncompressed_bytes)`` without decompressing."""
    data = bytes(payload)
    if data.startswith(_MAGIC) and len(data) >= _HEADER.size:
        return len(data), int(_HEADER.unpack(data[:_HEADER.size])[2])
    return len(data), len(data)

