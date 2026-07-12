from __future__ import annotations

import pytest

from inference_engine.distributed.capability import CompressionCodec
from inference_engine.distributed.prefill_compression import (
    compress_payload,
    decompress_payload,
    payload_sizes,
)


def test_zlib_round_trip_and_sizes():
    raw = (b"prefill-kv-" * 1000)
    compressed = compress_payload(raw, CompressionCodec.ZLIB)
    assert len(compressed) < len(raw)
    assert payload_sizes(compressed) == (len(compressed), len(raw))
    assert decompress_payload(
        compressed,
        max_uncompressed_bytes=len(raw),
    ) == raw


def test_none_is_backward_compatible_raw_payload():
    raw = b"legacy-kpkv"
    assert compress_payload(raw, CompressionCodec.NONE) == raw
    assert decompress_payload(raw, max_uncompressed_bytes=100) == raw
    assert payload_sizes(raw) == (len(raw), len(raw))


def test_compression_and_import_limits_validate():
    with pytest.raises(ValueError, match="zlib level"):
        compress_payload(b"x", CompressionCodec.ZLIB, level=10)
    with pytest.raises(ValueError, match="unsupported compression"):
        compress_payload(b"x", 99)  # type: ignore[arg-type]
    framed = compress_payload(b"x" * 100, CompressionCodec.ZLIB)
    with pytest.raises(ValueError, match="import budget"):
        decompress_payload(framed, max_uncompressed_bytes=10)
    with pytest.raises(ValueError, match="import budget"):
        decompress_payload(b"x" * 11, max_uncompressed_bytes=10)


def test_truncated_and_corrupt_payloads_fail():
    import hashlib
    import struct

    framed = compress_payload(b"x" * 100, CompressionCodec.ZLIB)
    with pytest.raises(ValueError):
        decompress_payload(framed[:10], max_uncompressed_bytes=1000)
    damaged = framed[:-1] + bytes([framed[-1] ^ 1])
    with pytest.raises((ValueError, __import__("zlib").error)):
        decompress_payload(damaged, max_uncompressed_bytes=1000)
    magic = b"KPC1"
    with pytest.raises(ValueError, match="unsupported compression codec"):
        decompress_payload(
            struct.pack("<4sBQ32s", magic, 99, 1, hashlib.sha256(b"x").digest())
            + b"x",
            max_uncompressed_bytes=100,
        )
    with pytest.raises(ValueError, match="unsupported framed"):
        decompress_payload(
            struct.pack(
                "<4sBQ32s",
                magic,
                int(CompressionCodec.NONE),
                1,
                hashlib.sha256(b"x").digest(),
            ) + b"x",
            max_uncompressed_bytes=100,
        )
    with pytest.raises(ValueError, match="decompressed payload size"):
        decompress_payload(
            framed[:5] + struct.pack("<Q", 101) + framed[13:],
            max_uncompressed_bytes=1000,
        )
    wrong_sha = bytearray(framed)
    wrong_sha[13] ^= 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        decompress_payload(bytes(wrong_sha), max_uncompressed_bytes=1000)
    # Header lies that the payload is smaller than the decompressed stream.
    bomb = (
        struct.pack(
            "<4sBQ32s",
            magic,
            int(CompressionCodec.ZLIB),
            10,
            hashlib.sha256(b"x" * 100).digest(),
        )
        + __import__("zlib").compress(b"x" * 100)
    )
    with pytest.raises(ValueError, match="import budget"):
        decompress_payload(bomb, max_uncompressed_bytes=50)
    with pytest.raises(ValueError, match="trailing data"):
        decompress_payload(framed + b"junk", max_uncompressed_bytes=1000)

