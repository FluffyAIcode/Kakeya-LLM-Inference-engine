"""Bit-packed KakeyaLattice D4 codec for portable MLX prefill snapshots."""
from __future__ import annotations

import json
import struct
from typing import Any

import torch

from inference_engine.backends.mlx.prefill_snapshot import _pack, _unpack
from inference_engine.distributed.tensor_codec import (
    from_proto_fields,
    to_proto_fields,
    torch_to_wire,
    wire_to_torch,
)

_MAGIC = b"KPKL1"
_HEADER_LEN = struct.Struct("<I")
_TORCH_DTYPES = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int32": torch.int32,
    "float16": torch.float16,
}
_WIRE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _imports():
    try:
        from kakeyalattice import V14KakeyaZamirLatticeGPU
        from kakeyalattice.hf.bitpack import (
            pack_lattice_codes,
            unpack_lattice_codes,
        )
        from kakeyalattice.hf.quantized_cache import (
            decode_from_indices,
            encode_to_indices,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "KakeyaLattice bit-packed prefill requires kakeyalattice>=1.6.1",
        ) from exc
    return (
        V14KakeyaZamirLatticeGPU,
        pack_lattice_codes,
        unpack_lattice_codes,
        encode_to_indices,
        decode_from_indices,
    )


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode_snapshot(payload: bytes, *, q_range: int = 38) -> bytes:
    """Quantize every K/V tensor and serialize losslessly packed lattice codes."""
    (
        codec_cls,
        pack_lattice_codes,
        _unpack_lattice_codes,
        encode_to_indices,
        _decode_from_indices,
    ) = _imports()
    snapshot_metadata, tensors = _unpack(payload)
    device = _device()
    codecs: dict[int, Any] = {}
    parts: list[bytes] = []
    records: list[dict[str, Any]] = []

    def add_part(tensor: torch.Tensor) -> dict[str, Any]:
        value = tensor.detach().contiguous().cpu()
        dtype = str(value.dtype).removeprefix("torch.")
        raw = value.numpy().tobytes()
        record = {
            "offset": sum(len(part) for part in parts),
            "length": len(raw),
            "shape": list(value.shape),
            "dtype": dtype,
        }
        parts.append(raw)
        return record

    for name, (wire, framework) in tensors.items():
        if not name.startswith("layer."):
            dtype, shape, data = to_proto_fields(wire)
            records.append({
                "name": name,
                "kind": "raw",
                "framework": framework,
                "wire_dtype": dtype,
                "wire_shape": shape,
                "data": add_part(torch.frombuffer(bytearray(data), dtype=torch.uint8)),
            })
            continue
        original = wire_to_torch(wire).to(device)
        head_dim = int(original.shape[-1])
        codec = codecs.get(head_dim)
        if codec is None:
            codec = codec_cls(D=head_dim, q_range=q_range, device=str(device))
            codecs[head_dim] = codec
        codes, norms, qmax = encode_to_indices(codec, original)
        packed = pack_lattice_codes(codes, "d4", q_range)
        records.append({
            "name": name,
            "kind": "kakeyalattice-d4",
            "framework": framework,
            "wire_dtype": wire.dtype,
            "head_dim": head_dim,
            "q_range": q_range,
            "packed": {
                "width": packed["width"],
                "mode": packed["mode"],
                "n_blocks": packed["n_blocks"],
                "shape": list(packed["shape"]),
                "bd": packed["bd"],
                "variant": packed["variant"],
                "buf": add_part(packed["buf"]),
                "exc_idx": add_part(packed["exc_idx"]),
                "exc_vals": add_part(packed["exc_vals"]),
            },
            "norms": add_part(norms),
            "qmax": add_part(qmax),
        })
    header = json.dumps(
        {"snapshot": snapshot_metadata, "tensors": records},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _MAGIC + _HEADER_LEN.pack(len(header)) + header + b"".join(parts)


def decode_snapshot(payload: bytes) -> bytes:
    """Restore a standard KPKV1 snapshot from bit-packed lattice tensors."""
    (
        codec_cls,
        _pack_lattice_codes,
        unpack_lattice_codes,
        _encode_to_indices,
        decode_from_indices,
    ) = _imports()
    if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + 4:
        raise ValueError("invalid KakeyaLattice prefill snapshot magic")
    header_len = _HEADER_LEN.unpack(payload[len(_MAGIC):len(_MAGIC) + 4])[0]
    start = len(_MAGIC) + 4
    end = start + header_len
    if end > len(payload):
        raise ValueError("truncated KakeyaLattice prefill header")
    metadata = json.loads(payload[start:end])
    raw = memoryview(payload)[end:]
    device = _device()
    codecs: dict[int, Any] = {}

    def read_part(record: dict[str, Any], *, device_: torch.device | None = None):
        offset = int(record["offset"])
        part_end = offset + int(record["length"])
        if offset < 0 or part_end > len(raw):
            raise ValueError("truncated KakeyaLattice tensor component")
        dtype = _TORCH_DTYPES[record["dtype"]]
        if part_end == offset:
            tensor = torch.empty(record["shape"], dtype=dtype)
        else:
            tensor = torch.frombuffer(
                bytearray(raw[offset:part_end]),
                dtype=dtype,
            ).reshape(record["shape"])
        return tensor.to(device_) if device_ is not None else tensor

    tensors = []
    for record in metadata["tensors"]:
        if record["kind"] == "raw":
            data = read_part(record["data"]).numpy().tobytes()
            wire = from_proto_fields(
                record["wire_dtype"],
                record["wire_shape"],
                data,
            )
        else:
            head_dim = int(record["head_dim"])
            q_range = int(record["q_range"])
            codec = codecs.get(head_dim)
            if codec is None:
                codec = codec_cls(D=head_dim, q_range=q_range, device=str(device))
                codecs[head_dim] = codec
            packed_record = record["packed"]
            packed = {
                "width": int(packed_record["width"]),
                "mode": packed_record["mode"],
                "n_blocks": int(packed_record["n_blocks"]),
                "shape": tuple(int(v) for v in packed_record["shape"]),
                "bd": int(packed_record["bd"]),
                "q_range": q_range,
                "variant": packed_record["variant"],
                "buf": read_part(packed_record["buf"], device_=device),
                "exc_idx": read_part(packed_record["exc_idx"], device_=device),
                "exc_vals": read_part(packed_record["exc_vals"], device_=device),
            }
            codes = unpack_lattice_codes(packed)
            restored = decode_from_indices(
                codec,
                codes,
                read_part(record["norms"], device_=device),
                read_part(record["qmax"], device_=device),
                out_dtype=_WIRE_DTYPES[record["wire_dtype"]],
            )
            wire = torch_to_wire(restored.cpu())
        tensors.append((record["name"], wire, record["framework"]))
    return _pack(tensors, metadata=metadata["snapshot"])
