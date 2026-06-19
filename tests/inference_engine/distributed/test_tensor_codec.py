"""Unit tests for the F3 bulk-tensor codec (inference_engine/distributed/tensor_codec)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from inference_engine.distributed import tensor_codec as tc


def _roundtrip(arr: np.ndarray, dtype: str | None = None) -> tc.WireTensor:
    wire = tc.encode_array(arr, dtype=dtype)
    name, shape, data = tc.to_proto_fields(wire)
    return tc.from_proto_fields(name, shape, data)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int32", "int64", "uint32", "bool"])
def test_roundtrip_preserves_values_and_shape(dtype):
    rng = np.random.default_rng(0)
    if dtype == "bool":
        arr = rng.integers(0, 2, size=(2, 3, 4)).astype(bool)
    elif dtype.startswith("float"):
        arr = rng.standard_normal((2, 3, 4)).astype(dtype)
    else:
        arr = rng.integers(0, 100, size=(2, 3, 4)).astype(dtype)
    out = _roundtrip(arr)
    assert out.dtype == dtype
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out.data, arr)


def test_encode_rejects_non_ndarray():
    with pytest.raises(TypeError):
        tc.encode_array([1, 2, 3])  # type: ignore[arg-type]


def test_encode_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        tc.encode_array(np.zeros(2, dtype=np.float64))


def test_encode_with_explicit_bfloat16_tag_uses_uint16_buffer():
    bits = np.array([0x3F80, 0x4000], dtype=np.uint16)  # bf16 1.0, 2.0
    wire = tc.encode_array(bits, dtype="bfloat16")
    assert wire.is_bfloat16
    name, shape, data = tc.to_proto_fields(wire)
    assert name == "bfloat16"
    out = tc.from_proto_fields(name, shape, data)
    assert out.is_bfloat16
    np.testing.assert_array_equal(out.data, bits)


def test_from_proto_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        tc.from_proto_fields("float64", [2], b"\x00" * 16)


def test_from_proto_rejects_byte_count_mismatch():
    with pytest.raises(ValueError, match="byte count"):
        tc.from_proto_fields("float32", [4], b"\x00" * 8)  # 4*4=16 expected


def test_from_proto_rejects_negative_dim():
    with pytest.raises(ValueError, match="negative dim"):
        tc.from_proto_fields("int32", [-1], b"")


def test_nbytes_matches_payload():
    wire = tc.encode_array(np.zeros((3, 5), dtype=np.float32))
    assert tc.nbytes(wire) == 3 * 5 * 4


def test_to_proto_fields_returns_contiguous_bytes_for_noncontiguous_input():
    arr = np.asfortranarray(np.arange(6, dtype=np.int32).reshape(2, 3))
    out = _roundtrip(arr)
    np.testing.assert_array_equal(out.data, arr)


def test_torch_bridge_roundtrip_float32():
    t = torch.randn(2, 3, dtype=torch.float32)
    wire = tc.torch_to_wire(t)
    name, shape, data = tc.to_proto_fields(wire)
    back = tc.wire_to_torch(tc.from_proto_fields(name, shape, data))
    assert back.dtype == torch.float32
    torch.testing.assert_close(back, t)


def test_torch_bridge_roundtrip_bfloat16():
    t = (torch.arange(6, dtype=torch.float32).reshape(2, 3)).to(torch.bfloat16)
    wire = tc.torch_to_wire(t)
    assert wire.is_bfloat16
    name, shape, data = tc.to_proto_fields(wire)
    back = tc.wire_to_torch(tc.from_proto_fields(name, shape, data))
    assert back.dtype == torch.bfloat16
    torch.testing.assert_close(back, t)


def test_torch_bridge_handles_noncontiguous():
    t = torch.randn(4, 5)[:, ::2]  # non-contiguous view
    back = tc.wire_to_torch(tc.torch_to_wire(t))
    torch.testing.assert_close(back, t.contiguous())
