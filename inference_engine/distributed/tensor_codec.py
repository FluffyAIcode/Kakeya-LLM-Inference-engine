"""Framework-agnostic tensor <-> proto codec for the F3 bulk-tensor data plane
(ADR 0009 §4: shipping aux-hidden / restored-K/V between a verifier host and a
remote DFlash+f_θ proposer host).

The wire form is a tiny self-describing blob: a dtype string, an int64 shape,
and the raw little-endian buffer (``numpy.ndarray.tobytes``). The endpoints
convert to/from torch or mlx with the thin helpers below, so the codec itself
has **no** torch/mlx dependency and is unit-testable anywhere numpy is present.

Why raw numpy bytes rather than ``torch.save`` (what the old co-located
``k3_specdecode_gpu_bench`` used): it is framework-neutral (an MLX verifier on
the Mac and a torch DFlash proposer on the GPU must interoperate), it has no
pickle/security surface, and the byte count is exactly the tensor payload so
RTT/bandwidth accounting is honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

# Dtypes we allow on the wire. bfloat16 has no numpy scalar type, so it is
# carried as raw uint16 pairs under the logical name "bfloat16" and rebuilt by
# the framework bridge (torch.bfloat16 / mlx.bfloat16) at the endpoint.
_ALLOWED_DTYPES = frozenset(
    {"float32", "float16", "bfloat16", "int32", "int64", "uint32", "bool"}
)


@dataclass(frozen=True)
class WireTensor:
    """A decoded tensor still in framework-neutral form.

    ``data`` is a numpy array EXCEPT for ``bfloat16``, where numpy has no native
    scalar: ``data`` is then a ``uint16`` array carrying the raw bf16 bit
    patterns and ``dtype`` is the logical string ``"bfloat16"`` so the bridge
    can reinterpret it.
    """

    dtype: str
    shape: Tuple[int, ...]
    data: np.ndarray

    @property
    def is_bfloat16(self) -> bool:
        return self.dtype == "bfloat16"


def encode_array(array: np.ndarray, *, dtype: str | None = None) -> WireTensor:
    """Encode a numpy array into a :class:`WireTensor`.

    ``dtype`` overrides the logical dtype name (used to tag a uint16 buffer as
    logical ``bfloat16``); otherwise it is inferred from ``array.dtype``.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"encode_array expects np.ndarray, got {type(array).__name__}")
    logical = dtype or str(array.dtype)
    if logical not in _ALLOWED_DTYPES:
        raise ValueError(f"unsupported wire dtype {logical!r}")
    contiguous = np.ascontiguousarray(array)
    return WireTensor(dtype=logical, shape=tuple(int(d) for d in array.shape),
                      data=contiguous)


def to_proto_fields(wire: WireTensor) -> Tuple[str, List[int], bytes]:
    """Flatten a :class:`WireTensor` to the (dtype, shape, data-bytes) triple
    that fills a proto ``Tensor`` message."""
    return wire.dtype, [int(d) for d in wire.shape], wire.data.tobytes()


def from_proto_fields(dtype: str, shape: Sequence[int], data: bytes) -> WireTensor:
    """Rebuild a :class:`WireTensor` from proto ``Tensor`` fields, validating the
    byte count matches ``shape × itemsize`` so a truncated/garbled blob fails
    loudly instead of silently mis-shaping."""
    if dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"unsupported wire dtype {dtype!r}")
    np_dtype = np.uint16 if dtype == "bfloat16" else np.dtype(dtype)
    count = 1
    for d in shape:
        if d < 0:
            raise ValueError(f"negative dim in shape {tuple(shape)}")
        count *= int(d)
    expected = count * np.dtype(np_dtype).itemsize
    if len(data) != expected:
        raise ValueError(
            f"tensor byte count {len(data)} != shape {tuple(shape)} × "
            f"{np.dtype(np_dtype).itemsize}B = {expected} (dtype {dtype})")
    flat = np.frombuffer(data, dtype=np_dtype, count=count).reshape(tuple(shape))
    return WireTensor(dtype=dtype, shape=tuple(int(d) for d in shape), data=flat)


def nbytes(wire: WireTensor) -> int:
    """Payload size of the tensor buffer in bytes (for RTT/bandwidth accounting)."""
    return int(wire.data.nbytes)


# --------------------------------------------------------------------------- #
# Framework bridges. Imported lazily so the codec works without torch/mlx.
# --------------------------------------------------------------------------- #
def torch_to_wire(tensor: Any) -> WireTensor:
    """torch.Tensor -> WireTensor (bf16 -> logical bfloat16 over uint16 bits)."""
    import torch

    t = tensor.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        bits = t.view(torch.uint16).numpy()
        return encode_array(bits, dtype="bfloat16")
    return encode_array(t.numpy())


def wire_to_torch(wire: WireTensor) -> Any:
    """WireTensor -> torch.Tensor (rebuilds bfloat16 from the uint16 bit buffer)."""
    import torch

    if wire.is_bfloat16:
        return torch.from_numpy(wire.data.copy()).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(wire.data).copy())


def mlx_to_wire(array: Any) -> WireTensor:  # pragma: no cover - requires mlx runtime
    """mlx.array -> WireTensor. mlx bfloat16 is bridged through uint16 bits."""
    import mlx.core as mx

    if array.dtype == mx.bfloat16:
        bits = np.array(array.view(mx.uint16), copy=True)
        return encode_array(bits.astype(np.uint16), dtype="bfloat16")
    return encode_array(np.array(array, copy=True))


def wire_to_mlx(wire: WireTensor) -> Any:  # pragma: no cover - requires mlx runtime
    """WireTensor -> mlx.array (rebuilds bfloat16 from the uint16 bit buffer)."""
    import mlx.core as mx

    if wire.is_bfloat16:
        return mx.array(wire.data).view(mx.bfloat16)
    return mx.array(np.ascontiguousarray(wire.data))
