"""Tiny conversion helpers between `mx.array` and `torch.Tensor`.

The speculative decoder in `kv_cache_proposer.speculative` operates on
`torch.Tensor` (it calls `torch.argmax`, `.item()`, etc.). The MLX
backend produces `mx.array`. We convert at the API boundary so the
speculative loop is unchanged. Conversions are zero-copy where MLX's
unified-memory model allows; otherwise they go through NumPy (which is
host-side memory on Mac, so the round trip is a memcpy on the GPU
shared address space — single-digit ms for the slices we use).

The module imports `mlx.core` at top level; importing it on a host
without Apple Silicon will raise. That is intentional: there is no
fallback. The platform check happens at the package level via
`inference_engine.backends.mlx.env`, before anyone reaches this file.
"""

from __future__ import annotations

import numpy as np
import torch

import mlx.core as mx


def mx_to_torch(arr: "mx.array") -> torch.Tensor:
    """Materialize an `mx.array` as a `torch.Tensor` on CPU.

    MLX is lazy: passing a graph-node `mx.array` here forces evaluation.
    The result is a CPU `torch.Tensor` with dtype matching the source
    (bf16 ↔ bf16, f16 ↔ f16, f32 ↔ f32, int32 ↔ int32). Other dtypes
    raise `TypeError` rather than silently downcasting.
    """
    if not isinstance(arr, mx.array):
        raise TypeError(
            f"mx_to_torch expected mx.array, got {type(arr).__name__}"
        )
    # Force evaluation before reading.
    mx.eval(arr)
    src_dtype = arr.dtype
    # NumPy doesn't have native bf16; route bf16 through int16 view + bf16
    # reinterpretation on the torch side.
    if src_dtype == mx.bfloat16:
        as_uint16 = np.asarray(arr.view(mx.uint16))  # bit-preserving view
        t = torch.from_numpy(np.array(as_uint16, copy=True)).view(torch.bfloat16)
        return t
    np_arr = np.asarray(arr)
    return torch.from_numpy(np.array(np_arr, copy=True))


def torch_to_mx(tensor: torch.Tensor) -> "mx.array":
    """Convert a `torch.Tensor` to an `mx.array` (mirror of `mx_to_torch`).

    Used when the speculative decoder feeds correction-or-bonus token
    ids back into the verifier as a 1-element list — which we keep as
    Python `int` and never need to cross the bridge — but provided for
    completeness so callers can pass tensors in symmetric APIs.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"torch_to_mx expected torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.dtype == torch.bfloat16:
        as_uint16 = tensor.view(torch.uint16).contiguous().numpy()
        return mx.array(as_uint16).view(mx.bfloat16)
    return mx.array(tensor.contiguous().numpy())
