"""Round-trip tests for `mx.array` <-> `torch.Tensor` conversion.

Mac-only. Skipped via `pytest.importorskip` on hosts without `mlx`.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np
import torch

from inference_engine.backends.mlx._torch_bridge import mx_to_torch, torch_to_mx


# ---------------------------------------------------------------------------
# mx_to_torch — type / shape preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mx_dtype,torch_dtype",
    [
        (mx.bfloat16, torch.bfloat16),
        (mx.float16, torch.float16),
        (mx.float32, torch.float32),
        (mx.int32, torch.int32),
    ],
)
def test_mx_to_torch_preserves_dtype(mx_dtype, torch_dtype) -> None:
    a = mx.zeros((2, 3, 4), dtype=mx_dtype)
    t = mx_to_torch(a)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch_dtype
    assert tuple(t.shape) == (2, 3, 4)


def test_mx_to_torch_preserves_values_fp32() -> None:
    a = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    t = mx_to_torch(a)
    expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    assert torch.allclose(t, expected)


def test_mx_to_torch_preserves_values_bf16() -> None:
    a = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.bfloat16)
    t = mx_to_torch(a)
    expected = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16
    )
    # bf16 has 7 bits of mantissa; small integers round-trip exactly.
    assert torch.equal(t, expected)


def test_mx_to_torch_rejects_non_mx_array() -> None:
    with pytest.raises(TypeError, match="expected mx.array"):
        mx_to_torch([1, 2, 3])


# ---------------------------------------------------------------------------
# torch_to_mx
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "torch_dtype,mx_dtype",
    [
        (torch.bfloat16, mx.bfloat16),
        (torch.float32, mx.float32),
        (torch.int32, mx.int32),
    ],
)
def test_torch_to_mx_preserves_dtype(torch_dtype, mx_dtype) -> None:
    t = torch.zeros((2, 3), dtype=torch_dtype)
    a = torch_to_mx(t)
    assert isinstance(a, mx.array)
    assert a.dtype == mx_dtype
    assert tuple(a.shape) == (2, 3)


def test_torch_to_mx_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        torch_to_mx([1, 2, 3])


def test_round_trip_torch_mx_torch_bf16() -> None:
    src = torch.tensor([1.0, -2.0, 3.5, -4.25], dtype=torch.bfloat16)
    a = torch_to_mx(src)
    rt = mx_to_torch(a)
    assert rt.dtype == torch.bfloat16
    assert torch.equal(rt, src)


def test_round_trip_mx_torch_mx_fp32() -> None:
    src = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    t = mx_to_torch(src)
    rt = torch_to_mx(t)
    assert rt.dtype == mx.float32
    # Compare via numpy because mx.array equality is element-wise tensor.
    assert np.array_equal(np.asarray(rt), np.asarray(src))


def test_mx_to_torch_lazy_arrays_get_evaluated() -> None:
    """If we pass an unevaluated mx graph node, mx_to_torch must
    eval it before reading."""
    a = mx.zeros((4, 4), dtype=mx.float32)
    b = a + 1.0  # lazy
    # Don't call mx.eval(b); rely on mx_to_torch to do it.
    t = mx_to_torch(b)
    assert torch.allclose(t, torch.ones((4, 4)))
