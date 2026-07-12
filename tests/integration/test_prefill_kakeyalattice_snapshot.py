from __future__ import annotations

import pytest
import torch

pytest.importorskip("kakeyalattice.hf.bitpack")

from inference_engine.backends.mlx.prefill_snapshot import _pack, _unpack
from inference_engine.distributed.prefill_kakeyalattice import (
    decode_snapshot,
    encode_snapshot,
)
from inference_engine.distributed.tensor_codec import torch_to_wire, wire_to_torch


def test_bitpacked_snapshot_handles_heterogeneous_head_dimensions():
    torch.manual_seed(7)
    original = {
        "layer.0.k": torch.randn(1, 2, 16, 256, dtype=torch.bfloat16),
        "layer.0.v": torch.randn(1, 2, 16, 256, dtype=torch.bfloat16),
        "layer.1.k": torch.randn(1, 1, 16, 512, dtype=torch.bfloat16),
        "layer.1.v": torch.randn(1, 1, 16, 512, dtype=torch.bfloat16),
    }
    raw = _pack(
        [(name, torch_to_wire(value), "torch") for name, value in original.items()],
        metadata={"layer_count": 2, "token_count": 16},
    )
    restored_raw = decode_snapshot(encode_snapshot(raw))
    metadata, restored = _unpack(restored_raw)
    assert metadata["token_count"] == 16
    for name, expected in original.items():
        actual = wire_to_torch(restored[name][0])
        assert actual.shape == expected.shape
        relative_mse = (
            (actual.float() - expected.float()).square().sum()
            / expected.float().square().sum()
        )
        assert float(relative_mse) < 0.002
