from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from inference_engine.backends.mlx.prefill_snapshot import (  # noqa: E402
    export_mlx_prefill_snapshot,
    import_mlx_prefill_snapshot,
)
from inference_engine.distributed.capability import CacheCompatibility  # noqa: E402


class Layer:
    def __init__(self, value: float = 0.0):
        self.keys = mx.full((1, 2, 3, 4), value)
        self.values = mx.full((1, 2, 3, 4), value + 1)
        self.offset = 3

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, value):
        self.keys, self.values = value


def test_snapshot_round_trip_and_compatibility():
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=3)
    source = [Layer(1), Layer(2)]
    payload = export_mlx_prefill_snapshot(
        source,
        token_count=3,
        cached_token_ids=[1, 2, 3],
        compatibility=compatibility,
        next_token_logits=torch.tensor([1.0, 2.0]),
    )
    target = [Layer(9), Layer(9)]
    imported = import_mlx_prefill_snapshot(
        payload,
        target,
        compatibility=compatibility,
    )
    assert imported.token_count == 3
    assert imported.cached_token_ids == (1, 2, 3)
    assert torch.equal(imported.next_token_logits, torch.tensor([1.0, 2.0]))
    assert bool(mx.all(target[0].keys == source[0].keys))
    assert target[0].offset == 3
    with pytest.raises(ValueError, match="compatibility"):
        import_mlx_prefill_snapshot(
            payload,
            target,
            compatibility=CacheCompatibility(model_id="other"),
        )


def test_snapshot_rejects_empty_layer_and_bad_payload():
    compatibility = CacheCompatibility(model_id="m")
    layer = Layer()
    layer.keys = None
    with pytest.raises(ValueError, match="empty"):
        export_mlx_prefill_snapshot(
            [layer],
            token_count=1,
            cached_token_ids=[1],
            compatibility=compatibility,
        )
    with pytest.raises(ValueError, match="magic"):
        import_mlx_prefill_snapshot(
            b"bad",
            [Layer()],
            compatibility=compatibility,
        )
