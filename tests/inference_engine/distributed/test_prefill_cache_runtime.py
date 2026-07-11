from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from inference_engine.distributed.capability import CacheCompatibility  # noqa: E402
from inference_engine.distributed.prefill_cache import PrefixCacheStore  # noqa: E402
from inference_engine.distributed.prefill_cache_runtime import (  # noqa: E402
    DistributedPrefillCacheHook,
)


class Layer:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, value):
        self.keys, self.values = value


class Verifier:
    def __init__(self):
        self.cache = None
        self.cached_token_sequence = []
        self.next_global_position = 0
        self.next_token_logits = torch.zeros(2)
        self.prefill_calls = 0
        self.forwarded = 0

    def reset(self):
        self.cache = [Layer()]
        self.cached_token_sequence = []
        self.next_global_position = 0

    def prefill(self, tokens):
        self.reset()
        self.prefill_calls += 1
        self._append(tokens)

    def forward_block(self, tokens):
        self.forwarded += len(tokens)
        self._append(tokens)
        return torch.stack([torch.tensor([float(t), 0.0]) for t in tokens])

    def commit_or_truncate(self, *, forwarded, accepted):
        assert forwarded == accepted

    def _append(self, tokens):
        values = mx.array(tokens, dtype=mx.float32).reshape(1, 1, -1, 1)
        layer = self.cache[0]
        layer.keys = values if layer.keys is None else mx.concatenate([layer.keys, values], axis=2)
        layer.values = layer.keys + 1
        layer.offset += len(tokens)
        self.cached_token_sequence.extend(tokens)
        self.next_global_position += len(tokens)
        self.next_token_logits = torch.tensor([float(tokens[-1]), 1.0])


def test_local_snapshot_hit_skips_prefill_and_computes_suffix():
    compatibility = CacheCompatibility(model_id="m", block_size_tokens=2)
    store = PrefixCacheStore(compatibility, max_bytes=1 << 20, node_id="head")
    reused_events = []
    hook = DistributedPrefillCacheHook(store, on_reuse=reused_events.append)

    first = Verifier()
    assert hook.prepare(first, [1, 2, 3, 4]) == 0
    assert first.prefill_calls == 1
    assert store.stats().entry_count == 2

    exact = Verifier()
    assert hook.prepare(exact, [1, 2, 3, 4]) == 4
    assert exact.prefill_calls == 0
    assert exact.forwarded == 0
    assert exact.cached_token_sequence == [1, 2, 3, 4]

    suffix = Verifier()
    assert hook.prepare(suffix, [1, 2, 3, 4, 5, 6]) == 4
    assert suffix.prefill_calls == 0
    assert suffix.forwarded == 2
    assert suffix.cached_token_sequence == [1, 2, 3, 4, 5, 6]
    assert hook.stats.local_hits == 2
    assert hook.stats.tokens_reused == 8
    assert reused_events == [4, 4]
