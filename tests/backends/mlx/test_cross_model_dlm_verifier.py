"""Linux-CI tests for the MLX cross-model DLM-restored verifier helpers.

Only the non-MLX (model-structure) helpers are exercised here — ``mlx`` is
imported lazily inside the MLX-touching functions, so this module imports and
these helpers run on Linux without Apple Silicon. The MLX forward/injection
path is validated on a Mac by
``scripts/research/k3_integrated_niah_eval_mac.py``.
"""

from __future__ import annotations

import pytest

from inference_engine.backends.mlx import cross_model_dlm_verifier as cmv


class _Attn:
    def __init__(self, head_dim, layer_type, has_kv, layer_idx):
        self.head_dim = head_dim
        self.layer_type = layer_type
        self.has_kv = has_kv
        self.layer_idx = layer_idx


class _Layer:
    def __init__(self, attn):
        self.self_attn = attn


class _TextModel:
    def __init__(self, layers, previous_kvs=None):
        self.layers = layers
        self.embed_tokens = object()
        if previous_kvs is not None:
            self.previous_kvs = previous_kvs


def _gemma4_like(num_kv_shared=0):
    """30 layers: full-attention (head_dim 512) at 5,11,17,23,29, else sliding
    (256). KV sharing for the last `num_kv_shared` layers (same-type source)."""
    n = 30
    full = {5, 11, 17, 23, 29}
    layers = []
    for i in range(n):
        hd = 512 if i in full else 256
        lt = "full_attention" if i in full else "sliding_attention"
        has_kv = i < n - num_kv_shared
        layers.append(_Layer(_Attn(hd, lt, has_kv, i)))
    prev = list(range(n))
    if num_kv_shared > 0:
        m = n - num_kv_shared
        by_type = {}
        for i in range(m):
            by_type[layers[i].self_attn.layer_type] = i
        for j in range(m, n):
            prev[j] = by_type[layers[j].self_attn.layer_type]
    return _TextModel(layers, prev)


class _Wrapper:
    """Mimics mlx_lm wrapper: .model is the text model."""
    def __init__(self, tm):
        self.model = tm


def test_resolve_text_model_via_model_attr():
    tm = _gemma4_like()
    assert cmv.resolve_mlx_text_model(_Wrapper(tm)) is tm


def test_resolve_text_model_direct():
    tm = _gemma4_like()
    # text-only wrapper: object whose .model is the text model
    assert cmv.resolve_mlx_text_model(_Wrapper(tm)) is tm


def test_full_attention_layer_indices_gemma4():
    tm = _gemma4_like()
    assert cmv.mlx_full_attention_layer_indices(tm) == [5, 11, 17, 23, 29]


def test_full_attention_layer_indices_uniform_returns_empty():
    layers = [_Layer(_Attn(256, "sliding_attention", True, i)) for i in range(4)]
    tm = _TextModel(layers, list(range(4)))
    assert cmv.mlx_full_attention_layer_indices(tm) == []


def test_kv_source_map_no_sharing_is_identity():
    tm = _gemma4_like(num_kv_shared=0)
    assert cmv.kv_source_layer_map(tm) == list(range(30))


def test_kv_source_map_with_sharing_points_to_source():
    tm = _gemma4_like(num_kv_shared=10)
    src = cmv.kv_source_layer_map(tm)
    # The last 10 layers (20..29) are sharers; each maps to an earlier
    # same-type source layer (< 20), never to itself.
    for j in range(20, 30):
        assert src[j] < 20
        assert src[j] != j
    # has_kv layers map to themselves.
    for i in range(20):
        assert src[i] == i


def test_resolve_raises_without_text_model():
    class Bad:
        pass
    with pytest.raises(AttributeError):
        cmv.resolve_mlx_text_model(Bad())
