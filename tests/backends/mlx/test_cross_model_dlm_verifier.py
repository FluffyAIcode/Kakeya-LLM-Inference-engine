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


def _gemma4_geom_model():
    """Mock with n_kv_heads/head_dim per layer (8/256 sliding, 2/512 full)."""
    full = {5, 11, 17, 23, 29}
    layers = []
    for i in range(30):
        if i in full:
            layers.append(_Layer(_AttnGeom(2, 512, "full_attention", i)))
        else:
            layers.append(_Layer(_AttnGeom(8, 256, "sliding_attention", i)))
    return _TextModel(layers, list(range(30)))


class _AttnGeom(_Attn):
    def __init__(self, n_kv, head_dim, layer_type, layer_idx):
        super().__init__(head_dim, layer_type, True, layer_idx)
        self.n_kv_heads = n_kv


def test_kv_memory_report_s5_vs_naive():
    tm = _gemma4_geom_model()
    full = [5, 11, 17, 23, 29]
    s5 = cmv.kv_memory_report(
        tm, sink_size=4, window_size=64, seq_len=5500, exact_layer_indices=full)
    naive = cmv.kv_memory_report(
        tm, sink_size=5500, window_size=0, seq_len=5500,
        exact_layer_indices=list(range(30)))
    # S5 dramatically smaller than naive full-KV; growth = 5 full layers only.
    assert s5["total_resident_bytes"] < naive["total_resident_bytes"] / 5
    # per-token growth = 5 full layers * (2 * 2 kv * 512 * 2 bytes) = 20480 B
    assert s5["per_token_growth_bytes"] == 5 * (2 * 2 * 512 * 2)


def test_kv_memory_report_compression_shrinks_slope():
    tm = _gemma4_geom_model()
    full = [5, 11, 17, 23, 29]
    exact = cmv.kv_memory_report(
        tm, sink_size=4, window_size=64, seq_len=5500, exact_layer_indices=full)
    comp = cmv.kv_memory_report(
        tm, sink_size=4, window_size=64, seq_len=5500, exact_layer_indices=full,
        compress_full_bits_per_token_per_head=3232.0)
    # KakeyaLattice (~2.5x) shrinks the full-layer term + the linear slope.
    assert comp["total_resident_bytes"] < exact["total_resident_bytes"]
    assert comp["per_token_growth_bytes"] < exact["per_token_growth_bytes"]


def test_per_layer_kv_geometry():
    tm = _gemma4_geom_model()
    geom = cmv.per_layer_kv_geometry(tm)
    assert geom[0] == (8, 256, "sliding_attention")
    assert geom[5] == (2, 512, "full_attention")
