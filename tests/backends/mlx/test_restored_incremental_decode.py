"""Linux-CI tests for the MLX incremental restored-decode wrappers
(``restored_prefill_cache`` / ``restored_incremental_generate``).

These functions import ``mlx`` / ``mlx_lm`` lazily, so to exercise their
control flow on Linux (no Apple Silicon) we inject minimal fake ``mlx.core``
and ``mlx_lm`` modules via ``monkeypatch.setitem(sys.modules, ...)`` (auto
reverted). The real MLX kernels/cache behaviour are validated on a Mac by
``scripts/research/k3_integrated_niah_eval_mac.py --incremental``; here we lock
in the wrapper logic: which layers get the inject config, cache plumbing, and
the argmax/EOS/stop-condition decode loop.
"""

from __future__ import annotations

import sys
import types

import pytest

from inference_engine.backends.mlx import cross_model_dlm_verifier as cmv


# --------------------------------------------------------------------------- #
# Fake model structure
# --------------------------------------------------------------------------- #
class _FakeAttn:
    def __init__(self, layer_idx, has_kv=True):
        self.layer_idx = layer_idx
        self.has_kv = has_kv

    def __call__(self, *a, **k):  # present so _patched_attention_class can swap
        raise AssertionError("attn should not be invoked by the fake model")


class _FakeLayer:
    def __init__(self, attn):
        self.self_attn = attn


class _FakeTextModel:
    def __init__(self, n=6, shared=()):
        self.layers = [_FakeLayer(_FakeAttn(i, has_kv=i not in shared))
                       for i in range(n)]
        self.previous_kvs = list(range(n))
        self.embed_tokens = object()   # resolve_mlx_text_model sentinel


class _Logits:
    """Supports ``logits[0, -1]`` -> the last-row vocab list."""
    def __init__(self, row):
        self._row = row

    def __getitem__(self, key):
        assert key == (0, -1)
        return list(self._row)


class _FakeModel:
    """mlx_lm-like wrapper: ``.model`` is the text model and it is callable."""
    def __init__(self, tm, last_row):
        self.model = tm
        self._row = last_row
        self.captured_inject = None
        self.last_cache = "UNSET"

    def __call__(self, ids, cache=None):
        self.captured_inject = [
            l.self_attn.layer_idx for l in self.model.layers
            if getattr(l.self_attn, "_kakeya_inject", None)
            and l.self_attn._kakeya_inject.get("mode") == "inject"
        ]
        self.last_cache = cache
        return _Logits(self._row)


# --------------------------------------------------------------------------- #
# Fake mlx / mlx_lm modules
# --------------------------------------------------------------------------- #
class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


def _install_fakes(monkeypatch, *, prompt_cache="CACHE", gen_stream=()):
    mx = types.ModuleType("mlx.core")
    mx.array = lambda x, **k: x
    mx.eval = lambda *a, **k: None
    mx.argmax = lambda row, **k: _Scalar(int(max(range(len(row)),
                                                  key=lambda i: row[i])))
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx

    base = types.ModuleType("mlx_lm.models.base")
    base.scaled_dot_product_attention = lambda *a, **k: None
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model, **k: prompt_cache
    gen_mod = types.ModuleType("mlx_lm.generate")

    def _generate_step(prompt, model, *, prompt_cache=None, max_tokens=256, **k):
        for i, tok in enumerate(gen_stream):
            if i >= max_tokens:
                break
            yield tok, 0.0
    gen_mod.generate_step = _generate_step

    models_pkg = types.ModuleType("mlx_lm.models")
    mlx_lm_pkg = types.ModuleType("mlx_lm")
    for name, mod in [
        ("mlx", mlx_pkg), ("mlx.core", mx),
        ("mlx_lm", mlx_lm_pkg), ("mlx_lm.models", models_pkg),
        ("mlx_lm.models.base", base), ("mlx_lm.models.cache", cache_mod),
        ("mlx_lm.generate", gen_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


# --------------------------------------------------------------------------- #
# restored_prefill_cache
# --------------------------------------------------------------------------- #
def test_prefill_injects_only_source_layers_with_restored_kv(monkeypatch):
    _install_fakes(monkeypatch)
    tm = _FakeTextModel(n=6, shared=(5,))     # layer 5 is a KV-sharer
    model = _FakeModel(tm, last_row=[0.1, 0.9, 0.2])
    rk = {0: "k0", 2: "k2", 5: "k5"}          # 5 is sharer -> skipped
    rv = {0: "v0", 2: "v2", 5: "v5"}
    cache, last = cmv.restored_prefill_cache(
        model, [10, 11, 12, 13],
        restored_k_per_layer=rk, restored_v_per_layer=rv,
        evicted_positions=[1, 2])
    # Only has_kv layers present in rk get injected (0, 2). Layer 5 is a sharer
    # (skipped); layers 1,3,4 have no restored K/V (skipped).
    assert model.captured_inject == [0, 2]
    # Cache from make_prompt_cache is threaded into the forward and returned.
    assert cache == "CACHE"
    assert model.last_cache == "CACHE"
    # Last-row logits returned (predicts first token).
    assert last == [0.1, 0.9, 0.2]


def test_prefill_evicted_mask_clamped_and_attention_restored(monkeypatch):
    _install_fakes(monkeypatch)
    tm = _FakeTextModel(n=3)
    attn_cls = type(tm.layers[0].self_attn)
    orig_call = attn_cls.__call__
    model = _FakeModel(tm, last_row=[1.0, 0.0])
    # out-of-range evicted positions are ignored (clamped to prompt length)
    cmv.restored_prefill_cache(
        model, [7, 8], restored_k_per_layer={0: "k"}, restored_v_per_layer={0: "v"},
        evicted_positions=[0, 99, -1])
    # Attention __call__ restored after the context manager and inject config
    # cleared from every layer.
    assert attn_cls.__call__ is orig_call
    for l in tm.layers:
        assert not hasattr(l.self_attn, "_kakeya_inject")


# --------------------------------------------------------------------------- #
# restored_incremental_generate
# --------------------------------------------------------------------------- #
def test_generate_single_token_when_max_tokens_one(monkeypatch):
    _install_fakes(monkeypatch, gen_stream=[5, 6, 7])
    model = _FakeModel(_FakeTextModel(), last_row=None)
    out = cmv.restored_incremental_generate(
        model, "CACHE", [0.0, 0.0, 1.0], max_tokens=1)
    assert out == [2]                          # argmax of first_logits, no decode


def test_generate_stops_when_first_is_eos(monkeypatch):
    _install_fakes(monkeypatch, gen_stream=[5, 6])
    model = _FakeModel(_FakeTextModel(), last_row=None)
    out = cmv.restored_incremental_generate(
        model, "CACHE", [0.0, 9.0], max_tokens=16, eos_ids=[1])
    assert out == [1]                          # first token is EOS -> stop


def test_generate_streams_until_eos(monkeypatch):
    _install_fakes(monkeypatch, gen_stream=[5, 6, 99, 7])
    model = _FakeModel(_FakeTextModel(), last_row=None)
    out = cmv.restored_incremental_generate(
        model, "CACHE", [0.0, 0.0, 1.0], max_tokens=16, eos_ids=[99])
    # first = argmax([..1.0]) = 2, then stream 5,6 then EOS 99 (included, stops)
    assert out == [2, 5, 6, 99]


def test_generate_streams_until_max_tokens(monkeypatch):
    _install_fakes(monkeypatch, gen_stream=[5, 6, 7, 8, 9])
    model = _FakeModel(_FakeTextModel(), last_row=None)
    out = cmv.restored_incremental_generate(
        model, "CACHE", [9.0, 0.0], max_tokens=3)
    # first = argmax([9,0]) = 0, then generate_step capped at max_tokens-1 = 2
    assert out == [0, 5, 6]
