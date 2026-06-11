"""Linux-CI tests for the MLX native restored-cache primitive.

Covers the control flow + native ``.state`` writes with injected fake
``mlx`` / ``mlx_lm`` modules (no Apple Silicon). The real native prefill /
quantized decode are validated on a Mac via
``k3_integrated_niah_eval_mac.py --native-cache``.
"""

from __future__ import annotations

import sys
import types

import pytest

from inference_engine.backends.mlx import native_restored_cache as nrc


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, idx): self.idx = idx
    def __eq__(self, o): return isinstance(o, _Out) and o.idx == self.idx
    def __hash__(self): return self.idx


class _Layer:
    def __init__(self, idx): self.layer_idx = idx
    def __call__(self, x, *a, **k): return (_Out(self.layer_idx), None, 0)


class _Embed:
    def __call__(self, ids): return 1.0          # numeric so *embed_scale works
    def as_linear(self, h): return "L"


class _TextModel:
    def __init__(self, n=4):
        self.layers = [_Layer(i) for i in range(n)]
        self.embed_tokens = _Embed()


class _Logits:
    def __init__(self, row): self._row = row
    def __getitem__(self, k):
        assert k == (0, -1)
        return self._row


class _Model:
    def __init__(self, tm, row="LAST"):
        self.model = tm
        self._row = row
        self.last_cache = "UNSET"
        self.chunk_lens = []                      # query-lengths seen per forward

    def __call__(self, ids, cache=None):
        self.last_cache = cache
        self.chunk_lens.append(len(ids[0]))
        return _Logits(self._row)


class _KVCacheLayer:
    """Fake native KVCache: ``.state`` setter + nbytes + to_quantized + empty."""
    def __init__(self, nbytes=1000, empty=False):
        self._state = None
        self._nbytes = nbytes
        self._empty = empty
        self.quantized = False

    @property
    def state(self): return self._state

    @state.setter
    def state(self, v): self._state = v

    @property
    def nbytes(self): return self._nbytes

    def empty(self): return self._empty

    def to_quantized(self, *, group_size, bits):
        q = _KVCacheLayer(nbytes=self._nbytes // (16 // bits), empty=self._empty)
        q.quantized = True
        q.qparams = (group_size, bits)
        return q


class _Scalar:
    def __init__(self, v): self._v = v
    def item(self): return self._v


def _install_mlx(monkeypatch, prompt_cache=None):
    mx = types.ModuleType("mlx.core")
    mx.array = lambda x, **k: x
    mx.eval = lambda *a, **k: None
    mx.argmax = lambda r, **k: _Scalar(int(max(range(len(r)), key=lambda i: r[i])))
    mlx_pkg = types.ModuleType("mlx"); mlx_pkg.core = mx
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model, **k: (
        prompt_cache if prompt_cache is not None else ["c0", "c1"])
    gen_mod = types.ModuleType("mlx_lm.generate")
    gen_mod.generate_step = lambda *a, **k: iter(())
    for name, mod in [
        ("mlx", mlx_pkg), ("mlx.core", mx),
        ("mlx_lm", types.ModuleType("mlx_lm")),
        ("mlx_lm.models", types.ModuleType("mlx_lm.models")),
        ("mlx_lm.models.cache", cache_mod),
        ("mlx_lm.generate", gen_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return mx


# --------------------------------------------------------------------------- #
# set_kv_cache_state / inject_restored_into_native_cache
# --------------------------------------------------------------------------- #
def test_set_kv_cache_state_writes_state():
    c = _KVCacheLayer()
    nrc.set_kv_cache_state(c, "K", "V")
    assert c.state == ("K", "V")


def test_inject_restored_only_present_layers():
    cache = [_KVCacheLayer() for _ in range(4)]
    rk = {0: "k0", 2: "k2", 9: "k9"}             # 9 out of range -> ignored
    rv = {0: "v0", 2: "v2", 9: "v9"}
    out = nrc.inject_restored_into_native_cache(cache, rk, rv)
    assert cache[0].state == ("k0", "v0")
    assert cache[2].state == ("k2", "v2")
    assert cache[1].state is None and cache[3].state is None


def test_inject_restored_respects_layer_indices():
    cache = [_KVCacheLayer() for _ in range(3)]
    rk = {0: "k0", 1: "k1", 2: "k2"}
    nrc.inject_restored_into_native_cache(cache, rk, rk, layer_indices=[1])
    assert cache[1].state == ("k1", "k1")
    assert cache[0].state is None and cache[2].state is None


# --------------------------------------------------------------------------- #
# build_native_prefill_cache
# --------------------------------------------------------------------------- #
def test_native_prefill_chunks_the_prompt(monkeypatch):
    cache_layers = [_KVCacheLayer(), _KVCacheLayer()]
    _install_mlx(monkeypatch, prompt_cache=cache_layers)
    model = _Model(_TextModel(2), row="LASTROW")
    cache, last = nrc.build_native_prefill_cache(
        model, [1, 2, 3, 4, 5], prefill_step_size=2)
    # 5 tokens, step 2 -> chunked forwards of [2, 2, 1] (no single full forward).
    assert model.chunk_lens == [2, 2, 1]
    assert cache is cache_layers                 # native make_prompt_cache
    assert last == "LASTROW"                     # logits[0,-1] from final chunk


def test_native_prefill_single_chunk_when_step_large(monkeypatch):
    cache_layers = [_KVCacheLayer()]
    _install_mlx(monkeypatch, prompt_cache=cache_layers)
    model = _Model(_TextModel(1), row="R")
    nrc.build_native_prefill_cache(model, [1, 2, 3], prefill_step_size=512)
    assert model.chunk_lens == [3]


def test_native_prefill_rejects_empty_prompt(monkeypatch):
    _install_mlx(monkeypatch)
    with pytest.raises(ValueError):
        nrc.build_native_prefill_cache(_Model(_TextModel(2)), [])


# --------------------------------------------------------------------------- #
# quantize_full_attn_layers / cache_resident_bytes
# --------------------------------------------------------------------------- #
def test_quantize_full_attn_layers():
    cache = [_KVCacheLayer(nbytes=1600) for _ in range(6)]
    cache[3] = _KVCacheLayer(nbytes=1600, empty=True)   # empty -> skipped
    out = nrc.quantize_full_attn_layers(cache, [1, 3, 5, 99], bits=8, group_size=64)
    assert out[1].quantized and out[1].qparams == (64, 8)
    assert out[5].quantized
    assert not out[3].quantized                  # empty layer untouched
    assert not out[0].quantized                  # not in full-attn list
    # quantized layer reports smaller nbytes (real memory win)
    assert out[1].nbytes < cache[0].nbytes


def test_cache_resident_bytes_sums_nbytes():
    cache = [_KVCacheLayer(nbytes=100), _KVCacheLayer(nbytes=250)]
    assert nrc.cache_resident_bytes(cache) == 350


def test_native_decode_delegates(monkeypatch):
    _install_mlx(monkeypatch)
    # generate_step yields nothing -> just the argmax first token
    out = nrc.native_restored_decode(_Model(_TextModel(2)), ["C"], [0.0, 1.0, 0.0],
                                     max_tokens=8)
    assert out == [1]                            # argmax of first_logits


def _install_mlx_genstream(monkeypatch, stream, prompt_cache):
    mx = _install_mlx(monkeypatch, prompt_cache=prompt_cache)
    gen_mod = sys.modules["mlx_lm.generate"]
    seen = {}

    def _gen_step(prompt, model, *, prompt_cache=None, max_tokens=256, **k):
        seen["prompt"] = prompt
        seen["kwargs"] = k
        for i, t in enumerate(stream):
            if i >= max_tokens:
                break
            yield t, 0.0
    gen_mod.generate_step = _gen_step
    return seen


def test_native_generate_end_to_end_and_eos(monkeypatch):
    cache_layers = [_KVCacheLayer()]
    seen = _install_mlx_genstream(monkeypatch, [5, 6, 99, 7], cache_layers)
    out, cache = nrc.native_generate(
        _Model(_TextModel(1)), [1, 2, 3], max_tokens=16, eos_ids=[99],
        prefill_step_size=128, kv_bits=8)
    assert out == [5, 6, 99]                     # stops at EOS (included)
    assert cache is cache_layers
    assert seen["prompt"] == [1, 2, 3]           # whole prompt handed to generate_step
    assert seen["kwargs"]["prefill_step_size"] == 128
    assert seen["kwargs"]["kv_bits"] == 8


def test_native_generate_rejects_empty(monkeypatch):
    _install_mlx(monkeypatch)
    with pytest.raises(ValueError):
        nrc.native_generate(_Model(_TextModel(1)), [], max_tokens=4)
