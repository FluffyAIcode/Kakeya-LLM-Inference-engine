"""Linux-CI tests for the MLX fused DFlash spec-decode engine.

The fused loop (``fused_specdecode_generate``) takes all MLX/torch ops as
injected callables, so its accept/reject/commit/extend control flow is tested
**without MLX**. The MLX-touching wrappers (``_build_aux``, ``capture_aux_hidden``,
``MLXRestoredIncrementalVerifier``, ``make_bridge_embed_lm_head``) are tested by
injecting fake ``mlx`` / ``mlx_lm`` modules. Real MLX kernels are validated on a
Mac by ``k3_integrated_niah_eval_mac.py --fused-specdecode``.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from inference_engine.backends.mlx import fused_specdecode as fsd


# =========================================================================== #
# 1) Fused loop control flow (no MLX) — verifier truth = successor (last+1).
# =========================================================================== #
class _FakeAdapter:
    def __init__(self, prompt_len, first_token, hidden=4):
        self._past_len = prompt_len
        self.next_token_logits = first_token
        self.hidden = hidden
        self._capture_aux = False
        self._last_aux = None
        self.commits = []
        self.appends = []

    def forward_block(self, candidate):
        # verifier greedy continuation: prediction after token t is t+1.
        block_logits = [candidate[i] + 1 for i in range(len(candidate))]
        if self._capture_aux:
            L = len(candidate)
            self._last_aux = [torch.arange(L * self.hidden).float().reshape(L, self.hidden)]
        return block_logits

    def commit_or_truncate(self, *, forwarded, accepted):
        self.commits.append((forwarded, accepted))
        self._past_len += accepted

    def append_token(self, token_id):
        bl = self.forward_block([token_id])
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = bl[-1]
        self.appends.append(token_id)
        return self.next_token_logits

    def last_aux_torch_slice(self, start=0, end=None):
        # Mirror MLXRestoredIncrementalVerifier.last_aux_torch_slice: per-aux-layer
        # torch rows of the most recent forward_block, sliced [start:end].
        aux = self._last_aux or [torch.zeros(1, self.hidden)]
        return [a[start:end] for a in aux]


class _FakeDrafter:
    def __init__(self, drafts):
        self.cfg = types.SimpleNamespace(aux_layer_ids=(2,))
        self._drafts = list(drafts)
        self.make_calls = 0
        self.extend_calls = 0

    def make_context_kv(self, aux, positions):
        self.make_calls += 1
        return ("ctx", self.make_calls)

    def extend_context_kv(self, ctx_kv, new_kv):
        self.extend_calls += 1
        return ("ctx_ext", self.extend_calls)

    def draft_block_cached(self, ctx_kv, bonus, embed_fn, lm_head_fn,
                           *, block_size, context_len):
        return list(self._drafts.pop(0)) if self._drafts else []


def _loop_kwargs(drafter, **over):
    kw = dict(
        aux_prompt=[torch.zeros(1, 5, 4)],
        embed_fn=lambda x: x, lm_head_fn=lambda x: x,
        argmax_fn=lambda row: int(row), arange_fn=lambda s, e: torch.arange(s, e),
        cat_aux_fn=lambda parts: torch.cat(list(parts), dim=0).unsqueeze(0),
    )
    kw.update(over)
    return kw


def test_fused_loop_full_acceptance():
    adapter = _FakeAdapter(prompt_len=5, first_token=100)
    drafter = _FakeDrafter(drafts=[[101, 102], [200, 201]])
    res = fsd.fused_specdecode_generate(
        adapter, drafter, gen_tokens=5, block_size=4, eos_ids=(),
        **_loop_kwargs(drafter))
    # Block1: candidate=[100,101,102] fully accepted (3). On FULL acceptance the
    # loop reuses block_logits[-1] (=103) as the next distribution and does NOT
    # append a correction token. next=103.
    # Block2: L=2 -> candidate=[103,200]; accept 103 (1), reject 200, correction
    #   =104 appended -> commit [103,104]; total 5 tokens.
    assert res["tokens"] == [100, 101, 102, 103, 104]
    assert res["blocks"] == 2
    assert res["mean_accept_len"] == 2.0          # (3 + 1) / 2
    assert adapter.commits[0] == (3, 3)           # block1 verify-commit
    assert adapter.appends == [104]               # only block2's correction
    # capture flag toggled on during loop, off after.
    assert adapter._capture_aux is False
    # context K/V extended once per block.
    assert drafter.extend_calls == 2


def test_fused_loop_partial_rejection_and_correction():
    adapter = _FakeAdapter(prompt_len=5, first_token=100)
    drafter = _FakeDrafter(drafts=[[101, 777]])   # 777 mismatches verifier 102
    res = fsd.fused_specdecode_generate(
        adapter, drafter, gen_tokens=3, block_size=4, eos_ids=(),
        **_loop_kwargs(drafter))
    # candidate=[100,101,777]: accept 100,101 (2), reject 777, correction=102.
    assert res["tokens"] == [100, 101, 102]
    assert res["blocks"] == 1
    assert res["mean_accept_len"] == 2.0
    # commit_or_truncate(forwarded=3, accepted=2) then append correction (1,1).
    assert adapter.commits[0] == (3, 2)


def test_fused_loop_stops_on_eos():
    adapter = _FakeAdapter(prompt_len=5, first_token=100)
    drafter = _FakeDrafter(drafts=[[101, 102]])
    res = fsd.fused_specdecode_generate(
        adapter, drafter, gen_tokens=50, block_size=4, eos_ids=(103,),
        **_loop_kwargs(drafter))
    # Block1 fully accepts [100,101,102] (no correction appended on full accept),
    # leaving next=103. Block2's bonus is then 103 (EOS), committed and stopped.
    assert res["tokens"] == [100, 101, 102, 103]
    assert res["blocks"] == 2


def test_fused_loop_greedy_fallback_on_low_acceptance():
    adapter = _FakeAdapter(prompt_len=5, first_token=100)
    # Each block accepts only the bonus (drafts mismatch the verifier), so after
    # 2 blocks mean acceptance = 1.0 < 1.5 and the loop switches to plain greedy
    # to finish the budget (no aux capture, no drafter extension past the blocks).
    drafter = _FakeDrafter(drafts=[[999, 999, 999], [999, 999, 999]])
    res = fsd.fused_specdecode_generate(
        adapter, drafter, gen_tokens=10, block_size=4, eos_ids=(),
        **_loop_kwargs(drafter))
    # blocks 1-2 commit [100,101] then [102,103]; greedy fallback adds 104..109.
    assert res["tokens"] == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    assert res["blocks"] == 2              # only the speculative blocks are counted
    assert res["mean_accept_len"] == 1.0   # (1 + 1) / 2
    assert adapter._capture_aux is False   # turned off for the greedy tail
    assert drafter.extend_calls == 2       # extended only during the spec blocks


# =========================================================================== #
# 2) MLX-touching wrappers with fake mlx / mlx_lm.
# =========================================================================== #
class _Out:
    """Stand-in for a layer's [1, L, hidden] output; ``[0]`` strips batch."""
    def __init__(self, idx):
        self.idx = idx

    def __getitem__(self, k):
        return ("row", self.idx)

    def __eq__(self, o):
        return isinstance(o, _Out) and o.idx == self.idx

    def __hash__(self):
        return self.idx


class _Layer:
    def __init__(self, idx):
        self.layer_idx = idx

    def __call__(self, x, *a, **k):
        return (_Out(self.layer_idx), None, 0)   # (h, kvs, offset)


class _TextModel:
    def __init__(self, n=4):
        self.layers = [_Layer(i) for i in range(n)]
        self.embed_tokens = self._Embed()

    class _Embed:
        def __call__(self, ids):
            return "EMB"

        def as_linear(self, h):
            return "LOGITS"


class _Model:
    def __init__(self, tm, row=None):
        self.model = tm
        self._row = row
        self.last_cache = "UNSET"

    def __call__(self, ids, cache=None):
        # drive the (patched) layers so their _aux_record gets populated
        tm = self.model
        for l in tm.layers:
            l(None)
        self.last_cache = cache
        if self._row is not None:
            class _L:
                def __init__(self, r): self._r = r
                def __getitem__(self, k):
                    assert k == 0
                    return self._r
            return _L(self._row)
        return None


def _install_mlx(monkeypatch, trim_log=None):
    mx = types.ModuleType("mlx.core")
    mx.array = lambda x, **k: x
    mx.eval = lambda *a, **k: None
    mx.argmax = lambda r, **k: r
    mx.tanh = lambda x: x / 2          # non-identity marker so softcap is visible
    mlx_pkg = types.ModuleType("mlx"); mlx_pkg.core = mx
    cache_mod = types.ModuleType("mlx_lm.models.cache")

    def _trim(cache, n):
        if trim_log is not None:
            trim_log.append((cache, n))
    cache_mod.trim_prompt_cache = _trim
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


def test_build_aux_indexing(monkeypatch):
    _install_mlx(monkeypatch)
    tm = _TextModel(n=4)

    class _E:
        def __call__(self, ids): return 1.0     # numeric so *embed_scale works
        def as_linear(self, h): return "L"
    tm.embed_tokens = _E()
    sink = {0: "h0", 1: "h1", 2: "h2", 3: "h3"}
    # hs = [scaled_embeds(=2.0), h0, h1, h2, h3]; hs[a] = output of layer a-1.
    aux = fsd._build_aux(tm, "ids", sink, embed_scale=2.0, aux_layer_ids=[0, 1, 3])
    assert aux == [2.0, "h0", "h2"]              # hs[0]=embeds, hs[1]=h0, hs[3]=h2


def test_capture_aux_hidden_runs_layers_and_indexes(monkeypatch):
    _install_mlx(monkeypatch)
    tm = _TextModel(n=3)

    class _E:
        def __call__(self, ids): return 1.0
        def as_linear(self, h): return "L"
    tm.embed_tokens = _E()
    model = _Model(tm)
    aux = fsd.capture_aux_hidden(model, [1, 2], aux_layer_ids=[1, 3],
                                 embed_scale=10.0)
    # hs[1] = output of layer 0; hs[3] = output of layer 2.
    assert aux == [_Out(0), _Out(2)]
    # _aux_record cleared from layers after capture.
    for l in tm.layers:
        assert not hasattr(l, "_aux_record")


def test_adapter_prefill_forward_commit(monkeypatch):
    trim_log = []
    _install_mlx(monkeypatch, trim_log=trim_log)
    tm = _TextModel(n=3)

    class _E:
        def __call__(self, ids): return 1.0
        def as_linear(self, h): return "L"
    tm.embed_tokens = _E()
    model = _Model(tm, row="ROW")

    # patch restored_prefill_cache to a sentinel (its own test covers internals)
    monkeypatch.setattr(fsd, "restored_prefill_cache",
                        lambda m, ids, **k: ("CACHE", "FIRST"))
    adapter = fsd.MLXRestoredIncrementalVerifier(
        model, embed_scale=10.0, aux_layer_ids=(1,),
        bridge_to_torch=lambda a: ("torch", a))
    adapter.prefill([1, 2, 3], restored_k_per_layer={}, restored_v_per_layer={},
                    evicted_positions=[1])
    assert adapter._cache == "CACHE"
    assert adapter.next_token_logits == "FIRST"
    assert adapter._past_len == 3

    # forward_block with aux capture -> bridges hs[1] = layer-0 output
    adapter._capture_aux = True
    logits = adapter.forward_block([7, 8])
    assert logits == "ROW"                       # _Model returns row at [0]
    # aux = [hs[1]] = [layer-0 output], captured LAZILY in MX (_last_aux_mx);
    # _last_aux stays None and the torch bridge happens on demand.
    assert adapter._last_aux is None
    assert adapter.last_aux_torch_slice() == [("torch", ("row", 0))]

    # commit_or_truncate trims by (forwarded - accepted) and advances _past_len
    adapter.commit_or_truncate(forwarded=2, accepted=1)
    assert trim_log == [("CACHE", 1)]
    assert adapter._past_len == 4

    # no trim when fully accepted
    trim_log.clear()
    adapter.commit_or_truncate(forwarded=2, accepted=2)
    assert trim_log == []
    assert adapter._past_len == 6


def test_adapter_append_token_and_non_aux_path(monkeypatch):
    _install_mlx(monkeypatch)
    tm = _TextModel(n=2)
    model = _Model(tm, row=[10, 11, 12])         # forward_block -> row at [0]
    adapter = fsd.MLXRestoredIncrementalVerifier(model, embed_scale=1.0)
    adapter._cache = "C"
    adapter._past_len = 5
    # _capture_aux stays False -> non-aux branch; append_token commits (1,1).
    nxt = adapter.append_token(99)
    assert adapter._last_aux is None             # non-aux path
    assert nxt == 12                             # logits[-1]
    assert adapter._past_len == 6


def test_adapter_prefill_rejects_empty_prompt(monkeypatch):
    _install_mlx(monkeypatch)
    adapter = fsd.MLXRestoredIncrementalVerifier(_Model(_TextModel(2)), embed_scale=1.0)
    with pytest.raises(ValueError):
        adapter.prefill([], restored_k_per_layer={}, restored_v_per_layer={},
                        evicted_positions=[])


def test_make_full_kv_prompt_cache_all_kvcache(monkeypatch):
    # Fake mlx_lm.models.cache with make_prompt_cache (count) + a KVCache class.
    import types as _t
    class _FakeKV:
        instances = 0
        def __init__(self): type(self).instances += 1
    cache_mod = _t.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model, **k: ["a", "b", "c", "d"]  # 4 layers
    cache_mod.KVCache = _FakeKV
    monkeypatch.setitem(sys.modules, "mlx_lm", _t.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models", _t.ModuleType("mlx_lm.models"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    out = fsd.make_full_kv_prompt_cache(object())
    assert len(out) == 4 and all(isinstance(c, _FakeKV) for c in out)
    assert _FakeKV.instances == 4   # every layer is a fresh full KVCache


def test_patched_decoder_layers_empty_is_noop(monkeypatch):
    _install_mlx(monkeypatch)
    tm = _TextModel(0)
    with fsd._patched_decoder_layers(tm):
        pass                                     # no layers -> no-op guard


def test_adapter_commit_validates_accepted(monkeypatch):
    _install_mlx(monkeypatch)
    tm = _TextModel(n=2)
    adapter = fsd.MLXRestoredIncrementalVerifier(_Model(tm), embed_scale=1.0)
    adapter._cache = "C"
    with pytest.raises(ValueError):
        adapter.commit_or_truncate(forwarded=2, accepted=3)


def test_adapter_forward_block_requires_prefill(monkeypatch):
    _install_mlx(monkeypatch)
    tm = _TextModel(n=2)
    adapter = fsd.MLXRestoredIncrementalVerifier(_Model(tm), embed_scale=1.0)
    with pytest.raises(RuntimeError):
        adapter.forward_block([1])
    adapter._cache = "C"
    with pytest.raises(ValueError):
        adapter.forward_block([])


def test_bridge_embed_is_unscaled_and_lm_head_softcaps(monkeypatch):
    mx = _install_mlx(monkeypatch)
    tm = _TextModel(n=2)
    seen = {}

    class _E:
        def __call__(self, ids):
            seen["embed_ids"] = ids
            return "RAW_EMB"                       # NOT multiplied by embed_scale

        def as_linear(self, h):
            seen["as_linear_h"] = h
            return 100.0
    tm.embed_tokens = _E()

    embed_fn, lm_head_fn = fsd.make_bridge_embed_lm_head(
        tm, mx_to_torch=lambda a, **k: ("mt", a),
        torch_to_mx=lambda h: ("tm", h),
        device="cpu", torch_dtype="f32", softcap=50.0)

    class _Ids:
        def detach(self): return self
        def to(self, d): return self
        def tolist(self): return [[1, 2]]
    out_emb = embed_fn(_Ids())
    assert out_emb == ("mt", "RAW_EMB")            # plain lookup (Gap-B: no *scale)

    out_logits = lm_head_fn("H")
    # softcap*tanh(as_linear/softcap): 50*tanh(100/50)=50*(2/2)=50 (fake tanh=x/2)
    assert seen["as_linear_h"] == ("tm", "H")
    assert out_logits == ("mt", 50.0)              # softcap path applied
