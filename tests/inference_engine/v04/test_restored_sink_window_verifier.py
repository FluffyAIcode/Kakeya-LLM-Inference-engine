"""Unit tests for the Gap 1 + Gap 2 served-path integration.

Covers, on CPU with tiny synthetic stand-ins (no real models):

* :class:`CrossModelRestoredSinkWindowVerifier` — the full
  ``SinkWindowVerifier`` public surface, with assertions that
  ``forward_block`` is bit-equivalent to the underlying restored forward.
* End-to-end :class:`SpeculativeDecoder` integration over the restored
  adapter (accept-all path and reject-all path), proving the served
  output equals greedy restored-AR.
* :func:`build_restored_speculative_decoder` factory.

The heavy ``load_restored_verifier`` model loader is coverage-exempt
(``# pragma: no cover``) and validated by GPU integration runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from inference_engine.v04 import (
    CrossModelRestoredSinkWindowVerifier,
    build_restored_speculative_decoder,
)

V = 16  # synthetic vocab size


# --------------------------------------------------------------------------- #
# Synthetic stand-ins
# --------------------------------------------------------------------------- #
class _Cfg:
    """Verifier text-config shape consumed by the adapter's KV accounting."""

    def __init__(
        self,
        num_hidden_layers=3,
        num_key_value_heads=4,
        head_dim=8,
        hidden_size=32,
        num_attention_heads=4,
    ):
        self.num_hidden_layers = num_hidden_layers
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads
        if head_dim is not None:
            self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


class _Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2, bias=False)


class _FakeVerifierModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg or _Cfg()
        self.lin = nn.Linear(2, 2, bias=False)


class _FakeRestored:
    """Deterministic stand-in for CrossModelDLMRestoredVerifier.

    Implements an "increment" language model: the predicted next token
    after seeing token ``x`` is ``(x + 1) % V``. ``forward`` returns
    ``[1, T, V]`` logits whose argmax at position ``t`` is
    ``(seq[t] + 1) % V``. This makes greedy restored-AR fully predictable.
    """

    def __init__(self, sink_size=2, window_size=4, bare_tensor=False, cfg=None):
        self.sink_size = sink_size
        self.window_size = window_size
        self.verifier_model = _FakeVerifierModel(cfg)
        self.drafter = _Param()
        self.f_theta = _Param()
        self._bare_tensor = bare_tensor
        self.seen_helpers = []

    def forward(
        self,
        input_ids,
        *,
        apply_rotary_pos_emb=None,
        eager_attention_forward=None,
        all_attention_functions=None,
    ):
        self.seen_helpers.append(
            (apply_rotary_pos_emb, eager_attention_forward, all_attention_functions)
        )
        seq = input_ids[0].tolist()
        T = len(seq)
        logits = torch.full((1, T, V), -10.0)
        for t, tok in enumerate(seq):
            logits[0, t, (int(tok) + 1) % V] = 10.0
        if self._bare_tensor:
            return logits
        return SimpleNamespace(logits=logits)


class _FakeProposer:
    """Minimal DLMProposer stand-in: ``propose_block`` returns whatever
    ``predict_fn(committed, L)`` yields. Carries the stats attributes the
    SpeculativeDecoder resets/reads."""

    def __init__(self, predict_fn):
        self._predict = predict_fn
        self.stats = SimpleNamespace(
            total_blocks=0,
            total_diffusion_steps=0,
            total_forward_passes=0,
            peak_activation_bytes=0,
            weight_bytes=0,
        )

    def propose_block(self, committed_token_ids, block_size, num_steps):
        toks = self._predict(list(committed_token_ids), block_size)
        return SimpleNamespace(tokens=list(toks))


def _make_adapter(**kw):
    restored = _FakeRestored(**kw)
    sentinel_aprp = object()
    sentinel_eager = object()
    sentinel_all = object()
    adapter = CrossModelRestoredSinkWindowVerifier(
        restored,
        apply_rotary_pos_emb=sentinel_aprp,
        eager_attention_forward=sentinel_eager,
        all_attention_functions=sentinel_all,
        device="cpu",
    )
    return adapter, restored, (sentinel_aprp, sentinel_eager, sentinel_all)


# --------------------------------------------------------------------------- #
# Construction / accounting
# --------------------------------------------------------------------------- #
def test_construction_basic():
    adapter, restored, _ = _make_adapter(sink_size=2, window_size=4)
    assert adapter.sink_size == 2
    assert adapter.window_size == 4
    assert adapter.cache is None
    assert adapter.cache_logical_size == 0
    assert adapter.next_global_position == 0
    assert adapter.next_token_logits is None
    assert adapter.cached_token_sequence == []
    assert adapter.model is restored.verifier_model
    # weight_bytes sums verifier + drafter + f_theta params (>0).
    assert adapter.stats.weight_bytes > 0
    assert adapter._bytes_per_kv_token > 0


def test_weight_bytes_skips_module_without_parameters():
    restored = _FakeRestored()
    restored.drafter = object()  # no .parameters → exercised `continue`
    adapter = CrossModelRestoredSinkWindowVerifier(
        restored,
        apply_rotary_pos_emb=None,
        eager_attention_forward=None,
        all_attention_functions=None,
    )
    assert adapter.stats.weight_bytes > 0  # verifier + f_theta still counted


def test_bytes_per_kv_token_head_dim_present():
    adapter, _, _ = _make_adapter()
    cfg = _Cfg(num_hidden_layers=3, num_key_value_heads=4, head_dim=8)
    expected = 3 * 4 * 8 * 4 * 2  # layers*kv_heads*head_dim*itemsize(fp32)*2
    assert adapter._bytes_per_kv_token == expected


def test_bytes_per_kv_token_head_dim_derived_from_hidden():
    cfg = _Cfg(num_key_value_heads=2, head_dim=None,
               hidden_size=32, num_attention_heads=4)
    adapter, _, _ = _make_adapter(cfg=cfg)
    # head_dim = hidden_size // num_attention_heads = 32 // 4 = 8
    expected = 3 * 2 * 8 * 4 * 2
    assert adapter._bytes_per_kv_token == expected


def test_bytes_per_kv_token_default_itemsize_when_no_params():
    # Verifier model with no parameters → itemsize loop does zero
    # iterations → default itemsize (4) is used.
    class _NoParamVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg(num_hidden_layers=3, num_key_value_heads=4,
                               head_dim=8)

    restored = _FakeRestored()
    restored.verifier_model = _NoParamVerifier()
    adapter = CrossModelRestoredSinkWindowVerifier(
        restored,
        apply_rotary_pos_emb=None,
        eager_attention_forward=None,
        all_attention_functions=None,
    )
    assert adapter._bytes_per_kv_token == 3 * 4 * 8 * 4 * 2


def test_bytes_per_kv_token_kv_heads_fallback_and_zero_qheads():
    # No num_key_value_heads → falls back to num_attention_heads; head_dim
    # None and num_attention_heads=0 → head_dim resolves to 0.
    cfg = _Cfg(num_key_value_heads=None, head_dim=None,
               hidden_size=0, num_attention_heads=0)
    adapter, _, _ = _make_adapter(cfg=cfg)
    assert adapter._bytes_per_kv_token == 0


# --------------------------------------------------------------------------- #
# prefill
# --------------------------------------------------------------------------- #
def test_prefill_empty_raises():
    adapter, _, _ = _make_adapter()
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        adapter.prefill([])


def test_prefill_sets_next_token_logits_and_passes_helpers():
    adapter, restored, sentinels = _make_adapter(sink_size=2, window_size=4)
    prompt = [5, 6, 7]
    adapter.prefill(prompt)
    # next_token_logits predicts (last_token + 1) % V
    assert int(torch.argmax(adapter.next_token_logits)) == (7 + 1) % V
    assert adapter.next_global_position == 3
    assert adapter.cached_token_sequence == [5, 6, 7]  # <= budget=6
    assert adapter.cache_logical_size == 3
    assert adapter.stats.forward_calls == 1
    assert adapter.stats.tokens_consumed == 3
    assert adapter.stats.peak_activation_bytes > 0
    assert adapter.stats.peak_kv_bytes > 0
    # the configured HF helpers were threaded through to restored.forward
    assert restored.seen_helpers[-1] == sentinels


def test_prefill_bounds_resident_cache_when_over_budget():
    adapter, _, _ = _make_adapter(sink_size=2, window_size=4)  # budget 6
    prompt = list(range(10))  # length 10 > 6
    adapter.prefill(prompt)
    # sink (first 2) + window (last 4)
    assert adapter.cached_token_sequence == [0, 1, 6, 7, 8, 9]
    assert adapter.cache_logical_size == 6
    assert adapter.next_global_position == 10  # logical length unbounded


# --------------------------------------------------------------------------- #
# forward_block
# --------------------------------------------------------------------------- #
def test_forward_block_requires_prefill():
    adapter, _, _ = _make_adapter()
    with pytest.raises(RuntimeError, match="not prefilled"):
        adapter.forward_block([1, 2])


def test_forward_block_empty_raises():
    adapter, _, _ = _make_adapter()
    adapter.prefill([1, 2, 3])
    with pytest.raises(ValueError, match="tokens must be non-empty"):
        adapter.forward_block([])


def test_forward_block_equivalent_to_restored_forward():
    adapter, restored, _ = _make_adapter(sink_size=2, window_size=4)
    prompt = [3, 4, 5]
    adapter.prefill(prompt)
    block = [9, 1]
    out = adapter.forward_block(block)  # [2, V]
    assert tuple(out.shape) == (2, V)
    # Equivalence: forward_block rows == restored.forward(prompt+block) slice
    ref = restored.forward(
        torch.tensor([prompt + block]),
    ).logits[0]
    assert torch.equal(out, ref[len(prompt):len(prompt) + len(block)])
    # argmax rows predict (token+1)%V
    assert int(torch.argmax(out[0])) == (9 + 1) % V
    assert int(torch.argmax(out[1])) == (1 + 1) % V
    # provisional resident size = committed + L (un-trimmed pre-commit)
    assert adapter.cache_logical_size == 3 + 2
    assert adapter.stats.forward_calls == 2  # prefill + this block


# --------------------------------------------------------------------------- #
# commit_or_truncate
# --------------------------------------------------------------------------- #
def test_commit_invalid_accepted_raises():
    adapter, _, _ = _make_adapter()
    adapter.prefill([1, 2, 3])
    adapter.forward_block([4, 5])
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        adapter.commit_or_truncate(forwarded=2, accepted=3)


def test_commit_accept_partial_extends_committed():
    adapter, _, _ = _make_adapter(sink_size=2, window_size=4)
    adapter.prefill([1, 2, 3])
    adapter.forward_block([4, 5])
    adapter.commit_or_truncate(forwarded=2, accepted=1)  # keep only 4
    assert adapter.next_global_position == 4
    assert adapter.cached_token_sequence == [1, 2, 3, 4]
    assert adapter._committed == [1, 2, 3, 4]


def test_commit_accept_zero_keeps_committed():
    adapter, _, _ = _make_adapter()
    adapter.prefill([1, 2, 3])
    adapter.forward_block([4, 5])
    adapter.commit_or_truncate(forwarded=2, accepted=0)
    assert adapter._committed == [1, 2, 3]
    assert adapter.next_global_position == 3


# --------------------------------------------------------------------------- #
# append_token
# --------------------------------------------------------------------------- #
def test_append_token_advances_and_predicts():
    adapter, _, _ = _make_adapter(sink_size=2, window_size=4)
    adapter.prefill([1, 2, 3])
    nt = adapter.append_token(8)
    assert adapter._committed == [1, 2, 3, 8]
    assert adapter.next_global_position == 4
    # predicts (8 + 1) % V
    assert int(torch.argmax(nt)) == (8 + 1) % V


# --------------------------------------------------------------------------- #
# CacheInspector accessors
# --------------------------------------------------------------------------- #
def test_cache_inspector_accessors():
    adapter, _, _ = _make_adapter(sink_size=2, window_size=4)
    adapter.prefill(list(range(10)))
    assert adapter.k_seq_length(object()) == 6
    assert adapter.kv_live_bytes(object()) == 6 * adapter._bytes_per_kv_token
    assert adapter.live_kv_bytes() == 6 * adapter._bytes_per_kv_token


# --------------------------------------------------------------------------- #
# _sync_bounded_state window edge + _restored_logits bare-tensor + peak
# --------------------------------------------------------------------------- #
def test_sync_zero_window_keeps_only_sink():
    # window_size = 0 → budget == sink, keep_window <= 0 branch.
    adapter, _, _ = _make_adapter(sink_size=2, window_size=0)
    adapter.prefill([1, 2, 3, 4, 5])
    assert adapter.cached_token_sequence == [1, 2]
    assert adapter.cache_logical_size == 2


def test_restored_forward_returns_bare_tensor():
    adapter, _, _ = _make_adapter(bare_tensor=True, sink_size=2, window_size=4)
    adapter.prefill([2, 3, 4])
    assert int(torch.argmax(adapter.next_token_logits)) == (4 + 1) % V


def test_record_peak_activation_keeps_max():
    adapter, _, _ = _make_adapter()
    big = torch.zeros(1, 100, V)
    small = torch.zeros(1, 1, V)
    adapter._record_peak_activation(big)
    peak = adapter.stats.peak_activation_bytes
    adapter._record_peak_activation(small)  # not greater → unchanged
    assert adapter.stats.peak_activation_bytes == peak


# --------------------------------------------------------------------------- #
# Incremental-decode path (Gap-A throughput) — exercised with a fake model
# that uses a real transformers DynamicCache so the cache bookkeeping
# (build / append / truncate / position tracking) is covered on CPU.
# --------------------------------------------------------------------------- #
class _FakeIncVerifierModel(nn.Module):
    def __init__(self, n_layers, V):
        super().__init__()
        self.config = _Cfg()
        self.lin = nn.Linear(2, 2, bias=False)
        self.model = SimpleNamespace(layers=[object() for _ in range(n_layers)])
        self._n = n_layers
        self._V = V

    def forward(self, input_ids=None, position_ids=None, cache_position=None,
                past_key_values=None, use_cache=False, **kw):
        seq = input_ids[0].tolist()
        L = len(seq)
        logits = torch.full((1, L, self._V), -10.0)
        for t, tk in enumerate(seq):
            logits[0, t, (int(tk) + 1) % self._V] = 10.0
        if past_key_values is not None:
            for i in range(self._n):
                past_key_values.update(
                    torch.zeros(1, 2, L, 4), torch.zeros(1, 2, L, 4), i)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class _FakeRestoredInc:
    def __init__(self, n_layers=3, V=16, sink=2, window=4, incomplete=False):
        self.sink_size = sink
        self.window_size = window
        self.verifier_model = _FakeIncVerifierModel(n_layers, V)
        self.drafter = _Param()
        self.f_theta = _Param()
        self._n = n_layers
        self._V = V
        self._incomplete = incomplete

    def forward(self, input_ids, *, apply_rotary_pos_emb=None,
                eager_attention_forward=None, all_attention_functions=None,
                capture_kv=None):
        seq = input_ids[0].tolist()
        T = len(seq)
        logits = torch.full((1, T, self._V), -10.0)
        for t, tk in enumerate(seq):
            logits[0, t, (int(tk) + 1) % self._V] = 10.0
        if capture_kv is not None:
            for i in range(self._n):
                if self._incomplete and i == self._n - 1:
                    continue  # leave a None to trigger the guard
                capture_kv[i] = (torch.zeros(1, 2, T, 4), torch.zeros(1, 2, T, 4))
        return SimpleNamespace(logits=logits)


def _make_inc_adapter(**kw):
    restored = _FakeRestoredInc(**kw)
    return CrossModelRestoredSinkWindowVerifier(
        restored, apply_rotary_pos_emb=None, eager_attention_forward=None,
        all_attention_functions=None, incremental=True), restored


def test_incremental_prefill_builds_cache():
    a, _ = _make_inc_adapter(sink=2, window=4)
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])  # T=8 > budget 6 → eviction → capture
    assert a._past is not None
    assert a._past_len == 8
    assert len(a._past.layers) == 3
    assert int(a.next_token_logits.argmax()) == (8 + 1) % V


def test_incremental_capture_incomplete_raises():
    a, _ = _make_inc_adapter(incomplete=True)
    with pytest.raises(RuntimeError, match="not captured"):
        a.prefill([1, 2, 3, 4, 5, 6, 7, 8])


def test_incremental_forward_block_native_and_commit_accept_all():
    a, _ = _make_inc_adapter(sink=2, window=4)
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    blk = a.forward_block([9, 1])
    assert int(blk[0].argmax()) == (9 + 1) % V
    assert int(blk[1].argmax()) == (1 + 1) % V
    assert a._past.layers[0].keys.shape[2] == 8 + 2  # appended
    a.commit_or_truncate(forwarded=2, accepted=2)
    assert a._past_len == 10
    assert a._committed[-2:] == [9, 1]


def test_incremental_commit_truncates_rejected_tail():
    a, _ = _make_inc_adapter(sink=2, window=4)
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    a.forward_block([9, 1])  # cache → 10
    a.commit_or_truncate(forwarded=2, accepted=1)  # drop 1
    assert a._past_len == 9
    assert a._past.layers[0].keys.shape[2] == 9
    assert a._committed[-1] == 9


def test_incremental_append_token_advances():
    a, _ = _make_inc_adapter(sink=2, window=4)
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    nt = a.append_token(5)
    assert a._past_len == 9
    assert int(nt.argmax()) == (5 + 1) % V


def test_incremental_reset_clears_past():
    a, _ = _make_inc_adapter()
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    a.reset()
    assert a._past is None and a._past_len == 0


def test_incremental_prefill_twice_reuses_num_layers():
    a, _ = _make_inc_adapter(sink=2, window=4)
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    n1 = a._num_layers_cache
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # _num_layers_cache already set
    assert a._num_layers_cache == n1 == 3


def test_incremental_commit_skips_empty_layer():
    a, _ = _make_inc_adapter(sink=2, window=4)
    a.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    a.forward_block([9, 1])
    # Defensive: a layer with keys=None must be skipped during truncation.
    a._past.layers.append(SimpleNamespace(keys=None, values=None))
    a.commit_or_truncate(forwarded=2, accepted=1)
    assert a._past_len == 9
    assert a._past.layers[0].keys.shape[2] == 9


# --------------------------------------------------------------------------- #
# End-to-end SpeculativeDecoder integration (Gap 1 + factory)
# --------------------------------------------------------------------------- #
def _greedy_reference(prompt, n):
    """Greedy restored-AR: predict (x+1)%V repeatedly from prompt[-1]."""
    out = []
    x = prompt[-1]
    for _ in range(n):
        x = (x + 1) % V
        out.append(x)
    return out


def test_spec_decode_accept_all_matches_greedy():
    adapter, _, _ = _make_adapter(sink_size=4, window_size=64)
    # Proposer that proposes the *correct* continuation → all accepted.
    def predict(committed, L):
        x = committed[-1]
        toks = []
        for _ in range(L):
            x = (x + 1) % V
            toks.append(x)
        return toks

    decoder = build_restored_speculative_decoder(
        _FakeProposer(predict), adapter, block_size=4, num_diffusion_steps=2,
    )
    prompt = [1, 2, 3]
    res = decoder.generate(prompt_ids=prompt, max_new_tokens=10)
    assert res.output_token_ids == _greedy_reference(prompt, 10)
    assert res.acceptance_rate > 0.0  # tokens were accepted


def test_spec_decode_reject_all_still_matches_greedy():
    adapter, _, _ = _make_adapter(sink_size=4, window_size=64)
    # Proposer that always proposes a token the verifier won't predict
    # (offset by 2) → accepted=0 each block; verifier emits the correction.
    def predict(committed, L):
        x = committed[-1]
        return [(x + 2) % V] * L

    decoder = build_restored_speculative_decoder(
        _FakeProposer(predict), adapter, block_size=4, num_diffusion_steps=2,
    )
    prompt = [1, 2, 3]
    res = decoder.generate(prompt_ids=prompt, max_new_tokens=6)
    # Even with 0 acceptance, the verifier's correction token each block
    # is the greedy next token → output still equals greedy restored-AR.
    assert res.output_token_ids == _greedy_reference(prompt, 6)
    assert res.total_accepted == 0
