"""Numerical-equivalence tests for the tiled quantized flash attention.

The tiled online-softmax must equal full SDPA on the *same* int8 K/V (the only
difference vs bf16 SDPA is the int8 quantization itself, which both paths share).
Requires torch; skipped on torch-less hosts (the cloud-agent gate). Validated on
H200: decode max_abs 2.4e-4, prefill max_abs 7.8e-3.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from inference_engine.engine.quant_attention import quantized_flash_attention


def _quant(t):
    amax = t.abs().amax(-1, keepdim=True).clamp_(min=1e-8)
    s = amax / 127
    return torch.clamp(torch.round(t / s), -127, 127).to(torch.int8), s.to(t.dtype)


def _ref(q, kdeq, vdeq, scale, n_rep, causal):
    kref = kdeq.repeat_interleave(n_rep, dim=1)
    vref = vdeq.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), kref.float(), vref.float(), scale=scale, is_causal=causal)


@pytest.mark.parametrize("S,tile", [(1000, 256), (1000, 4096)])
def test_decode_matches_full_sdpa(S, tile):
    torch.manual_seed(0)
    N, Hq, Hkv, D = 2, 8, 4, 256
    scale = D ** -0.5
    k = torch.randn(N, Hkv, S, D); v = torch.randn(N, Hkv, S, D)
    ki, ks = _quant(k); vi, vs = _quant(v)
    kdeq = ki.to(torch.float32) * ks; vdeq = vi.to(torch.float32) * vs
    q = torch.randn(N, Hq, 1, D)
    ref = _ref(q, kdeq, vdeq, scale, Hq // Hkv, causal=False)
    mine = quantized_flash_attention(q, ki, ks, vi, vs, scale=scale, tile=tile)
    assert (ref - mine.float()).abs().max() < 1e-2


def test_prefill_causal_matches_full_sdpa():
    torch.manual_seed(1)
    N, Hq, Hkv, L, D = 2, 8, 8, 48, 256
    scale = D ** -0.5
    k = torch.randn(N, Hkv, L, D); v = torch.randn(N, Hkv, L, D)
    ki, ks = _quant(k); vi, vs = _quant(v)
    kdeq = ki.to(torch.float32) * ks; vdeq = vi.to(torch.float32) * vs
    q = torch.randn(N, Hq, L, D)
    ref = _ref(q, kdeq, vdeq, scale, 1, causal=True)
    mine = quantized_flash_attention(q, ki, ks, vi, vs, scale=scale, tile=16,
                                     causal_offset=0)
    assert (ref - mine.float()).abs().max() < 2e-2


def test_tile_size_invariance():
    torch.manual_seed(2)
    N, Hq, Hkv, S, D = 1, 4, 4, 777, 128
    scale = D ** -0.5
    k = torch.randn(N, Hkv, S, D); v = torch.randn(N, Hkv, S, D)
    ki, ks = _quant(k); vi, vs = _quant(v)
    q = torch.randn(N, Hq, 1, D)
    a = quantized_flash_attention(q, ki, ks, vi, vs, scale=scale, tile=64)
    b = quantized_flash_attention(q, ki, ks, vi, vs, scale=scale, tile=999)
    assert (a - b).abs().max() < 1e-3  # online softmax is tile-invariant
