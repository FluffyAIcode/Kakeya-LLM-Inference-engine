"""Quantized flash attention (KIE-v1.1.y) — tiled online-softmax over int8 K/V.

The bounded-decode floor (KIE-v1.1.x) was blocked not by storage but by the
**decode-time dequant transient**: a cache that hands bf16 K/V to SDPA
materializes the full ``[N, H, S, D]`` per exact layer, so int8/packed *storage*
shrinks but the *peak* doesn't (N=34 OOM).

This module removes the transient: attention reads the **int8** (or packed) K/V
and computes the result with a **flash-style online-softmax over tiles of the
key/value sequence**, dequantizing only one tile at a time. The transient is
``O(tile)`` instead of ``O(S)``, so the int8 storage saving (≈2×) turns into
concurrency (target N→60 at 62k). Pure PyTorch (no custom CUDA kernel); the
online-softmax recurrence is numerically exact vs full SDPA up to the int8
quantization error.
"""

from __future__ import annotations

from typing import Optional


def _repeat_kv(x, n_rep: int):
    """[N, Hkv, S, D] -> [N, Hkv*n_rep, S, D] (GQA expand)."""
    if n_rep == 1:
        return x
    N, Hkv, S, D = x.shape
    return (x[:, :, None, :, :]
            .expand(N, Hkv, n_rep, S, D)
            .reshape(N, Hkv * n_rep, S, D))


def quantized_flash_attention(
    q,                      # [N, Hq, Lq, D] bf16/fp16
    k_int8, k_scale,        # [N, Hkv, S, D] int8, [N, Hkv, S, 1] dtype
    v_int8, v_scale,        # same
    *,
    scale: float,
    tile: int = 4096,
    causal_offset: Optional[int] = None,
):
    """Flash-style attention over an int8 K/V cache, tile-dequantized.

    Online softmax (running max/sum/acc) over key tiles; only one tile's bf16
    K/V (`[N, Hq, tile, D]`) is materialized at a time. Returns `[N, Hq, Lq, D]`.

    ``causal_offset`` : if set, key position ``j`` is visible to query ``i`` iff
    ``j <= causal_offset + i`` (prefill). For decode (Lq=1 attending to all S),
    leave None.
    """
    import torch

    N, Hq, Lq, D = q.shape
    Hkv = k_int8.shape[1]
    S = k_int8.shape[2]
    n_rep = Hq // Hkv
    dtype = q.dtype
    neg_inf = torch.finfo(torch.float32).min

    m = q.new_full((N, Hq, Lq, 1), neg_inf, dtype=torch.float32)
    l = q.new_zeros((N, Hq, Lq, 1), dtype=torch.float32)
    acc = q.new_zeros((N, Hq, Lq, D), dtype=torch.float32)
    qf = q.to(torch.float32)

    for t0 in range(0, S, tile):
        t1 = min(t0 + tile, S)
        kt = (k_int8[:, :, t0:t1].to(dtype) * k_scale[:, :, t0:t1])  # [N,Hkv,w,D]
        vt = (v_int8[:, :, t0:t1].to(dtype) * v_scale[:, :, t0:t1])
        kt = _repeat_kv(kt, n_rep).to(torch.float32)
        vt = _repeat_kv(vt, n_rep).to(torch.float32)
        s = torch.matmul(qf, kt.transpose(-1, -2)) * scale          # [N,Hq,Lq,w]
        if causal_offset is not None:
            w = t1 - t0
            qi = torch.arange(Lq, device=q.device).view(Lq, 1) + causal_offset
            kj = torch.arange(t0, t1, device=q.device).view(1, w)
            s = torch.where(qi >= kj, s, torch.full_like(s, neg_inf))
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        p = torch.exp(s - m_new)                                    # [N,Hq,Lq,w]
        corr = torch.exp(m - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        acc = acc * corr + torch.matmul(p, vt)
        m = m_new

    out = acc / l.clamp_min(1e-20)
    return out.to(dtype)
