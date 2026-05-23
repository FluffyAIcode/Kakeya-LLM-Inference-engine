"""MLX-backed sink+window AR verifier.

API parity with :class:`kv_cache_proposer.verifier.SinkWindowVerifier`
so the speculative decoder is unchanged when this drop-in replacement
is used. Internals:

  * Model is loaded via ``mlx_lm.load(repo_id)`` and runs on Apple's
    unified-memory GPU via Metal.
  * KV cache is a list of
    :class:`inference_engine.backends.mlx.cache.SinkWindowKVCache`
    (one per transformer layer). Each cache trims itself atomically
    inside ``update_and_fetch`` — there is no post-hoc state surgery.
    See `cache.py`'s module docstring for why direct mutation of
    ``mlx_lm.models.cache.KVCache`` was the source of the MLX-1b
    divergence-after-trim bug.
  * Returned logits are converted from ``mx.array`` to
    ``torch.Tensor`` at the API boundary so the speculative loop's
    ``torch.argmax`` / ``.item()`` work unchanged. The conversion is
    small-tensor (next-token slice or block-of-L slice), single-digit
    ms.

The module imports ``mlx.core`` and ``mlx_lm`` at top level. Non-Apple-
Silicon hosts cannot load this file — that is the intended behavior.

Failure modes (no fallback):
  * any forward producing logits of unexpected shape -> RuntimeError
  * commit/truncate beyond cache contents -> RuntimeError
  * unsupported config dtype -> ValueError
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx_lm
import torch

from kv_cache_proposer.verifier import VerifierConfig, VerifierStats

from . import cache as cache_ops
from ._torch_bridge import mx_to_torch
from .cache import SinkWindowKVCache, make_sink_window_cache
from .env import require_environment


def _model_weight_bytes(model) -> int:
    """Sum bytes across a flat tree of mlx parameters."""
    total = 0

    def _accum(obj):
        nonlocal total
        if isinstance(obj, mx.array):
            total += obj.size * obj.dtype.size
        elif isinstance(obj, dict):
            for v in obj.values():
                _accum(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _accum(v)

    _accum(model.parameters())
    return total


class MLXSinkWindowVerifier:
    """Drop-in MLX replacement for `SinkWindowVerifier`.

    Construction:
        cfg = VerifierConfig(model_id="Qwen/Qwen3-1.7B", sink_size=4,
                             window_size=64)
        verifier = MLXSinkWindowVerifier(cfg)

    Reuses ``VerifierConfig`` / ``VerifierStats`` so the speculative
    decoder doesn't know which backend is in use. The ``device`` field
    of the config is accepted but largely ignored (MLX uses its own
    device model). The ``dtype`` field is honored only as a
    diagnostic — ``mlx_lm.load`` picks the checkpoint's stored dtype.
    """

    def __init__(self, config: Optional[VerifierConfig] = None) -> None:
        require_environment()
        self.config = config or VerifierConfig()
        if self.config.sink_size < 0 or self.config.window_size <= 0:
            raise ValueError(
                "sink_size must be >= 0 and window_size must be > 0"
            )

        self.model, self.tokenizer = mlx_lm.load(self.config.model_id)
        self._mx_dtype = _map_torch_dtype_to_mx(self.config.dtype)

        self.cache: Optional[List[SinkWindowKVCache]] = None
        # Number of generated/seen tokens — same field name as the
        # PyTorch verifier, used by the speculative decoder for
        # bookkeeping. Distinct from per-layer cache.offset (also
        # the global position; redundant but mirrors the PyTorch class).
        self.cache_logical_size: int = 0
        self.next_global_position: int = 0
        self.next_token_logits: Optional[torch.Tensor] = None

        self.stats = VerifierStats(weight_bytes=_model_weight_bytes(self.model))

    # ---------------------------- public API ---------------------------- #

    def reset(self) -> None:
        self.cache = make_sink_window_cache(
            self.model,
            sink_size=self.config.sink_size,
            window_size=self.config.window_size,
        )
        self.cache_logical_size = 0
        self.next_global_position = 0
        self.next_token_logits = None

    def prefill(self, prompt_ids: List[int]) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        L = len(prompt_ids)
        arr = mx.array([prompt_ids], dtype=mx.int32)
        logits_mx = self.model(arr, cache=self.cache)
        mx.eval(logits_mx)
        if logits_mx.ndim != 3 or int(logits_mx.shape[0]) != 1:
            raise RuntimeError(
                f"prefill: model returned logits of shape {tuple(logits_mx.shape)} "
                "(expected [1, T, V])"
            )

        self._record_peak_activation(logits_mx)
        self.next_token_logits = mx_to_torch(logits_mx[0, -1])
        self.next_global_position = L
        self.cache_logical_size = self._cache_buffer_size()
        self._record_peak_kv()
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L

    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("Verifier not prefilled.")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        L = len(tokens)
        arr = mx.array([tokens], dtype=mx.int32)
        logits_mx = self.model(arr, cache=self.cache)
        mx.eval(logits_mx)
        if logits_mx.ndim != 3 or int(logits_mx.shape[0]) != 1 \
                or int(logits_mx.shape[1]) != L:
            raise RuntimeError(
                f"forward_block: expected logits [1,{L},V], got "
                f"{tuple(logits_mx.shape)}"
            )

        self._record_peak_activation(logits_mx)
        # Persistent cache size is bounded by sink+window after the
        # SinkWindowKVCache.update_and_fetch trim. Read directly from
        # the cache rather than tracking it ourselves.
        self.cache_logical_size = self._cache_buffer_size()
        block_logits = mx_to_torch(logits_mx[0])  # [L, V]
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        return block_logits

    def commit_or_truncate(self, forwarded: int, accepted: int) -> None:
        if accepted < 0 or accepted > forwarded:
            raise ValueError("accepted must satisfy 0 <= accepted <= forwarded")
        drop = forwarded - accepted
        if drop > 0:
            trims = [layer.trim(drop) for layer in self.cache]
            # All layers must trim by exactly `drop`. If any layer has fewer
            # than `drop` slots, that's a real cross-layer inconsistency
            # we want surfaced (no fallback).
            if any(t != drop for t in trims):
                raise RuntimeError(
                    f"per-layer trim mismatch (asked drop={drop}, got {trims}); "
                    "SinkWindowKVCache state diverged across layers"
                )
        self.cache_logical_size = self._cache_buffer_size()
        self.next_global_position += accepted
        self._record_peak_kv()

    def append_token(self, token_id: int) -> torch.Tensor:
        logits = self.forward_block([token_id])
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1].clone()
        return self.next_token_logits

    # --------------------------- internals --------------------------- #

    def _cache_buffer_size(self) -> int:
        """Return the actual seq dim of the cache buffer (post-trim)."""
        if self.cache is None:
            return 0
        return cache_ops.cache_seq_length(self.cache)

    def _record_peak_kv(self) -> None:
        if self.cache is None:
            return
        total = cache_ops.total_kv_bytes(self.cache)
        if total > self.stats.peak_kv_bytes:
            self.stats.peak_kv_bytes = total

    def _record_peak_activation(self, logits_mx: "mx.array") -> None:
        bytes_ = int(logits_mx.size) * int(logits_mx.dtype.size)
        if bytes_ > self.stats.peak_activation_bytes:
            self.stats.peak_activation_bytes = bytes_


def _map_torch_dtype_to_mx(dtype) -> "mx.Dtype":
    """Best-effort mapping for diagnostic purposes only."""
    import torch as _torch
    table = {
        _torch.bfloat16: mx.bfloat16,
        _torch.float16: mx.float16,
        _torch.float32: mx.float32,
    }
    if dtype not in table:
        raise ValueError(
            f"VerifierConfig.dtype={dtype} has no MLX equivalent; "
            "supported: bfloat16, float16, float32"
        )
    return table[dtype]
