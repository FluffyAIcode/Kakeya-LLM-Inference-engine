"""MLX-backed sink+window AR verifier.

API parity with `kv_cache_proposer.verifier.SinkWindowVerifier`: the
speculative decoder is unchanged when this drop-in replacement is
used. The differences are entirely internal:

  * Model is loaded via `mlx_lm.load(repo_id)` and runs on Apple's
    unified-memory GPU via Metal.
  * KV cache is the list returned by
    `mlx_lm.models.cache.make_prompt_cache(model)` — one
    `KVCache` per transformer layer.
  * Sink+window trimming is applied via direct attribute mutation on
    each `KVCache` (`keys`, `values`, `offset`); see `cache.py`.
  * Returned logits are converted from `mx.array` to `torch.Tensor` at
    the API boundary so the speculative loop's `torch.argmax` /
    `.item()` work unchanged. The conversion is small-tensor only
    (next-token slice or block-of-L slice), single-digit ms.

The module imports `mlx.core` and `mlx_lm` at top level. On non-Apple-
Silicon hosts, this import fails — that is the intended behavior. The
backend's package `__init__` guards against accidental imports by
exposing only `env` at the top level; callers must opt in explicitly.

Failure modes (no fallback):
  * any forward producing logits of unexpected shape -> RuntimeError
  * any KV cache layer with K/V seq lengths disagreeing -> RuntimeError
  * commit/truncate beyond cache contents -> RuntimeError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx_lm
import torch

from kv_cache_proposer.verifier import VerifierConfig, VerifierStats

from . import cache as cache_ops
from ._torch_bridge import mx_to_torch
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

    The class deliberately reuses the PyTorch ``VerifierConfig`` /
    ``VerifierStats`` dataclasses so the speculative decoder doesn't
    need to know which backend is in use.

    The ``device`` and ``dtype`` fields of the config are accepted but
    largely ignored: MLX uses its own device + dtype model. We honor
    ``dtype`` when it can be cast to an `mx.Dtype` (bfloat16 / float16
    / float32) and emit a clear error otherwise.
    """

    def __init__(self, config: Optional[VerifierConfig] = None) -> None:
        require_environment()  # raises if MLX/Metal not usable
        self.config = config or VerifierConfig()
        if self.config.sink_size < 0 or self.config.window_size <= 0:
            raise ValueError(
                "sink_size must be >= 0 and window_size must be > 0"
            )

        self.model, self.tokenizer = mlx_lm.load(self.config.model_id)
        # Map config.dtype to an mx.Dtype for any caller that wants to
        # know what the model loaded as (we do NOT cast — mlx_lm.load
        # picks the checkpoint's stored dtype).
        self._mx_dtype = _map_torch_dtype_to_mx(self.config.dtype)

        self.cache: Optional[List] = None
        self.cache_logical_size: int = 0
        self.next_global_position: int = 0
        self.next_token_logits: Optional[torch.Tensor] = None

        self.stats = VerifierStats(weight_bytes=_model_weight_bytes(self.model))

    # ---------------------------- public API ---------------------------- #

    def reset(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache
        self.cache = make_prompt_cache(self.model)
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
        # Force evaluation so the cache writes complete before we trim.
        mx.eval(logits_mx)
        if logits_mx.ndim != 3 or int(logits_mx.shape[0]) != 1:
            raise RuntimeError(
                f"prefill: model returned logits of shape {tuple(logits_mx.shape)} "
                "(expected [1, T, V])"
            )

        self._record_peak_activation(logits_mx)

        # Convert only the last position's logits (predicting the first
        # generated token) — that's all the speculative loop needs.
        self.next_token_logits = mx_to_torch(logits_mx[0, -1])
        self.cache_logical_size = L
        self.next_global_position = L

        self._trim_cache_in_place()
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
        self.cache_logical_size += L

        # Convert all L positions' logits (the speculative loop reads
        # position-by-position; converting once is cheaper than L
        # individual cross-backend trips).
        block_logits = mx_to_torch(logits_mx[0])  # [L, V]
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        return block_logits

    def commit_or_truncate(self, forwarded: int, accepted: int) -> None:
        if accepted < 0 or accepted > forwarded:
            raise ValueError("accepted must satisfy 0 <= accepted <= forwarded")
        drop = forwarded - accepted
        if drop > 0:
            cache_ops.truncate_caches_tail(
                self.cache,
                drop=drop,
                # The new offset is the global position AFTER accepting
                # `accepted` of the L forwarded tokens.
                new_offset=self.next_global_position + accepted,
            )
        self.cache_logical_size -= drop
        self.next_global_position += accepted
        self._trim_cache_in_place()
        self._record_peak_kv()

    def append_token(self, token_id: int) -> torch.Tensor:
        logits = self.forward_block([token_id])
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1].clone()
        return self.next_token_logits

    # --------------------------- internals --------------------------- #

    def _budget(self) -> int:
        return self.config.sink_size + self.config.window_size

    def _trim_cache_in_place(self) -> None:
        if self.cache is None:
            raise RuntimeError("No cache to trim.")
        if self.cache_logical_size <= self._budget():
            return
        cache_ops.trim_caches_to_sink_window(
            self.cache,
            sink_size=self.config.sink_size,
            window_size=self.config.window_size,
            keep_offset=self.next_global_position,
        )
        self.cache_logical_size = self._budget()

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
    """Best-effort mapping for diagnostic purposes only.

    We don't actually cast the model's parameters — mlx_lm.load uses
    the checkpoint's stored dtype. We only record what the caller
    *intended* so mismatches are obvious if a user asks for fp32 and
    sees bf16 acting.
    """
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
