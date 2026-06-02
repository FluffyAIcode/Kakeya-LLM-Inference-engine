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
from .quantization import QuantizationInfo, detect_quantization


def _model_weight_bytes(model) -> int:
    """Sum bytes across a flat tree of mlx parameters.

    Thin wrapper retained for backward compatibility with
    :mod:`inference_engine.backends.mlx.proposer` and existing tests.
    Equivalent to ``detect_quantization(model).total_weight_bytes``;
    new code should call :func:`detect_quantization` directly to also
    get bits / group_size / per-region byte breakdown.
    """
    return detect_quantization(model).total_weight_bytes


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
        # Parallel record of the token id at every K/V cache slot, in
        # the same physical order as ``self.cache[*].keys``. See the CPU
        # verifier for the full motivation; in short, this is required
        # by ADR 0007 §2.2 (path-selection needs token-id-level
        # comparison against the cache) and §2.9 INV-1 (parallel-
        # sequence consistency). Maintained synchronously with the
        # K/V tensors by every cache mutation method below.
        self.cached_token_sequence: List[int] = []

        self.quantization: QuantizationInfo = detect_quantization(self.model)
        self.stats = VerifierStats(weight_bytes=self.quantization.total_weight_bytes)

        # PR-E1c: precompute per-K/V-token byte cost for the
        # ``kv_live_bytes`` accessor. Mirrors the CPU verifier;
        # reads dims from the wrapped HF config so GQA / MQA via
        # ``num_key_value_heads`` is honored.
        cfg = self.model.config if hasattr(self.model, "config") else self.model
        num_layers = int(getattr(cfg, "num_hidden_layers"))
        num_kv_heads = int(
            getattr(cfg, "num_key_value_heads", None)
            or getattr(cfg, "num_attention_heads")
        )
        head_dim = int(
            getattr(cfg, "head_dim", None)
            or (cfg.hidden_size // cfg.num_attention_heads)
        )
        itemsize = torch.tensor([], dtype=self.config.dtype).element_size()
        self._bytes_per_kv_token = (
            num_layers * num_kv_heads * head_dim * itemsize * 2
        )

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
        self.cached_token_sequence = []
        self._assert_cache_invariant_1()

    def prefill(self, prompt_ids: List[int]) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        L = len(prompt_ids)
        arr = mx.array([prompt_ids], dtype=mx.int32)
        logits_mx = self.model(arr, cache=self.cache)
        mx.eval(logits_mx)
        if logits_mx.ndim != 3 or int(logits_mx.shape[0]) != 1:  # pragma: no cover - mlx_lm contract
            raise RuntimeError(
                f"prefill: model returned logits of shape {tuple(logits_mx.shape)} "
                "(expected [1, T, V])"
            )

        self._record_peak_activation(logits_mx)
        self.next_token_logits = mx_to_torch(logits_mx[0, -1])
        self.next_global_position = L
        self.cache_logical_size = self._cache_buffer_size()
        # Compute the post-trim parallel token sequence directly. The
        # MLX SinkWindowKVCache trims inside update_and_fetch on every
        # forward, so by the time we get here the per-layer K/V tensors
        # already hold the sink+window slice of ``prompt_ids``. We
        # mirror that slice on cached_token_sequence so INV-1 holds.
        self.cached_token_sequence = self._sink_window_slice(prompt_ids)
        self._record_peak_kv()
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        self._assert_cache_invariant_1()

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
            raise RuntimeError(  # pragma: no cover - mlx_lm contract
                f"forward_block: expected logits [1,{L},V], got "
                f"{tuple(logits_mx.shape)}"
            )

        self._record_peak_activation(logits_mx)
        # Persistent cache size is bounded by sink+window after the
        # SinkWindowKVCache.update_and_fetch trim. Read directly from
        # the cache rather than tracking it ourselves.
        self.cache_logical_size = self._cache_buffer_size()
        # Mirror the same trim on the parallel sequence: take the
        # current sequence concatenated with the new tokens and apply
        # the sink+window slice.
        extended = self.cached_token_sequence + list(tokens)
        self.cached_token_sequence = self._sink_window_slice(extended)
        block_logits = mx_to_torch(logits_mx[0])  # [L, V]
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        self._assert_cache_invariant_1()
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
            # Mirror the tail truncation on the parallel sequence.
            self.cached_token_sequence = self.cached_token_sequence[:-drop]
        self.cache_logical_size = self._cache_buffer_size()
        self.next_global_position += accepted
        self._record_peak_kv()
        self._assert_cache_invariant_1()

    def append_token(self, token_id: int) -> torch.Tensor:
        logits = self.forward_block([token_id])
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1].clone()
        return self.next_token_logits

    # ---------------- CacheInspector protocol (ADR 0008 PR-A3b) ---------------- #
    # Mirrors the CPU verifier's CacheInspector implementation. The session
    # argument is accepted for protocol conformance but ignored in v0.3
    # single-tenant scope (one verifier instance binds to one session at a
    # time, see ADR 0008 §2.5). PR-A3c plumbs session-scoped binding.

    def k_seq_length(self, session: object) -> int:
        """Return the K/V tensor sequence length for the bound session.

        Implements the :class:`inference_engine.session.store.CacheInspector`
        Protocol. The ``session`` argument is unused in v0.3 (single
        bound session per verifier). Returns 0 when no cache is
        allocated.
        """
        del session  # unused in v0.3 single-tenant scope
        return self._cache_buffer_size()

    def kv_live_bytes(self, session: object) -> int:
        """Return the live K/V cache size in bytes for ``session``.

        Mirrors the CPU verifier's :meth:`kv_live_bytes`; computed as
        ``k_seq_length × num_layers × num_kv_heads × head_dim ×
        itemsize × 2``. PR-E1c — feeds ``GetSessionInfo.kv_live_bytes``
        through the coordinator's slab-write-through.
        """
        del session  # unused in v0.3 single-tenant scope
        return self._cache_buffer_size() * self._bytes_per_kv_token

    # --------------------------- internals --------------------------- #

    def _cache_buffer_size(self) -> int:
        """Return the actual seq dim of the cache buffer (post-trim)."""
        if self.cache is None:
            return 0
        return cache_ops.cache_seq_length(self.cache)

    def _sink_window_slice(self, sequence: List[int]) -> List[int]:
        """Return ``sequence`` after the sink+window trim that the K/V
        cache applies.

        Mirrors ``SinkWindowKVCache.update_and_fetch``'s trim logic at
        the token-id level: if the input length exceeds the budget,
        keep the first ``sink_size`` entries and the last
        ``window_size`` entries; otherwise return unchanged.
        """
        budget = self.config.sink_size + self.config.window_size
        if len(sequence) <= budget:
            return list(sequence)
        return (
            list(sequence[: self.config.sink_size])
            + list(sequence[-self.config.window_size :])
        )

    def _assert_cache_invariant_1(self) -> None:
        """ADR 0007 §2.9 INV-1: parallel-sequence consistency.

        After every cache mutation, ``len(self.cached_token_sequence)``
        must equal the K/V tensor sequence dimension. Violation
        indicates a bug in the verifier's cache-mutation path; per ADR
        0007 §2.9 the implementation must raise — never silently
        recover, never fall back, never re-sync.
        """
        actual = len(self.cached_token_sequence)
        expected = self._cache_buffer_size()
        if actual != expected:
            raise AssertionError(
                f"INV-1 violated (parallel-sequence consistency): "
                f"cached_token_sequence has {actual} entries but K/V "
                f"cache seq dim is {expected}. This is a bug in the "
                f"verifier's cache-mutation path; per ADR 0007 §2.9 it "
                f"must surface as a critical error rather than be "
                f"silently recovered. cache_logical_size="
                f"{self.cache_logical_size}, "
                f"next_global_position={self.next_global_position}."
            )

    def live_kv_bytes(self) -> int:
        """Return the current size of the verifier's live KV cache in bytes.

        This is the *now* size, not a peak. Reads from any thread:
        ``cache_ops.total_kv_bytes`` walks the per-layer
        :class:`SinkWindowKVCache` instances and sums
        ``keys.size * keys.dtype.size`` + same for values, all of
        which are integer attributes that don't tear under a
        concurrent reader. The HTTP ``/metrics`` handler relies on
        this property to scrape KV usage during in-flight generation.

        Returns 0 when the cache has not been allocated yet (between
        ``reset()`` and the next ``prefill()``).
        """
        if self.cache is None:
            return 0
        return cache_ops.total_kv_bytes(self.cache)

    def _record_peak_kv(self) -> None:
        total = self.live_kv_bytes()
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
