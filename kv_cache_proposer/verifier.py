"""AR Verifier with sink+window KV cache.

The verifier is an autoregressive Qwen3 model. Its KV cache is bounded to
`sink_size + window_size` tokens at all times. After every model call we
*physically* slice each layer's K and V tensors to enforce that bound
(StreamingLLM-style attention sink + sliding window).

Important correctness notes:
  * `position_ids` for new tokens always use the **global** sequence position,
    so RoPE on new queries and new keys is rotated at their true distance.
    The surviving sink/window K vectors retain the RoPE rotation they had at
    their original positions — attention dot-products use the correct
    relative-position phase per surviving token.
  * `cache_position` tracks where in the *trimmed* cache the new K/V land,
    which is what the causal mask uses to enforce ``q_i attends k_{<= i}``
    inside the trimmed cache layout.
  * No fallback. If the cache layout becomes inconsistent we raise.
  * Greedy decoding (argmax) is used. With block_size+sink+window covering the
    full sequence, this is bit-equivalent to vanilla greedy AR — the
    speculative loop's correctness test relies on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass
class VerifierConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"
    sink_size: int = 4
    window_size: int = 64


@dataclass
class VerifierStats:
    forward_calls: int = 0
    tokens_consumed: int = 0
    peak_kv_bytes: int = 0
    weight_bytes: int = 0
    peak_activation_bytes: int = 0
    """Largest single-forward activation footprint observed.

    We approximate activation peak by the size of the logits buffer
    (`[B, T, V_vocab]`), which is the dominant transient tensor of a
    Qwen3 forward at long contexts. Other intermediates (Q, attn probs,
    MLP buffers) are smaller per layer and overlapping in lifetime."""


class SinkWindowVerifier:
    def __init__(self, config: Optional[VerifierConfig] = None) -> None:
        self.config = config or VerifierConfig()
        if self.config.sink_size < 0 or self.config.window_size <= 0:
            raise ValueError("sink_size must be >= 0 and window_size must be > 0")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            dtype=self.config.dtype,
        )
        self.model.to(self.config.device).eval()

        self.cache: Optional[DynamicCache] = None
        # Number of K/V slots currently in cache (after trimming).
        self.cache_logical_size: int = 0
        # Next global token position to emit (== number of tokens the verifier
        # has *seen*, counting both prompt and generated tokens).
        self.next_global_position: int = 0
        # Logits at `next_global_position` predicting the next token. Updated
        # after every forward pass.
        self.next_token_logits: Optional[torch.Tensor] = None
        # Parallel record of the token id at every K/V cache slot, in the
        # same physical order as ``self.cache.layers[*].keys``. Maintained
        # synchronously with the K/V tensors by every cache mutation
        # method below. Required by ADR 0007 §2.2 + §2.9 INV-1: the
        # path-selection algorithm (PR 7-2) needs token-id-level prefix
        # matching against the cache, and the K/V tensors don't expose
        # token ids.
        #
        # Storage: at most ``sink_size + window_size`` int entries, so
        # bounded at the same constant the K/V cache is bounded at
        # (e.g. 68 entries × 8 bytes per Python int = 544 bytes,
        # negligible vs the 7.4 MiB K/V).
        #
        # Invariant INV-1 (ADR 0007 §2.9): after every cache mutation,
        # ``len(self.cached_token_sequence)`` equals the K/V tensor
        # sequence dimension. Enforced by ``_assert_cache_invariant_1``.
        self.cached_token_sequence: List[int] = []

        self.stats = VerifierStats(
            weight_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters())
        )

    # ---------------------------- public API ---------------------------- #
    def reset(self) -> None:
        self.cache = DynamicCache(config=self.model.config)
        self.cache_logical_size = 0
        self.next_global_position = 0
        self.next_token_logits = None
        self.cached_token_sequence = []
        self._assert_cache_invariant_1()

    @torch.no_grad()
    def prefill(self, prompt_ids: List[int]) -> None:
        """Run the prompt through the verifier, then trim KV cache."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        device = self.config.device
        L = len(prompt_ids)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        cache_position = torch.arange(L, dtype=torch.long, device=device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self.cache_logical_size = L
        self.next_global_position = L
        self.next_token_logits = outputs.logits[0, -1].clone()

        # Update parallel token sequence in lockstep with the K/V cache.
        # After this prefill the cache holds K/V for all L tokens; the
        # subsequent ``_trim_cache_in_place`` will drop middle entries
        # to enforce sink+window. We mirror that exact transformation
        # on ``cached_token_sequence``.
        self.cached_token_sequence = list(prompt_ids)

        self._record_peak_activation(outputs.logits)
        self._trim_cache_in_place()
        self._record_peak_kv()
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        self._assert_cache_invariant_1()

    @torch.no_grad()
    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        """Forward `tokens` through the verifier with the trimmed cache.

        Returns a [len(tokens), V] tensor of next-token logits, where
        ``logits[i]`` is the verifier's prediction for the token *after*
        ``tokens[i]``. K/V for these tokens is appended to the cache
        (subject to subsequent trimming via :meth:`commit_or_truncate`).
        """
        if self.cache is None:
            raise RuntimeError("Verifier not prefilled.")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        device = self.config.device
        L = len(tokens)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        global_start = self.next_global_position
        position_ids = torch.arange(
            global_start, global_start + L, dtype=torch.long, device=device
        ).unsqueeze(0)
        cache_start = self.cache_logical_size
        cache_position = torch.arange(
            cache_start, cache_start + L, dtype=torch.long, device=device
        )

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        # Cache provisionally has cache_start + L slots until commit/truncate.
        self.cache_logical_size = cache_start + L
        # Mirror the provisional extension on the parallel sequence;
        # commit_or_truncate will drop the unaccepted tail in lockstep.
        self.cached_token_sequence = self.cached_token_sequence + list(tokens)
        self._record_peak_activation(outputs.logits)
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += L
        self._assert_cache_invariant_1()
        # Don't trim yet — caller decides how many tokens were accepted.
        return outputs.logits[0].clone()  # [L, V]

    def commit_or_truncate(
        self,
        forwarded: int,
        accepted: int,
    ) -> None:
        """Reconcile cache state after a verification pass.

        ``forwarded`` is the number of tokens passed to :meth:`forward_block`.
        ``accepted`` is how many of those tokens speculative decoding kept.
        The unaccepted tail (``forwarded - accepted``) is sliced out of the
        cache; ``next_global_position`` is advanced by ``accepted``. The cache
        is then trimmed to sink+window.
        """
        if accepted < 0 or accepted > forwarded:
            raise ValueError("accepted must satisfy 0 <= accepted <= forwarded")
        drop = forwarded - accepted
        if drop > 0:
            self._truncate_tail_in_place(drop)
            # Mirror the tail truncation on the parallel sequence.
            self.cached_token_sequence = self.cached_token_sequence[:-drop]
        self.cache_logical_size -= drop
        self.next_global_position += accepted
        self._trim_cache_in_place()
        self._record_peak_kv()
        self._assert_cache_invariant_1()

    @torch.no_grad()
    def append_token(self, token_id: int) -> torch.Tensor:
        """Forward a single token (e.g., correction or bonus) into the cache.

        Returns the logits predicting the token *after* `token_id`.
        """
        logits = self.forward_block([token_id])
        # commit it: this single token is accepted
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1].clone()
        return self.next_token_logits

    # ------------------------- internal helpers ------------------------- #
    def _budget(self) -> int:
        return self.config.sink_size + self.config.window_size

    def _trim_cache_in_place(self) -> None:
        if self.cache is None:
            raise RuntimeError("No cache to trim.")
        budget = self._budget()
        if self.cache_logical_size <= budget:
            return
        sink = self.config.sink_size
        keep_window = budget - sink

        for layer in self.cache.layers:
            keys: torch.Tensor = layer.keys
            values: torch.Tensor = layer.values
            if keys is None or values is None:
                continue
            if keys.shape[2] != self.cache_logical_size:
                raise RuntimeError(
                    f"Cache shape {keys.shape} inconsistent with logical "
                    f"size {self.cache_logical_size}; layout invariant violated."
                )
            sink_k = keys[:, :, :sink, :]
            sink_v = values[:, :, :sink, :]
            tail_k = keys[:, :, -keep_window:, :]
            tail_v = values[:, :, -keep_window:, :]
            # Allocate fresh contiguous tensors so the originals are GC'd —
            # otherwise CPython retains the trimmed slices' parent storage and
            # peak_kv_bytes would over-report.
            layer.keys = torch.cat([sink_k, tail_k], dim=2).contiguous()
            layer.values = torch.cat([sink_v, tail_v], dim=2).contiguous()
        # Mirror the same sink+window slice on the parallel token sequence
        # so cached_token_sequence stays in lockstep with the K/V tensors.
        self.cached_token_sequence = (
            self.cached_token_sequence[:sink]
            + self.cached_token_sequence[-keep_window:]
        )
        self.cache_logical_size = budget

    def _truncate_tail_in_place(self, drop: int) -> None:
        if drop <= 0:
            return
        if self.cache is None:
            raise RuntimeError("No cache to truncate.")
        if drop > self.cache_logical_size:
            raise RuntimeError(
                f"Cannot drop {drop} tokens from cache of size {self.cache_logical_size}"
            )
        keep = self.cache_logical_size - drop
        for layer in self.cache.layers:
            keys: torch.Tensor = layer.keys
            values: torch.Tensor = layer.values
            if keys is None or values is None:
                continue
            layer.keys = keys[:, :, :keep, :].contiguous()
            layer.values = values[:, :, :keep, :].contiguous()

    def _cache_seq_length(self) -> int:
        """Return the seq dim of the cache K/V tensors, or 0 if empty.

        Reads from the first non-empty layer; all layers share the same
        seq dim by construction (every K/V mutation in this class
        applies the same shape transformation across all layers).
        """
        if self.cache is None:
            return 0
        for layer in self.cache.layers:
            keys = getattr(layer, "keys", None)
            if keys is not None:
                return int(keys.shape[2])
        return 0

    # --------------- CacheInspector protocol (ADR 0008 PR-A3b) --------------- #
    # The CPU verifier satisfies inference_engine.session.store.CacheInspector
    # structurally so SessionStore can use it as the INV-1 source of truth.
    # The session argument is unused — the verifier holds a single cache for
    # the session it is currently bound to (single-tenant, max_concurrent=1
    # in v0.3, see ADR 0008 §2.5). PR-A3c will plumb session-scoped binding
    # so multiple sessions can each query their own slab.

    def k_seq_length(self, session: object) -> int:
        """Return the K/V tensor sequence length for the bound session.

        Implements the :class:`inference_engine.session.store.CacheInspector`
        Protocol. The ``session`` argument is accepted for protocol
        conformance but ignored in v0.3 (one verifier instance binds to
        one session at a time). Returns 0 when no cache is allocated.
        """
        del session  # unused in v0.3 single-tenant scope
        return self._cache_seq_length()

    def _assert_cache_invariant_1(self) -> None:
        """ADR 0007 §2.9 INV-1: parallel-sequence consistency.

        After every cache mutation, ``len(self.cached_token_sequence)``
        must equal the K/V tensor sequence dimension. Violation
        indicates a bug in the cache-mutation path; per ADR 0007 §2.9
        the implementation must raise — never silently recover, never
        fall back, never re-sync.
        """
        actual = len(self.cached_token_sequence)
        expected = self._cache_seq_length()
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

        This is the *now* size, not a peak. Reads cleanly from any
        thread (no locks): in CPython, walking ``self.cache.layers``
        and reading ``Tensor.numel()`` / ``element_size()`` on each
        is safe even while the worker thread is mutating the cache —
        a concurrent write produces a value somewhere between the
        two adjacent stable states, never garbage. The HTTP
        ``/metrics`` handler relies on this property.

        Returns 0 when the cache has not been allocated yet (between
        ``reset()`` and the next ``prefill()``).
        """
        if self.cache is None:
            return 0
        total = 0
        for layer in self.cache.layers:
            if layer.keys is not None:
                total += layer.keys.numel() * layer.keys.element_size()
            if layer.values is not None:
                total += layer.values.numel() * layer.values.element_size()
        return total

    def _record_peak_kv(self) -> None:
        total = self.live_kv_bytes()
        self.stats.peak_kv_bytes = max(self.stats.peak_kv_bytes, total)

    def _record_peak_activation(self, logits: torch.Tensor) -> None:
        bytes_ = int(logits.numel() * logits.element_size())
        if bytes_ > self.stats.peak_activation_bytes:
            self.stats.peak_activation_bytes = bytes_
