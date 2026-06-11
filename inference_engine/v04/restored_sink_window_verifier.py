"""Gap 1 — incremental, stateful verifier adapter for K/V Restoration.

This module bridges the *validated* (but full-forward / eval-only)
:class:`inference_engine.v04.cross_model_dlm_verifier.CrossModelDLMRestoredVerifier`
to the **stateful, incremental** verifier contract that the speculative
decoder (:class:`kv_cache_proposer.speculative.SpeculativeDecoder`) and the
gRPC session coordinators expect — i.e. the public surface of
:class:`kv_cache_proposer.verifier.SinkWindowVerifier`:

  * ``prefill(prompt_ids)``
  * ``forward_block(tokens) -> [L, V]``
  * ``commit_or_truncate(forwarded, accepted)``
  * ``append_token(token_id) -> next_token_logits``
  * ``next_token_logits`` / ``next_global_position`` / ``cached_token_sequence``
  * ``cache_logical_size`` / ``cache``
  * ``k_seq_length(session)`` / ``kv_live_bytes(session)`` / ``live_kv_bytes()``
  * ``stats`` (:class:`kv_cache_proposer.verifier.VerifierStats`)
  * ``model`` (the verifier ``nn.Module``, for KV-dim resolution)

Once an instance of this adapter is constructed, it is a drop-in
replacement for ``SinkWindowVerifier`` everywhere those callers use it —
that is *both* Gap 1 (the speculative accept/reject loop) and Gap 2 (the
server: ``SessionStore`` / ``AppendTokensCoordinator`` /
``GenerationCoordinator`` only depend on this contract).

Beta semantics (honest)
-----------------------

Each ``prefill`` / ``forward_block`` / ``append_token`` re-runs the
restored full-forward over the committed prefix (+ the block being
verified). This is **bit-equivalent to the validated gate forward** — it
*is* that forward — and realizes the headline Kakeya property: the
verifier holds only a sink+window resident cache (``cache_logical_size``
is bounded by ``sink+window``), and the evicted-position K/V are
reconstructed each step from the cache-free drafter (ADR 0008 §11.3: the
proposer is a constant-memory K/V reconstruction source) plus the S5
exact full-attention layers.

The compute is O(T)/step (O(T^2) per generation), same as the eval
harness. The per-step O(1) persistent-cache optimization (reusing the
verifier's resident sink+window K/V across steps and amortizing the
drafter forward with the proposer's) is the K2.A.2 follow-up — it does
not change *outputs*, only speed. Keeping this adapter a thin,
provably-equivalent wrapper is deliberate: it lets the recall gate that
passed on the full-forward path carry over to the served path unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from kv_cache_proposer.verifier import VerifierStats

from inference_engine.v04.cross_model_dlm_verifier import (
    CrossModelDLMRestoredVerifier,
    get_verifier_decoder,
    resolve_text_config,
)


class CrossModelRestoredSinkWindowVerifier:
    """Stateful sink+window verifier backed by f_θ + S5 K/V Restoration.

    Wraps a constructed :class:`CrossModelDLMRestoredVerifier` and exposes
    the :class:`~kv_cache_proposer.verifier.SinkWindowVerifier` public API.
    """

    def __init__(
        self,
        restored: CrossModelDLMRestoredVerifier,
        *,
        apply_rotary_pos_emb: Callable,
        eager_attention_forward: Callable,
        all_attention_functions: Optional[Any] = None,
        device: str = "cpu",
        incremental: bool = False,
    ) -> None:
        self._restored = restored
        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self._eager_attention_forward = eager_attention_forward
        self._all_attention_functions = all_attention_functions
        self._device = torch.device(device)
        # Incremental decode mode (Gap-A throughput optimization): capture the
        # restored K/V into a persistent KV cache at prefill, then decode the
        # new tokens with the verifier's NATIVE incremental forward (O(L)/block)
        # instead of re-running the O(T) restored forward each step. Recall is
        # carried by the full-attention (S5) layers whose captured K/V are the
        # verifier's own at every position (== native AR for those layers).
        self._incremental = bool(incremental)
        self._past = None            # transformers Cache holding restored K/V
        self._past_len = 0           # number of positions in the cache
        self._num_layers_cache = None  # resolved lazily (incremental path only)
        # Fused-engine (component A): optionally capture the verifier's aux-layer
        # hidden states DURING the incremental verify forward, so the DFlash
        # drafter's context can be extended incrementally instead of via a
        # separate O(C) clean-aux forward each block. Gated off by default so
        # the plain Gap-A decode path pays no overhead.
        drafter_cfg = getattr(getattr(restored, "drafter", None), "cfg", None)
        self._aux_layer_ids = tuple(getattr(drafter_cfg, "aux_layer_ids", ()) or ())
        self._capture_aux = False
        self._last_aux = None        # list[Tensor [L, hidden]] from the last verify

        self.sink_size = restored.sink_size
        self.window_size = restored.window_size

        # No persistent DynamicCache: the bounded sink+window resident K/V
        # are conceptual here (re-derived each forward). ``cache is None``
        # makes SpeculativeDecoder._kv_bytes return 0; the bounded-KV story
        # is carried by stats.peak_kv_bytes / kv_live_bytes instead.
        self.cache = None
        self.cache_logical_size: int = 0
        self.next_global_position: int = 0
        self.next_token_logits: Optional[torch.Tensor] = None
        self.cached_token_sequence: List[int] = []

        # Full committed prefix (prompt + accepted/correction tokens). This
        # is what drives restoration; it is NOT bounded (it is the logical
        # sequence), while ``cached_token_sequence`` is the bounded resident
        # mirror used by the CacheInspector accessors.
        self._committed: List[int] = []
        # Tokens passed to the most recent forward_block, pending a
        # commit_or_truncate decision.
        self._pending: List[int] = []

        self.stats = VerifierStats(weight_bytes=self._compute_weight_bytes())
        self._bytes_per_kv_token = self._compute_bytes_per_kv_token()

    # ------------------------------------------------------------------ #
    # Introspection used by the server (scripts/start_grpc_runtime_server)
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        """The verifier ``nn.Module`` (exposes ``.config`` for KV dims)."""
        return self._restored.verifier_model

    # ------------------------------------------------------------------ #
    # Construction-time accounting
    # ------------------------------------------------------------------ #
    def _compute_weight_bytes(self) -> int:
        total = 0
        for module in (
            getattr(self._restored, "verifier_model", None),
            getattr(self._restored, "drafter", None),
            getattr(self._restored, "f_theta", None),
        ):
            params = getattr(module, "parameters", None)
            if params is None:
                continue
            for p in params():
                total += p.numel() * p.element_size()
        return total

    def _compute_bytes_per_kv_token(self) -> int:
        cfg = resolve_text_config(self._restored.verifier_model.config)
        num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        num_kv_heads = int(
            getattr(cfg, "num_key_value_heads", None)
            or getattr(cfg, "num_attention_heads", 0)
            or 0
        )
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            hidden = getattr(cfg, "hidden_size", 0) or 0
            num_q = getattr(cfg, "num_attention_heads", 0) or 0
            head_dim = (hidden // num_q) if num_q else 0
        head_dim = int(head_dim)
        # itemsize from the verifier's own parameters (fp32 on CPU / bf16 GPU)
        itemsize = 4
        for p in self._restored.verifier_model.parameters():
            itemsize = p.element_size()
            break
        # ``× 2`` = K + V
        return num_layers * num_kv_heads * head_dim * itemsize * 2

    # ------------------------------------------------------------------ #
    # Core: run the restored forward over a sequence → per-position logits
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _restored_logits(self, seq_ids: List[int]) -> torch.Tensor:
        """Return ``[T, V]`` logits for ``seq_ids`` via the restored forward."""
        input_ids = torch.tensor(
            [seq_ids], dtype=torch.long, device=self._device
        )
        out = self._restored.forward(
            input_ids,
            apply_rotary_pos_emb=self._apply_rotary_pos_emb,
            eager_attention_forward=self._eager_attention_forward,
            all_attention_functions=self._all_attention_functions,
        )
        logits = out.logits if hasattr(out, "logits") else out
        return logits[0]  # [T, V]

    # ------------------------------------------------------------------ #
    # SinkWindowVerifier public API
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._committed = []
        self._pending = []
        self.cached_token_sequence = []
        self.cache_logical_size = 0
        self.next_global_position = 0
        self.next_token_logits = None
        self._past = None
        self._past_len = 0

    # ------------------------------------------------------------------ #
    # Incremental-decode helpers (Gap-A throughput path)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_restored_cache(self, prompt_ids):
        """Run the restored forward over the prompt ONCE, capturing the
        per-layer post-norm/RoPE/injection K/V into a transformers
        ``DynamicCache``. Returns (cache, last_logits)."""
        from transformers.cache_utils import DynamicCache
        if self._num_layers_cache is None:
            self._num_layers_cache = len(
                get_verifier_decoder(self._restored.verifier_model).layers)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        capture: list = [None] * self._num_layers_cache
        out = self._restored.forward(
            input_ids,
            apply_rotary_pos_emb=self._apply_rotary_pos_emb,
            eager_attention_forward=self._eager_attention_forward,
            all_attention_functions=self._all_attention_functions,
            capture_kv=capture,
        )
        logits = (out.logits if hasattr(out, "logits") else out)[0]
        if any(c is None for c in capture):
            raise RuntimeError(
                "Incremental prefill requires an evicted-position restored "
                "forward (prompt must exceed sink+window); some layers were "
                "not captured. Use a longer prompt or incremental=False."
            )
        cache = DynamicCache()
        for li, (k, v) in enumerate(capture):
            cache.update(k, v, li)
        return cache, logits

    @torch.no_grad()
    def _native_forward(self, tokens):
        """Native incremental verifier forward over ``tokens`` against the
        persistent restored cache. Appends tokens' K/V to the cache.
        Returns ``[len(tokens), V]`` logits."""
        L = len(tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=self._device)
        pos = torch.arange(self._past_len, self._past_len + L, device=self._device)
        want_aux = self._capture_aux and bool(self._aux_layer_ids)
        out = self._restored.verifier_model(
            input_ids=ids,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            past_key_values=self._past,
            use_cache=True,
            output_hidden_states=want_aux,
        )
        self._past = out.past_key_values
        if want_aux:
            hs = out.hidden_states  # tuple; hs[a] = [B, L, hidden]
            self._last_aux = [hs[a][0].detach() for a in self._aux_layer_ids]
        return out.logits[0]

    @torch.no_grad()
    def prefill(self, prompt_ids: List[int]) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        self.reset()
        self._committed = list(prompt_ids)
        if self._incremental:
            self._past, logits = self._build_restored_cache(self._committed)
            self._past_len = len(self._committed)
        else:
            logits = self._restored_logits(self._committed)  # [L, V]
        self.next_token_logits = logits[-1].clone()
        self._sync_bounded_state()
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += len(prompt_ids)
        self._record_peak_activation(logits)
        self._record_peak_kv()

    @torch.no_grad()
    def forward_block(self, tokens: List[int]) -> torch.Tensor:
        if not self._committed:
            raise RuntimeError("Verifier not prefilled.")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        self._pending = list(tokens)
        if self._incremental:
            block = self._native_forward(self._pending).clone()  # [L, V]
        else:
            seq = self._committed + self._pending
            logits = self._restored_logits(seq)  # [len(seq), V]
            start = len(self._committed)
            block = logits[start : start + len(tokens)].clone()  # [L, V]
        # Provisional resident size mirrors SinkWindowVerifier (un-trimmed
        # until commit_or_truncate); _sync_bounded_state re-bounds on commit.
        self.cache_logical_size = len(self._committed) + len(tokens)
        self.stats.forward_calls += 1
        self.stats.tokens_consumed += len(tokens)
        self._record_peak_activation(block)
        return block

    def commit_or_truncate(self, forwarded: int, accepted: int) -> None:
        if accepted < 0 or accepted > forwarded:
            raise ValueError("accepted must satisfy 0 <= accepted <= forwarded")
        if self._incremental and self._past is not None:
            # forward_block appended `forwarded` tokens' K/V to the cache;
            # drop the rejected tail so the cache reflects only committed.
            drop = forwarded - accepted
            if drop > 0:
                keep = self._past_len + forwarded - drop  # == _past_len + accepted
                for layer in self._past.layers:
                    if getattr(layer, "keys", None) is not None:
                        layer.keys = layer.keys[:, :, :keep, :].contiguous()
                        layer.values = layer.values[:, :, :keep, :].contiguous()
            self._past_len += accepted
        if accepted:
            self._committed.extend(self._pending[:accepted])
        self._pending = []
        self._sync_bounded_state()
        self._record_peak_kv()

    @torch.no_grad()
    def append_token(self, token_id: int) -> torch.Tensor:
        logits = self.forward_block([token_id])
        self.commit_or_truncate(forwarded=1, accepted=1)
        self.next_token_logits = logits[-1].clone()
        return self.next_token_logits

    # ------------------------------------------------------------------ #
    # CacheInspector protocol (used by SessionStore / coordinators)
    # ------------------------------------------------------------------ #
    def k_seq_length(self, session: object) -> int:
        del session  # single-tenant: one verifier per bound session
        return len(self.cached_token_sequence)

    def kv_live_bytes(self, session: object) -> int:
        del session
        return len(self.cached_token_sequence) * self._bytes_per_kv_token

    def live_kv_bytes(self) -> int:
        return len(self.cached_token_sequence) * self._bytes_per_kv_token

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _budget(self) -> int:
        return self.sink_size + self.window_size

    def _sync_bounded_state(self) -> None:
        """Recompute the bounded sink+window resident mirror + counters."""
        budget = self._budget()
        seq = self._committed
        if len(seq) <= budget:
            self.cached_token_sequence = list(seq)
        else:
            keep_window = budget - self.sink_size
            self.cached_token_sequence = (
                seq[: self.sink_size] + seq[-keep_window:]
                if keep_window > 0
                else seq[: self.sink_size]
            )
        self.cache_logical_size = len(self.cached_token_sequence)
        self.next_global_position = len(self._committed)

    def _record_peak_activation(self, logits: torch.Tensor) -> None:
        n = int(logits.numel() * logits.element_size())
        if n > self.stats.peak_activation_bytes:
            self.stats.peak_activation_bytes = n

    def _record_peak_kv(self) -> None:
        self.stats.peak_kv_bytes = max(
            self.stats.peak_kv_bytes, self.live_kv_bytes()
        )
