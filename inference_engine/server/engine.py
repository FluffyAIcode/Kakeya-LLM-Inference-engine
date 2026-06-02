"""Engine protocol + concrete :class:`SpeculativeEngine` adapter.

The HTTP routes consume the :class:`Engine` protocol — never a
SpeculativeDecoder directly. This separation gives us three things:

  1. Tests can plug in a deterministic test engine that emits a fixed
     token sequence, so route-level tests verify HTTP behaviour
     without ever loading a real model.
  2. Future engine variants (continuous-batching engine in E4, async
     pipeline engine in E5) can subclass / replace this one
     transparently to the route layer.
  3. The protocol surface is *narrow* (3 methods) and *typed*; it is
     the documented contract between "how generation works" and "how
     HTTP works". Changes to one side cannot accidentally couple to
     the other.

The protocol is :class:`Engine`. The single concrete production
implementation in this commit is :class:`SpeculativeEngine`, which
adapts ``kv_cache_proposer.SpeculativeDecoder`` to the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, runtime_checkable

from kv_cache_proposer.speculative import SpeculativeDecoder

from .tokenizer import Tokenizer


@dataclass(frozen=True)
class EngineResult:
    """Aggregate generation result returned by :meth:`Engine.generate`.

    Mirrors the fields of ``SpeculativeDecoder``'s result that the
    HTTP layer needs to populate OpenAI usage and finish_reason
    fields. Frozen so it can be passed safely across worker threads
    without ownership confusion.
    """

    output_token_ids: List[int]
    acceptance_rate: float
    proposer_forward_calls: int
    verifier_forward_calls: int
    stopped_on_eos: bool


@runtime_checkable
class Engine(Protocol):
    """Generation contract consumed by HTTP routes.

    Implementations:
      * :class:`SpeculativeEngine` — production, wraps a real
        ``SpeculativeDecoder`` over real verifier + proposer.
      * Test doubles in ``tests/inference_engine/server/`` that
        implement this protocol with deterministic outputs (they are
        regular concrete classes, not ``unittest.mock`` objects).

    Methods:
        generate
            Run generation to completion. ``on_token`` is called once
            per *committed* token from the worker thread; if it
            returns ``True``, generation stops at that token boundary.
            The callback is the only way streaming routes inject
            cancellation signals (e.g. client disconnect).
        kv_state
            Return the engine's current verifier KV-cache size in
            bytes, or 0 if the engine has no real KV cache (test
            doubles). Read on every ``/metrics`` scrape to populate
            the ``scheduler_kv_live_bytes`` gauge so the ADR 0006
            §2.3 long-session memory-stability claim is verifiable
            in production.
    """

    @property
    def tokenizer(self) -> Tokenizer:
        ...  # pragma: no cover - Protocol body, never executed

    @property
    def model_id_label(self) -> str:
        ...  # pragma: no cover - Protocol body, never executed

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: List[int],
        on_token: Optional[Callable[[int], bool]] = None,
    ) -> EngineResult:
        ...  # pragma: no cover - Protocol body, never executed

    def kv_state(self) -> int:
        ...  # pragma: no cover - Protocol body, never executed


class SpeculativeEngine:
    """Concrete :class:`Engine` backed by a real SpeculativeDecoder.

    Construction is intentionally explicit — callers must hand in a
    fully-constructed decoder, the tokenizer it shares with its
    verifier, and the user-facing model id label that the server
    reports via ``/v1/models`` and embeds in completion responses.
    The label is decoupled from the decoder's internal verifier id so
    operators can present (e.g.) ``"kakeya-v1"`` to clients while the
    underlying weights are ``Qwen/Qwen3-1.7B``.

    No fallback paths: if the tokenizer reports no EOS at all (the
    canonical EOS *and* ``<|im_end|>`` are both missing), construction
    raises ``ValueError`` immediately rather than letting the engine
    silently generate without termination conditions.
    """

    def __init__(
        self,
        decoder: SpeculativeDecoder,
        tokenizer: Tokenizer,
        model_id_label: str,
    ) -> None:
        if not model_id_label.strip():
            raise ValueError("model_id_label must be a non-empty string")
        # Defensive EOS check — the decoder will accept generation
        # without EOS but the result is always cut to max_tokens, which
        # is rarely what the user wants. Surface this at construction.
        from .tokenizer import resolve_eos_ids
        if not resolve_eos_ids(tokenizer):
            raise ValueError(
                "tokenizer has no EOS token id and no <|im_end|> sentinel; "
                "the engine cannot determine when to stop"
            )
        self._decoder = decoder
        self._tokenizer = tokenizer
        self._model_id_label = model_id_label

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def model_id_label(self) -> str:
        return self._model_id_label

    @property
    def decoder(self) -> SpeculativeDecoder:
        """Underlying decoder — exposed for diagnostic / metric use only.

        Routes should never reach in here; the protocol surface is
        intentionally narrower. This getter exists for callers that
        need the decoder's stats (acceptance rate distribution over
        time, peak KV bytes) outside the per-request generate path.
        """
        return self._decoder

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: List[int],
        on_token: Optional[Callable[[int], bool]] = None,
    ) -> EngineResult:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )
        if not eos_token_ids:
            raise ValueError("eos_token_ids must be non-empty")

        result = self._decoder.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            on_token=on_token,
        )
        eos_set = set(eos_token_ids)
        stopped_on_eos = bool(
            result.output_token_ids
            and result.output_token_ids[-1] in eos_set
        )
        return EngineResult(
            output_token_ids=list(result.output_token_ids),
            acceptance_rate=float(result.acceptance_rate),
            proposer_forward_calls=int(result.proposer_forward_calls),
            verifier_forward_calls=int(result.verifier_forward_calls),
            stopped_on_eos=stopped_on_eos,
        )

    def kv_state(self) -> int:
        """Live KV cache bytes from the underlying verifier.

        Reads ``self._decoder.verifier.live_kv_bytes()`` if the
        verifier exposes that method (both the CPU and MLX
        verifiers in this repository do). Returns 0 if the verifier
        is older / a stub that does not. Called from the
        ``/metrics`` handler on every scrape and must be safe to
        call concurrently with the worker thread that is mutating
        the verifier's cache (see verifier docstrings for the
        thread-safety argument).
        """
        live = getattr(self._decoder.verifier, "live_kv_bytes", None)
        if live is None:
            return 0
        return int(live())
