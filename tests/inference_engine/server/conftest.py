"""Shared test doubles + fixtures for the HTTP server tests.

These are **real concrete classes** that satisfy the
:class:`~inference_engine.server.tokenizer.Tokenizer` and
:class:`~inference_engine.server.engine.Engine` protocols
structurally; they are not ``unittest.mock`` objects, and they do not
patch or wrap any production class. The "deterministic" qualifier
means their outputs are computed from constructor arguments rather
than from a real model, which is what makes route-level tests fast
and reproducible without HF cache.

The same doubles are used by:

  * tests/inference_engine/server/test_app_routes.py
  * tests/inference_engine/server/test_app_streaming.py
  * tests/inference_engine/server/test_streaming.py

Tests of the *real* :class:`SpeculativeEngine` adapter live in
test_engine.py and use the codebase's existing test verifier /
proposer fakes from ``tests/conftest.py`` (also real concrete
classes — see kv_cache_proposer.speculative tests for precedent).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import pytest

from inference_engine.server.engine import EngineResult


class DeterministicTokenizer:
    """Tiny deterministic tokenizer that maps words to integer ids.

    Vocabulary: each unique word in any input becomes a fresh id.
    Two reserved sentinel tokens are predefined so chat-template and
    EOS resolution have something to work with:

        id 0  -> ``<|im_end|>``     (also reported as eos_token_id)
        id 1  -> ``<|unk|>``        (reported as unk_token_id)

    ``apply_chat_template`` is implemented with a minimal but
    deterministic format::

        ROLE: <role>
        CONTENT: <content>
        ...

    flattened to whitespace-separated words and mapped through the
    vocabulary. ``add_generation_prompt=True`` appends the literal
    string ``"ASSISTANT:"``. This is sufficient for route-level tests
    to exercise full request -> tokenize -> generate -> decode loops
    without depending on transformers.
    """

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {"<|im_end|>": 0, "<|unk|>": 1}
        self._id_to_token: dict[int, str] = {0: "<|im_end|>", 1: "<|unk|>"}
        self.eos_token_id: Optional[int] = 0
        self.unk_token_id: Optional[int] = 1

    def _intern(self, word: str) -> int:
        if word not in self._token_to_id:
            new_id = len(self._token_to_id)
            self._token_to_id[word] = new_id
            self._id_to_token[new_id] = word
        return self._token_to_id[word]

    def apply_chat_template(
        self,
        messages: List[dict],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
        enable_thinking: bool = False,
    ) -> Any:
        if not tokenize or return_dict:
            raise ValueError(
                "DeterministicTokenizer only supports tokenize=True, return_dict=False"
            )
        words: List[str] = []
        for msg in messages:
            words.append(msg["role"].upper() + ":")
            words.extend(msg["content"].split())
        if add_generation_prompt:
            words.append("ASSISTANT:")
        return [self._intern(w) for w in words]

    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = False) -> str:
        out: List[str] = []
        for tid in token_ids:
            tok = self._id_to_token.get(int(tid), "<|unk|>")
            if skip_special_tokens and tok in {"<|im_end|>", "<|unk|>"}:
                continue
            out.append(tok)
        return " ".join(out)

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        return self._token_to_id.get(token)


class DeterministicEngine:
    """Engine test double that emits a fixed token sequence.

    Implements the :class:`~inference_engine.server.engine.Engine`
    protocol structurally without subclassing it. The ``generate``
    method walks a pre-baked token sequence, invoking ``on_token`` per
    committed token and respecting both ``max_new_tokens`` and the
    EOS list. The engine therefore exercises every cancellation and
    termination branch in the streaming layer without ever loading a
    real model.

    Special token ids:
      * ``0`` is treated as ``<|im_end|>`` by the paired
        DeterministicTokenizer; if it appears in ``fixed_tokens`` and
        ``0 in eos_token_ids`` (the default), generation stops at it.
    """

    def __init__(
        self,
        fixed_tokens: List[int],
        tokenizer: DeterministicTokenizer,
        model_id_label: str = "kakeya-test",
        per_token_delay_s: float = 0.0,
    ) -> None:
        if not fixed_tokens:
            raise ValueError("fixed_tokens must be non-empty")
        if not model_id_label.strip():
            raise ValueError("model_id_label must be non-empty")
        if per_token_delay_s < 0:
            raise ValueError("per_token_delay_s must be >= 0")
        self._fixed_tokens = list(fixed_tokens)
        self._tokenizer = tokenizer
        self._model_id_label = model_id_label
        self._per_token_delay_s = per_token_delay_s

    @property
    def tokenizer(self) -> DeterministicTokenizer:
        return self._tokenizer

    @property
    def model_id_label(self) -> str:
        return self._model_id_label

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
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if not eos_token_ids:
            raise ValueError("eos_token_ids must be non-empty")
        eos_set = set(int(i) for i in eos_token_ids)
        emitted: List[int] = []
        stopped_on_eos = False
        for tok in self._fixed_tokens:
            if len(emitted) >= max_new_tokens:
                break
            if self._per_token_delay_s > 0:  # pragma: no cover - timing aid
                import time
                time.sleep(self._per_token_delay_s)
            emitted.append(int(tok))
            if on_token is not None and on_token(int(tok)):
                break
            if int(tok) in eos_set:
                stopped_on_eos = True
                break
        return EngineResult(
            output_token_ids=emitted,
            acceptance_rate=1.0,
            proposer_forward_calls=len(emitted),
            verifier_forward_calls=len(emitted),
            stopped_on_eos=stopped_on_eos,
        )


@pytest.fixture
def tokenizer() -> DeterministicTokenizer:
    return DeterministicTokenizer()


@pytest.fixture
def short_engine(tokenizer: DeterministicTokenizer) -> DeterministicEngine:
    """Engine that emits 3 tokens then EOS."""
    # Pre-intern the words we want the tokens to decode to.
    hello = tokenizer._intern("hello")
    world = tokenizer._intern("world")
    bang = tokenizer._intern("!")
    eos = tokenizer.eos_token_id
    assert eos is not None
    return DeterministicEngine(
        fixed_tokens=[hello, world, bang, eos],
        tokenizer=tokenizer,
        model_id_label="kakeya-test-short",
    )


@pytest.fixture
def long_engine(tokenizer: DeterministicTokenizer) -> DeterministicEngine:
    """Engine that emits 50 tokens (no EOS in the sequence) — used to
    exercise the ``max_tokens`` truncation path and disconnect-mid-
    stream paths."""
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    return DeterministicEngine(
        fixed_tokens=ids,
        tokenizer=tokenizer,
        model_id_label="kakeya-test-long",
    )
