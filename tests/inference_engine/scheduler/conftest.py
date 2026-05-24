"""Shared fixtures for scheduler tests.

Defines local copies of the deterministic test doubles
(``DeterministicTokenizer``, ``DeterministicEngine``) so this branch
can be tested independently of the E2 server branch. When both land,
a follow-up commit consolidates them into a single shared location.

These are real concrete classes — not ``unittest.mock`` objects.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import pytest
import torch

from inference_engine.memory.pool import SlabPool
from inference_engine.memory.slab import SlabConfig
from inference_engine.scheduler.config import AdmissionPolicy, SchedulerConfig
from inference_engine.scheduler.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Test doubles (local copies; identical behaviour to E2's versions)
# ---------------------------------------------------------------------------


class DeterministicTokenizer:
    """Minimal HF-AutoTokenizer-shaped tokenizer; word-id mapping."""

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

    def apply_chat_template(  # pragma: no cover - unused by scheduler tests
        self, *args, **kwargs
    ) -> Any:
        raise NotImplementedError

    def decode(  # pragma: no cover - unused by scheduler tests
        self, token_ids, *, skip_special_tokens=False
    ):
        raise NotImplementedError

    def convert_tokens_to_ids(  # pragma: no cover - unused by scheduler tests
        self, token: str
    ) -> Optional[int]:
        return self._token_to_id.get(token)


class DeterministicEngine:
    """Engine test double emitting a fixed token sequence."""

    def __init__(
        self,
        fixed_tokens: List[int],
        tokenizer: DeterministicTokenizer,
        model_id_label: str = "kakeya-test",
        per_token_delay_s: float = 0.0,
    ) -> None:
        if not fixed_tokens:
            raise ValueError("fixed_tokens must be non-empty")
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
    ):
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )
        if not eos_token_ids:
            raise ValueError("eos_token_ids must be non-empty")
        eos_set = set(int(i) for i in eos_token_ids)
        emitted: List[int] = []
        for tok in self._fixed_tokens:
            if len(emitted) >= max_new_tokens:
                break
            if self._per_token_delay_s > 0:
                import time
                time.sleep(self._per_token_delay_s)
            emitted.append(int(tok))
            if on_token is not None and on_token(int(tok)):
                break
            if int(tok) in eos_set:
                break

        # Lightweight result struct identical to what
        # SpeculativeDecoder.GenerationResult exposes (only the fields
        # the scheduler actually reads).
        class _Result:
            def __init__(self, output_token_ids):
                self.output_token_ids = output_token_ids
                self.acceptance_rate = 1.0
                self.proposer_forward_calls = len(output_token_ids)
                self.verifier_forward_calls = len(output_token_ids)

        return _Result(emitted)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slab_config() -> SlabConfig:
    return SlabConfig(
        num_layers=2, num_heads=2, sink_size=1,
        window_size=2, head_dim=4, dtype=torch.float32,
    )


@pytest.fixture
def small_pool(slab_config: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=3, slab_config=slab_config)


@pytest.fixture
def single_pool(slab_config: SlabConfig) -> SlabPool:
    return SlabPool(num_slabs=1, slab_config=slab_config)


@pytest.fixture
def tokenizer() -> DeterministicTokenizer:
    return DeterministicTokenizer()


@pytest.fixture
def short_engine(tokenizer: DeterministicTokenizer) -> DeterministicEngine:
    hello = tokenizer._intern("hello")
    world = tokenizer._intern("world")
    bang = tokenizer._intern("!")
    return DeterministicEngine(
        fixed_tokens=[hello, world, bang, tokenizer.eos_token_id],
        tokenizer=tokenizer,
    )


@pytest.fixture
def long_engine(tokenizer: DeterministicTokenizer) -> DeterministicEngine:
    ids = [tokenizer._intern(f"tok{i}") for i in range(50)]
    return DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer, model_id_label="long",
    )


@pytest.fixture
def slow_engine(tokenizer: DeterministicTokenizer) -> DeterministicEngine:
    ids = [tokenizer._intern(f"slow{i}") for i in range(20)]
    return DeterministicEngine(
        fixed_tokens=ids, tokenizer=tokenizer,
        model_id_label="slow", per_token_delay_s=0.01,
    )


@pytest.fixture
def reject_scheduler(short_engine, small_pool):
    return Scheduler(
        engine=short_engine, pool=small_pool,
        config=SchedulerConfig(
            max_concurrent=small_pool.total_count,
            admission_policy=AdmissionPolicy.REJECT,
        ),
    )


@pytest.fixture
def queue_scheduler(short_engine, small_pool):
    return Scheduler(
        engine=short_engine, pool=small_pool,
        config=SchedulerConfig(
            max_concurrent=small_pool.total_count,
            admission_policy=AdmissionPolicy.QUEUE,
            queue_max_wait_s=2.0,
        ),
    )
