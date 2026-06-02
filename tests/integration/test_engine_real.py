"""Integration tests for :class:`SpeculativeEngine`.

PR-N3 migration of the former Linux-side ``test_engine.py`` (which
used ``_DecoderDouble`` + ``_VerifierDouble`` test mirrors). The
SpeculativeEngine wrapper's contract is generation
orchestration (forward result → ``EngineResult``); validating it
against real Qwen3-0.6B numerics confirms the wrapper preserves
the underlying decoder's behavior.
"""

from __future__ import annotations

import pytest

from inference_engine.server.engine import EngineResult


@pytest.fixture
def engine(real_speculative_engine):
    return real_speculative_engine


def test_engine_exposes_tokenizer_and_model_id_label(engine):
    assert engine.tokenizer is not None
    assert isinstance(engine.model_id_label, str)
    assert engine.model_id_label  # non-empty


def test_engine_generate_returns_engine_result(engine):
    eos = engine.tokenizer.eos_token_id
    eos_ids = [int(eos)] if eos is not None else [0]
    result = engine.generate(
        prompt_ids=engine.tokenizer.encode(
            "Reply with one word.", add_special_tokens=False,
        ),
        max_new_tokens=4,
        eos_token_ids=eos_ids,
    )
    assert isinstance(result, EngineResult)
    assert isinstance(result.output_token_ids, list)
    assert len(result.output_token_ids) >= 1
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.proposer_forward_calls >= 0
    assert result.verifier_forward_calls >= 0
    assert isinstance(result.stopped_on_eos, bool)


def test_engine_generate_respects_max_new_tokens(engine):
    eos = engine.tokenizer.eos_token_id
    # Use a synthetic eos id well outside the vocab so generation
    # cannot stop on EOS, forcing the max_new_tokens stop path.
    result = engine.generate(
        prompt_ids=engine.tokenizer.encode(
            "Tell me a story.", add_special_tokens=False,
        ),
        max_new_tokens=3,
        eos_token_ids=[10**9],
    )
    assert len(result.output_token_ids) <= 3
    assert result.stopped_on_eos is False


def test_engine_generate_rejects_empty_prompt(engine):
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        engine.generate(
            prompt_ids=[],
            max_new_tokens=4,
            eos_token_ids=[int(engine.tokenizer.eos_token_id) or 0],
        )


def test_engine_generate_rejects_zero_max_tokens(engine):
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        engine.generate(
            prompt_ids=[1, 2, 3],
            max_new_tokens=0,
            eos_token_ids=[int(engine.tokenizer.eos_token_id) or 0],
        )


def test_engine_on_token_callback_invoked_per_committed_token(engine):
    callbacks: list[int] = []

    def on_token(tid: int) -> bool:
        callbacks.append(int(tid))
        return False  # never request early stop

    eos_ids = [int(engine.tokenizer.eos_token_id) or 0]
    result = engine.generate(
        prompt_ids=engine.tokenizer.encode(
            "Hi.", add_special_tokens=False,
        ),
        max_new_tokens=4,
        eos_token_ids=eos_ids,
        on_token=on_token,
    )
    # Callback fires once per committed token.
    assert callbacks == result.output_token_ids


def test_engine_on_token_callback_can_request_early_stop(engine):
    seen = []

    def on_token(tid: int) -> bool:
        seen.append(int(tid))
        return True  # stop after the first emitted token

    eos_ids = [10**9]  # avoid EOS path
    result = engine.generate(
        prompt_ids=engine.tokenizer.encode(
            "One.", add_special_tokens=False,
        ),
        max_new_tokens=10,
        eos_token_ids=eos_ids,
        on_token=on_token,
    )
    assert len(result.output_token_ids) == 1
    assert seen == result.output_token_ids
