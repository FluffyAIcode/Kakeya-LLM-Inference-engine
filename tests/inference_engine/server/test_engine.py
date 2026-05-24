"""Unit tests for :class:`SpeculativeEngine` adapter.

The engine adapter is intentionally thin (delegate + result translation),
so a focused suite verifies:

  * construction validates inputs (model_id_label, EOS availability)
  * ``generate()`` forwards args correctly to the underlying decoder
  * ``generate()`` translates result fields (output_token_ids,
    acceptance_rate, etc.) verbatim
  * ``stopped_on_eos`` is computed from the last output token vs the
    eos_token_ids set, regardless of the decoder's own framing
  * defensive validation on ``prompt_ids`` / ``max_new_tokens`` /
    ``eos_token_ids``

We use a real concrete ``_DecoderDouble`` class — not a mock — that
implements the SpeculativeDecoder.generate signature with deterministic
behaviour. The engine accepts it via duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pytest

from inference_engine.server.engine import EngineResult, SpeculativeEngine

# We import the conftest fixture (DeterministicTokenizer) implicitly.


# ---------------------------------------------------------------------------
# Real concrete decoder double
# ---------------------------------------------------------------------------


@dataclass
class _DecoderResult:
    """Minimal duck-type of SpeculativeDecoder's GenerationResult.

    Only the fields the engine adapter reads need to exist; we omit
    the rest to avoid coupling the test to fields the engine does
    not look at.
    """

    output_token_ids: List[int]
    acceptance_rate: float
    proposer_forward_calls: int
    verifier_forward_calls: int


class _DecoderDouble:
    """Concrete decoder stand-in with deterministic output.

    Records the last ``generate`` call's args so tests can assert
    forwarding correctness. Also exposes ``call_count`` for the
    adapter-doesn't-double-call invariant.
    """

    def __init__(self, fixed_tokens: List[int], acceptance: float = 0.5,
                 proposer_calls: int = 7, verifier_calls: int = 3) -> None:
        self._fixed_tokens = list(fixed_tokens)
        self._acceptance = acceptance
        self._proposer_calls = proposer_calls
        self._verifier_calls = verifier_calls
        self.call_count = 0
        self.last_kwargs: Optional[dict] = None

    def generate(
        self,
        *,
        prompt_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: List[int],
        on_token: Optional[Callable[[int], bool]] = None,
    ) -> _DecoderResult:
        self.call_count += 1
        self.last_kwargs = dict(
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=list(eos_token_ids),
            on_token=on_token,
        )
        emitted: List[int] = []
        for tok in self._fixed_tokens[:max_new_tokens]:
            emitted.append(tok)
            if on_token is not None and on_token(tok):
                break
            if tok in set(eos_token_ids):
                break
        return _DecoderResult(
            output_token_ids=emitted,
            acceptance_rate=self._acceptance,
            proposer_forward_calls=self._proposer_calls,
            verifier_forward_calls=self._verifier_calls,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_succeeds_with_valid_inputs(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10, 0])
    engine = SpeculativeEngine(
        decoder=decoder, tokenizer=tokenizer, model_id_label="m"
    )
    assert engine.tokenizer is tokenizer
    assert engine.model_id_label == "m"
    assert engine.decoder is decoder


def test_construction_rejects_empty_model_id_label(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10])
    with pytest.raises(ValueError, match="model_id_label must be a non-empty"):
        SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="")


def test_construction_rejects_whitespace_model_id_label(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10])
    with pytest.raises(ValueError, match="model_id_label must be a non-empty"):
        SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="   ")


def test_construction_rejects_tokenizer_with_no_eos():
    """A tokenizer that reports neither eos nor <|im_end|> is a real
    misconfiguration; the engine refuses to start in that state."""

    class _NoEos:
        eos_token_id = None
        unk_token_id = None

        def apply_chat_template(self, *a, **kw):  # pragma: no cover - unused
            raise NotImplementedError

        def decode(self, *a, **kw):  # pragma: no cover - unused
            return ""

        def convert_tokens_to_ids(self, token):
            return None

    decoder = _DecoderDouble(fixed_tokens=[10])
    with pytest.raises(ValueError, match="no EOS token id"):
        SpeculativeEngine(decoder=decoder, tokenizer=_NoEos(), model_id_label="m")


# ---------------------------------------------------------------------------
# generate(): forwarding & translation
# ---------------------------------------------------------------------------


def test_generate_forwards_prompt_ids_max_new_tokens_eos(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10, 11, 0])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    engine.generate(
        prompt_ids=[1, 2, 3], max_new_tokens=5, eos_token_ids=[0]
    )
    assert decoder.last_kwargs["prompt_ids"] == [1, 2, 3]
    assert decoder.last_kwargs["max_new_tokens"] == 5
    assert decoder.last_kwargs["eos_token_ids"] == [0]


def test_generate_forwards_on_token_callback(tokenizer):
    seen: List[int] = []

    def cb(tid: int) -> bool:
        seen.append(tid)
        return False

    decoder = _DecoderDouble(fixed_tokens=[10, 11, 0])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    engine.generate(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0], on_token=cb)
    # The callback must have been invoked through the decoder.
    assert seen == [10, 11, 0]


def test_generate_translates_result_fields(tokenizer):
    decoder = _DecoderDouble(
        fixed_tokens=[10, 11, 0], acceptance=0.42,
        proposer_calls=17, verifier_calls=4,
    )
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    res = engine.generate(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    assert isinstance(res, EngineResult)
    assert res.output_token_ids == [10, 11, 0]
    assert res.acceptance_rate == pytest.approx(0.42)
    assert res.proposer_forward_calls == 17
    assert res.verifier_forward_calls == 4


def test_generate_stopped_on_eos_when_last_token_is_eos(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10, 11, 0])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    res = engine.generate(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    assert res.stopped_on_eos is True


def test_generate_stopped_on_eos_false_when_last_token_not_eos(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10, 11, 12])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    res = engine.generate(prompt_ids=[1], max_new_tokens=2, eos_token_ids=[0])
    # Truncated by max_new_tokens; last is 11, not in eos.
    assert res.stopped_on_eos is False


def test_generate_stopped_on_eos_false_when_output_empty(tokenizer):
    """If decoder somehow returns empty output, stopped_on_eos must be
    False (no last token to inspect)."""
    decoder = _DecoderDouble(fixed_tokens=[])

    # _DecoderDouble's loop is a no-op for empty fixed_tokens.
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    res = engine.generate(prompt_ids=[1], max_new_tokens=5, eos_token_ids=[0])
    assert res.output_token_ids == []
    assert res.stopped_on_eos is False


def test_generate_only_calls_decoder_once(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10, 0])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    engine.generate(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    assert decoder.call_count == 1


# ---------------------------------------------------------------------------
# Defensive validation
# ---------------------------------------------------------------------------


def test_generate_rejects_empty_prompt_ids(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        engine.generate(prompt_ids=[], max_new_tokens=5, eos_token_ids=[0])


def test_generate_rejects_zero_max_new_tokens(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        engine.generate(prompt_ids=[1], max_new_tokens=0, eos_token_ids=[0])


def test_generate_rejects_negative_max_new_tokens(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        engine.generate(prompt_ids=[1], max_new_tokens=-3, eos_token_ids=[0])


def test_generate_rejects_empty_eos_token_ids(tokenizer):
    decoder = _DecoderDouble(fixed_tokens=[10])
    engine = SpeculativeEngine(decoder=decoder, tokenizer=tokenizer, model_id_label="m")
    with pytest.raises(ValueError, match="eos_token_ids must be non-empty"):
        engine.generate(prompt_ids=[1], max_new_tokens=5, eos_token_ids=[])
