"""Unit tests for `kv_cache_proposer.speculative.SpeculativeDecoder`."""

from __future__ import annotations

import pytest
import torch

from kv_cache_proposer.proposer import DLMProposer, ProposerConfig
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig
from kv_cache_proposer.speculative import (
    SpeculativeDecoder,
    SpeculativeRunResult,
)


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


@pytest.fixture(scope="module")
def decoder(proposer_session: DLMProposer) -> SpeculativeDecoder:
    verifier = SinkWindowVerifier(
        VerifierConfig(dtype=torch.bfloat16, device="cpu", sink_size=4, window_size=64)
    )
    return SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier, block_size=4, num_diffusion_steps=4
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_block", [0, -1])
def test_decoder_rejects_nonpositive_block_size(
    proposer_session: DLMProposer, fresh_verifier_factory, bad_block: int
) -> None:
    with pytest.raises(ValueError, match="block_size must be > 0"):
        SpeculativeDecoder(
            proposer=proposer_session,
            verifier=fresh_verifier_factory(),
            block_size=bad_block,
            num_diffusion_steps=4,
        )


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_decoder_rejects_nonpositive_num_diffusion_steps(
    proposer_session: DLMProposer, fresh_verifier_factory, bad_steps: int
) -> None:
    with pytest.raises(ValueError, match="num_diffusion_steps must be > 0"):
        SpeculativeDecoder(
            proposer=proposer_session,
            verifier=fresh_verifier_factory(),
            block_size=4,
            num_diffusion_steps=bad_steps,
        )


def test_generate_rejects_nonpositive_max_new_tokens(decoder: SpeculativeDecoder) -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        decoder.generate([1, 2, 3], max_new_tokens=0)


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------

def test_generate_produces_tokens_and_stats(
    decoder: SpeculativeDecoder, proposer_session: DLMProposer, short_chat_messages
) -> None:
    prompt = proposer_session.encode_chat(short_chat_messages)
    eos = _eos_ids(decoder.verifier.tokenizer)
    result = decoder.generate(prompt, max_new_tokens=8, eos_token_ids=eos)
    assert isinstance(result, SpeculativeRunResult)
    # Basic counters
    assert result.proposer_forward_calls > 0
    assert result.proposer_diffusion_steps == result.proposer_forward_calls
    assert result.verifier_forward_calls >= 1  # at least prefill
    assert result.verifier_tokens_consumed >= len(prompt)
    assert result.verifier_peak_kv_bytes > 0
    assert result.verifier_final_kv_bytes > 0
    assert result.verifier_peak_activation_bytes > 0
    assert result.proposer_peak_activation_bytes > 0
    assert result.wall_time_seconds > 0
    # Per-block trace shape
    assert len(result.accepted_per_block) == len(result.proposed_per_block)
    # acceptance_rate is in [0, 1]
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_generate_with_no_proposed_tokens_returns_zero_acceptance() -> None:
    """If max_new_tokens=1 and the very first verifier prediction is EOS,
    we may finish before any proposer block — exercise the empty-history
    `acceptance_rate` path."""
    r = SpeculativeRunResult(
        output_token_ids=[],
        accepted_per_block=[],
        proposed_per_block=[],
        proposer_forward_calls=0,
        proposer_diffusion_steps=0,
        verifier_forward_calls=0,
        verifier_tokens_consumed=0,
        proposer_peak_activation_bytes=0,
        proposer_weight_bytes=0,
        verifier_peak_kv_bytes=0,
        verifier_final_kv_bytes=0,
        verifier_peak_activation_bytes=0,
        verifier_weight_bytes=0,
        verifier_final_kv_token_count=0,
        wall_time_seconds=0.0,
    )
    assert r.acceptance_rate == 0.0
    assert r.total_proposed == 0
    assert r.total_accepted == 0


def test_generate_eos_in_accepted_prefix_truncates(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """When EOS appears inside the accepted span of a block, generation
    must stop there and the trailing accepted tokens must be discarded."""
    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session,
        verifier=verifier,
        block_size=8,
        num_diffusion_steps=8,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier.tokenizer)
    result = decoder.generate(prompt, max_new_tokens=64, eos_token_ids=eos)
    # The model with this prompt should hit EOS before max_new_tokens.
    assert any(t in eos for t in result.output_token_ids)
    # No tokens after the first EOS
    eos_pos = next(i for i, t in enumerate(result.output_token_ids) if t in eos)
    assert eos_pos == len(result.output_token_ids) - 1


def test_kv_bytes_static_helper_returns_zero_for_no_cache(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    verifier = fresh_verifier_factory()
    # No prefill yet; cache is None.
    assert SpeculativeDecoder._kv_bytes(verifier) == 0


# ---------------------------------------------------------------------------
# Streaming callback (on_token)
# ---------------------------------------------------------------------------

def test_on_token_callback_emits_each_committed_token_in_order(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """The on_token callback must fire once per committed token, in the
    same order as result.output_token_ids."""
    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=4, num_diffusion_steps=4,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier.tokenizer)
    streamed: list[int] = []

    def _cb(tok_id: int):
        streamed.append(tok_id)
        return False

    result = decoder.generate(
        prompt_ids=prompt, max_new_tokens=16,
        eos_token_ids=eos, on_token=_cb,
    )
    assert streamed == result.output_token_ids


def test_on_token_callback_can_request_early_stop(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """Returning truthy from the callback halts the loop at the next
    safe checkpoint (between blocks). The streamed tokens form a
    *prefix* of the output sequence; generation may commit a small
    number of additional tokens between the callback's True return and
    the loop exit (at most one block of L tokens, plus the
    correction/bonus token that block produces)."""
    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=4, num_diffusion_steps=4,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Count: one, two, three"},
    ]
    prompt = proposer_session.encode_chat(msgs)
    streamed: list[int] = []

    def _cb_stop_after_first(tok_id: int):
        streamed.append(tok_id)
        return True  # stop on the very first token

    result = decoder.generate(
        prompt_ids=prompt, max_new_tokens=64,
        eos_token_ids=None, on_token=_cb_stop_after_first,
    )
    assert len(streamed) >= 1
    # streamed is a prefix of result.output_token_ids
    assert streamed == result.output_token_ids[: len(streamed)]
    # Generation overshoots by at most one block + correction (block_size=4).
    assert len(result.output_token_ids) - len(streamed) <= 4 + 1


def test_on_token_callback_can_stop_on_accepted_token(
    monkeypatch, proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """Cover the accepted-block emit-then-stop branch explicitly.

    Force the proposer to emit exactly the verifier's greedy
    continuation (so accepted == block_size > 0 in the first block).
    A callback that returns True on the very first emission then
    exercises the ``_emit(d[:accepted])`` early-stop path."""
    # Build an oracle: the verifier's own greedy first 4 tokens, used
    # as the proposer's draft so they all get accepted.
    verifier = fresh_verifier_factory(sink=4, window=64)
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    verifier.prefill(list(prompt))
    greedy: list[int] = []
    for _ in range(4):
        tok = int(torch.argmax(verifier.next_token_logits).item())
        greedy.append(tok)
        verifier.append_token(tok)

    real_propose = proposer_session.propose_block

    def _oracle_propose(committed_token_ids, block_size, num_steps):
        proposal = real_propose(committed_token_ids, block_size, num_steps)
        return type(proposal)(
            tokens=greedy[:block_size],
            diffusion_steps=proposal.diffusion_steps,
            forward_passes=proposal.forward_passes,
            peak_activation_bytes=proposal.peak_activation_bytes,
        )

    monkeypatch.setattr(proposer_session, "propose_block", _oracle_propose)

    fresh = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=fresh,
        block_size=4, num_diffusion_steps=4,
    )
    streamed: list[int] = []

    def _cb(tok_id: int):
        streamed.append(tok_id)
        return True  # stop on the first emission

    result = decoder.generate(
        prompt_ids=list(prompt), max_new_tokens=16,
        eos_token_ids=None, on_token=_cb,
    )
    # The first proposed block was 4 oracle-tokens (all accepted), so
    # the first emission lands inside _emit(d[:accepted]) — line 206-208.
    assert result.acceptance_rate > 0  # we verified at least one accept
    assert len(streamed) == 1


def test_on_token_callback_can_stop_on_correction_token(
    monkeypatch, proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """Cover the correction/bonus emit-then-stop branch explicitly.

    Forcing the proposer to always emit token 0 (always wrong) means
    every block's accepted=0 and the only emission is the correction
    token. A callback that returns True on the first correction
    therefore exercises the post-correction `_emit([correction_or_bonus])`
    branch."""
    real_propose = proposer_session.propose_block

    def _always_wrong(committed_token_ids, block_size, num_steps):
        proposal = real_propose(committed_token_ids, block_size, num_steps)
        return type(proposal)(
            tokens=[0] * block_size,
            diffusion_steps=proposal.diffusion_steps,
            forward_passes=proposal.forward_passes,
            peak_activation_bytes=proposal.peak_activation_bytes,
        )

    monkeypatch.setattr(proposer_session, "propose_block", _always_wrong)

    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=2, num_diffusion_steps=2,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    streamed: list[int] = []

    def _cb(tok_id: int):
        streamed.append(tok_id)
        return True  # stop after the first emission

    result = decoder.generate(
        prompt_ids=prompt, max_new_tokens=8,
        eos_token_ids=None, on_token=_cb,
    )
    # acceptance=0 in every block (proposer is always wrong) — every
    # emission is a correction-or-bonus token, exercising the
    # post-correction _emit branch.
    assert result.acceptance_rate == 0.0
    assert len(streamed) == 1
    assert streamed == result.output_token_ids[:1]


def test_on_token_callback_fires_on_eos_termination(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """When EOS is in the accepted prefix and ends generation, the
    callback must still see those tokens (including the EOS)."""
    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=8, num_diffusion_steps=8,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier.tokenizer)
    streamed: list[int] = []

    def _cb(tok_id: int):
        streamed.append(tok_id)
        return False

    result = decoder.generate(
        prompt_ids=prompt, max_new_tokens=64,
        eos_token_ids=eos, on_token=_cb,
    )
    assert streamed == result.output_token_ids
    # The model with this prompt EOSes quickly.
    assert any(t in set(eos) for t in streamed)


def test_on_token_callback_none_is_no_op(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """When on_token is None, the loop runs without any callback overhead
    and produces the exact same result as a callback-free run."""
    verifier1 = fresh_verifier_factory(sink=4, window=64)
    verifier2 = fresh_verifier_factory(sink=4, window=64)
    decoder1 = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier1,
        block_size=4, num_diffusion_steps=4,
    )
    decoder2 = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier2,
        block_size=4, num_diffusion_steps=4,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier1.tokenizer)

    no_cb = decoder1.generate(
        prompt_ids=prompt, max_new_tokens=16, eos_token_ids=eos,
    )
    streamed: list[int] = []
    with_cb = decoder2.generate(
        prompt_ids=prompt, max_new_tokens=16, eos_token_ids=eos,
        on_token=lambda t: streamed.append(t) or False,
    )
    assert no_cb.output_token_ids == with_cb.output_token_ids
    assert streamed == with_cb.output_token_ids


def test_generate_appends_correction_or_bonus(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """A non-EOS-prone prompt should run multiple blocks, exercising the
    append_token path (lines 183-185) for the correction/bonus token at
    each iteration where EOS is not yet reached."""
    verifier = fresh_verifier_factory(sink=4, window=128)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=2, num_diffusion_steps=2,
    )
    msgs = [
        {"role": "system", "content": "You are a counter."},
        {"role": "user", "content": "Count: one, two,"},
    ]
    prompt = proposer_session.encode_chat(msgs)
    # No EOS set → forces every iteration through the append-correction path
    # because we cannot stop on EOS.
    result = decoder.generate(prompt, max_new_tokens=4, eos_token_ids=None)
    assert len(result.output_token_ids) == 4
    # If we generated 4 tokens with block_size=2, we ran at least 2 outer
    # iterations and therefore committed at least one correction or bonus.
    assert len(result.accepted_per_block) >= 2


def test_generate_max_new_tokens_short_circuits_after_block(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """When the accepted span of a block fills `max_new_tokens` exactly
    (and contains no EOS), the loop should break before append_token —
    exercising the early-exit branch on line 177."""
    verifier = fresh_verifier_factory(sink=4, window=128)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=4, num_diffusion_steps=4,
    )
    msgs = [
        {"role": "system", "content": "You are a counter."},
        {"role": "user", "content": "Count: one, two,"},
    ]
    prompt = proposer_session.encode_chat(msgs)
    # No EOS set; max_new_tokens not a multiple of block_size — last block
    # is shrunk to fit, and after it we must break before appending bonus.
    result = decoder.generate(prompt, max_new_tokens=3, eos_token_ids=None)
    assert len(result.output_token_ids) == 3


def test_generate_correction_token_can_be_eos(
    proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """When the correction or bonus token is itself an EOS id, the loop
    must commit it and stop (lines 186-188)."""
    # Use a prompt that strongly biases the verifier to emit EOS quickly;
    # use a small block_size so the correction-token branch is the most
    # likely path to encounter EOS rather than the in-accepted-prefix path.
    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=1, num_diffusion_steps=1,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier.tokenizer)
    result = decoder.generate(prompt, max_new_tokens=8, eos_token_ids=eos)
    # The output must terminate with an EOS token, possibly via the
    # correction path or via the accepted-prefix path. Either way, the
    # generated sequence ends with EOS.
    assert any(t in eos for t in result.output_token_ids)
    assert result.output_token_ids[-1] in eos


def test_generate_correction_is_eos_via_rejected_proposal(
    monkeypatch, proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """Hit the EOS-on-correction branch (lines 186-188).

    Strategy: have the proposer ALWAYS emit token id 0 (almost certainly
    wrong). Every iteration the verifier will reject, the correction will
    be the verifier's own argmax, the correction is committed via
    append_token. Eventually for the prompt 'Reply with exactly: OK.'
    the verifier's argmax is `<|im_end|>` (EOS), and that gets committed
    as a correction → branch fires.
    """
    verifier = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=verifier,
        block_size=1, num_diffusion_steps=1,
    )
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier.tokenizer)
    real_propose = proposer_session.propose_block

    def _always_wrong(committed_token_ids, block_size, num_steps):
        proposal = real_propose(committed_token_ids, block_size, num_steps)
        return type(proposal)(
            tokens=[0] * block_size,  # token 0 is essentially never the verifier's argmax for this prompt
            diffusion_steps=proposal.diffusion_steps,
            forward_passes=proposal.forward_passes,
            peak_activation_bytes=proposal.peak_activation_bytes,
        )

    monkeypatch.setattr(proposer_session, "propose_block", _always_wrong)
    result = decoder.generate(list(prompt), max_new_tokens=16, eos_token_ids=eos)
    # Acceptance rate must be 0 (every proposal rejected); we generated
    # solely via corrections, and the final correction is EOS.
    assert result.acceptance_rate == 0.0
    assert result.output_token_ids[-1] in eos
    # All tokens before the last are non-EOS (corrections leading up to EOS).
    assert all(t not in eos for t in result.output_token_ids[:-1])


def test_generate_eos_in_middle_of_accepted_block_drops_trailing(
    monkeypatch, proposer_session: DLMProposer, fresh_verifier_factory
) -> None:
    """Hit the lines 170-171 trim branch.

    To enter that branch we need: a block where the verifier accepts
    AT LEAST 3 tokens AND an EOS token sits at index <= accepted - 2 of
    the accepted span. We cannot predict naturally where the verifier
    will emit EOS, so we use the verifier *itself* as an oracle to
    discover, for the prompt at hand, the exact 4-token sequence
    `[t0, t1, EOS, t3]` such that the verifier would greedily accept
    all four. Then we monkeypatch the proposer to emit exactly that
    sequence (the proposer pays its real diffusion cost; we only swap
    the returned token ids).
    """
    verifier = fresh_verifier_factory(sink=4, window=64)
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    prompt = proposer_session.encode_chat(msgs)
    eos = _eos_ids(verifier.tokenizer)
    eos_id = eos[0]

    # Step 1: discover the verifier's greedy continuation up to 4 tokens.
    # We do this with a fresh prefill, then four single-token argmax steps.
    verifier.prefill(list(prompt))
    greedy = []
    for _ in range(4):
        tok = int(torch.argmax(verifier.next_token_logits).item())
        greedy.append(tok)
        verifier.append_token(tok)

    # Step 2: replace at most ONE token in the middle with EOS, in a
    # position chosen so accepted will be >=3 with EOS not at the end.
    # The verifier's natural sequence for "Reply with exactly: OK." is
    # short (often `OK . <|im_end|>`), and EOS will already appear in it.
    # Find the EOS position naturally:
    eos_positions = [i for i, t in enumerate(greedy) if t in eos]
    if not eos_positions or eos_positions[0] >= len(greedy) - 1:
        # Construct the artificial block: keep first 2 tokens of greedy,
        # insert EOS at position 2, then append another natural-looking
        # token at position 3. The verifier WILL accept the first 2
        # naturally; whether it accepts EOS at position 2 depends on
        # logits, but for "Reply ... OK." the verifier predicts
        # `<|im_end|>` after `OK.`, so the construction is plausible.
        block = greedy[:2] + [eos_id, greedy[3] if len(greedy) > 3 else greedy[0]]
    else:
        # Natural EOS in middle: just propose greedy[:4] and rely on
        # the in-block trim path.
        block = greedy[:4]

    real_propose = proposer_session.propose_block

    def _fake_propose(committed_token_ids, block_size, num_steps):
        proposal = real_propose(committed_token_ids, block_size, num_steps)
        # Pay the real diffusion cost, then swap tokens.
        return type(proposal)(
            tokens=block[:block_size],
            diffusion_steps=proposal.diffusion_steps,
            forward_passes=proposal.forward_passes,
            peak_activation_bytes=proposal.peak_activation_bytes,
        )

    fresh = fresh_verifier_factory(sink=4, window=64)
    decoder = SpeculativeDecoder(
        proposer=proposer_session, verifier=fresh,
        block_size=4, num_diffusion_steps=4,
    )
    monkeypatch.setattr(proposer_session, "propose_block", _fake_propose)
    result = decoder.generate(list(prompt), max_new_tokens=8, eos_token_ids=eos)
    # Must end at EOS, never include any trailing post-EOS tokens.
    assert any(t in eos for t in result.output_token_ids)
    assert result.output_token_ids[-1] in eos
    # No duplicate EOS or any token after the first EOS in output.
    first_eos_idx = next(i for i, t in enumerate(result.output_token_ids) if t in eos)
    assert first_eos_idx == len(result.output_token_ids) - 1


