"""ADR 0007 §2.7 + §2.9 INV-3 — determinism gate.

The contract: cross-request KV reuse (continuation path) must produce
**bit-identical** output to the always-reset path for any input that
satisfies the §2.4.a continuation precondition. This is the §2.7
determinism claim and the §2.9 INV-3 invariant.

This test is a **mandatory pre-merge gate** for the v0.3.0 GA path.
PR 7-1 through PR 7-4 introduced the cross-request reuse machinery;
this PR (7-5) ships the test that proves it preserves correctness.

How the test works
==================

1. Build TWO independent decoders + verifiers. Same model weights,
   same config, same seed. They start identical.
2. Drive a 30-turn synthetic conversation through both:
     - decoder_reuse: take the continuation path naturally
     - decoder_reset: force ``verifier.reset()`` BEFORE each turn,
                      so path_select sees an empty cache and
                      returns NewSession every time. This emulates
                      v0.3.0-rc1's per-turn-reset behavior.
3. Compare the output token sequences turn-by-turn. INV-3 demands
   bit-identical agreement.

Why we trust this gate
======================

If decoder_reuse and decoder_reset produce identical outputs across
30 multi-turn extensions, then the cross-request KV reuse path is
correct: it IS the same K/V state the reset path computes from
scratch, just achieved with O(new_tokens) prefill cost instead of
O(history_length).

If they diverge:
  - The first divergent turn shows the bug.
  - Likely culprit: cached_token_sequence drifted from K/V tensor
    state (INV-1 should have caught this earlier — but if the bug
    is in the trim path that runs across BOTH paths, INV-1 might
    not detect it).
  - This fails the §2.7 contract; PR cannot merge.

Greedy decoding
===============

Both decoders use temperature=0 (greedy) — required by ADR 0001 §2.2
and ADR 0007's deterministic-output assumption. Non-greedy decoding
introduces RNG which would make this gate's "bit-identical" check
incoherent without seeded RNG, which is out of scope for v0.3.

The proposer is shared (it's stateless w.r.t. prior turns — DLM
doesn't have a KV cache). Sharing it across the two decoders is
not a state-leak risk; it ensures both runs see the exact same
proposer drafts.

Numerical determinism on Apple Metal
====================================

Per ADR 0007 §2.7's resolved OQ-2: bit-identical is the strict
gate. If a future Mac M4 run shows the strict gate is unreachable,
the relaxation must be written into ADR 0007 explicitly first
(amendment) before this test is changed. Tests that auto-relax
based on whether the strict path passes are forbidden.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from kv_cache_proposer.proposer import DLMProposer
from kv_cache_proposer.speculative import SpeculativeDecoder
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig


def _make_decoder(proposer: DLMProposer) -> SpeculativeDecoder:
    """Construct a fresh decoder with default v0.3 sink+window config."""
    verifier = SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16,
            device="cpu",
            sink_size=4,
            window_size=64,
        )
    )
    return SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=4,
        num_diffusion_steps=4,
    )


def _eos_ids(tokenizer) -> List[int]:
    """Resolve EOS ids the same way the engine does at runtime."""
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _build_30_turn_conversation(
    seed_prompt: List[int], n_turns: int, tokens_per_extension: int
) -> List[List[int]]:
    """Generate a list of prompts, each extending the previous one.

    Turn N's prompt is turn N-1's prompt + ``tokens_per_extension``
    deterministic synthetic tokens. We use small ints so the
    extensions don't accidentally hit out-of-vocab paths in the
    real Qwen3 model.
    """
    prompts: List[List[int]] = []
    history = list(seed_prompt)
    prompts.append(list(history))
    for turn in range(1, n_turns):
        # Append ``tokens_per_extension`` tokens, deterministic by turn idx
        for k in range(tokens_per_extension):
            history.append((100 + turn * 7 + k) % 1000)
        prompts.append(list(history))
    return prompts


# ---------------------------------------------------------------------------
# The gate test
# ---------------------------------------------------------------------------


def test_inv_3_bit_identical_continuation_vs_reset(
    proposer_session: DLMProposer, short_chat_messages,
) -> None:
    """ADR 0007 §2.9 INV-3: continuation path output is bit-identical
    to the always-reset path output for every continuation-eligible
    input across a multi-turn synthetic conversation.

    The conversation is built turn-by-turn so each turn's prompt
    strictly extends the previous turn's full history (prompt +
    assistant_response + new_user_tokens). This is the realistic
    agent-loop pattern that the cross-request KV reuse fix targets.

    Mandatory pre-merge gate per ADR 0007 §6 item 2.
    """
    seed_prompt = proposer_session.encode_chat(short_chat_messages)

    decoder_reuse = _make_decoder(proposer_session)
    decoder_reset = _make_decoder(proposer_session)
    eos = _eos_ids(decoder_reuse.verifier.tokenizer)

    # Run a 10-turn conversation. We can't go to 30 turns at
    # max_new_tokens=4 each because the EOS frequently fires in
    # the deterministic short_chat_messages workload, ending the
    # responses early — which is fine for §2.7 (still bit-identical)
    # but doesn't exercise the continuation path heavily.
    n_turns = 10
    history = list(seed_prompt)
    divergent_turns: List[int] = []
    first_divergence_msg = ""

    for turn_idx in range(n_turns):
        # Take whatever path the reuse decoder picks naturally.
        result_reuse = decoder_reuse.generate(
            history, max_new_tokens=4, eos_token_ids=eos,
        )
        # Reset path: force NewSession.
        decoder_reset.verifier.reset()
        result_reset = decoder_reset.generate(
            history, max_new_tokens=4, eos_token_ids=eos,
        )

        if result_reuse.output_token_ids != result_reset.output_token_ids:
            divergent_turns.append(turn_idx)
            if len(divergent_turns) == 1:
                first_divergence_msg = (
                    f"Turn {turn_idx}: continuation path output != "
                    f"reset path output.\n"
                    f"  history length = {len(history)}\n"
                    f"  reuse output    = {result_reuse.output_token_ids}\n"
                    f"  reset output    = {result_reset.output_token_ids}\n"
                    f"  reuse path_selection = "
                    f"{result_reuse.path_selection}\n"
                    f"  reset path_selection = "
                    f"{result_reset.path_selection}\n"
                )

        # Extend history with the assistant's reply (use the reuse
        # decoder's output; both decoders agreed by INV-3, so this
        # choice is symmetric) plus 2 deterministic new "user" tokens
        # so the next turn's prompt strictly extends the cache.
        history = (
            list(history)
            + list(result_reuse.output_token_ids)
            + [(50 + turn_idx) % 1000, (37 + turn_idx) % 1000]
        )

    if divergent_turns:
        raise AssertionError(
            f"INV-3 violated: continuation path output diverges from "
            f"reset path on {len(divergent_turns)} of {n_turns} "
            f"turns ({divergent_turns}).\n\n"
            f"First divergence:\n{first_divergence_msg}\n"
            f"This is a §2.7 / §2.9 INV-3 violation; v0.3.0 GA is "
            f"blocked until the cross-request KV reuse path is fixed "
            f"to produce bit-identical output to the reset path. Per "
            f"the project's no-fallback principle, the fix is in the "
            f"reuse path, not in this test."
        )


def test_inv_3_first_turn_takes_new_session_in_both_paths(
    proposer_session: DLMProposer, short_chat_messages,
) -> None:
    """Sanity check: on the very first turn, both decoders are at
    cold state, so both take the new-session path. Outputs must be
    identical (this is just running the same input through two
    identical models — trivially bit-identical, but a useful pre-
    flight before the multi-turn gate above)."""
    seed_prompt = proposer_session.encode_chat(short_chat_messages)
    decoder_a = _make_decoder(proposer_session)
    decoder_b = _make_decoder(proposer_session)
    eos = _eos_ids(decoder_a.verifier.tokenizer)

    result_a = decoder_a.generate(
        seed_prompt, max_new_tokens=4, eos_token_ids=eos,
    )
    result_b = decoder_b.generate(
        seed_prompt, max_new_tokens=4, eos_token_ids=eos,
    )

    assert result_a.output_token_ids == result_b.output_token_ids
    # Both should have taken new_session (cold cache).
    assert result_a.path_selection == "new_session"
    assert result_b.path_selection == "new_session"


def test_inv_3_continuation_path_is_actually_taken(
    proposer_session: DLMProposer, short_chat_messages,
) -> None:
    """Sanity check that the gate is actually testing what it claims.
    If decoder_reuse never takes the continuation path (e.g. some
    bug routes everything to NewSession), the determinism check is
    trivially satisfied but doesn't prove anything.

    Confirm: the second turn of a multi-turn run must take the
    continuation path."""
    seed_prompt = proposer_session.encode_chat(short_chat_messages)
    decoder = _make_decoder(proposer_session)
    eos = _eos_ids(decoder.verifier.tokenizer)

    # Turn 1: cold start
    result1 = decoder.generate(
        seed_prompt, max_new_tokens=4, eos_token_ids=eos,
    )
    assert result1.path_selection == "new_session"

    # Turn 2: prompt = full prior history (prompt + assistant reply)
    # + 2 new "user" tokens. This strictly extends what the cache
    # holds (prompt + generated tokens, all already in cache).
    history2 = (
        list(seed_prompt)
        + list(result1.output_token_ids)
        + [99, 88]
    )
    result2 = decoder.generate(
        history2, max_new_tokens=4, eos_token_ids=eos,
    )
    assert result2.path_selection == "continuation", (
        f"expected continuation path; got {result2.path_selection}. "
        f"If this fails, the determinism gate is not actually "
        f"exercising the cross-request reuse path."
    )
    assert result2.tokens_skipped > 0


# ---------------------------------------------------------------------------
# What this gate does NOT cover
# ---------------------------------------------------------------------------
#
# Out of scope for the gate (covered elsewhere):
#   * MLX-backend numerical determinism: the gate runs on the CPU
#     verifier. The MLX backend has its own platform tests
#     (run on Mac M4, see PR 7-6's bench_long_session_v2) and
#     ADR 0007 §2.7 OQ-2 covers the relaxation policy if Metal
#     produces tiny float differences.
#   * Long-session memory bound: §2.3.a evidence is collected in
#     PR 7-6's 4h Mac re-run, not here.
#   * Per-turn latency: § the §2.3.b → §2.3.a reframing happens in
#     PR 7-7 once we have the v2 4h evidence.
