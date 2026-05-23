"""Smoke tests on real downloaded weights.

Validates:
  1) Both tokenizers tokenize the same chat prompt identically (the
     speculative loop relies on this).
  2) The verifier's prefill produces logits whose argmax over a prompt is
     stable (sanity).
  3) The proposer's masked-diffusion `propose_block` actually produces
     non-mask tokens and runs end to end on a tiny block.
  4) `SinkWindowVerifier.forward_block + commit_or_truncate + append_token`
     keeps the cache shape consistent with `sink + window`.

No mock, no fallback: every call is on real weights.
"""

from __future__ import annotations

import sys

import torch

from kv_cache_proposer.proposer import DLMProposer, ProposerConfig
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig


def _eq_or_fail(a, b, msg: str) -> None:
    if a != b:
        raise AssertionError(f"{msg}: a={a!r} b={b!r}")


def main() -> int:
    print("[smoke] loading proposer ...", flush=True)
    proposer = DLMProposer(ProposerConfig(dtype=torch.bfloat16, device="cpu"))
    print("[smoke] loading verifier ...", flush=True)
    verifier = SinkWindowVerifier(
        VerifierConfig(dtype=torch.bfloat16, device="cpu", sink_size=4, window_size=32)
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly 'OK'."},
    ]
    prop_ids = proposer.encode_chat(messages)
    ver_ids = verifier.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    print(f"[smoke] proposer prompt ids ({len(prop_ids)}): {prop_ids[:20]}...", flush=True)
    print(f"[smoke] verifier prompt ids ({len(ver_ids)}): {ver_ids[:20]}...", flush=True)
    _eq_or_fail(prop_ids, ver_ids, "tokenizers diverge on identical chat prompt")
    print("[smoke] (1) tokenizers agree on prompt", flush=True)

    # (2) verifier prefill
    print("[smoke] verifier prefill ...", flush=True)
    verifier.prefill(prop_ids)
    assert verifier.next_token_logits is not None
    pred = int(torch.argmax(verifier.next_token_logits).item())
    pred_str = verifier.tokenizer.decode([pred], skip_special_tokens=False)
    print(f"[smoke] verifier next token id {pred} -> {pred_str!r}", flush=True)
    print(
        f"[smoke] cache_logical_size after prefill+trim: {verifier.cache_logical_size} "
        f"(budget={verifier.config.sink_size + verifier.config.window_size})",
        flush=True,
    )
    assert verifier.cache_logical_size == min(
        len(prop_ids), verifier.config.sink_size + verifier.config.window_size
    ), "cache size after prefill mismatches budget"
    print("[smoke] (2) verifier prefill OK", flush=True)

    # (3) proposer block
    print("[smoke] proposer.propose_block (L=4, K=4) ...", flush=True)
    blk = proposer.propose_block(prop_ids, block_size=4, num_steps=4)
    print(f"[smoke] proposed block: {blk.tokens} -> {proposer.tokenizer.decode(blk.tokens)!r}", flush=True)
    if any(t == proposer.mask_id for t in blk.tokens):
        raise AssertionError("Proposer leaked <|mask|> tokens into block output.")
    print("[smoke] (3) proposer block OK", flush=True)

    # (4) verifier round trip on the proposed block
    print("[smoke] verifier.forward_block + commit_or_truncate + append_token ...", flush=True)
    forwarded = len(blk.tokens)
    block_logits = verifier.forward_block(blk.tokens)
    assert block_logits.shape[0] == forwarded
    # accept only the first 2 tokens (synthetic, just for path coverage)
    accept = min(2, forwarded)
    verifier.commit_or_truncate(forwarded=forwarded, accepted=accept)
    print(f"[smoke] cache after partial accept (forwarded={forwarded}, accepted={accept}): "
          f"size={verifier.cache_logical_size}", flush=True)
    # add one correction token
    correction = int(torch.argmax(block_logits[accept - 1] if accept > 0 else verifier.next_token_logits).item())
    verifier.append_token(correction)
    print(f"[smoke] appended correction token={correction!r}; cache_size={verifier.cache_logical_size}", flush=True)

    # check budget
    budget = verifier.config.sink_size + verifier.config.window_size
    if verifier.cache_logical_size > budget:
        raise AssertionError(
            f"Cache size {verifier.cache_logical_size} exceeds budget {budget}"
        )
    # check actual K/V tensor shapes match logical size
    layer0 = verifier.cache.layers[0]
    real_size = layer0.keys.shape[2]
    if real_size != verifier.cache_logical_size:
        raise AssertionError(
            f"Layer-0 K shape {layer0.keys.shape} ({real_size}) != logical "
            f"size {verifier.cache_logical_size}"
        )
    print("[smoke] (4) verifier cache layout invariant OK", flush=True)
    print("\n[smoke] ALL SMOKE TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
