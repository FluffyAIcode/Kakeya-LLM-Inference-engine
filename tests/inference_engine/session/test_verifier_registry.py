"""PR-A3c per-session binding: registry + coordinator resolver isolation.

Proves that with the per-session verifier registry, concurrent/interleaved
sessions have ISOLATED KV state (no cross-session corruption) — the
correctness prerequisite for true multi-tenant serving. Linux-runnable with a
deterministic fake verifier (no model weights).
"""

from __future__ import annotations

import torch

from inference_engine.session.coordinator import AppendTokensCoordinator
from inference_engine.session.generator import GenerationCoordinator, TokenEvent
from inference_engine.session.store import SessionStore
from inference_engine.session.verifier_registry import PerSessionVerifierRegistry

_V = 100


class _FakeVerifier:
    """Deterministic verifier: predicts ``(last_committed + 1) % V``. Each
    instance owns its own committed sequence (per-session state)."""

    def __init__(self) -> None:
        self.cached_token_sequence: list = []
        self.next_global_position: int = 0
        self.next_token_logits = torch.zeros(_V)

    @staticmethod
    def _onehot(tok: int) -> torch.Tensor:
        v = torch.zeros(_V)
        v[(tok + 1) % _V] = 1.0
        return v

    def spawn(self) -> "_FakeVerifier":
        return _FakeVerifier()

    def prefill(self, prompt_ids):
        self.cached_token_sequence = list(prompt_ids)
        self.next_global_position = len(prompt_ids)
        self.next_token_logits = self._onehot(prompt_ids[-1])

    def forward_block(self, tokens):
        return torch.stack([self._onehot(t) for t in tokens])

    def commit_or_truncate(self, *, forwarded: int, accepted: int):
        # committed tokens were the pending block; mirror them into the
        # bounded sequence + advance position.
        # (the coordinator passes forwarded==accepted for prompt/greedy)
        pass

    def k_seq_length(self, session) -> int:
        return len(self.cached_token_sequence)

    def kv_live_bytes(self, session) -> int:
        return len(self.cached_token_sequence) * 2


def _commit(verifier, tok):
    verifier.cached_token_sequence.append(tok)
    verifier.next_global_position += 1


def test_registry_creates_and_isolates_per_session():
    reg = PerSessionVerifierRegistry(factory=_FakeVerifier)
    a = reg.get("s-a")
    b = reg.get("s-b")
    assert a is not b
    assert reg.get("s-a") is a              # cached
    assert reg.active_sessions() == 2
    reg.remove("s-a")
    assert reg.active_sessions() == 1
    assert reg.get("s-a") is not a          # recreated fresh


def test_cache_inspector_routes_per_session():
    reg = PerSessionVerifierRegistry(factory=_FakeVerifier)
    reg.get("s-a").cached_token_sequence = [1, 2, 3]
    reg.get("s-b").cached_token_sequence = [9]

    class _S:
        def __init__(self, sid):
            self.session_id = sid
    assert reg.k_seq_length(_S("s-a")) == 3
    assert reg.k_seq_length(_S("s-b")) == 1


def _make(store, reg):
    base = _FakeVerifier()
    return (AppendTokensCoordinator(store, base, resolver=reg.get),
            GenerationCoordinator(store, base, resolver=reg.get))


def test_interleaved_sessions_do_not_corrupt_each_other():
    """Two sessions, interleaved append+generate, keep separate token streams.

    Session A is seeded at token 10 (so it must emit 11,12,13,...); B at 20
    (21,22,23,...). Interleaving B's work between A's steps must NOT shift A's
    stream — that only holds if each session has its own verifier (per-session
    binding). With a single shared verifier this test would fail.
    """
    reg = PerSessionVerifierRegistry(factory=_FakeVerifier)
    store = SessionStore(capacity=8, cache_inspector=reg)
    append, gen = _make(store, reg)

    sa = store.create_session().session_id
    sb = store.create_session().session_id
    append.append_tokens(sa, [10])
    append.append_tokens(sb, [20])

    def step(sid):
        for ev in gen.generate(sid, max_tokens=1):
            if isinstance(ev, TokenEvent):
                # mirror the commit the fake didn't do (test-only bookkeeping)
                _commit(reg.get(sid), ev.token_id)
                return ev.token_id
        return None

    # Interleave: A, B, A, B, A, B
    a_stream = []
    b_stream = []
    for _ in range(3):
        a_stream.append(step(sa))
        b_stream.append(step(sb))

    assert a_stream == [11, 12, 13]
    assert b_stream == [21, 22, 23]
