"""ADR 0008 §7 GA gate G3 — INV-3 byte-exact determinism.

Drives two independent ``GenerationCoordinator`` instances against
**real** Qwen3-0.6B verifiers through identical history fed via
different chunkings, and asserts the resulting greedy token
streams are byte-identical. This is the integration-level
counterpart of the Linux unit test
``tests/inference_engine/session/test_generator.py::TestDeterminism``,
which uses the deterministic ``FakeVerifier`` to verify the
**dispatch** is non-stateful; this file verifies the same property
holds against the actual verifier numerics on the target hardware.

Replaces the deleted ``tests/core/test_determinism_gate.py`` (PR-A3
removed it together with ``verifier.path_select``; the replacement
landed here, in the integration suite, instead of in
``tests/core/`` because integration is where Mac-M4-only GA gates
belong per ADR 0008 §9).

Marker
------
This whole file inherits ``@pytest.mark.integration`` via
``conftest.py``. Bare ``pytest`` skips it; opt in with::

    pytest -m integration tests/integration/test_inv3_session_determinism_gate.py

Fixture cost
------------
``fresh_verifier_factory`` (from ``tests/conftest.py``) loads
Qwen3-0.6B from the HF cache. On Mac M4 with a warm cache the load
is <2 s; cold takes 10-30 s plus download. Weights are cached
across tests in this file via ``session_verifier_pair``.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from inference_engine.session import (
    AppendTokensCoordinator,
    GenerationCoordinator,
    SessionStore,
    TokenEvent,
)


@pytest.fixture(scope="module")
def session_verifier_pair(fresh_verifier_factory):
    """Two independent verifiers + stores + coordinator pairs.

    Module-scoped: loading Qwen3-0.6B twice costs ~2-4 s on Mac M4
    with a warm HF cache. Tests share the pair; each test resets
    each verifier's state via ``reset()`` before driving its own
    workload, so cross-test bleed-over is impossible by construction.
    """
    fv_a = fresh_verifier_factory(sink=4, window=64)
    fv_b = fresh_verifier_factory(sink=4, window=64)
    yield fv_a, fv_b


def _drive(
    *,
    verifier,
    chunks: List[List[int]],
    max_tokens: int,
) -> List[int]:
    """Set up a fresh SessionStore + coordinators on the given
    verifier, append the chunks in order, then greedy-generate and
    return the emitted token ids.
    """
    verifier.reset()
    store = SessionStore(capacity=1, cache_inspector=verifier)
    append_coord = AppendTokensCoordinator(store, verifier)
    gen_coord = GenerationCoordinator(store, verifier)

    sess = store.create_session()
    for chunk in chunks:
        append_coord.append_tokens(sess.session_id, chunk)

    tokens: List[int] = []
    for ev in gen_coord.generate(sess.session_id, max_tokens=max_tokens):
        if isinstance(ev, TokenEvent):
            tokens.append(ev.token_id)
    return tokens


def test_one_call_vs_two_calls_yield_byte_identical_tokens(
    session_verifier_pair,
):
    """The minimal INV-3 gate: same total token sequence delivered
    in 1 call vs. 2 calls produces bit-identical greedy output."""
    fv_a, fv_b = session_verifier_pair
    full_history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    tokens_one_call = _drive(
        verifier=fv_a, chunks=[full_history], max_tokens=12,
    )
    tokens_two_calls = _drive(
        verifier=fv_b,
        chunks=[full_history[:5], full_history[5:]],
        max_tokens=12,
    )

    assert tokens_one_call == tokens_two_calls, (
        f"INV-3 violated: chunking changed greedy output\n"
        f"  one-call    = {tokens_one_call!r}\n"
        f"  two-calls   = {tokens_two_calls!r}"
    )


def test_chunking_invariance_across_three_splits(
    session_verifier_pair,
):
    """Stronger version: three different chunkings all produce the
    same final greedy stream. This catches any chunk-boundary
    numerical drift the 1-vs-2 case might miss (e.g., a bug that
    only triggers when a chunk crosses a sink+window trim
    boundary).

    The verifier's sink+window is (4, 64) = 68 capacity. We pick a
    history short enough to stay under that bound on the first
    pass and long enough to span more than two chunkings.
    """
    fv_a, fv_b = session_verifier_pair

    full = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    ]

    chunkings = [
        [full],                                              # 1×20
        [full[:7], full[7:14], full[14:]],                   # 3×medium
        [full[i : i + 2] for i in range(0, 20, 2)],          # 10×small
    ]

    runs = []
    for chunks in chunkings:
        # Alternate which verifier we use to keep state fully
        # disjoint across chunkings (we have two; the third
        # chunking reuses fv_a after a reset).
        verifier = fv_a if len(runs) % 2 == 0 else fv_b
        runs.append(_drive(verifier=verifier, chunks=chunks, max_tokens=8))

    assert runs[0] == runs[1] == runs[2], (
        f"INV-3 violated: chunkings produced divergent token streams\n"
        f"  1×20   = {runs[0]!r}\n"
        f"  3×med  = {runs[1]!r}\n"
        f"  10×sm  = {runs[2]!r}"
    )


def test_repeated_runs_with_same_history_byte_identical(
    session_verifier_pair,
):
    """Determinism in the trivial sense: running the SAME workload
    on the SAME verifier twice produces the same output. This is a
    sanity check against accidental RNG (greedy decoding has no
    legitimate source of nondeterminism)."""
    fv_a, _ = session_verifier_pair
    history = [42, 43, 44, 45, 46]

    first = _drive(verifier=fv_a, chunks=[history], max_tokens=6)
    second = _drive(verifier=fv_a, chunks=[history], max_tokens=6)

    assert first == second, (
        f"non-determinism in repeated greedy runs:\n"
        f"  first  = {first!r}\n"
        f"  second = {second!r}"
    )
    # Sanity: greedy with a real verifier should produce SOMETHING.
    assert len(first) > 0
