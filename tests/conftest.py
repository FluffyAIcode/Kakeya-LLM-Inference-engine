"""Shared fixtures for the platform-neutral test suite.

Per the project's no-mock rule, fixtures load real Qwen3 weights from
HuggingFace and reuse them across tests via session scope. The first
session that touches them pays the load cost (~3-5 s on CPU); subsequent
tests pay nothing.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from kv_cache_proposer.proposer import DLMProposer, ProposerConfig
from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig
from kv_cache_proposer.baseline import BaselineDecoder, BaselineConfig


@pytest.fixture(scope="session")
def proposer_session() -> DLMProposer:
    return DLMProposer(ProposerConfig(dtype=torch.bfloat16, device="cpu"))


@pytest.fixture(scope="session")
def baseline_decoder() -> BaselineDecoder:
    return BaselineDecoder(BaselineConfig(dtype=torch.bfloat16, device="cpu"))


def _build_verifier(sink: int = 4, window: int = 64) -> SinkWindowVerifier:
    return SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16,
            device="cpu",
            sink_size=sink,
            window_size=window,
        )
    )


@pytest.fixture(scope="session")
def verifier_session() -> SinkWindowVerifier:
    return _build_verifier()


@pytest.fixture
def fresh_verifier_factory():
    """Each test that needs a clean verifier with custom (sink, window)
    can call this. Returns a factory; the verifier loads weights once
    per call (cheap on CPU with cached HF blob)."""
    return _build_verifier


@pytest.fixture
def short_chat_messages() -> List[dict]:
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly 'OK'."},
    ]
