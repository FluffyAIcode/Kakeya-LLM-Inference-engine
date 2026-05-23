"""Shared fixtures for the inference_engine test suite."""

from __future__ import annotations

import pytest
import torch

from inference_engine.proposer import SparseLogitsProposer
from kv_cache_proposer.proposer import ProposerConfig


@pytest.fixture(scope="session")
def sparse_proposer() -> SparseLogitsProposer:
    return SparseLogitsProposer(
        ProposerConfig(dtype=torch.bfloat16, device="cpu")
    )
