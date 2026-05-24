"""System-test fixtures.

System tests exercise the full engine — real Qwen3 + dllm-hub
weights, real speculative decoder, real HTTP server, real scheduler
— end to end. They are slow (model load + generation) and require
the HuggingFace cache to be populated. Tests auto-skip when:

  * HF cache lacks ``Qwen/Qwen3-1.7B`` or
    ``dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1``,
  * or ``transformers`` is unavailable (it should always be
    available per requirements.txt, but we defend against unusual
    install states).

There is **no mocking** anywhere in system tests. The whole point
is to verify integration on real components. Tests assert
structural / behavioral invariants (status codes, response shapes,
type predicates, streaming semantics, scheduler concurrency
counts) — never specific token outputs (no overfit).

Session-scoped fixtures keep model-load cost amortized across the
test suite; constructing the verifier is the dominant cost (~3 GB
download + bf16 weight load).
"""

from __future__ import annotations

import pytest

# Try to detect cache before any heavy imports.
try:
    from huggingface_hub import try_to_load_from_cache
    _have_cache_helpers = True
except ImportError:  # pragma: no cover - huggingface_hub is required by deps
    _have_cache_helpers = False


_REQUIRED_REPOS = (
    ("Qwen/Qwen3-1.7B", "config.json"),
    ("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "config.json"),
)


def _hf_cache_ready() -> bool:
    if not _have_cache_helpers:
        return False
    for repo, fname in _REQUIRED_REPOS:
        if not try_to_load_from_cache(repo_id=repo, filename=fname):
            return False
    return True


_skip_if_no_cache = pytest.mark.skipif(
    not _hf_cache_ready(),
    reason=(
        "system tests require HuggingFace cache for "
        "Qwen/Qwen3-1.7B and dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1; "
        "run ./scripts/setup_mac.sh (or setup_cuda.sh) on a host that "
        "has internet access first."
    ),
)


def pytest_collection_modifyitems(config, items):
    """Apply the skip marker to every collected item in tests/system/."""
    for item in items:
        if "tests/system" in str(item.fspath):
            item.add_marker(_skip_if_no_cache)


@pytest.fixture(scope="session")
def real_speculative_engine():
    """Real :class:`SpeculativeEngine` over real Qwen3 + dllm-hub.

    Session-scoped: we pay the model-load cost once. The decoder
    underneath does its own per-request prefill/decode so this is
    safe to share across tests.
    """
    import torch

    from inference_engine.proposer import SparseLogitsProposer
    from inference_engine.server.engine import SpeculativeEngine
    from kv_cache_proposer.proposer import ProposerConfig
    from kv_cache_proposer.speculative import SpeculativeDecoder
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        model_id="Qwen/Qwen3-1.7B",
        dtype=torch.bfloat16, device="cpu",
        sink_size=4, window_size=64,
    )
    proposer = SparseLogitsProposer(proposer_cfg)
    verifier = SinkWindowVerifier(verifier_cfg)
    decoder = SpeculativeDecoder(
        proposer=proposer, verifier=verifier,
        block_size=8, num_diffusion_steps=2,
    )
    return SpeculativeEngine(
        decoder=decoder,
        tokenizer=verifier.tokenizer,
        model_id_label="kakeya-system-test",
    )


@pytest.fixture
def server_app(real_speculative_engine):
    """Fresh FastAPI app per test, but shared engine."""
    from inference_engine.server.app import create_app
    from inference_engine.server.config import ServerConfig

    return create_app(real_speculative_engine, ServerConfig())
