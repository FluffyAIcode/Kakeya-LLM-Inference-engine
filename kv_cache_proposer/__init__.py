"""KV-cache-saving speculative decoding using a DLM Proposer + AR Verifier.

Components:
  * DLMProposer  - wraps `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` and proposes
                   a block of L tokens via masked-diffusion denoising.
  * SinkWindowVerifier - wraps an AR Qwen3 model whose KV cache is bounded to
                   `sink_size + window_size` entries, evicting older K/V
                   tensors in place after every step.
  * speculative_generate - greedy speculative decoding loop that is
                   distribution-equivalent to the verifier's own greedy decode.
  * baseline_generate - reference greedy AR decode with full KV cache, used
                   to prove output equivalence and measure the KV baseline.
"""

from .proposer import DLMProposer
from .verifier import SinkWindowVerifier
from .speculative import SpeculativeDecoder, SpeculativeRunResult
from .baseline import BaselineDecoder, BaselineRunResult
from .metrics import (
    cache_kv_bytes,
    cache_token_count,
    measure_proposer_weight_bytes,
    NBTReport,
)

__all__ = [
    "DLMProposer",
    "SinkWindowVerifier",
    "SpeculativeDecoder",
    "SpeculativeRunResult",
    "BaselineDecoder",
    "BaselineRunResult",
    "cache_kv_bytes",
    "cache_token_count",
    "measure_proposer_weight_bytes",
    "NBTReport",
]
