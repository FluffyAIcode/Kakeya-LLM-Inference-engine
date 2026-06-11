"""Gap 2 — factories that wire K/V Restoration into the served paths.

Two entry points:

* :func:`build_restored_speculative_decoder` — wrap a proposer + a
  restored verifier in a :class:`kv_cache_proposer.speculative.SpeculativeDecoder`.
  Pure plumbing over already-tested library code; the restored verifier
  (:class:`~inference_engine.v04.restored_sink_window_verifier.CrossModelRestoredSinkWindowVerifier`)
  implements the ``SinkWindowVerifier`` contract, so the decoder needs no
  changes.

* :func:`load_restored_verifier` — load the Gemma 4 verifier + DFlash
  drafter + trained f_θ from disk and build the restored adapter, ready
  to hand to the gRPC server (``GenerationCoordinator`` AR path) or to
  :func:`build_restored_speculative_decoder`. This is a heavy model
  loader (multi-GB ``from_pretrained``); per the repo convention for
  model loaders (e.g. ``scripts/start_grpc_runtime_server.py``) its body
  is exempt from unit-test coverage and validated by GPU integration runs.
"""

from __future__ import annotations

from typing import Any, Optional

from inference_engine.v04.restored_sink_window_verifier import (
    CrossModelRestoredSinkWindowVerifier,
)


def build_restored_speculative_decoder(
    proposer: Any,
    verifier: CrossModelRestoredSinkWindowVerifier,
    *,
    block_size: int = 16,
    num_diffusion_steps: int = 16,
):
    """Return a :class:`SpeculativeDecoder` over ``proposer`` + restored
    ``verifier`` (the f_θ + S5 K/V-Restoration verifier).

    The restored verifier exposes the full ``SinkWindowVerifier`` API, so
    the speculative accept/reject loop runs unchanged — the only
    difference from the vanilla path is that every verifier forward
    reconstructs the evicted-position K/V (bounded resident cache).
    """
    from kv_cache_proposer.speculative import SpeculativeDecoder

    return SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )


def load_restored_verifier(
    *,
    verifier_id: str,
    drafter_id: str,
    f_theta_dir: str,
    sink_size: int = 4,
    window_size: int = 64,
    s5_exact_full_attn: bool = True,
    device: str = "cpu",
    dtype: Optional[Any] = None,
) -> CrossModelRestoredSinkWindowVerifier:  # pragma: no cover - heavy model loader
    """Load Gemma 4 verifier + DFlash drafter + f_θ and build the restored
    sink+window verifier adapter.

    Coverage-exempt (model-loading plumbing): validated by GPU integration
    runs, mirroring ``scripts/research/k3_integrated_niah_eval.py``.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.gemma4.modeling_gemma4 import (  # type: ignore
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    from inference_engine.v04 import DFlashDrafter, FThetaProjection
    from inference_engine.v04.cross_model_dlm_verifier import (
        CrossModelDLMRestoredVerifier,
        full_attention_layer_indices,
    )

    dev = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    verifier = AutoModelForCausalLM.from_pretrained(
        verifier_id,
        dtype=dtype,
        attn_implementation="eager",
        device_map="auto" if dev.type == "cuda" else None,
    ).eval()
    for p in verifier.parameters():
        p.requires_grad_(False)

    drafter = DFlashDrafter.from_pretrained(drafter_id, dtype=dtype).to(dev).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)

    f_theta = FThetaProjection.from_pretrained(
        f_theta_dir, dtype=torch.float32, device=dev,
    )

    exact_layers = full_attention_layer_indices(verifier) if s5_exact_full_attn else None

    restored = CrossModelDLMRestoredVerifier(
        verifier_model=verifier,
        drafter=drafter,
        f_theta=f_theta,
        sink_size=sink_size,
        window_size=window_size,
        exact_layer_indices=exact_layers,
    )
    return CrossModelRestoredSinkWindowVerifier(
        restored,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=ALL_ATTENTION_FUNCTIONS,
        device=device,
    )
