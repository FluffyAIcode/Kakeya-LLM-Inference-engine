"""Real-MLX gate: imported prefill state must preserve continuation logits."""
from __future__ import annotations

import os

import pytest


def test_real_mlx_prefill_snapshot_preserves_continuation_logits():
    pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")

    from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
    from inference_engine.distributed.capability import (
        CacheCompatibility,
        CompressionCodec,
    )
    from inference_engine.distributed.prefill_cache import PrefixCacheStore
    from inference_engine.distributed.prefill_cache_runtime import (
        DistributedPrefillCacheHook,
    )
    from kv_cache_proposer.verifier import VerifierConfig

    model_path = os.environ["KAKEYA_MAC_VERIFIER_PATH"]
    verifier = MLXSinkWindowVerifier(VerifierConfig(
        model_id=model_path,
        dtype=torch.bfloat16,
        device="cpu",
        sink_size=4,
        window_size=64,
    ))
    prompt = (
        "Kakeya distributed prefill equivalence test. "
        "The imported cache must produce exactly the same continuation logits. "
    ) * 8
    token_ids = verifier.tokenizer.encode(prompt)
    compatibility = CacheCompatibility(
        model_id=os.environ.get("KAKEYA_CACHE_MODEL_ID", model_path),
        model_revision=os.environ.get("KAKEYA_MODEL_REVISION", ""),
        tokenizer_revision=os.environ.get("KAKEYA_TOKENIZER_REVISION", ""),
        cache_format_version="kakeya-prefill-v2-zlib",
        quantization="4bit-mlx",
        # This is a single-verifier round-trip gate; production head/workers
        # derive and compare the real geometry hash separately. Keep the
        # namespace stable without requiring another machine-local env var.
        layer_geometry_hash=os.environ.get(
            "KAKEYA_LAYER_GEOMETRY_HASH",
            "integration-single-verifier",
        ),
        kv_dtype="bfloat16",
        block_size_tokens=64,
        tenant_namespace="integration",
    )
    store = PrefixCacheStore(
        compatibility,
        max_bytes=1 << 30,
        node_id="integration-head",
    )
    hook = DistributedPrefillCacheHook(
        store,
        compression=CompressionCodec.ZLIB,
    )
    try:
        assert hook.prepare(verifier, token_ids) == 0
        baseline_logits = verifier.next_token_logits.clone()
        baseline_argmax = int(torch.argmax(baseline_logits).item())
        baseline_tokens = list(verifier.cached_token_sequence)

        reused = hook.prepare(verifier, token_ids)
        assert reused == len(token_ids)
        assert int(torch.argmax(verifier.next_token_logits).item()) == baseline_argmax
        assert torch.equal(verifier.next_token_logits, baseline_logits)
        assert verifier.cached_token_sequence == baseline_tokens

        # One real decode step must remain bit-identical after re-import.
        next_token = baseline_argmax
        imported_row = verifier.forward_block([next_token])[-1].clone()
        hook.prepare(verifier, token_ids)
        local_row = verifier.forward_block([next_token])[-1].clone()
        assert torch.equal(imported_row, local_row)
    finally:
        hook.close()

