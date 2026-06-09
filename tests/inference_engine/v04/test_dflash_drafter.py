"""Unit tests for inference_engine.v04.dflash_drafter (no network).

Reuses the synthetic-checkpoint pattern from test_dflash_loader.py.
The drafter wrapper layers thin extras on top of the loader (already
unit-tested in test_dflash_loader.py); these tests verify the wrapper
behavior:

  1. from_pretrained loads via the loader and wires tokenizer + extras
  2. Properties (block_size, target_layer_ids, num_layers, model_type)
     surface DFlash config fields correctly
  3. propose_kv runs and returns a KVCapture
  4. require_trained_embed=True raises on random-init embed_tokens
  5. require_trained_embed=False bypasses the check
  6. summary() is JSON-serialisable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from inference_engine.v04 import DFlashDrafter
from inference_engine.v04.dflash_drafter import _resolve_device

# Reuse the exact same synthetic-checkpoint helper as the loader tests
# so the two test files exercise consistent fixtures.
from tests.inference_engine.v04.test_dflash_loader import (
    _tiny_qwen3_config,
    _write_tiny_checkpoint,
)


def _write_drafter_checkpoint(
    tmp_path: Path,
    *,
    use_drafter_prefix: bool = False,
    include_extras: bool = True,
    embed_tokens_trained: bool = True,
) -> Path:
    """Build a synthetic on-disk DFlash-shaped checkpoint AND write
    a tokenizer config so AutoTokenizer.from_pretrained works.

    The loader tests don't exercise the tokenizer load (they call
    load_dflash_drafter, which doesn't load a tokenizer). DFlashDrafter
    DOES load a tokenizer, so we need the tokenizer files alongside.
    """
    d = _write_tiny_checkpoint(
        tmp_path,
        use_drafter_prefix=use_drafter_prefix,
        include_extras=include_extras,
        embed_tokens_trained=embed_tokens_trained,
    )

    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=False,
        cache_dir=str(tmp_path / "_hf_cache"),
    )
    src.save_pretrained(d)

    return d


# ---------------------------------------------------------------------------
# 1. Device resolution
# ---------------------------------------------------------------------------


class TestResolveDevice:

    def test_explicit_pass_through(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("mps") == "mps"

    def test_auto_falls_back_to_cpu_when_no_accelerators(self, monkeypatch):
        import torch as _torch
        monkeypatch.setattr(_torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: False)
        assert _resolve_device("auto") == "cpu"
        assert _resolve_device(None) == "cpu"

    def test_auto_picks_mps_when_available(self, monkeypatch):
        import torch as _torch
        monkeypatch.setattr(_torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        assert _resolve_device("auto") == "mps"


# ---------------------------------------------------------------------------
# 2. from_pretrained loading
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestFromPretrainedHF:
    """Tests that need network access (HF Hub for the tokenizer).

    Marked so CI without network credentials can skip via
    `pytest -m 'not network'`. The test_dflash_loader.py tests work
    fully offline because they don't load tokenizers; this class
    is the network-touching subset.
    """


class TestFromPretrainedLocal:
    """Local-path tests using a synthetic on-disk checkpoint. The
    only network access needed is the one-time tokenizer fetch via
    _write_drafter_checkpoint, which caches under tmp_path/_hf_cache.
    Marked as network-using by the fixture path through transformers.
    """

    @pytest.mark.network
    def test_loads_clean_checkpoint(self, tmp_path):
        d = _write_drafter_checkpoint(tmp_path, embed_tokens_trained=True)
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        assert type(drafter.model).__name__ == "Qwen3ForCausalLM"
        assert drafter.tokenizer is not None
        assert drafter.embed_tokens_trained is True
        assert drafter.source == str(d)

    @pytest.mark.network
    def test_extras_surfaced(self, tmp_path):
        d = _write_drafter_checkpoint(
            tmp_path, include_extras=True, embed_tokens_trained=True,
        )
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        assert drafter.extras is not None
        param_names = [n for n, _ in drafter.extras.named_parameters()]
        assert any("fc__weight" in n for n in param_names)
        assert any("hidden_norm__weight" in n for n in param_names)

    @pytest.mark.network
    def test_no_extras_when_checkpoint_lacks_them(self, tmp_path):
        d = _write_drafter_checkpoint(
            tmp_path, include_extras=False, embed_tokens_trained=True,
        )
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        assert drafter.extras is None

    @pytest.mark.network
    def test_require_trained_embed_default_raises_on_random(self, tmp_path):
        d = _write_drafter_checkpoint(
            tmp_path, embed_tokens_trained=False,
        )
        with pytest.raises(ValueError, match="NOT trained"):
            DFlashDrafter.from_pretrained(
                str(d), dtype=torch.float32, device="cpu",
                trust_remote_code=False,
            )

    @pytest.mark.network
    def test_require_trained_embed_false_bypasses(self, tmp_path):
        d = _write_drafter_checkpoint(
            tmp_path, embed_tokens_trained=False,
        )
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False, require_trained_embed=False,
        )
        assert drafter.embed_tokens_trained is False
        # Architectural warning should still be in the list as evidence
        warning_text = " ".join(drafter.architectural_warnings)
        assert "embed_tokens.weight.var()" in warning_text

    @pytest.mark.network
    def test_model_eval_and_no_grad(self, tmp_path):
        d = _write_drafter_checkpoint(tmp_path, embed_tokens_trained=True)
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        assert drafter.model.training is False
        for p in drafter.model.parameters():
            assert p.requires_grad is False


# ---------------------------------------------------------------------------
# 3. Properties
# ---------------------------------------------------------------------------


class TestProperties:

    @pytest.mark.network
    def test_property_surface(self, tmp_path):
        d = _write_drafter_checkpoint(tmp_path, embed_tokens_trained=True)
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        cfg = _tiny_qwen3_config()
        assert drafter.block_size == cfg["block_size"]
        assert drafter.target_layer_ids == cfg["target_layer_ids"]
        assert drafter.num_layers == cfg["num_hidden_layers"]
        assert drafter.model_type == cfg["model_type"]

    @pytest.mark.network
    def test_summary_is_json_serializable(self, tmp_path):
        d = _write_drafter_checkpoint(tmp_path, embed_tokens_trained=True)
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        summary = drafter.summary()
        roundtripped = json.loads(json.dumps(summary))
        assert roundtripped["kind"] == "dflash_drafter"
        assert roundtripped["model_type"] == "qwen3"
        assert roundtripped["embed_tokens_trained"] is True
        assert roundtripped["extras_attached"] is True

    @pytest.mark.network
    def test_repr_includes_key_fields(self, tmp_path):
        d = _write_drafter_checkpoint(tmp_path, embed_tokens_trained=True)
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        text = repr(drafter)
        assert "DFlashDrafter" in text
        assert "model_type='qwen3'" in text
        assert "block_size=" in text


# ---------------------------------------------------------------------------
# 4. propose_kv (proposer-role primitive)
# ---------------------------------------------------------------------------


class TestProposeKV:

    @pytest.mark.network
    def test_propose_kv_returns_kvcapture(self, tmp_path):
        d = _write_drafter_checkpoint(tmp_path, embed_tokens_trained=True)
        drafter = DFlashDrafter.from_pretrained(
            str(d), dtype=torch.float32, device="cpu",
            trust_remote_code=False,
        )
        from inference_engine.v04 import KVCapture

        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        capture = drafter.propose_kv(input_ids)
        assert isinstance(capture, KVCapture)
        assert capture.num_layers == drafter.num_layers
        assert capture.seq_len == 5
        # KVCapture.keys[i] has shape [B, T, num_kv_heads, head_dim]
        for layer_keys in capture.keys:
            assert layer_keys.shape[0] == 1   # B
            assert layer_keys.shape[1] == 5   # T
