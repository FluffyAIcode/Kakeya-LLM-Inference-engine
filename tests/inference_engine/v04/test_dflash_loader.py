"""Unit tests for inference_engine.v04.dflash_loader (no network).

These tests exercise the loader's three orthogonal concerns:

  1. Key-remap heuristics — does ``_propose_key_remap`` correctly
     map prefixed checkpoint keys to Qwen3-expected keys?
  2. Extras classification — does ``inspect_dflash_checkpoint``
     correctly identify ``fc.*`` and ``hidden_norm.*`` keys as
     DFlash extras?
  3. ``embed_tokens_trained`` decision — does
     ``load_dflash_drafter``'s post-load verification correctly
     flag random-init vs trained embeddings?

We use synthetic on-disk safetensors checkpoints + a real
Qwen3 config to keep the tests deterministic and offline. The
config is small (4 layers, 64 hidden) so each test runs in
< 1 s on a CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

torch = pytest.importorskip("torch")

# Skip the whole module if transformers isn't installed at all
# (keeps the suite usable in minimal Python envs).
transformers = pytest.importorskip("transformers")

from inference_engine.v04 import dflash_loader  # noqa: E402
from inference_engine.v04.dflash_loader import (  # noqa: E402
    EMBED_TOKENS_TRAINED_VAR_THRESHOLD,
    _looks_like_local_path,
    _propose_key_remap,
    _resolve_local_dir,
    inspect_dflash_checkpoint,
    load_dflash_drafter,
)


# ---------------------------------------------------------------------------
# Synthetic Qwen3 config: tiny so tests are fast.
# ---------------------------------------------------------------------------


def _tiny_qwen3_config() -> Dict:
    """A minimal valid Qwen3 config with ``model_type: qwen3``.

    Includes the DFlash-specific config keys so we can verify
    ``inspect_dflash_checkpoint`` strips them before consulting
    transformers (those keys are not part of the Qwen3 schema and
    transformers 4.x and 5.x both reject unknown kwargs at config
    construction).
    """
    return {
        "model_type": "qwen3",
        "architectures": ["DFlashDraftModel"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "head_dim": 16,
        "block_size": 16,
        "target_layer_ids": [0, 1],
        "dflash_config": {
            "mask_token_id": 4,
            "target_layer_ids": [0, 1],
        },
    }


def _write_tiny_checkpoint(
    tmp_path: Path,
    *,
    use_drafter_prefix: bool,
    include_extras: bool,
    embed_tokens_trained: bool,
) -> Path:
    """Build a synthetic on-disk DFlash-shaped checkpoint.

    We construct a real (small) Qwen3ForCausalLM, take its
    state_dict, optionally rename keys with a ``drafter.`` prefix,
    optionally add DFlash extras (``fc.weight``, ``hidden_norm.weight``),
    and optionally overwrite ``embed_tokens.weight`` with a near-zero
    random init (variance ~ 1e-6) to simulate "newly initialised".

    Saves the result via safetensors at ``tmp_path / model.safetensors``,
    plus ``config.json``.
    """
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _tiny_qwen3_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    cfg_for_model = dict(cfg)
    cfg_for_model.pop("dflash_config", None)
    cfg_for_model.pop("target_layer_ids", None)
    cfg_for_model.pop("block_size", None)
    model_type = cfg_for_model.pop("model_type")
    hf_cfg = AutoConfig.for_model(model_type, **cfg_for_model)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(hf_cfg)
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)

    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}

    embed_key = "model.embed_tokens.weight"
    if embed_tokens_trained:
        # Real trained embeddings on production-scale models have
        # per-token variance around 1e-3 to 1e-2 (verified spot
        # checks on Gemma 3, Qwen 3, Llama 3). std=0.05 gives
        # variance ~ 2.5e-3, comfortably above the loader's 1e-3
        # "trained" threshold.
        if embed_key in state:
            state[embed_key] = (torch.randn_like(state[embed_key]) * 0.05).contiguous()
    else:
        if embed_key in state:
            state[embed_key] = (torch.randn_like(state[embed_key]) * 1e-4).contiguous()

    if use_drafter_prefix:
        state = {f"drafter.{k}": v for k, v in state.items()}

    if include_extras:
        d_model = cfg["hidden_size"]
        state["fc.weight"] = torch.randn(d_model, d_model * 6) * 0.02
        state["fc.bias"] = torch.zeros(d_model)
        state["hidden_norm.weight"] = torch.ones(d_model)

    save_file(state, str(tmp_path / "model.safetensors"))
    return tmp_path


# ---------------------------------------------------------------------------
# 1. _propose_key_remap unit tests (no checkpoint, pure logic)
# ---------------------------------------------------------------------------


class TestProposeKeyRemap:

    def test_identity_remap(self):
        ckpt = ["model.embed_tokens.weight", "lm_head.weight"]
        expected = ["model.embed_tokens.weight", "lm_head.weight"]
        remap, unmapped, extras, fc, hn = _propose_key_remap(ckpt, expected)
        assert remap == {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
        }
        assert unmapped == []
        assert extras == []

    def test_drafter_prefix_strip(self):
        ckpt = ["drafter.model.embed_tokens.weight", "drafter.lm_head.weight"]
        expected = ["model.embed_tokens.weight", "lm_head.weight"]
        remap, unmapped, extras, fc, hn = _propose_key_remap(ckpt, expected)
        assert remap == {
            "drafter.model.embed_tokens.weight": "model.embed_tokens.weight",
            "drafter.lm_head.weight": "lm_head.weight",
        }
        assert unmapped == []
        assert extras == []

    def test_extras_classification(self):
        ckpt = ["model.embed_tokens.weight", "fc.weight", "hidden_norm.weight"]
        expected = ["model.embed_tokens.weight"]
        remap, unmapped, extras, fc, hn = _propose_key_remap(ckpt, expected)
        assert remap == {"model.embed_tokens.weight": "model.embed_tokens.weight"}
        assert unmapped == []
        assert sorted(extras) == ["fc.weight", "hidden_norm.weight"]
        assert fc == ["fc.weight"]
        assert hn == ["hidden_norm.weight"]

    def test_unmapped_qwen3_keys_are_reported(self):
        ckpt = ["model.embed_tokens.weight"]
        expected = ["model.embed_tokens.weight", "lm_head.weight"]
        remap, unmapped, extras, fc, hn = _propose_key_remap(ckpt, expected)
        assert "lm_head.weight" in unmapped

    def test_no_double_assignment(self):
        ckpt = ["model.embed_tokens.weight", "drafter.model.embed_tokens.weight"]
        expected = ["model.embed_tokens.weight"]
        remap, unmapped, extras, fc, hn = _propose_key_remap(ckpt, expected)
        assert len(remap) == 1
        target = next(iter(remap.values()))
        assert target == "model.embed_tokens.weight"
        assert len(extras) == 1


# ---------------------------------------------------------------------------
# 2. inspect_dflash_checkpoint integration tests (synthetic on-disk checkpoint)
# ---------------------------------------------------------------------------


class TestInspectDFlashCheckpoint:

    def test_clean_checkpoint(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=True, embed_tokens_trained=True,
        )
        result = inspect_dflash_checkpoint(str(d))
        assert result.config["model_type"] == "qwen3"
        assert "model.embed_tokens.weight" in result.checkpoint_keys
        assert "model.embed_tokens.weight" in result.qwen3_expected_keys
        assert "model.embed_tokens.weight" in result.key_remap
        assert "fc.weight" in result.fc_keys
        assert "hidden_norm.weight" in result.hidden_norm_keys
        assert result.qwen3_unmapped == []

    def test_drafter_prefix_checkpoint(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=True,
            include_extras=True, embed_tokens_trained=True,
        )
        result = inspect_dflash_checkpoint(str(d))
        assert "drafter.model.embed_tokens.weight" in result.checkpoint_keys
        assert "model.embed_tokens.weight" in result.qwen3_expected_keys
        assert (
            result.key_remap.get("drafter.model.embed_tokens.weight")
            == "model.embed_tokens.weight"
        )
        assert result.qwen3_unmapped == []
        assert "fc.weight" in result.fc_keys

    def test_no_extras_warning(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=False, embed_tokens_trained=True,
        )
        result = inspect_dflash_checkpoint(str(d))
        msgs = " ".join(result.warnings)
        assert "no `fc.*` keys" in msgs
        assert "no `hidden_norm.*` keys" in msgs

    def test_dflash_config_keys_stripped(self, tmp_path):
        """The DFlash-only config keys (block_size, target_layer_ids,
        dflash_config) must not be passed to AutoConfig — that would
        raise on transformers 4.x AND 5.x because Qwen3Config doesn't
        accept them. ``_enumerate_qwen3_expected_keys`` strips them
        before constructing the config.
        """
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=True, embed_tokens_trained=True,
        )
        result = inspect_dflash_checkpoint(str(d))
        assert "block_size" in result.config
        assert "dflash_config" in result.config
        assert len(result.qwen3_expected_keys) > 0


# ---------------------------------------------------------------------------
# 3. load_dflash_drafter integration tests
# ---------------------------------------------------------------------------


class TestLoadDFlashDrafter:

    def test_clean_load_trained_embed(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=True, embed_tokens_trained=True,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        assert result.expected_class_name == "Qwen3ForCausalLM"
        assert result.embed_tokens_trained is True
        assert result.embed_tokens_var > EMBED_TOKENS_TRAINED_VAR_THRESHOLD
        assert result.extras is not None
        names = [n for n, _ in result.extras.named_parameters()]
        assert any("fc__weight" in n for n in names)
        assert any("hidden_norm__weight" in n for n in names)

    def test_drafter_prefix_load(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=True,
            include_extras=True, embed_tokens_trained=True,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        assert result.expected_class_name == "Qwen3ForCausalLM"
        assert result.embed_tokens_trained is True
        embed = result.model.get_input_embeddings()
        assert embed.weight.shape[0] == 256

    def test_random_init_embed_flagged(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=True, embed_tokens_trained=False,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        assert result.embed_tokens_trained is False
        assert result.embed_tokens_var <= EMBED_TOKENS_TRAINED_VAR_THRESHOLD
        warning_text = " ".join(result.architectural_warnings)
        assert "embed_tokens.weight.var()" in warning_text
        assert "Block C f_θ training MUST NOT proceed" in warning_text

    def test_extras_attached_with_correct_count(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=True, embed_tokens_trained=True,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        params = list(result.extras.named_parameters())
        # 2 fc params (weight + bias) + 1 hidden_norm param = 3
        assert len(params) == 3

    def test_no_extras_when_checkpoint_has_none(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=False, embed_tokens_trained=True,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        assert result.extras is None

    def test_inspection_attached_to_load_result(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=True,
            include_extras=True, embed_tokens_trained=True,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        assert result.inspection is not None
        assert isinstance(result.inspection.key_remap, dict)
        # Extras keep their on-disk names (no drafter. prefix), since
        # _write_tiny_checkpoint deliberately adds extras OUTSIDE the
        # prefix loop — mirrors the real DFlash repo where the extras
        # are at the top level even though the qwen3 weights may carry
        # a prefix from the upstream training pipeline.
        assert "fc.weight" in result.inspection.fc_keys
        assert "hidden_norm.weight" in result.inspection.hidden_norm_keys

    def test_dtype_propagation(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path, use_drafter_prefix=False,
            include_extras=False, embed_tokens_trained=True,
        )
        result = load_dflash_drafter(str(d), dtype=torch.float32)
        embed = result.model.get_input_embeddings()
        assert embed.weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. CLI inspect mode (round-trip JSON)
# ---------------------------------------------------------------------------


class TestCLIInspect:

    def test_cli_inspect_writes_json(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path / "checkpoint", use_drafter_prefix=False,
            include_extras=True, embed_tokens_trained=True,
        )
        out = tmp_path / "inspection.json"
        rc = dflash_loader._cli_main(
            ["inspect", str(d), "--output", str(out)]
        )
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["repo_or_path"] == str(d)
        assert "key_remap" in payload
        assert "fc.weight" in payload["fc_keys"]
        assert "hidden_norm.weight" in payload["hidden_norm_keys"]


# ---------------------------------------------------------------------------
# 5. Local-path heuristic + fail-fast resolver
#    (Regression for 2026-06-09 user-side bug: DRAFTER_ID=models/dflash-
#    kakeya-baseline silently fell through to HF Hub fetch when the local
#    LFS-pulled directory wasn't present, and HF returned 404 with a
#    misleading error message far from the root cause.)
# ---------------------------------------------------------------------------


class TestLooksLikeLocalPath:

    def test_models_prefix(self):
        assert _looks_like_local_path("models/dflash-kakeya-baseline") is True
        assert _looks_like_local_path("models/foo") is True

    def test_relative_prefixes(self):
        assert _looks_like_local_path("./models/foo") is True
        assert _looks_like_local_path("../models/foo") is True

    def test_absolute_prefix(self):
        assert _looks_like_local_path("/Users/me/models/foo") is True
        assert _looks_like_local_path("/tmp/checkpoint") is True

    def test_hf_repo_id_does_not_match(self):
        assert _looks_like_local_path("z-lab/gemma-4-26B-A4B-it-DFlash") is False
        assert _looks_like_local_path("google/gemma-3-1b-it") is False
        assert _looks_like_local_path("FakeRockert543/gemma-4-26b-a4b-it-MLX-4bit") is False
        assert _looks_like_local_path("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1") is False


class TestResolveLocalDirFailFast:
    """Regression for 2026-06-09 user-side bug.

    User report: ``DRAFTER_ID=models/dflash-kakeya-baseline`` was treated
    as an HF repo id and the script tried to download from HF, which
    returned 404. Pre-fix _resolve_local_dir silently fell through to
    huggingface_hub.snapshot_download for any non-existent local path.
    Post-fix it raises FileNotFoundError with an actionable message.
    """

    def test_local_path_that_exists_is_returned(self, tmp_path):
        d = _write_tiny_checkpoint(
            tmp_path / "models" / "dflash-kakeya-baseline",
            use_drafter_prefix=False, include_extras=True,
            embed_tokens_trained=True,
        )
        result = _resolve_local_dir(str(d), {})
        assert result == d

    def test_local_path_that_does_not_exist_raises(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            _resolve_local_dir("models/dflash-kakeya-baseline", {})
        msg = str(excinfo.value)
        assert "does not exist on disk" in msg
        assert "git lfs pull" in msg
        assert "current working directory" in msg.lower()

    def test_relative_path_that_does_not_exist_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_local_dir("./models/missing", {})
        with pytest.raises(FileNotFoundError):
            _resolve_local_dir("../models/missing", {})

    def test_absolute_path_that_does_not_exist_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_local_dir("/tmp/definitely-not-a-real-checkpoint", {})

    def test_hf_repo_id_falls_through_to_hf_hub(self, monkeypatch):
        """For a non-local-looking input (HF repo id format), the resolver
        should still call huggingface_hub.snapshot_download — the fail-fast
        path is local-path-specific."""
        called = {}

        def fake_snapshot_download(**kwargs):
            called["kwargs"] = kwargs
            return "/tmp/fake-cached-snapshot"

        import huggingface_hub
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download,
        )
        result = _resolve_local_dir("z-lab/gemma-4-26B-A4B-it-DFlash", {})
        assert called["kwargs"]["repo_id"] == "z-lab/gemma-4-26B-A4B-it-DFlash"
        assert str(result) == "/tmp/fake-cached-snapshot"
