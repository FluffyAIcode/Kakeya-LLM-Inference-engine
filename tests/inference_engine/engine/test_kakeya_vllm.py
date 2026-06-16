"""Unit tests for the KIE-v0.5 (CUDA) entrypoint config — torch/vllm-free.

These cover the pure config surface of :mod:`inference_engine.engine.kakeya_vllm`
(the Kakeya bounded-window injection + the vLLM kwargs builder). The ``vllm``
import is deferred to :class:`KakeyaVLLM` construction, so importing and testing
the config layer requires no GPU / vllm / torch.
"""

from __future__ import annotations

import pytest

from inference_engine.engine.kakeya_vllm import (
    DEFAULT_SINK,
    DEFAULT_WINDOW,
    KakeyaVLLMConfig,
    kakeya_hf_overrides,
    kakeya_window_total,
)


def test_window_total_default_is_s5():
    # Kakeya restored-S5 default: 4 sink + 64 window = 68.
    assert kakeya_window_total() == DEFAULT_SINK + DEFAULT_WINDOW == 68


@pytest.mark.parametrize("sink,window,expected", [(4, 64, 68), (0, 32, 32), (8, 256, 264)])
def test_window_total_param(sink, window, expected):
    assert kakeya_window_total(sink, window) == expected


@pytest.mark.parametrize("sink,window", [(-1, 64), (4, 0), (4, -5)])
def test_window_total_rejects_bad_args(sink, window):
    with pytest.raises(ValueError):
        kakeya_window_total(sink, window)


def test_hf_overrides_flat_by_default():
    # Default is flat (top-level only) — safe for text-only models (Qwen/Llama);
    # injecting text_config into a model that has none breaks vLLM.
    ov = kakeya_hf_overrides(4, 64)
    assert ov == {"sliding_window": 68}


def test_hf_overrides_nested_opt_in():
    # gemma-4 (multimodal) nests sliding_window under text_config; opt in.
    ov = kakeya_hf_overrides(4, 64, nested_text_config=True)
    assert ov == {"sliding_window": 68, "text_config": {"sliding_window": 68}}


def test_config_to_vllm_kwargs_keeps_graphs_and_injects_window():
    # nest_text_config forced True simulates the resolved gemma-4 path.
    cfg = KakeyaVLLMConfig(
        model="google/gemma-4-26b-a4b-it", max_model_len=16384, nest_text_config=True
    )
    kw = cfg.to_vllm_kwargs()
    # CUDA graphs (and thus fused-MoE graph capture) MUST stay on.
    assert kw["enforce_eager"] is False
    # Kakeya window injected as hf_overrides (nested for gemma-4).
    assert kw["hf_overrides"] == {"sliding_window": 68, "text_config": {"sliding_window": 68}}
    assert kw["model"] == "google/gemma-4-26b-a4b-it"
    assert kw["max_model_len"] == 16384
    # No quantization key when None.
    assert "quantization" not in kw


def test_config_to_vllm_kwargs_flat_for_text_only():
    # Auto/None resolves to flat in to_vllm_kwargs (text-only safe default).
    cfg = KakeyaVLLMConfig(model="Qwen/Qwen3-4B", max_model_len=2048)
    kw = cfg.to_vllm_kwargs()
    assert kw["hf_overrides"] == {"sliding_window": 68}


def test_config_passes_through_extra_kwargs_and_quant():
    cfg = KakeyaVLLMConfig(
        model="m", quantization="fp8", extra_vllm_kwargs={"tensor_parallel_size": 2}
    )
    kw = cfg.to_vllm_kwargs()
    assert kw["quantization"] == "fp8"
    assert kw["tensor_parallel_size"] == 2


def test_window_total_property_matches_helper():
    cfg = KakeyaVLLMConfig(model="m", sink=4, window=64)
    assert cfg.window_total == kakeya_window_total(4, 64) == 68
