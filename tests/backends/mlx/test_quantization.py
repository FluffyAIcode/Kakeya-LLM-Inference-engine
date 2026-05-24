"""Unit tests for :mod:`inference_engine.backends.mlx.quantization`.

These tests build small synthetic parameter trees that match the shape
of an mlx_lm model's ``parameters()`` output (nested dicts of
``mx.array``). They are skipped on non-Apple-Silicon hosts at import
time via ``pytest.importorskip``; coverage is enforced only when the
runner is invoked with ``--backend=mlx``.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from inference_engine.backends.mlx.quantization import (  # noqa: E402
    QuantizationInfo,
    _is_quantized_linear_dict,
    _read_args_quantization,
    detect_quantization,
)


# ---------------------------------------------------------------------------
# Synthetic model builders
# ---------------------------------------------------------------------------


class _FakeArgs:
    """Stand-in for an mlx_lm ``ModelArgs``-style config."""

    def __init__(self, quantization=None):
        self.quantization = quantization


class _FakeModel:
    """Minimal duck-type of an mlx_lm model.

    Stores a parameter tree (nested dicts/lists of ``mx.array``)
    returned by :meth:`parameters` and an optional ``args`` carrying
    quantization config. Mirrors only what
    :func:`detect_quantization` actually consumes; nothing else.
    """

    def __init__(self, params, args=None):
        self._params = params
        self.args = args

    def parameters(self):
        return self._params


def _make_quantized_layer(out_dim: int, in_dim: int, *, bits: int, group_size: int):
    """Build a (weight, scales, biases) trio matching mlx_lm's QuantizedLinear.

    ``weight`` shape: ``[out_dim, in_dim * bits / 32]`` packed in uint32.
    ``scales`` / ``biases`` shape: ``[out_dim, in_dim / group_size]``.

    Values are zero-filled — detection only inspects shape / dtype, not
    contents.
    """
    if (in_dim * bits) % 32 != 0:
        raise ValueError("test fixture: in_dim*bits must be multiple of 32")
    if in_dim % group_size != 0:
        raise ValueError("test fixture: in_dim must be multiple of group_size")
    packed_cols = in_dim * bits // 32
    n_groups = in_dim // group_size
    return {
        "weight": mx.zeros((out_dim, packed_cols), dtype=mx.uint32),
        "scales": mx.zeros((out_dim, n_groups), dtype=mx.bfloat16),
        "biases": mx.zeros((out_dim, n_groups), dtype=mx.bfloat16),
    }


def _make_full_precision_layer(out_dim: int, in_dim: int, dtype=None):
    """Build a regular dense linear weight tensor."""
    dtype = dtype or mx.bfloat16
    return {"weight": mx.zeros((out_dim, in_dim), dtype=dtype)}


# ---------------------------------------------------------------------------
# Detection: full precision
# ---------------------------------------------------------------------------


def test_detect_unquantized_model_reports_no_quantization():
    params = {
        "embed": _make_full_precision_layer(128, 16),
        "layers": [
            {"q_proj": _make_full_precision_layer(16, 16),
             "norm": {"weight": mx.zeros((16,), dtype=mx.bfloat16)}}
        ],
        "lm_head": _make_full_precision_layer(128, 16),
    }
    info = detect_quantization(_FakeModel(params))
    assert info.is_quantized is False
    assert info.bits is None
    assert info.group_size is None
    assert info.quantized_weight_bytes == 0
    assert info.full_precision_weight_bytes > 0
    assert info.total_weight_bytes == info.full_precision_weight_bytes
    assert info.quantized_param_count == 0
    assert info.full_precision_param_count > 0


def test_detect_unquantized_render_short_says_unquantized():
    params = {"embed": _make_full_precision_layer(128, 16)}
    info = detect_quantization(_FakeModel(params))
    rendered = info.render_short()
    assert "unquantized" in rendered
    assert "GB" in rendered


# ---------------------------------------------------------------------------
# Detection: quantized via args
# ---------------------------------------------------------------------------


def test_detect_quantized_with_args_metadata_reports_correct_bits():
    params = {
        "embed": _make_full_precision_layer(128, 64),
        "layers": [
            {"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}
        ],
        "lm_head": _make_full_precision_layer(128, 64),
    }
    args = _FakeArgs(quantization={"bits": 4, "group_size": 64})
    info = detect_quantization(_FakeModel(params, args=args))
    assert info.is_quantized is True
    assert info.bits == 4
    assert info.group_size == 64
    assert info.quantized_weight_bytes > 0
    assert info.full_precision_weight_bytes > 0
    assert info.quantized_param_count == 64 * 64
    assert info.full_precision_param_count == 128 * 64 + 128 * 64


def test_detect_quantized_render_short_includes_bits_and_group():
    params = {
        "layers": [{"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}]
    }
    args = _FakeArgs(quantization={"bits": 4, "group_size": 64})
    info = detect_quantization(_FakeModel(params, args=args))
    rendered = info.render_short()
    assert "4-bit" in rendered
    assert "group=64" in rendered
    assert "effective=" in rendered


def test_effective_bits_per_param_in_expected_range_for_4bit():
    # Make embeddings small so quantized portion dominates and we see
    # something close to the 4-bit + scales overhead.
    params = {
        "layers": [
            {"q_proj": _make_quantized_layer(1024, 1024, bits=4, group_size=64)}
            for _ in range(16)
        ],
        "embed": _make_full_precision_layer(8, 1024),
        "lm_head": _make_full_precision_layer(8, 1024),
    }
    args = _FakeArgs(quantization={"bits": 4, "group_size": 64})
    info = detect_quantization(_FakeModel(params, args=args))
    # 4 bits weight + (16 + 16 = 32 bits / 64-element group) ~= 4.5 bits per param
    assert 4.4 < info.effective_bits_per_param < 4.7


# ---------------------------------------------------------------------------
# Detection: quantized via inference (args missing or malformed)
# ---------------------------------------------------------------------------


def test_detect_quantized_inferred_when_args_missing():
    params = {"layers": [{"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}]}
    info = detect_quantization(_FakeModel(params, args=None))
    assert info.is_quantized is True
    assert info.bits == 4
    assert info.group_size == 64


def test_detect_quantized_inferred_when_args_lacks_quantization():
    params = {"layers": [{"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}]}
    args = _FakeArgs(quantization=None)
    info = detect_quantization(_FakeModel(params, args=args))
    assert info.is_quantized is True
    assert info.bits == 4


def test_detect_quantized_inferred_when_args_quantization_not_dict():
    params = {"layers": [{"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}]}
    args = _FakeArgs(quantization="not a dict")
    info = detect_quantization(_FakeModel(params, args=args))
    assert info.is_quantized is True
    assert info.bits == 4


def test_detect_quantized_inferred_when_args_quantization_lacks_bits():
    params = {"layers": [{"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}]}
    args = _FakeArgs(quantization={"group_size": 64})
    info = detect_quantization(_FakeModel(params, args=args))
    assert info.is_quantized is True
    assert info.bits == 4


def test_detect_quantized_inferred_when_args_quantization_has_non_int_bits():
    params = {"layers": [{"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}]}
    args = _FakeArgs(quantization={"bits": "four", "group_size": 64})
    info = detect_quantization(_FakeModel(params, args=args))
    assert info.is_quantized is True
    assert info.bits == 4


def test_inference_handles_8bit_group64():
    params = {"layers": [{"q_proj": _make_quantized_layer(64, 64, bits=8, group_size=64)}]}
    info = detect_quantization(_FakeModel(params, args=None))
    assert info.bits == 8
    assert info.group_size == 64


# ---------------------------------------------------------------------------
# Tree traversal: lists, nested dicts, mixed quantization
# ---------------------------------------------------------------------------


def test_walk_traverses_lists_of_layer_dicts():
    layers = [
        {"q_proj": _make_quantized_layer(64, 64, bits=4, group_size=64)}
        for _ in range(3)
    ]
    params = {"layers": layers}
    info = detect_quantization(_FakeModel(params, args=_FakeArgs({"bits": 4, "group_size": 64})))
    # 3x the per-layer quantized bytes
    one_layer = detect_quantization(
        _FakeModel({"layers": [layers[0]]}, args=_FakeArgs({"bits": 4, "group_size": 64}))
    )
    assert info.quantized_weight_bytes == 3 * one_layer.quantized_weight_bytes


def test_walk_handles_top_level_array_leaf():
    # Unusual but tolerated: an mx.array directly at the top of the tree.
    info = detect_quantization(_FakeModel(mx.zeros((4, 4), dtype=mx.bfloat16)))
    assert info.is_quantized is False
    assert info.full_precision_weight_bytes == 4 * 4 * 2  # bfloat16 = 2 bytes


def test_walk_handles_nested_tuples():
    params = (
        _make_full_precision_layer(8, 8),
        (_make_quantized_layer(64, 64, bits=4, group_size=64),),
    )
    info = detect_quantization(_FakeModel(params, args=_FakeArgs({"bits": 4, "group_size": 64})))
    assert info.is_quantized is True
    assert info.full_precision_weight_bytes == 8 * 8 * 2


def test_walk_ignores_non_tensor_metadata_in_dict():
    params = {
        "name": "qwen3-1.7b",
        "version": 7,
        "embed": _make_full_precision_layer(8, 8),
    }
    info = detect_quantization(_FakeModel(params))
    assert info.is_quantized is False
    assert info.full_precision_weight_bytes == 8 * 8 * 2


# ---------------------------------------------------------------------------
# Quantized-linear dict identification
# ---------------------------------------------------------------------------


def test_is_quantized_linear_dict_true_on_well_formed_trio():
    layer = _make_quantized_layer(64, 64, bits=4, group_size=64)
    assert _is_quantized_linear_dict(layer) is True


def test_is_quantized_linear_dict_false_when_weight_is_bfloat16():
    layer = {
        "weight": mx.zeros((64, 64), dtype=mx.bfloat16),
        "scales": mx.zeros((64, 1), dtype=mx.bfloat16),
        "biases": mx.zeros((64, 1), dtype=mx.bfloat16),
    }
    assert _is_quantized_linear_dict(layer) is False


def test_is_quantized_linear_dict_false_when_scales_missing():
    layer = {
        "weight": mx.zeros((64, 8), dtype=mx.uint32),
        "biases": mx.zeros((64, 1), dtype=mx.bfloat16),
    }
    assert _is_quantized_linear_dict(layer) is False


def test_is_quantized_linear_dict_false_when_scales_not_array():
    layer = {
        "weight": mx.zeros((64, 8), dtype=mx.uint32),
        "scales": "not a tensor",
        "biases": mx.zeros((64, 1), dtype=mx.bfloat16),
    }
    assert _is_quantized_linear_dict(layer) is False


# ---------------------------------------------------------------------------
# Args quantization extraction
# ---------------------------------------------------------------------------


def test_read_args_quantization_returns_none_for_missing_args():
    class _NoArgs:
        pass

    assert _read_args_quantization(_NoArgs()) == (None, None)


def test_read_args_quantization_returns_none_for_args_quantization_none():
    args = _FakeArgs(quantization=None)

    class _Wrapper:
        def __init__(self, a):
            self.args = a

    assert _read_args_quantization(_Wrapper(args)) == (None, None)


def test_read_args_quantization_returns_pair_when_present():
    args = _FakeArgs(quantization={"bits": 4, "group_size": 64})

    class _Wrapper:
        def __init__(self, a):
            self.args = a

    assert _read_args_quantization(_Wrapper(args)) == (4, 64)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_detect_raises_on_model_without_parameters_method():
    class _BogusModel:
        pass

    with pytest.raises(TypeError, match="has no callable .parameters"):
        detect_quantization(_BogusModel())


def test_detect_raises_on_non_callable_parameters():
    class _BogusModel:
        parameters = "not callable"

    with pytest.raises(TypeError, match="has no callable .parameters"):
        detect_quantization(_BogusModel())


# ---------------------------------------------------------------------------
# Inference fallback: returns None when ratio doesn't match known table
# ---------------------------------------------------------------------------


def test_inference_returns_none_for_pathological_ratio():
    """Hand-build a tree whose packed/scales ratio matches none of the
    known (bits, group_size) pairs. Detection should report quantized
    = True (because uint32 weights are present) but bits/group_size
    = None (because it can't infer them)."""
    bogus = {
        "weight": mx.zeros((4, 7), dtype=mx.uint32),  # 28 packed elements
        "scales": mx.zeros((4, 5), dtype=mx.bfloat16),  # 20 scale elements
        "biases": mx.zeros((4, 5), dtype=mx.bfloat16),
    }
    info = detect_quantization(_FakeModel({"layer": bogus}, args=None))
    assert info.is_quantized is True
    assert info.bits is None
    assert info.group_size is None
    assert info.quantized_param_count == 0  # no bits -> can't compute


def test_inference_returns_none_when_no_scales_seen():
    """Edge case: a uint32 weight without any scales sibling — we do
    not classify it as quantized at all."""
    params = {"weight_only": {"weight": mx.zeros((4, 8), dtype=mx.uint32)}}
    info = detect_quantization(_FakeModel(params))
    # No scales/biases -> not classified as quantized linear
    assert info.is_quantized is False


# ---------------------------------------------------------------------------
# Frozen dataclass invariant
# ---------------------------------------------------------------------------


def test_quantization_info_is_frozen():
    info = QuantizationInfo(
        is_quantized=False,
        bits=None,
        group_size=None,
        quantized_weight_bytes=0,
        full_precision_weight_bytes=100,
        total_weight_bytes=100,
        full_precision_param_count=50,
        quantized_param_count=0,
        effective_bits_per_param=16.0,
    )
    with pytest.raises(Exception):
        info.bits = 4  # type: ignore[misc]
