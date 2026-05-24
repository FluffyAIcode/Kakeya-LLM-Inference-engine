"""Quantization detection and accounting for MLX-backed verifiers.

Why this module exists
----------------------

ADR 0002 §2.2 mandates 4-bit quantization for verifiers ≥ 4 B parameters,
and recommends it as an option for smaller verifiers when memory headroom
matters. ``mlx_lm.load(repo_id)`` transparently handles both
full-precision and quantized checkpoints — at the loader level, calling
sites do not have to branch. But for **memory accounting** and **ADR
0002 §2.2's 60 % memory rule** to be enforced and reported, the engine
needs to know:

1. Whether the loaded model is quantized at all.
2. If quantized, the bits-per-weight and group_size.
3. How many parameters are in the quantized portion vs the full-
   precision portion (typically embedding + lm_head stay full-precision
   even in a 4-bit checkpoint, because their per-vocab outputs are
   sensitive to quantization noise).
4. The effective bits-per-parameter averaged across the whole model,
   which is what the user actually pays for in unified-memory bytes.

This module exposes a small, MLX-API-aware helper that pulls all of
this from a loaded ``mlx_lm`` model and returns a plain, serializable
dataclass. Stats reporting (`MLXSinkWindowVerifier.stats`,
`bench_mlx_verifier_quant.py`, future engine HTTP API metrics) all
consume it.

What this module is NOT
-----------------------

* It does not decide *whether* to quantize — that is ADR 0002's
  prerogative and is already encoded in the choice of ``model_id``.
* It does not perform quantization itself — ``mlx_lm.convert -q``
  is the canonical tool; we only inspect already-quantized
  checkpoints loaded via ``mlx_lm.load``.
* It is not platform-portable: it imports MLX at module top level.
  Importing this module on non-Apple-Silicon hosts will fail at the
  ``import mlx.core`` line; that is intentional, not a bug.

API contract
------------

The two public symbols are :class:`QuantizationInfo` (immutable
dataclass) and :func:`detect_quantization` (model -> info). The
detection routine accepts any object that quacks like an
``mlx_lm`` model (has a ``parameters()`` returning the standard tree
of ``mx.array`` values, and optionally an ``args`` attribute carrying
the original config). All defensive paths through "the model doesn't
quack right" raise informative errors; we do not silently fall back
to "assume full precision".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx


# (bits, group_size) pairs that mlx_lm and mlx-community ship with.
# Ordered from most-common to least-common; the first match wins when
# inferring from the parameter tree.
_KNOWN_BITS_GROUPS: Tuple[Tuple[int, int], ...] = (
    (4, 64), (4, 32), (4, 128),
    (8, 64), (8, 32), (8, 128),
    (3, 64), (2, 64), (6, 64),
)


@dataclass(frozen=True)
class QuantizationInfo:
    """Quantization metadata for a loaded MLX model.

    Attributes
    ----------
    is_quantized:
        ``True`` if any subset of the model's weights is stored in a
        quantized format (``mx.array`` of dtype ``uint32`` packed with
        sub-byte values, accompanied by ``scales`` and ``biases`` per
        quantization group).
    bits:
        Bits per quantized weight (e.g. 4 for ``mlx-community/*-4bit``).
        ``None`` if ``is_quantized`` is ``False``.
    group_size:
        Number of consecutive weights sharing one ``(scale, bias)``
        pair. Common values: 32, 64, 128. ``None`` if not quantized.
    quantized_weight_bytes:
        Bytes occupied by the *quantized* parameters (packed weight
        data + scales + biases, as ``mx.array`` storage). ``0`` if
        not quantized.
    full_precision_weight_bytes:
        Bytes occupied by all *non-quantized* parameters (typically
        embedding tables, lm_head, layer norms). Equals total weight
        bytes minus ``quantized_weight_bytes``.
    total_weight_bytes:
        Sum of the two above. Same number returned by the existing
        ``_model_weight_bytes`` walker; included here so callers don't
        have to add it themselves.
    full_precision_param_count:
        Logical parameter count in the full-precision portion.
    quantized_param_count:
        Logical parameter count in the quantized portion (i.e. the
        number of "weights" the user thinks of, not the number of
        ``uint32`` storage elements).
    effective_bits_per_param:
        ``8 * total_weight_bytes / (full + quantized) param count``.
        For a Qwen3-1.7B 4-bit checkpoint this typically lands at
        ~4.5 (4 bits of weight + scales/biases overhead amortized
        over a group of 64).
    """

    is_quantized: bool
    bits: Optional[int]
    group_size: Optional[int]
    quantized_weight_bytes: int
    full_precision_weight_bytes: int
    total_weight_bytes: int
    full_precision_param_count: int
    quantized_param_count: int
    effective_bits_per_param: float

    def render_short(self) -> str:
        """One-line human-readable summary suitable for stats output."""
        if not self.is_quantized:
            return (
                f"unquantized | weights={self.total_weight_bytes / 1e9:.2f} GB"
            )
        return (
            f"{self.bits}-bit (group={self.group_size}) | "
            f"weights={self.total_weight_bytes / 1e9:.2f} GB | "
            f"effective={self.effective_bits_per_param:.2f} bits/param"
        )


def detect_quantization(model) -> QuantizationInfo:
    """Inspect a loaded ``mlx_lm`` model and return its quantization info.

    Parameters
    ----------
    model:
        Anything with the standard mlx_lm model interface — namely a
        ``parameters()`` method returning a (possibly nested) tree of
        ``mx.array`` parameters, and optionally an ``args`` attribute
        whose ``quantization`` field holds ``{"bits": int,
        "group_size": int}`` for quantized checkpoints.

        We do not require ``args`` to be present; if it is missing we
        fall back to weight-tree inspection for the bits/group_size
        (described below). We *do* require ``parameters()`` to work,
        because byte accounting flows through it.

    Returns
    -------
    QuantizationInfo

    Detection algorithm
    -------------------
    1. Walk the parameter tree, classifying each leaf ``mx.array``
       as "quantized payload" (a ``weight`` of ``uint32`` whose
       sibling dict also has matching ``scales``/``biases`` arrays)
       or "full precision". Per-tensor byte and element counts are
       collected from each leaf — no global dtype assumptions.
    2. If ``model.args`` exposes ``quantization``, use those numbers
       authoritatively for ``bits`` / ``group_size``.
    3. Otherwise, infer ``(bits, group_size)`` from the global ratio
       of packed-``uint32`` elements to ``scales`` elements. The
       relation
            packed_u32_elems / scales_elems = group_size * bits / 32
       has a unique integer solution within the
       :data:`_KNOWN_BITS_GROUPS` table for any well-formed
       checkpoint.

    Raises
    ------
    TypeError
        If the model lacks ``parameters()``.
    """
    if not hasattr(model, "parameters") or not callable(model.parameters):
        raise TypeError(
            f"detect_quantization: model of type {type(model).__name__} "
            "has no callable .parameters(); expected an mlx_lm model"
        )

    walker = _ParamTreeWalker()
    walker.walk(model.parameters())

    is_quantized = walker.quantized_weight_bytes > 0
    bits: Optional[int] = None
    group_size: Optional[int] = None

    if is_quantized:
        cfg_bits, cfg_group = _read_args_quantization(model)
        if cfg_bits is not None and cfg_group is not None:
            bits, group_size = cfg_bits, cfg_group
        else:
            bits, group_size = walker.infer_bits_and_group_size()

    quantized_param_count = (
        walker.packed_uint32_elements * (32 // bits) if bits else 0
    )

    total_bytes = walker.quantized_weight_bytes + walker.full_precision_weight_bytes
    total_params = walker.full_precision_element_count + quantized_param_count
    effective_bits = (8.0 * total_bytes / total_params) if total_params > 0 else 0.0

    return QuantizationInfo(
        is_quantized=is_quantized,
        bits=bits,
        group_size=group_size,
        quantized_weight_bytes=walker.quantized_weight_bytes,
        full_precision_weight_bytes=walker.full_precision_weight_bytes,
        total_weight_bytes=total_bytes,
        full_precision_param_count=walker.full_precision_element_count,
        quantized_param_count=quantized_param_count,
        effective_bits_per_param=effective_bits,
    )


def _read_args_quantization(model) -> Tuple[Optional[int], Optional[int]]:
    """Pull ``(bits, group_size)`` from ``model.args.quantization`` if present.

    Returns ``(None, None)`` if any of the following hold:
      * the model lacks an ``args`` attribute,
      * ``args.quantization`` is missing or ``None``,
      * the value is not a dict, or
      * the dict lacks an ``int`` ``bits`` or ``group_size`` field.

    We do not raise on missing config — inference from the parameter
    tree (in :meth:`_ParamTreeWalker.infer_bits_and_group_size`) is
    a valid fallback.
    """
    args = getattr(model, "args", None)
    if args is None:
        return None, None
    quant = getattr(args, "quantization", None)
    if quant is None:
        return None, None
    if not isinstance(quant, dict):
        return None, None
    bits = quant.get("bits")
    group_size = quant.get("group_size")
    if not isinstance(bits, int) or not isinstance(group_size, int):
        return None, None
    return bits, group_size


class _ParamTreeWalker:
    """Walks an mlx_lm parameter tree and accumulates byte / element counts.

    A "quantized linear" leaf is identified by the presence of three
    sibling tensors in the same dict:
      * ``weight``  : ``uint32`` packed payload
      * ``scales``  : per-group scale (float)
      * ``biases``  : per-group bias  (float)

    All other ``mx.array`` leaves (including any ``weight`` whose
    dtype is not ``uint32``) are full-precision. Lists and tuples
    are traversed recursively. Anything that is neither a tensor nor
    a container is ignored (mlx_lm parameter trees occasionally carry
    metadata strings).
    """

    def __init__(self) -> None:
        self.quantized_weight_bytes = 0
        self.full_precision_weight_bytes = 0
        self.full_precision_element_count = 0
        self.packed_uint32_elements = 0
        self.scale_elements = 0

    def walk(self, tree) -> None:
        if isinstance(tree, dict):
            self._walk_dict(tree)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                self.walk(v)
        elif isinstance(tree, mx.array):
            # Top-of-tree leaf: extremely rare but tolerated.
            self._add_full_precision(tree)

    def _walk_dict(self, d: dict) -> None:
        if _is_quantized_linear_dict(d):
            weight = d["weight"]
            scales = d["scales"]
            biases = d["biases"]

            self.quantized_weight_bytes += (
                _bytes(weight) + _bytes(scales) + _bytes(biases)
            )
            self.packed_uint32_elements += int(weight.size)
            self.scale_elements += int(scales.size)

            for k, v in d.items():
                if k not in {"weight", "scales", "biases"}:
                    self.walk(v)
            return

        for v in d.values():
            if isinstance(v, mx.array):
                self._add_full_precision(v)
            else:
                self.walk(v)

    def _add_full_precision(self, arr: "mx.array") -> None:
        self.full_precision_weight_bytes += _bytes(arr)
        self.full_precision_element_count += int(arr.size)

    def infer_bits_and_group_size(self) -> Tuple[Optional[int], Optional[int]]:
        """Reverse-engineer ``(bits, group_size)`` from packed/scale ratios.

        Used only when ``model.args.quantization`` is absent. The
        global identity is

            packed_u32_elems * (32 / bits)  ==  group_size * scales_elems

        i.e. for each quantized weight (logical), there are
        ``1 / group_size`` scales. Rearranging:

            packed_u32_elems / scales_elems == group_size * bits / 32

        We test each known ``(bits, group_size)`` combination against
        this ratio and return the first match. Returns ``(None, None)``
        if no combination fits — the caller treats that as "quantized
        but format unknown".
        """
        if self.scale_elements == 0:
            return None, None
        ratio = self.packed_uint32_elements / self.scale_elements
        for bits, gs in _KNOWN_BITS_GROUPS:
            expected = gs * bits / 32.0
            if abs(ratio - expected) < 1e-9:
                return bits, gs
        return None, None


def _is_quantized_linear_dict(d: dict) -> bool:
    """Is this dict the ``(weight, scales, biases)`` trio of a QuantizedLinear?

    All three keys must be present, ``weight`` must be a ``uint32`` array,
    and ``scales`` / ``biases`` must be ``mx.array`` (any float dtype).
    """
    if "weight" not in d or "scales" not in d or "biases" not in d:
        return False
    w = d["weight"]
    s = d["scales"]
    b = d["biases"]
    if not (
        isinstance(w, mx.array)
        and isinstance(s, mx.array)
        and isinstance(b, mx.array)
    ):
        return False
    if w.dtype != mx.uint32:
        return False
    return True


def _bytes(arr: "mx.array") -> int:
    return int(arr.size) * int(arr.dtype.size)
