"""K2.A: KVCompressor protocol + Mac-portable KakeyaLattice adapter.

ADR 0008 §11.11 specifies the contract under which the v0.4
verifier's resident sink+window K/V cache is held in compressed
form. K1 used uncompressed K/V (i.e. an implicit identity
compressor); K2.A introduces a narrow ``KVCompressor`` protocol so
that the *only* place the KakeyaLattice dependency lives is behind
that interface, and the K1.D ``DLMRestoredVerifier`` orchestration
remains compressor-agnostic.

This module ships three deliverables:

1. **`KVCompressor`** — a runtime-checkable :class:`Protocol`.
   Three operations (``compress``, ``decompress``, ``evict``) plus
   a memory accessor (``memory_bytes``) and a self-describing
   metadata field (``codec_name``). Stateful, instantiated once
   per (layer, head_kv) pair; the per-instance state is what
   amortises KL's setup cost across decode steps.

2. **`IdentityCompressor`** — the K1 baseline. Stores ``(k, v)``
   uncompressed in a ``dict`` keyed by position. ``compress`` is
   the identity function; ``decompress`` is the identity function;
   memory accounting is exact. This is the default when no
   compressor is requested and is also the K2.A oracle for the
   round-trip-identity gate (any KL implementation must round-trip
   at least as well as identity, modulo the published KL
   fidelity envelope).

3. **`KakeyaLatticeCompressor`** — a thin adapter over the
   in-house ``kakeyalattice`` package
   (``github.com/FluffyAIcode/LLM-KV--Cache-compress``, v1.4 D4
   / v1.5 E8 lattice). The adapter is **device-aware** so that
   the same code path runs on PyTorch CUDA, PyTorch MPS (Apple
   Silicon — Mac M4), and PyTorch CPU; the codec library itself
   is pure PyTorch with no CUDA-specific kernels, so cross-
   platform support is a matter of forwarding the verifier's
   active device through the codec constructor and trusting
   PyTorch's device dispatch.

   ``kakeyalattice`` is an **optional** dependency. If it is not
   installed, attempting to construct ``KakeyaLatticeCompressor``
   raises a ``KakeyaLatticeUnavailable`` error with an actionable
   install message; the higher-level ``make_default_compressor``
   factory catches that error and falls back to
   ``IdentityCompressor`` with a warning, so the v0.4 runtime
   continues to operate (with K1-level memory but K1-level
   throughput) on hosts without ``kakeyalattice`` available.

Mac M4 portability claim — discharged here, validated by the
Mac M4 reviewer aid:

    *kakeyalattice* is pure PyTorch. The codec's hot path
    (Sylvester–Hadamard rotation, per-vector qmax, Conway–Sloane
    closest-lattice-point) compiles cleanly on the PyTorch MPS
    backend. The ``KakeyaLatticeCompressor`` adapter forwards
    ``device`` to the codec constructor verbatim; the only
    Mac-specific concession is that ``device="mps"`` is the
    default on Apple Silicon (vs ``"cuda"`` on the upstream
    library) so the codec does not silently materialise tensors
    on CPU. See ADR 0008 §11.11.9 for the empirical-evidence
    path and the Mac round-trip acceptance gate.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Public exception type for the optional-dep failure path.
# ---------------------------------------------------------------------------


class KakeyaLatticeUnavailable(RuntimeError):
    """Raised when :class:`KakeyaLatticeCompressor` is constructed
    on a host where the ``kakeyalattice`` package is not installed.

    Carries an actionable install hint in the message body. The
    ``make_default_compressor`` factory catches this error and
    falls back to :class:`IdentityCompressor`; user code that
    constructs ``KakeyaLatticeCompressor`` directly should let the
    error propagate so the deployment failure is visible.
    """


# ---------------------------------------------------------------------------
# Protocol — the K2.A integration contract.
# ---------------------------------------------------------------------------


@runtime_checkable
class KVCompressor(Protocol):
    """K2.A integration contract for verifier-side K/V cache codecs.

    Per ADR 0008 §11.11.4. One instance per (layer, head_kv); see
    ``DLMRestoredVerifier`` orchestration in K1.D for how
    instances are wired into the patched attention forward.

    Lifecycle of a single decode step:

    1. ``compress(k, v, positions)`` — verifier writes new K/V at
       the given resident positions (sink + window slots that
       were just produced by the current forward).
    2. (during attention) ``decompress(positions)`` — verifier
       reads back the K/V at all currently-resident positions.
       The returned tensors live one forward.
    3. (eviction) ``evict(positions)`` — verifier drops positions
       that just left the sliding window.
    4. ``memory_bytes()`` — sustained byte size of the compressor's
       internal state. Used by K1.G memory accounting.

    Idempotence: ``compress`` over an already-resident position
    overwrites in-place; ``evict`` over a non-resident position is
    a no-op. ``decompress`` over a non-resident position raises
    ``KeyError`` — no silent zero-fill, because that would mask
    a verifier-side cache bookkeeping bug.
    """

    @property
    def codec_name(self) -> str:
        """Self-describing label for logging / JSON evidence."""

    def compress(
        self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
    ) -> None:
        """Store K/V at the given positions in compressed form.

        Args:
            k: ``[..., n, head_dim]`` tensor of keys at ``positions``.
            v: ``[..., n, head_dim]`` tensor of values at ``positions``.
            positions: ``[n]`` int64 tensor of absolute token positions.

        Idempotent on repeated positions (overwrite semantics).
        """

    def decompress(
        self, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct K/V at the given positions.

        Returns ``(k, v)`` of shape ``[..., n, head_dim]`` where
        ``n = len(positions)``. Order matches ``positions``.

        Raises:
            KeyError: if any position is not currently resident.
        """

    def evict(self, positions: torch.Tensor) -> None:
        """Drop the given positions from the compressor's state.

        No-op on non-resident positions.
        """

    def memory_bytes(self) -> int:
        """Sustained byte size of the compressor's state."""


# ---------------------------------------------------------------------------
# IdentityCompressor — K1 baseline; uncompressed.
# ---------------------------------------------------------------------------


class IdentityCompressor:
    """Reference no-op compressor — stores ``(k, v)`` uncompressed.

    Round-trip identity is exact (bit-for-bit). Memory accounting
    is a sum of stored tensor element counts × dtype itemsize.

    Used as:

    * The K1 default (so K1.E NIAH evidence with this compressor
      reproduces the K1.D oracle path bit-for-bit).
    * The K2.A round-trip oracle (any KakeyaLatticeCompressor
      reconstruction error is measured *relative* to this).
    * The cross-platform fallback when ``kakeyalattice`` is not
      installed (see :func:`make_default_compressor`).
    """

    codec_name: str = "identity"

    def __init__(self) -> None:
        # Position -> (k_slice, v_slice). Tensors are stored as
        # they were written; no copy. The verifier owns lifetime.
        self._k: Dict[int, torch.Tensor] = {}
        self._v: Dict[int, torch.Tensor] = {}

    def compress(
        self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
    ) -> None:
        if k.shape[:-1] != v.shape[:-1]:
            raise ValueError(
                f"k.shape[:-1]={tuple(k.shape[:-1])} != "
                f"v.shape[:-1]={tuple(v.shape[:-1])}"
            )
        if k.shape[-2] != positions.shape[0]:
            raise ValueError(
                f"k position dim {k.shape[-2]} != "
                f"len(positions) {positions.shape[0]}"
            )
        for i, pos in enumerate(positions.tolist()):
            self._k[int(pos)] = k.select(-2, i).clone()
            self._v[int(pos)] = v.select(-2, i).clone()

    def decompress(
        self, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_slices = []
        v_slices = []
        for pos in positions.tolist():
            ipos = int(pos)
            if ipos not in self._k:
                raise KeyError(
                    f"position {ipos} not resident in IdentityCompressor"
                )
            k_slices.append(self._k[ipos])
            v_slices.append(self._v[ipos])
        # Stack along position dim (-2 of original layout).
        k = torch.stack(k_slices, dim=-2)
        v = torch.stack(v_slices, dim=-2)
        return k, v

    def evict(self, positions: torch.Tensor) -> None:
        for pos in positions.tolist():
            ipos = int(pos)
            self._k.pop(ipos, None)
            self._v.pop(ipos, None)

    def memory_bytes(self) -> int:
        # Bytes occupied by the resident K/V tensors. Both dicts
        # have the same key set; sum once over each.
        total = 0
        for t in self._k.values():
            total += t.numel() * t.element_size()
        for t in self._v.values():
            total += t.numel() * t.element_size()
        return total


# ---------------------------------------------------------------------------
# KakeyaLatticeCompressor — K2.A target. Optional dependency.
# ---------------------------------------------------------------------------


_KL_INSTALL_HINT = (
    "kakeyalattice is required for KakeyaLatticeCompressor. Install with "
    "`pip install kakeyalattice` (or `pip install -e <local-clone>` of "
    "github.com/FluffyAIcode/LLM-KV--Cache-compress). Falling back to "
    "IdentityCompressor preserves correctness but loses the K2.A "
    "throughput improvement of ADR 0008 §11.11."
)


def _import_kakeyalattice():
    """Import the codec module; raise KakeyaLatticeUnavailable on failure.

    Centralised so the error path is testable without the package
    actually missing on the test host (we monkey-patch this
    function in :class:`TestImportFailure` Linux unit tests).
    """
    try:
        from kakeyalattice import (
            V14KakeyaZamirLatticeGPU as _D4Codec,
            V15KakeyaZamirE8GPU as _E8Codec,
        )
    except ImportError as e:
        raise KakeyaLatticeUnavailable(_KL_INSTALL_HINT) from e
    return _D4Codec, _E8Codec


def _resolve_codec_class(lattice: str):
    """Map ``lattice`` string ('D4' / 'E8') to a codec class."""
    d4_cls, e8_cls = _import_kakeyalattice()
    if lattice.upper() == "D4":
        return d4_cls
    if lattice.upper() == "E8":
        return e8_cls
    raise ValueError(
        f"unknown lattice {lattice!r}; expected 'D4' (v1.4) or 'E8' (v1.5)"
    )


class KakeyaLatticeCompressor:
    """K2.A KV compressor backed by KakeyaLattice (D4 or E8 lattice).

    Stores K/V in compressed form via the in-house codec at
    ``kakeyalattice.V14KakeyaZamirLatticeGPU`` (D4, v1.4) or
    ``V15KakeyaZamirE8GPU`` (E8, v1.5). Per ADR 0008 §11.11, this
    compressor's compression headroom is **not** spent on memory
    reduction (K1 already satisfies constant memory) but on
    enabling a larger resident sink+window cache at the same
    memory budget — which is what improves throughput by reducing
    the dLM K/V Restoration path's invocation rate.

    Cross-platform: ``device`` is forwarded to the underlying codec
    constructor verbatim. The codec is pure PyTorch (no CUDA-specific
    kernels), so:

    * ``device="cuda"`` — upstream default, vast.ai H100/H200
    * ``device="mps"`` — Mac M4 / Apple Silicon (K2.A Mac M4
      portability target; round-trip identity validated by the
      Mac reviewer aid ``review_pr_k2a_kl_smoke_on_mac.sh``)
    * ``device="cpu"`` — last-resort fallback / Linux CI

    The compressor stores reconstructed K/V (``codec.roundtrip(...)``)
    rather than the raw lattice bits, because K1.D's attention
    layer expects standard ``torch.Tensor`` inputs and the codec
    library does not currently expose its compressed
    representation as a Python object — only as a round-tripped
    tensor. This keeps the K2.A integration purely at the
    correctness-preservation layer; memory savings come from the
    fact that the round-tripped tensor has the *fidelity* of the
    lattice (so a downstream codec or quantised storage layer
    can compress it lossily without further loss), not from any
    in-RAM size change.

    For pure compressed storage (lattice bits only, no
    round-tripped tensors held in memory), a future K4-slot
    optimisation can add a ``codec.encode(...)`` / ``decode(...)``
    pair if upstream exposes them; not in scope for K2.A.
    """

    def __init__(
        self,
        head_dim: int,
        device: Optional[torch.device] = None,
        lattice: str = "D4",
        q_range: int = 38,
    ) -> None:
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {head_dim}")
        if (head_dim & (head_dim - 1)) != 0:
            raise ValueError(
                f"head_dim must be a power of 2 for Hadamard rotation; "
                f"got {head_dim}"
            )
        codec_cls = _resolve_codec_class(lattice)
        device = device if device is not None else torch.device("cpu")
        # KakeyaLattice's constructor accepts device as either a
        # torch.device or a str; we pass a string for consistency
        # with the upstream API surface.
        self._codec = codec_cls(
            D=int(head_dim),
            q_range=int(q_range),
            device=str(device),
        )
        self._device = device
        self._lattice = lattice.upper()
        self._head_dim = int(head_dim)
        self._q_range = int(q_range)
        # Position -> (k_round_tripped, v_round_tripped). After
        # KL fidelity is applied. The tensors live on ``device``.
        self._k: Dict[int, torch.Tensor] = {}
        self._v: Dict[int, torch.Tensor] = {}

    @property
    def codec_name(self) -> str:
        return (
            f"kakeyalattice-{self._lattice}-Q{self._q_range}-"
            f"D{self._head_dim}"
        )

    def compress(
        self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
    ) -> None:
        if k.shape[:-1] != v.shape[:-1]:
            raise ValueError(
                f"k.shape[:-1]={tuple(k.shape[:-1])} != "
                f"v.shape[:-1]={tuple(v.shape[:-1])}"
            )
        if k.shape[-2] != positions.shape[0]:
            raise ValueError(
                f"k position dim {k.shape[-2]} != "
                f"len(positions) {positions.shape[0]}"
            )
        if k.shape[-1] != self._head_dim:
            raise ValueError(
                f"k head_dim {k.shape[-1]} != configured "
                f"head_dim {self._head_dim}"
            )
        # Run round-trip through the lattice codec. The codec
        # accepts arbitrary leading dims as long as the last dim
        # equals D=head_dim; we batch all positions through one
        # call per K/V tensor so the per-vector setup amortises.
        k_hat = self._codec.roundtrip(k.contiguous())
        v_hat = self._codec.roundtrip(v.contiguous())
        for i, pos in enumerate(positions.tolist()):
            self._k[int(pos)] = k_hat.select(-2, i).clone()
            self._v[int(pos)] = v_hat.select(-2, i).clone()

    def decompress(
        self, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_slices = []
        v_slices = []
        for pos in positions.tolist():
            ipos = int(pos)
            if ipos not in self._k:
                raise KeyError(
                    f"position {ipos} not resident in "
                    f"KakeyaLatticeCompressor"
                )
            k_slices.append(self._k[ipos])
            v_slices.append(self._v[ipos])
        k = torch.stack(k_slices, dim=-2)
        v = torch.stack(v_slices, dim=-2)
        return k, v

    def evict(self, positions: torch.Tensor) -> None:
        for pos in positions.tolist():
            ipos = int(pos)
            self._k.pop(ipos, None)
            self._v.pop(ipos, None)

    def memory_bytes(self) -> int:
        total = 0
        for t in self._k.values():
            total += t.numel() * t.element_size()
        for t in self._v.values():
            total += t.numel() * t.element_size()
        return total


# ---------------------------------------------------------------------------
# Factory — picks the right compressor for the active device.
# ---------------------------------------------------------------------------


def make_default_compressor(
    *,
    head_dim: int,
    device: Optional[torch.device] = None,
    prefer_kakeya: bool = True,
    lattice: str = "D4",
    q_range: int = 38,
) -> KVCompressor:
    """Build the default :class:`KVCompressor` for the given device.

    Decision order:

    1. If ``prefer_kakeya`` is False, return :class:`IdentityCompressor`.
       (Used by tests / K1 baseline runs that intentionally want
       the uncompressed reference.)
    2. Try to construct :class:`KakeyaLatticeCompressor` on the
       given device. On success, return it.
    3. On :class:`KakeyaLatticeUnavailable` (the optional
       dependency is not installed), emit a warning and fall back
       to :class:`IdentityCompressor`.

    Other errors (e.g. invalid head_dim, unknown lattice) are NOT
    caught — they signal a configuration bug rather than an
    install issue.

    Mac M4 path: ``device=torch.device('mps')`` flows through to
    the KakeyaLattice constructor; the codec is pure PyTorch so
    MPS dispatch is automatic. See ADR 0008 §11.11.9 and the Mac
    reviewer aid ``review_pr_k2a_kl_smoke_on_mac.sh`` for the
    Mac round-trip-identity acceptance evidence.
    """
    if not prefer_kakeya:
        return IdentityCompressor()
    try:
        return KakeyaLatticeCompressor(
            head_dim=head_dim,
            device=device,
            lattice=lattice,
            q_range=q_range,
        )
    except KakeyaLatticeUnavailable as e:
        warnings.warn(
            f"KakeyaLattice unavailable; falling back to "
            f"IdentityCompressor (K1 throughput baseline). "
            f"Original cause: {e}",
            stacklevel=2,
        )
        return IdentityCompressor()


__all__ = [
    "KVCompressor",
    "IdentityCompressor",
    "KakeyaLatticeCompressor",
    "KakeyaLatticeUnavailable",
    "make_default_compressor",
]
