"""Linux CI unit tests for inference_engine/v04/kv_compressor.py.

Covers the K2.A KVCompressor scaffold per ADR 0008 §11.11.4. The
optional ``kakeyalattice`` dependency is not present on the Linux
CI agent, so:

* ``IdentityCompressor`` — fully tested here (Linux CPU, no
  optional deps).
* ``KakeyaLatticeCompressor`` — import-failure path tested here
  (validates KakeyaLatticeUnavailable + the install hint
  message). Round-trip fidelity validation is done on Mac M4 by
  ``scripts/review_pr_k2a_kl_smoke_on_mac.sh`` — see ADR 0008
  §11.11.9 for the empirical-evidence path.
* ``make_default_compressor`` — fallback path tested here (the
  warning-and-Identity path when KL is unavailable). The
  KL-success path is covered by the Mac smoke.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from inference_engine.v04.kv_compressor import (
    IdentityCompressor,
    KakeyaLatticeCompressor,
    KakeyaLatticeUnavailable,
    KVCompressor,
    make_default_compressor,
)


# ---------------------------------------------------------------------------
# Protocol shape — runtime_checkable membership.
# ---------------------------------------------------------------------------


class TestProtocolMembership:
    def test_identity_is_kv_compressor(self):
        assert isinstance(IdentityCompressor(), KVCompressor)

    def test_random_class_is_not_kv_compressor(self):
        class NotACompressor:
            pass

        assert not isinstance(NotACompressor(), KVCompressor)

    def test_protocol_method_set(self):
        # Sanity that the runtime_checkable Protocol covers all
        # four ops we expect; if a future refactor drops one, the
        # check above would silently start passing for incomplete
        # impls. Pin the membership via attribute presence.
        c = IdentityCompressor()
        for method in ("compress", "decompress", "evict", "memory_bytes"):
            assert callable(getattr(c, method))
        assert isinstance(c.codec_name, str)


# ---------------------------------------------------------------------------
# IdentityCompressor — core correctness.
# ---------------------------------------------------------------------------


class TestIdentityCompressor:
    def _kv(self, n: int, head_dim: int = 8, dtype=torch.float32):
        # Layout: [num_kv_heads, n, head_dim] (K1.D's per-layer call).
        torch.manual_seed(42)
        k = torch.randn(2, n, head_dim, dtype=dtype)
        v = torch.randn(2, n, head_dim, dtype=dtype)
        positions = torch.arange(n, dtype=torch.int64)
        return k, v, positions

    def test_round_trip_exact(self):
        c = IdentityCompressor()
        k, v, positions = self._kv(n=5)
        c.compress(k, v, positions)
        k_rt, v_rt = c.decompress(positions)
        # Exact bit-for-bit equality (Identity is the round-trip oracle).
        assert torch.equal(k, k_rt)
        assert torch.equal(v, v_rt)

    def test_partial_decompress_preserves_order(self):
        c = IdentityCompressor()
        k, v, positions = self._kv(n=10)
        c.compress(k, v, positions)
        # Decompress a permuted subset.
        wanted = torch.tensor([7, 2, 5, 0], dtype=torch.int64)
        k_rt, v_rt = c.decompress(wanted)
        assert k_rt.shape == (2, 4, 8)
        for i, pos in enumerate(wanted.tolist()):
            assert torch.equal(k_rt[..., i, :], k[..., pos, :])
            assert torch.equal(v_rt[..., i, :], v[..., pos, :])

    def test_overwrite_is_idempotent_to_latest_value(self):
        c = IdentityCompressor()
        k, v, positions = self._kv(n=4)
        c.compress(k, v, positions)
        # Now overwrite position 1 with a known sentinel.
        sentinel_k = torch.full((2, 1, 8), 7.0)
        sentinel_v = torch.full((2, 1, 8), -3.0)
        c.compress(
            sentinel_k, sentinel_v, torch.tensor([1], dtype=torch.int64),
        )
        k_rt, v_rt = c.decompress(torch.tensor([1], dtype=torch.int64))
        assert torch.equal(k_rt, sentinel_k)
        assert torch.equal(v_rt, sentinel_v)
        # Other positions undisturbed.
        k_rt0, _ = c.decompress(torch.tensor([0], dtype=torch.int64))
        assert torch.equal(k_rt0, k[..., 0:1, :])

    def test_evict_drops_state(self):
        c = IdentityCompressor()
        k, v, positions = self._kv(n=4)
        c.compress(k, v, positions)
        c.evict(torch.tensor([1, 3], dtype=torch.int64))
        # 1 and 3 gone; 0 and 2 remain.
        k_rt, _ = c.decompress(torch.tensor([0, 2], dtype=torch.int64))
        assert k_rt.shape == (2, 2, 8)
        with pytest.raises(KeyError, match="position 1"):
            c.decompress(torch.tensor([1], dtype=torch.int64))

    def test_evict_nonresident_is_no_op(self):
        c = IdentityCompressor()
        k, v, positions = self._kv(n=2)
        c.compress(k, v, positions)
        c.evict(torch.tensor([99, 100], dtype=torch.int64))  # never seen
        # State undisturbed.
        k_rt, _ = c.decompress(positions)
        assert torch.equal(k_rt, k)

    def test_decompress_nonresident_raises_keyerror(self):
        c = IdentityCompressor()
        with pytest.raises(KeyError, match="position 5"):
            c.decompress(torch.tensor([5], dtype=torch.int64))

    def test_memory_bytes_matches_stored_tensors(self):
        c = IdentityCompressor()
        # 4 positions × 2 kv-heads × 8 head_dim × 4 bytes (fp32) × 2 (K+V)
        k, v, positions = self._kv(n=4, head_dim=8, dtype=torch.float32)
        c.compress(k, v, positions)
        # Per-position slice is [num_kv_heads, head_dim] = 16 elem * 4B = 64B.
        # 4 positions × 2 (K + V) × 64B = 512B
        assert c.memory_bytes() == 512

    def test_memory_bytes_grows_then_shrinks_with_evict(self):
        c = IdentityCompressor()
        k, v, positions = self._kv(n=4, head_dim=8)
        c.compress(k, v, positions)
        before = c.memory_bytes()
        c.evict(torch.tensor([0, 1], dtype=torch.int64))
        after = c.memory_bytes()
        assert after == before // 2

    def test_codec_name_is_identity(self):
        assert IdentityCompressor().codec_name == "identity"

    def test_shape_mismatch_in_compress_raises(self):
        c = IdentityCompressor()
        k = torch.randn(2, 3, 8)
        v = torch.randn(2, 4, 8)  # mismatched position dim
        positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        with pytest.raises(ValueError, match="shape"):
            c.compress(k, v, positions)

    def test_position_count_mismatch_raises(self):
        c = IdentityCompressor()
        k = torch.randn(2, 3, 8)
        v = torch.randn(2, 3, 8)
        positions = torch.tensor([0, 1], dtype=torch.int64)  # only 2
        with pytest.raises(ValueError, match="position"):
            c.compress(k, v, positions)


# ---------------------------------------------------------------------------
# KakeyaLatticeCompressor — import-failure path is the only one
# Linux CI exercises (the package is not installed on the agent VM).
# Round-trip fidelity is validated on Mac M4 by the smoke script.
# ---------------------------------------------------------------------------


class TestKakeyaLatticeUnavailableHandling:
    def test_construct_raises_when_kakeyalattice_missing(self, monkeypatch):
        # Force the import to fail even if kakeyalattice happens
        # to be installed on the runner (defensive).
        from inference_engine.v04 import kv_compressor

        def _raise(*args, **kwargs):
            raise KakeyaLatticeUnavailable(
                "test: kakeyalattice not installed"
            )

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice", _raise,
        )
        with pytest.raises(KakeyaLatticeUnavailable):
            KakeyaLatticeCompressor(head_dim=128)

    def test_install_hint_in_error_message(self, monkeypatch):
        from inference_engine.v04 import kv_compressor

        def _real_import_attempt():
            # Trigger the real import; if kakeyalattice is missing,
            # this surfaces the install hint; if it's present, skip.
            try:
                from kakeyalattice import V14KakeyaZamirLatticeGPU  # noqa: F401
            except ImportError:
                return
            pytest.skip("kakeyalattice is installed; install-hint test n/a")

        _real_import_attempt()
        # Manually trigger the centralised import helper.
        with pytest.raises(KakeyaLatticeUnavailable) as exc_info:
            kv_compressor._import_kakeyalattice()
        msg = str(exc_info.value)
        assert "kakeyalattice" in msg
        assert "pip install kakeyalattice" in msg
        assert "IdentityCompressor" in msg

    def test_invalid_head_dim_raises_before_import(self):
        # head_dim validation runs before the codec is constructed;
        # this error path is independent of whether kakeyalattice
        # is installed.
        with pytest.raises(ValueError, match="head_dim"):
            KakeyaLatticeCompressor(head_dim=0)
        with pytest.raises(ValueError, match="head_dim"):
            KakeyaLatticeCompressor(head_dim=-4)
        with pytest.raises(ValueError, match="power of 2"):
            KakeyaLatticeCompressor(head_dim=100)

    def test_invalid_lattice_raises(self, monkeypatch):
        # Patch the import to succeed with stub codec classes so
        # we can exercise the lattice-name validation branch
        # without requiring kakeyalattice to be installed.
        from inference_engine.v04 import kv_compressor

        class _StubCodec:
            def __init__(self, *a, **kw):
                pass

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_StubCodec, _StubCodec),
        )
        with pytest.raises(ValueError, match="unknown lattice"):
            KakeyaLatticeCompressor(head_dim=128, lattice="Z4")


# ---------------------------------------------------------------------------
# KakeyaLatticeCompressor with a stub codec — verifies the
# adapter's K/V routing logic without needing the real codec on
# Linux CI. The actual round-trip fidelity is verified on Mac M4.
# ---------------------------------------------------------------------------


class _StubKakeyaLatticeCodec:
    """Minimal stand-in that records what was round-tripped.

    Returns the input unchanged from ``roundtrip(...)`` so the
    adapter behaves like IdentityCompressor with a different
    codec_name. Lets us validate routing without depending on
    the real KL package on Linux CI.
    """

    def __init__(self, *, D, q_range, device):
        self.D = D
        self.q_range = q_range
        self.device = device
        self.roundtrip_calls = 0

    def roundtrip(self, x):
        self.roundtrip_calls += 1
        return x


class TestKakeyaLatticeCompressorRouting:
    def _make(self, monkeypatch, head_dim=128, lattice="D4"):
        from inference_engine.v04 import kv_compressor
        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_StubKakeyaLatticeCodec, _StubKakeyaLatticeCodec),
        )
        c = KakeyaLatticeCompressor(
            head_dim=head_dim, lattice=lattice, q_range=38,
        )
        return c

    def test_codec_name_self_describing(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=128, lattice="D4")
        # 'kakeyalattice-D4-Q38-D128'
        assert "kakeyalattice" in c.codec_name
        assert "D4" in c.codec_name
        assert "Q38" in c.codec_name
        assert "D128" in c.codec_name

    def test_e8_codec_name(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=128, lattice="E8")
        assert "E8" in c.codec_name

    def test_compress_invokes_roundtrip(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=8)
        k = torch.randn(2, 3, 8)
        v = torch.randn(2, 3, 8)
        positions = torch.tensor([10, 11, 12], dtype=torch.int64)
        c.compress(k, v, positions)
        # One roundtrip per K/V tensor (not per position).
        assert c._codec.roundtrip_calls == 2

    def test_decompress_returns_compressed_state(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=8)
        k = torch.randn(2, 3, 8)
        v = torch.randn(2, 3, 8)
        positions = torch.tensor([10, 11, 12], dtype=torch.int64)
        c.compress(k, v, positions)
        # With Identity-stub roundtrip, output == input bit-for-bit.
        k_rt, v_rt = c.decompress(positions)
        assert torch.equal(k, k_rt)
        assert torch.equal(v, v_rt)

    def test_evict_clears_position(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=8)
        k = torch.randn(2, 2, 8)
        v = torch.randn(2, 2, 8)
        c.compress(k, v, torch.tensor([5, 6], dtype=torch.int64))
        c.evict(torch.tensor([5], dtype=torch.int64))
        with pytest.raises(KeyError):
            c.decompress(torch.tensor([5], dtype=torch.int64))
        # 6 still resident.
        c.decompress(torch.tensor([6], dtype=torch.int64))

    def test_memory_bytes_accounts_compressed_state(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=8)
        k = torch.randn(2, 4, 8, dtype=torch.float32)
        v = torch.randn(2, 4, 8, dtype=torch.float32)
        c.compress(k, v, torch.tensor([0, 1, 2, 3], dtype=torch.int64))
        # 4 positions × 2 kv-heads × 8 head_dim × 4 bytes × 2 (K+V) = 512
        assert c.memory_bytes() == 512

    def test_head_dim_mismatch_in_compress_raises(self, monkeypatch):
        c = self._make(monkeypatch, head_dim=8)
        k = torch.randn(2, 1, 16)  # wrong head_dim
        v = torch.randn(2, 1, 16)
        with pytest.raises(ValueError, match="head_dim"):
            c.compress(k, v, torch.tensor([0], dtype=torch.int64))

    def test_device_string_passed_to_codec(self, monkeypatch):
        # The adapter must forward a string device to the codec
        # constructor (kakeyalattice's published API takes str).
        from inference_engine.v04 import kv_compressor

        captured = {}

        class _CapturingStub(_StubKakeyaLatticeCodec):
            def __init__(self, *, D, q_range, device):
                captured["D"] = D
                captured["q_range"] = q_range
                captured["device"] = device

            def roundtrip(self, x):
                return x

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_CapturingStub, _CapturingStub),
        )
        KakeyaLatticeCompressor(
            head_dim=128, device=torch.device("cpu"),
        )
        assert captured["D"] == 128
        assert captured["q_range"] == 38
        # Critically: device passed as a STRING (KL's API).
        assert isinstance(captured["device"], str)
        assert captured["device"] == "cpu"

    def test_mps_device_forwarded_as_string(self, monkeypatch):
        # Mac M4 portability: device='mps' must reach the codec
        # unmodified. This is the K2.A Mac portability load-bearing
        # property — without it, the codec would silently materialise
        # tensors on CPU even though the verifier is on MPS.
        from inference_engine.v04 import kv_compressor

        captured = {}

        class _CapturingStub(_StubKakeyaLatticeCodec):
            def __init__(self, *, D, q_range, device):
                captured["device"] = device

            def roundtrip(self, x):
                return x

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_CapturingStub, _CapturingStub),
        )
        KakeyaLatticeCompressor(
            head_dim=128, device=torch.device("mps"),
        )
        assert captured["device"] == "mps"


# ---------------------------------------------------------------------------
# make_default_compressor — fallback semantics.
# ---------------------------------------------------------------------------


class TestMakeDefaultCompressor:
    def test_prefer_kakeya_false_returns_identity(self):
        c = make_default_compressor(head_dim=128, prefer_kakeya=False)
        assert isinstance(c, IdentityCompressor)

    def test_kakeya_unavailable_falls_back_with_warning(self, monkeypatch):
        from inference_engine.v04 import kv_compressor

        def _raise(*a, **kw):
            raise KakeyaLatticeUnavailable("test: not installed")

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice", _raise,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c = make_default_compressor(head_dim=128, prefer_kakeya=True)
        assert isinstance(c, IdentityCompressor)
        # Warning emitted with the install hint.
        assert any(
            "KakeyaLattice unavailable" in str(w.message) for w in caught
        )

    def test_kakeya_success_returns_kl_compressor(self, monkeypatch):
        from inference_engine.v04 import kv_compressor

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_StubKakeyaLatticeCodec, _StubKakeyaLatticeCodec),
        )
        c = make_default_compressor(head_dim=128, prefer_kakeya=True)
        assert isinstance(c, KakeyaLatticeCompressor)
        assert "kakeyalattice" in c.codec_name

    def test_lattice_passthrough(self, monkeypatch):
        from inference_engine.v04 import kv_compressor

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_StubKakeyaLatticeCodec, _StubKakeyaLatticeCodec),
        )
        d4 = make_default_compressor(head_dim=128, lattice="D4")
        e8 = make_default_compressor(head_dim=128, lattice="E8")
        assert "D4" in d4.codec_name
        assert "E8" in e8.codec_name

    def test_q_range_passthrough(self, monkeypatch):
        from inference_engine.v04 import kv_compressor

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_StubKakeyaLatticeCodec, _StubKakeyaLatticeCodec),
        )
        c = make_default_compressor(head_dim=128, q_range=152)
        assert "Q152" in c.codec_name

    def test_invalid_head_dim_propagates(self, monkeypatch):
        # Configuration bug — should NOT be caught. Distinct from
        # the "kakeyalattice not installed" path (which is caught
        # and logged). Use stub codec so the path reaches the
        # KakeyaLatticeCompressor constructor where head_dim is
        # validated; IdentityCompressor doesn't take head_dim, so
        # prefer_kakeya=False would be a no-op.
        from inference_engine.v04 import kv_compressor

        monkeypatch.setattr(
            kv_compressor, "_import_kakeyalattice",
            lambda: (_StubKakeyaLatticeCodec, _StubKakeyaLatticeCodec),
        )
        with pytest.raises(ValueError, match="head_dim"):
            make_default_compressor(head_dim=0, prefer_kakeya=True)
