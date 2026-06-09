"""Linux CI tests for scripts/research/k3_dflash_mlx_bridge.py.

Covers ONLY the testable surface: the ``mx_to_torch`` bridge utility's
behaviour with numpy stand-ins for ``mx.array`` (numpy arrays support
the ``__array__`` protocol that ``np.asarray`` uses).

The MLX-touching paths (:class:`MLXVerifierAuxProvider`,
:func:`build_mlx_verifier_callbacks`, :func:`mlx_verify_block`) require
``mlx`` and a real Gemma 4 verifier to validate end-to-end; their
correctness is proven by ``scripts/research/k3_dflash_specdecode_eval_mac.py``
running on Mac M4 hardware and producing acceptance evidence
comparable to PR #93's CUDA evidence.

This test module's job is to ensure the dtype/device/shape contract on
the bridge utility doesn't drift silently — a regression in the
numpy-intermediate code path would corrupt every spec decode loop on
Mac.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/research to path so we can import the bridge module.
RESEARCH_DIR = Path(__file__).parent.parent.parent / "scripts" / "research"
sys.path.insert(0, str(RESEARCH_DIR))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

# Import only the bridge utilities; mx_to_torch / torch_to_mx live at
# module top-level. The MLX-touching paths are conditional imports
# inside their respective functions.
from k3_dflash_mlx_bridge import mx_to_torch  # type: ignore  # noqa: E402


class TestMxToTorch:
    """Verify mx_to_torch's contract using numpy arrays as stand-in
    for ``mx.array`` (np arrays support the same ``__array__`` /
    ``np.asarray`` interface that the bridge uses internally).
    """

    def test_basic_shape_preserved(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = mx_to_torch(x)
        assert tuple(t.shape) == (2, 3, 4)
        assert t.dtype == torch.float32

    def test_dtype_override(self):
        x = np.ones((4, 4), dtype=np.float32)
        t = mx_to_torch(x, dtype=torch.float32)
        assert t.dtype == torch.float32

        t = mx_to_torch(x, dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_device_default_cpu(self):
        x = np.zeros((2, 2), dtype=np.float32)
        t = mx_to_torch(x)
        assert t.device.type == "cpu"

    def test_device_override_to_cpu_explicit(self):
        x = np.zeros((2, 2), dtype=np.float32)
        t = mx_to_torch(x, device="cpu")
        assert t.device.type == "cpu"

    def test_values_preserved(self):
        x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        t = mx_to_torch(x)
        assert torch.allclose(t, torch.tensor([[1.5, 2.5], [3.5, 4.5]]))

    def test_copy_independence(self):
        """Bridge must copy — mutating the source numpy array after
        the bridge must NOT affect the resulting torch tensor.
        Otherwise the spec decode loop could see verifier hidden
        state mutate mid-bridge."""
        x = np.ones((3, 3), dtype=np.float32)
        t = mx_to_torch(x)
        x[0, 0] = 99.0
        assert t[0, 0].item() == 1.0  # NOT 99.0

    def test_int_dtype_passthrough(self):
        """Logits/argmax outputs may come through as int dtypes;
        bridge must not silently convert."""
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        t = mx_to_torch(x)
        assert t.dtype == torch.int64
        assert tuple(t.shape) == (2, 3)
        assert t[1, 2].item() == 6

    def test_high_rank_tensor(self):
        """Spec decode aux hiddens are [B, T, hidden] — rank 3.
        Drafter K/V are [B, H, T, D] — rank 4. Bridge must
        handle arbitrary rank correctly."""
        x = np.ones((1, 2, 4, 8, 16), dtype=np.float32)
        t = mx_to_torch(x)
        assert tuple(t.shape) == (1, 2, 4, 8, 16)


class TestModuleInterface:
    """Smoke tests on the module's public surface — catch import-time
    regressions / typos / signature drift in the API contract."""

    def test_module_exposes_expected_public_names(self):
        import k3_dflash_mlx_bridge as br
        assert hasattr(br, "mx_to_torch")
        assert hasattr(br, "torch_to_mx")
        assert hasattr(br, "MLXVerifierAuxProvider")
        assert hasattr(br, "build_mlx_verifier_callbacks")
        assert hasattr(br, "mlx_verify_block")
        assert "mx_to_torch" in br.__all__
        assert "MLXVerifierAuxProvider" in br.__all__

    def test_aux_provider_init_signature(self):
        """Constructor signature stable: (mlx_model, aux_layer_ids,
        bridge_dtype=None, bridge_device='cpu') — the rest of the
        spec decode loop depends on this."""
        import inspect
        from k3_dflash_mlx_bridge import MLXVerifierAuxProvider
        sig = inspect.signature(MLXVerifierAuxProvider.__init__)
        params = sig.parameters
        assert "mlx_model" in params
        assert "aux_layer_ids" in params
        assert "bridge_dtype" in params
        assert "bridge_device" in params

    def test_build_callbacks_signature(self):
        import inspect
        from k3_dflash_mlx_bridge import build_mlx_verifier_callbacks
        sig = inspect.signature(build_mlx_verifier_callbacks)
        params = sig.parameters
        assert "mlx_model" in params
        assert "hidden_size" in params
        assert "softcap" in params
        assert "bridge_dtype" in params
        assert "bridge_device" in params

    def test_verify_block_signature(self):
        import inspect
        from k3_dflash_mlx_bridge import mlx_verify_block
        sig = inspect.signature(mlx_verify_block)
        params = sig.parameters
        assert "mlx_model" in params
        assert "committed" in params
        assert "draft" in params
