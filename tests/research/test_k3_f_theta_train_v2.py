"""Linux CI tests for the v2 trainer pieces in
``scripts/research/k3_f_theta_train``: cosine+magnitude loss, NIAH
synthetic prompts, cosine LR schedule.

These are the trainer-side fixes for the recall=0 evidence in PR #103
(f_θ v1). See the script docstring for v2 motivation.

The training loop itself requires CUDA + a 26B verifier and is
validated empirically via vast.ai (see scripts/review_pr_k3_f_theta_
train_on_vast.sh). Linux CI verifies the building blocks.
"""

from __future__ import annotations

import math

import pytest
import torch

# Import the v2 helpers — the script is importable as a module via the
# scripts/research package convention used elsewhere in the codebase.
import importlib.util
import pathlib
import sys

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts" / "research" / "k3_f_theta_train.py"
)
_spec = importlib.util.spec_from_file_location("k3_f_theta_train", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register in sys.modules BEFORE exec so @dataclass (which probes
# sys.modules[cls.__module__] for KW_ONLY type-id check) doesn't trip.
sys.modules["k3_f_theta_train"] = _mod
_spec.loader.exec_module(_mod)


# ---------------------------------------------------------------------------
# Per-vector cosine + magnitude loss
# ---------------------------------------------------------------------------


class TestPerVectorCosineMagLoss:

    def test_identical_vectors_give_zero_loss(self):
        x = torch.randn(2, 5, 4, 8)
        loss, cos, mag = _mod._per_vector_cosine_mag_loss(x, x)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)
        assert float(cos) == pytest.approx(0.0, abs=1e-5)
        assert float(mag) == pytest.approx(0.0, abs=1e-5)

    def test_negated_vectors_give_cos_loss_2(self):
        x = torch.randn(2, 5, 4, 8)
        loss, cos, mag = _mod._per_vector_cosine_mag_loss(x, -x)
        # cos sim between x and -x is -1, so 1-cos = 2
        assert float(cos) == pytest.approx(2.0, abs=1e-3)
        # magnitude is the same so mag_loss ≈ 0
        assert float(mag) == pytest.approx(0.0, abs=1e-3)

    def test_orthogonal_vectors_give_cos_loss_1(self):
        # Two orthogonal unit vectors — cos sim = 0 → cos_loss = 1
        pred = torch.zeros(1, 1, 1, 4)
        pred[..., 0] = 1.0
        tgt = torch.zeros(1, 1, 1, 4)
        tgt[..., 1] = 1.0
        _, cos, mag = _mod._per_vector_cosine_mag_loss(pred, tgt)
        assert float(cos) == pytest.approx(1.0, abs=1e-5)
        assert float(mag) == pytest.approx(0.0, abs=1e-5)

    def test_scaled_vector_gives_zero_cos_nonzero_mag(self):
        # pred = 2 * tgt → same direction (cos=1, cos_loss=0)
        # but different magnitude (‖pred‖ = 2‖tgt‖)
        tgt = torch.randn(2, 5, 4, 8)
        pred = 2.0 * tgt
        _, cos, mag = _mod._per_vector_cosine_mag_loss(pred, tgt)
        assert float(cos) == pytest.approx(0.0, abs=1e-3)
        assert float(mag) > 0.0

    def test_loss_is_differentiable(self):
        pred = torch.randn(2, 5, 4, 8, requires_grad=True)
        tgt = torch.randn(2, 5, 4, 8)
        loss, _, _ = _mod._per_vector_cosine_mag_loss(pred, tgt)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.norm().item() > 0.0


# ---------------------------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------------------------


class TestLRSchedule:

    def test_const_schedule_returns_peak(self):
        for s in [1, 100, 10000]:
            assert _mod._lr_at_step(
                s, peak_lr=1e-3, total_steps=1000,
                warmup_steps=100, schedule="const",
            ) == 1e-3

    def test_cosine_warmup_starts_below_peak(self):
        lr_step1 = _mod._lr_at_step(
            1, peak_lr=1e-3, total_steps=1000, warmup_steps=100,
            schedule="cosine",
        )
        # step 1 of 100 warmup → lr = 1e-3 * 1/100 = 1e-5
        assert lr_step1 == pytest.approx(1e-5, rel=1e-6)

    def test_cosine_warmup_reaches_peak(self):
        lr = _mod._lr_at_step(
            100, peak_lr=1e-3, total_steps=1000, warmup_steps=100,
            schedule="cosine",
        )
        assert lr == pytest.approx(1e-3, rel=1e-6)

    def test_cosine_decay_reaches_floor(self):
        # At final step, cosine should be ≈ peak / 100
        lr_final = _mod._lr_at_step(
            1000, peak_lr=1e-3, total_steps=1000, warmup_steps=100,
            schedule="cosine",
        )
        assert lr_final == pytest.approx(1e-5, rel=1e-3)

    def test_cosine_midway_above_floor(self):
        # halfway through decay (step ≈ 550), cosine factor = cos(π/2) = 0
        # → lr = floor + (peak - floor) * 0.5 ≈ 5e-4
        lr_mid = _mod._lr_at_step(
            550, peak_lr=1e-3, total_steps=1000, warmup_steps=100,
            schedule="cosine",
        )
        assert lr_mid == pytest.approx(5e-4, rel=0.05)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="unknown schedule"):
            _mod._lr_at_step(
                1, peak_lr=1e-3, total_steps=1000, warmup_steps=100,
                schedule="exponential",
            )


# ---------------------------------------------------------------------------
# NIAH-style synthetic training prompts
# ---------------------------------------------------------------------------


class TestNIAHTrainingPrompts:

    def test_returns_requested_count(self):
        prompts = _mod._make_niah_training_prompts(8, seed=1234)
        assert len(prompts) == 8
        assert all(isinstance(p, str) for p in prompts)

    def test_prompts_contain_needle(self):
        prompts = _mod._make_niah_training_prompts(4, seed=42)
        for p in prompts:
            assert "secret code is" in p.lower(), \
                "needle pattern missing"
            assert "Question: What is the secret code?" in p, \
                "question line missing"

    def test_seed_determinism(self):
        a = _mod._make_niah_training_prompts(4, seed=99)
        b = _mod._make_niah_training_prompts(4, seed=99)
        assert a == b

    def test_different_seeds_give_different_prompts(self):
        a = _mod._make_niah_training_prompts(4, seed=1)
        b = _mod._make_niah_training_prompts(4, seed=2)
        assert a != b

    def test_haystack_size_respected(self):
        prompts = _mod._make_niah_training_prompts(
            4, seed=1, haystack_min_lines=10, haystack_max_lines=12,
        )
        for p in prompts:
            # Count haystack lines: split on newlines, drop the
            # introductory + trailing question blocks.
            body = p.split("\n\n", 1)[1].rsplit("\n\n", 1)[0]
            n_lines = len(body.split("\n"))
            assert 10 <= n_lines <= 12, f"got {n_lines}"

    def test_no_eval_seed_collision(self):
        """The trainer uses seed = args.seed + 1000 to avoid colliding
        with the eval's needle generator. Verify the trainer's NIAH
        prompts at seed 1000+default are not byte-identical to a
        trivially-seeded set the eval might use (seed 0 or 42)."""
        train_seed_default = 0 + 1000  # default training NIAH seed
        train_prompts = _mod._make_niah_training_prompts(
            10, seed=train_seed_default,
        )
        eval_seed_42 = _mod._make_niah_training_prompts(10, seed=42)
        eval_seed_0 = _mod._make_niah_training_prompts(10, seed=0)
        assert train_prompts != eval_seed_42
        assert train_prompts != eval_seed_0
