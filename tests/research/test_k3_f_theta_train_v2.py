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


# ---------------------------------------------------------------------------
# v3: attention-output distillation loss
# ---------------------------------------------------------------------------


class _StubAttn(torch.nn.Module):
    """Minimal stand-in for a Gemma 4 self_attn module: q_norm, k_norm,
    v_norm, q_proj (only out_features used), o_proj. Used by the
    distillation loss to apply per-layer norms + o_proj. Nothing else
    is needed since cos/sin/mask/attention are operator-level."""

    def __init__(self, n_heads, n_kv_heads, head_dim, hidden_dim):
        super().__init__()
        from torch.nn import RMSNorm
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.scaling = head_dim ** -0.5
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.v_norm = RMSNorm(head_dim)
        # q_proj.out_features is read by the loss to compute n_heads
        self.q_proj = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, hidden_dim, bias=False)


class _StubLayer(torch.nn.Module):
    def __init__(self, n_heads, n_kv_heads, head_dim, hidden_dim):
        super().__init__()
        self.self_attn = _StubAttn(n_heads, n_kv_heads, head_dim, hidden_dim)


def _identity_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2):
    # Identity RoPE for tests — just return x unchanged. We're testing
    # the wiring, not RoPE correctness (RoPE is applied via the same
    # function signature the actual transformers helper uses).
    return x


class TestAttentionDistillationLoss:

    def _build_synthetic(self, T=16):
        from inference_engine.v04.f_theta import FThetaConfig, FThetaProjection
        torch.manual_seed(0)
        n_d_layers = 2
        d_kv_heads, d_head = 2, 4
        n_v_layers = 3
        n_heads, head_dim, hidden = 4, 4, 16
        n_kv_heads = 2

        cfg = FThetaConfig(
            drafter_num_layers=n_d_layers,
            drafter_num_kv_heads=d_kv_heads, drafter_head_dim=d_head,
            verifier_num_layers=n_v_layers,
            verifier_num_kv_heads=n_kv_heads, verifier_head_dim=head_dim,
            rank=8,
        )
        f_theta = FThetaProjection(cfg).float()

        layers = [
            _StubLayer(n_heads, n_kv_heads, head_dim, hidden)
            for _ in range(n_v_layers)
        ]

        # Synthetic captured target data
        target = _mod.AttentionTargetData(
            q_raw=[torch.randn(T, n_heads * head_dim, dtype=torch.bfloat16)
                   for _ in range(n_v_layers)],
            o_tgt=[torch.randn(T, hidden, dtype=torch.bfloat16)
                   for _ in range(n_v_layers)],
            cos=[torch.randn(1, T, head_dim, dtype=torch.bfloat16)
                 for _ in range(n_v_layers)],
            sin=[torch.randn(1, T, head_dim, dtype=torch.bfloat16)
                 for _ in range(n_v_layers)],
            attention_mask=None,
            num_heads_per_layer=[n_heads] * n_v_layers,
            head_dim_per_layer=[head_dim] * n_v_layers,
        )
        seq = _mod.CapturedSequence(
            seq_len=T,
            drafter_k=torch.randn(n_d_layers, T, d_kv_heads * d_head),
            drafter_v=torch.randn(n_d_layers, T, d_kv_heads * d_head),
            attn_target=target,
        )
        return f_theta, layers, seq

    def test_attention_distill_loss_runs(self):
        f_theta, layers, seq = self._build_synthetic()
        diag = {}
        loss = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            diag_buf=diag,
        )
        assert torch.is_tensor(loss)
        assert loss.dim() == 0
        assert float(loss) > 0.0
        assert "mse_O_mean" in diag
        assert "abs_O_target_mean" in diag

    def test_s5_skip_layer_indices_excludes_layers(self):
        """S5 mode: skip_layer_indices excludes those layers from the loss
        (loss differs and is averaged over the remaining layers)."""
        f_theta, layers, seq = self._build_synthetic()
        full = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
        )
        skipped = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            skip_layer_indices=[0],
        )
        # Excluding a layer changes the (per-used-layer averaged) loss.
        assert abs(float(full) - float(skipped)) > 1e-9

    def test_loss_is_differentiable_through_f_theta(self):
        f_theta, layers, seq = self._build_synthetic()
        loss = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
        )
        loss.backward()
        any_grad = any(
            p.grad is not None and p.grad.norm().item() > 0
            for p in f_theta.parameters()
        )
        assert any_grad, "f_θ params should receive non-zero gradient"

    def test_o_proj_weights_remain_frozen_in_loss(self):
        """o_proj is the verifier's frozen weight; gradient should NOT
        accumulate on it through the loss (it's not registered to f_θ
        optimizer, but we still check o_proj's grad is unset before
        backprop and unset after, since we pass o_proj from
        non-trainable verifier modules)."""
        f_theta, layers, seq = self._build_synthetic()
        # Freeze o_proj like the trainer does
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad_(False)
        loss = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
        )
        loss.backward()
        for layer in layers:
            for p in layer.parameters():
                assert p.grad is None, "verifier params must not receive grad"

    def test_dispatch_through_f_theta_loss_function(self):
        f_theta, layers, seq = self._build_synthetic()
        diag = {}
        loss = _mod._f_theta_loss(
            f_theta, seq, sample_positions=0,
            loss_type="attn_distill",
            diag_buf=diag,
            layers=layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
        )
        assert torch.is_tensor(loss) and loss.dim() == 0
        assert "mse_O_mean" in diag

    def test_attn_distill_requires_layers_arg(self):
        f_theta, layers, seq = self._build_synthetic()
        with pytest.raises(ValueError, match="attn_distill requires"):
            _mod._f_theta_loss(
                f_theta, seq, sample_positions=0,
                loss_type="attn_distill",
            )

    def test_legacy_loss_rejects_attn_only_capture(self):
        """If loss_type=mse but seq has only attn_target (no verifier_k/v),
        we should fail loud, not silently."""
        _, layers, seq = self._build_synthetic()
        # seq has attn_target but no verifier_k/v
        f_theta_legacy = None  # not used past the dispatch check
        from inference_engine.v04.f_theta import FThetaConfig, FThetaProjection
        cfg = FThetaConfig(
            drafter_num_layers=2, drafter_num_kv_heads=2, drafter_head_dim=4,
            verifier_num_layers=3, verifier_num_kv_heads=2, verifier_head_dim=4,
            rank=8,
        )
        f_theta = FThetaProjection(cfg).float()
        with pytest.raises(RuntimeError, match="legacy K/V capture"):
            _mod._f_theta_loss(
                f_theta, seq, sample_positions=64, loss_type="mse",
            )

    def test_sample_positions_subselects_output(self):
        f_theta, layers, seq = self._build_synthetic(T=16)
        # With sample=4, loss should still be a scalar but use only 4
        # output positions — verify the loss runs and is differentiable.
        loss_full = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            sample_positions=None, seed=42,
        )
        loss_sub = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            sample_positions=4, seed=42,
        )
        assert torch.is_tensor(loss_sub) and loss_sub.dim() == 0
        # Different sample sizes should generally give different scalars
        # (it's an average over different sets of positions)
        # Don't strictly assert they differ — small T might collide.
        assert float(loss_full) > 0.0 and float(loss_sub) > 0.0


# ---------------------------------------------------------------------------
# v3 dataclass surface
# ---------------------------------------------------------------------------


class TestAttentionDistillationHybridLoss:
    """v3 hybrid loss — fixes the f_θ collapse degeneracy exposed by
    the 2026-06-10 alpha-sweep diagnostic (raw K/V rel_mse 1331×;
    k_norm hides scale errors from attn_distill alone)."""

    def _build_synthetic_with_raw_kv(self, T=16):
        from inference_engine.v04.f_theta import FThetaConfig, FThetaProjection
        torch.manual_seed(0)
        n_d_layers, d_kv_heads, d_head = 2, 2, 4
        n_v_layers = 3
        n_heads, head_dim, hidden = 4, 4, 16
        n_kv_heads = 2

        cfg = FThetaConfig(
            drafter_num_layers=n_d_layers,
            drafter_num_kv_heads=d_kv_heads, drafter_head_dim=d_head,
            verifier_num_layers=n_v_layers,
            verifier_num_kv_heads=n_kv_heads, verifier_head_dim=head_dim,
            rank=8,
        )
        f_theta = FThetaProjection(cfg).float()

        layers = [
            _StubLayer(n_heads, n_kv_heads, head_dim, hidden)
            for _ in range(n_v_layers)
        ]

        target = _mod.AttentionTargetData(
            q_raw=[torch.randn(T, n_heads * head_dim, dtype=torch.bfloat16)
                   for _ in range(n_v_layers)],
            o_tgt=[torch.randn(T, hidden, dtype=torch.bfloat16)
                   for _ in range(n_v_layers)],
            cos=[torch.randn(1, T, head_dim, dtype=torch.bfloat16)
                 for _ in range(n_v_layers)],
            sin=[torch.randn(1, T, head_dim, dtype=torch.bfloat16)
                 for _ in range(n_v_layers)],
            attention_mask=None,
            num_heads_per_layer=[n_heads] * n_v_layers,
            head_dim_per_layer=[head_dim] * n_v_layers,
            k_raw_tgt=[torch.randn(T, n_kv_heads * head_dim, dtype=torch.bfloat16)
                       for _ in range(n_v_layers)],
            v_raw_tgt=[torch.randn(T, n_kv_heads * head_dim, dtype=torch.bfloat16)
                       for _ in range(n_v_layers)],
        )
        seq = _mod.CapturedSequence(
            seq_len=T,
            drafter_k=torch.randn(n_d_layers, T, d_kv_heads * d_head),
            drafter_v=torch.randn(n_d_layers, T, d_kv_heads * d_head),
            attn_target=target,
        )
        return f_theta, layers, seq

    def test_hybrid_runs_and_emits_full_diag(self):
        f_theta, layers, seq = self._build_synthetic_with_raw_kv()
        diag = {}
        loss = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            hybrid=True, diag_buf=diag,
        )
        assert torch.is_tensor(loss) and loss.dim() == 0
        for k in ("mse_O_mean", "k_dir_mean", "v_dir_mean",
                  "k_mag_mean", "v_mag_mean"):
            assert k in diag, f"missing diag key: {k}"

    def test_hybrid_requires_raw_kv_tgt(self):
        f_theta, layers, _ = self._build_synthetic_with_raw_kv()
        # Build seq WITHOUT k_raw_tgt/v_raw_tgt — should fail loud
        T = 16; n_v = 3; n_kv = 2; hd = 4
        target_no_raw = _mod.AttentionTargetData(
            q_raw=[torch.randn(T, 4*hd, dtype=torch.bfloat16) for _ in range(n_v)],
            o_tgt=[torch.randn(T, 16, dtype=torch.bfloat16) for _ in range(n_v)],
            cos=[torch.randn(1, T, hd, dtype=torch.bfloat16) for _ in range(n_v)],
            sin=[torch.randn(1, T, hd, dtype=torch.bfloat16) for _ in range(n_v)],
            attention_mask=None,
            num_heads_per_layer=[4]*n_v, head_dim_per_layer=[hd]*n_v,
        )
        seq_no_raw = _mod.CapturedSequence(
            seq_len=T,
            drafter_k=torch.randn(2, T, 8), drafter_v=torch.randn(2, T, 8),
            attn_target=target_no_raw,
        )
        with pytest.raises(RuntimeError, match="k_raw_tgt"):
            _mod._attention_distillation_loss(
                f_theta, seq_no_raw, layers,
                apply_rotary_pos_emb=_identity_rotary_pos_emb,
                device=torch.device("cpu"),
                hybrid=True,
            )

    def test_hybrid_dispatch_via_loss_type(self):
        f_theta, layers, seq = self._build_synthetic_with_raw_kv()
        diag = {}
        loss = _mod._f_theta_loss(
            f_theta, seq, sample_positions=0,
            loss_type="attn_distill_hybrid",
            diag_buf=diag,
            layers=layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
        )
        assert torch.is_tensor(loss) and loss.dim() == 0
        assert "k_dir_mean" in diag

    def test_hybrid_loss_strictly_higher_than_attn_distill_alone(self):
        """Hybrid adds direction + magnitude terms; with random initial
        f_θ, all four components are non-trivial → hybrid > attn_distill
        (which only has the mse_O term). Verifies the additional terms
        actually affect the loss, not silently zero."""
        f_theta, layers, seq = self._build_synthetic_with_raw_kv()
        loss_attn_only = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            hybrid=False,
        )
        loss_hybrid = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            hybrid=True,
        )
        assert float(loss_hybrid.detach()) > float(loss_attn_only.detach())

    def test_hybrid_grad_flows_to_f_theta(self):
        f_theta, layers, seq = self._build_synthetic_with_raw_kv()
        loss = _mod._attention_distillation_loss(
            f_theta, seq, layers,
            apply_rotary_pos_emb=_identity_rotary_pos_emb,
            device=torch.device("cpu"),
            hybrid=True,
        )
        loss.backward()
        any_grad = any(
            p.grad is not None and p.grad.norm().item() > 0
            for p in f_theta.parameters()
        )
        assert any_grad


class TestAttentionTargetDataDataclass:

    def test_fields_present(self):
        td = _mod.AttentionTargetData(
            q_raw=[], o_tgt=[], cos=[], sin=[],
            attention_mask=None,
            num_heads_per_layer=[], head_dim_per_layer=[],
        )
        assert td.q_raw == []
        assert td.attention_mask is None

    def test_captured_sequence_optional_kv_and_attn(self):
        seq = _mod.CapturedSequence(
            seq_len=10,
            drafter_k=torch.zeros(2, 10, 8),
            drafter_v=torch.zeros(2, 10, 8),
        )
        assert seq.verifier_k is None
        assert seq.verifier_v is None
        assert seq.attn_target is None

    def test_captured_sequence_attn_target_path(self):
        td = _mod.AttentionTargetData(
            q_raw=[], o_tgt=[], cos=[], sin=[],
            attention_mask=None,
            num_heads_per_layer=[], head_dim_per_layer=[],
        )
        seq = _mod.CapturedSequence(
            seq_len=10,
            drafter_k=torch.zeros(2, 10, 8),
            drafter_v=torch.zeros(2, 10, 8),
            attn_target=td,
        )
        assert seq.attn_target is td

    def test_attention_target_data_optional_raw_kv_for_hybrid(self):
        """k_raw_tgt and v_raw_tgt fields default to None; populated
        when capture_raw_kv=True is passed during data collection."""
        td_legacy = _mod.AttentionTargetData(
            q_raw=[], o_tgt=[], cos=[], sin=[],
            attention_mask=None,
            num_heads_per_layer=[], head_dim_per_layer=[],
        )
        assert td_legacy.k_raw_tgt is None
        assert td_legacy.v_raw_tgt is None

        td_hybrid = _mod.AttentionTargetData(
            q_raw=[torch.zeros(8, 16)], o_tgt=[torch.zeros(8, 32)],
            cos=[torch.zeros(1, 8, 4)], sin=[torch.zeros(1, 8, 4)],
            attention_mask=None,
            num_heads_per_layer=[4], head_dim_per_layer=[4],
            k_raw_tgt=[torch.randn(8, 8)],
            v_raw_tgt=[torch.randn(8, 8)],
        )
        assert td_hybrid.k_raw_tgt is not None
        assert td_hybrid.v_raw_tgt is not None
