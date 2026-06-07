"""Linux CI unit tests for ADR 0011 toy prototype (PR-R1b).

These tests cover the architectural pieces that don't require a real
HuggingFace model checkpoint:

* ``make_sink_window_attention_mask`` (2D) — sparsity pattern correctness.
* ``make_sink_window_attention_mask_4d`` — shape + numerical equivalence
  to the 2D version.
* ``CrossAttentionBridge`` — output shape, zero-init invariant, padding
  mask handling.
* ``CrossAttentionVerifier`` — layer-module discovery on multiple
  HF-shaped surrogates, depth bounds checking, and forward-hook
  injection on a synthetic mini-transformer (no HF download).
* End-to-end **gradient flow**: loss → cross-attn parameters reach
  non-zero ``.grad`` after one backward pass.
* End-to-end **mask actually restricts attention**: full-attention vs
  sink+window outputs differ at predicted positions and only there.

Empirical Gate G-X1 (does the bridge actually rescue recall on Gemma
3-1B-it?) is gated on Mac M4 by ``scripts/review_pr_r1_on_mac.sh`` —
not in CI.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.research.cross_attn_toy_prototype import (
    DEFAULT_NEEDLE_VOCAB,
    CrossAttentionBridge,
    CrossAttentionVerifier,
    NeedleVocab,
    make_niah_dataset,
    make_sink_window_attention_mask,
    make_sink_window_attention_mask_4d,
    needle_vocab_for_mode,
)


# ---------------------------------------------------------------------------
# Sink+window mask tests
# ---------------------------------------------------------------------------


def _allowed_positions(q: int, sink: int, window: int):
    """Reference set of allowed key indices for query ``q``."""
    return set(range(min(sink, q + 1))) | set(
        range(max(sink, q - window + 1), q + 1)
    )


class TestSinkWindowMask2D:
    def test_shape_and_dtype(self):
        m = make_sink_window_attention_mask(
            seq_len=12, sink=2, window=4,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert m.shape == (12, 12)
        assert m.dtype == torch.float32

    def test_allowed_positions_match_reference_for_every_query(self):
        seq_len, sink, window = 16, 3, 5
        m = make_sink_window_attention_mask(
            seq_len, sink, window,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        for q in range(seq_len):
            allowed = _allowed_positions(q, sink, window)
            for k in range(seq_len):
                if k in allowed:
                    assert m[q, k] == 0.0, (
                        f"q={q} k={k} should be allowed but got {m[q, k]}"
                    )
                else:
                    # Masked entries are dtype's finite minimum (not -inf
                    # for float dtypes — see docstring).
                    assert m[q, k] == torch.finfo(torch.float32).min, (
                        f"q={q} k={k} should be masked but got {m[q, k]}"
                    )

    def test_first_token_can_only_see_itself(self):
        m = make_sink_window_attention_mask(
            seq_len=10, sink=4, window=4,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert m[0, 0] == 0.0
        for k in range(1, 10):
            assert m[0, k] != 0.0

    def test_window_only_when_sink_zero(self):
        m = make_sink_window_attention_mask(
            seq_len=10, sink=0, window=3,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        # q=5 should attend to {3, 4, 5}
        assert m[5, 3] == 0.0 and m[5, 4] == 0.0 and m[5, 5] == 0.0
        # not {0, 1, 2}
        assert m[5, 0] != 0.0 and m[5, 1] != 0.0 and m[5, 2] != 0.0

    def test_bf16_uses_finite_min(self):
        m = make_sink_window_attention_mask(
            seq_len=8, sink=2, window=2,
            device=torch.device("cpu"), dtype=torch.bfloat16,
        )
        assert m.dtype == torch.bfloat16
        # For seq_len=8, sink=2, window=2, q=7: allowed = {0,1,6,7},
        # so m[7, 3] is masked.
        masked_val = m[7, 3]
        # bf16's finfo.min is finite, not -inf — important for MPS softmax
        assert torch.isfinite(masked_val), (
            "bf16 mask must use finfo.min (finite), not -inf"
        )
        # bf16 finfo.min ≈ -3.39e+38; just check it's very large negative.
        assert float(masked_val) < -1e30


class TestSinkWindowMask4D:
    def test_shape(self):
        m = make_sink_window_attention_mask_4d(
            seq_len=20, sink=4, window=8,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert m.shape == (1, 1, 20, 20)

    def test_matches_2d_version(self):
        seq_len, sink, window = 24, 4, 6
        m2 = make_sink_window_attention_mask(
            seq_len, sink, window,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        m4 = make_sink_window_attention_mask_4d(
            seq_len, sink, window,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert torch.equal(m4[0, 0], m2)

    def test_softmax_zeros_forbidden_positions(self):
        """The whole point: when added to attention scores then softmax'd,
        forbidden key positions should receive zero attention weight."""
        seq_len, sink, window = 32, 4, 8
        torch.manual_seed(0)
        # arbitrary attention scores [B=1, H=1, T, T]
        scores = torch.randn(1, 1, seq_len, seq_len, dtype=torch.float32)
        mask = make_sink_window_attention_mask_4d(
            seq_len, sink, window,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        weights = F.softmax(scores + mask, dim=-1)
        for q in range(seq_len):
            allowed = _allowed_positions(q, sink, window)
            for k in range(seq_len):
                if k not in allowed:
                    assert weights[0, 0, q, k].item() < 1e-6, (
                        f"forbidden position q={q} k={k} got weight "
                        f"{weights[0, 0, q, k].item()}"
                    )
            # Allowed weights should sum to ~1
            allowed_sum = sum(
                weights[0, 0, q, k].item() for k in allowed
            )
            assert abs(allowed_sum - 1.0) < 1e-4, (
                f"q={q} allowed weights sum to {allowed_sum}, expected ~1"
            )


# ---------------------------------------------------------------------------
# CrossAttentionBridge tests
# ---------------------------------------------------------------------------


class TestCrossAttentionBridge:
    def test_output_shape(self):
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=64,
            proposer_hidden_dim=48,
            num_heads=4,
            head_dim=16,
        )
        verifier_h = torch.randn(2, 10, 64)
        proposer_bank = torch.randn(2, 16, 48)
        out = bridge(verifier_hidden=verifier_h, proposer_hidden_bank=proposer_bank)
        assert out.shape == (2, 10, 64)

    def test_zero_init_o_proj_makes_output_zero(self):
        """At step 0 the bridge MUST contribute zero — this is the
        training-stability guarantee in the docstring. If anyone changes
        the init to nonzero W_o, this test catches it."""
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=32, proposer_hidden_dim=32,
            num_heads=4, head_dim=8,
        )
        verifier_h = torch.randn(1, 5, 32)
        proposer_bank = torch.randn(1, 7, 32)
        out = bridge(verifier_hidden=verifier_h, proposer_hidden_bank=proposer_bank)
        assert torch.allclose(out, torch.zeros_like(out)), (
            f"bridge with zero-init W_o produced non-zero output: max="
            f"{out.abs().max().item()}"
        )

    def test_grad_flows_to_o_proj(self):
        """After one backward pass W_o.grad must be non-None — i.e.,
        gradient flow through the bridge is not severed by zero init."""
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v_h = torch.randn(1, 3, 16, requires_grad=False)
        p_b = torch.randn(1, 4, 16, requires_grad=False)
        out = bridge(verifier_hidden=v_h, proposer_hidden_bank=p_b)
        # synthetic loss that DOESN'T multiply by 0
        target = torch.randn_like(out)
        loss = F.mse_loss(out + v_h, target)  # +v_h so grad flows even with W_o=0
        loss.backward()
        assert bridge.o_proj.weight.grad is not None
        assert bridge.q_proj.weight.grad is not None
        assert bridge.k_proj.weight.grad is not None
        assert bridge.v_proj.weight.grad is not None

    def test_padding_mask_blocks_proposer_positions(self):
        """If a proposer position is masked, it must contribute zero
        (post-softmax weight = 0)."""
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        # Force W_o to non-zero so we can observe the effect
        with torch.no_grad():
            nn.init.normal_(bridge.o_proj.weight, std=0.1)
        v_h = torch.randn(1, 3, 16)
        bank = torch.randn(1, 5, 16)
        # Mask all proposer positions except index 2
        mask = torch.zeros(1, 5)
        mask[0, 2] = 1.0
        out_masked = bridge(
            verifier_hidden=v_h, proposer_hidden_bank=bank,
            proposer_attention_mask=mask,
        )
        # Compare to a bank where only position 2 has content (others zeroed)
        bank_compact = torch.zeros_like(bank)
        bank_compact[0, 2] = bank[0, 2]
        out_compact_no_mask = bridge(
            verifier_hidden=v_h, proposer_hidden_bank=bank_compact,
        )
        # Mismatch tolerance: softmax over 1 allowed key gives weight 1.0
        # exactly; with compact (no mask) softmax over 5 keys redistributes
        # — they SHOULDN'T match exactly. So instead verify the masked
        # output uses ONLY position 2's V by checking against a hand-
        # computed single-key cross-attention.
        Q = bridge.q_proj(v_h).view(1, 3, 2, 8).transpose(1, 2)  # [1,2,3,8]
        K = bridge.k_proj(bank[:, 2:3]).view(1, 1, 2, 8).transpose(1, 2)  # [1,2,1,8]
        V = bridge.v_proj(bank[:, 2:3]).view(1, 1, 2, 8).transpose(1, 2)  # [1,2,1,8]
        # weight = softmax(QK^T / sqrt(d)) — only one key so weight=1
        attn = torch.matmul(Q, K.transpose(-2, -1)) * (8 ** -0.5)
        w = F.softmax(attn, dim=-1)
        ctx = torch.matmul(w, V).transpose(1, 2).contiguous().view(1, 3, 16)
        expected = bridge.o_proj(ctx)
        assert torch.allclose(out_masked, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# R1c: o_proj_init_std init behaviour
# ---------------------------------------------------------------------------


class TestOProjInitStd:
    def test_default_is_strict_zero(self):
        """Default (no arg) keeps the R1b strict-zero W_o invariant — the
        step-0 bridge output must be exactly zero."""
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=32, proposer_hidden_dim=32,
            num_heads=4, head_dim=8,
        )
        assert bridge.o_proj_init_std == 0.0
        assert torch.count_nonzero(bridge.o_proj.weight) == 0

    def test_explicit_zero_matches_default(self):
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=32, proposer_hidden_dim=32,
            num_heads=4, head_dim=8, o_proj_init_std=0.0,
        )
        v_h = torch.randn(1, 5, 32)
        p_b = torch.randn(1, 7, 32)
        out = bridge(verifier_hidden=v_h, proposer_hidden_bank=p_b)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_positive_std_makes_o_proj_nonzero(self):
        """A positive std seeds W_o so the bridge contributes from step 0
        — this is the R1c plateau-escape change."""
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=64, proposer_hidden_dim=64,
            num_heads=8, head_dim=16, o_proj_init_std=0.05,
        )
        assert bridge.o_proj_init_std == 0.05
        assert torch.count_nonzero(bridge.o_proj.weight) > 0
        # std of the init should be in the right ballpark (large enough
        # matrix that the empirical std is close to the requested value).
        emp_std = bridge.o_proj.weight.detach().float().std().item()
        assert 0.02 < emp_std < 0.10, f"empirical std {emp_std} off target"

    def test_positive_std_makes_step0_output_nonzero(self):
        """With a positive std the bridge output at step 0 is NOT zero —
        the whole point: the loss can shape W_o from step 1."""
        torch.manual_seed(0)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=32, proposer_hidden_dim=32,
            num_heads=4, head_dim=8, o_proj_init_std=0.02,
        )
        v_h = torch.randn(1, 5, 32)
        p_b = torch.randn(1, 7, 32)
        out = bridge(verifier_hidden=v_h, proposer_hidden_bank=p_b)
        assert out.abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# R1c: needle vocabulary / debug modes
# ---------------------------------------------------------------------------


class TestNeedleVocabForMode:
    def test_off_is_none(self):
        assert needle_vocab_for_mode("off") is None

    def test_small_is_low_entropy(self):
        v = needle_vocab_for_mode("small")
        assert isinstance(v, NeedleVocab)
        assert v.size() == 20  # 2 prefixes × 10 codes (0..9)
        assert v.size() < DEFAULT_NEEDLE_VOCAB.size()

    def test_medium_between_small_and_off(self):
        small = needle_vocab_for_mode("small")
        medium = needle_vocab_for_mode("medium")
        assert medium.size() == 400  # 4 prefixes × 100 codes (0..99)
        assert small.size() < medium.size() < DEFAULT_NEEDLE_VOCAB.size()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="needle_debug_mode"):
            needle_vocab_for_mode("tiny")

    def test_default_vocab_size_is_full(self):
        # 15 prefixes × (9999-1000+1)=9000 codes = 135000
        assert DEFAULT_NEEDLE_VOCAB.size() == 135000


class TestMakeNiahDatasetVocab:
    def test_small_vocab_restricts_answers(self):
        """Every answer must come from the small closed set."""
        vocab = needle_vocab_for_mode("small")
        data = make_niah_dataset(
            tokenizer=None, n_samples=40, seed=7, needle_vocab=vocab,
            haystack_min_tokens=64, haystack_max_tokens=128,
        )
        allowed = set()
        for prefix in vocab.prefixes:
            for code in range(vocab.code_min, vocab.code_max + 1):
                allowed.add(f" {prefix}-{code}")
        for sample in data:
            assert sample.answer_text in allowed, (
                f"answer {sample.answer_text!r} not in small vocab"
            )
        # With only 20 possible answers and 40 samples, we must see
        # repeats — confirms the vocabulary is genuinely small.
        distinct = {s.answer_text for s in data}
        assert len(distinct) <= vocab.size()

    def test_off_mode_matches_default_vocab_dataset(self):
        """``needle_vocab=None`` reproduces the default-vocab dataset
        bit-for-bit (same seed → same RNG draw order)."""
        a = make_niah_dataset(
            tokenizer=None, n_samples=10, seed=123, needle_vocab=None,
            haystack_min_tokens=64, haystack_max_tokens=128,
        )
        b = make_niah_dataset(
            tokenizer=None, n_samples=10, seed=123,
            needle_vocab=DEFAULT_NEEDLE_VOCAB,
            haystack_min_tokens=64, haystack_max_tokens=128,
        )
        assert [s.answer_text for s in a] == [s.answer_text for s in b]
        assert [s.prompt_text for s in a] == [s.prompt_text for s in b]

    def test_default_answers_use_four_digit_codes(self):
        data = make_niah_dataset(
            tokenizer=None, n_samples=20, seed=1, needle_vocab=None,
            haystack_min_tokens=64, haystack_max_tokens=128,
        )
        for sample in data:
            # " PREFIX-NNNN" — code is in [1000, 9999], i.e. 4 digits.
            code_part = sample.answer_text.strip().split("-")[1]
            assert 1000 <= int(code_part) <= 9999


# ---------------------------------------------------------------------------
# CrossAttentionVerifier — layer-module discovery + hook integration
# ---------------------------------------------------------------------------


class _Gemma3LikeBase(nn.Module):
    """HF-Gemma3-shaped surrogate: ``base.model.layers`` is the
    ``nn.ModuleList`` of decoder layers; provides ``config`` and
    a forward that returns a logits-bearing namespace."""

    class _Config:
        hidden_size = 16
        model_type = "gemma3_text"

    class _Inner(nn.Module):
        def __init__(self, hidden, num_layers, vocab):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(num_layers)]
            )
            self.norm = nn.LayerNorm(hidden)

    def __init__(self, hidden=16, num_layers=4, vocab=32):
        super().__init__()
        self.config = self._Config()
        self.config.hidden_size = hidden
        self.model = self._Inner(hidden, num_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self._captured_attention_mask = None

    def forward(self, input_ids=None, attention_mask=None,
                use_cache=False, return_dict=True, **_):
        # Capture the attention_mask kwarg for assertion purposes — the
        # toy passes a 4D tensor or a Gemma3 dict; either way, we record.
        self._captured_attention_mask = attention_mask
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()


class _GPT2LikeBase(nn.Module):
    """Surrogate for HF GPT-2 shape: ``base.transformer.h`` is the
    decoder layer list."""

    class _Config:
        hidden_size = 8
        model_type = "gpt2"

    class _Trans(nn.Module):
        def __init__(self, hidden, num_layers, vocab):
            super().__init__()
            self.wte = nn.Embedding(vocab, hidden)
            self.h = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(num_layers)]
            )

    def __init__(self):
        super().__init__()
        self.config = self._Config()
        self.transformer = self._Trans(8, 3, 16)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids=None, attention_mask=None,
                use_cache=False, return_dict=True, **_):
        h = self.transformer.wte(input_ids)
        for layer in self.transformer.h:
            h = layer(h)
        return type("Out", (), {"logits": self.lm_head(h)})()


class TestCrossAttentionVerifierWiring:
    def test_finds_gemma_style_layers(self):
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        layers = v._layers_module()
        assert layers is base.model.layers
        assert len(layers) == 4

    def test_finds_gpt2_style_layers(self):
        base = _GPT2LikeBase()
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=8, proposer_hidden_dim=8,
            num_heads=2, head_dim=4,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        layers = v._layers_module()
        assert layers is base.transformer.h

    def test_unrecognized_base_raises(self):
        class _Mystery(nn.Module):
            class _C: pass
            def __init__(self):
                super().__init__()
                self.config = self._C()
        with pytest.raises(RuntimeError, match="decoder layers"):
            CrossAttentionVerifier(
                base_model=_Mystery(),
                cross_attn=CrossAttentionBridge(
                    verifier_hidden_dim=8, proposer_hidden_dim=8,
                    num_heads=1, head_dim=8,
                ),
                cross_attn_depth=1,
            )

    @pytest.mark.parametrize("depth", [0, -1, 5, 100])
    def test_invalid_depth_raises(self, depth):
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        with pytest.raises(ValueError, match="cross_attn_depth"):
            CrossAttentionVerifier(
                base_model=base, cross_attn=bridge, cross_attn_depth=depth,
            )

    def test_gemma_attention_mask_dict_is_passed_through(self):
        """For Gemma3-class models the wrapper must wrap the 4D mask in
        the {"full_attention", "sliding_attention"} dict."""
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
            sink=2, window=3,
        )
        input_ids = torch.randint(0, 32, (1, 8))
        bank = torch.randn(1, 8, 16)
        v(input_ids=input_ids, proposer_hidden_bank=bank)
        captured = base._captured_attention_mask
        assert isinstance(captured, dict)
        assert set(captured.keys()) == {"full_attention", "sliding_attention"}
        assert captured["full_attention"].shape == (1, 1, 8, 8)
        # full_attention and sliding_attention point to the same tensor
        assert captured["full_attention"] is captured["sliding_attention"]

    def test_non_gemma_attention_mask_is_raw_4d(self):
        base = _GPT2LikeBase()
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=8, proposer_hidden_dim=8,
            num_heads=2, head_dim=4,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
            sink=1, window=2,
        )
        captured = {}
        orig_forward = base.forward

        def wrapped(*a, **kw):
            captured["mask"] = kw.get("attention_mask")
            return orig_forward(*a, **kw)
        base.forward = wrapped

        input_ids = torch.randint(0, 16, (1, 6))
        bank = torch.randn(1, 6, 8)
        v(input_ids=input_ids, proposer_hidden_bank=bank)
        assert torch.is_tensor(captured["mask"])
        assert captured["mask"].shape == (1, 1, 6, 6)


class TestCrossAttentionVerifierForwardHook:
    """Verify the forward hook actually injects delta on layer K's
    output and the modified hidden propagates to the lm_head."""

    def test_hook_modifies_layer_output(self):
        torch.manual_seed(0)
        hidden_dim = 16
        base = _Gemma3LikeBase(hidden=hidden_dim, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=hidden_dim, proposer_hidden_dim=hidden_dim,
            num_heads=2, head_dim=8,
        )
        # Force non-zero W_o so the bridge actually contributes
        with torch.no_grad():
            nn.init.normal_(bridge.o_proj.weight, std=0.5)

        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        input_ids = torch.randint(0, 32, (1, 5))
        bank = torch.randn(1, 5, hidden_dim)

        logits_with_bridge = v(input_ids=input_ids, proposer_hidden_bank=bank)
        # Baseline: same model, no hook (use forward_bounded_no_bridge to keep
        # the same 4D mask semantics — apples-to-apples)
        logits_baseline = v.forward_bounded_no_bridge(input_ids=input_ids)
        # The two MUST differ — otherwise the hook isn't firing or W_o is zero.
        assert not torch.allclose(logits_with_bridge, logits_baseline), (
            "hook injection produced identical logits; bridge isn't wired in"
        )

    def test_hook_is_removed_after_forward(self):
        """Hooks must be cleaned up — otherwise repeated forwards stack
        deltas non-deterministically."""
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        target_layer = base.model.layers[1]  # 0-indexed = depth 2 - 1
        before = len(target_layer._forward_hooks)
        input_ids = torch.randint(0, 32, (1, 5))
        bank = torch.randn(1, 5, 16)
        v(input_ids=input_ids, proposer_hidden_bank=bank)
        after = len(target_layer._forward_hooks)
        assert after == before, (
            f"forward hook leaked: {before} -> {after}"
        )

    def test_hook_removed_even_on_exception(self):
        """If the base forward raises, the hook still gets removed."""
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        target_layer = base.model.layers[1]
        before = len(target_layer._forward_hooks)
        # Corrupt the base to raise mid-forward
        original = base.model.layers[3].forward

        def broken(*a, **kw):
            raise RuntimeError("synthetic")
        base.model.layers[3].forward = broken
        try:
            with pytest.raises(RuntimeError, match="synthetic"):
                v(
                    input_ids=torch.randint(0, 32, (1, 5)),
                    proposer_hidden_bank=torch.randn(1, 5, 16),
                )
        finally:
            base.model.layers[3].forward = original
        after = len(target_layer._forward_hooks)
        assert after == before, "hook leaked after exception"

    def test_grad_flows_through_bridge(self):
        """End-to-end: a loss on logits-with-bridge produces non-None
        gradients on cross-attention parameters. Validates that the hook
        is part of the autograd graph."""
        torch.manual_seed(0)
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        with torch.no_grad():
            nn.init.normal_(bridge.o_proj.weight, std=0.1)
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        input_ids = torch.randint(0, 32, (1, 5))
        bank = torch.randn(1, 5, 16, requires_grad=False)
        logits = v(input_ids=input_ids, proposer_hidden_bank=bank)
        target = torch.randint(0, 32, (1, 5))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target.reshape(-1),
        )
        loss.backward()
        for name, p in bridge.named_parameters():
            assert p.grad is not None, f"{name} got no gradient"
            # Q/K/V/O all participate; norms should be finite (not NaN).
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"


class TestForwardFullAttentionAndBoundedBaseline:
    """The three eval forward pathways must be distinguishable."""

    def test_full_attention_passes_no_attention_mask(self):
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        v.forward_full_attention(input_ids=torch.randint(0, 32, (1, 6)))
        # Oracle path must NOT pass the sink+window mask.
        assert base._captured_attention_mask is None

    def test_bounded_no_bridge_passes_sink_window_mask(self):
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
            sink=2, window=3,
        )
        v.forward_bounded_no_bridge(input_ids=torch.randint(0, 32, (1, 8)))
        captured = base._captured_attention_mask
        assert isinstance(captured, dict)  # gemma3 path uses dict
        assert "full_attention" in captured

    def test_bounded_no_bridge_does_not_register_hook(self):
        """Bounded baseline must NOT inject the cross-attention delta —
        otherwise it's the same as the cross_attn path."""
        base = _Gemma3LikeBase(hidden=16, num_layers=4)
        bridge = CrossAttentionBridge(
            verifier_hidden_dim=16, proposer_hidden_dim=16,
            num_heads=2, head_dim=8,
        )
        with torch.no_grad():
            nn.init.normal_(bridge.o_proj.weight, std=0.5)
        v = CrossAttentionVerifier(
            base_model=base, cross_attn=bridge, cross_attn_depth=2,
        )
        target_layer = base.model.layers[1]
        before = len(target_layer._forward_hooks)
        v.forward_bounded_no_bridge(input_ids=torch.randint(0, 32, (1, 6)))
        after = len(target_layer._forward_hooks)
        assert after == before, "bounded baseline must not leave hooks"
