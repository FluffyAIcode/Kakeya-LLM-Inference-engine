"""Tests for `inference_engine.backends.mlx.verifier.MLXSinkWindowVerifier`.

Mac-only: every test requires `mlx`, `mlx_lm`, and a working Metal
device. The real Qwen3-1.7B weights are loaded once per session via
the `mlx_verifier_session` fixture (model load is the slow step on
M-series, ~1.5 s; subsequent tests reuse it).

The headline correctness check is
`test_mlx_argmax_matches_pytorch_baseline`, which prefills both the
PyTorch and MLX verifiers with the same prompt and asserts their
argmax-of-next-token agrees. bf16 numerical noise across the two
backends could in principle flip an argmax — if it does we want to
know, so the test asserts equality (not approximate).
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

import torch

from kv_cache_proposer.verifier import VerifierConfig
from inference_engine.backends.mlx.verifier import (
    MLXSinkWindowVerifier,
    _model_weight_bytes,
    _map_torch_dtype_to_mx,
)


@pytest.fixture(scope="session")
def mlx_verifier_session() -> MLXSinkWindowVerifier:
    return MLXSinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=4, window_size=64,
        )
    )


def _build_mlx_verifier(sink: int = 4, window: int = 64) -> MLXSinkWindowVerifier:
    return MLXSinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=sink, window_size=window,
        )
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_config_loads(mlx_verifier_session: MLXSinkWindowVerifier) -> None:
    v = mlx_verifier_session
    assert v.config.sink_size == 4
    assert v.config.window_size == 64
    assert v.cache is None
    assert v.next_token_logits is None
    assert v.cache_logical_size == 0
    assert v.next_global_position == 0
    assert v.stats.weight_bytes > 0


@pytest.mark.parametrize(
    "sink,window,err",
    [
        (-1, 8, "sink_size must be >= 0"),
        (4, 0, "window_size must be > 0"),
    ],
)
def test_construction_validates_window_args(sink, window, err) -> None:
    with pytest.raises(ValueError, match=err):
        MLXSinkWindowVerifier(
            VerifierConfig(
                dtype=torch.bfloat16, device="cpu",
                sink_size=sink, window_size=window,
            )
        )


def test_construction_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="no MLX equivalent"):
        MLXSinkWindowVerifier(
            VerifierConfig(
                dtype=torch.float64, device="cpu",  # fp64 not in our table
                sink_size=4, window_size=8,
            )
        )


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------

def test_prefill_rejects_empty(mlx_verifier_session: MLXSinkWindowVerifier) -> None:
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        mlx_verifier_session.prefill([])


def test_prefill_under_budget() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    prompt = list(range(20))
    v.prefill(prompt)
    assert v.cache_logical_size == 20
    assert v.next_global_position == 20
    assert v.next_token_logits is not None
    assert v.next_token_logits.shape[-1] > 1000  # vocab size
    assert v.stats.forward_calls == 1


def test_prefill_over_budget_triggers_trim() -> None:
    v = _build_mlx_verifier(sink=4, window=8)
    v.prefill(list(range(50)))
    assert v.cache_logical_size == 12
    # All non-null layers must reflect the trimmed size physically.
    for layer in v.cache:
        if layer.keys is None:
            continue
        assert int(layer.keys.shape[2]) == 12


def test_prefill_zero_sink() -> None:
    v = _build_mlx_verifier(sink=0, window=8)
    v.prefill(list(range(20)))
    assert v.cache_logical_size == 8


# ---------------------------------------------------------------------------
# forward_block + commit_or_truncate
# ---------------------------------------------------------------------------

def test_forward_block_requires_prefill() -> None:
    v = _build_mlx_verifier()
    with pytest.raises(RuntimeError, match="not prefilled"):
        v.forward_block([1, 2, 3])


def test_forward_block_rejects_empty() -> None:
    v = _build_mlx_verifier()
    v.prefill([1, 2, 3])
    with pytest.raises(ValueError, match="tokens must be non-empty"):
        v.forward_block([])


def test_forward_block_returns_per_position_logits() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    L = 5
    block = list(range(100, 100 + L))
    logits = v.forward_block(block)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == L
    assert logits.shape[1] > 1000  # vocab


def test_commit_validates_args() -> None:
    v = _build_mlx_verifier()
    v.prefill([1, 2, 3])
    v.forward_block([4, 5, 6])
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        v.commit_or_truncate(forwarded=3, accepted=-1)
    with pytest.raises(ValueError, match="0 <= accepted <= forwarded"):
        v.commit_or_truncate(forwarded=3, accepted=4)


def test_commit_full_accept_no_drop() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102])
    v.commit_or_truncate(forwarded=3, accepted=3)
    assert v.cache_logical_size == 13
    assert v.next_global_position == 13


def test_commit_partial_accept_drops_tail() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102])
    v.commit_or_truncate(forwarded=3, accepted=1)
    assert v.cache_logical_size == 11
    assert v.next_global_position == 11


def test_commit_zero_accept_drops_all() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102])
    v.commit_or_truncate(forwarded=3, accepted=0)
    assert v.cache_logical_size == 10


def test_commit_post_trims_to_budget() -> None:
    v = _build_mlx_verifier(sink=4, window=8)  # budget = 12
    v.prefill(list(range(10)))
    v.forward_block([100, 101, 102, 103, 104])  # logical -> 15, then trim
    v.commit_or_truncate(forwarded=5, accepted=5)
    assert v.cache_logical_size == 12


# ---------------------------------------------------------------------------
# append_token
# ---------------------------------------------------------------------------

def test_append_token_advances_state() -> None:
    v = _build_mlx_verifier(sink=4, window=64)
    v.prefill(list(range(10)))
    pre_size = v.cache_logical_size
    pre_pos = v.next_global_position
    logits = v.append_token(123)
    assert v.cache_logical_size == pre_size + 1
    assert v.next_global_position == pre_pos + 1
    assert logits is v.next_token_logits
    assert logits.ndim == 1


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def test_trim_raises_when_no_cache() -> None:
    v = _build_mlx_verifier()
    with pytest.raises(RuntimeError, match="No cache to trim"):
        v._trim_cache_in_place()


def test_record_peak_kv_handles_null_cache() -> None:
    v = _build_mlx_verifier()
    pre = v.stats.peak_kv_bytes
    v._record_peak_kv()  # cache is None
    assert v.stats.peak_kv_bytes == pre


def test_record_peak_activation_grows_only() -> None:
    v = _build_mlx_verifier()
    a = mx.zeros((1, 4, 32), dtype=mx.bfloat16)
    b = mx.zeros((1, 8, 32), dtype=mx.bfloat16)
    v._record_peak_activation(a)
    pa1 = v.stats.peak_activation_bytes
    v._record_peak_activation(b)
    pa2 = v.stats.peak_activation_bytes
    v._record_peak_activation(a)
    assert pa1 > 0 and pa2 > pa1
    assert v.stats.peak_activation_bytes == pa2  # smaller doesn't regress


def test_reset_clears_state() -> None:
    v = _build_mlx_verifier()
    v.prefill([1, 2, 3])
    v.reset()
    assert v.next_token_logits is None
    assert v.cache_logical_size == 0
    assert v.next_global_position == 0
    assert v.cache is not None  # reset re-creates an empty cache list


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_model_weight_bytes_positive(mlx_verifier_session) -> None:
    bytes_ = _model_weight_bytes(mlx_verifier_session.model)
    # Qwen3-1.7B ≈ 3.4 GB at bf16 ≈ 3.4e9 bytes
    assert bytes_ > 1_000_000_000


@pytest.mark.parametrize(
    "torch_dtype,mx_dtype",
    [
        (torch.bfloat16, mx.bfloat16),
        (torch.float16, mx.float16),
        (torch.float32, mx.float32),
    ],
)
def test_dtype_mapping_supported(torch_dtype, mx_dtype) -> None:
    assert _map_torch_dtype_to_mx(torch_dtype) == mx_dtype


def test_dtype_mapping_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="no MLX equivalent"):
        _map_torch_dtype_to_mx(torch.float64)


# ---------------------------------------------------------------------------
# Headline cross-backend correctness
# ---------------------------------------------------------------------------

def test_mlx_argmax_matches_pytorch_baseline() -> None:
    """MLX verifier's first-token argmax must equal the PyTorch
    verifier's first-token argmax for the same prompt.

    bf16 logits computed on Metal vs PyTorch CPU can differ in
    individual values, but Qwen3-1.7B's argmax margin is large enough
    on chat prompts that the picks agree. This is the primary
    correctness gate for the MLX backend before anyone wires it into
    the speculative decoder.
    """
    from kv_cache_proposer.verifier import SinkWindowVerifier

    cpu_v = SinkWindowVerifier(
        VerifierConfig(
            dtype=torch.bfloat16, device="cpu",
            sink_size=4, window_size=64,
        )
    )
    mlx_v = _build_mlx_verifier(sink=4, window=64)

    # Build identical prompt token ids via the verifier's own tokenizer.
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Reply with exactly: OK."},
    ]
    cpu_ids = cpu_v.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    # mlx_lm's TokenizerWrapper exposes apply_chat_template too. We
    # don't require its ids to match cpu_ids byte-for-byte (the wrapper
    # may add minor surface differences); we feed the same `cpu_ids`
    # token list to both verifiers below.
    cpu_v.prefill(cpu_ids)
    mlx_v.prefill(cpu_ids)

    cpu_argmax = int(torch.argmax(cpu_v.next_token_logits).item())
    mlx_argmax = int(torch.argmax(mlx_v.next_token_logits).item())
    assert cpu_argmax == mlx_argmax, (
        f"first-token argmax differs across backends: "
        f"cpu={cpu_argmax}  mlx={mlx_argmax}"
    )
