"""Unit tests for `kv_cache_proposer.baseline.BaselineDecoder`."""

from __future__ import annotations

import pytest
import torch

from kv_cache_proposer.baseline import BaselineDecoder, BaselineConfig, BaselineRunResult


def test_baseline_loads(baseline_decoder: BaselineDecoder) -> None:
    assert baseline_decoder.tokenizer is not None
    assert baseline_decoder._weight_bytes > 0


def test_baseline_rejects_nonpositive_max_new_tokens(baseline_decoder: BaselineDecoder) -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        baseline_decoder.generate([1, 2, 3], max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        baseline_decoder.generate([1, 2, 3], max_new_tokens=-5)


def test_baseline_generates_and_grows_kv(
    baseline_decoder: BaselineDecoder, short_chat_messages
) -> None:
    prompt = baseline_decoder.tokenizer.apply_chat_template(
        short_chat_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    result = baseline_decoder.generate(prompt, max_new_tokens=4)
    assert isinstance(result, BaselineRunResult)
    assert 1 <= len(result.output_token_ids) <= 4
    assert result.forward_calls >= 1
    assert result.peak_kv_bytes > 0
    assert result.final_kv_bytes > 0
    assert result.weight_bytes == baseline_decoder._weight_bytes
    assert result.final_kv_token_count == len(prompt) + len(result.output_token_ids) \
        if result.output_token_ids[-1] not in {baseline_decoder.tokenizer.eos_token_id, baseline_decoder.tokenizer.convert_tokens_to_ids("<|im_end|>")} \
        else result.final_kv_token_count == len(prompt) + len(result.output_token_ids) - 1
    # Sanity: peak_kv_bytes is monotone non-decreasing
    assert result.peak_kv_bytes >= result.final_kv_bytes \
        or result.peak_kv_bytes == result.final_kv_bytes


def test_baseline_eos_stops_generation(
    baseline_decoder: BaselineDecoder, short_chat_messages
) -> None:
    prompt = baseline_decoder.tokenizer.apply_chat_template(
        short_chat_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    eos = baseline_decoder.tokenizer.convert_tokens_to_ids("<|im_end|>")
    result = baseline_decoder.generate(prompt, max_new_tokens=64, eos_token_ids=[eos])
    # The response to "Reply with exactly 'OK'." is short; should EOS quickly.
    assert eos in result.output_token_ids
    # Generation should stop at EOS, so output ends with EOS.
    eos_idx = result.output_token_ids.index(eos)
    assert eos_idx == len(result.output_token_ids) - 1
    # Each non-EOS generated token consumes one extra forward beyond prefill.
    assert result.forward_calls == eos_idx + 1


def test_baseline_kv_bytes_static_helper(baseline_decoder: BaselineDecoder) -> None:
    # Build a fresh DynamicCache via prefill, then read it via the static helper.
    cache_pre = baseline_decoder._kv_bytes
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=baseline_decoder.model.config)
    assert cache_pre(cache) == 0  # empty cache has zero bytes
