from scripts.agent_gan_inference_demo import (
    _agent_cache_gate,
    _output_metadata,
)


def test_agent_gate_requires_remote_warmup_and_primary_hot_inference():
    warm = {"remote_jobs": 1, "remote_hits": 1}
    actual = {
        "local_hits": 1,
        "remote_jobs": 0,
        "tokens_computed": 0,
        "fallbacks": 0,
    }
    assert _agent_cache_gate(warm, actual)
    assert not _agent_cache_gate({**warm, "remote_jobs": 0}, actual)
    assert not _agent_cache_gate(warm, {**actual, "local_hits": 0})
    assert not _agent_cache_gate(warm, {**actual, "fallbacks": 1})


def test_agent_output_report_is_redacted():
    result = _output_metadata("private model output")
    assert result["output_chars"] == 20
    assert len(result["output_hash"]) == 64
    assert "output" not in result
