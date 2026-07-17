from autoresearch.prefill.prepare import evaluate


class Candidate:
    PREFILL_COMPUTE_CHUNK_TOKENS = 256
    SNAPSHOT_MODE = "final_only"
    MAX_SEGMENT_SECONDS = 300


def _report(**overrides):
    stage = {
        "name": "agent_critic",
        "ok": True,
        "complete": True,
        "prefix_tokens": 1000,
        "warmup_wall_s": 1000,
        "review_scope": "full",
        "generator_full_tokens": 900,
        "critic_context_tokens": 900,
        "critic_omitted_tokens": 0,
        "critic_protocol": "recursive_proof_decomposition_v2",
        "delta": {"fallbacks": 0, "remote_job_failures": 0},
    }
    stage.update(overrides)
    return {"stages": [stage]}


def test_autoresearch_accepts_faster_full_context_candidate():
    result = evaluate(_report(), Candidate)
    assert result["accepted"]
    assert result["metric_cold_critic_prefill_s"] == 1000
    assert result["estimated_max_segment_s"] == 256
    assert all(result["constraints"].values())


def test_autoresearch_rejects_slow_segment_or_semantic_regression():
    slow = type("Slow", (), {
        "PREFILL_COMPUTE_CHUNK_TOKENS": 512,
        "SNAPSHOT_MODE": "final_only",
        "MAX_SEGMENT_SECONDS": 300,
    })
    assert not evaluate(_report(), slow)["accepted"]
    assert not evaluate(
        _report(critic_omitted_tokens=1),
        Candidate,
    )["accepted"]
    assert not evaluate(
        _report(delta={"fallbacks": 1, "remote_job_failures": 1}),
        Candidate,
    )["accepted"]
