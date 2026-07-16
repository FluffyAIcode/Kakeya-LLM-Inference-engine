from scripts.benchmark_prefill_architecture import _delta, _gate


def _stage(**delta):
    values = {
        "remote_jobs": 0,
        "remote_hits": 0,
        "local_hits": 0,
        "tokens_reused": 0,
        "tokens_computed": 0,
        "hot_promotions": 0,
        "fallbacks": 0,
        "remote_job_failures": 0,
    }
    values.update(delta)
    return {"delta": values}


def test_phase_gates_cover_worker_hot_and_offload_paths():
    assert _gate("remote_compute", _stage(
        remote_jobs=1, remote_hits=1, hot_promotions=1,
    )) == (True, "remote_worker")
    assert _gate("primary_hot_hit", _stage(local_hits=1)) == (
        True, "primary_hot",
    )
    assert _gate("allens_cold_restore", _stage(
        remote_hits=1, hot_promotions=1,
    )) == (True, "allens_offload")


def test_phase_gates_reject_primary_prefill_and_failures():
    assert _gate("remote_compute", _stage(tokens_computed=1))[0] is False
    assert _gate("remote_compute", _stage(fallbacks=1))[1] == (
        "fallback_or_remote_failure"
    )
    assert _gate("primary_hot_hit", _stage(remote_hits=1))[0] is False
    assert _gate("allens_cold_restore", _stage(remote_jobs=1))[0] is False


def test_metric_delta_uses_known_phase_keys():
    before = {"local_hits": 1, "remote_hits": 2}
    after = {"local_hits": 3, "remote_hits": 2}
    result = _delta(before, after)
    assert result["local_hits"] == 2
    assert result["remote_hits"] == 0
    assert result["fallbacks"] == 0
