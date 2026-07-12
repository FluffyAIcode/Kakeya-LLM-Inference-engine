from scripts.verify_remote_prefill_e2e import _acceptance


def test_cache_only_replay_is_accepted_without_worker_job():
    accepted, path = _acceptance(
        {"remote_hits": 0, "remote_jobs": 0, "tokens_reused": 0},
        {"remote_hits": 1, "remote_jobs": 0, "tokens_reused": 512},
        minimum_prefix_tokens=512,
        require_worker=False,
    )
    assert accepted
    assert path == "remote_cache"


def test_worker_mode_requires_compute_job_and_remote_import():
    before = {"remote_hits": 0, "remote_jobs": 0, "tokens_reused": 0}
    cache_only = {"remote_hits": 1, "remote_jobs": 0, "tokens_reused": 512}
    assert _acceptance(
        before,
        cache_only,
        minimum_prefix_tokens=512,
        require_worker=True,
    ) == (False, "remote_cache")
    assert _acceptance(
        before,
        {**cache_only, "remote_jobs": 1},
        minimum_prefix_tokens=512,
        require_worker=True,
    ) == (True, "remote_compute")


def test_insufficient_reuse_is_rejected():
    assert _acceptance(
        {"remote_hits": 0, "remote_jobs": 0, "tokens_reused": 0},
        {"remote_hits": 1, "remote_jobs": 0, "tokens_reused": 64},
        minimum_prefix_tokens=128,
        require_worker=False,
    ) == (False, "none")
