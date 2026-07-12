from inference_engine.distributed.cache_fill import CacheFillCapture


def test_capture_deduplicates_excludes_replay_and_drains():
    capture = CacheFillCapture(max_items=2)
    assert capture.observe(client_label="live", token_ids=[1, 2, 3])
    assert not capture.observe(client_label="live", token_ids=[1, 2, 3])
    assert not capture.observe(client_label="cache-fill-1", token_ids=[4])
    assert capture.stats() == {
        "queued": 1,
        "captured": 1,
        "duplicates": 1,
        "dropped": 0,
        "max_items": 2,
    }
    item = capture.drain(1)[0]
    assert item.token_ids == (1, 2, 3)
    assert item.token_count == 3
    assert capture.stats()["queued"] == 0


def test_capture_is_bounded_and_allows_recapture_after_drain():
    capture = CacheFillCapture(max_items=1)
    assert capture.observe(client_label="a", token_ids=[1])
    assert capture.observe(client_label="b", token_ids=[2])
    assert capture.stats()["dropped"] == 1
    assert capture.drain(1)[0].token_ids == (2,)
    assert capture.observe(client_label="a", token_ids=[1])


def test_capture_validates_limits_and_empty_input():
    try:
        CacheFillCapture(max_items=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected max_items validation")
    capture = CacheFillCapture()
    assert not capture.observe(client_label="live", token_ids=[])
    try:
        capture.drain(0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected drain validation")
