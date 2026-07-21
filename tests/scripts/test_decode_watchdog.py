from scripts.decode_watchdog import evaluate


def test_watchdog_requires_two_identical_stale_decode_observations():
    live = {
        "phase": "decode",
        "pid": 10,
        "session_id": "s",
        "token_index": 4,
        "updated_at_unix": 100.0,
    }
    first, restart = evaluate(live, {}, now=221.0, stall_seconds=120.0)
    assert not restart
    second, restart = evaluate(live, first, now=251.0, stall_seconds=120.0)
    assert restart
    assert second["consecutive"] == 2


def test_watchdog_resets_when_token_progresses_or_phase_is_idle():
    stale = {
        "phase": "decode",
        "pid": 10,
        "session_id": "s",
        "token_index": 4,
        "updated_at_unix": 100.0,
    }
    previous, _ = evaluate(stale, {}, now=221.0, stall_seconds=120.0)
    progressed = dict(stale, token_index=5, updated_at_unix=220.0)
    state, restart = evaluate(
        progressed, previous, now=250.0, stall_seconds=120.0,
    )
    assert not restart
    assert state["consecutive"] == 0
    idle = dict(stale, phase="idle")
    state, restart = evaluate(idle, previous, now=250.0, stall_seconds=120.0)
    assert not restart
    assert state["consecutive"] == 0
