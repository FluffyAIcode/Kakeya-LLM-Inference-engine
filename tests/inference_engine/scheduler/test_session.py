"""Unit tests for :class:`Session`."""

from __future__ import annotations

import pytest

from inference_engine.scheduler.session import Session, SessionState


def test_construction_defaults_to_pending():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    assert s.state is SessionState.PENDING
    assert s.is_terminal is False
    assert s.id.startswith("sess-")


def test_construction_rejects_empty_prompt():
    with pytest.raises(ValueError, match="prompt_ids must be non-empty"):
        Session(prompt_ids=[], max_new_tokens=10, eos_token_ids=[0])


def test_construction_rejects_zero_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        Session(prompt_ids=[1], max_new_tokens=0, eos_token_ids=[0])


def test_construction_rejects_negative_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        Session(prompt_ids=[1], max_new_tokens=-1, eos_token_ids=[0])


def test_construction_rejects_empty_eos():
    with pytest.raises(ValueError, match="eos_token_ids must be non-empty"):
        Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[])


def test_admit_transition():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    assert s.state is SessionState.ADMITTED
    assert s.admitted_at is not None
    assert s.is_terminal is False


def test_admit_twice_raises():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    with pytest.raises(RuntimeError, match="cannot admit"):
        s.mark_admitted()


def test_complete_transition():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    s.mark_completed()
    assert s.state is SessionState.COMPLETED
    assert s.is_terminal is True
    assert s.finished_at is not None


def test_cancel_transition():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    s.mark_cancelled()
    assert s.state is SessionState.CANCELLED
    assert s.is_terminal is True


def test_failed_transition_records_error():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    err = RuntimeError("boom")
    s.mark_failed(err)
    assert s.state is SessionState.FAILED
    assert s.error is err


def test_double_finalize_raises():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    s.mark_completed()
    with pytest.raises(RuntimeError, match="already finalized"):
        s.mark_cancelled()


def test_double_cancel_raises():
    s = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    s.mark_admitted()
    s.mark_cancelled()
    with pytest.raises(RuntimeError, match="already finalized"):
        s.mark_failed(RuntimeError("x"))


def test_unique_session_ids():
    a = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    b = Session(prompt_ids=[1], max_new_tokens=10, eos_token_ids=[0])
    assert a.id != b.id
