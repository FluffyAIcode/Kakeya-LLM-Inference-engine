import threading

import pytest

from kakeya.errors import InterTokenTimeoutError
from kakeya.session import Session


class BlockingCall:
    def __init__(self):
        self.released = threading.Event()
        self.cancelled = False

    def __iter__(self):
        return self

    def __next__(self):
        self.released.wait(timeout=1)
        raise StopIteration

    def cancel(self):
        self.cancelled = True
        self.released.set()


def test_generate_inter_token_timeout_cancels_stream():
    call = BlockingCall()
    client = type("Client", (), {
        "_stub": type("Stub", (), {"Generate": lambda _self, _request: call})(),
    })()
    session = Session(client=client, session_id="s")
    with pytest.raises(InterTokenTimeoutError, match="no Generate frame"):
        list(session.generate(max_tokens=1, inter_token_timeout_s=0.01))
    assert call.cancelled


def test_generate_rejects_non_positive_inter_token_timeout():
    client = type("Client", (), {"_stub": object()})()
    session = Session(client=client, session_id="s")
    with pytest.raises(ValueError, match="must be > 0"):
        list(session.generate(max_tokens=1, inter_token_timeout_s=0))
