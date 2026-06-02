"""Shared fixtures for server-side tests on Linux.

PR-N3 retired the previously-housed ``DeterministicEngine`` and
``DeterministicTokenizer`` test doubles plus their fixtures
(``tokenizer``, ``short_engine``, ``long_engine``). The HTTP shim's
runtime tests moved to ``tests/integration/test_http_shim_real.py``;
the engine + tokenizer wrapper tests moved to
``tests/integration/test_engine_real.py`` and
``tests/integration/test_tokenizer_real.py``.

What stays on Linux: the ``_reset_sse_starlette_app_status``
autouse fixture below, which fixes a sse-starlette / pytest-asyncio
event-loop-binding interaction that would otherwise corrupt async
streaming tests in the (still Linux-runnable) test_streaming.py.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# sse-starlette compatibility shim
#
# ``sse_starlette.sse.EventSourceResponse`` lazily creates a class-level
# ``anyio.Event`` (``AppStatus.should_exit_event``) the first time it is
# instantiated. The Event is bound to whichever asyncio event loop is
# running at that moment. pytest-asyncio (>=0.21, with the function-
# scoped event-loop default in 1.x) creates a fresh event loop per
# async test, so the *second* SSE-driven test inherits an Event bound
# to a now-closed loop and the SSE response raises::
#
#     RuntimeError: <asyncio.locks.Event ...> is bound to a different event loop
#
# The fix is to reset the class-level cached Event before every test
# that exercises the SSE app. Production code is unaffected — uvicorn
# stays on a single loop for the lifetime of the process and the lazy
# init runs exactly once there.
#
# We make this autouse at the package level so any future SSE test in
# this directory picks the reset up automatically.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sse_starlette_app_status():
    """Reset ``sse_starlette`` shutdown event between async tests.

    Without this, only the first async streaming test in a session
    succeeds; subsequent tests fail because the cached anyio.Event is
    still bound to the previous, now-closed event loop.
    """
    try:
        from sse_starlette.sse import AppStatus  # type: ignore
    except ImportError:  # pragma: no cover - sse_starlette is a hard dep
        yield
        return
    AppStatus.should_exit_event = None
    AppStatus.should_exit = False
    try:
        yield
    finally:
        AppStatus.should_exit_event = None
        AppStatus.should_exit = False
