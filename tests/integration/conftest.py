"""Shared fixtures and marker plumbing for the integration suite.

Tests under ``tests/integration/`` exercise the v0.3 runtime against
**real** model weights — typically the same Qwen3-0.6B verifier used
by ``tests/core/``. They are NOT part of the Linux unit-test gate
(coverage is platform-neutral; loading real weights is HF-cache- and
hardware-bound), and are NOT auto-discovered by a bare ``pytest``
invocation: every test in this directory carries the
``@pytest.mark.integration`` marker, and you opt in with::

    pytest -m integration tests/integration/

Per ADR 0008 §9, this suite is the binding GA gate. PR-E2 (a future
PR) will add a self-hosted Mac M4 GitHub Actions workflow that runs
``pytest -m integration`` on every PR labelled ``needs-mac-m4``;
until that workflow lands, contributors run the suite manually on
Mac M4 and push the resulting JSON / JUnit reports to the PR branch.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-mark every test under ``tests/integration/`` with
    ``@pytest.mark.integration`` so contributors don't have to
    repeat the decorator on every test in this directory.

    Standard pytest behavior: tests with this marker run only when
    explicitly selected via ``-m integration``; a bare ``pytest``
    invocation skips them.
    """
    for item in items:
        # str(item.fspath) is reliable across pytest versions; "rootpath"
        # comparisons would also work but require a config dependency.
        if "tests/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
