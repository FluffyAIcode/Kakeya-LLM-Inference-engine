"""Unit tests for :mod:`inference_engine.server.auth`."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from inference_engine.server.auth import (
    extract_bearer_token,
    is_public_path,
    verify_api_key,
)


# ---------------------------------------------------------------------------
# is_public_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/healthz", "/metrics"])
def test_public_paths_are_exempt(path):
    assert is_public_path(path) is True


@pytest.mark.parametrize("path", [
    "/v1/models",
    "/v1/chat/completions",
    "/anything-else",
    "/",
])
def test_other_paths_require_auth(path):
    assert is_public_path(path) is False


# ---------------------------------------------------------------------------
# extract_bearer_token
# ---------------------------------------------------------------------------


def test_extract_well_formed_bearer():
    assert extract_bearer_token("Bearer abc123") == "abc123"


def test_extract_handles_extra_whitespace():
    assert extract_bearer_token("  Bearer   xyz  ") == "xyz"


def test_extract_case_insensitive_scheme():
    assert extract_bearer_token("bearer abc") == "abc"
    assert extract_bearer_token("BEARER abc") == "abc"


def test_extract_returns_none_for_missing_header():
    assert extract_bearer_token(None) is None


def test_extract_returns_none_for_empty_header():
    assert extract_bearer_token("") is None
    assert extract_bearer_token("   ") is None


def test_extract_returns_none_for_non_bearer_scheme():
    assert extract_bearer_token("Basic abc") is None
    assert extract_bearer_token("Token abc") is None


def test_extract_returns_none_for_only_scheme():
    assert extract_bearer_token("Bearer") is None


def test_extract_returns_none_for_only_whitespace_token():
    """``"Bearer    "`` has no second whitespace-delimited token, so
    ``split(None, 1)`` returns a 1-element list and we treat the
    whole header as malformed."""
    assert extract_bearer_token("Bearer    ") is None


def test_extract_returns_none_for_bearer_only_no_token():
    """No second non-whitespace word -> None (caller treats this as
    missing/malformed and returns 401)."""
    assert extract_bearer_token("Bearer ") is None


def test_extract_strips_token():
    assert extract_bearer_token("Bearer  spaced  ") == "spaced"


# ---------------------------------------------------------------------------
# verify_api_key
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for starlette.requests.Request that exposes
    only what verify_api_key reads."""

    def __init__(self, *, path: str, method: str = "GET",
                 authorization: str | None = None) -> None:
        from types import SimpleNamespace

        self.url = SimpleNamespace(path=path)
        self.method = method
        # Headers must be case-insensitive lookup; SimpleNamespace
        # with a get() works for our use.
        h = {} if authorization is None else {"authorization": authorization}
        self.headers = h


def test_verify_no_keys_configured_passes(monkeypatch):
    request = _FakeRequest(path="/v1/chat/completions")
    # Empty set => auth disabled => no error.
    verify_api_key(request, valid_keys=frozenset())


def test_verify_valid_key_succeeds():
    request = _FakeRequest(
        path="/v1/chat/completions",
        authorization="Bearer secret",
    )
    verify_api_key(request, valid_keys=frozenset({"secret"}))


def test_verify_missing_header_raises_401():
    request = _FakeRequest(path="/v1/chat/completions")
    with pytest.raises(HTTPException) as excinfo:
        verify_api_key(request, valid_keys=frozenset({"secret"}))
    assert excinfo.value.status_code == 401
    assert "missing or malformed" in excinfo.value.detail


def test_verify_malformed_header_raises_401():
    request = _FakeRequest(
        path="/v1/chat/completions",
        authorization="Basic abc",
    )
    with pytest.raises(HTTPException) as excinfo:
        verify_api_key(request, valid_keys=frozenset({"secret"}))
    assert excinfo.value.status_code == 401


def test_verify_invalid_key_raises_401():
    request = _FakeRequest(
        path="/v1/chat/completions",
        authorization="Bearer wrong",
    )
    with pytest.raises(HTTPException) as excinfo:
        verify_api_key(request, valid_keys=frozenset({"secret"}))
    assert excinfo.value.status_code == 401
    assert "invalid API key" in excinfo.value.detail


@pytest.mark.parametrize("path", ["/healthz", "/metrics"])
def test_verify_public_paths_exempt_even_when_auth_enabled(path):
    request = _FakeRequest(path=path)
    # No auth header, no key -> still passes because path is public.
    verify_api_key(request, valid_keys=frozenset({"secret"}))


def test_verify_options_request_exempt():
    """CORS preflight always passes auth so browsers can probe."""
    request = _FakeRequest(path="/v1/chat/completions", method="OPTIONS")
    verify_api_key(request, valid_keys=frozenset({"secret"}))


def test_verify_supports_multiple_valid_keys():
    request = _FakeRequest(
        path="/v1/chat/completions",
        authorization="Bearer key2",
    )
    verify_api_key(request, valid_keys=frozenset({"key1", "key2", "key3"}))


def test_verify_includes_www_authenticate_header():
    request = _FakeRequest(path="/v1/chat/completions")
    with pytest.raises(HTTPException) as excinfo:
        verify_api_key(request, valid_keys=frozenset({"secret"}))
    assert excinfo.value.headers is not None
    assert "WWW-Authenticate" in excinfo.value.headers
