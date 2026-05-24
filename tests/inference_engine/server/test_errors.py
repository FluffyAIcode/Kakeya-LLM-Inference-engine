"""Unit tests for :mod:`inference_engine.server.errors`."""

from __future__ import annotations

import pytest

from inference_engine.server.errors import build_error_envelope


def test_envelope_minimal_shape():
    e = build_error_envelope(message="bad", status_code=400)
    assert e == {
        "error": {
            "message": "bad",
            "type": "invalid_request_error",
            "code": None,
            "param": None,
        }
    }


@pytest.mark.parametrize("code,expected_type", [
    (400, "invalid_request_error"),
    (401, "authentication_error"),
    (403, "permission_error"),
    (404, "not_found_error"),
    (422, "invalid_request_error"),
    (429, "rate_limit_error"),
    (500, "server_error"),
])
def test_envelope_status_to_type_mapping(code, expected_type):
    e = build_error_envelope(message="x", status_code=code)
    assert e["error"]["type"] == expected_type


@pytest.mark.parametrize("code", [402, 405, 418])
def test_envelope_unmapped_4xx_falls_back_to_invalid_request(code):
    e = build_error_envelope(message="x", status_code=code)
    assert e["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("code", [501, 502, 503, 504])
def test_envelope_unmapped_5xx_falls_back_to_server_error(code):
    e = build_error_envelope(message="x", status_code=code)
    assert e["error"]["type"] == "server_error"


def test_envelope_carries_code_and_param():
    e = build_error_envelope(
        message="x", status_code=400, code="invalid_param", param="messages.0.role",
    )
    assert e["error"]["code"] == "invalid_param"
    assert e["error"]["param"] == "messages.0.role"


# ---------------------------------------------------------------------------
# http_exception_handler / request_validation_exception_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_exception_handler_serializes_non_string_detail():
    """A route that raises ``HTTPException(detail=<dict>)`` should not
    crash the handler — we coerce non-string detail to a string."""
    from starlette.exceptions import HTTPException

    from inference_engine.server.errors import http_exception_handler

    exc = HTTPException(status_code=400, detail={"code": "weird", "field": "x"})
    response = await http_exception_handler(request=None, exc=exc)
    assert response.status_code == 400
    # body is encoded JSON; we don't deeply inspect, just confirm it
    # contains the dict-coerced message.
    body = response.body.decode("utf-8")
    assert "code" in body and "weird" in body


@pytest.mark.asyncio
async def test_request_validation_handler_with_multiple_errors():
    """When pydantic raises with N>1 errors, the message includes a
    summary count."""
    from fastapi.exceptions import RequestValidationError

    from inference_engine.server.errors import (
        request_validation_exception_handler,
    )

    # Construct an exception with 3 errors using pydantic's expected shape.
    errors = [
        {"loc": ("body", "messages"), "msg": "field required", "type": "value_error"},
        {"loc": ("body", "model"), "msg": "field required", "type": "value_error"},
        {"loc": ("body", "max_tokens"), "msg": "must be int", "type": "type_error"},
    ]
    exc = RequestValidationError(errors=errors)
    response = await request_validation_exception_handler(request=None, exc=exc)
    assert response.status_code == 422
    body = response.body.decode("utf-8")
    assert "and 2 more validation error(s)" in body
