"""OpenAI-compatible error envelope and FastAPI exception handlers.

The OpenAI API wraps every error in the same JSON shape::

    {
      "error": {
        "message": "<human-readable description>",
        "type": "<class>",
        "code": "<machine-readable code>" | null,
        "param": "<offending parameter>" | null
      }
    }

Off-the-shelf OpenAI clients (Python SDK, JS SDK, vercel/ai-sdk)
parse this envelope and surface ``error.message`` to the user. If we
return FastAPI's default ``{"detail": "..."}`` shape they fall back
to a generic "request failed" message and lose the actionable
information.

This module provides:

  * :func:`build_error_envelope` — serializer.
  * :func:`http_exception_handler` —
    converts ``starlette.exceptions.HTTPException`` raised inside
    routes into the OpenAI envelope, preserving the status code.
  * :func:`request_validation_exception_handler` — converts
    ``fastapi.exceptions.RequestValidationError`` (the 422 produced
    by pydantic on bad request bodies) into the envelope.
  * :func:`unhandled_exception_handler` — last-resort 500 wrapper
    so a route bug doesn't leak a Python traceback over the wire.

Mapping of status code to ``error.type``:

    400 invalid_request_error      (chat-template / encoding failed)
    401 authentication_error
    403 permission_error
    404 not_found_error
    422 invalid_request_error
    429 rate_limit_error
    500 server_error
    other 4xx invalid_request_error
    other 5xx server_error
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException


_STATUS_TO_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
}


def build_error_envelope(
    *,
    message: str,
    status_code: int,
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> dict:
    """Construct the OpenAI-shaped error body for ``status_code``.

    ``code`` and ``param`` mirror the OpenAI fields; both default to
    ``None`` (rendered as JSON ``null``).
    """
    if status_code in _STATUS_TO_TYPE:
        err_type = _STATUS_TO_TYPE[status_code]
    elif 400 <= status_code < 500:
        err_type = "invalid_request_error"
    else:
        err_type = "server_error"
    return {
        "error": {
            "message": message,
            "type": err_type,
            "code": code,
            "param": param,
        }
    }


async def http_exception_handler(
    request: Request, exc: HTTPException,
) -> JSONResponse:
    """Wrap an ``HTTPException`` into the OpenAI envelope."""
    detail = exc.detail
    # detail is usually a string; if a route raised with a dict (rare
    # but legal in starlette) we serialize it to the message field.
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_envelope(
            message=detail, status_code=exc.status_code,
        ),
        headers=exc.headers,
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    """Wrap a 422 from pydantic into the OpenAI envelope.

    ``exc.errors()`` returns a list of dicts describing each
    validation failure. We emit the first one's message + offending
    parameter; multi-error requests get a summary.
    """
    errors = exc.errors()
    if not errors:  # pragma: no cover - pydantic always yields >=1 error
        message = "request validation failed"
        param: Optional[str] = None
    else:
        first = errors[0]
        loc = first.get("loc") or ()
        # Drop the leading "body" / "query" / "path" prefix for clarity.
        loc_path = ".".join(str(p) for p in loc[1:]) or None
        param = loc_path
        if len(errors) > 1:
            message = (
                f"{first.get('msg', 'invalid request')} "
                f"(and {len(errors) - 1} more validation error(s))"
            )
        else:
            message = first.get("msg", "invalid request")
    return JSONResponse(
        status_code=422,
        content=build_error_envelope(
            message=message, status_code=422, param=param,
        ),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception,
) -> JSONResponse:
    """Last-resort 500 wrapper for unexpected exceptions in route handlers.

    We deliberately do not include the traceback in the response —
    that's a security risk. The exception still propagates to the
    server's structured logs (if logging is configured).
    """
    return JSONResponse(
        status_code=500,
        content=build_error_envelope(
            message=f"unhandled server error: {type(exc).__name__}",
            status_code=500,
        ),
    )
