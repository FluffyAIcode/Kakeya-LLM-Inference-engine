"""API key authentication.

Optional Bearer-token authentication for the OpenAI-compatible routes.
When :attr:`ServerConfig.api_keys` is non-empty, every request to
``/v1/*`` must carry an ``Authorization: Bearer <key>`` header whose
key is in the configured set; otherwise the request is rejected with
``401 authentication_error`` in the OpenAI envelope.

Public routes (``/healthz``, ``/metrics``) are exempt from auth so
liveness probes and Prometheus scrapers do not need credentials.

When :attr:`ServerConfig.api_keys` is empty (the default), all routes
are open. This is the v0.1.0 behavior — single-user dev mode does
not need keys.

Why a custom dependency rather than FastAPI's ``HTTPBearer``:

  * We want consistent OpenAI-shaped 401 bodies (FastAPI's bearer
    default returns ``{"detail": "Not authenticated"}``).
  * We want a dynamic exemption list (public routes), which is
    cleanest as a request-path predicate.
  * We want to verify against an explicit set rather than calling
    out to an external auth service.

The dependency is registered via ``app.dependency_overrides`` only
when ``api_keys`` is non-empty; otherwise the route table is
unchanged. This matches "pay-for-what-you-use" — single-user mode
has zero auth overhead.
"""

from __future__ import annotations

from typing import FrozenSet, Optional

from fastapi import HTTPException, Request, status


_PUBLIC_PATHS = frozenset({"/healthz", "/metrics"})


def is_public_path(path: str) -> bool:
    """Return True if ``path`` is exempt from auth.

    OPTIONS preflight requests are also exempt — the auth middleware
    sees them too but we don't want to return 401 on a CORS probe.
    """
    return path in _PUBLIC_PATHS


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Parse ``Authorization: Bearer <token>``; return the token or None.

    Returns ``None`` (rather than raising) so the caller can decide
    how to respond. Tolerates extra whitespace and case-insensitive
    "Bearer".
    """
    if authorization_header is None:
        return None
    raw = authorization_header.strip()
    if not raw:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    if not token:  # pragma: no cover - defense in depth; split() guarantees non-empty
        # split(None, 1) on a stripped string can only return a 2-element
        # list whose second element starts with a non-whitespace
        # character. Empty token is therefore unreachable through any
        # normal header. We keep the check as defense-in-depth in case
        # split semantics ever change.
        return None
    return token


def verify_api_key(
    request: Request,
    *,
    valid_keys: FrozenSet[str],
) -> None:
    """Ensure the request carries a valid API key, or raise 401.

    Public paths (see :func:`is_public_path`) and ``OPTIONS``
    preflight requests pass through unchanged. Otherwise the
    ``Authorization`` header is parsed; missing / malformed header
    or unrecognized key both raise ``HTTPException(401)``.

    ``valid_keys`` is taken as a parameter rather than read from
    request state so this function is a pure callable easy to test
    in isolation.
    """
    if not valid_keys:
        # Auth disabled.
        return
    if is_public_path(request.url.path):
        return
    if request.method.upper() == "OPTIONS":
        return
    token = extract_bearer_token(request.headers.get("authorization"))
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": 'Bearer realm="kakeya"'},
        )
    if token not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": 'Bearer realm="kakeya"'},
        )
