"""Kakeya Python SDK — public API surface (PR-B4 of ADR 0008).

Two top-level types power the entire surface:

  * :class:`Client` — connection to a Kakeya RuntimeService.
  * :class:`Session` — handle to one server-side session.

Plus a typed exception hierarchy rooted at :class:`KakeyaError`
(see :mod:`kakeya.errors`) that maps every gRPC status code from
the runtime to a Python class.

Example::

    from kakeya import Client

    with Client("localhost:50051") as client:
        with client.create_session(eos_token_ids=[151645]) as session:
            session.append([10, 20, 30])
            for token_id in session.generate(max_tokens=64):
                print(token_id)

Tokenization is intentionally NOT part of the SDK core — per
ADR 0008 §2.4 / §3.4, the runtime treats token ids as opaque
integers; rendering messages to tokens is the application's
responsibility (or an opt-in helper that lives in
``sdks/python/examples/``). Importing :mod:`transformers` and
calling ``apply_chat_template`` is one valid path; using a custom
serializer is another. The SDK takes no position.
"""

from kakeya.client import DEFAULT_ADDRESS, Client
from kakeya.errors import (
    InvalidArgumentError,
    InvariantViolationError,
    KakeyaError,
    ResourceExhaustedError,
    RpcCancelledError,
    SessionClosedError,
    SessionNotFoundError,
    UnimplementedError,
)
from kakeya.session import Session, SessionInfo

__all__ = [
    "Client",
    "DEFAULT_ADDRESS",
    "InvalidArgumentError",
    "InvariantViolationError",
    "KakeyaError",
    "ResourceExhaustedError",
    "RpcCancelledError",
    "Session",
    "SessionClosedError",
    "SessionInfo",
    "SessionNotFoundError",
    "UnimplementedError",
]
