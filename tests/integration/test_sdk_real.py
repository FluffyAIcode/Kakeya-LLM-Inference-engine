"""Integration tests for the Kakeya Python SDK against a real runtime.

PR-N4 migration of the former Linux-side ``test_client.py`` and
``test_session.py`` (which used a ``FakeVerifier`` / ``_MinimalVerifierStub``
test mirror behind a background-thread gRPC server). The SDK's
truth is wire-layer correctness — gRPC encode/decode, status code
mapping, streaming order, lifecycle. This file drives a real
runtime backed by Qwen3-0.6B; the SDK exercises the same wire
contract it would exercise in production.

What stays on Linux: ``tests/sdk/python/test_errors.py`` — pure
``_wrap_grpc_error`` mapping with synthesized ``grpc.RpcError``
objects; no server / verifier needed.
"""

from __future__ import annotations

import grpc
import pytest

from kakeya import Client
from kakeya.errors import (
    InvalidArgumentError,
    SessionNotFoundError,
)


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


class TestClient:
    def test_client_create_session_returns_session_with_server_id(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            session = client.create_session()
            try:
                assert isinstance(session.session_id, str)
                assert len(session.session_id) > 0
            finally:
                session.close()

    def test_client_create_session_with_eos_token_ids(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            session = client.create_session(eos_token_ids=[0, 7, 42])
            try:
                info = session.info()
                # Server records eos_token_ids; round-trip via info
                # is implicit because session is alive and well.
                assert info.history_length == 0
            finally:
                session.close()

    def test_client_close_idempotent(self, real_grpc_runtime_address):
        client = Client(real_grpc_runtime_address)
        client.close()
        client.close()  # second close is a no-op
        assert client.closed

    def test_client_address_property(self, real_grpc_runtime_address):
        with Client(real_grpc_runtime_address) as client:
            assert client.address == real_grpc_runtime_address


# ---------------------------------------------------------------------------
# Session.append + Session.generate end-to-end
# ---------------------------------------------------------------------------


class TestSession:
    def test_append_then_generate_yields_tokens(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            with client.create_session() as session:
                session.append([1, 2, 3])
                tokens = list(session.generate(max_tokens=4))
                # At least one token; iterator is exhausted by [DONE].
                assert len(tokens) >= 1
                # Metadata available after iteration.
                assert session.last_stop_reason is not None
                assert session.last_generated_token_count == len(tokens)

    def test_session_info_reports_history_after_append(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            with client.create_session() as session:
                session.append([10, 20, 30])
                info = session.info()
                assert info.history_length == 3

    def test_session_close_returns_final_history_length(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            session = client.create_session()
            session.append([10, 20])
            final = session.close()
            assert final == 2

    def test_session_close_is_idempotent_locally(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            session = client.create_session()
            session.append([1])
            session.close()
            assert session.close() == 0  # second close is local no-op


# ---------------------------------------------------------------------------
# Error mapping (gRPC status → typed Python class) end-to-end
# ---------------------------------------------------------------------------


class TestErrorsEndToEnd:
    def test_unknown_session_raises_session_not_found(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            from kakeya.session import Session
            phantom = Session(client=client, session_id="sess-does-not-exist")
            with pytest.raises(SessionNotFoundError):
                phantom.append([1, 2, 3])

    def test_invalid_argument_for_zero_max_tokens(
        self, real_grpc_runtime_address,
    ):
        with Client(real_grpc_runtime_address) as client:
            with client.create_session() as session:
                session.append([1, 2, 3])
                with pytest.raises(InvalidArgumentError):
                    list(session.generate(max_tokens=0))

    def test_session_closed_locally_then_append_raises(
        self, real_grpc_runtime_address,
    ):
        from kakeya.errors import SessionClosedError

        with Client(real_grpc_runtime_address) as client:
            session = client.create_session()
            session.close()
            with pytest.raises(SessionClosedError):
                session.append([1, 2, 3])
