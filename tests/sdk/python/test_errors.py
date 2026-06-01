"""Unit tests for :mod:`kakeya.errors` (PR-B4).

The error wrapper :func:`_wrap_grpc_error` is internal but
load-bearing — every gRPC failure on the SDK surface goes through
it. Tests verify the gRPC status -> typed Python class mapping for
every documented status code, plus the catch-all path for unknown
codes.
"""

from __future__ import annotations

import grpc
import pytest

from kakeya.errors import (
    InvalidArgumentError,
    InvariantViolationError,
    KakeyaError,
    ResourceExhaustedError,
    RpcCancelledError,
    SessionClosedError,
    SessionNotFoundError,
    UnimplementedError,
    _wrap_grpc_error,
)


class _SyntheticRpcError(grpc.RpcError):
    """Minimal grpc.RpcError stand-in for unit-testing the error
    wrapper. The real ``_InactiveRpcError`` constructor is private
    in grpcio, so we synthesize the same .code() / .details()
    surface."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class TestKakeyaErrorBase:
    def test_carries_message(self):
        err = KakeyaError("boom")
        assert str(err) == "boom"

    def test_carries_rpc_code_when_provided(self):
        err = KakeyaError("boom", rpc_code=grpc.StatusCode.NOT_FOUND)
        assert err.rpc_code is grpc.StatusCode.NOT_FOUND

    def test_rpc_code_defaults_to_none(self):
        err = KakeyaError("boom")
        assert err.rpc_code is None

    def test_session_closed_error_has_no_rpc_code(self):
        err = SessionClosedError("session closed locally")
        assert err.rpc_code is None
        assert "session closed" in str(err)


class TestSubclassHierarchy:
    @pytest.mark.parametrize("cls", [
        SessionNotFoundError,
        InvalidArgumentError,
        InvariantViolationError,
        ResourceExhaustedError,
        UnimplementedError,
        RpcCancelledError,
        SessionClosedError,
    ])
    def test_subclasses_kakeya_error(self, cls):
        assert issubclass(cls, KakeyaError)


class TestWrapGrpcError:
    @pytest.mark.parametrize("code, expected_cls", [
        (grpc.StatusCode.NOT_FOUND, SessionNotFoundError),
        (grpc.StatusCode.INVALID_ARGUMENT, InvalidArgumentError),
        (grpc.StatusCode.FAILED_PRECONDITION, InvariantViolationError),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, ResourceExhaustedError),
        (grpc.StatusCode.UNIMPLEMENTED, UnimplementedError),
        (grpc.StatusCode.CANCELLED, RpcCancelledError),
    ])
    def test_known_status_maps_to_typed_subclass(self, code, expected_cls):
        synthetic = _SyntheticRpcError(code, "details from server")
        wrapped = _wrap_grpc_error(synthetic)
        assert isinstance(wrapped, expected_cls)
        assert wrapped.rpc_code is code
        assert "details from server" in str(wrapped)

    def test_unknown_status_falls_back_to_kakeya_error(self):
        # Use a code that's not in the documented mapping.
        synthetic = _SyntheticRpcError(
            grpc.StatusCode.INTERNAL, "server exploded",
        )
        wrapped = _wrap_grpc_error(synthetic)
        # Falls back to base KakeyaError (not any specific subclass).
        assert type(wrapped) is KakeyaError
        assert wrapped.rpc_code is grpc.StatusCode.INTERNAL
        assert "server exploded" in str(wrapped)

    def test_empty_details_does_not_crash(self):
        synthetic = _SyntheticRpcError(grpc.StatusCode.NOT_FOUND, "")
        wrapped = _wrap_grpc_error(synthetic)
        assert isinstance(wrapped, SessionNotFoundError)
        assert wrapped.rpc_code is grpc.StatusCode.NOT_FOUND

    def test_rpc_error_without_details_method(self):
        # Real grpc.RpcError instances should always have .details(),
        # but the wrapper handles the bare-RpcError case defensively
        # by falling back to str(exc).
        bare = grpc.RpcError("bare")
        # bare doesn't have code(); we don't go through the wrapper
        # in production for objects like this. But the fallback path
        # in _wrap_grpc_error reads .details() guarded by hasattr —
        # let's confirm bare objects don't reach the wrapper by
        # construction. (Coverage of the hasattr branch is exercised
        # below with a code-only synthetic.)

        class _CodeOnlyError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.NOT_FOUND
            # no details() method

        wrapped = _wrap_grpc_error(_CodeOnlyError("fallback"))
        assert isinstance(wrapped, SessionNotFoundError)
        assert "fallback" in str(wrapped)
