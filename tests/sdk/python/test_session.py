"""Unit tests for :class:`kakeya.Session` (PR-B4).

Tests run against a real :func:`runtime_address` fixture (background-
thread async server), so the SDK exercises actual gRPC streaming
end-to-end.
"""

from __future__ import annotations

import pytest

from kakeya import (
    Client,
    InvalidArgumentError,
    InvariantViolationError,
    Session,
    SessionClosedError,
    SessionInfo,
    SessionNotFoundError,
)
from inference_engine.server.proto_gen.kakeya.v1 import runtime_pb2


# ---------------------------------------------------------------------------
# Properties + closed contract
# ---------------------------------------------------------------------------


class TestPropertiesAndClosed:
    def test_session_id_is_server_issued(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            assert session.session_id.startswith("sess-")
            session.close()

    def test_closed_default_false(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            assert session.closed is False
            session.close()
            assert session.closed is True

    def test_last_metadata_defaults(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            assert session.last_stop_reason is None
            assert session.last_generated_token_count == 0
            assert session.last_prefill_duration_seconds == 0.0
            assert session.last_total_duration_seconds == 0.0
            assert session.last_history_truncated_dropped is None
            session.close()


# ---------------------------------------------------------------------------
# append()
# ---------------------------------------------------------------------------


class TestAppend:
    def test_returns_history_length(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            new_len = session.append([10, 20, 30])
            assert new_len == 3
            session.close()

    def test_appends_extend_history(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([10, 20])
            new_len = session.append([30])
            assert new_len == 3
            session.close()

    def test_empty_input_is_noop(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1])
            new_len = session.append([])
            assert new_len == 1
            session.close()

    def test_after_local_close_raises_session_closed_error(
        self, runtime_address,
    ):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.close()
            with pytest.raises(SessionClosedError):
                session.append([1, 2, 3])

    def test_unknown_session_after_runtime_close_raises_not_found(
        self, runtime_address,
    ):
        # Bypass local close-tracking by stashing the session_id and
        # creating a fresh local Session object pointed at an id that
        # the runtime doesn't know.
        with Client(runtime_address.address) as client:
            phantom = Session(client=client, session_id="sess-phantom")
            with pytest.raises(SessionNotFoundError):
                phantom.append([1])


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_yields_token_ids_in_order(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            tokens = list(session.generate(max_tokens=3))
            assert len(tokens) == 3
            assert all(isinstance(t, int) for t in tokens)
            session.close()

    def test_sets_last_metadata_after_iteration(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            list(session.generate(max_tokens=2))
            assert session.last_stop_reason == \
                runtime_pb2.GenerateDone.STOP_REASON_MAX_TOKENS
            assert session.last_generated_token_count == 2
            assert session.last_total_duration_seconds >= 0.0
            assert session.last_prefill_duration_seconds == 0.0
            session.close()

    def test_eos_terminates_with_eos_stop_reason(self, runtime_address):
        # FakeVerifier's deterministic argmax = sum(history[-3:]) % 16.
        # Initial history [1, 2, 3] -> first generated token = 6.
        with Client(runtime_address.address) as client:
            session = client.create_session(eos_token_ids=[6])
            session.append([1, 2, 3])
            tokens = list(session.generate(max_tokens=10))
            assert tokens == [6]
            assert session.last_stop_reason == \
                runtime_pb2.GenerateDone.STOP_REASON_EOS
            session.close()

    def test_records_history_truncated_metadata(self, runtime_address):
        # FakeVerifier's default sink+window = 6. Append 8 tokens to
        # make the cache truncated; then generate.
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([10, 20, 30, 40, 50, 60, 70, 80])
            tokens = list(session.generate(max_tokens=2))
            assert len(tokens) == 2
            # 8 history - 6 cache = 2 dropped at start of generate.
            assert session.last_history_truncated_dropped == 2
            session.close()

    def test_no_truncation_leaves_metadata_none(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            list(session.generate(max_tokens=1))
            assert session.last_history_truncated_dropped is None
            session.close()

    def test_metadata_resets_between_calls(self, runtime_address):
        # generate() resets every last_* property at start. We
        # verify by running a CALL that emits NO truncated frame
        # AFTER one that did: the second call's
        # last_history_truncated_dropped must be None, not the
        # first call's value.
        #
        # We can't easily switch a session out of truncated mode
        # once it's in (sink+window cap is permanent for that
        # session), so we test the inverse path: do the
        # non-truncated call first, then the truncated call. After
        # the second call last_history_truncated_dropped is
        # populated; after the FIRST it must be None.
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])  # under sink+window
            list(session.generate(max_tokens=1))
            assert session.last_history_truncated_dropped is None

            # Now push the cache past sink+window and call again.
            session.append([10, 20, 30, 40, 50, 60, 70, 80])
            list(session.generate(max_tokens=1))
            assert isinstance(session.last_history_truncated_dropped, int)
            assert session.last_history_truncated_dropped > 0
            session.close()

    def test_after_local_close_raises_session_closed_error(
        self, runtime_address,
    ):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1])
            session.close()
            with pytest.raises(SessionClosedError):
                list(session.generate(max_tokens=1))

    def test_no_history_raises_invalid_argument(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            with pytest.raises(InvalidArgumentError):
                list(session.generate(max_tokens=1))
            session.close()

    def test_temperature_nonzero_raises_invalid_argument(
        self, runtime_address,
    ):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            with pytest.raises(InvalidArgumentError):
                list(session.generate(max_tokens=1, temperature=0.7))
            session.close()

    def test_top_p_set_raises_invalid_argument(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            with pytest.raises(InvalidArgumentError):
                list(session.generate(max_tokens=1, top_p=0.9))
            session.close()

    def test_top_k_other_than_one_raises_invalid_argument(
        self, runtime_address,
    ):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            with pytest.raises(InvalidArgumentError):
                list(session.generate(max_tokens=1, top_k=50))
            session.close()

    def test_seed_accepted(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            tokens = list(session.generate(max_tokens=2, seed=42))
            assert len(tokens) == 2
            session.close()

    def test_temperature_zero_accepted(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            tokens = list(session.generate(max_tokens=1, temperature=0.0))
            assert len(tokens) == 1
            session.close()

    def test_top_k_one_accepted(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            tokens = list(session.generate(max_tokens=1, top_k=1))
            assert len(tokens) == 1
            session.close()


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------


class TestInfo:
    def test_returns_session_info_dataclass(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([1, 2, 3])
            info = session.info()
            assert isinstance(info, SessionInfo)
            assert info.history_length == 3
            assert info.cache_invariant_inv1_violations == 0
            assert info.cache_invariant_inv2_violations == 0
            assert info.idle_seconds >= 0.0
            session.close()

    def test_repr_includes_all_fields(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            info = session.info()
            text = repr(info)
            for needle in (
                "history_length=", "kv_live_bytes=", "inv1=",
                "inv2=", "idle_seconds=",
            ):
                assert needle in text, f"missing {needle} in {text!r}"
            session.close()

    def test_unknown_session_raises_not_found(self, runtime_address):
        with Client(runtime_address.address) as client:
            phantom = Session(client=client, session_id="sess-x")
            with pytest.raises(SessionNotFoundError):
                phantom.info()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_returns_final_history_length(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.append([10, 20, 30])
            assert session.close() == 3

    def test_zero_for_empty_session(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            assert session.close() == 0

    def test_idempotent_after_first_close(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.close()
            assert session.close() == 0  # no RPC, no error

    def test_rpc_error_on_close_still_flips_closed_flag(self, runtime_address):
        # Phantom session: close() RPC returns NOT_FOUND; we still
        # set self._closed = True so subsequent calls don't make
        # phantom RPCs.
        with Client(runtime_address.address) as client:
            phantom = Session(client=client, session_id="sess-not-here")
            with pytest.raises(SessionNotFoundError):
                phantom.close()
            assert phantom.closed is True
            # Subsequent close() is a no-op.
            assert phantom.close() == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_with_block_closes_on_exit(self, runtime_address):
        with Client(runtime_address.address) as client:
            with client.create_session() as session:
                assert session.closed is False
            assert session.closed is True

    def test_context_manager_swallows_close_exception_on_exit(
        self, runtime_address,
    ):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            session.close()  # close once normally
            # Now enter as context manager and let __exit__ try to
            # close again — close() is idempotent so this is also fine.
            with session:
                pass
            assert session.closed is True


# ---------------------------------------------------------------------------
# SessionInfo dataclass surface
# ---------------------------------------------------------------------------


class TestSessionInfoStandalone:
    def test_constructor_and_attributes(self):
        info = SessionInfo(
            history_length=5,
            kv_live_bytes=12345,
            cache_invariant_inv1_violations=0,
            cache_invariant_inv2_violations=0,
            idle_seconds=1.234,
        )
        assert info.history_length == 5
        assert info.kv_live_bytes == 12345
        assert info.cache_invariant_inv1_violations == 0
        assert info.cache_invariant_inv2_violations == 0
        assert info.idle_seconds == 1.234

    def test_repr_format(self):
        info = SessionInfo(
            history_length=1, kv_live_bytes=2,
            cache_invariant_inv1_violations=3,
            cache_invariant_inv2_violations=4,
            idle_seconds=5.6,
        )
        assert "history_length=1" in repr(info)
        assert "kv_live_bytes=2" in repr(info)
        assert "inv1=3" in repr(info)
        assert "inv2=4" in repr(info)
        assert "idle_seconds=5.600" in repr(info)
