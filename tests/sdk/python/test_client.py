"""Unit tests for :class:`kakeya.Client` (PR-B4).

Tests run against a real :func:`runtime_address` fixture (background-
thread async server), so the SDK exercises actual gRPC machinery
end-to-end.
"""

from __future__ import annotations

import pytest

from kakeya import (
    Client,
    DEFAULT_ADDRESS,
    ResourceExhaustedError,
    Session,
)


class TestConstruction:
    def test_default_address_constant(self):
        assert DEFAULT_ADDRESS == "localhost:50051"

    def test_address_property(self):
        client = Client("127.0.0.1:99999")
        assert client.address == "127.0.0.1:99999"
        client.close()

    def test_closed_property_default_false(self):
        client = Client("127.0.0.1:99999")
        assert client.closed is False
        client.close()

    def test_channel_options_keyword_accepted(self):
        # We pass a benign option to confirm the path; the option's
        # effect is grpcio's domain.
        client = Client(
            "127.0.0.1:99999",
            channel_options=[("grpc.enable_retries", 0)],
        )
        client.close()


class TestClose:
    def test_close_flips_closed_flag(self):
        client = Client("127.0.0.1:99999")
        client.close()
        assert client.closed is True

    def test_close_is_idempotent(self):
        client = Client("127.0.0.1:99999")
        client.close()
        client.close()  # must not raise
        assert client.closed is True


class TestContextManager:
    def test_with_block_closes_on_exit(self):
        with Client("127.0.0.1:99999") as client:
            assert client.closed is False
        assert client.closed is True

    def test_context_manager_closes_on_exception(self):
        client = None
        with pytest.raises(RuntimeError, match="boom"):
            with Client("127.0.0.1:99999") as c:
                client = c
                raise RuntimeError("boom")
        assert client is not None and client.closed is True


class TestCreateSession:
    def test_returns_session_with_server_issued_id(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            assert isinstance(session, Session)
            assert session.session_id.startswith("sess-")
            session.close()

    def test_eos_token_ids_passed_through(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session(eos_token_ids=[7, 11, 13])
            store_session = runtime_address.store.get_session(
                session.session_id,
            )
            assert store_session.eos_token_ids == (7, 11, 13)
            session.close()

    def test_client_label_passed_through(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session(client_label="demo-1")
            store_session = runtime_address.store.get_session(
                session.session_id,
            )
            assert store_session.client_label == "demo-1"
            session.close()

    def test_default_args_produce_empty_eos(self, runtime_address):
        with Client(runtime_address.address) as client:
            session = client.create_session()
            store_session = runtime_address.store.get_session(
                session.session_id,
            )
            assert store_session.eos_token_ids == ()
            assert store_session.client_label == ""
            session.close()

    def test_resource_exhausted_raises_typed_exception(self):
        # Use a runtime with capacity > num_slabs so the second
        # create_session can't be satisfied by LRU eviction (still
        # within capacity) and the slab pool exhausts. The fixture-
        # constructed runtime accepts a slab_pool kwarg so we can
        # build this scenario without re-implementing the thread/
        # loop dance inline.
        from inference_engine.memory.pool import SlabPool
        from inference_engine.memory.slab import SlabConfig
        from tests.sdk.python.conftest import _start_runtime, _stop_runtime

        cfg = SlabConfig(
            num_layers=1, num_heads=1, sink_size=1,
            window_size=2, head_dim=4,
        )
        pool = SlabPool(num_slabs=1, slab_config=cfg)
        fixture, thread, loop, holder = _start_runtime(
            cache_inspector_enabled=False,
            slab_pool=pool,
            capacity=4,
        )
        try:
            with Client(fixture.address) as client:
                client.create_session()  # consumes the only slab
                with pytest.raises(ResourceExhaustedError) as exc:
                    client.create_session()  # pool empty
                assert exc.value.rpc_code is not None
                assert "slab pool exhausted" in str(exc.value)
        finally:
            _stop_runtime(thread, loop, holder)
