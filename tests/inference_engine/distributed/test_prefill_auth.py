from __future__ import annotations

import pytest

from inference_engine.distributed.prefill_auth import (
    FleetAuthConfig,
    PrefillAuthError,
    metadata_pairs,
    signed_metadata,
    verify_metadata,
)
from inference_engine.server.proto_gen.kakeya.v1 import distributed_pb2


def _request():
    return distributed_pb2.FetchBlocksRequest(lease_id="lease")


def _config():
    return FleetAuthConfig(b"x" * 32, "tenant-a", "node-a", 30)


def test_sign_and_verify_round_trip():
    request = _request()
    metadata = signed_metadata(request, _config(), now=100)
    assert verify_metadata(metadata, request, _config(), now=110) == (
        "tenant-a", "node-a",
    )


def test_tenant_hash_keys_are_isolated():
    a = _config().tenant_hash_key()
    b = FleetAuthConfig(b"x" * 32, "tenant-b", "node-a").tenant_hash_key()
    assert a != b


@pytest.mark.parametrize("mutator,match", [
    (lambda md: tuple((k, "tenant-b" if k.endswith("tenant-id") else v)
                      for k, v in md), "tenant mismatch"),
    (lambda md: tuple((k, "bad" if k.endswith("auth-mac") else v)
                      for k, v in md), "invalid prefill authentication MAC"),
    (lambda md: tuple((k, v) for k, v in md if not k.endswith("auth-mac")),
     "missing prefill authentication metadata"),
])
def test_auth_rejects_bad_metadata(mutator, match):
    request = _request()
    with pytest.raises(PrefillAuthError, match=match):
        verify_metadata(
            mutator(signed_metadata(request, _config(), now=100)),
            request,
            _config(),
            now=100,
        )


def test_auth_rejects_replay_and_bad_timestamp():
    request = _request()
    metadata = signed_metadata(request, _config(), now=100)
    with pytest.raises(PrefillAuthError, match="replay window"):
        verify_metadata(metadata, request, _config(), now=200)
    bad = tuple(("x-kakeya-auth-ts", "nan") if k == "x-kakeya-auth-ts"
                else (k, v) for k, v in metadata)
    with pytest.raises(PrefillAuthError, match="invalid authentication timestamp"):
        verify_metadata(bad, request, _config(), now=100)


def test_auth_rejects_tampered_request():
    request = _request()
    metadata = signed_metadata(request, _config(), now=100)
    with pytest.raises(PrefillAuthError, match="invalid prefill authentication MAC"):
        verify_metadata(
            metadata,
            distributed_pb2.FetchBlocksRequest(lease_id="other"),
            _config(),
            now=100,
        )


def test_config_validation_and_file(tmp_path):
    with pytest.raises(ValueError):
        FleetAuthConfig(b"short", "t", "n")
    with pytest.raises(ValueError):
        FleetAuthConfig(b"x" * 32, "", "n")
    with pytest.raises(ValueError):
        FleetAuthConfig(b"x" * 32, "t", "")
    with pytest.raises(ValueError):
        FleetAuthConfig(b"x" * 32, "t", "n", 0)
    path = tmp_path / "psk"
    path.write_bytes(b"y" * 32 + b"\n")
    assert FleetAuthConfig.from_file(
        str(path), tenant_id="t", node_id="n",
    ).psk == b"y" * 32


def test_metadata_pairs_accepts_tuples():
    assert metadata_pairs((("a", "b"),)) == (("a", "b"),)

    class Item:
        key = "c"
        value = "d"

    assert metadata_pairs((Item(),)) == (("c", "d"),)


def test_sign_rejects_non_protobuf():
    with pytest.raises(TypeError):
        signed_metadata(object(), _config())

