"""Fleet-PSK authentication and tenant isolation for prefill RPCs.

The transport can still be a private-network insecure gRPC channel, but every
request is authenticated before allocation/compute. The signature covers the
deterministic protobuf bytes plus caller/tenant/timestamp metadata. Prefix
hashes use a tenant-derived HMAC key so one tenant cannot probe another
tenant's prompt prefixes.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

AUTH_TENANT = "x-kakeya-tenant-id"
AUTH_NODE = "x-kakeya-node-id"
AUTH_TS = "x-kakeya-auth-ts"
AUTH_MAC = "x-kakeya-auth-mac"


class PrefillAuthError(ValueError):
    """Authentication or replay-window failure."""


@dataclass(frozen=True)
class FleetAuthConfig:
    psk: bytes
    tenant_id: str
    node_id: str
    max_clock_skew_s: float = 60.0

    def __post_init__(self) -> None:
        if len(self.psk) < 16:
            raise ValueError("fleet PSK must be at least 16 bytes")
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.max_clock_skew_s <= 0:
            raise ValueError("max_clock_skew_s must be > 0")

    @classmethod
    def from_file(
        cls, path: str, *, tenant_id: str, node_id: str,
        max_clock_skew_s: float = 60.0,
    ) -> "FleetAuthConfig":
        with open(path, "rb") as fh:
            secret = fh.read().strip()
        return cls(secret, tenant_id, node_id, max_clock_skew_s)

    def tenant_hash_key(self) -> bytes:
        return hmac.new(
            self.psk,
            b"kakeya-prefill-tenant\0" + self.tenant_id.encode(),
            hashlib.sha256,
        ).digest()


def _request_bytes(request) -> bytes:
    serialize = getattr(request, "SerializeToString", None)
    if serialize is None:
        raise TypeError("authenticated request must be a protobuf message")
    return serialize(deterministic=True)


def _mac(
    request, *, psk: bytes, tenant_id: str, node_id: str, timestamp: str,
) -> str:
    body_hash = hashlib.sha256(_request_bytes(request)).digest()
    payload = b"\0".join((
        tenant_id.encode(),
        node_id.encode(),
        timestamp.encode(),
        body_hash,
    ))
    return hmac.new(psk, payload, hashlib.sha256).hexdigest()


def signed_metadata(
    request, config: FleetAuthConfig, *, now: float | None = None,
) -> Tuple[Tuple[str, str], ...]:
    timestamp = str(int(time.time() if now is None else now))
    return (
        (AUTH_TENANT, config.tenant_id),
        (AUTH_NODE, config.node_id),
        (AUTH_TS, timestamp),
        (AUTH_MAC, _mac(
            request,
            psk=config.psk,
            tenant_id=config.tenant_id,
            node_id=config.node_id,
            timestamp=timestamp,
        )),
    )


def verify_metadata(
    metadata: Iterable[Tuple[str, str]],
    request,
    config: FleetAuthConfig,
    *,
    now: float | None = None,
) -> Tuple[str, str]:
    values = {key.lower(): value for key, value in metadata}
    tenant = values.get(AUTH_TENANT, "")
    node = values.get(AUTH_NODE, "")
    timestamp = values.get(AUTH_TS, "")
    supplied = values.get(AUTH_MAC, "")
    if tenant != config.tenant_id:
        raise PrefillAuthError("tenant mismatch")
    if not node or not timestamp or not supplied:
        raise PrefillAuthError("missing prefill authentication metadata")
    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise PrefillAuthError("invalid authentication timestamp") from exc
    current = time.time() if now is None else now
    if abs(current - ts) > config.max_clock_skew_s:
        raise PrefillAuthError("authentication timestamp outside replay window")
    expected = _mac(
        request,
        psk=config.psk,
        tenant_id=tenant,
        node_id=node,
        timestamp=timestamp,
    )
    if not hmac.compare_digest(supplied, expected):
        raise PrefillAuthError("invalid prefill authentication MAC")
    return tenant, node


def metadata_pairs(metadata: Sequence) -> Tuple[Tuple[str, str], ...]:
    """Normalize grpc metadata objects or plain pairs for verification."""
    result = []
    for item in metadata:
        if hasattr(item, "key") and hasattr(item, "value"):
            result.append((item.key, item.value))
        else:
            result.append((item[0], item[1]))
    return tuple(result)

