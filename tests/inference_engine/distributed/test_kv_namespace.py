from inference_engine.distributed.capability import CacheCompatibility
from inference_engine.distributed.kv_namespace import VirtualKVNamespace


def test_virtual_namespace_aggregates_matching_cache_mounts():
    namespace = VirtualKVNamespace(CacheCompatibility(
        model_id="gemma",
        tenant_namespace="private",
    ))
    result = namespace.describe([
        {
            "id": "head",
            "cache": {
                "model_id": "gemma",
                "bytes_used": 10,
                "bytes_free": 20,
                "entry_count": 2,
            },
            "endpoint": {"address": "head:1", "network": "thunderbolt"},
        },
        {
            "id": "peer",
            "cache": {
                "model_id": "other",
                "bytes_used": 99,
                "bytes_free": 1,
                "entry_count": 9,
            },
            "endpoint": None,
        },
        {"id": "worker", "cache": None},
    ])
    assert result["uri"].startswith("kv://private/gemma/")
    assert result["coherent_shared_memory"] is False
    assert result["bytes_used"] == 10
    assert result["bytes_free"] == 20
    assert result["entry_count"] == 2
    assert result["mounts"] == [{
        "node_id": "head",
        "address": "head:1",
        "bytes_used": 10,
        "bytes_free": 20,
        "entry_count": 2,
        "network": "thunderbolt",
    }]


def test_virtual_namespace_defaults_tenant_and_endpoint():
    namespace = VirtualKVNamespace(CacheCompatibility(model_id="m"))
    result = namespace.describe([{
        "id": "cache",
        "cache": {"model_id": "m"},
    }])
    assert result["uri"].startswith("kv://default/m/")
    assert result["mounts"][0]["address"] == ""
    assert result["mounts"][0]["network"] == "default"
