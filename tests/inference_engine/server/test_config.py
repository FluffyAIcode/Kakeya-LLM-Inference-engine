"""Unit tests for :mod:`inference_engine.server.config`."""

from __future__ import annotations

import pytest

from inference_engine.server.config import ServerConfig


def test_default_construction_succeeds():
    cfg = ServerConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.default_max_new_tokens == 1024
    assert cfg.request_timeout_s == 120.0
    assert cfg.model_id_label == "kakeya-v1"
    assert cfg.log_level == "info"


def test_explicit_construction_succeeds():
    cfg = ServerConfig(
        host="0.0.0.0", port=9000, default_max_new_tokens=256,
        request_timeout_s=30.0, model_id_label="custom",
        log_level="debug",
    )
    assert cfg.port == 9000
    assert cfg.model_id_label == "custom"


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_port_out_of_range_raises(port):
    with pytest.raises(ValueError, match="port must be in"):
        ServerConfig(port=port)


@pytest.mark.parametrize("value", [0, -100])
def test_non_positive_max_new_tokens_raises(value):
    with pytest.raises(ValueError, match="default_max_new_tokens must be positive"):
        ServerConfig(default_max_new_tokens=value)


@pytest.mark.parametrize("value", [0.0, -0.5])
def test_non_positive_request_timeout_raises(value):
    with pytest.raises(ValueError, match="request_timeout_s must be positive"):
        ServerConfig(request_timeout_s=value)


def test_empty_model_label_raises():
    with pytest.raises(ValueError, match="model_id_label must be non-empty"):
        ServerConfig(model_id_label="")


def test_whitespace_only_model_label_raises():
    with pytest.raises(ValueError, match="model_id_label must be non-empty"):
        ServerConfig(model_id_label="   ")


def test_unknown_log_level_raises():
    with pytest.raises(ValueError, match="log_level must be one of"):
        ServerConfig(log_level="VERBOSE")


def test_from_env_with_no_vars_matches_defaults():
    cfg = ServerConfig.from_env(env={})
    assert cfg == ServerConfig()


def test_from_env_reads_all_supported_vars():
    cfg = ServerConfig.from_env(env={
        "KAKEYA_HOST": "10.0.0.1",
        "KAKEYA_PORT": "1234",
        "KAKEYA_MAX_NEW_TOKENS": "512",
        "KAKEYA_REQUEST_TIMEOUT_S": "45.5",
        "KAKEYA_MODEL_ID_LABEL": "myengine",
        "KAKEYA_LOG_LEVEL": "warning",
    })
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 1234
    assert cfg.default_max_new_tokens == 512
    assert cfg.request_timeout_s == 45.5
    assert cfg.model_id_label == "myengine"
    assert cfg.log_level == "warning"


def test_from_env_ignores_unknown_vars():
    cfg = ServerConfig.from_env(env={"PATH": "/usr/bin", "HOME": "/root"})
    assert cfg == ServerConfig()


def test_from_env_invalid_int_raises():
    with pytest.raises(ValueError, match="not an integer"):
        ServerConfig.from_env(env={"KAKEYA_PORT": "not-a-number"})


def test_from_env_invalid_float_raises():
    with pytest.raises(ValueError, match="not a float"):
        ServerConfig.from_env(env={"KAKEYA_REQUEST_TIMEOUT_S": "abc"})


def test_from_env_int_validation_caught_by_post_init():
    with pytest.raises(ValueError, match="port must be in"):
        ServerConfig.from_env(env={"KAKEYA_PORT": "0"})


def test_from_env_uses_os_environ_by_default(monkeypatch):
    monkeypatch.delenv("KAKEYA_PORT", raising=False)
    monkeypatch.setenv("KAKEYA_HOST", "172.16.0.1")
    cfg = ServerConfig.from_env()
    assert cfg.host == "172.16.0.1"
    monkeypatch.delenv("KAKEYA_HOST", raising=False)


def test_frozen_dataclass_rejects_mutation():
    cfg = ServerConfig()
    with pytest.raises(Exception):
        cfg.port = 9000  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scheduler-related fields (added when E2 ↔ E4 integrated)
# ---------------------------------------------------------------------------


def test_default_max_concurrent_is_one():
    cfg = ServerConfig()
    assert cfg.max_concurrent == 1


def test_default_admission_policy_is_reject():
    from inference_engine.scheduler.config import AdmissionPolicy
    cfg = ServerConfig()
    assert cfg.admission_policy is AdmissionPolicy.REJECT


def test_default_queue_wait_is_zero():
    cfg = ServerConfig()
    assert cfg.queue_max_wait_s == 0.0


@pytest.mark.parametrize("n", [0, -1, -100])
def test_non_positive_max_concurrent_raises(n):
    with pytest.raises(ValueError, match="max_concurrent must be positive"):
        ServerConfig(max_concurrent=n)


@pytest.mark.parametrize("w", [-0.1, -1.0])
def test_negative_queue_wait_raises(w):
    with pytest.raises(ValueError, match="queue_max_wait_s must be >= 0"):
        ServerConfig(queue_max_wait_s=w)


def test_from_env_max_concurrent():
    cfg = ServerConfig.from_env(env={"KAKEYA_MAX_CONCURRENT": "8"})
    assert cfg.max_concurrent == 8


def test_from_env_invalid_max_concurrent_raises():
    with pytest.raises(ValueError, match="not an integer"):
        ServerConfig.from_env(env={"KAKEYA_MAX_CONCURRENT": "many"})


def test_from_env_admission_policy_reject():
    from inference_engine.scheduler.config import AdmissionPolicy
    cfg = ServerConfig.from_env(env={"KAKEYA_ADMISSION_POLICY": "reject"})
    assert cfg.admission_policy is AdmissionPolicy.REJECT


def test_from_env_admission_policy_queue():
    from inference_engine.scheduler.config import AdmissionPolicy
    cfg = ServerConfig.from_env(env={"KAKEYA_ADMISSION_POLICY": "queue"})
    assert cfg.admission_policy is AdmissionPolicy.QUEUE


def test_from_env_admission_policy_case_insensitive():
    from inference_engine.scheduler.config import AdmissionPolicy
    cfg = ServerConfig.from_env(env={"KAKEYA_ADMISSION_POLICY": "  QUEUE  "})
    assert cfg.admission_policy is AdmissionPolicy.QUEUE


def test_from_env_invalid_admission_policy_raises():
    with pytest.raises(ValueError, match="not a valid AdmissionPolicy"):
        ServerConfig.from_env(env={"KAKEYA_ADMISSION_POLICY": "drop"})


def test_from_env_queue_max_wait_s():
    cfg = ServerConfig.from_env(env={"KAKEYA_QUEUE_MAX_WAIT_S": "5.5"})
    assert cfg.queue_max_wait_s == 5.5


def test_from_env_invalid_queue_wait_raises():
    with pytest.raises(ValueError, match="not a float"):
        ServerConfig.from_env(env={"KAKEYA_QUEUE_MAX_WAIT_S": "soon"})


# ---------------------------------------------------------------------------
# api_keys validation + env-var parsing
# ---------------------------------------------------------------------------


def test_api_keys_default_is_empty():
    cfg = ServerConfig()
    assert cfg.api_keys == frozenset()


def test_api_keys_explicit_set():
    cfg = ServerConfig(api_keys=frozenset({"sk-a", "sk-b"}))
    assert cfg.api_keys == frozenset({"sk-a", "sk-b"})


def test_api_keys_rejects_non_string():
    with pytest.raises(ValueError, match="api_keys must be strings"):
        ServerConfig(api_keys=frozenset({123}))  # type: ignore[arg-type]


def test_api_keys_rejects_empty_string():
    with pytest.raises(ValueError, match="non-empty whitespace-free"):
        ServerConfig(api_keys=frozenset({""}))


def test_api_keys_rejects_whitespace_in_key():
    with pytest.raises(ValueError, match="non-empty whitespace-free"):
        ServerConfig(api_keys=frozenset({"sk has space"}))


def test_from_env_api_keys_csv_parsing():
    cfg = ServerConfig.from_env(env={"KAKEYA_API_KEYS": "sk-a,sk-b,sk-c"})
    assert cfg.api_keys == frozenset({"sk-a", "sk-b", "sk-c"})


def test_from_env_api_keys_strips_whitespace_around_entries():
    cfg = ServerConfig.from_env(env={"KAKEYA_API_KEYS": " sk-a , sk-b "})
    assert cfg.api_keys == frozenset({"sk-a", "sk-b"})


def test_from_env_api_keys_drops_empty_entries():
    cfg = ServerConfig.from_env(env={"KAKEYA_API_KEYS": "sk-a,,sk-b,"})
    assert cfg.api_keys == frozenset({"sk-a", "sk-b"})
