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
