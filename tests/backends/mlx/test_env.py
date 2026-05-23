"""Unit tests for `inference_engine.backends.mlx.env`.

The module's branching depends on the real platform and the real
`mlx` install state. We exercise both halves of every branch by
selectively monkey-patching `platform.machine`, `importlib.metadata`,
and `importlib.import_module` — the *system under test* is never
mocked; we only mock its environment dependencies, which is the only
way to reach the "mlx absent" branches on a Mac and the "mlx present"
branches on Linux.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
import types

import pytest

from inference_engine.backends.mlx import env as env_mod
from inference_engine.backends.mlx.env import (
    MLXEnvironment,
    MLXEnvironmentError,
    probe_environment,
    require_environment,
    _safe_dist_version,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _force_arm64(monkeypatch) -> None:
    """Pretend the host is Apple Silicon so the early-out doesn't trip."""
    monkeypatch.setattr(platform, "machine", lambda: "arm64")


def _make_fake_mx_core(*, has_metal=True, is_avail=True, raise_on_call=False):
    """Build a minimal stand-in for `mlx.core` that exposes the surface
    the env module looks at: a `metal` submodule with `is_available()`."""
    mx_core = types.ModuleType("mlx.core")
    if has_metal:
        metal = types.ModuleType("mlx.core.metal")
        if raise_on_call:
            def _bad():
                raise OSError("mocked metal probe error")
            metal.is_available = _bad
        elif callable(is_avail) or isinstance(is_avail, bool):
            metal.is_available = (lambda v: (lambda: v))(is_avail) \
                if isinstance(is_avail, bool) else is_avail
        else:
            metal.is_available = "not_callable"
        mx_core.metal = metal
    return mx_core


def _patch_import(monkeypatch, fake_mx_core=None, raise_on_import=None):
    """Override `importlib.import_module` so it returns our fake mlx.core
    (or raises) for that specific name; falls through for everything else."""
    real_import = importlib.import_module

    def _mocked(name, *args, **kwargs):
        if name == "mlx.core":
            if raise_on_import is not None:
                raise raise_on_import
            return fake_mx_core
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(env_mod.importlib, "import_module", _mocked)


def _patch_dist_version(monkeypatch, *, mlx, mlx_lm):
    """Replace `_safe_dist_version` so we can inject installed-package
    states without actually installing or uninstalling anything."""
    def _fake(name):
        if name == "mlx":
            return mlx
        if name in ("mlx-lm", "mlx_lm"):
            return mlx_lm
        return None

    monkeypatch.setattr(env_mod, "_safe_dist_version", _fake)


# ---------------------------------------------------------------------------
# _safe_dist_version helper
# ---------------------------------------------------------------------------

def test_safe_dist_version_missing_returns_none(monkeypatch) -> None:
    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(env_mod.importlib.metadata, "version", _raise)
    assert _safe_dist_version("definitely-not-installed") is None


def test_safe_dist_version_present_returns_string() -> None:
    # pytest is always installed in the test env
    v = _safe_dist_version("pytest")
    assert v is not None
    assert isinstance(v, str)
    assert len(v) > 0


# ---------------------------------------------------------------------------
# Hard refusal paths (architecture)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["x86_64", "i386", "aarch64", "ppc64le"])
def test_non_arm64_returns_unavailable(monkeypatch, arch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: arch)
    _patch_dist_version(monkeypatch, mlx="0.31.0", mlx_lm="0.20.0")
    env = probe_environment()
    assert env.is_available is False
    assert env.machine == arch
    assert "arm64" in env.failure_reason


# ---------------------------------------------------------------------------
# Missing mlx package
# ---------------------------------------------------------------------------

def test_arm64_but_mlx_not_installed(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx=None, mlx_lm=None)
    env = probe_environment()
    assert env.is_available is False
    assert env.mlx_version is None
    assert env.failure_reason == "mlx package is not installed"


# ---------------------------------------------------------------------------
# mlx.core import failure
# ---------------------------------------------------------------------------

def test_mlx_core_import_raises(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.0", mlx_lm=None)
    _patch_import(monkeypatch, raise_on_import=OSError("mock dlopen failed"))
    env = probe_environment()
    assert env.is_available is False
    assert "mlx.core import failed" in env.failure_reason
    assert "OSError" in env.failure_reason


# ---------------------------------------------------------------------------
# mlx.core present but malformed (no .metal, or bad is_available)
# ---------------------------------------------------------------------------

def test_mlx_core_missing_metal_submodule(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.0", mlx_lm=None)
    fake = _make_fake_mx_core(has_metal=False)
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = probe_environment()
    assert env.is_available is False
    assert "no `metal` submodule" in env.failure_reason


def test_mlx_metal_is_available_not_callable(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.0", mlx_lm=None)
    fake = _make_fake_mx_core(has_metal=True, is_avail="not_callable")
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = probe_environment()
    assert env.is_available is False
    assert "is not callable" in env.failure_reason


def test_mlx_metal_is_available_raises(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.0", mlx_lm=None)
    fake = _make_fake_mx_core(has_metal=True, raise_on_call=True)
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = probe_environment()
    assert env.is_available is False
    assert "is_available() raised" in env.failure_reason


def test_mlx_metal_is_available_returns_false(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.0", mlx_lm=None)
    fake = _make_fake_mx_core(has_metal=True, is_avail=False)
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = probe_environment()
    assert env.is_available is False
    assert "returned False" in env.failure_reason
    assert env.metal_available is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_mlx_fully_available(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.1", mlx_lm="0.20.0")
    fake = _make_fake_mx_core(has_metal=True, is_avail=True)
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = probe_environment()
    assert env.is_available is True
    assert env.mlx_version == "0.31.1"
    assert env.mlx_lm_version == "0.20.0"
    assert env.metal_available is True
    assert env.failure_reason == ""


def test_mlx_available_without_mlx_lm(monkeypatch) -> None:
    """mlx-lm is a soft dependency; mlx alone should yield is_available=True."""
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.1", mlx_lm=None)
    fake = _make_fake_mx_core(has_metal=True, is_avail=True)
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = probe_environment()
    assert env.is_available is True
    assert env.mlx_lm_version is None


# ---------------------------------------------------------------------------
# render() output
# ---------------------------------------------------------------------------

def test_render_when_available(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.1", mlx_lm="0.20.0")
    fake = _make_fake_mx_core(has_metal=True, is_avail=True)
    _patch_import(monkeypatch, fake_mx_core=fake)
    s = probe_environment().render()
    assert s.startswith("mlx OK:")
    assert "mlx=0.31.1" in s
    assert "metal=True" in s


def test_render_when_unavailable(monkeypatch) -> None:
    _patch_dist_version(monkeypatch, mlx=None, mlx_lm=None)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    s = probe_environment().render()
    assert s.startswith("mlx UNAVAILABLE")
    assert "arm64" in s  # the failure reason mentions arm64


# ---------------------------------------------------------------------------
# require_environment
# ---------------------------------------------------------------------------

def test_require_raises_when_unavailable(monkeypatch) -> None:
    _patch_dist_version(monkeypatch, mlx=None, mlx_lm=None)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    with pytest.raises(MLXEnvironmentError, match="arm64"):
        require_environment()


def test_require_returns_when_available(monkeypatch) -> None:
    _force_arm64(monkeypatch)
    _patch_dist_version(monkeypatch, mlx="0.31.1", mlx_lm="0.20.0")
    fake = _make_fake_mx_core(has_metal=True, is_avail=True)
    _patch_import(monkeypatch, fake_mx_core=fake)
    env = require_environment()
    assert env.is_available is True


# ---------------------------------------------------------------------------
# Sanity: real probe on this host doesn't crash
# ---------------------------------------------------------------------------

def test_real_probe_is_well_formed() -> None:
    """Whatever host runs this — Linux x86 cloud agent, Mac M-series,
    CUDA box — `probe_environment()` must return a coherent
    `MLXEnvironment` and not raise."""
    env = probe_environment()
    assert isinstance(env, MLXEnvironment)
    # `is_available` must agree with `metal_available` and `failure_reason`.
    if env.is_available:
        assert env.metal_available is True
        assert env.failure_reason == ""
    else:
        assert env.failure_reason != ""
    # render() never crashes
    s = env.render()
    assert isinstance(s, str) and len(s) > 0
