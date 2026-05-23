"""MLX environment detection.

Pure-stdlib module so it can be imported on any host (including the
Linux x86 cloud agent) without pulling in `mlx`. Its only job is to
report whether the MLX runtime is actually available on this machine
and, if so, surface enough metadata for the rest of the backend to
make decisions (chip family, Metal availability, mlx-lm presence,
etc.).

Design choices:

  * **No fallback.** If a caller asks `require_environment()` for an
    available environment and one is not present, we raise
    `MLXEnvironmentError`. The caller does not silently fall back to
    PyTorch / CPU.
  * **No mock.** All checks query the real interpreter / OS / module
    metadata. The Linux test path exercises the
    "MLX absent → reports unavailable" branch end-to-end on its own
    machine, and the Mac test path exercises the
    "MLX present → reports available" branch on its own.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from typing import Optional


class MLXEnvironmentError(RuntimeError):
    """Raised when the MLX runtime is required but is not usable on this host."""


@dataclass(frozen=True)
class MLXEnvironment:
    """Structured snapshot of MLX availability on the current host.

    Attributes
    ----------
    is_available
        True if `mlx.core` imports successfully AND `mx.metal.is_available()`
        returns True. Both conditions must hold; an arm64 macOS host
        without a usable Metal device is *not* available.
    mlx_version
        Version string of the `mlx` distribution as reported by
        `importlib.metadata`. None when MLX is not installed.
    mlx_lm_version
        Version of the `mlx-lm` dist (used for loading Qwen3 weights),
        or None if not installed. mlx-lm is a soft dependency for
        higher-level utilities; mlx alone is sufficient for the
        backend's primitives.
    metal_available
        Result of `mlx.core.metal.is_available()`. Distinct from
        `is_available` because we record it even when no Metal device
        is present (so diagnostics are explicit about WHICH check
        failed).
    platform_str
        `platform.platform()` for diagnostics.
    machine
        `platform.machine()` — `arm64` on Apple Silicon. We refuse to
        treat MLX as available on non-arm64 hosts even if `mlx` happens
        to import (e.g. a future Linux build) because the rest of the
        codebase assumes Apple unified memory.
    python_version
        `platform.python_version()` for diagnostics.
    failure_reason
        Human-readable message when `is_available` is False. Empty
        string when MLX is available.
    """

    is_available: bool
    mlx_version: Optional[str]
    mlx_lm_version: Optional[str]
    metal_available: bool
    platform_str: str
    machine: str
    python_version: str
    failure_reason: str

    def render(self) -> str:
        """Return a stable single-line summary for logs / report files."""
        if self.is_available:
            return (
                f"mlx OK: mlx={self.mlx_version} mlx_lm={self.mlx_lm_version} "
                f"metal={self.metal_available} arch={self.machine} "
                f"python={self.python_version}"
            )
        return (
            f"mlx UNAVAILABLE ({self.failure_reason}): "
            f"mlx={self.mlx_version} mlx_lm={self.mlx_lm_version} "
            f"metal={self.metal_available} arch={self.machine} "
            f"python={self.python_version}"
        )


def _safe_dist_version(name: str) -> Optional[str]:
    """Return the installed distribution version or None.

    Exists as a small helper because `importlib.metadata.version` raises
    `PackageNotFoundError` rather than returning None when the dist is
    missing — we want the latter for our snapshot semantics.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment() -> MLXEnvironment:
    """Detect MLX availability without raising.

    The function is pure — it calls only `importlib`, `platform`, and
    (when available) `mlx.core.metal.is_available()`. It never imports
    anything beyond the lightweight `mlx.core` module, and even that is
    guarded so a missing or broken MLX install just sets
    `failure_reason` rather than crashing the caller.
    """
    machine = platform.machine()
    platform_str = platform.platform()
    python_version = platform.python_version()
    mlx_version = _safe_dist_version("mlx")
    mlx_lm_version = _safe_dist_version("mlx-lm") or _safe_dist_version("mlx_lm")

    # Hard refusal: mlx targets Apple Silicon. Even if a future Linux
    # build ships, the rest of our stack assumes unified memory.
    if machine != "arm64":
        return MLXEnvironment(
            is_available=False,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason=(
                f"machine={machine!r} is not arm64; "
                "MLX backend requires Apple Silicon"
            ),
        )

    if mlx_version is None:
        return MLXEnvironment(
            is_available=False,
            mlx_version=None,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason="mlx package is not installed",
        )

    # mlx is on the path; try to import its core and ask Metal directly.
    try:
        mx_core = importlib.import_module("mlx.core")
    except Exception as e:
        return MLXEnvironment(
            is_available=False,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason=f"mlx.core import failed: {type(e).__name__}: {e}",
        )

    metal_mod = getattr(mx_core, "metal", None)
    if metal_mod is None:
        return MLXEnvironment(
            is_available=False,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason="mlx.core has no `metal` submodule",
        )

    is_avail_fn = getattr(metal_mod, "is_available", None)
    if not callable(is_avail_fn):
        return MLXEnvironment(
            is_available=False,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason="mlx.core.metal.is_available is not callable",
        )

    try:
        metal_available = bool(is_avail_fn())
    except Exception as e:
        return MLXEnvironment(
            is_available=False,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason=f"metal.is_available() raised: {type(e).__name__}: {e}",
        )

    if not metal_available:
        return MLXEnvironment(
            is_available=False,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            metal_available=False,
            platform_str=platform_str,
            machine=machine,
            python_version=python_version,
            failure_reason="metal.is_available() returned False",
        )

    return MLXEnvironment(
        is_available=True,
        mlx_version=mlx_version,
        mlx_lm_version=mlx_lm_version,
        metal_available=True,
        platform_str=platform_str,
        machine=machine,
        python_version=python_version,
        failure_reason="",
    )


def require_environment() -> MLXEnvironment:
    """Return an :class:`MLXEnvironment` or raise.

    Use this at the top of any code path that needs a working MLX
    runtime. Callers that just want a structured availability snapshot
    (e.g. a runner or a diagnostic) should use :func:`probe_environment`
    instead.
    """
    env = probe_environment()
    if not env.is_available:
        raise MLXEnvironmentError(env.failure_reason)
    return env
