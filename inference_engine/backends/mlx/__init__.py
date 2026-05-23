"""MLX (Apple Silicon) backend for the inference engine.

This package is loadable on every host because its public surface is
the platform-neutral `env` module. The real MLX-dependent submodules
(`verifier`, `cache`, `_torch_bridge`) deliberately do *not* re-export
at package import time — the caller imports them explicitly, and that
import will fail on non-Mac hosts. There is no try/except fallback.

Typical use:

    from inference_engine.backends.mlx import probe_environment
    info = probe_environment()
    if info.is_available:
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        # ...

Submodule import contract:
  * `env`             — pure stdlib, importable everywhere
  * `_torch_bridge`   — imports `mlx.core`, Mac-only
  * `cache`           — imports `mlx.core`, Mac-only
  * `verifier`        — imports `mlx.core` and `mlx_lm`, Mac-only
"""

from .env import (
    MLXEnvironment,
    MLXEnvironmentError,
    probe_environment,
    require_environment,
)

__all__ = [
    "MLXEnvironment",
    "MLXEnvironmentError",
    "probe_environment",
    "require_environment",
]
