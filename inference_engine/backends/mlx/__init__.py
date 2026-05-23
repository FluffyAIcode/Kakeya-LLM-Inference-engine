"""MLX (Apple Silicon) backend for the inference engine.

This package is loadable on every host because its public surface is
the platform-neutral `env` module. The real MLX-dependent submodules
(verifier, proposer, cache) deliberately do *not* re-export at package
import time — the caller imports them explicitly, and that import will
fail on non-Mac hosts. There is no try/except fallback.

Typical use:

    from inference_engine.backends.mlx.env import probe_environment
    info = probe_environment()
    if info.is_available:
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        # ...
"""

from .env import (
    MLXEnvironment,
    MLXEnvironmentError,
    probe_environment,
)

__all__ = [
    "MLXEnvironment",
    "MLXEnvironmentError",
    "probe_environment",
]
