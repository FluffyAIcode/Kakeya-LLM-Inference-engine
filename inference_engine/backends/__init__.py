"""Hardware backends.

Each subpackage provides a backend-specific implementation of the
inference-engine primitives (verifier, proposer, KV cache, sampling)
that match the same API as the platform-neutral reference in
`kv_cache_proposer/`. Subpackages here import their hardware
runtime (MLX / CUDA) at module load — importing
`inference_engine.backends.mlx` from a host without Apple Silicon
will raise. That is intentional: there is no fallback path that
silently substitutes a different backend.
"""
