"""Local inference engine.

This package wraps the algorithmic core in `kv_cache_proposer/` with the
production-grade components described in
`docs/local-inference-engine.md`. Each subpackage corresponds to one of
the L4–L5 layers of the architecture:

  * proposer/  — proposer-side optimizations (sparse logits, …)
  * memory/    — fixed-slab KV pool, NF4 KV quantization (future)
  * scheduler/ — continuous batching (future)
  * server/    — OpenAI-compat HTTP API (future)
  * backends/  — MLX (Mac) and CUDA (Linux) backend bindings (future)
"""
