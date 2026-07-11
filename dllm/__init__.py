"""Compatibility namespace for legacy dllm-hub remote model imports.

The legacy Qwen diffusion checkpoint declares ``dllm`` in its Transformers
auto-map source even though Kakeya does not import runtime symbols from that
package. Keeping this empty namespace on PYTHONPATH satisfies the static
dependency check consistently in CI and source checkouts.
"""
