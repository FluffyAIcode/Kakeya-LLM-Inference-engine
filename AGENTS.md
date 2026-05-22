# Agents

## Cursor Cloud specific instructions

This is a **Python-only** ML research project (DLM Proposer + AR Verifier speculative-decoding engine). No web UI, Docker, or external services are required. Everything runs on CPU.

### Quick reference

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Smoke tests | `PYTHONPATH=. python3 scripts/smoke_test.py` |
| Equivalence-regime demo | `PYTHONPATH=. python3 -m kv_cache_proposer.run_demo --max-new-tokens 32 --block-size 8 --num-diffusion-steps 8 --sink-size 4 --window-size 64 --prompt "Reply with exactly: OK."` |
| Compression-regime demo | `PYTHONPATH=. python3 -m kv_cache_proposer.run_demo --max-new-tokens 64 --block-size 16 --num-diffusion-steps 16 --sink-size 4 --window-size 24 --batch-size-for-amortization 64 --prompt "Write a one-paragraph explanation of why prime numbers are infinite, suitable for a high school student."` |

### Gotchas

- **`dllm` stub required**: The HuggingFace model `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` references a broken `dllm` package in its `if __name__` block. The update script creates a no-op stub at `<site-packages>/dllm/__init__.py` so `transformers`' `check_imports` passes. If you see `ModuleNotFoundError: No module named 'dllm'`, the stub is missing.
- **`PYTHONPATH=.`** is required for all commands run from the repo root, since the `kv_cache_proposer` package is not installed as an editable package.
- **First run downloads ~5 GB** of model weights from HuggingFace Hub (Qwen3-0.6B-diffusion ~1.5 GB + Qwen3-1.7B ~3.4 GB). Subsequent runs use the HF cache (`~/.cache/huggingface/`).
- **Memory**: Both models load in bfloat16 and fit within 15 GB RAM on CPU.
- **No linter/formatter** is configured in the repo. There is no `pyproject.toml`, `setup.cfg`, or linting configuration.
- **No formal test framework** (pytest, unittest). The only automated tests are `scripts/smoke_test.py` which validates component integration on real weights.
