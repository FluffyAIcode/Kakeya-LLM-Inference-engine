"""Pure-Python aggregation helpers used by ``scripts/bench_agentic/``.

These helpers are split out of the CLI scripts so they can be unit-
tested under the Linux 100% coverage gate. The CLI scripts that
import them are themselves exempt from the coverage gate (CLI
plumbing convention; see ``scripts/serve.py`` for precedent).
"""
