"""Architecture 9 direct-cutover entry points."""
from __future__ import annotations

from autoresearch.prefill.architecture_v7 import (
    run_architecture_v7_entry,
    run_host_definition_gate,
)
from autoresearch.prefill.cursor_strategy import CursorStrategyAdapter


def run_architecture_v9_entry(*args, strategy_adapter=None, **kwargs):
    """Production entry: Cursor strategy is mandatory and has no fallback."""
    return run_architecture_v7_entry(
        *args,
        strategy_adapter=strategy_adapter or CursorStrategyAdapter(),
        **kwargs,
    )


__all__ = ["run_architecture_v9_entry", "run_host_definition_gate"]
