"""Streaming detokenizer for text-delta emission.

Provides :class:`_StreamingDetokenizer`, the per-token incremental
text-delta emitter consumed by ``app.py``'s SSE handler. The earlier
``iter_token_deltas`` / ``run_blocking`` helpers (engine-direct
streaming and blocking generation) have been removed in favor of
routing every request through the :class:`Scheduler` — see
``inference_engine.server.app`` for the integrated path.

The detokenizer's job is non-trivial: HuggingFace tokenizers can
decode partial id sequences, but they do *not* guarantee that
``decode([id_n])`` is a substring of ``decode([id_0..id_n])`` because
BPE merges and special-token handling reshape the prefix. The robust
pattern is::

    full = tokenizer.decode(all_ids_so_far, skip_special_tokens=True)
    delta = full[len(decoded_so_far):]
    decoded_so_far = full

which is what we replicate here. ``feed(token_id)`` returns the new
text since the last call, which may be the empty string if the new
token contributes only the first byte of a multi-byte UTF-8 sequence
(the next call will then yield both bytes).
"""

from __future__ import annotations

from typing import List

from .tokenizer import Tokenizer


class _StreamingDetokenizer:
    """Incremental decoder that emits valid text deltas only."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._all_ids: List[int] = []
        self._decoded_so_far: str = ""

    def feed(self, token_id: int) -> str:
        self._all_ids.append(int(token_id))
        full = self._tokenizer.decode(
            self._all_ids, skip_special_tokens=True
        )
        delta = full[len(self._decoded_so_far):]
        self._decoded_so_far = full
        return delta
