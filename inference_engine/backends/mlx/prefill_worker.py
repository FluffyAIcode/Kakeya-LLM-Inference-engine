"""MLX prefill-only compute engine for ADR 0017 worker nodes."""
from __future__ import annotations

import threading
from typing import Callable, Sequence

from inference_engine.backends.mlx.prefill_snapshot import (
    export_mlx_prefill_snapshot,
)
from inference_engine.distributed.capability import (
    CacheCompatibility,
    CompressionCodec,
)
from inference_engine.distributed.prefill_cache import CacheBlock
from inference_engine.distributed.prefill_compression import compress_payload


class MLXPrefillComputeEngine:
    """Serially runs prefill with a loaded MLX verifier and exports one snapshot."""

    def __init__(
        self,
        verifier,
        compatibility: CacheCompatibility,
        *,
        compute_chunk_tokens: int = 256,
    ) -> None:
        if compute_chunk_tokens <= 0:
            raise ValueError("compute_chunk_tokens must be > 0")
        self.verifier = verifier
        self.compatibility = compatibility
        self.compute_chunk_tokens = int(compute_chunk_tokens)
        self._progress_callback: Callable[[int], None] | None = None
        self._lock = threading.Lock()

    def set_progress_callback(
        self,
        callback: Callable[[int], None] | None,
    ) -> None:
        self._progress_callback = callback

    def _report_progress(self, token_count: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(int(token_count))

    def compute_prefill(
        self,
        token_ids: Sequence[int],
        block_hashes: Sequence[bytes],
        *,
        compression: CompressionCodec,
        cancelled: threading.Event,
    ) -> Sequence[CacheBlock]:
        tokens = [int(token) for token in token_ids]
        if not tokens or not block_hashes:
            raise ValueError("token_ids and block_hashes must be non-empty")
        size = self.compatibility.block_size_tokens
        expected_blocks = (len(tokens) + size - 1) // size
        if len(block_hashes) != expected_blocks:
            raise ValueError(
                f"expected {expected_blocks} block hashes, got {len(block_hashes)}",
            )
        with self._lock:
            if cancelled.is_set():
                raise InterruptedError("prefill job cancelled")
            first_end = min(self.compute_chunk_tokens, len(tokens))
            self.verifier.prefill(tokens[:first_end])
            self._report_progress(first_end)
            for start in range(
                first_end,
                len(tokens),
                self.compute_chunk_tokens,
            ):
                if cancelled.is_set():
                    raise InterruptedError("prefill job cancelled")
                block = tokens[start:start + self.compute_chunk_tokens]
                logits = self.verifier.forward_block(block)
                self.verifier.commit_or_truncate(
                    forwarded=len(block),
                    accepted=len(block),
                )
                self.verifier.next_token_logits = logits[-1].clone()
                self._report_progress(min(start + len(block), len(tokens)))
            # Export exactly once. Intermediate full snapshots make encoding
            # and compression quadratic in prompt length and are not required
            # for correctness because the final chained hash commits every
            # preceding token block.
            return (
                self._snapshot(
                    token_count=len(tokens),
                    block_hash=block_hashes[-1],
                    compression=compression,
                ),
            )

    def _snapshot(
        self,
        *,
        token_count: int,
        block_hash: bytes,
        compression: CompressionCodec,
    ) -> CacheBlock:
        raw = export_mlx_prefill_snapshot(
            self.verifier.cache,
            token_count=token_count,
            cached_token_ids=self.verifier.cached_token_sequence,
            compatibility=self.compatibility,
            next_token_logits=self.verifier.next_token_logits,
            block_hash=block_hash,
        )
        return CacheBlock.create(
            bytes(block_hash),
            token_count,
            compress_payload(raw, compression),
        )

