"""Bounded, in-memory capture queue for maintenance cache-fill replays."""
from __future__ import annotations

import hashlib
import secrets
import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CapturedPrefix:
    capture_id: str
    token_ids: tuple[int, ...]
    token_count: int


class CacheFillCapture:
    """Capture first appends without persisting prompt or token content."""

    def __init__(
        self,
        *,
        max_items: int = 256,
        excluded_label_prefix: str = "cache-fill-",
    ) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self.max_items = int(max_items)
        self.excluded_label_prefix = excluded_label_prefix
        self._salt = secrets.token_bytes(32)
        self._items: deque[CapturedPrefix] = deque()
        self._seen: set[bytes] = set()
        self._lock = threading.Lock()
        self.captured = 0
        self.duplicates = 0
        self.dropped = 0

    def observe(
        self,
        *,
        client_label: str,
        token_ids: Iterable[int],
    ) -> bool:
        if client_label.startswith(self.excluded_label_prefix):
            return False
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return False
        digest = hashlib.sha256(
            self._salt
            + b"".join(token.to_bytes(4, "little", signed=False) for token in tokens)
        ).digest()
        with self._lock:
            if digest in self._seen:
                self.duplicates += 1
                return False
            if len(self._items) >= self.max_items:
                evicted = self._items.popleft()
                self._seen.discard(bytes.fromhex(evicted.capture_id))
                self.dropped += 1
            item = CapturedPrefix(digest.hex(), tokens, len(tokens))
            self._items.append(item)
            self._seen.add(digest)
            self.captured += 1
            return True

    def drain(self, max_items: int) -> list[CapturedPrefix]:
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        output = []
        with self._lock:
            while self._items and len(output) < max_items:
                item = self._items.popleft()
                self._seen.discard(bytes.fromhex(item.capture_id))
                output.append(item)
        return output

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "queued": len(self._items),
                "captured": self.captured,
                "duplicates": self.duplicates,
                "dropped": self.dropped,
                "max_items": self.max_items,
            }
