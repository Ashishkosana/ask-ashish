"""In-memory per-key sliding window. One process, which matches a single Railway instance."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600) -> None:
        self.limit = limit
        self.window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        moment = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            cutoff = moment - self.window
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(moment)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        moment = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 1
            return max(1, int(self.window - (moment - events[0])) + 1)
