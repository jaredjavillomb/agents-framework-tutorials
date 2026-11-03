"""One definition of "a tool call", shared by every agent's tools.

Each agent's tools module owns one ToolCallLog. Every call to a tool goes through `attempt()`, which
records it and says whether it may run. The runner reads `names()` for the results file and calls
`reset()` between scenarios or questions. Thread-safe, because some frameworks run tools in parallel.
"""

from __future__ import annotations

import threading


class ToolCallLog:
    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self._names: list[str] = []
        self._refused = 0
        self._lock = threading.Lock()

    def attempt(self, name: str) -> bool:
        """Record a call to `name`. Return False if it is over the limit and must not run."""
        with self._lock:
            self._names.append(name)
            allowed = self.limit is None or len(self._names) <= self.limit
            if not allowed:
                self._refused += 1
            return allowed

    def names(self) -> list[str]:
        """Every attempted call in order, refused ones included."""
        with self._lock:
            return list(self._names)

    def budget_exhausted(self) -> bool:
        """True if at least one call was refused for being over the limit."""
        with self._lock:
            return self._refused > 0

    def reset(self) -> None:
        with self._lock:
            self._names.clear()
            self._refused = 0
