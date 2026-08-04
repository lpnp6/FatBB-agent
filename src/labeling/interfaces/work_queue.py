"""WorkQueue — pluggable FIFO boundary for future distributed workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WorkQueue(ABC):
    """Queue contract for optional asynchronous or distributed execution."""

    @abstractmethod
    async def load(self) -> None:
        """Initialize the queue connection or local state."""
        ...

    @abstractmethod
    async def enqueue(self, items: list[dict[str, Any]]) -> None:
        """Add work items to the queue."""
        ...

    @abstractmethod
    async def dequeue(self, count: int) -> list[dict[str, Any]]:
        """Return up to *count* work items for processing."""
        ...
