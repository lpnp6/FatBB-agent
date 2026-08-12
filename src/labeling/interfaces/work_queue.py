"""WorkQueue — pluggable FIFO boundary for future distributed workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .dedup_store import DedupEntry


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

    @abstractmethod
    async def join(self) -> None:
        """Wait until all work enqueued so far has reached a final outcome."""
        ...

    @abstractmethod
    async def submit_results(
        self, outcomes: list[tuple[dict[str, Any], DedupEntry]],
    ) -> None:
        """Persist validated results, then mark and acknowledge their tasks."""
        ...

    @abstractmethod
    async def submit_retries(
        self, failures: list[tuple[dict[str, Any], Exception]],
    ) -> None:
        """Record failed attempts and arrange their next delivery."""
        ...

    @abstractmethod
    async def reclaim_stale(self) -> int:
        """Return unfinished deliveries to the queue."""
        ...
