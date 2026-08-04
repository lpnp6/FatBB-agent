"""WorkQueue — abstract FIFO queue with crash recovery for the labeling pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WorkQueue(ABC):
    """Persistent FIFO queue for labeling work items.

    :meth:`dequeue` destructively removes items from the queue.  If the
    process crashes between *dequeue* and the dedup-store write, the item
    is re-discovered by the sampler on the next run (its hash is not yet
    in the dedup store) and re-enqueued.
    """

    @abstractmethod
    async def load(self) -> None:
        """Restore queue state from storage.  No-op on first run."""
        ...

    @abstractmethod
    async def enqueue(self, items: list[dict[str, Any]]) -> None:
        """Add *items* to the queue.  Known ids are left untouched.

        Each item dict must contain at least ``"id"``.
        """
        ...

    @abstractmethod
    async def dequeue(self, count: int) -> list[dict[str, Any]]:
        """Remove and return up to *count* items.

        Items are permanently removed from the queue.  If the process
        crashes the sampler re-discovers unprocessed items on the next
        run and re-enqueues them.
        """
        ...
