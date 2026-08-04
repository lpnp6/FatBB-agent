"""CheckpointStore — abstract interface for labeling progress tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class ItemStatus(str, Enum):
    """Lifecycle of a single item in the labeling pipeline."""

    PENDING = "pending"
    """Item registered but not yet processed."""

    IN_FLIGHT = "in_flight"
    """Processing started, not yet complete."""

    COMPLETED = "completed"
    """Processing succeeded, result written to training output."""

    REJECTED = "rejected"
    """All retries exhausted. May be retried on next run (no cross-run limit)."""


class CheckpointStore(ABC):
    """Persist per-item labeling progress with atomic state transitions.

    Responsibilities:
        1. Track every manifest item through PENDING → IN_FLIGHT →
           COMPLETED / REJECTED.
        2. Survive crashes: on restart, IN_FLIGHT items are retried
           (attempt counter preserved), REJECTED items get another chance.
        3. Support concurrent workers: single-process, multi-process, or
           distributed, depending on the concrete implementation.

    Planned implementations:
        - FileCheckpointStore   (JSON file, single-process, zero deps)
        - MemoryCheckpointStore (in-memory dict, for unit tests)
        - SQLiteCheckpointStore (SQLite WAL, multi-process safe)
        - RedisCheckpointStore  (Redis Hash, distributed)
    """

    # ---- lifecycle -----------------------------------------------------------

    @abstractmethod
    async def load(self) -> None:
        """Initialize or resume checkpoint state from storage.

        Called once at pipeline startup. After this call the store is ready
        for ensure_items() and state transitions. State is fully encapsulated
        — callers query via get_status() / get_attempt().
        """
        ...

    @abstractmethod
    async def ensure_items(self, item_ids: list[str]) -> None:
        """Register items that don't yet exist as PENDING. Idempotent.

        Called at the start of every run with the full manifest. Existing
        items (any status) are left untouched. Only truly new item_ids are
        added in PENDING state.

        Args:
            item_ids: Every item id from the current manifest.
        """
        ...

    # ---- accessors -----------------------------------------------------------

    @abstractmethod
    def get_status(self, item_id: str) -> ItemStatus:
        """Return the current lifecycle status of *item_id*.

        Raises:
            KeyError: If *item_id* was never registered via ensure_items().
        """
        ...

    @abstractmethod
    def get_attempt(self, item_id: str) -> int:
        """Return the number of times this item has been marked IN_FLIGHT.

        Raises:
            KeyError: If *item_id* was never registered via ensure_items().
        """
        ...

    # ---- state transitions ---------------------------------------------------

    @abstractmethod
    async def mark_in_flight(self, item_id: str, recipe_card_hash: str) -> int:
        """Atomically transition PENDING / REJECTED → IN_FLIGHT.

        Increments the attempt counter and records the content hash for
        provenance. Returns the new attempt number (1-based after the first
        call, 2 after the first retry, etc.).

        Concrete implementations must make this atomic — via asyncio.Lock,
        SQLite transaction, or Redis WATCH/MULTI — so two concurrent workers
        cannot both transition the same item.

        Args:
            item_id: The manifest item to transition.
            recipe_card_hash: Content fingerprint for provenance.

        Returns:
            The attempt number AFTER incrementing.
        """
        ...

    @abstractmethod
    async def mark_completed(self, item_id: str) -> None:
        """Transition IN_FLIGHT → COMPLETED."""
        ...

    @abstractmethod
    async def mark_completed_batch(self, item_ids: list[str]) -> None:
        """Transition multiple items to COMPLETED in a single persist.

        Semantically equivalent to calling :meth:`mark_completed` for each id,
        but implementations should batch the write.
        """
        ...

    @abstractmethod
    async def mark_rejected(self, item_id: str, error: str) -> None:
        """Transition IN_FLIGHT → REJECTED.

        Called when all retries for this item are exhausted within a single
        run. The item may still be retried on the next run — there is no
        cross-run rejection limit.

        Args:
            item_id: The manifest item that exhausted retries.
            error: The last error message for diagnostics.
        """
        ...

    @abstractmethod
    async def mark_pending(self, item_id: str) -> None:
        """Reset an interrupted item from IN_FLIGHT to PENDING."""
        ...

    # ---- monitoring ----------------------------------------------------------

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        """Return item counts keyed by status.

        Example return value:
            {"pending": 3, "in_flight": 0, "completed": 490, "rejected": 7}
        """
        ...
