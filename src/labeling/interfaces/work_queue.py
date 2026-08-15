"""WorkQueue — pluggable FIFO boundary for future distributed workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WorkQueue(ABC):
    """Queue contract for optional asynchronous or distributed execution.

    Two Redis Stream channels (see the Redis implementation):

    - **task stream** — orchestrator → workers. ``enqueue`` publishes work items,
      ``dequeue`` pulls them. ``reclaim_stale`` returns abandoned deliveries.
    - **result stream** — workers → orchestrator. Workers publish labeled results
      with ``publish_results``; the orchestrator consumes them with
      ``consume_results`` and persists dedup/checkpoint state locally (single
      writer). ``reclaim_stale_results`` returns abandoned result deliveries.

    The queue is a pure transport layer — it owns no dedup/checkpoint state.
    That state lives with the orchestrator, which writes it as results arrive.

    Result payload shape (the second element of each ``publish_results`` tuple):

    - always: ``source_id``, ``recipe_card_hash``, ``raw_text``, ``outcome``
      (``"success"`` | ``"failure"``)
    - success: ``model``, ``output``
    - failure: ``last_error``
    """

    @abstractmethod
    async def load(self) -> None:
        """Initialize both consumer groups (task + result streams)."""
        ...

    @abstractmethod
    async def enqueue(self, items: list[dict[str, Any]]) -> None:
        """Publish work items to the task stream."""
        ...

    @abstractmethod
    async def dequeue(self, count: int) -> list[dict[str, Any]]:
        """Return up to *count* work items from the task stream."""
        ...

    @abstractmethod
    async def publish_results(
        self, results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Publish result payloads and acknowledge their tasks in one transaction.

        Each element is ``(task, result)``: *task* is the work item returned by
        ``dequeue`` (its ``_message_id`` is acknowledged), *result* is a plain
        dict (see class docstring for its shape). The task is acknowledged in the
        same transaction as its result is published, so a result is never lost
        before its task is released.
        """
        ...

    @abstractmethod
    async def consume_results(self, count: int) -> list[dict[str, Any]]:
        """Return up to *count* result payloads from the result stream."""
        ...

    @abstractmethod
    async def reclaim_stale(self) -> int:
        """Return unfinished task deliveries to the task stream."""
        ...

    @abstractmethod
    async def discard_pending(self) -> int:
        """Acknowledge and drop abandoned task deliveries.

        Dead workers strand their in-flight tasks in the consumer group's
        pending-entries list. The orchestrator re-enqueues every PENDING
        checkpoint item on the next run, so these abandoned deliveries are
        stale duplicates — discard them (ack without re-publish) so they are
        never processed twice. Returns the number discarded.
        """
        ...

    @abstractmethod
    async def reclaim_stale_results(self) -> int:
        """Return unfinished result deliveries to the result stream."""
        ...
