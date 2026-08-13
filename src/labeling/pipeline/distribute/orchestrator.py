"""Distributed producer — discover, deduplicate, enqueue, and drain batches.

The orchestrator owns dedup/checkpoint state. Workers only label and publish
results over the result stream; this module consumes those results and persists
them, so dedup/checkpoint remains single-writer (local to the orchestrator).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...interfaces.checkpoint_store import CheckpointStore
from ...interfaces.dedup_store import DedupEntry, DedupStore, HashStatus
from ...interfaces.orchestrator import Orchestrator
from ...interfaces.sampler import Sampler
from ...interfaces.work_queue import WorkQueue

logger = logging.getLogger(__name__)


class DistributedProductionOrchestrator(Orchestrator):
    """Discover and drain unique labeling-task batches through remote workers."""

    def __init__(
        self,
        *,
        dedup_store: DedupStore,
        sampler: Sampler,
        checkpoint: CheckpointStore,
        task_queue: WorkQueue,
        retries: int = 2,
    ) -> None:
        self._dedup_store = dedup_store
        self._sampler = sampler
        self._checkpoint = checkpoint
        self._task_queue = task_queue
        self._retries = retries

    async def run(
        self,
        root: Path | str,
        *,
        glob: str = "**/*.md",
    ) -> dict[str, Any]:
        """Queue every unique task, draining each batch before the next."""
        await self._checkpoint.load()
        await self._task_queue.load()
        # Reset items left IN_FLIGHT by a previous crash (same-run recovery).
        self._dedup_store.expire_stale(timeout_minutes=0)
        await self._checkpoint.expire_stale()

        scanned = 0
        queued = 0

        async for batch in self._sampler.sample(root, glob=glob):
            scanned += len(batch)
            tasks = [
                {
                    "source_id": source_id,
                    "recipe_card_hash": recipe_card_hash,
                    "raw_text": raw_text,
                }
                for source_id, recipe_card_hash, raw_text in batch
            ]
            if not tasks:
                continue

            self._dedup_store.register_batch([
                DedupEntry(
                    recipe_card_hash=str(task["recipe_card_hash"]),
                    status=HashStatus.IN_FLIGHT,
                    source_id=str(task["source_id"]),
                )
                for task in tasks
            ])
            await self._checkpoint.mark_in_flight_batch([
                (str(task["source_id"]), str(task["recipe_card_hash"]))
                for task in tasks
            ])
            await self._task_queue.enqueue(tasks)
            await self._drain_batch(tasks)
            queued += len(tasks)

        return {
            "outcomes": {"queued": queued},
            "total_scanned": scanned,
            "total_processed": queued,
        }

    async def _drain_batch(self, tasks: list[dict[str, Any]]) -> None:
        """Consume results until every task in the batch is settled.

        Success marks the item COMPLETED; failure re-enqueues it (with a retry
        counter) until ``retries`` is exhausted, then marks it REJECTED.
        """
        active = {str(task["source_id"]) for task in tasks}
        attempts: dict[str, int] = {}

        while active:
            results = await self._task_queue.consume_results(len(active))
            for result in results:
                source_id = str(result["source_id"])
                if source_id not in active:
                    continue  # stale/duplicate result from a previous run
                if result["outcome"] == "success":
                    self._dedup_store.register_batch([self._accepted_entry(result)])
                    await self._checkpoint.mark_completed_batch([source_id])
                    active.discard(source_id)
                    continue

                attempts[source_id] = attempts.get(source_id, 0) + 1
                if attempts[source_id] > self._retries:
                    await self._checkpoint.mark_rejected(
                        source_id, str(result.get("last_error", "")),
                    )
                    logger.error(
                        "Labeling exhausted retries id=%s error=%s",
                        source_id, result.get("last_error"),
                    )
                    active.discard(source_id)
                    continue

                logger.warning(
                    "Retrying id=%s attempt=%d error=%s",
                    source_id, attempts[source_id], result.get("last_error"),
                )
                await self._checkpoint.mark_pending(source_id)
                await self._checkpoint.mark_in_flight(
                    source_id, str(result["recipe_card_hash"]),
                )
                await self._task_queue.enqueue([{
                    "source_id": source_id,
                    "recipe_card_hash": result["recipe_card_hash"],
                    "raw_text": result["raw_text"],
                    "last_error": result.get("last_error", ""),
                }])

    @staticmethod
    def _accepted_entry(result: dict[str, Any]) -> DedupEntry:
        return DedupEntry(
            recipe_card_hash=str(result["recipe_card_hash"]),
            status=HashStatus.ACCEPTED,
            source_id=str(result["source_id"]),
            raw_text=str(result["raw_text"]),
            model=result.get("model"),
            output=result.get("output"),
        )
