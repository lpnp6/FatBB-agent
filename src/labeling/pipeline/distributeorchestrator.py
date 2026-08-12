"""Distributed producer orchestration contracts.

This module keeps the existing sampler, deduplication, and checkpoint
interfaces.  It only adds the queue boundary needed to hand work to remote
workers; queue implementations own delivery and result persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..interfaces.checkpoint_store import CheckpointStore
from ..interfaces.dedup_store import DedupEntry, DedupStore, HashStatus
from ..interfaces.orchestrator import Orchestrator
from ..interfaces.sampler import Sampler
from ..interfaces.work_queue import WorkQueue


class DistributedProductionOrchestrator(Orchestrator):
    """Discover and drain unique labeling-task batches through remote workers."""

    def __init__(
        self,
        *,
        dedup_store: DedupStore,
        sampler: Sampler,
        checkpoint: CheckpointStore,
        task_queue: WorkQueue,
    ) -> None:
        self._dedup_store = dedup_store
        self._sampler = sampler
        self._checkpoint = checkpoint
        self._task_queue = task_queue

    async def run(
        self,
        root: Path | str,
        *,
        glob: str = "**/*.md",
    ) -> dict[str, Any]:
        """Queue every unique task, waiting for each batch to drain."""

        await self._checkpoint.load()
        await self._task_queue.load()

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
            await self._task_queue.join()
            queued += len(tasks)

        return {
            "outcomes": {"queued": queued},
            "total_scanned": scanned,
            "total_processed": queued,
        }
