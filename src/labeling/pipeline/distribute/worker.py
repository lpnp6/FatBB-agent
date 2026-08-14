"""Worker that consumes tasks from a WorkQueue and labels them via a LabelingClient."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ...interfaces.labeling_client import LabelingClient
from ...interfaces.work_queue import WorkQueue
from ...utils.validator import OutputValidationError, OutputValidator

logger = logging.getLogger(__name__)


class Worker:
    """Dequeue tasks, call the model, and publish results back.

    The worker is stateless with respect to dedup/checkpoint: it only labels and
    ships a self-contained result dict over the queue. The orchestrator owns all
    persistence.
    """

    def __init__(
        self,
        *,
        queue: WorkQueue,
        client: LabelingClient,
        validator: OutputValidator | None = None,
        repair: bool = True,
    ) -> None:
        self._queue = queue
        self._client = client
        self._validator = validator or OutputValidator()
        self._repair = repair

    async def run(self, *, count: int = 1) -> None:
        """Block in a dequeue → label → publish loop, publishing as tasks land."""
        await self._queue.load()
        while True:
            tasks = await self._queue.dequeue(count)
            if not tasks:
                continue
            for coro in asyncio.as_completed(self._process_and_pair(t) for t in tasks):
                task, result = await coro
                await self._queue.publish_results([(task, result)])

    async def _process_and_pair(
        self, task: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Label one task and return it alongside its result so publish_results
        can ack the correct ``_message_id``."""
        return task, await self._process_one(task)

    async def _process_one(self, task: dict[str, Any]) -> dict[str, Any]:
        source_id = str(task["source_id"])
        recipe_card_hash = str(task["recipe_card_hash"])
        raw_text = str(task["raw_text"])

        try:
            result = await self._client.label(raw_text)
            try:
                parsed = self._validator.parse(result.raw_output)
            except OutputValidationError as exc:
                if not self._repair:
                    raise
                result = await self._client.repair(result.raw_output, str(exc))
                parsed = self._validator.parse(result.raw_output)
        except Exception as exc:
            logger.warning("Task failed source_id=%s error=%s", source_id, exc)
            return {
                "source_id": source_id,
                "recipe_card_hash": recipe_card_hash,
                "raw_text": raw_text,
                "outcome": "failure",
                "last_error": str(exc),
            }

        return {
            "source_id": source_id,
            "recipe_card_hash": recipe_card_hash,
            "raw_text": raw_text,
            "outcome": "success",
            "model": result.model,
            "output": json.dumps(parsed.normalized_json, ensure_ascii=False),
        }
